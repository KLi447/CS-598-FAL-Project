from typing import List, Set, override, Dict
import threading
import time

from mlora.config.dispatcher import DispatcherConfig
from mlora.executor.task import Task
from mlora.model.args import Masks, MLoRAData, MLoRADataConfig, Tokens
from mlora.utils.gpu_state import GPUState, query_all_gpus, AdapterProfile
import logging

from .backend_dispatcher import BackendDispatcher


class TensorParallelDispatcher(BackendDispatcher):
    list_lock_: threading.Lock

    lock_set_: Set[str]

    def __init__(self, config: DispatcherConfig, adapter_profiles: Dict[str, AdapterProfile]) -> None:
        super().__init__(config)
        self.list_lock_ = threading.Lock()
        self.lock_set_ = set()
        self.adapter_profiles_ = adapter_profiles
        self.concurrency_num_ = config.concurrency_num_
        self.tp_world_size_ = 2  # FIXME

    def update_adapter_profiles(self, adapter_profiles: Dict[str, AdapterProfile]) -> None:
        self.adapter_profiles_ = adapter_profiles

    def _query_tp_group_gpus(self) -> List[GPUState]:
        return query_all_gpus()

    def _select_tasks_for_batch(self, candidates: List[Task]) -> List[Task]:
        #needs to be fixed
        gpu_states = self._query_tp_group_gpus()
        if not gpu_states:
            return []

        bottleneck_gpu = min(gpu_states, key=lambda gpu: gpu.free_mem)
        free_memory = bottleneck_gpu.free_mem

        def score_task(task: Task) -> float:
            wait_boost = min(task.waiting / 20.0, 1.0)
            aging_score = 1.0 + (wait_boost**2)
            return aging_score

        sorted_candidates = sorted(candidates, key=score_task, reverse=True)

        selected_tasks: List[Task] = []
        estimated_mem_usage = 0
        
        max_tokens_in_batch = 0

        for task in sorted_candidates:
            if len(selected_tasks) >= self.concurrency_num_:
                break

            adapter_name = task.adapter_name()[0]
            prof = self.adapter_profiles_[adapter_name]

            task_mem_estimate = prof.activations_memory_estimate(task.config_.batch_size_, 256) / self.tp_world_size_
            task_mem_estimate += prof.adapter_memory_estimate()

            if estimated_mem_usage + task_mem_estimate > free_memory:
                logging.info(f"Skipping task {task.task_name()} due to memory constraints.")
                continue

            selected_tasks.append(task)
            estimated_mem_usage += task_mem_estimate
            max_tokens_in_batch += task.config_.batch_size_ * 256

        logging.info(f"Selected {len(selected_tasks)} tasks for the next batch.")
        return selected_tasks

    @override
    def _dispatch_task_in(self):
        with self.list_lock_:
            terminate_ready = [task for task in self.ready_ if task.is_terminate()]
            if terminate_ready:
                self.ready_ = [task for task in self.ready_ if not task.is_terminate()]
                for task in terminate_ready:
                    logging.info(f"Task {task.task_name()} terminated from ready queue.")
                    self.terminate_event_.notify(task)

            terminate_running = [task for task in self.running_ if task.is_terminate()]
            if terminate_running:
                self.running_ = [task for task in self.running_ if not task.is_terminate()]
                for task in terminate_running:
                    logging.info(f"Task {task.task_name()} terminated from running queue.")
                    self.terminate_event_.notify(task)

    def find_the_task(self, task_name: str) -> Task | None:
        """Finds a task in any of the queues."""
        with self.list_lock_:
            for task in self.running_:
                if task.task_name() == task_name:
                    return task
            for task in self.ready_:
                if task.task_name() == task_name:
                    return task
        return None

    def dispatch_task_to_run(self, task_name: str):
        task = self.find_the_task(task_name)
        if task:
            self.running_event_.notify(task)
        else:
            logging.warning(f"Cannot dispatch task {task_name} to run: not found.")

    def dispatch_task_to_done(self, task_name: str):
        task = self.find_the_task(task_name)
        if task:
            self.done_event_.notify(task)
        else:
            logging.warning(f"Cannot dispatch task {task_name} to done: not found.")

    def lock_task(self, name: str):
        self.lock_set_.add(name)

    def unlock_task(self, name: str):
        self.lock_set_.discard(name)

    def is_lock(self, name: str) -> bool:
        return name in self.lock_set_

    @override
    def data(self) -> MLoRAData | None:
        self._dispatch_task_in()

        with self.list_lock_:
            candidate_tasks = [
                task for task in self.ready_ if not self.is_lock(task.task_name())
            ]

            if not candidate_tasks:
                return None

            tasks_for_batch = self._select_tasks_for_batch(candidate_tasks)

            if not tasks_for_batch:
                return None

            batch_tokens: List[Tokens] = []
            data_configs: List[MLoRADataConfig] = []
            
            for task in tasks_for_batch:
                self.ready_.remove(task)
                self.running_.append(task)
                self.running_event_.notify(task)
                self.lock_task(task.task_name())
                logging.info(f"Task {task.task_name()} locked and moved to running.")

            scheduled_task_names = {t.task_name() for t in tasks_for_batch}
            for task in self.ready_:
                if task.task_name() not in scheduled_task_names:
                    task.waiting += 1
                else:
                    # This should not happen
                    task.waiting = 0

            # Collate data from all tasks in the batch
            start_idx = 0
            for task in tasks_for_batch:
                data, data_config = task.data(start_idx)
                for item in data_config:
                    item.task_name_ = task.task_name()
                
                data_configs.extend(data_config)
                batch_tokens.extend(data)
                start_idx += len(data)

        if not batch_tokens:
            return None

        batch_tokens, batch_masks = self._align_batch_tokens(batch_tokens, data_configs)

        return MLoRAData(
            batch_tokens=batch_tokens, batch_mask=batch_masks, data_config=data_configs
        )

    def task_step(self, task_name: str):
        with self.list_lock_:
            task_to_update = None
            for task in self.running_:
                if task.task_name() == task_name:
                    task_to_update = task
                    break
            
            if not task_to_update:
                logging.warning(f"Task {task_name} requested step but not found in running list.")

                self.unlock_task(task_name)
                return

            task_to_update.step()
            self.step_event_.notify(task_to_update)
            self.unlock_task(task_name)

            self.running_.remove(task_to_update)

            if task_to_update.is_done():
                logging.info(f"Task {task_name} is done after step.")
                self.done_event_.notify(task_to_update)
            elif task_to_update.is_terminate():
                logging.info(f"Task {task_name} is terminated after step.")
                self.terminate_event_.notify(task_to_update)
            else:
                logging.info(f"Task {task_name} completed step, moving back to ready.")
                self.ready_.append(task_to_update)
                self.ready_event_.notify(task_to_update)