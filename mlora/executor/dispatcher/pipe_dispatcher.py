from typing import List, Set, override, Dict
import threading
import time

from mlora.config.dispatcher import DispatcherConfig
from mlora.executor.task import Task
from mlora.model.args import Masks, MLoRAData, MLoRADataConfig, Tokens
from mlora.utils.gpu_state import GPUState, query_all_gpus, AdapterProfile
import logging

from .backend_dispatcher import BackendDispatcher


class PipeDispatcher(BackendDispatcher):
    
    # set of task names that are currently "locked" (ie being processed or 
    # should not be processed again until it is unlocked). Used to prevent 
    # the same task from being processed multiple times concurrently
    lock_set_: Set[str]

    def __init__(self, config: DispatcherConfig, adapter_profiles: Dict[str, AdapterProfile]) -> None:
        super().__init__(config)
        self.lock_set_ = set()
        self.adapter_profiles = adapter_profiles
        self.concurrency_num_ = config.concurrency_num_
        self.list_lock = threading.Lock()

    def _compute_placement(self, candidates: List[Task]):
        gpu_state = query_all_gpus()[0]

        def score_task(task):
            adapter = task.adapter_name()[0]
            prof    = self.adapter_profiles[adapter]

            free_memory = gpu_state.free_mem
            gpu_util = gpu_state.gpu_util

            extra_mem = free_memory - prof.total_memory_estimate(
                task.config_.batch_size_, 256)
            if extra_mem < 0:
                return float('-inf')

            memory_score = (free_memory - extra_mem) / free_memory 

            gpu_util_decimal = gpu_util / 100.0
            gpu_availability_fraction = 1.0 - gpu_util_decimal
            gpu_availability_fraction = max(0.0, min(1.0, gpu_availability_fraction))

            task_normalized_flops = prof.flops_per_token_fwd / max([p.flops_per_token_fwd for s, p in self.adapter_profiles.items()])

            compute_score = task_normalized_flops * gpu_availability_fraction

            wait_boost = min(task.waiting / 20, 1.0)
            aging_score = 1.0 + (wait_boost ** 2)

            logging.info(f"name: {task.task_name()} mem: {memory_score} comp: {compute_score} wait: {aging_score}")

            return (0.3 * memory_score) + (0.7 * compute_score) * aging_score + (0.1 * wait_boost)
        
        for c in candidates:
            logging.info(f"C: {c.config_.name_}, epoch: {c.now_epoch_}, waiting: {c.waiting}, score: {score_task(c)}")

        # return a copy of the list so that we don't modify candidates list itself
        return sorted(candidates, key=score_task, reverse=True)

    @override
    def _dispatch_task_in(self):
        # Update the terminate and ready queues and then dispatch tasks from the 
        # ready queue to the running queue as long as there is room 
        
        # ready task to terminate
        with self.list_lock:
            terminate_ready = [task for task in self.ready_ if task.is_terminate()]
            if terminate_ready:
                self.ready_ = [task for task in self.ready_ if not task.is_terminate()]
                for task in terminate_ready:
                    logging.info(f"Task {task.task_name()} terminated while in ready queue.")
                    self.terminate_event_.notify(task)

            terminate_running = [task for task in self.running_ if task.is_terminate()]
            if terminate_running:
                 self.running_ = [task for task in self.running_ if not task.is_terminate()]
                 for task in terminate_running:
                    logging.info(f"Task {task.task_name()} terminated while in running queue.")
                    self.terminate_event_.notify(task)

    def find_the_task(self, task_name: str) -> Task:
        # the worker do not really dispather the task
        # so we just find it in the read
        with self.list_lock:
             for task in self.running_:
                 if task.task_name() == task_name:
                     return task
             for task in self.ready_:
                 if task.task_name() == task_name:
                     return task
        return None

    # if not the head worker, we need to manully dispatch the task
    def dispatch_task_to_run(self, task_name: str):
        task = self.find_the_task(task_name)
        if task:
            self.running_event_.notify(task)
        else:
             logging.warning(f"Cannot dispatch task {task_name} to run: not found.")
            

    def dispatch_task_to_ready(self, task_name: str):
        task = self.find_the_task(task_name)
        if task:
            self.ready_event_.notify(task)
        else:
             logging.warning(f"Cannot dispatch task {task_name} to ready: not found.")

    def dispatch_task_to_done(self, task_name: str):
        task = self.find_the_task(task_name)
        if task:
            self.done_event_.notify(task)
        else:
             logging.warning(f"Cannot dispatch task {task_name} to done: not found.")

    def dispatch_task_to_terminal(self, task_name: str):
        task = self.find_the_task(task_name)
        if task:
            self.terminate_event_.notify(task)
        else:
             logging.warning(f"Cannot dispatch task {task_name} to terminate: not found.")

    def dispatch_task_to_step(self, task_name: str):
        task = self.find_the_task(task_name)
        if task:
            task.step()
            self.step_event_.notify(task)
        else:
             logging.warning(f"Cannot dispatch task {task_name} to step: not found.")

    def lock_task(self, name: str):
        self.lock_set_.add(name)

    def unlock_task(self, name: str):
        if name not in self.lock_set_:
            return
        self.lock_set_.remove(name)

    # used to check if a task is locked
    def is_lock(self, name: str):
        return name in self.lock_set_

    @override
    def data(self) -> MLoRAData | None:
        
        # re-update the queues before grabbing the new task to be scheduled
        self._dispatch_task_in()

        batch_tokens: List[Tokens] = []
        batch_masks: List[Masks] = []
        data_configs: List[MLoRADataConfig] = []

        # avoid locked tasks
        with self.list_lock:
            can_run_tasks = [
                task for task in self.ready_ if not self.is_lock(task.task_name())
            ]

            if len(can_run_tasks) == 0:
                return None
            
            # get all train data
            start_idx: int = 0
            # pipe dispatcher just run one task

            num_to_run = min(len(can_run_tasks), self.concurrency_num_ - len(self.running_))
            if num_to_run <= 0:
                return None

            task = self._compute_placement(can_run_tasks)[0]
            logging.info(f"Selected task: {task.task_name()}")

            for t in self.ready_:
                if t != task:
                    t.waiting += 1
                else:
                    t.waiting = 0

            try:
                self.ready_.remove(task)
                self.running_.append(task)
                self.running_event_.notify(task)
                self.lock_task(task.task_name())
                logging.info(f"Task {task.task_name()} moved to running.")
            except ValueError:
                    # Handle error: maybe skip task, maybe raise exception
                    return None

            data, data_config = task.data(start_idx)

        # for unlock the task
            for item in data_config:
                item.task_name_ = task.task_name()

            data_configs.extend(data_config)
            batch_tokens.extend(data)
            start_idx = start_idx + len(data)
            self.lock_task(task.task_name())

        # post process this batch data
            batch_tokens, batch_masks = self._align_batch_tokens(batch_tokens, data_configs)

        return MLoRAData(
            batch_tokens=batch_tokens, batch_mask=batch_masks, data_config=data_configs
        )

    def task_step(self, task_name: str):
        # in head worker the task must in running
        with self.list_lock:
            task_found = None
            for task in self.running_:
                if task.task_name() != task_name:
                    continue
                task_found = task
                break
            if not task_found:
                logging.warning(f"Task {task_name} requested step but not found in running list.")
                self.unlock_task(task)
            

            task_found.step()
            self.step_event_.notify(task_found)
            self.unlock_task(task_name)

            if task_found.is_done():
                logging.info(f"Task {task_name} is done after step.")
                self.running_.remove(task_found)
                self.done_event_.notify(task_found)
            elif task_found.is_terminate():
                 logging.info(f"Task {task_name} is terminated after step.")
                 self.running_.remove(task_found)
                 self.terminate_event_.notify(task_found)
            else:
                logging.info(f"Task {task_name} completed step, moving back to ready.")
                self.running_.remove(task_found)
                self.ready_.append(task_found)
                self.ready_event_.notify(task_found)

            # self._dispatch_task_out()
