import logging
import time
import uuid
import threading
from typing import Any, Dict, List, OrderedDict, Tuple, cast
from dataclasses import dataclass

import torch
import torch.distributed as dist

from mlora.config import MLoRAConfig
from mlora.config.task import TaskConfig
from mlora.config.config import DictConfig
from mlora.model.args import LinearInfo, MLoRAData, ModelData
from mlora.model.llm import LLMModel
from mlora.model.llm.model_llama_tp import LlamaModel_TP
from mlora.model.tokenizer import Tokenizer
from mlora.utils.gpu_state import AdapterProfile

from .dispatcher import DISPATCHER_CLASS, TensorParallelDispatcher
from .executor import Executor
from .task import Task
from flops_profiler.profiler import get_model_profile
from collections import namedtuple

import pynvml

class TPExecutor(Executor):
    device_: str
    rank_: int
    world_size_: int

    model_: LlamaModel_TP
    tokenizer_: Tokenizer
    mlora_config: DictConfig

    dispatcher_: TensorParallelDispatcher

    def __init__(
        self,
        model: LlamaModel_TP,
        tokenizer: Tokenizer,
        config: MLoRAConfig,
        device: str,
        rank: int,
        world_size: int,
    ) -> None:
        self.model_ = model
        self.tokenizer_ = tokenizer
        self.mlora_config = config

        self.device_ = device
        self.rank_ = rank
        self.world_size_ = world_size

        self.model_.to(self.device_)

        self.dispatcher_ = DISPATCHER_CLASS["tensor"](config.dispatcher_, {})

        hook_func = {
            "init": self.__task_init_hook,
            "running": self.__task_to_running_hook,
            "ready": self.__task_to_ready_hook,
            "done": self.__task_to_done_hook,
            "terminate": self.__task_to_terminate_hook,
        }

        for hook, cb in hook_func.items():
            self.dispatcher_.register_hook(hook, cb)

    def _poll_gpu_stats(self, stop_event: threading.Event, results: List[int]):
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(self.device_.split(":")[-1]))
            while not stop_event.is_set():
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                results.append(util.gpu)
                time.sleep(0.01)  # Poll every 10ms
        except Exception as e:
            logging.warning(f"GPU poll thread failed: {e}")
        finally:
            pynvml.nvmlShutdown()

    def _start_gpu_monitor(self) -> Tuple[threading.Event, List[int], threading.Thread]:
        stop_event = threading.Event()
        util_results = []
        monitor_thread = threading.Thread(
            target=self._poll_gpu_stats, args=(stop_event, util_results), daemon=True
        )
        monitor_thread.start()
        return stop_event, util_results, monitor_thread

    def _stop_gpu_monitor(
        self,
        context: str,
        stop_event: threading.Event,
        util_results: List[int],
        monitor_thread: threading.Thread,
    ):
        stop_event.set()
        monitor_thread.join()
        if util_results:
            avg_util = sum(util_results) / len(util_results)
            max_util = max(util_results)
            logging.info(
                f"GPU UTIL @ {context} (Rank {self.rank_}): "
                f"Avg: {avg_util:.2f}%, Max: {max_util}%"
            )

    def calculate_costs(self):
        ##FIXME
        adapter_profiles = {}
        for task in self.dispatcher_.ready_:
            name = task.config_.adapter_.name_
            logging.info(f"Creating dummy profile for: {name}")

            adapter_profiles[name] = AdapterProfile(
                param_bytes=1,
                flops_per_token_fwd=1,
                flops_per_token_bwd=1,
                latency_ms=1,
            )
            logging.info(f"... Dummy profile for {name}: {adapter_profiles[name]}")

        self.dispatcher_.update_adapter_profiles(adapter_profiles)

    def execute(self) -> None:
        while True:
            train_data: MLoRAData | None = self.dispatcher_.data()
            if train_data is None:
                time.sleep(1 / 100000)
                continue

            self.process_batch(train_data)

    def process_batch(self, train_data: MLoRAData):
        tokens = torch.tensor(
            train_data.batch_tokens_,
            dtype=torch.long,
            device=self.device_,
        )
        labels = torch.tensor(train_data.batch_tokens_, dtype=torch.long)
        masks = torch.tensor(train_data.batch_mask_)

        fwd_start_event = torch.cuda.Event(enable_timing=True)
        fwd_end_event = torch.cuda.Event(enable_timing=True)
        fwd_start_event.record()
        stop_event, results, thread = self._start_gpu_monitor()

        output = self.model_(train_data.model_data())

        self._stop_gpu_monitor("Forward Pass", stop_event, results, thread)
        fwd_end_event.record()
        torch.cuda.synchronize()
        fwd_latency_ms = fwd_start_event.elapsed_time(fwd_end_event)
        logging.info(f"    Forward Pass Latency (Rank {self.rank_}): {fwd_latency_ms:.4f} ms")

        total_loss = None
        for config in train_data.data_config_:
            loss = config.loss_fn_(output, labels, masks)
            if loss is not None:
                logging.info(f"    Component Loss ({config.adapter_name_}): {loss.item()}")
                total_loss = loss if total_loss is None else total_loss + loss

        if total_loss is not None:
            logging.info(f"Total Batch Loss: {total_loss.item()}")
            bwd_start_event = torch.cuda.Event(enable_timing=True)
            bwd_end_event = torch.cuda.Event(enable_timing=True)
            bwd_start_event.record()
            stop_event, results, thread = self._start_gpu_monitor()

            total_loss.backward()

            self._stop_gpu_monitor("Backward Pass", stop_event, results, thread)
            bwd_end_event.record()
            torch.cuda.synchronize()
            bwd_latency_ms = bwd_start_event.elapsed_time(bwd_end_event)
            logging.info(f"    Backward Pass Latency (Rank {self.rank_}): {bwd_latency_ms:.4f} ms")

            for param in self.model_.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad /= self.world_size_
        else:
            logging.warning("Batch produced no loss value.")

        task_names = {item.task_name_ for item in train_data.data_config_}
        for task_name in task_names:
            self.dispatcher_.task_step(task_name)
            self.dispatcher_.unlock_task(task_name)

    def add_task(self, config: TaskConfig):
        self.dispatcher_.add_task(config, self.model_.name_or_path_)

    def __task_init_hook(self, task: Task):
        logging.info(
            f"Init {task.task_type()} : {task.task_name()} "
            + f"task with adapters: {task.adapter_name()}"
        )
        task.prepare(self.__linears_info(), self.tokenizer_)

    def __task_to_running_hook(self, task: Task):
        logging.info(f"Task to running, need to load adapters: {task.adapter_name()}")
        task.switch_device(self.device_)
        for adapter_model in task.adapter_model():
            self.model_.load_adapter(task.adapter_name()[0], adapter_model)

    def __task_to_ready_hook(self, task: Task):
        logging.info(f"Base model offload adapters: {task.adapter_name()}")
        task.switch_device("cpu")
        for adapter_name in task.adapter_name():
            self.model_.offload_adapter(adapter_name)

    def __task_to_done_hook(self, task: Task):
        logging.info(f"Finish and base model offload adapter - {task.adapter_name()}")
        task.switch_device("cpu")
        for adapter_name in task.adapter_name():
            self.model_.offload_adapter(adapter_name)
        task.done()

    def __task_to_terminate_hook(self, task: Task):
        logging.info(f"Task - {task.task_name()} terminate.")
        task.switch_device("cpu")
        for adapter_name in task.adapter_name():
            self.model_.offload_adapter(adapter_name)
        task.terminate()

    def __linears_info(self) -> OrderedDict[str, LinearInfo]:
        return self.model_.linears_info()