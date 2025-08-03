import logging
import time
import uuid
import threading
from enum import Enum, auto
from typing import Any, Dict, List, OrderedDict, Tuple, cast
from dataclasses import dataclass

import torch

from mlora.config import MLoRAConfig
from mlora.config.task import TaskConfig
from mlora.config.config import DictConfig
from mlora.model.args import LinearInfo, MLoRAData, ModelData
from mlora.model.llm import LLMModel
from mlora.model.llm.model_llama import precompute_mask
from mlora.model.tokenizer import Tokenizer
from mlora.utils.gpu_state import AdapterProfile

from .dispatcher import DISPATCHER_CLASS, PipeDispatcher
from .executor import Executor
from .pipeline.function import RecvOperator, SendOperator
from .pipeline.messages import PipeMessage, PipeMessageType
from .pipeline.queue import DeviceSwapQueue
from .pipeline.rpc_transport import RpcTransport
from .pipeline.stream import CudaStream
from .task import Task
from flops_profiler.profiler import get_model_profile
from collections import namedtuple

import pynvml

class WorkerRole(Enum):
    HEAD = auto()
    MID = auto()
    TAIL = auto()


class PipeExecutor(Executor):
    role_: WorkerRole
    device_: str

    rank_: int
    world_size_: int
    # balance_: List[int]

    # info about model
    partial_model_: torch.nn.Sequential
    heads_: int
    model_name_: str
    recompute_: bool
    mlora_config: DictConfig

    input_queue_: DeviceSwapQueue
    transport_: RpcTransport

    # cache some tensor
    backward_cache_: Dict[int, torch.Tensor]
    input_cache_: Dict[int, MLoRAData]

    # also this
    dispatcher_: PipeDispatcher

    def __init__(
        self,
        model: LLMModel,
        tokenizer: Tokenizer,
        config: MLoRAConfig,
        device: str,
        rank: int,
        nodes: int,
        recompute: bool = False,
    ) -> None:
        self.model_ = model
        self.tokenizer_ = tokenizer
        self.heads_ = self.model_.n_heads_
        self.model_name_ = self.model_.name_or_path_
        self.mlora_config = config

        self.device_ = device
        self.rank_ = rank
        self.world_size_ = nodes

        self.hidden_size_ = self.model_.dim_

        self.backward_cache_ = {}
        self.input_cache_ = {}
        self.latency_events_ = {}

        self.recompute_ = recompute

        self.__init_worker()
        self.__init_partition()

        a = 0
        for partial_layer in self.partial_model_:
            logging.info(f"Layer {a}: {partial_layer.name()}")
            a += 1

        self.default_stream_ = CudaStream(torch.cuda.default_stream(self.device_))
        
        # n = len(self.mlora_config.adapters().items())
        n = config.dispatcher_.concurrency_num_

        # init the rpc and wait the cluster node ready
        self.transport_ = RpcTransport(
            self.rank_, self.world_size_, torch.device(self.device_)
        )

        # config.dispatcher_.concurrency_num_ = n # eventually I should get rid of the logic to process this entirely

        self.dispatcher_: PipeDispatcher = cast(
            PipeDispatcher, DISPATCHER_CLASS["pipe"](config.dispatcher_, {})
        )

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

    def __init_worker(self):
        # init the different worker
        if self.rank_ == 0:
            self.role_ = WorkerRole.HEAD
            self.input_queue_ = DeviceSwapQueue(
                torch.device("cpu"), torch.device(self.device_), 4, "input_data_queue"
            )
            self.input_queue_.start()
        elif self.rank_ == self.world_size_ - 1:
            self.role_ = WorkerRole.TAIL
        else:
            self.role_ = WorkerRole.MID

    def __init_partition(self) -> None:
        seq_model: torch.nn.Sequential = self.model_.sequential()

        balance = [len(seq_model) // self.world_size_] * self.world_size_
        for i in range(len(seq_model) % self.world_size_):
            balance[i] += 1 # LLMs are homogenous, add wherever (for now)

        start_module_idx = sum(balance[: self.rank_])
        end_module_idx = start_module_idx + balance[self.rank_]

        assert sum(balance) == len(seq_model) # otherwise I've done some math wrong :/

        self.partial_model_ = torch.nn.Sequential()

        logging.info(
            f"RANK-{self.rank_} in device {self.device_} to load module layers "
            f"from {start_module_idx} to {end_module_idx}."
        )

        for idx in range(start_module_idx, end_module_idx):
            # logging.info(seq_model[idx])
            self.partial_model_.append(seq_model[idx])

        assert len(self.partial_model_) == balance[self.rank_]

        del seq_model[:start_module_idx]
        if end_module_idx < sum(balance) - 1:
            del seq_model[end_module_idx+1:]
        del self.model_

        self.partial_model_.to_empty(device=self.device_)

        for name, module in self.partial_model_.named_modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        torch.cuda.empty_cache()

    def calculate_costs(self):
        adapter_profiles = {}

        for task in self.dispatcher_.ready_:
            name = task.config_.adapter_.name_
            logging.info(f"Profiling: {name}")

            DummyLoRAConfig = namedtuple("DummyLoRAConfig", ["adapter_name_"])

            task.switch_device(self.device_)
            for adapter_model in task.adapter_model():
                for partial_layer in self.partial_model_:
                    if partial_layer.name() == "Decoder":
                        partial_layer.wrapper_module_.load_adapter(adapter_model)

            first_module_name = self.partial_model_[0].name()
            if first_module_name == "Embedding":
                dummy_input = torch.ones((1, 1), dtype=torch.long, device=self.device_)
            else:
                dummy_input = torch.ones(
                    (1, 1, self.hidden_size_), dtype=torch.float16, device=self.device_)

            dummy_batch_data = ModelData(
                random_id_=0, task_name_=[name], batch_tokens_=None,
                batch_mask_=None, data_config_=[DummyLoRAConfig(adapter_name_=name)], enable_checkpoint_=False
            )
            dummy_mask = precompute_mask(dummy_input, self.heads_, self.device_, None)
            dummy_input_tuple = (dummy_input, dummy_mask, dummy_batch_data, False)

            fwd_flops, _, params = get_model_profile(
                    model=self.partial_model_, args=(dummy_input_tuple,),
                    print_profile=False, as_string=False,
                    ignore_modules=[torch.nn.Dropout, torch.nn.LayerNorm]
            )

            for _ in range(5):
                self.partial_model_(dummy_input_tuple)
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            self.partial_model_(dummy_input_tuple)
            end_event.record()
            
            torch.cuda.synchronize()
            latency_ms = start_event.elapsed_time(end_event)

            param_bytes = (params * 2) / (1024 * 1024)
            bwd_flops = fwd_flops * 2
            adapter_profiles[name] = AdapterProfile(
                param_bytes=1,
                flops_per_token_fwd=1,
                flops_per_token_bwd=1,
                latency_ms=1,
            )
            logging.info(f"... Profile for {name}: {adapter_profiles[name]}")

            for adapter_name in task.adapter_name():
                for partial_layer in self.partial_model_:
                    if partial_layer.name() == "Decoder":
                        partial_layer.wrapper_module_.offload_adapter(adapter_name)

            task.switch_device("cpu")

        self.dispatcher_.update_adapter_profiles(adapter_profiles)

    def __head_worker_run(self):
        while True:
            # we get the model's output, and calc the loss
            self.__process_comm()
            self.__process_backward()
            self.__process_output()
            self.__process_input()
            time.sleep(1 / 100000)

    def __not_head_worker_run(self):
        while True:
            self.__process_comm()
            self.__process_backward()
            self.__process_forward()
            time.sleep(1 / 100000)

    def __head_process_step(self, message: PipeMessage):
        assert message.model_data_ is not None
        train_data: MLoRAData = self.input_cache_[message.model_data_.random_id_]

        # like dpo one task have two data config
        task_names = set()
        for item in train_data.data_config_:
            task_names.add(item.task_name_)

        for task_name in task_names:
            self.dispatcher_.task_step(task_name)
            self.dispatcher_.unlock_task(task_name)

        assert message.model_data_ is not None
        del self.input_cache_[message.model_data_.random_id_]

    def __process_backward(self):
        message = self.transport_.recv_message(PipeMessageType.GRADIENTS, block=False)
        if message is None:
            return

        logging.debug(
            f"Recv the gradients - {str(message.msg_id_)[:8]} from {message.src_}."
        )

        msg_id = message.msg_id_

        tn = message.model_data_.task_name_[0]
        key = (tn, msg_id)

        assert key in self.backward_cache_

        phony: torch.Tensor = self.backward_cache_.pop(key)
        phony.grad_fn.grad_from_next_worker = message.tensor_data_  # type: ignore

        gradient_tensor = message.tensor_data_.to(self.device_)
        phony.grad_fn.grad_from_next_worker = gradient_tensor

        if self.role_ == WorkerRole.HEAD:
            bwd_start_event = torch.cuda.Event(enable_timing=True)
            bwd_end_event = torch.cuda.Event(enable_timing=True)

            bwd_start_event.record()
            stop_event, results, thread = self._start_gpu_monitor()
            
            phony.backward()

            self._stop_gpu_monitor("Backward Pass", stop_event, results, thread)
            bwd_end_event.record()

            model_data = message.model_data_
            batch_id = model_data.random_id_
            
            # Check if we have a start event for this batch
            if batch_id in self.latency_events_:
                start_event = self.latency_events_.pop(batch_id) # pop to clean up
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                
                torch.cuda.synchronize()
                
                total_latency_ms = start_event.elapsed_time(end_event)
                logging.info(
                    f"   Total Batch Latency (Fwd->Bwd): {total_latency_ms:.4f} ms"
                )
        else:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            stop_event, results, thread = self._start_gpu_monitor()
    
            phony.backward()

            self._stop_gpu_monitor("Backward Pass", stop_event, results, thread)
            end_event.record()
            
            torch.cuda.synchronize()
            latency_ms = start_event.elapsed_time(end_event)
            logging.info(f"   Backward Pass Latency (Rank {self.rank_}): {latency_ms:.4f} ms")

        if self.role_ == WorkerRole.HEAD:
            self.__head_process_step(message)
        else:
            assert message.model_data_ is not None
            for task_name in message.model_data_.task_name_:
                self.dispatcher_.dispatch_task_to_step(task_name)

    def __process_forward(self):
        assert self.role_ != WorkerRole.HEAD

        # recv the tensors from prev-worker
        message = self.transport_.recv_message(PipeMessageType.ACTIVATIONS, block=False)
        if message is None:
            return

        logging.debug(
            f"Recv the activations - {str(message.msg_id_)[:8]} from {message.src_}."
        )

        data = RecvOperator.apply(
            torch.tensor(1.0, requires_grad=True), self.transport_, message
        )

        # we need to wait the default stream calcuate all tensor
        # and then send it, so we hook the pre stage fn to poll the stream
        data.grad_fn.pre_stage_fn = self.default_stream_.poll  # type: ignore
        assert message.model_data_ is not None

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        stop_event, results, thread = self._start_gpu_monitor()
        
        data = self.__forward(data, message.model_data_)

        self._stop_gpu_monitor("Mid/Tail Forward Pass", stop_event, results, thread)
        end_event.record()
        
        torch.cuda.synchronize() # Wait for the forward pass to complete
        latency_ms = start_event.elapsed_time(end_event)
        logging.info(f"   Forward Pass Latency (Rank {self.rank_}): {latency_ms:.4f} ms")

        self.default_stream_.poll()
        assert message.model_data_ is not None
        return self.__send_activations(data, message.model_data_)

    def __process_comm(self):
        try:
            msg: PipeMessage = self.transport_.recv_comm(
                PipeMessageType.COMM, block=False
            )
            comm_data = msg.comm_data_
        except Exception:
            return

        if comm_data["comm"] == "task_add":
            self.add_task(comm_data["data"])
        elif comm_data["comm"] == "task_running":
            self.dispatcher_.dispatch_task_to_run(comm_data["data"])
        elif comm_data["comm"] == "task_ready":
            self.dispatcher_.dispatch_task_to_ready(comm_data["data"])
        elif comm_data["comm"] == "task_done":
            self.dispatcher_.dispatch_task_to_done(comm_data["data"])
        elif comm_data["comm"] == "task_terminal":
            self.dispatcher_.dispatch_task_to_terminal(comm_data["data"])
        else:
            raise NotImplementedError

    def __process_output(self):
        assert self.role_ == WorkerRole.HEAD

        # recv the tensors from prev-worker
        message = self.transport_.recv_message(PipeMessageType.ACTIVATIONS, block=False)
        if message is None:
            return

        logging.debug(
            f"Recv the activations - {str(message.msg_id_)[:8]} from {message.src_}."
        )

        output: torch.Tensor = RecvOperator.apply(
            torch.tensor(1.0, requires_grad=True), self.transport_, message
        )

        # we need to wait the default stream calcuate all tensor
        # and then send it, so we hook the pre stage fn to poll the stream
        output.grad_fn.pre_stage_fn = self.default_stream_.poll  # type: ignore

        assert message.model_data_ is not None
        train_data: MLoRAData = self.input_cache_[message.model_data_.random_id_]
        labels = torch.tensor(train_data.batch_tokens_, dtype=torch.long)
        masks = torch.tensor(train_data.batch_mask_)

        total_loss: torch.Tensor | None = None
        
        for config in train_data.data_config_:
            loss = config.loss_fn_(output, labels, masks)
            if loss is None:
                continue
            
            logging.info(f"    Component Loss ({config.adapter_name_}): {loss.item()}")
            
            total_loss = loss if total_loss is None else total_loss + loss

        if total_loss is not None:
            logging.info(f"Total Batch Loss: {total_loss.item()}")
            total_loss.backward()
        else:
            logging.warning("Batch produced no loss value.")

    def __process_input(self):
        train_data: MLoRAData | None = self.dispatcher_.data()
        if train_data is None:
            return
        # step1. get the model data and execute the forward
        tensor_data = torch.tensor(
            train_data.batch_tokens_,
            dtype=torch.long,
            device=self.device_,
            requires_grad=False,
        )

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        stop_event, results, thread = self._start_gpu_monitor()
        self.latency_events_[train_data.model_data().random_id_] = start_event

        hidden_data = self.__forward(tensor_data, train_data.model_data())

        self._stop_gpu_monitor("Mid/Tail Forward Pass", stop_event, results, thread)
        end_event.record()
        torch.cuda.synchronize()
        latency_ms = start_event.elapsed_time(end_event)

        logging.info(f"    Head Node Forward Latency: {latency_ms:.4f} ms")

        # step2. then send the hidden state to next worker
        self.default_stream_.poll()
        self.__send_activations(hidden_data, train_data.model_data())

        # step3. cache the input, we need it to calc the loss
        self.input_cache_[train_data.model_data().random_id_] = train_data

    def __send_activations(self, tensor_data: torch.Tensor, batch_data: ModelData):
        assert isinstance(tensor_data, torch.Tensor)
        assert batch_data is None or isinstance(batch_data, ModelData)

        msg_id = uuid.uuid4().int
        assert msg_id not in self.backward_cache_

        phony: torch.Tensor = SendOperator.apply(
            torch.tensor(1.0, requires_grad=True),
            tensor_data,
            self.transport_,
            msg_id,
            batch_data,
        )

        tn = batch_data.task_name_[0]
        self.backward_cache_[(tn, msg_id)] = phony

    def __send_comm(self, data: Any):
        self.transport_.send_comm(PipeMessageType.COMM, data)

    def __forward(self, tensor_data: torch.Tensor, batch_data: ModelData):
        mask = precompute_mask(
            tensor_data, self.heads_, self.device_, batch_data.batch_mask_
        )
        data = (tensor_data, mask, batch_data, self.recompute_)

        for seq in self.partial_model_:
            data = seq.forward(data)

        return data[0]

    def execute(self) -> None:
        if self.role_ == WorkerRole.HEAD:
            self.__head_worker_run()
        elif self.role_ == WorkerRole.MID or self.role_ == WorkerRole.TAIL:
            self.__not_head_worker_run()
        else:
            raise NotImplementedError

    def add_task(self, config: TaskConfig):
        if self.role_ != WorkerRole.TAIL:
            self.__send_comm({"comm": "task_add", "data": config})
        if self.role_ != WorkerRole.HEAD:
            # only the head worker need to load dataset
            config.dataset_ = None
        self.dispatcher_.add_task(config, self.model_name_)

    def __task_init_hook(self, task: Task):
        logging.info(
            f"Init {task.task_type()} : {task.task_name()} "
            + f"task with adapters: {task.adapter_name()}"
        )
        task.prepare(self.__linears_info(), self.tokenizer_)

        # task.switch_device(self.device_)
        # for adapter_model in task.adapter_model():
        #     for partial_layer in self.partial_model_:
        #         if partial_layer.name() != "Decoder":
        #             continue
        #         partial_layer.wrapper_module_.load_adapter(adapter_model)

    def __task_to_running_hook(self, task: Task):
        logging.info(f"Task to running, need to load adapters: {task.adapter_name()}")
        if self.role_ != WorkerRole.TAIL:
            self.__send_comm({"comm": "task_running", "data": task.task_name()})

        task.switch_device(self.device_)
        for adapter_model in task.adapter_model():
            for partial_layer in self.partial_model_:
                if partial_layer.name() != "Decoder":
                    continue
                partial_layer.wrapper_module_.load_adapter(adapter_model)

    def __task_to_ready_hook(self, task: Task):
        logging.info(f"Base model offload adapters: {task.adapter_name()}")
        if self.role_ != WorkerRole.TAIL:
            self.__send_comm({"comm": "task_ready", "data": task.task_name()})

        task.switch_device("cpu")
        for adapter_name in task.adapter_name():
            for partial_layer in self.partial_model_:
                if partial_layer.name() != "Decoder":
                    continue
                partial_layer.wrapper_module_.offload_adapter(adapter_name)

    def __task_to_done_hook(self, task: Task):
        logging.info(f"Finish and base model offload adapter - {task.adapter_name()}")
        if self.role_ != WorkerRole.TAIL:
            self.__send_comm({"comm": "task_done", "data": task.task_name()})

        task.switch_device("cpu")
        for adapter_name in task.adapter_name():
            for partial_layer in self.partial_model_:
                if partial_layer.name() != "Decoder":
                    continue
                partial_layer.wrapper_module_.offload_adapter(adapter_name)
        task.done(is_pipeline=self.rank_)

    def __task_to_terminate_hook(self, task: Task):
        logging.info(f"Task - {task.task_name()} terminate.")
        if self.role_ != WorkerRole.TAIL:
            self.__send_comm({"comm": "task_terminal", "data": task.task_name()})

        task.switch_device("cpu")
        for adapter_name in task.adapter_name():
            for partial_layer in self.partial_model_:
                if partial_layer.name() != "Decoder":
                    continue
                partial_layer.wrapper_module_.offload_adapter(adapter_name)
        task.terminate()

    def __linears_info(self) -> OrderedDict[str, LinearInfo]:
        ret_val = OrderedDict()
        for module in self.partial_model_:
            if module.name() != "Decoder":
                continue
            ret_val.update(module.wrapper_module_.linears_info())
        return ret_val
