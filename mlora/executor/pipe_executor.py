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
from mlora.model.args import LinearInfo, MLoRAData, ModelData
from mlora.model.llm import LLMModel
from mlora.model.llm.model_llama import precompute_mask
from mlora.model.tokenizer import Tokenizer
from mlora.utils.gpu_state import AdapterProfile
from mlora.utils.shutdown import is_shutdown_requested

from .dispatcher import DISPATCHER_CLASS, PipeDispatcher
from .executor import Executor
from .pipeline.function import RecvOperator, SendOperator
from .pipeline.messages import PipeMessage, PipeMessageType
from .pipeline.queue import DeviceSwapQueue
from .pipeline.rpc_transport import RpcTransport
from .pipeline.rpc_transport_block import RpcTransportBlock
from .pipeline.stream import CudaStream
from .task import Task


class WorkerRole(Enum):
    HEAD = auto()
    MID = auto()
    TAIL = auto()


class PipeExecutor(Executor):
    role_: WorkerRole
    device_: str

    rank_: int
    world_size_: int
    balance_: List[int]

    # info about model
    partial_model_: torch.nn.Sequential
    heads_: int
    model_name_: str
    recompute_: bool

    input_queue_: DeviceSwapQueue
    transport_: RpcTransport

    # cache some tensor
    backward_cache_: Dict[int, torch.Tensor]
    input_cache_: Dict[int, MLoRAData]

    cache_forward_: torch.Tensor

    # also this
    adapter_profiles: Dict[str, AdapterProfile]
    dispatcher_: PipeDispatcher
    stream_pools: List[Tuple[CudaStream, CudaStream, CudaStream]]
    stream_taken: List[bool]
    task_slot: Dict[str, int]
    slot_lock: threading.Lock

    def __init__(
        self,
        model: LLMModel,
        tokenizer: Tokenizer,
        config: MLoRAConfig,
        device: str,
        rank: int,
        balance: List[int],
        recompute: bool = False,
        is_block: bool = False,
    ) -> None:
        self.model_ = model
        self.tokenizer_ = tokenizer
        self.heads_ = self.model_.n_heads_
        self.model_name_ = self.model_.name_or_path_

        self.device_ = device
        self.rank_ = rank
        self.balance_ = balance
        self.world_size_ = len(balance)

        self.slot_lock = threading.Lock()

        self.backward_cache_ = {}
        self.input_cache_ = {}

        self.cache_forward_ = None

        self.recompute_ = recompute

        self.__init_worker()
        self.__init_partition()

        self.__calculate_costs()

        self.default_stream_ = CudaStream(torch.cuda.default_stream(self.device_))
        
        n = len(self.mlora_config.adapters().items())
        # n = config.dispatcher_.concurrency_num_

        self.stream_pools = []
        self.stream_taken = [False] * n
        for _ in range(n):
            self.stream_pools.append((
                CudaStream(torch.cuda.Stream(device=self.device_)),
                CudaStream(torch.cuda.Stream(device=self.device_)),
                CudaStream(torch.cuda.Stream(device=self.device_)),
            ))
        self.task_slot = {}

        # init the rpc and wait the cluster node ready
        if is_block:
            self.transport_ = RpcTransportBlock(
                self.rank_, self.world_size_, torch.device(self.device_)
            )
        else:
            self.transport_ = RpcTransport(
                self.rank_, self.world_size_, torch.device(self.device_)
            )

        # config.dispatcher_.concurrency_num_ = n # eventually I should get rid of the logic to process this entirely

        self.dispatcher_: PipeDispatcher = cast(
            PipeDispatcher, DISPATCHER_CLASS["pipe"](config.dispatcher_, self.adapter_profiles)
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
        balance = self.balance_[self.rank_]
        start_module_idx = sum(self.balance_[: self.rank_])
        end_module_idx = start_module_idx + balance
        logging.info(
            f"RANK-{self.rank_} in device {self.device_} to load module layers "
            f"from {start_module_idx} to {end_module_idx}."
        )

        seq_model: torch.nn.Sequential = self.model_.sequential()
        assert sum(self.balance_) == len(seq_model)

        self.partial_model_ = torch.nn.Sequential()

        for idx in range(start_module_idx, end_module_idx):
            self.partial_model_.append(seq_model[idx])

        assert len(self.partial_model_) == balance

        del seq_model[:start_module_idx]
        del seq_model[balance:]
        del self.model_

        torch.cuda.empty_cache()

    def __calculate_costs(self):
        self.adapter_profiles: Dict[str, AdapterProfile] = {}

        base_layers = list(self.partial_model_)

        for name, adapter in self.mlora_config.adapters().items():
            adapter = adapter.export()
            if adapter["peft_type"] != "LORA":
                continue

            r = adapter["r"]
            targets = adapter["target_modules"]

            total_params     = 0
            total_fwd_flops  = 0
            total_bwd_flops  = 0

            for layer in base_layers:
                decoder = getattr(layer, "wrapper_module_", None)
                if decoder is None:
                    continue

                # attention (is all you need)
                attn = getattr(decoder, "attn_", None)
                if attn is not None:
                    for proj in ("wq_", "wk_", "wv_", "wo_"):
                        lin = getattr(attn, proj, None)
                        real_lin = lin.weight_ if hasattr(lin, "weight_") else lin
                        in_f, out_f = getattr(real_lin, "in_features", None), getattr(real_lin, "out_features", None)
                        mods = r * in_f + out_f * r
                        total_params    += mods
                        total_fwd_flops += mods
                        total_bwd_flops += 2 * mods

                # mlp
                mlp = getattr(decoder, "mlp_", None)
                if mlp is not None:
                    for proj in ("gate_", "down_", "up_"):
                        lin = getattr(mlp, proj, None)
                        real_lin = lin.weight_ if hasattr(lin, "weight_") else lin
                        in_f, out_f = getattr(real_lin, "in_features", None), getattr(real_lin, "out_features", None)
                        mods = r * in_f + out_f * r
                        total_params    += mods
                        total_fwd_flops += mods
                        total_bwd_flops += 2 * mods

            param_bytes = total_params * 2 // (1024 * 1024)  # fp16
            self.adapter_profiles[name] = AdapterProfile(
                param_bytes           = param_bytes,
                flops_per_token_fwd   = total_fwd_flops,
                flops_per_token_bwd   = total_bwd_flops,
            )

    def __head_worker_run(self):
        logging.info(f"PipeExecutor (Rank {self.rank_}) Head worker run starting")
        while not is_shutdown_requested():
            # we get the model's output, and calc the loss
            self.__process_comm()
            self.__process_backward()
            self.__process_output()
            self.__process_input()
            
            if is_shutdown_requested():
                logging.info(f"PipeExecutor (Rank {self.rank_}) Head worker detected shutdown. Exiting loop.")
                break
            time.sleep(1 / 100000)
            
        logging.info(f"PipeExecutor (Rank {self.rank_}) Head worker loop finished.")

    def __not_head_worker_run(self):
        logging.info(f"PipeExecutor (Rank {self.rank_}) Non-head worker run starting")
        while not is_shutdown_requested():
            self.__process_comm()
            self.__process_backward()
            self.__process_forward()
            
            if is_shutdown_requested():
                logging.info(f"PipeExecutor (Rank {self.rank_}) Non-Head worker detected shutdown. Exiting loop.")
                break
            
            time.sleep(1 / 100000)
            
        logging.info(f"PipeExecutor (Rank {self.rank_}) Non-Head worker loop finished.")

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

        logging.info(
            f"Recv the gradients - {str(message.msg_id_)[:8]} from {message.src_}."
        )

        msg_id = message.msg_id_

        tn = message.model_data_.task_name_[0]
        key = (tn, msg_id)

        assert key in self.backward_cache_

        phony: torch.Tensor = self.backward_cache_.pop(key)
        phony.grad_fn.grad_from_next_worker = message.tensor_data_  # type: ignore

        # ----- dev/kev -------
        slot = self.task_slot[tn]
        _, comp_stream, _ = self.stream_pools[slot]

        gradient_tensor = message.tensor_data_.to(self.device_)
        phony.grad_fn.grad_from_next_worker = gradient_tensor

        with torch.cuda.stream(comp_stream.stream_):
            phony.backward()
            # TODO: profiler had a del self.backward_cache_[msg_id] right after phony.backward() and before the 
            # * if self.cache_forward_ is not None if statement
        comp_stream.stream_.synchronize() ##HERE, needs fix
        # ----- dev/kev -------
        
        # TODO: the profiler added this self.cache_forward if-statement below - double check that 
        # this is fine with the added code above
        if self.cache_forward_ is not None:
            self.cache_forward_.grad_fn.model_data_.communication_time_ = (
                message.model_data_.communication_time_
            )

        if self.role_ == WorkerRole.HEAD:
            logging.info(
                f"total time: {time.time() - message.model_data_.computation_time_} communication time: {message.model_data_.communication_time_}"
            )
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

        logging.info(
            f"Recv the activations - {str(message.msg_id_)[:8]} from {message.src_}."
        )

        tn = message.model_data_.task_name_[0]
        slot = self.task_slot[tn]
        recv_stream, comp_stream, _ = self.stream_pools[slot]

        with torch.cuda.stream(recv_stream.stream_):
            data = RecvOperator.apply(
                torch.tensor(1.0, requires_grad=True), self.transport_, message
            )

        comp_stream.stream_.wait_stream(recv_stream.stream_)

        # we need to wait the default stream calcuate all tensor
        # and then send it, so we hook the pre stage fn to poll the stream
        # data.grad_fn.pre_stage_fn = comp_stream.poll  # type: ignore
        assert message.model_data_ is not None

        # profiling
        self.cache_forward_ = data

        
        with torch.cuda.stream(comp_stream.stream_):
            data = self.__forward(data, message.model_data_)
            # todo: here
        
        # comp_stream.stream_.synchronize()   # UNCOMMENT FOR NANs
        comp_stream.poll()
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

        tn = message.model_data_.task_name_[0]
        slot = self.task_slot[tn]
        recv_stream, comp_stream, _ = self.stream_pools[slot]

        with torch.cuda.stream(recv_stream.stream_):
            output: torch.Tensor = RecvOperator.apply(
                torch.tensor(1.0, requires_grad=True), self.transport_, message
            )

        comp_stream.stream_.wait_stream(recv_stream.stream_)

        # we need to wait the default stream calcuate all tensor
        # and then send it, so we hook the pre stage fn to poll the stream
        # output.grad_fn.pre_stage_fn = comp_stream.poll  # type: ignore

        assert message.model_data_ is not None
        train_data: MLoRAData = self.input_cache_[message.model_data_.random_id_]
        labels = torch.tensor(train_data.batch_tokens_, dtype=torch.long)
        masks = torch.tensor(train_data.batch_mask_)

        total_loss: torch.Tensor | None = None

        with torch.cuda.stream(comp_stream.stream_):
            for config in train_data.data_config_:
                loss = config.loss_fn_(output, labels, masks)
                if loss is None:
                    continue
                total_loss = loss if total_loss is None else total_loss + loss

                if total_loss is not None:
                    total_loss.backward()   # TODO: here
        # comp_stream.stream_.synchronize() ##HERE, needs fix - UNCOMMENT IF NANs

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


        # ----- dev/kev -------
        tn = train_data.data_config_[0].task_name_
        slot = self.task_slot[tn]
        _, comp_stream, _ = self.stream_pools[slot]

        with torch.cuda.stream(comp_stream.stream_):
            # ----- profiling -------  
            train_data.computation_time_ = time.time()
            # -- dev/kev ---
            hidden_data = self.__forward(tensor_data, train_data.model_data())
            # TODO: here

        # comp_stream.stream_.synchronize() # UNCOMMENT IF NANs
        # step2. then send the hidden state to next worker
        comp_stream.poll()
        self.__send_activations(hidden_data, train_data.model_data())

        # step3. cache the input, we need it to calc the loss
        self.input_cache_[train_data.model_data().random_id_] = train_data

    def __send_activations(self, tensor_data: torch.Tensor, batch_data: ModelData):
        assert isinstance(tensor_data, torch.Tensor)
        assert batch_data is None or isinstance(batch_data, ModelData)

        msg_id = uuid.uuid4().int
        assert msg_id not in self.backward_cache_

        tn = batch_data.task_name_[0]
        slot = self.task_slot[tn]
        _, comp_stream, send_stream = self.stream_pools[slot]

        send_stream.stream_.wait_stream(comp_stream.stream_)

        with torch.cuda.stream(send_stream.stream_):
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
        try:
            if self.role_ == WorkerRole.HEAD:
                self.__head_worker_run()
            elif self.role_ == WorkerRole.MID or self.role_ == WorkerRole.TAIL:
                self.__not_head_worker_run()
            else:
                raise NotImplementedError
        finally:
            logging.info(f"PipeExecutor (Rank: {self.rank_}) stopping RPC transport. In execute()")
            self.transport_.stop()
            logging.info(f"PipeExecutor (Rank: {self.rank_}) stopped RPC transport. In execute()")

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

        task.switch_device(self.device_)
        for adapter_model in task.adapter_model():
            for partial_layer in self.partial_model_:
                if partial_layer.name() != "Decoder":
                    continue
                partial_layer.wrapper_module_.load_adapter(adapter_model)

    def __task_to_running_hook(self, task: Task):
        logging.info(f"Task to running, need to load adapters: {task.adapter_name()}")
        if self.role_ != WorkerRole.TAIL:
            self.__send_comm({"comm": "task_running", "data": task.task_name()})
        
        with self.slot_lock:
            for i, in_use in enumerate(self.stream_taken):
                if not in_use:
                    self.stream_taken[i]      = True
                    self.task_slot[task.task_name()] = i
                    break
            else:
                raise RuntimeError("No free stream slot for new task") #should never happen

        task.switch_device(self.device_)

    def __task_to_ready_hook(self, task: Task):
        logging.info(f"Base model offload adapters: {task.adapter_name()}")
        if self.role_ != WorkerRole.TAIL:
            self.__send_comm({"comm": "task_ready", "data": task.task_name()})

        if task.task_name() in self.task_slot:
            slot = self.task_slot[task.task_name()]
            recv_stream, comp_stream, send_stream = self.stream_pools[slot]

            self.stream_taken[slot] = False

        task.switch_device("cpu")

    def __task_to_done_hook(self, task: Task):
        logging.info(f"Finish and base model offload adapter - {task.adapter_name()}")
        if self.role_ != WorkerRole.TAIL:
            self.__send_comm({"comm": "task_done", "data": task.task_name()})

        slot = self.task_slot.pop(task.task_name(), None)
        if slot is not None:
            recv_stream, comp_stream, send_stream = self.stream_pools[slot]

            recv_stream.stream_.synchronize()
            comp_stream.stream_.synchronize()
            send_stream.stream_.synchronize()

            self.stream_taken[slot] = False

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

        slot = self.task_slot.pop(task.task_name(), None)
        if slot is not None:
            recv_stream, comp_stream, send_stream = self.stream_pools[slot]

            recv_stream.stream_.synchronize()
            comp_stream.stream_.synchronize()
            send_stream.stream_.synchronize()

            self.stream_taken[slot] = False

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
