import logging
import time
import uuid
import threading
import queue
import io
from enum import Enum, auto
from typing import Any, Dict, List, OrderedDict, Tuple, cast, Optional
from dataclasses import dataclass

import torch

from mlora.config import MLoRAConfig
from mlora.config.task import TaskConfig
from mlora.config.config import DictConfig
from mlora.model.args import LinearInfo, MLoRAData, ModelData, ModelDataConfig
from mlora.model.llm import LLMModel
from mlora.model.llm.model_llama import precompute_mask
from mlora.model.tokenizer import Tokenizer
from mlora.utils.gpu_state import AdapterProfile
from torch.profiler import profile as TorchProfile, schedule, ProfilerActivity, record_function, tensorboard_trace_handler
from torch.utils.tensorboard import SummaryWriter

from .dispatcher import DISPATCHER_CLASS, PipeDispatcher
from .executor import Executor
from .pipeline.function import RecvOperator, SendOperator
from .pipeline.nccl_transport import NcclTransport, PipeMessage, PipeMessageType
from .pipeline.stream import CudaStream
from .task import Task
from flops_profiler.profiler import get_model_profile
from collections import namedtuple
import os
import subprocess
from queue import Queue, Empty
import torch
import torch.nn.functional as F
import re

class WorkerRole(Enum):
    HEAD = auto()
    MID = auto()
    TAIL = auto()


def loss_fn(
    config: ModelDataConfig,
    input: torch.Tensor,        # [B, T, V] logits
    target: torch.Tensor,       # [B, T] token ids
    mask: torch.Tensor | None,  # [B, T] {0,1}
) -> torch.Tensor:
    bs = getattr(config, "batch_start_idx_", 0) or 0
    be = getattr(config, "batch_end_idx_", input.size(0)) or input.size(0)

    logits = input[bs:be, :-1, :]
    tgt    = target[bs:be, 1:]
    m      = mask[bs:be, 1:] if mask is not None else None

    V = logits.size(-1)
    logits = logits.reshape(-1, V).float()
    tgt    = tgt.reshape(-1).to(logits.device)

    if m is not None:
        m = m.reshape(-1).to(logits.device).bool()
        tgt = tgt.masked_fill(~m, -100)

    return F.cross_entropy(logits, tgt, ignore_index=-100, reduction="mean")

class PipeExecutor(Executor):
    role_: WorkerRole
    device_: str

    rank_: int
    world_size_: int

    # info about model
    partial_model_: torch.nn.Sequential
    heads_: int
    model_name_: str
    recompute_: bool
    mlora_config: DictConfig

    transport_: NcclTransport

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
        world_size: int,
        recompute: bool = False,
    ) -> None:
        self.model_ = model
        self.tokenizer_ = tokenizer
        self.heads_ = self.model_.n_heads_
        self.model_name_ = self.model_.name_or_path_
        self.mlora_config = config

        self.device_ = device
        self.rank_ = rank
        self.world_size_ = world_size

        self.hidden_size_ = self.model_.dim_

        self.backward_cache_ = {}
        self.input_cache_ = {}
        self.latency_events_ = {}

        self.recompute_ = recompute
        self.log_dir_ = getattr(config, "tb_log_dir_", None) or os.environ.get(
            "MLOTRA_TB_LOGDIR", f"/projects/beis/akanodia/CS-598-FAL-Project/runs/llama_job_1/rank{self.rank_}"
        )
        os.makedirs(self.log_dir_, exist_ok=True)

        wait_steps   = int(os.environ.get("MLOTRA_PROF_WAIT", "1"))
        warmup_steps = int(os.environ.get("MLOTRA_PROF_WARMUP", "1"))
        active_steps = int(os.environ.get("MLOTRA_PROF_ACTIVE", "47")) # profile the rest 47 etc after skipping first 3 (consider when colocated job finished, stop when bigger batch size finishes first)
        repeat_spans = int(os.environ.get("MLOTRA_PROF_REPEAT", "0"))  # 0 = run once

        self.profiler_ = TorchProfile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=wait_steps, warmup=warmup_steps, active=active_steps, repeat=repeat_spans),
            on_trace_ready=tensorboard_trace_handler(self.log_dir_),   # writes Chrome trace to TB
            record_shapes=True,
            profile_memory=True,                                       # includes CUDA memory
            with_stack=False,
        )
        self.profiler_.start()
        self.writer_ = SummaryWriter(log_dir=self.log_dir_, filename_suffix=f"_rank{self.rank_}")
        self.global_step_ = 0
        self.__init_worker()
        self.__init_partition()

        self.default_stream_ = CudaStream(torch.cuda.default_stream(self.device_))

        n = config.dispatcher_.concurrency_num_

        self.header_queue = queue.Queue()
        self.transport_ = NcclTransport(rank, world_size, device, header_queue=self.header_queue)

        # pipeline neighbors
        self.prev_rank = rank - 1 if rank > 0 else None
        self.next_rank = rank + 1 if rank < world_size - 1 else None

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


        # Optional: start nvidia-smi dmon collector if enabled
        if os.environ.get("MLOTRA_DMON", "0") == "1":
            self._start_dmon()

    # ---------------------------
    # GPU / Memory instrumentation
    # ---------------------------

    def _poll_gpu_stats(self, stop_event: threading.Event, results: List[int]):
        """Background sampler for instantaneous GPU utilization (%) via NVML."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(self.device_.split(":")[-1]))
            while not stop_event.is_set():
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                results.append(util.gpu)  # %
                time.sleep(0.01)
        except Exception as e:
            logging.warning(f"GPU poll thread failed: {e}")
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

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
                f"GPU UTIL @ {context} (Rank {self.rank_}): Avg: {avg_util:.2f}%, Max: {max_util}%"
            )

    def _device_total_bytes(self) -> int:
        try:
            dev = torch.device(self.device_)
            props = torch.cuda.get_device_properties(dev)
            return int(getattr(props, "total_memory", 0)) or 0
        except Exception:
            return 0

    def _tb_log_gpu_now(self, tag_prefix: str):
        """
        Push instantaneous GPU + memory stats to TensorBoard.
        Each call also increments step to guarantee plots advance.
        """
        if self.writer_ is None:
            return None
        try:
            dev = torch.device(self.device_)
            if torch.cuda.is_available():
                torch.cuda.synchronize(dev)

                mem_alloc = torch.cuda.memory_allocated(dev)
                mem_resv  = torch.cuda.memory_reserved(dev)
                device_total_bytes = self._device_total_bytes()

                util_now, nvml_used, nvml_total = None, None, None
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    idx = int(str(self.device_).split(":")[-1])
                    h = pynvml.nvmlDeviceGetHandleByIndex(idx)
                    u = pynvml.nvmlDeviceGetUtilizationRates(h)
                    util_now = float(u.gpu)
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(h)
                    nvml_used, nvml_total = float(meminfo.used), float(meminfo.total)
                    pynvml.nvmlShutdown()
                except Exception as e:
                    logging.debug(f"NVML unavailable: {e}")

                step = self.global_step_

                if util_now is not None:
                    self.writer_.add_scalar(f"{tag_prefix}/gpu_util_pct", util_now, step)

                self.writer_.add_scalar(f"{tag_prefix}/cuda_mem/allocated_bytes", mem_alloc, step)
                self.writer_.add_scalar(f"{tag_prefix}/cuda_mem/reserved_bytes",  mem_resv,  step)

                if device_total_bytes > 0:
                    self.writer_.add_scalar(f"{tag_prefix}/cuda_mem/allocated_pct", 
                                            (mem_alloc/device_total_bytes)*100.0, step)
                    self.writer_.add_scalar(f"{tag_prefix}/cuda_mem/reserved_pct",  
                                            (mem_resv/device_total_bytes)*100.0, step)

                if nvml_used is not None and nvml_total:
                    self.writer_.add_scalar(f"{tag_prefix}/nvml/device_used_bytes", nvml_used, step)
                    self.writer_.add_scalar(f"{tag_prefix}/nvml/device_used_pct", (nvml_used/nvml_total)*100.0, step)

                # immediately flush to disk so TB frontend sees updates
                self.writer_.flush()

                logging.info(
                    f"[Rank {self.rank_}] step {step} | GPU util={util_now}% "
                    f"| alloc={mem_alloc/1e6:.1f}MB ({(mem_alloc/device_total_bytes*100.0):.1f}%) "
                    f"| reserved={mem_resv/1e6:.1f}MB ({(mem_resv/device_total_bytes*100.0):.1f}%)"
                )
                # increment after logging so each call produces new X-axis point
                self.global_step_ += 1
                return util_now
        except Exception as e:
            logging.warning(f"TB GPU log failed: {e}")
        return None

    def _profiler_step(self, where: str):
        """Advance profiler once per pipeline 'step' and also log pipeline step marker."""
        if self.writer_:
            self.writer_.add_scalar("pipeline/step", self.global_step_, self.global_step_)
            self.writer_.flush()
        if self.profiler_ is not None:
            self.profiler_.step()


    # -------- dmon (optional) --------
    def _start_dmon(self):
        """Start `nvidia-smi dmon` and a reader thread to emit TB scalars. Set MLOTRA_DMON=1 to enable."""
        try:
            self._dmon_stop_evt_ = threading.Event()
            self._dmon_queue_ = Queue()
            self._dmon_proc_ = subprocess.Popen(
                ["nvidia-smi", "dmon", "-s", "pmf", "-d", "1", "-o", "DT"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )

            def _reader():
                header = None
                for line in self._dmon_proc_.stdout:
                    if self._dmon_stop_evt_.is_set():
                        break
                    line = line.strip()
                    if not line or line.startswith("#"):
                        if "pwr" in line and "sm" in line:
                            header = re.split(r"\s+", line.lstrip("#").strip())
                        continue
                    if header is None:
                        continue
                    cols = re.split(r"\s+", line)
                    row = dict(zip(header, cols))
                    self._dmon_queue_.put(row)

            def _consumer():
                while not self._dmon_stop_evt_.is_set():
                    try:
                        row = self._dmon_queue_.get(timeout=1.0)
                    except Empty:
                        continue
                    try:
                        if self.writer_:
                            for k in ("pwr", "sm", "mem", "fb"):
                                if k in row:
                                    self.writer_.add_scalar(f"dmon/{k}_rank{self.rank_}", float(row[k]), self.global_step_)
                            for k in ("rxpci", "txpci"):
                                if k in row:
                                    self.writer_.add_scalar(f"dmon/{k}_MBs_rank{self.rank_}", float(row[k]), self.global_step_)
                    except Exception:
                        pass

            self._dmon_thread_ = threading.Thread(target=_reader, daemon=True)
            self._dmon_thread_.start()
            threading.Thread(target=_consumer, daemon=True).start()
            logging.info("nvidia-smi dmon started for TB logging.")
        except Exception as e:
            logging.warning(f"Could not start nvidia-smi dmon: {e}")

    def _stop_dmon(self):
        if getattr(self, "_dmon_stop_evt_", None):
            self._dmon_stop_evt_.set()
        if getattr(self, "_dmon_proc_", None):
            try:
                self._dmon_proc_.terminate()
            except Exception:
                pass

    # ---------------------------
    # Pipeline setup
    # ---------------------------

    def __init_worker(self):
        # init the different worker
        if self.rank_ == 0:
            self.role_ = WorkerRole.HEAD
        elif self.rank_ == self.world_size_ - 1:
            self.role_ = WorkerRole.TAIL
        else:
            self.role_ = WorkerRole.MID

    def __init_partition(self) -> None:
        seq_model: torch.nn.Sequential = self.model_.sequential()

        balance = [len(seq_model) // self.world_size_] * self.world_size_
        for i in range(len(seq_model) % self.world_size_):
            balance[i] += 1

        start_module_idx = sum(balance[: self.rank_])
        end_module_idx = start_module_idx + balance[self.rank_]

        assert sum(balance) == len(seq_model)

        self.partial_model_ = torch.nn.Sequential()

        logging.info(
            f"RANK-{self.rank_} in device {self.device_} to load module layers "
            f"from {start_module_idx} to {end_module_idx}."
        )

        for idx in range(start_module_idx, end_module_idx):
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
        """
        Profile each READY task's adapter on this rank's partition and update the dispatcher
        with AdapterProfile(param_bytes, flops_per_token_fwd, flops_per_token_bwd, latency_ms).
        """
        adapter_profiles = {}

        for task in self.dispatcher_.ready_:
            name = task.config_.adapter_.name_
            logging.info(f"Profiling: {name}")

            task.switch_device(self.device_)

            for adapter_model in task.adapter_model():
                for partial_layer in self.partial_model_:
                    if partial_layer.name() == "Decoder":
                        try:
                            partial_layer.wrapper_module_.load_adapter(adapter_model)
                        except AssertionError:
                            pass

            first_module_name = self.partial_model_[0].name()
            if first_module_name == "Embedding":
                dummy_input = torch.ones((1, 1), dtype=torch.long, device=self.device_)
            else:
                dummy_input = torch.ones(
                    (1, 1, self.hidden_size_), dtype=torch.float16, device=self.device_
                )

            DummyLoRAConfig = namedtuple("DummyLoRAConfig", ["adapter_name_"])
            dummy_batch_data = ModelData(
                random_id_=0,
                task_name_=[name],
                batch_tokens_=None,
                batch_mask_=None,
                data_config_=[DummyLoRAConfig(adapter_name_=name)],
                enable_checkpoint_=False,
            )
            dummy_mask = precompute_mask(dummy_input, self.heads_, self.device_, None)
            dummy_tuple = (dummy_input, dummy_mask, dummy_batch_data, False)

            for _ in range(3):
                _ = self.partial_model_(dummy_tuple)

            try:
                fwd_flops, _, params = get_model_profile(
                    model=self.partial_model_,
                    args=(dummy_tuple,),
                    print_profile=False,
                    as_string=False,
                    ignore_modules=[torch.nn.Dropout, torch.nn.LayerNorm],
                )
            except Exception as e:
                logging.warning(f"get_model_profile failed on rank {self.rank_}: {e}")
                fwd_flops, params = 0.0, 0

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(self.device_)
            start_event.record()
            _ = self.partial_model_(dummy_tuple)
            end_event.record()
            torch.cuda.synchronize(self.device_)
            latency_ms = max(0.0, float(start_event.elapsed_time(end_event)))

            param_bytes = float(params) * 2.0
            bwd_flops = float(fwd_flops) * 2.0

            adapter_profiles[name] = AdapterProfile(
                param_bytes=param_bytes,
                flops_per_token_fwd=float(fwd_flops),
                flops_per_token_bwd=float(bwd_flops),
                latency_ms=latency_ms,
            )
            logging.info(f"... Profile for {name}: {adapter_profiles[name]}")

            for adapter_name in task.adapter_name():
                for partial_layer in self.partial_model_:
                    if partial_layer.name() == "Decoder":
                        try:
                            partial_layer.wrapper_module_.offload_adapter(adapter_name)
                        except Exception:
                            pass

            task.switch_device("cpu")

        self.dispatcher_.update_adapter_profiles(adapter_profiles)

    # ---------------------------
    # Pipeline loops
    # ---------------------------

    def __head_worker_run(self):
        while True:
            self.__recv_comm()
            self.__process_backward()
            self.__process_output()
            self.__process_input()
            time.sleep(1 / 100000)

    def __not_head_worker_run(self):
        while True:
            self.__recv_comm()
            self.__process_backward()
            self.__process_forward()
            time.sleep(1 / 100000)

    def __tail_worker_run(self):
        while True:
            self.__recv_comm()
            self.__process_tail_forward_and_backward()
            time.sleep(1 / 100000)

    def _fmt_peak_with_pct(self, peak_mb: float) -> str:
        total_mb = self._device_total_bytes() / (1024 * 1024)
        pct = (peak_mb / total_mb * 100.0) if total_mb else 0.0
        return f"{peak_mb:.2f} MB ({pct:.2f}% of device)"

    def __head_process_step(self, message):
        train_data: MLoRAData = self.input_cache_[message.random_id_]

        task_names = set()
        for item in train_data.data_config_:
            task_names.add(item.task_name_)

        for task_name in task_names:
            self.dispatcher_.task_step(task_name)
            self.dispatcher_.unlock_task(task_name)

        del self.input_cache_[message.random_id_]

    def __process_backward(self):
        message = self.transport_.recv_message("next")
        if message is None:
            return

        logging.info(f"[Rank {self.rank_}] Received gradient message from next.")
        logging.info(f"backward cache: {self.backward_cache_}")

        model_data = None
        if getattr(message, "meta_tensor_", None) is not None:
            meta_bytes = bytes(message.meta_tensor_.cpu().tolist())
            buffer = io.BytesIO(meta_bytes)
            model_data = torch.load(buffer)

        tn = model_data.task_name_[0]
        key = model_data.random_id_

        assert key in self.backward_cache_, f"Backward cache miss for key {key}"

        torch.cuda.reset_peak_memory_stats(device=self.device_)

        phony: torch.Tensor = self.backward_cache_.pop(key)
        gradient_tensor = message.tensor_data_.to(self.device_)
        phony.grad_fn.grad_from_next_worker = gradient_tensor

        if self.role_ == WorkerRole.HEAD:
            bwd_start_event = torch.cuda.Event(enable_timing=True)
            bwd_end_event = torch.cuda.Event(enable_timing=True)

            bwd_start_event.record()
            phony.backward()
            bwd_end_event.record()

            if key is not None and key in self.latency_events_:
                start_event = self.latency_events_.pop(key)
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                torch.cuda.synchronize()

                fwd_peak_memory_mb = torch.cuda.max_memory_allocated(self.device_) / (1024 * 1024)
                logging.info(f"Forward pass peak memory (Rank {self.rank_}): {self._fmt_peak_with_pct(fwd_peak_memory_mb)}")

                bwd_latency_ms = bwd_start_event.elapsed_time(bwd_end_event)
                logging.info(f"   Head Backward Pass Latency: {bwd_latency_ms:.4f} ms")

                total_latency_ms = start_event.elapsed_time(end_event)
                logging.info(f"   Total Batch Latency (Fwd->Bwd): {total_latency_ms:.4f} ms")

            # snapshot GPU/mem after head backward
            self._tb_log_gpu_now(f"gpu_rank{self.rank_}")
            self._profiler_step("backward")

        else:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.reset_peak_memory_stats(device=self.device_)

            start_event.record()
            phony.backward()
            end_event.record()

            bwd_peak_memory_mb = torch.cuda.max_memory_allocated(self.device_) / (1024 * 1024)
            logging.info(f"Backward pass peak memory (Rank {self.rank_}): {self._fmt_peak_with_pct(bwd_peak_memory_mb)}")

            torch.cuda.synchronize()
            latency_ms = start_event.elapsed_time(end_event)
            logging.info(f"   Backward Pass Latency (Rank {self.rank_}): {latency_ms:.4f} ms")

            # snapshot for mid workers as well
            self._tb_log_gpu_now(f"gpu_rank{self.rank_}")
            self._profiler_step("backward")

        if self.role_ == WorkerRole.HEAD:
            self.__head_process_step(model_data)
        else:
            for task_name in model_data.task_name_:
                self.dispatcher_.dispatch_task_to_step(task_name)

    def __process_forward(self):
        assert self.role_ != WorkerRole.HEAD

        message = self.transport_.recv_message("prev", only_type=PipeMessageType.TENSOR)
        if message is None:
            return

        logging.info(f"[Rank {self.rank_}] Received activation message from prev.")

        model_data = None
        if getattr(message, "meta_tensor_", None) is not None:
            meta_bytes = bytes(message.meta_tensor_.cpu().tolist())
            buffer = io.BytesIO(meta_bytes)
            model_data = torch.load(buffer)

        data = RecvOperator.apply(
            torch.tensor(1.0, requires_grad=True, device=self.device_),
            self.transport_,
            message,
            "tail"
        )

        torch.cuda.reset_peak_memory_stats(device=self.device_)
        data.grad_fn.pre_stage_fn = self.default_stream_.poll

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        data = self.__forward(data, model_data)
        end_event.record()

        fwd_peak_memory_mb = torch.cuda.max_memory_allocated(self.device_) / (1024 * 1024)
        logging.info(f"Forward pass peak memory (Rank {self.rank_}): {self._fmt_peak_with_pct(fwd_peak_memory_mb)}")

        torch.cuda.synchronize()
        latency_ms = start_event.elapsed_time(end_event)
        logging.info(f"   Forward Pass Latency (Rank {self.rank_}): {latency_ms:.4f} ms")

        self.default_stream_.poll()

        # snapshot after forward
        self._tb_log_gpu_now(f"gpu_rank{self.rank_}")
        self._profiler_step("mid_forward") 
        return self.__send_activations(data, model_data)

    def __process_tail_forward_and_backward(self):
        assert self.role_ != WorkerRole.HEAD

        message = self.transport_.recv_message("prev", only_type=PipeMessageType.TENSOR)
        if message is None:
            return

        logging.info(f"[Rank {self.rank_}] Received activation message from prev.")

        model_data = None
        if getattr(message, "meta_tensor_", None) is not None:
            meta_bytes = bytes(message.meta_tensor_.cpu().tolist())
            buffer = io.BytesIO(meta_bytes)
            model_data = torch.load(buffer)

        data = RecvOperator.apply(
            torch.tensor(1.0, requires_grad=True, device=self.device_),
            self.transport_,
            message,
            "tail"
        )

        data.requires_grad_()

        torch.cuda.reset_peak_memory_stats(device=self.device_)
        data.grad_fn.pre_stage_fn = self.default_stream_.poll

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        data = self.__forward(data, model_data)
        data.retain_grad()
        end_event.record()

        fwd_peak_memory_mb = torch.cuda.max_memory_allocated(self.device_) / (1024 * 1024)
        logging.info(f"Forward pass peak memory (Rank {self.rank_}): {self._fmt_peak_with_pct(fwd_peak_memory_mb)}")

        torch.cuda.synchronize()
        latency_ms = start_event.elapsed_time(end_event)
        logging.info(f"   Forward Pass Latency (Rank {self.rank_}): {latency_ms:.4f} ms")

        self.default_stream_.poll()

        labels = torch.tensor(model_data.batch_tokens_, dtype=torch.long, device=self.device_)
        masks = torch.tensor(model_data.batch_mask_, device=self.device_)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.reset_peak_memory_stats(device=self.device_)

        start_event.record()

        total_loss = None
        for config in model_data.data_config_:
            loss = loss_fn(config, data, labels, masks)
            if loss is None:
                continue
            total_loss = loss if total_loss is None else total_loss + loss

        if total_loss is None:
            logging.warning("tail: no loss computed")
            return

        logging.info(f"[Tail] computed loss {total_loss.item()}; calling backward()")
        total_loss.backward()
        end_event.record()

        if self.writer_ is not None:
            self.writer_.add_scalar(f"loss/total_rank{self.rank_}", float(total_loss.item()), self.global_step_)
            self.writer_.add_scalar(f"time/forward_ms_rank{self.rank_}", float(latency_ms), self.global_step_)
            # GPU + memory (with percentages) to TB and logs
            self._tb_log_gpu_now(f"gpu_rank{self.rank_}")
        self._profiler_step("tail_backward")

        bwd_peak_memory_mb = torch.cuda.max_memory_allocated(self.device_) / (1024 * 1024)
        logging.info(f"Backward pass peak memory (Rank {self.rank_}): {self._fmt_peak_with_pct(bwd_peak_memory_mb)}")

        torch.cuda.synchronize()
        latency_ms = start_event.elapsed_time(end_event)
        logging.info(f"   Backward Pass Latency (Rank {self.rank_}): {latency_ms:.4f} ms")

        gradient_to_send = data.grad
        assert gradient_to_send is not None, "Input gradient is None after backward pass."
        logging.info(f"[Rank {self.rank_}] Sending gradient back to previous stage.")

        buffer = io.BytesIO()
        torch.save(model_data, buffer)
        buffer.seek(0)
        meta_bytes = torch.ByteTensor(list(buffer.getvalue())).to(self.device_)

        grad_message = PipeMessage(
            msg_type=PipeMessageType.TENSOR,
            tensor=gradient_to_send,
            comm_data=None,
            meta_tensor=meta_bytes
        )

        self.transport_.send_message(grad_message, "prev")

        for task_name in model_data.task_name_:
            self.dispatcher_.dispatch_task_to_step(task_name)

    def __handle_comm(self, comm_data):
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
            raise NotImplementedError(f"Unknown comm type: {comm_data}")

    def __process_output(self):
        assert self.role_ == WorkerRole.HEAD

        message = self.transport_.recv_message("prev", block=False)
        if message is None or message.msg_type_ != PipeMessageType.TENSOR:
            return

        logging.info(
            f"Recv the activations - {str(message.msg_id_)[:8]} from prev."
        )

        output: torch.Tensor = RecvOperator.apply(
            torch.tensor(1.0, requires_grad=True, device=self.device_),
            self.transport_,
            message,
            "head"
        )

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

        # snapshot after head forward/send path as well
        self._tb_log_gpu_now(f"gpu_rank{self.rank_}")
        self._profiler_step("head_output") 

    def __process_input(self):
        train_data: MLoRAData | None = self.dispatcher_.data()
        if train_data is None:
            return

        tensor_data = torch.tensor(
            train_data.batch_tokens_,
            dtype=torch.long,
            device=self.device_,
            requires_grad=False,
        )

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        self.latency_events_[train_data.model_data().random_id_] = start_event

        hidden_data = self.__forward(tensor_data, train_data.model_data())

        end_event.record()
        torch.cuda.synchronize()
        latency_ms = start_event.elapsed_time(end_event)
        logging.info(f"    Head Node Forward Latency: {latency_ms:.4f} ms")

        self.default_stream_.poll()

        buffer = io.BytesIO()
        torch.save(train_data.model_data(), buffer)
        buffer.seek(0)
        meta_bytes = torch.ByteTensor(list(buffer.getvalue())).to(self.device_)

        send_msg = PipeMessage(
            PipeMessageType.TENSOR,
            tensor=hidden_data,
            meta_tensor=meta_bytes,
            comm_data=None,
        )
        phony = SendOperator.apply(hidden_data, self.transport_, send_msg, "head")

        self.input_cache_[train_data.model_data().random_id_] = train_data
        self.backward_cache_[int(train_data.model_data().random_id_)] = phony

        # snapshot after head produces activations too
        self._tb_log_gpu_now(f"gpu_rank{self.rank_}")
        self._profiler_step("head_forward")

    def __send_activations(self, tensor_data: torch.Tensor, batch_data: ModelData):
        ##FIXME
        if self.next_rank == None:
            return

        assert isinstance(tensor_data, torch.Tensor)
        assert batch_data is None or isinstance(batch_data, ModelData)

        batch_random_id = batch_data.random_id_ if batch_data is not None else uuid.uuid4().int
        rid = batch_random_id % (2**63 - 1)

        msg = PipeMessage(
            PipeMessageType.TENSOR,
            tensor=tensor_data,
            meta_tensor=torch.tensor([rid], dtype=torch.int64, device=self.device_),
            comm_data=batch_data,
        )

        phony: torch.Tensor = SendOperator.apply(
            torch.tensor(1.0, requires_grad=True, device=self.device_),
            self.transport_,
            msg,
            "tail"
        )

        tn = batch_data.task_name_[0] if batch_data is not None else "unknown"
        self.backward_cache_[(tn, int(rid))] = phony

    def __send_comm(self, data: Any, dst: Optional[int] = None) -> None:
        if dst is None:
            if hasattr(self, "next_rank"):
                dst = self.next_rank
            else:
                try:
                    dst = self._name_to_rank(self.next_worker_name)
                except Exception:
                    dst = None

        if dst is None:
            logging.info(f"[Rank {self.rank_}] __send_comm called but no dst available.")
            return

        msg = PipeMessage(PipeMessageType.COMM, tensor=None, comm_data=data, meta_tensor=None)
        if dst == (self.rank_ + 1):
            neighbor = "next"
        elif dst == (self.rank_ - 1):
            neighbor = "prev"
        else:
            neighbor = "next" if dst > self.rank_ else "prev"

        self.transport_.send_message(msg, neighbor)

    def __recv_comm(self):
        for neighbor in ["prev", "next"]:
            if neighbor == "next" and self.next_rank is None:
                continue
            elif neighbor == "prev" and self.prev_rank is None:
                continue

            msg = self.transport_.recv_message(neighbor, block=True, only_type=PipeMessageType.COMM)
            if msg is None:
                continue

            logging.info(f"[Rank {self.rank_}] Received comm from {neighbor}")
            self.__handle_comm(msg.comm_data_)

    def send_tensor(self, tensor, dst):
        msg = PipeMessage(PipeMessageType.TENSOR, tensor=tensor)
        self.transport_.send_message(msg, dst)

    def recv_tensor(self, src):
        msg = self.transport_.recv_message(src)
        assert msg.msg_type_ == PipeMessageType.TENSOR
        return msg.tensor_data_

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
        elif self.role_ == WorkerRole.MID:
            self.__not_head_worker_run()
        elif self.role_ == WorkerRole.TAIL:
            self.__tail_worker_run()
        else:
            raise NotImplementedError

    def add_task(self, config: TaskConfig):
        if self.role_ != WorkerRole.TAIL:
            self.__send_comm({"comm": "task_add", "data": config})
        if self.role_ != WorkerRole.HEAD:
            config.dataset_ = None
        self.dispatcher_.add_task(config, self.model_name_)

    def __task_init_hook(self, task: Task):
        logging.info(
            f"Init {task.task_type()} : {task.task_name()} "
            + f"task with adapters: {task.adapter_name()}"
        )
        task.prepare(self.__linears_info(), self.tokenizer_)

    def __task_to_running_hook(self, task: Task):
        logging.info(f"Task to running, need to load adapters: {task.adapter_name()}")
        if self.role_ != WorkerRole.TAIL:
            self.__send_comm({"comm": "task_running", "data": task.task_name()})

        task.switch_device(self.device_)
        for adapter_model in task.adapter_model():
            for partial_layer in self.partial_model_:
                if partial_layer.name() == "Decoder":
                    try:
                        partial_layer.wrapper_module_.load_adapter(adapter_model)
                    except AssertionError:
                        pass

    def __task_to_ready_hook(self, task: Task):
        logging.info(f"Base model offload adapters: {task.adapter_name()}")
        if self.role_ != WorkerRole.TAIL:
            self.__send_comm({"comm": "task_ready", "data": task.task_name()})

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