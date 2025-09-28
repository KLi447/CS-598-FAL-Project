import logging
from typing import Callable, Dict, Optional

import torch

import mlora.profiler
from mlora.config import MLoRAConfig, TaskConfig
from mlora.model.args import MLoRAData
from mlora.model.llm import LLMModel
from mlora.model.tokenizer import Tokenizer

from .dispatcher import DISPATCHER_CLASS, Dispatcher
from .task import Task

import os
from torch.utils.tensorboard import SummaryWriter

# NVML is optional; we guard its usage in code paths
try:
    import pynvml
    _NVML_OK = True
    # Don't init globally here for multi-rank safety—init on demand in the logger
except Exception:
    _NVML_OK = False


class Executor:
    model_: LLMModel
    tokenizer_: Tokenizer

    dispatcher_: Dispatcher

    def __init__(
        self, model: LLMModel, tokenizer: Tokenizer, config: MLoRAConfig
    ) -> None:
        self.model_ = model
        self.tokenizer_ = tokenizer

        dispatcher_name = config.dispatcher_.name_
        assert dispatcher_name in DISPATCHER_CLASS
        self.dispatcher_ = DISPATCHER_CLASS[dispatcher_name](config.dispatcher_)
        self.global_step_ = 0

        # rank/host for clearer TB directories (mirrors your PipeExecutor convention)
        self.rank_ = int(os.environ.get("RANK", "0"))
        host = os.environ.get("SLURMD_NODENAME", os.uname().nodename)
        base_logdir = os.environ.get("MLOTRA_TB_LOGDIR", "/projects/beis/akanodia/CS-598-FAL-Project/runs/qwen_job_3")
        self.log_dir_ = os.path.join(base_logdir, f"host-{host}", f"rank{self.rank_}")
        os.makedirs(self.log_dir_, exist_ok=True)

        self.writer_ = SummaryWriter(log_dir=self.log_dir_, filename_suffix=f"_rank{self.rank_}")

        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]

        profiler_schedule = torch.profiler.schedule(wait=2, warmup=1, active=47, repeat=1)

        self.profiler_ = torch.profiler.profile(
            activities=activities,
            schedule=profiler_schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.log_dir_),
            record_shapes=True,
            with_stack=True
        )
        self.profiler_.start()

        # --- Hook registration (unchanged) ---
        hook_func = {
            "init": self.__task_init_hook,
            "running": self.__task_to_running_hook,
            "ready": self.__task_to_ready_hook,
            "done": self.__task_to_done_hook,
            "terminate": self.__task_to_terminate_hook,
        }
        for hook, cb in hook_func.items():
            self.dispatcher_.register_hook(hook, cb)

    # ===========================
    # Helpers for GPU/TB logging
    # ===========================

    def _device_total_bytes(self) -> int:
        """Total device memory in bytes for the current model device."""
        try:
            dev = torch.device(self.model_.device_)
            props = torch.cuda.get_device_properties(dev)
            return int(getattr(props, "total_memory", 0)) or 0
        except Exception:
            return 0

    def _tb_log_gpu_now(self, tag_prefix: str) -> Optional[float]:
        """
        Push instantaneous GPU + memory stats to TensorBoard.
        Mirrors the style from PipeExecutor:
          - NVML GPU util %
          - CUDA allocated/reserved bytes
          - Allocated/Reserved as % of device total
        Also flushes TB and increments global_step_.
        Returns util% if available.
        """
        if self.writer_ is None:
            return None

        try:
            dev = torch.device(self.model_.device_)
            if torch.cuda.is_available():
                torch.cuda.synchronize(dev)

                mem_alloc = torch.cuda.memory_allocated(dev)
                mem_resv  = torch.cuda.memory_reserved(dev)
                device_total_bytes = self._device_total_bytes()

                util_now, nvml_used, nvml_total = None, None, None

                # Try NVML (safe-guarded)
                if _NVML_OK:
                    try:
                        pynvml.nvmlInit()
                        # If device string is like "cuda:1", take the index
                        try:
                            idx = int(str(self.model_.device_).split(":")[-1])
                        except Exception:
                            idx = 0
                        h = pynvml.nvmlDeviceGetHandleByIndex(idx)
                        u = pynvml.nvmlDeviceGetUtilizationRates(h)
                        util_now = float(u.gpu)
                        meminfo = pynvml.nvmlDeviceGetMemoryInfo(h)
                        nvml_used, nvml_total = float(meminfo.used), float(meminfo.total)
                        pynvml.nvmlShutdown()
                    except Exception:
                        # NVML may not be available in some environments; just skip
                        pass

                step = self.global_step_

                if util_now is not None:
                    self.writer_.add_scalar(f"{tag_prefix}/gpu_util_pct", util_now, step)

                # Raw CUDA memory (allocator perspective)
                self.writer_.add_scalar(f"{tag_prefix}/cuda_mem/allocated_bytes", mem_alloc, step)
                self.writer_.add_scalar(f"{tag_prefix}/cuda_mem/reserved_bytes",  mem_resv,  step)

                # Normalize by device total, if available
                if device_total_bytes > 0:
                    self.writer_.add_scalar(f"{tag_prefix}/cuda_mem/allocated_pct",
                                            (mem_alloc / device_total_bytes) * 100.0, step)
                    self.writer_.add_scalar(f"{tag_prefix}/cuda_mem/reserved_pct",
                                            (mem_resv / device_total_bytes) * 100.0, step)

                # NVML device view (if present)
                if nvml_used is not None and nvml_total:
                    self.writer_.add_scalar(f"{tag_prefix}/nvml/device_used_bytes", nvml_used, step)
                    self.writer_.add_scalar(f"{tag_prefix}/nvml/device_used_pct", (nvml_used / nvml_total) * 100.0, step)

                # Also log a simple pipeline step marker to keep charts moving
                self.writer_.add_scalar("pipeline/step", step, step)

                # Flush immediately so TB frontend updates
                self.writer_.flush()

                # Bump step to ensure a new X-axis point next time
                self.global_step_ += 1

                # Emit a concise log line (like PipeExecutor)
                total_b = device_total_bytes if device_total_bytes else 1
                try:
                    alloc_pct = (mem_alloc / total_b) * 100.0
                    resv_pct  = (mem_resv  / total_b) * 100.0
                except Exception:
                    alloc_pct = resv_pct = 0.0
                logging.info(
                    f"[Rank {self.rank_}] step {step} | GPU util={util_now}% "
                    f"| alloc={mem_alloc/1e6:.1f}MB ({alloc_pct:.1f}%) "
                    f"| reserved={mem_resv/1e6:.1f}MB ({resv_pct:.1f}%)"
                )
                return util_now
        except Exception as e:
            logging.warning(f"TB GPU log failed: {e}")
        return None

    def _profiler_step(self, where: str):
        """Advance profiler & write a simple tag (kept lightweight like in PipeExecutor)."""
        try:
            if self.writer_:
                # already logged 'pipeline/step' inside _tb_log_gpu_now; keep step sync by calling tb first
                pass
            if self.profiler_ is not None:
                self.profiler_.step()
        except Exception:
            pass

    # ===========================
    # Task hooks (unchanged)
    # ===========================

    def register_hook(self, name: str, cb: Callable):
        self.dispatcher_.register_hook(name, cb)

    def __task_init_hook(self, task: Task):
        logging.info(
            f"Init {task.task_type()} : {task.task_name()} "
            + f"task with adapters: {task.adapter_name()}"
        )
        task.prepare(self.model_.linears_info(), self.tokenizer_)

    def __task_to_running_hook(self, task: Task):
        logging.info(f"Base model load adapters: {task.adapter_name()}")
        task.switch_device(self.model_.device_)
        for adapter_model in task.adapter_model():
            self.model_.load_adapter(adapter_model)

    def __task_to_ready_hook(self, task: Task):
        logging.info(f"Base model offload adapters: {task.adapter_name()}")
        for adapter_name in task.adapter_name():
            self.model_.offload_adapter(adapter_name)
        task.switch_device("cpu")

    def __task_to_done_hook(self, task: Task):
        logging.info(f"Finish and base model offload adapter - {task.adapter_name()}")
        for adapter_name in task.adapter_name():
            self.model_.offload_adapter(adapter_name)
        task.switch_device("cpu")
        task.done()

    def __task_to_terminate_hook(self, task: Task):
        logging.info(f"Task - {task.task_name()} terminate.")
        for adapter_name in task.adapter_name():
            self.model_.offload_adapter(adapter_name)
        task.switch_device("cpu")
        task.terminate()

    def dispatcher_info(self) -> Dict[str, str]:
        return self.dispatcher_.info()

    def add_task(self, config: TaskConfig):
        self.dispatcher_.add_task(config, self.model_.name_or_path_)

    def notify_terminate_task(self, task_name: str):
        self.dispatcher_.notify_terminate_task(task_name)

    # ===========================
    # Training loop
    # ===========================

    def execute(self) -> None:
        mm_collect_step = 0

        while not self.dispatcher_.is_done():
            data: MLoRAData | None = self.dispatcher_.data()
            assert data is not None

            torch.cuda.reset_peak_memory_stats(device=self.model_.device_)

            batch_size = data.batch_size()
            token_len = data.token_len()

            fwd_start_event = torch.cuda.Event(enable_timing=True)
            fwd_end_event = torch.cuda.Event(enable_timing=True)

            fwd_start_event.record()

            output = self.model_.forward(data.model_data())

            fwd_end_event.record()

            labels = torch.tensor(data.batch_tokens_, dtype=torch.long)

            fwd_peak_memory_mb = torch.cuda.max_memory_allocated(self.model_.device_) / (1024 * 1024)
            logging.info(f"Forward pass peak memory: {fwd_peak_memory_mb:.2f} MB")

            torch.cuda.synchronize()
            fwd_latency_ms = fwd_start_event.elapsed_time(fwd_end_event)
            logging.info(f"    Forward Pass Latency: {fwd_latency_ms:.4f} ms")
            if self.writer_:
                self.writer_.add_scalar("time/forward_ms", float(fwd_latency_ms), self.global_step_)

            # --- NEW: GPU/memory snapshot after forward (same style as PipeExecutor) ---
            self._tb_log_gpu_now(f"gpu_rank{self.rank_}")
            self._profiler_step("forward")

            total_loss: Optional[torch.Tensor] = None

            for config in data.data_config_:
                loss = config.loss_fn_(output, labels, torch.tensor(data.batch_mask_))
                if loss is None:
                    continue
                total_loss = loss if total_loss is not None else loss

                if total_loss is not loss:
                    total_loss = total_loss + loss

            if total_loss is not None:
                bwd_start_event = torch.cuda.Event(enable_timing=True)
                bwd_end_event = torch.cuda.Event(enable_timing=True)

                torch.cuda.reset_peak_memory_stats(self.model_.device_)
                bwd_start_event.record()

                total_loss.backward()

                # Keep profiler step, but we also call _profiler_step() to mirror PipeExecutor semantics
                self.profiler_.step()
                bwd_end_event.record()

                bwd_peak_memory_mb = torch.cuda.max_memory_allocated(self.model_.device_) / (1024 * 1024)
                logging.info(f"Backward pass peak memory: {bwd_peak_memory_mb:.2f} MB")

                torch.cuda.synchronize()
                bwd_latency_ms = bwd_start_event.elapsed_time(bwd_end_event)
                logging.info(f"    Backward Pass Latency: {bwd_latency_ms:.4f} ms")

                logging.info(f"    Total Latency:{(fwd_latency_ms + bwd_latency_ms):.4f} ms")

                if self.writer_:
                    self.writer_.add_scalar("loss/total", float(total_loss.item()), self.global_step_)
                    self.writer_.add_scalar("time/backward_ms", float(bwd_latency_ms), self.global_step_)
                    self.writer_.add_scalar(
                        "memory/max_alloc_bytes",
                        torch.cuda.max_memory_allocated(device=self.model_.device_),
                        self.global_step_,
                    )

                # --- NEW: GPU/memory snapshot after backward (same style as PipeExecutor) ---
                self._tb_log_gpu_now(f"gpu_rank{self.rank_}")
                self._profiler_step("backward")
            else:
                # Still keep charts moving if no loss was produced
                self._tb_log_gpu_now(f"gpu_rank{self.rank_}")
                self._profiler_step("no_loss_step")

            self.dispatcher_.step()
            mm_collect_step += 1
            if self.writer_:
                self.writer_.flush()

            # Note: self.global_step_ is incremented inside _tb_log_gpu_now()

            mlora.profiler.metric_log_dict(
                "memory",
                {
                    "batch_size": batch_size,
                    "token_len": token_len,
                    "memory": torch.cuda.max_memory_allocated(
                        device=self.model_.device_
                    ),
                },
                mm_collect_step,
            )