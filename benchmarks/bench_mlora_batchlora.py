import logging
import os
import random
import time
from typing import List, Tuple, override

import torch

import mlora.config
import mlora.executor
import mlora.executor.dispatcher
import mlora.model
import mlora.utils
from mlora.config.adapter import LoRAConfig
from mlora.config.dispatcher import DispatcherConfig
from mlora.config.task import TrainTaskConfig
from mlora.executor.task import TrainTask, register_task_class
from mlora.model.args import MLoRADataConfig, Tokens

g_start_time: int | None = None
g_total_token: int = 0


g_batch_size: int = int(os.getenv("BATCH_SIZE"))
g_concurrency_num: int = int(os.getenv("TASK"))


class BenchmarkArgs:
    seq_len_: int = 512
    test_epochs_: int = 20


class BenchmarkConfig(mlora.config.MLoRAConfig):
    dispatcher_: DispatcherConfig

    def __init__(self):
        # just for init
        self.dispatcher_ = DispatcherConfig(
            {"name": "default", "concurrency_num": g_concurrency_num}
        )


class BenchmarkTask(TrainTask):
    def __init__(self, config: mlora.config.TaskConfig, llm_name: str) -> None:
        super().__init__(config, llm_name)

    @override
    def data(self, start_idx: int) -> Tuple[List[Tokens], List[MLoRADataConfig]]:
        global g_batch_size

        ret_tokens = [
            [random.randint(1, 10000) for _ in range(BenchmarkArgs.seq_len_)]
        ] * g_batch_size

        end_idx = start_idx + len(ret_tokens)

        def loss_fn(
            input: torch.Tensor, target: torch.Tensor, _: torch.Tensor
        ) -> torch.Tensor:
            vacab_size = input.shape[-1]
            loss_input = (
                input[start_idx:end_idx, :-1, :].contiguous().view(-1, vacab_size)
            )
            loss_target = (
                target[start_idx:end_idx, 1:]
                .contiguous()
                .view(-1)
                .to(loss_input.device)
            )
            loss: torch.Tensor = self.context_.loss_fn_(loss_input, loss_target)

            return loss

        data_config = MLoRADataConfig(
            self.context_.name_,
            self.context_.type_,
            start_idx,
            end_idx,
            self._expand_batch_tokens,
            loss_fn,
            self.task_name(),
        )

        return ret_tokens, [data_config]

    @override
    def step(self):
        self.now_step_ += 1
        self.now_epoch_ += 1
        self.context_.step()

        global g_start_time
        global g_total_token
        global g_batch_size

        if g_start_time is not None:
            g_total_token = g_total_token + (g_batch_size * BenchmarkArgs.seq_len_)
            logging.info(
                f"average {g_total_token / (time.time() - g_start_time) : .2f} tokens/s"
            )
        else:
            g_start_time = time.time()

        logging.info(f"task {self.context_.name_} step")


def generate_task_config(task_idx: int) -> TrainTaskConfig:

    adapters = {
        f"test_{task_idx}": LoRAConfig(
            {
                "type": "lora",
                "name": f"test_{task_idx}",
                "path": f"adapters/test_{task_idx}",
                "r": 16,
                "alpha": 16,
                "dropout": 0.05,
                "target_modules": {
                    "q_proj": True,
                    "k_proj": True,
                    "v_proj": True,
                    "o_proj": True,
                },
                "optimizer": "adamw",
                "lr": 1e-3,
            }
        )
    }
    datasets = {f"test_{task_idx}": None}

    return TrainTaskConfig(
        {
            "batch_size": g_batch_size,
            "mini_batch_size": g_batch_size,
            "num_epochs": BenchmarkArgs.test_epochs_,
            "cutoff_len": BenchmarkArgs.seq_len_,
            "save_step": 10000000,
            "name": f"test_{task_idx}",
            "type": "benchmark",
            "adapter": f"test_{task_idx}",
            "dataset": f"test_{task_idx}",
        },
        adapters,
        datasets,
    )


if __name__ == "__main__":
    args = mlora.utils.get_cmd_args()

    mlora.utils.setup_seed(args.seed)
    mlora.utils.setup_logging(args.log_level, args.log_file)
    mlora.utils.setup_cuda_check()
    mlora.utils.setup_metric_logger(args.metric_file)

    register_task_class("benchmark", BenchmarkTask)

    # enable the trace mode for profiling performance
    if args.trace:
        mlora.utils.setup_trace_mode()

    tokenizer, model = mlora.model.load_model(args)

    config = BenchmarkConfig()

    # init all task from config file
    executor = mlora.executor.Executor(
        model,
        tokenizer,
        config,
    )

    # only the header node can add task
    for idx in range(0, g_concurrency_num):
        executor.add_task(generate_task_config(idx))

    executor.execute()
