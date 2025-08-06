# m-LoRA: Efficient Multi-LoRA Fine Tuning with Shared-Based Model
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright (C) 2024 All Rights Reserved.
#
# Github:  https://github.com/TUDB-Labs/mLoRA

import mlora.model
import mlora.utils
import mlora.executor
import mlora.config
import torch
import torch.distributed as dist
import os
import logging

def setup_distributed(args):
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        dist.init_process_group(backend="nccl")

        torch.cuda.set_device(local_rank)

        args.rank = rank
        args.nodes = world_size
        args.device = f"cuda:{local_rank}"

        logging.info(f"Distributed training enabled. Rank: {rank}/{world_size}, Device: {args.device}")
        return True

    except KeyError:
        logging.info("Not a distributed run. Running in single-process mode.")
        args.rank = 0
        args.nodes = 1
        return False

if __name__ == "__main__":
    args = mlora.utils.get_cmd_args()

    is_distributed = setup_distributed(args)

    try:

        mlora.utils.setup_seed(args.seed)
        mlora.utils.setup_logging(args.log_level, args.log_file)
        mlora.utils.setup_cuda_check()
        mlora.utils.setup_metric_logger(args.metric_file)

        if args.trace:
            mlora.utils.setup_trace_mode()

        tokenizer, model = mlora.model.load_model(args)
        config = mlora.config.MLoRAConfig(args.config)

        tasks_to_run = []

        if args.rank == 0:
            tasks_to_run = config.tasks_
            logging.info(f"Rank 0: Loaded {len(tasks_to_run)} tasks from config.")

        if is_distributed:
            object_list = [tasks_to_run]
            dist.broadcast_object_list(object_list, src=0)
            tasks_to_run = object_list[0]
            logging.info(f"Rank {args.rank}: Received {len(tasks_to_run)} tasks from Rank 0.")

        executor = mlora.executor.TPExecutor(
            model, tokenizer, config, args.device, args.rank, args.nodes, args.recompute
        )

        for item in config.tasks_:
            executor.add_task(item)

            executor.calculate_costs() #not implemented

        executor.execute()

    finally:
        if is_distributed:
            dist.destroy_process_group()
            logging.info("Distributed process group destroyed.")