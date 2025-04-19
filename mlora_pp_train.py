#!/usr/bin/env python3
# m-LoRA: Efficient Multi-LoRA Fine Tuning with Shared-Based Model
# [License header truncated for brevity]

import mlora.model
import mlora.utils
import mlora.executor
import mlora.config

if __name__ == "__main__":
    args = mlora.utils.get_cmd_args()
    mlora.utils.setup_seed(args.seed)
    mlora.utils.setup_logging(args.log_level, args.log_file)
    mlora.utils.setup_cuda_check()
    mlora.utils.setup_metric_logger(args.metric_file)
    if args.trace:
        mlora.utils.setup_trace_mode()

    tokenizer, model = mlora.model.load_model(args)
    config = mlora.config.MLoRAConfig(args.config)

    if args.fast_lora:
        print("Using fast (per-example) LoRA mode as specified.")
        for adapter_name, adapter_conf in config.adapters.items():
            if adapter_conf.type_ == "lora":
                # If desired, override normal LoRA adapters to fast mode.
                adapter_conf.type_ = "flora_per_example"
    else:
        print("Using standard LoRA mode.")

    executor = mlora.executor.PipeExecutor(
        model, tokenizer, config, args.device, args.rank, args.balance, args.recompute
    )

    if args.rank == 0:
        for item in config.tasks_:
            executor.add_task(item)

    executor.execute()
