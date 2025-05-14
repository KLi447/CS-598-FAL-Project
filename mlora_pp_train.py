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
from mlora.utils.shutdown import request_shutdown, is_shutdown_requested, get_shutdown_event
import signal
import logging
import sys

count = 0

def sigint_handler(sig, frame):
    global count
    count += 1
    logging.info("Caught Signal in handler")
    if count == 2:
        exit(0)
    request_shutdown(signal.Signals(sig).name)

if __name__ == "__main__":
    args = mlora.utils.get_cmd_args()
    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGTERM, sigint_handler)

    mlora.utils.setup_seed(args.seed)
    mlora.utils.setup_logging(args.log_level, args.log_file)
    mlora.utils.setup_cuda_check()
    mlora.utils.setup_metric_logger(args.metric_file)

    # enable the trace mode for profiling performance
    if args.trace:
        mlora.utils.setup_trace_mode()

    tokenizer, model = mlora.model.load_model(args)
    config = mlora.config.MLoRAConfig(args.config)

    # init all task from config file
    executor = mlora.executor.PipeExecutor(
        model, tokenizer, config, args.device, args.rank, args.nodes, args.recompute
    )

    # only the header node can add task
    if args.rank == 0:
        for item in config.tasks_:
            executor.add_task(item)

    try:
        executor.execute()
    except KeyboardInterrupt:
        logging.error(f"Keyboard Interrupt in main", exc_info=True)
        request_shutdown("ExceptionInMain") # Ensure shutdown on other errors too
    finally:
        logging.info("Main: Performing final cleanup...")
        if hasattr(executor, 'transport_') and executor.transport_ is not None and not executor.transport_.stop_:
            logging.info(f"Main: Attempting to stop transport for executor (Rank {executor.rank_}).")
            executor.transport_.stop()
            logging.info(f"Main: Transport for executor (Rank {executor.rank_}) stopped.")
            
    logging.info("mLoRA training process finished. End of main()")
    
    # if args.rank == 0:
    #     # sys.exit(0) # force shutdown in case of other threads still running - temporary solution to their bad code
    #     exit()