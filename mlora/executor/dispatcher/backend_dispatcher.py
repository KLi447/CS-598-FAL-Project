import logging
import threading
from typing import override

from mlora.config.dispatcher import DispatcherConfig
from mlora.config.task import TaskConfig
from mlora.utils import is_shutdown_requested

from .dispatcher import Dispatcher


class BackendDispatcher(Dispatcher):
    sem_: threading.Semaphore

    def __init__(self, config: DispatcherConfig) -> None:
        super().__init__(config)
        self.sem_ = threading.Semaphore(0)

    @override
    def add_task(self, config: TaskConfig, llm_name: str):
        super().add_task(config, llm_name)
        self.sem_.release()

    @override
    def is_done(self) -> bool:
        while len(self.running_) == 0 and len(self.ready_) == 0:
            if is_shutdown_requested():
                logging.info("Dispatcher is_done(): detected shutdown. Exiting loop.")
                return True
            
            # block until some task be add to the queue
            logging.info("Dispatcher no task, wait...")
            did_acquire = self.sem_.acquire(timeout=5)  # TODO: added a timeout here since it can hang
            if not did_acquire:
                # logging.info("Dispatcher is_done(): sem.acquire() timed out. Returning True indicating we're done!")
                # return True    # failed to get a task due to the timeout, so return true indicating we're basically done
                # TODO: can potentially change this to just 'continue' so that it checks if it shutdown is requested. that way it only exits when a shutdown is requested
                continue
        return False
