from typing import Dict, Type

from .backend_dispatcher import BackendDispatcher
from .dispatcher import Dispatcher
from .pipe_dispatcher import PipeDispatcher
from .tensor_dispatcher import TensorParallelDispatcher

DISPATCHER_CLASS: Dict[str, Type[Dispatcher]] = {
    "default": Dispatcher,
    "backend": BackendDispatcher,
    "pipe": PipeDispatcher,
    "tensor": TensorParallelDispatcher
}

__all__ = ["Dispatcher", "BackendDispatcher", "PipeDispatcher", "TensorParallelDispatcher", "DISPATCHER_CLASS"]
