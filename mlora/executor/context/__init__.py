from typing import Dict, Type

from .context import TaskContext
from .dora import InferenceDoRAContext, TrainDoRAContext
from .inference import InferenceTaskContext
from .lora import InferenceLoRAContext, TrainLoRAContext
from .loraplus import TrainLoRAPlusContext
from .train import TrainTaskContext
from .vera import InferenceVeRAContext, TrainVeRAContext
from .flora_per_example import InferenceFloraPerExampleContext,TrainFloraPerExampleContext

TRAINCONTEXT_CLASS: Dict[str, Type[TrainTaskContext]] = {
    "lora": TrainLoRAContext,
    "loraplus": TrainLoRAPlusContext,
    "vera": TrainVeRAContext,
    "dora": TrainDoRAContext,
    "flora_per_example": TrainFloraPerExampleContext,
}

INFERENCECONTEXT_CLASS: Dict[str, Type[InferenceTaskContext]] = {
    "lora": InferenceLoRAContext,
    "loraplus": InferenceLoRAContext,
    "vera": InferenceVeRAContext,
    "dora": InferenceDoRAContext,
    "flora_per_example": InferenceFloraPerExampleContext,
}


__all__ = [
    "TRAINCONTEXT_CLASS",
    "INFERENCECONTEXT_CLASS",
    "TaskContext",
    "TrainTaskContext",
    "TrainLoRAContext",
    "InferenceLoRAContext",
    "TrainLoRAPlusContext",
]
