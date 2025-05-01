import logging
from typing import Dict, Mapping, Optional, Type

from .adapter import AdapterConfig
from .config import DictConfig
from .dataset import DatasetConfig
from .train import TrainTaskConfig
from .cpo import CPOTaskConfig
from .ppo import PPOTaskConfig
from .cit import CITTaskConfig
from .dpo import DPOTaskConfig
from .patent import PatentTaskConfig

class TaskConfig(DictConfig):
    name_: str
    type_: str

    adapter_: AdapterConfig
    dataset_: DatasetConfig | None

    __params_map: Dict[str, str] = {
        "name_": "name",
        "type_": "type",
    }

    def __init__(
        self,
        config: Dict[str, str],
        adapters: Mapping[str, AdapterConfig],
        datasets: Mapping[str, DatasetConfig],
    ):
        super().__init__(config)
        self.init(self.__params_map, config)

        if isinstance(config["adapter"], dict):
            self.reward_adapter_ = adapters[config["adapter"]["reward_adapter"]]
            self.actor_adapter_ = adapters[config["adapter"]["actor_adapter"]]
            self.critic_adapter_ = adapters[config["adapter"]["critic_adapter"]]
        else:
            self.adapter_ = adapters[config["adapter"]]

        self.dataset_: DatasetConfig | None = datasets[config["dataset"]]

TASKCONFIG_CLASS: Dict[str, Type["TaskConfig"]] = {
    "train": TrainTaskConfig,
    "cpo": CPOTaskConfig,
    "ppo": PPOTaskConfig,
    "cit": CITTaskConfig,
    "dpo": DPOTaskConfig,
    "patent": PatentTaskConfig,
} 