import logging
from typing import Dict, Mapping

from .adapter import AdapterConfig
from .dataset import DatasetConfig

class TaskConfig:
    name_: str
    type_: str
    adapter_: AdapterConfig
    dataset_: DatasetConfig

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
        self.init(self.__params_map, config)

        adapter_name = config["adapter"]
        dataset_name = config["dataset"]

        self.adapter_ = adapters[adapter_name]
        self.dataset_ = datasets[dataset_name]

    def init(self, params_map: Dict[str, str], config: Dict[str, str]):
        for key, value in params_map.items():
            if value not in config:
                logging.warning(f"Missing config {value} for {key}")
                continue
            setattr(self, key, config[value]) 