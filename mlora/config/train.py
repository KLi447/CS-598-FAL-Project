import logging
from typing import Dict, Mapping

from .adapter import AdapterConfig
from .dataset import DatasetConfig
from .base import TaskConfig

class TrainTaskConfig(TaskConfig):
    batch_size_: int
    mini_batch_size_: int
    num_epochs_: int
    cutoff_len_: int
    save_step_: int
    compute_cost_: float

    __params_map: Dict[str, str] = {
        "batch_size_": "batch_size",
        "mini_batch_size_": "mini_batch_size",
        "num_epochs_": "num_epochs",
        "cutoff_len_": "cutoff_len",
        "save_step_": "save_step",
    }

    def __init__(
        self,
        config: Dict[str, str],
        adapters: Mapping[str, AdapterConfig],
        datasets: Mapping[str, DatasetConfig],
    ):
        super().__init__(config, adapters, datasets)
        self.init(self.__params_map, config)

        self.batch_size_ = int(self.batch_size_)
        self.mini_batch_size_ = int(self.mini_batch_size_)
        self.num_epochs_ = int(self.num_epochs_)
        self.cutoff_len_ = int(self.cutoff_len_)
        self.save_step_ = int(self.save_step_)

        adapter_name = config["adapter"]
        adapter_conf = adapters[adapter_name]
        if adapter_conf.type_ == "flora_per_example":
            r = adapter_conf.rank_
            # maybe cost = r * batch_size or something
            self.compute_cost_ = r * float(self.batch_size_)
        else:
            self.compute_cost_ = float(self.batch_size_)

        assert self.mini_batch_size_ <= self.batch_size_
        assert self.batch_size_ % self.mini_batch_size_ == 0

    @property
    def accumulate_step_(self) -> int:
        return self.batch_size_ // self.mini_batch_size_ 