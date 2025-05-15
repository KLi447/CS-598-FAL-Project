# File: mlora/executor/context/flora_per_example.py

from collections import OrderedDict
from typing import Dict, Optional, override

import torch

from mlora.config import FloraPerExampleConfig
from mlora.model.args import LinearInfo
from mlora.model.modules.flora_per_example import FFloraPerExampleAdapter

from .context import TaskContext
from .inference import InferenceTaskContext
from .train import TrainTaskContext


class InferenceFloraPerExampleContext(InferenceTaskContext):
    config_: FloraPerExampleConfig

    def __init__(
        self,
        config: FloraPerExampleConfig,
        linears_info: OrderedDict[str, LinearInfo],
    ) -> None:
        super().__init__(config, linears_info)

    @override
    def load_weight(self, linears_info: OrderedDict[str, LinearInfo]):
        # instantiate adapter (no training)
        
        for name, info in linears_info.items():
            target_name = name.split(".")[3]
            if target_name not in self.config_.target_:
                continue
            if self.config_.target_[target_name] is not True:
                continue
                
            adapter = FFloraPerExampleAdapter(
                adapter_name=self.config_.name_,
                batch_size=self.config_.batch_size_,
                in_dim=info.in_dim_,
                out_dim=info.out_dim_,
                r=self.config_.rank_,
                device=self.device_,
                activation_fn=None,
                alpha=self.config_.alpha_,
                dropout=self.config_.dropout_,
                shared=self.config_.shared_,
            )
            # placeholder init
            adapter.init_weight(None, None)
            self.adapter_model_[name] = adapter


class TrainFloraPerExampleContext(TrainTaskContext):
    config_: FloraPerExampleConfig

    def __init__(
        self,
        config: FloraPerExampleConfig,
        linears_info: OrderedDict[str, LinearInfo],
    ) -> None:
        super().__init__(config, linears_info)
        self.loss_fn_ = torch.nn.CrossEntropyLoss()

    @override
    def load_weight(self, linears_info: OrderedDict[str, LinearInfo]):
        # build & init adapter for each linear
        for name, info in linears_info.items():
            target_name = name.split(".")[3]
            if target_name not in self.config_.target_:
                continue
            if self.config_.target_[target_name] is not True:
                continue

            opt_cfg = self.config_.optimizer_config_
            optim_name = opt_cfg.optimizer_ if opt_cfg else "adamw"
            optim_args = opt_cfg.to_fn_parameters() if opt_cfg else {"lr": 3e-4}

            adapter = FFloraPerExampleAdapter(
                adapter_name=self.config_.name_,
                batch_size=self.config_.batch_size_,
                in_dim=info.in_dim_,
                out_dim=info.out_dim_,
                r=self.config_.rank_,
                device=self.device_,
                activation_fn=None,
                alpha=self.config_.alpha_,
                dropout=self.config_.dropout_,
                shared=self.config_.shared_,
            )

            # optionally warm‑start
            B_init = torch.load(self.config_.B_init_file_) if self.config_.B_init_file_ else None
            A_init = torch.load(self.config_.A_init_file_) if self.config_.A_init_file_ else None
            adapter.init_weight(B_init, A_init)

            self.adapter_model_[name] = adapter

    @override
    def weight_dict(self) -> Dict[str, torch.Tensor]:
        prefix = "base_model.model.model."
        out: Dict[str, torch.Tensor] = {}
        for name, adapter in self.adapter_model_.items():
            if adapter.shared_:
                out[prefix + name + ".flora_B"] = adapter.B
                out[prefix + name + ".flora_A"] = adapter.A
            else:
                out[prefix + name + ".flora_B_all"] = adapter.B_all
                out[prefix + name + ".flora_A_all"] = adapter.A_all
        return out

    @override
    def recover_weight(self, weight_dict: Dict[str, torch.Tensor]):
        prefix = "base_model.model.model."
        for name, adapter in self.adapter_model_.items():
            if adapter.shared_:
                adapter.B.copy_(weight_dict[prefix + name + ".flora_B"])
                adapter.A.copy_(weight_dict[prefix + name + ".flora_A"])
            else:
                adapter.B_all.copy_(weight_dict[prefix + name + ".flora_B_all"])
                adapter.A_all.copy_(weight_dict[prefix + name + ".flora_A_all"])
