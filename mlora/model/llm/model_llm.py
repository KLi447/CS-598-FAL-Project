from collections import OrderedDict
from typing import List, Optional

import torch

from mlora.model.args import LinearInfo, ModelData
from mlora.model.modules import AdapterModel

class LLMModel(torch.nn.Module):
    name_or_path_: str
    device_: str
    vocab_size_: int
    n_heads_: int
    dim_: int

    def __init__(self):
        super().__init__()

    def forward(self, input: ModelData):
        raise NotImplementedError

    @staticmethod
    def from_pretrained(
        path: str,
        device: str,
        precision: str,
        partial_model_to_device: Optional[List[int]] = None,
    ) -> "LLMModel":
        raise NotImplementedError

    def load_adapter(self, adapter_model: AdapterModel):
        raise NotImplementedError

    def offload_adapter(self, adapter_name: str):
        raise NotImplementedError

    def linears_info(self) -> OrderedDict[str, LinearInfo]:
        raise NotImplementedError

    def sequential(self) -> torch.nn.Sequential:
        raise NotImplementedError