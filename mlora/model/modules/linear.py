# File: mlora/model/modules/linear.py

from typing import Callable, List, MutableMapping, Optional, Tuple

import torch
import torch.nn.functional as F

from mlora.model.args import ModelData
from mlora.profiler import nvtx_range, set_backward_tracepoint
from mlora.utils import is_package_available

if is_package_available("bitsandbytes"):
    from bitsandbytes.nn import Linear4bit, Linear8bitLt
else:
    from mlora.utils import Linear8bitLt, Linear4bit

from .adapter import Adapter
from .dora import DoRA
from .lora import LoRA, LoRAFunction, get_range_tensor
from .vera import VeRA
from .flora_per_example import FFloraPerExampleAdapter


class Linear(torch.nn.Module):
    def __init__(self, weight: torch.nn.Module):
        # wrap the actual Llama Linear or bitsandbytes Linear4/8bitLt
        super().__init__()

        if not isinstance(weight, torch.nn.Linear):
            assert isinstance(weight, Linear8bitLt) or isinstance(
                weight, Linear4bit
            ), f"error type - {type(weight)}."
        else:
            weight.requires_grad_(False)

        self.device_   = weight.weight.device
        self.weight_   = weight
        self.adapters_: MutableMapping[str, Adapter] = {}

    def forward(self, data: torch.Tensor, input_args: ModelData) -> torch.Tensor:
        # data: [batch*seq_len, dim]  or  [batch, seq_len, dim]
        if len(self.adapters_) == 0:
            return self.weight_.forward(data)

        # base pass
        with nvtx_range("f_linear"):
            result = self.weight_.forward(data)
        set_backward_tracepoint(result.grad_fn, "b_linear")

        # sequence of adapter hooks
        for func in (
            self.__lora_forward,
            self.__vera_forward,
            self.__dora_forward,
            self.__flora_per_example_forward,
        ):
            result = func(data, input_args, result)

        return result

    def __lora_forward(
        self, data: torch.Tensor, input_args: ModelData, result: torch.Tensor
    ) -> torch.Tensor:
        dropouts: List[Optional[float]] = []
        scalings: List[Optional[float]] = []
        loras: Tuple[torch.Tensor | None, ...] = ()

        for cfg in input_args.data_config_:
            name = cfg.adapter_name_
            if name not in self.adapters_ or not isinstance(
                self.adapters_[name], LoRA
            ):
                loras += (None, None)
                dropouts.append(None)
                scalings.append(None)
                continue

            adapter = self.adapters_[name]
            loras += (adapter.lora_a_, adapter.lora_b_)
            dropouts.append(adapter.dropout_)
            scalings.append(adapter.scaling_)

        with nvtx_range("f_lora"):
            result = LoRAFunction.apply(
                result, data, input_args, dropouts, scalings, *loras
            )
        set_backward_tracepoint(result.grad_fn, "b_lora")
        return result

    def __vera_forward(
        self, data: torch.Tensor, input_args: ModelData, result: torch.Tensor
    ) -> torch.Tensor:
        lora_range = get_range_tensor(data.device, data.shape[0])
        for cfg in input_args.data_config_:
            name = cfg.adapter_name_
            if name not in self.adapters_ or not isinstance(
                self.adapters_[name], VeRA
            ):
                continue

            adapter = self.adapters_[name]
            start, end = cfg.batch_start_idx_, cfg.batch_end_idx_

            with nvtx_range("f_vera"):
                part = data[start:end]
                part = F.dropout(part, p=adapter.dropout_, training=True)
                part = part.mul(adapter.scaling_)
                part = part @ adapter.lora_a_.T
                part = part * adapter.d_vec_
                part = part @ adapter.lora_b_.T
                part = part * adapter.b_vec_
                part = part.to(result.dtype)

                result = result.index_add(
                    dim=0, index=lora_range[start:end], source=part
                )

        set_backward_tracepoint(result.grad_fn, "b_vera")
        return result

    def __dora_forward(
        self, data: torch.Tensor, input_args: ModelData, result: torch.Tensor
    ) -> torch.Tensor:
        lora_range = get_range_tensor(data.device, data.shape[0])
        for cfg in input_args.data_config_:
            name = cfg.adapter_name_
            if name not in self.adapters_ or not isinstance(
                self.adapters_[name], DoRA
            ):
                continue

            adapter = self.adapters_[name]
            start, end = cfg.batch_start_idx_, cfg.batch_end_idx_

            with nvtx_range("f_dora"):
                weight_norm   = adapter.get_weight_norm()
                mag_norm_scale = (adapter.magnitude_ / weight_norm).view(1, -1)

                part = data[start:end]
                part = F.dropout(part, p=adapter.dropout_, training=True)
                part = part @ adapter.lora_a_.T
                part = part @ adapter.lora_b_.T
                part = mag_norm_scale * part * adapter.scaling_

                base = result[start:end] * (mag_norm_scale - 1) + part
                result = result.index_copy(
                    dim=0, index=lora_range[start:end], source=base
                )

        set_backward_tracepoint(result.grad_fn, "b_dora")
        return result

    def __flora_per_example_forward(
        self, data: torch.Tensor, input_args: ModelData, result: torch.Tensor
    ) -> torch.Tensor:
        """
        Fast‑LoRA per‑example path. Must return a [B, S, C] tensor just like `result`
        so the decoder’s residual add will work.
        """
        # Find the adapter of type "flora_per_example"
        for cfg in input_args.data_config_:
            name = cfg.adapter_name_
            if name not in self.adapters_:
                continue
            adapter = self.adapters_[name]
            if adapter.adapter_type_ != "flora_per_example":
                continue

            if data.dim() == 2:
                # [B*S, d] → [B, S, d]
                BS, d    = data.shape
                S        = input_args["seq_len"]
                B        = BS // S
                data3d   = data.view(B, S, d)
            else:
                # already [B, S, d]
                B, S, d  = data.shape
                data3d   = data

            # W0: original Llama weight of shape [out_dim, in_dim]
            W0 = self.weight_.weight
            W0_t = W0.transpose(0,1)                   # [in_dim, out_dim]

            # Run our adapter: returns [B, S, out_dim]
            out3d = adapter.forward_per_example(data3d, W0_t)

            # **DO NOT** flatten back to 2D.  Return the 3D tensor:
            result = out3d.to(result.dtype)

        return result

    def load_adapter(self, adapter: Adapter):
        assert adapter.adapter_name_ not in self.adapters_
        self.adapters_[adapter.adapter_name_] = adapter

    def offload_adapter(self, adapter_name: str):
        if adapter_name in self.adapters_:
            del self.adapters_[adapter_name]
