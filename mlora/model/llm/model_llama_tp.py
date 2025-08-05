import logging
import os
from collections import OrderedDict
from typing import Dict, List, Optional, override

import torch
import torch.distributed as dist
from torch.nn import Module, ModuleList, Parameter, Sequential
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

from mlora.model.args import LinearInfo, LLMModelArgs, ModelData
from mlora.model.modules import AdapterModel
from .model_llm import LLMModel

def get_tensor_parallel_rank():
    return dist.get_rank() if dist.is_initialized() else 0

def get_tensor_parallel_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1

class _CopyToTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if dist.is_initialized():
            dist.all_reduce(grad_output)
        return grad_output

class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        if dist.is_initialized():
            dist.all_reduce(input_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class ColumnParallelLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, device=None, dtype=None):
        super().__init__()
        world_size = get_tensor_parallel_world_size()
        self.out_features_per_partition = out_features // world_size

        self.weight = Parameter(torch.empty(
            self.out_features_per_partition, in_features, device=device, dtype=dtype
        ))
        if bias:
            self.bias = Parameter(torch.empty(
                self.out_features_per_partition, device=device, dtype=dtype
            ))
        else:
            self.register_parameter('bias', None)

    def forward(self, input_):
        parallel_input = _CopyToTensorParallelRegion.apply(input_)
        output_parallel = F.linear(parallel_input, self.weight, self.bias)
        return output_parallel

class RowParallelLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, device=None, dtype=None):
        super().__init__()
        world_size = get_tensor_parallel_world_size()
        self.in_features_per_partition = in_features // world_size

        self.weight = Parameter(torch.empty(
            out_features, self.in_features_per_partition, device=device, dtype=dtype
        ))
        if bias:
            self.bias = Parameter(torch.empty(
                out_features, device=device, dtype=dtype
            ))
        else:
            self.register_parameter('bias', None)

    def forward(self, input_):
        output_parallel = F.linear(input_, self.weight)
        output = _ReduceFromTensorParallelRegion.apply(output_parallel)
        if self.bias is not None:
            output = output + self.bias
        return output

class TensorParallelLoRALayer(Module):
    def __init__(self, base_layer: Module, r: int, scaling: float, lora_a: torch.Tensor, lora_b: torch.Tensor):
        super().__init__()
        self.r = r
        self.scaling = scaling
        self.is_row_parallel = isinstance(base_layer, RowParallelLinear)
        self.lora_a = Parameter(lora_a)
        self.lora_b = Parameter(lora_b)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora_output = (x @ self.lora_a.T) @ self.lora_b.T * self.scaling
        if self.is_row_parallel:
            return _ReduceFromTensorParallelRegion.apply(lora_output)
        else:
            return lora_output

class TensorParallelAttention(torch.nn.Module):
    def __init__(self, args: LLMModelArgs):
        super().__init__()
        self.n_heads = args.n_heads_
        self.n_kv_heads = args.n_kv_heads_
        self.head_dim = args.dim_ // args.n_heads_
        world_size = get_tensor_parallel_world_size()

        self.n_heads_per_partition = self.n_heads // world_size
        self.n_kv_heads_per_partition = self.n_kv_heads // world_size if self.n_kv_heads > 0 else 0

        self.q_proj = ColumnParallelLinear(args.dim_, args.dim_, bias=False, device=args.device_, dtype=args.dtype_)
        self.k_proj = ColumnParallelLinear(args.dim_, self.n_kv_heads * self.head_dim, bias=False, device=args.device_, dtype=args.dtype_)
        self.v_proj = ColumnParallelLinear(args.dim_, self.n_kv_heads * self.head_dim, bias=False, device=args.device_, dtype=args.dtype_)
        self.o_proj = RowParallelLinear(args.dim_, args.dim_, bias=False, device=args.device_, dtype=args.dtype_)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, data: ModelData):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads_per_partition, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads_per_partition, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads_per_partition, self.head_dim).transpose(1, 2)

        if self.n_kv_heads_per_partition > 0:
            num_query_groups = self.n_heads_per_partition // self.n_kv_heads_per_partition
            if num_query_groups > 1:
                k = k.repeat_interleave(num_query_groups, dim=1)
                v = v.repeat_interleave(num_query_groups, dim=1)

        scores = torch.matmul(q, k.transpose(2, 3)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores + mask
        
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.o_proj(attn_output)

class TensorParallelMLP(torch.nn.Module):
    def __init__(self, args: LLMModelArgs, config):
        super().__init__()
        hidden_dim = config.intermediate_size
        
        self.gate_proj = ColumnParallelLinear(args.dim_, hidden_dim, bias=False, device=args.device_, dtype=args.dtype_)
        self.up_proj = ColumnParallelLinear(args.dim_, hidden_dim, bias=False, device=args.device_, dtype=args.dtype_)
        self.down_proj = RowParallelLinear(hidden_dim, args.dim_, bias=False, device=args.device_, dtype=args.dtype_)
        self.act_fn = torch.nn.SiLU()

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(self.act_fn(gate) * up)

class TensorParallelDecoderLayer(torch.nn.Module):
    def __init__(self, args: LLMModelArgs, config):
        super().__init__()
        self.self_attn = TensorParallelAttention(args)
        self.mlp = TensorParallelMLP(args, config)
        self.input_layernorm = RMSNorm(args.dim_, eps=args.norm_eps_)
        self.post_attention_layernorm = RMSNorm(args.dim_, eps=args.norm_eps_)
        self.adapters: Dict[str, List] = {}

    def forward(self, x: torch.Tensor, mask: torch.Tensor, data: ModelData):
        residual = x
        hidden_states = self.input_layernorm(x)
        hidden_states = self.self_attn(hidden_states, mask, data)
        x = residual + hidden_states

        residual = x
        hidden_states = self.post_attention_layernorm(x)
        hidden_states = self.mlp(hidden_states)
        x = residual + hidden_states

        return x

    def linears_info(self) -> OrderedDict[str, LinearInfo]:
        world_size = get_tensor_parallel_world_size()
        info = OrderedDict()
        for name, module in self.named_modules():
            submodule_name = name.split('.')[-1]
            if isinstance(module, ColumnParallelLinear):
                info[name] = LinearInfo(
                    name_=submodule_name,
                    in_dim_=module.weight.shape[1], 
                    out_dim_=module.out_features_per_partition * world_size, 
                    base_weight_=module
                )
            elif isinstance(module, RowParallelLinear):
                info[name] = LinearInfo(
                    name_=submodule_name,
                    in_dim_=module.in_features_per_partition * world_size, 
                    out_dim_=module.weight.shape[0], 
                    base_weight_=module
                )
        return info

    def load_adapter(self, layer_prefix: str, adapter_name: str, adapter_config: dict):
        rank = get_tensor_parallel_rank()
        world_size = get_tensor_parallel_world_size()

        if adapter_name not in self.adapters:
            self.adapters[adapter_name] = []

        for full_layer_name, lora_config in adapter_config.items():
            if not full_layer_name.startswith(f"{layer_prefix}."):
                continue
    
            submodule_path = full_layer_name.replace(f"{layer_prefix}.", "", 1)
            try:
                base_layer = self.get_submodule(submodule_path)
            except AttributeError:
                logging.warning(f"Could not find submodule {submodule_path} in layer {layer_prefix}")
                continue

            lora_a_full = lora_config.lora_a_
            lora_b_full = lora_config.lora_b_

            def hook(module, input, output, lora_layer):
                return output + lora_layer(input[0])

            if isinstance(base_layer, ColumnParallelLinear):
                in_dim = base_layer.weight.shape[1]
                out_dim = base_layer.out_features_per_partition * world_size
                
                if lora_a_full.shape[1] != in_dim:
                    raise ValueError(f"LoRA A matrix for {full_layer_name} has incorrect in_features: expected {in_dim}, got {lora_a_full.shape[1]}")
                if lora_b_full.shape[0] != out_dim:
                    raise ValueError(f"LoRA B matrix for {full_layer_name} has incorrect out_features: expected {out_dim}, got {lora_b_full.shape[0]}")

                lora_a_sharded = lora_a_full
                lora_b_sharded = lora_b_full.chunk(world_size, dim=0)[rank]
                
                lora_layer = TensorParallelLoRALayer(
                    base_layer, r=lora_config.r_, scaling=lora_config.scaling_,
                    lora_a=lora_a_sharded.to(base_layer.weight.device),
                    lora_b=lora_b_sharded.to(base_layer.weight.device)
                )

                hook_handle = base_layer.register_forward_hook(
                    lambda m, i, o, l=lora_layer: hook(m, i, o, l)
                )
                self.adapters[adapter_name].append({"name": full_layer_name, "hook": hook_handle})

            elif isinstance(base_layer, RowParallelLinear):
                in_dim = base_layer.in_features_per_partition * world_size
                out_dim = base_layer.weight.shape[0]

                if lora_a_full.shape[1] != in_dim:
                    raise ValueError(f"LoRA A matrix for {full_layer_name} has incorrect in_features: expected {in_dim}, got {lora_a_full.shape[1]}")
                if lora_b_full.shape[0] != out_dim:
                    raise ValueError(f"LoRA B matrix for {full_layer_name} has incorrect out_features: expected {out_dim}, got {lora_b_full.shape[0]}")

                lora_a_sharded = lora_a_full.chunk(world_size, dim=1)[rank]
                lora_b_sharded = lora_b_full

                lora_layer = TensorParallelLoRALayer(
                    base_layer, r=lora_config.r_, scaling=lora_config.scaling_,
                    lora_a=lora_a_sharded.to(base_layer.weight.device),
                    lora_b=lora_b_sharded.to(base_layer.weight.device)
                )

                hook_handle = base_layer.register_forward_hook(
                    lambda m, i, o, l=lora_layer: hook(m, i, o, l)
                )
                self.adapters[adapter_name].append({"name": full_layer_name, "hook": hook_handle})

    def offload_adapter(self, adapter_name: str):
        if adapter_name not in self.adapters: return
        for adapter_info in self.adapters[adapter_name]:
            adapter_info["hook"].remove()
        del self.adapters[adapter_name]

def precompute_mask(input_tokens, n_heads, device, dtype):
    batch_size, seq_len = input_tokens.shape
    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask

class LlamaModel_TP(LLMModel):
    def __init__(self, args: LLMModelArgs, config):
        super().__init__()
        
        self.args = args
        self.name_or_path_ = args.name_or_path_
        self.vocab_size_ = args.vocab_size_
        self.pad_token_id_ = args.pad_token_id_
        self.device_ = args.device_
        self.n_heads_ = args.n_heads_

        self.embed_tokens = torch.nn.Embedding(args.vocab_size_, args.dim_, self.pad_token_id_)
        self.layers = ModuleList([TensorParallelDecoderLayer(args, config) for _ in range(args.n_layers_)])
        self.norm = RMSNorm(args.dim_, eps=args.norm_eps_)
        self.lm_head = torch.nn.Linear(args.dim_, args.vocab_size_, bias=False)

    @override
    def forward(self, input: ModelData) -> torch.Tensor:
        tokens = torch.tensor(input.batch_tokens_, dtype=torch.int64, device=self.device_)
        mask = precompute_mask(tokens, self.n_heads_, self.device_, self.args.dtype_)
        
        hidden_states = self.embed_tokens(tokens)
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask, input)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits

    @override
    @staticmethod
    def from_pretrained(path: str, device: str, precision: str, **kwargs) -> "LlamaModel_TP":
        rank = get_tensor_parallel_rank()
        world_size = get_tensor_parallel_world_size()

        config = AutoConfig.from_pretrained(path)
        llama_args = LLMModelArgs(config)
        llama_args.device_ = device

        load_type_dict = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }

        additional_load_args = {
            "low_cpu_mem_usage": True
        }

        if precision in load_type_dict:
            additional_load_args["torch_dtype"] = load_type_dict[precision]
        else:
            load_4bit = precision in ["nf4", "fp4"]
            load_8bit = precision == "int8"

            additional_load_args["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            additional_load_args["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=load_4bit,
                load_in_8bit=load_8bit,
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type=precision,
            )
        
        llama_args.dtype_ = additional_load_args["torch_dtype"]

        with torch.device('meta'):
            model = LlamaModel_TP(llama_args, config)

        model.to_empty(device=device)

        logging.info(f"Rank {rank} loading checkpoint from disk with precision {precision}...")
        hf_model = AutoModelForCausalLM.from_pretrained(path, **additional_load_args)
        state_dict = {k.replace("model.", ""): v for k, v in hf_model.state_dict().items()}
        del hf_model

        if world_size > 1:
            dist.barrier()

        for name, param in model.named_parameters():
            if name not in state_dict:
                logging.warning(f"Weight {name} not found in checkpoint")
                continue
            
            source_tensor = state_dict[name]
            
            if "q_proj" in name or "k_proj" in name or "v_proj" in name or "gate_proj" in name or "up_proj" in name:
                sharded_tensor = source_tensor.chunk(world_size, dim=0)[rank]
                param.data.copy_(sharded_tensor)
            elif "o_proj" in name or "down_proj" in name:
                sharded_tensor = source_tensor.chunk(world_size, dim=1)[rank]
                param.data.copy_(sharded_tensor)
            else: 
                param.data.copy_(source_tensor)

        logging.info(f"Rank {rank} successfully loaded its shard of the model to {device}.")
        del state_dict
        return model

    @override
    def load_adapter(self, adapter_name: str, adapter_config: dict):
        for i, layer in enumerate(self.layers):
            layer.load_adapter(f"layers.{i}", adapter_name, adapter_config)

    @override
    def offload_adapter(self, adapter_name: str):
        for layer in self.layers:
            layer.offload_adapter(adapter_name)

    @override
    def linears_info(self) -> OrderedDict[str, LinearInfo]:
        ret_val = OrderedDict()
        for i, layer in enumerate(self.layers):
            layer_info = layer.linears_info()
            for name, info in layer_info.items():
                ret_val[f"layers.{i}.{name}"] = info
        return ret_val

    @override
    def sequential(self) -> Sequential:
        raise NotImplementedError("The tensor-parallel model does not have a sequential structure.")
