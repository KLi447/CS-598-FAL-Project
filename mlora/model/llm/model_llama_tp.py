import logging
import os
from collections import OrderedDict
from typing import Dict, List, Optional, override

import torch
import torch.distributed as dist
from torch.nn import Module, ModuleList, Parameter, Sequential
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM

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
        dist.all_reduce(grad_output)
        return grad_output

class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
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
            dist.all_reduce(lora_output)
        return lora_output

class TensorParallelAttention(torch.nn.Module):
    def __init__(self, args: LLMModelArgs):
        super().__init__()
        self.n_heads = args.n_heads_
        self.n_kv_heads = args.n_kv_heads_
        self.head_dim = args.dim_ // args.n_heads_
        world_size = get_tensor_parallel_world_size()

        self.n_heads_per_partition = self.n_heads // world_size
        self.n_kv_heads_per_partition = self.n_kv_heads // world_size

        self.q_proj = ColumnParallelLinear(args.dim_, args.dim_, bias=False, device=args.device_, dtype=args.dtype_)
        self.k_proj = ColumnParallelLinear(args.dim_, self.n_kv_heads * self.head_dim, bias=False, device=args.device_, dtype=args.dtype_)
        self.v_proj = ColumnParallelLinear(args.dim_, self.n_kv_heads * self.head_dim, bias=False, device=args.device_, dtype=args.dtype_)
        self.o_proj = RowParallelLinear(args.dim_, args.dim_, bias=False, device=args.device_, dtype=args.dtype_)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, data: ModelData):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads_per_partition, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads_per_partition, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads_per_partition, self.head_dim).transpose(1, 2)

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
        x = self.input_layernorm(x)
        x = self.self_attn(x, mask, data)
        x = residual + x
        
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        
        return x

    def linears_info(self) -> OrderedDict[str, LinearInfo]:
        world_size = get_tensor_parallel_world_size()
        info = OrderedDict()
        for name, module in self.named_modules():
            submodule_name = name.split('.')[-1]
            if isinstance(module, ColumnParallelLinear):
                info[submodule_name] = LinearInfo(module.weight.shape[1], module.out_features_per_partition * world_size)
            elif isinstance(module, RowParallelLinear):
                info[submodule_name] = LinearInfo(module.in_features_per_partition * world_size, module.weight.shape[0])
        return info

    def load_adapter(self, adapter_model: AdapterModel):
        rank = get_tensor_parallel_rank()
        world_size = get_tensor_parallel_world_size()
        self.adapters[adapter_model.name] = []
        
        for name, base_layer in self.named_modules():
            clean_name = name.split('.')[-1]
            if clean_name not in adapter_model.target_modules:
                continue

            lora_a_full = adapter_model.lora_a_weights[clean_name]
            lora_b_full = adapter_model.lora_b_weights[clean_name]
            
            if isinstance(base_layer, ColumnParallelLinear):
                lora_b_sharded = lora_b_full.chunk(world_size, dim=0)[rank]
                lora_a_sharded = lora_a_full
            elif isinstance(base_layer, RowParallelLinear):
                lora_a_sharded = lora_a_full.chunk(world_size, dim=1)[rank]
                lora_b_sharded = lora_b_full
            else:
                continue
            
            lora_layer = TensorParallelLoRALayer(
                base_layer, adapter_model.r, adapter_model.scaling, 
                lora_a_sharded.to(base_layer.weight.device), 
                lora_b_sharded.to(base_layer.weight.device)
            )

            def hook(module, input, output):
                return output + lora_layer(input[0])
            
            hook_handle = base_layer.register_forward_hook(hook)
            self.adapters[adapter_model.name].append({"name": name, "hook": hook_handle})

    def offload_adapter(self, adapter_name: str):
        if adapter_name not in self.adapters: return
        for adapter_info in self.adapters[adapter_name]:
            adapter_info["hook"].remove()
        del self.adapters[adapter_name]

def precompute_mask(input_tokens, n_heads, device, dtype):
    batch_size, seq_len = input_tokens.shape
    mask = torch.full((batch_size, 1, seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask

class LlamaModel_TP(LLMModel):
    def __init__(self, args: LLMModelArgs, config):
        super().__init__()
        
        self.args = args
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
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
        llama_args.dtype_ = dtype

        with torch.device('meta'):
            model = LlamaModel_TP(llama_args, config)

        logging.info(f"Rank {rank} loading checkpoint from disk...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        state_dict = {k.replace("model.", ""): v for k, v in hf_model.state_dict().items()}
        del hf_model

        if world_size > 1:
            dist.barrier()

        for name, param in state_dict.items():
            module_path, _, param_name = name.rpartition('.')
            
            if "q_proj" in name or "k_proj" in name or "v_proj" in name or "gate_proj" in name or "up_proj" in name:
                sharded_param = param.chunk(world_size, dim=0)[rank]
                model.get_submodule(module_path).weight.data.copy_(sharded_param)
            elif "o_proj" in name or "down_proj" in name:
                sharded_param = param.chunk(world_size, dim=1)[rank]
                model.get_submodule(module_path).weight.data.copy_(sharded_param)
            else: # Replicated weights
                model.get_submodule(module_path).get_parameter(param_name).data.copy_(param)

        model.to(device)
        logging.info(f"Rank {rank} successfully loaded its shard of the model to {device}.")
        del state_dict
        return model

    @override
    def load_adapter(self, adapter_model: AdapterModel):
        for layer in self.layers:
            layer.load_adapter(adapter_model)

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
