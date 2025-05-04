from __future__ import annotations
from dataclasses import dataclass
from typing import List

import pynvml as nvml
import torch

from flops_profiler.profiler import FlopsProfiler

@dataclass
class AdapterProfile:
    def __init__(self, param_bytes, flops_per_token_fwd, flops_per_token_bwd, 
                 activation_memory=0, gradient_memory=0, optimizer_memory=0):
        self.param_bytes = param_bytes
        self.flops_per_token_fwd = flops_per_token_fwd
        self.flops_per_token_bwd = flops_per_token_bwd
        self.activation_memory = activation_memory
        self.gradient_memory = gradient_memory
        self.optimizer_memory = optimizer_memory
        
    def total_memory_estimate(self, batch_size, seq_length):
        act_mem = self.activation_memory * batch_size * seq_length
        grad_mem = self.param_bytes
        opt_mem = self.optimizer_memory or (self.param_bytes * 2)
        return self.param_bytes + act_mem + grad_mem + opt_mem


@dataclass
class GPUState:
    id:        int
    total_mem: int # in MiB
    free_mem:  int
    used_mem:  int
    reserved_mem: int
    gpu_util:  int # in percent
    mem_util:  int

def get_gpu_memory_info(device_id=0):
    free_memory, total_memory = torch.cuda.mem_get_info(device_id)
    allocated = torch.cuda.memory_allocated(device_id)
    reserved = torch.cuda.memory_reserved(device_id)
    return {
        "free": free_memory,
        "total": total_memory,
        "allocated": allocated,
        "reserved": reserved
    }


def query_all_gpus() -> List[GPUState]:
    nvml.nvmlInit()
    try:
        num = nvml.nvmlDeviceGetCount()
        states: List[GPUState] = []
        for i in range(num):
            h = nvml.nvmlDeviceGetHandleByIndex(i)

            mem = get_gpu_memory_info(i)
            util = nvml.nvmlDeviceGetUtilizationRates(h)
            states.append(
                GPUState(
                    id=i,
                    total_mem=mem["total"] // (1024 * 1024),
                    free_mem=mem["free"] // (1024 * 1024),
                    used_mem=mem["allocated"] // (1024 * 1024),
                    reserved_mem=mem["reserved"] // (1024 * 1024),
                    gpu_util=util.gpu,
                    mem_util=util.memory,
                )
            )
        return states
    finally:
        nvml.nvmlShutdown()

def profile_adapter(model, adapter_name, sample_input):
    model.set_adapter(adapter_name)
    
    prof = FlopsProfiler(model)
    prof.start_profile()
    
    output = model(sample_input)
    loss = output.mean()

    loss.backward()
    
    flops = prof._get_total_flops()
    params = prof._get_total_params()
    prof.end_profile()
    
    return flops, params