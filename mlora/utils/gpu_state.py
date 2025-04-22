from __future__ import annotations
from dataclasses import dataclass
from typing import List

import pynvml as nvml

@dataclass
class AdapterProfile:
    param_bytes:         int
    flops_per_token_fwd: int
    flops_per_token_bwd: int

@dataclass
class GPUState:
    id:        int
    total_mem: int # in MiB
    free_mem:  int
    used_mem:  int
    gpu_util:  int # in percent
    mem_util:  int

    def can_fit(self, prof, util_target: int = 90) -> bool:
        would_use_mem = self.used_mem + prof.param_bytes // (1024 * 1024)
        would_use_util = self.gpu_util + prof.compute_util
        return would_use_mem < self.total_mem and would_use_util < util_target


def query_all_gpus() -> List[GPUState]:
    nvml.nvmlInit()
    try:
        num = nvml.nvmlDeviceGetCount()
        states: List[GPUState] = []
        for i in range(num):
            h = nvml.nvmlDeviceGetHandleByIndex(i)

            mem = nvml.nvmlDeviceGetMemoryInfo(h)
            util = nvml.nvmlDeviceGetUtilizationRates(h)
            states.append(
                GPUState(
                    id=i,
                    total_mem=mem.total // (1024 * 1024),
                    free_mem=mem.free // (1024 * 1024),
                    used_mem=mem.used // (1024 * 1024),
                    gpu_util=util.gpu,
                    mem_util=util.memory,
                )
            )
        return states
    finally:
        nvml.nvmlShutdown()