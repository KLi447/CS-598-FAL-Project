# File: mlora/model/modules/flora_per_example.py
# Fast‑LoRA per‑example adapter (shared & non‑shared)

from __future__ import annotations
import math
from typing import List, Optional
import logging
import torch
import torch.nn.functional as F
from mlora.model.modules.adapter import Adapter


class FFloraPerExampleAdapter(Adapter):
    """
    Fast‑LoRA adapter that supports

        shared=True   – one (B, A) pair for the whole batch
        shared=False  – distinct (Bᵢ, Aᵢ) for each example
    """

    # ────────────────────────────────────────────────────────────
    def __init__(
        self,
        adapter_name: str,
        batch_size: int,
        in_dim: int,
        out_dim: int,
        r: int,
        *,
        device: torch.device = torch.device("cpu"),
        activation_fn=None,
        optimizer: str = "adamw",
        lr: float = 3e-4,
        alpha: int = 64,
        dropout: float = 0.0,
        shared: bool = False,
    ):
        super().__init__("flora_per_example", adapter_name)

        self.batch_size_ = batch_size
        self.in_dim_     = in_dim
        self.out_dim_    = out_dim
        self.r_          = r
        self.shared_     = shared
        self.device_     = device
        self.act_        = activation_fn
        self.scaling_    = alpha / float(r)
        self.dropout_    = dropout
        
        if self.shared_:
            logging.info(
                "shared now"
            )
            self.B = torch.empty(in_dim, r, device=device, requires_grad=True)
            self.A = torch.empty(r,      out_dim, device=device, requires_grad=True)
            self._kaiming_(self.B)
            self._kaiming_(self.A)
        else:
            logging.info(
                "not shared now"
            )
            self.B_all = torch.empty(
                batch_size, in_dim, r, device=device, requires_grad=True
            )
            self.A_all = torch.empty(
                batch_size, r, out_dim, device=device, requires_grad=True
            )
            self._kaiming_(self.B_all)
            self._kaiming_(self.A_all)

    # ────────────────────────────────────────────────────────────
    @staticmethod
    def _kaiming_(t: torch.Tensor):
        torch.nn.init.kaiming_normal_(t, a=math.sqrt(5))

    # ----- hooks required by the framework ----------------------
    def init_weight(
        self,
        B_init: Optional[torch.Tensor] = None,
        A_init: Optional[torch.Tensor] = None,
    ):
        """
        Optional warm‑start called by TrainFloraPerExampleContext
        """
        if B_init is not None:
            if self.shared_:
                self.B.data.copy_(B_init.to(self.B))
            else:
                self.B_all.data.copy_(B_init.to(self.B_all))

        if A_init is not None:
            if self.shared_:
                self.A.data.copy_(A_init.to(self.A))
            else:
                self.A_all.data.copy_(A_init.to(self.A_all))

    def get_trainable_tensors(self) -> List[torch.Tensor]:
        return [self.B, self.A] if self.shared_ else [self.B_all, self.A_all]

    def get_all_tensors(self) -> List[torch.Tensor]:
        return self.get_trainable_tensors()

    def disable_grad(self):
        for p in self.get_trainable_tensors():
            p.requires_grad_(False)

    def enable_grad(self):
        for p in self.get_trainable_tensors():
            p.requires_grad_(True)

    # ------------------------------------------------------------
    def forward_per_example(self, X: torch.Tensor, W0: torch.Tensor) -> torch.Tensor:
        """
        X  : [B, S, in_dim]
        W0 : [in_dim, out_dim]  (frozen base weight)
        """
        B_sz, S, _ = X.shape
        if not self.shared_ and B_sz != self.batch_size_:
            raise ValueError("Batch‑size mismatch with non‑shared adapter")

        if self.shared_:
            # shared (B,A)
            BX  = X.unsqueeze(-1) * self.B                     # [B,S,in_dim,r]
            out = torch.einsum("bsir,io->bsro", BX, W0) * self.A   # [B,S,r,o]
        else:
            B = self.B_all.unsqueeze(1)                         # [B,1,in_dim,r]
            A = self.A_all.unsqueeze(1)                         # [B,1,r,o]
            BX = X.unsqueeze(-1) * B                            # [B,S,in_dim,r]
            out = torch.einsum("bsir,io->bsro", BX, W0) * A

        Y = out.sum(dim=2).mul(self.scaling_)
        if self.act_ is not None:
            Y = self.act_(Y)
        return Y
