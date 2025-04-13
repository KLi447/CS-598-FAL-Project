# File: mlora/model/modules/flora_per_example.py

import math
import torch
import torch.nn.functional as F
from typing import Optional, List

from mlora.model.modules.adapter import Adapter  # <-- Inherit from here

class FFloraPerExampleAdapter(Adapter):
    """
    Per-example Fast LoRA adapter, inheriting from the base "Adapter" class.

    Instead of having a single [in_dim, r] / [r, out_dim] for an entire domain,
    we store a separate B_i and A_i for each example i in the batch:
      B_all has shape [batch_size, in_dim, r]
      A_all has shape [batch_size, r, out_dim]

    Then we do eq. (6) from the "Batched Low-Rank Adaptation" paper:
      Y = A_i ∘ ( (B_i ∘ X_i) W0 )
    for each example i, all in one pass.
    """

    def __init__(
        self,
        adapter_name: str,
        batch_size: int,
        in_dim: int,
        out_dim: int,
        r: int,
        device: torch.device = torch.device("cpu"),
        activation_fn = None,
        optimizer: str = "adamw",
        lr: float = 3e-4,
        alpha: int = 64,
        dropout: float = 0.0,
    ):
        # Call the base Adapter constructor
        super().__init__("flora_per_example", adapter_name)

        self.batch_size_ = batch_size
        self.in_dim_ = in_dim
        self.out_dim_ = out_dim
        self.r_ = r
        self.device_ = device
        self.activation_ = activation_fn

        self.optimizer_ = optimizer
        self.lr_ = lr
        self.alpha_ = alpha
        self.dropout_ = dropout

        # Compute scaling factor (commonly alpha / r)
        self.scaling_ = self.alpha_ / float(self.r_)

        # Store B_i and A_i for each example i
        self.B_all = torch.zeros(
            (batch_size, in_dim, r),
            dtype=torch.float32,
            device=device,
            requires_grad=True
        )
        self.A_all = torch.zeros(
            (batch_size, r, out_dim),
            dtype=torch.float32,
            device=device,
            requires_grad=True
        )

        # Optionally init them (kaiming)
        self.init_weight()

    def init_weight(
        self,
        B_init: Optional[torch.Tensor] = None,
        A_init: Optional[torch.Tensor] = None
    ):
        """
        Initialize each example's B_i, A_i either from user-provided Tensors or
        with Kaiming normal.
        """
        with torch.no_grad():
            if B_init is not None:
                assert B_init.shape == self.B_all.shape
                self.B_all.copy_(B_init)
            else:
                for i in range(self.batch_size_):
                    torch.nn.init.kaiming_normal_(self.B_all[i], a=math.sqrt(5))

            if A_init is not None:
                assert A_init.shape == self.A_all.shape
                self.A_all.copy_(A_init)
            else:
                for i in range(self.batch_size_):
                    torch.nn.init.kaiming_normal_(self.A_all[i], a=math.sqrt(5))

    ############################################################################
    # Implementation for Adapter base class
    ############################################################################
    def get_trainable_tensors(self) -> List[torch.Tensor]:
        # Return the per-example adapter weights that can be learned.
        return [self.B_all, self.A_all]

    def get_all_tensors(self) -> List[torch.Tensor]:
        # Same as trainable for now. If there were non-trainable data, we'd add it.
        return [self.B_all, self.A_all]

    def disable_grad(self):
        for t in self.get_trainable_tensors():
            t.requires_grad_(False)

    def enable_grad(self):
        for t in self.get_trainable_tensors():
            t.requires_grad_(True)

    ############################################################################
    # The core forward pass for eq. (6):
    ############################################################################
    def forward_per_example(self, X: torch.Tensor, W0: torch.Tensor) -> torch.Tensor:
        """
        X: shape [batch_size, seq_len, in_dim]
        W0: shape [in_dim, out_dim]
        Return: shape [batch_size, seq_len, out_dim]
        """
        B, seq_len, in_dim = X.shape
        assert B == self.batch_size_
        assert in_dim == self.in_dim_

        # (B_i ∘ X_i)
        X_4d = X.unsqueeze(-1)          # [B, seq_len, in_dim, 1]
        B_4d = self.B_all.unsqueeze(1)  # [B, 1, in_dim, r]
        BX_4d = X_4d * B_4d             # [B, seq_len, in_dim, r]

        # multiply by W0 => shape [in_dim, out_dim]
        # Flatten to do matmul in a single pass
        BX_3d = BX_4d.view(B * seq_len, in_dim, self.r_)

        tmp = torch.einsum("bir,io->bro", BX_3d, W0)  # => [B*seq_len, r, out_dim]

        # multiply by A_i
        # A_all: [B, r, out_dim], replicate each example across seq_len
        A_expanded = self.A_all.unsqueeze(1).expand(B, seq_len, self.r_, self.out_dim_)
        A_3d = A_expanded.contiguous().view(B * seq_len, self.r_, self.out_dim_)

        # => [B*seq_len, r, out_dim]
        out_3d = tmp * A_3d  

        # reduce across r dimension (mean)
        out_2d = out_3d.mean(dim=1)  # => [B*seq_len, out_dim]

        # reshape => [B, seq_len, out_dim]
        Y = out_2d.view(B, seq_len, self.out_dim_)

        # optional activation
        if self.activation_ is not None:
            Y = self.activation_(Y)

        return Y
