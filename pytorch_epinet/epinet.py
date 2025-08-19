from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F



@dataclass
class EpinetConfig:
    num_classes: int
    index_dim: int = 32                  # Dz
    hidden_sizes: Iterable[int] = (50, 50) # epinet MLP widths
    prior_scale: float = 0.5             # weight on frozen prior head
    include_inputs: bool = False         # if True and batch is a Tensor, concat flat(batch) to hidden
    stop_grad_features: bool = True      # detach hidden before epinet

# ---------------------------------------------------------------------
# Epinet components
# ---------------------------------------------------------------------

class GaussianIndexer(nn.Module):
    """z ~ N(0, I) with shape [Dz], shared across batch."""
    def __init__(self, index_dim: int):
        super().__init__()
        self.index_dim = index_dim
    @torch.no_grad()
    def forward(self, device=None, dtype=None) -> torch.Tensor:
        return torch.randn(self.index_dim, device=device, dtype=dtype or torch.float32)


def _mlp(in_dim: int, hidden: Iterable[int], out_dim: int) -> nn.Sequential:
    layers = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class ProjectedMLP(nn.Module):
    def __init__(self, num_classes: int, index_dim: int, hidden: Iterable[int], concat_index: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.index_dim = index_dim
        self.hidden = tuple(hidden)
        self.concat_index = concat_index
        self.core: Optional[nn.Sequential] = None  # lazy init

    def _build(self, in_dim: int):
        # _mlp(in_dim, hidden, out_dim) must have NO activation on the final layer
        self.core = _mlp(in_dim, self.hidden, self.num_classes * self.index_dim)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Expect x: [B, Din], z: [Dz]  (shared across batch)
        if x.dim() != 2 or z.dim() != 1:
            raise ValueError("Expected x:[B,Din], z:[Dz].")
        if z.shape[0] != self.index_dim:
            raise ValueError(f"z dim {z.shape[0]} != index_dim {self.index_dim}")

        if self.concat_index:
            B = x.shape[0]
            z_cat = z.unsqueeze(0).expand(B, -1)   # [B, Dz]
            h = torch.cat([x, z_cat], dim=-1)      # [B, Din+Dz]
        else:
            h = x

        if self.core is None:
            self._build(h.shape[-1])

        out = self.core(h)                         # [B, C*Dz]
        B = x.shape[0]
        m = out.view(B, self.num_classes, self.index_dim)  # [B, C, Dz]
        return torch.einsum('bcd,d->bc', m, z) 


class MLPEpinetWithPrior(nn.Module):
    """
    epinet(hidden, inputs, z) = train([hidden,(+flat inputs)], z) + prior_scale * prior(...)
    - hidden: [B, D_hidden]
    - inputs: Tensor or None (only used if include_inputs=True)
    - z:      [B, Dz]
    """
    def __init__(self, cfg: EpinetConfig):
        super().__init__()
        self.cfg = cfg
        C, Dz = cfg.num_classes, cfg.index_dim
        self.train_head = ProjectedMLP(C, Dz, cfg.hidden_sizes)
        self.prior_head = ProjectedMLP(C, Dz, cfg.hidden_sizes)
        for p in self.prior_head.parameters():
            p.requires_grad = False
        self.indexer = GaussianIndexer(Dz)

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return x.view(B, -1)

    def forward(self,
                hidden: torch.Tensor,
                inputs: Optional[torch.Tensor],
                z: torch.Tensor) -> torch.Tensor:
        if hidden.dim() != 2:
            raise ValueError("hidden must be [B, D].")
        if self.cfg.include_inputs and inputs is not None:
            epi_in = torch.cat([hidden, self._flatten(inputs).to(hidden)], dim=-1)
        else:
            epi_in = hidden
        train = self.train_head(epi_in, z)
        prior = self.prior_head(epi_in, z)
        return train + self.cfg.prior_scale * prior

# ---------------------------------------------------------------------
# epinet wrapper
# ---------------------------------------------------------------------

class EpinetWrapper(nn.Module):
    """
    Combine a regular model with an epinet.

      - base_model: nn.Module
      - feature_fn: (base_model, batch) -> (mu:[B,C], hidden:[B,D])  # tensors
      - cfg: EpinetConfig

    Call:
      logits = wrapper(batch, n_index_samples=K, return_all=False/True)

    Notes:
      - We infer B/device for z *from the returned `hidden`* (no assumptions on `batch` type).
      - If cfg.include_inputs=True, raw `batch` must be a Tensor (else raw inputs are ignored).
    """
    def __init__(self,
                 base_model: nn.Module,
                 feature_fn: Callable[[nn.Module, Any], Tuple[torch.Tensor, torch.Tensor]],
                 cfg: EpinetConfig):
        super().__init__()
        self.base = base_model
        self.feature_fn = feature_fn
        self.cfg = cfg
        self.epinet = MLPEpinetWithPrior(cfg)

    def forward(self,
                batch: Any,
                n_index_samples: int = 1,
                return_all: bool = False,
                z: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1) Feature extraction (user-defined): must return tensors
        mu, hidden = self.feature_fn(self.base, batch)    # [B, C], [B, D]
        hidden_for_epi = hidden.detach() if self.cfg.stop_grad_features else hidden

        # 2) Sample z using hidden's batch size & device
        B = hidden.shape[0]
        device = hidden.device
        z_dtype = hidden.dtype if hidden.is_floating_point() else next(self.parameters()).dtype

        if z is None:
            if n_index_samples == 1:
                z = self.epinet.indexer(device=device, dtype=z_dtype)             # [B, Dz]
            else:
                # [S, Dz]
                z = torch.stack(
                    [self.epinet.indexer(device=device, dtype=z_dtype) for _ in range(n_index_samples)],
                    dim=0
                )

        # 3) (Optional) raw inputs to epinet if enabled and batch is a Tensor
        inputs_for_epinet: Optional[torch.Tensor] = None
        if self.cfg.include_inputs and isinstance(batch, torch.Tensor):
            inputs_for_epinet = batch

        # 4) Single vs multi z
        if z.dim() == 1:
            epi = self.epinet(hidden_for_epi, inputs_for_epinet, z)                  # [B, C]
            return mu + epi

        outs = []
        for s in range(z.shape[0]):
            epi_s = self.epinet(hidden_for_epi, inputs_for_epinet, z[s])             # [B, C]
            outs.append(mu + epi_s)
        stacked = torch.stack(outs, dim=0)                                           # [S, B, C]
        return stacked if return_all else stacked.mean(0)
