"""
Route A regression model: shared trunk + 7 domain heads.

Input per sample:
  e_sat, e_ntl, e_sv, e_poi_text : (d_txt,)  text embeddings (frozen encoder)
  v_poi                          : (d_poi_in,)  POI count features (log1p + z-score)

Output:
  (B, 7) logits in the same space as targets (logit((score-1)/9)).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from decision.data.targets import DOMAINS


class PoiMLP(nn.Module):
    def __init__(self, d_in: int, d_hid: int = 32, d_out: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hid, d_out),
        )

    def forward(self, v_poi: torch.Tensor) -> torch.Tensor:
        return self.net(v_poi)


class SharedTrunk(nn.Module):
    def __init__(self, d_in: int, d_out: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DomainHead(nn.Module):
    def __init__(self, d_in: int, d_hid: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.GELU(),
            nn.Linear(d_hid, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class RouteAModel(nn.Module):
    """Concat(4 frozen text embs, POI MLP output) → trunk → 7 heads."""

    def __init__(
        self,
        d_txt: int = 1024,
        d_poi_in: int = 16,
        d_poi: int = 32,
        d_trunk: int = 256,
        head_hidden: int = 64,
        dropout: float = 0.2,
        domains: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.domains = list(domains) if domains else list(DOMAINS)
        self.poi_mlp = PoiMLP(d_poi_in, d_hid=max(32, d_poi), d_out=d_poi, dropout=dropout / 2)
        self.trunk = SharedTrunk(4 * d_txt + d_poi, d_out=d_trunk, dropout=dropout)
        self.heads = nn.ModuleList([DomainHead(d_trunk, d_hid=head_hidden) for _ in self.domains])

    def forward(
        self,
        e_sat: torch.Tensor,
        e_ntl: torch.Tensor,
        e_sv: torch.Tensor,
        e_poi_text: torch.Tensor,
        v_poi: torch.Tensor,
    ) -> torch.Tensor:
        poi_feat = self.poi_mlp(v_poi)
        x = torch.cat([e_sat, e_ntl, e_sv, e_poi_text, poi_feat], dim=-1)
        z = self.trunk(x)
        return torch.stack([h(z) for h in self.heads], dim=-1)  # (B, 7)
