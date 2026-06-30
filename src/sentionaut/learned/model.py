"""``UnifiedWorldModel``: FiLM-conditioned shared-weight learned world model.

A conv encoder of ``s_t`` + an action MLP, FiLM-conditioned on embeddings for the
percept-model id, the implant id, and continuous topography params, feeding a
conv decoder for ``s_{t+1}``. A ``mode`` switch collapses conditioning so the same
architecture trains per-model specialists (the ablation baseline).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FiLM(nn.Module):
    """Feature-wise linear modulation: per-channel scale/shift from conditioning."""

    def __init__(self, cond_dim: int, n_channels: int):
        super().__init__()
        self.fc = nn.Linear(cond_dim, n_channels * 2)
        self.n_channels = n_channels

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.fc(cond).chunk(2, dim=-1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return x * (1 + gamma) + beta


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm = nn.GroupNorm(min(8, cout), cout)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class UnifiedWorldModel(nn.Module):
    def __init__(
        self,
        action_dim: int,
        n_models: int,
        n_implants: int,
        base_channels: int = 32,
        cond_dim: int = 64,
        mode: str = "shared",
    ):
        super().__init__()
        assert mode in ("shared", "specialist")
        self.mode = mode
        self.cond_dim = cond_dim

        self.model_emb = nn.Embedding(n_models, cond_dim)
        self.implant_emb = nn.Embedding(n_implants, cond_dim)
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim + 2, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )

        c = base_channels
        self.enc1 = ConvBlock(1, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.film1 = FiLM(cond_dim, c)
        self.film2 = FiLM(cond_dim, c * 2)
        self.mid = ConvBlock(c * 2, c * 2)
        self.dec2 = ConvBlock(c * 2, c)
        self.dec1 = ConvBlock(c, c)
        self.head = nn.Conv2d(c, 1, 1)

    def conditioning(self, action, model_id, implant_id, topo_params) -> torch.Tensor:
        a = self.action_mlp(torch.cat([action, topo_params], dim=-1))
        if self.mode == "specialist":
            # Specialists drop the categorical conditioning (one model per net).
            return a
        return a + self.model_emb(model_id) + self.implant_emb(implant_id)

    def forward(self, s_t, action, model_id, implant_id, topo_params) -> torch.Tensor:
        cond = self.conditioning(action, model_id, implant_id, topo_params)
        x = self.enc1(s_t)
        x = self.film1(x, cond)
        x = self.enc2(x)
        x = self.film2(x, cond)
        x = self.mid(x)
        x = self.dec2(x)
        x = self.dec1(x)
        return self.head(x)
