"""``UnifiedWorldModel``: a transformer (ViT-style) learned world model.

The input state image is tokenized into patches via a linear patch embedding plus
2D sinusoidal positional embeddings, processed by a Transformer encoder
(multi-head self-attention blocks), and decoded back to the next-percept image by
a linear patch-unembedding head (pure tensor reshape, no transposed conv). There
are no convolutions anywhere.

Conditioning on (percept-model id, implant id, topography params) + continuous
action is injected two ways: as conditioning tokens prepended to the patch
sequence, and as FiLM modulation of the patch tokens. A ``mode`` switch collapses
the categorical conditioning so the same architecture trains per-model
specialists (the ablation baseline).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def sincos_2d_pos_embed(n_h: int, n_w: int, dim: int, device, dtype) -> torch.Tensor:
    """Parameter-free 2D sinusoidal positional embedding, shape ``(n_h*n_w, dim)``.

    Being computed on the fly keeps the model agnostic to the percept grid size
    (which varies across configs), unlike a fixed learned table.
    """
    assert dim % 4 == 0, "transformer dim must be divisible by 4 for 2D sincos"
    quarter = dim // 4
    omega = torch.arange(quarter, device=device, dtype=dtype) / quarter
    omega = 1.0 / (10000.0**omega)  # (quarter,)
    gy = torch.arange(n_h, device=device, dtype=dtype)
    gx = torch.arange(n_w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(gy, gx, indexing="ij")
    yy = yy.reshape(-1)[:, None] * omega[None, :]  # (N, quarter)
    xx = xx.reshape(-1)[:, None] * omega[None, :]
    pe = torch.cat([torch.sin(yy), torch.cos(yy), torch.sin(xx), torch.cos(xx)], dim=1)
    return pe  # (N, dim)


class FiLM(nn.Module):
    """Feature-wise linear modulation of token features from a conditioning vector."""

    def __init__(self, cond_dim: int, dim: int):
        super().__init__()
        self.fc = nn.Linear(cond_dim, dim * 2)

    def forward(self, tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.fc(cond).chunk(2, dim=-1)
        return tokens * (1 + gamma[:, None, :]) + beta[:, None, :]


class UnifiedWorldModel(nn.Module):
    def __init__(
        self,
        action_dim: int,
        n_models: int,
        n_implants: int,
        dim: int = 64,
        depth: int = 4,
        heads: int = 4,
        patch_size: int = 4,
        mlp_ratio: float = 2.0,
        mode: str = "shared",
    ):
        super().__init__()
        assert mode in ("shared", "specialist")
        assert dim % 4 == 0
        self.mode = mode
        self.dim = dim
        self.patch_size = patch_size
        self.in_channels = 1
        patch_dim = self.in_channels * patch_size * patch_size

        # Linear patch embedding / unembedding (no convolutions).
        self.patch_embed = nn.Linear(patch_dim, dim)
        self.patch_unembed = nn.Linear(dim, patch_dim)

        # Conditioning: action MLP + categorical embeddings.
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim + 2, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.model_emb = nn.Embedding(n_models, dim)
        self.implant_emb = nn.Embedding(n_implants, dim)
        # Learned type embeddings so cond tokens are distinguishable from patches.
        self.cond_type = nn.Parameter(torch.zeros(3, dim))
        nn.init.normal_(self.cond_type, std=0.02)
        self.film = FiLM(dim, dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(dim)

    # Kept for backward compatibility / introspection.
    def conditioning(self, action, model_id, implant_id, topo_params) -> torch.Tensor:
        a = self.action_mlp(torch.cat([action, topo_params], dim=-1))
        if self.mode == "specialist":
            return a
        return a + self.model_emb(model_id) + self.implant_emb(implant_id)

    def _to_patches(self, x: torch.Tensor):
        """(B, 1, H, W) -> patches (B, N, patch_dim) with padded grid dims."""
        b, c, h, w = x.shape
        p = self.patch_size
        pad_h = (p - h % p) % p
        pad_w = (p - w % p) % p
        if pad_h or pad_w:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = h + pad_h, w + pad_w
        n_h, n_w = hp // p, wp // p
        x = x.reshape(b, c, n_h, p, n_w, p).permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(b, n_h * n_w, c * p * p)
        return x, (h, w, n_h, n_w)

    def _from_patches(self, tokens: torch.Tensor, shape) -> torch.Tensor:
        """Inverse of ``_to_patches``: (B, N, patch_dim) -> (B, 1, H, W)."""
        h, w, n_h, n_w = shape
        b = tokens.shape[0]
        c, p = self.in_channels, self.patch_size
        x = tokens.reshape(b, n_h, n_w, c, p, p).permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(b, c, n_h * p, n_w * p)
        return x[:, :, :h, :w]

    def forward(self, s_t, action, model_id, implant_id, topo_params) -> torch.Tensor:
        patches, shape = self._to_patches(s_t)
        b, n, _ = patches.shape
        tokens = self.patch_embed(patches)

        pe = sincos_2d_pos_embed(shape[2], shape[3], self.dim, tokens.device, tokens.dtype)
        tokens = tokens + pe[None, :, :]

        action_tok = self.action_mlp(torch.cat([action, topo_params], dim=-1))
        cond_vec = action_tok
        cond_tokens = [action_tok + self.cond_type[0]]
        if self.mode == "shared":
            model_tok = self.model_emb(model_id)
            implant_tok = self.implant_emb(implant_id)
            cond_vec = action_tok + model_tok + implant_tok
            cond_tokens.append(model_tok + self.cond_type[1])
            cond_tokens.append(implant_tok + self.cond_type[2])

        tokens = self.film(tokens, cond_vec)
        cond_seq = torch.stack(cond_tokens, dim=1)  # (B, n_cond, dim)
        seq = torch.cat([cond_seq, tokens], dim=1)

        seq = self.encoder(seq)
        seq = self.norm(seq)

        patch_tokens = seq[:, -n:, :]
        out_patches = self.patch_unembed(patch_tokens)
        return self._from_patches(out_patches, shape)
