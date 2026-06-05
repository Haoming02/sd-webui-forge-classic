# https://github.com/Comfy-Org/ComfyUI/blob/v0.24.1/comfy/ldm/modules/diffusionmodules/mmdit.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.nn.lumina import timestep_embedding


class FeedForwardSwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        super().__init__()

        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, max_period=10000):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

    def forward(self, t, dtype, **kwargs):
        t_freq = timestep_embedding(t, self.frequency_embedding_size, max_period=self.max_period).to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, bias=True, drop=0.0):
        super().__init__()

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = nn.Identity()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos, device=None, dtype=torch.float32):
    omega = torch.arange(embed_dim // 2, device=device, dtype=dtype)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb
