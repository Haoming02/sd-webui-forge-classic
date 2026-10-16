# Adapted from OpenMOSS/MOSS-Image (Apache-2.0), comfyui/moss-image/model.py.
"""MOSS-Image transformer using Forge operations and attention."""

import torch
from torch import nn

from backend.attention import attention_function


def _situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def _rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def _attention(q, k, v, rope, heads):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = _rotate(q.reshape(shape), rope).transpose(1, 2)
    k = _rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    return attention_function(q, k, v, heads, skip_reshape=True)


class _TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads, device, dtype, operations):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = (
            operations.RMSNorm(dim, eps=1e-6, device=device, dtype=dtype),
            operations.RMSNorm(dim, eps=1e-6, device=device, dtype=dtype),
        )
        self.q_norm, self.k_norm = (
            operations.RMSNorm(dim // heads, eps=1e-6, device=device, dtype=dtype),
            operations.RMSNorm(dim // heads, eps=1e-6, device=device, dtype=dtype),
        )
        self.qkv = operations.Linear(dim, 3 * dim, device=device, dtype=dtype)
        self.proj = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.ffn_gate, self.ffn_up = (
            operations.Linear(dim, ffn_dim, device=device, dtype=dtype),
            operations.Linear(dim, ffn_dim, device=device, dtype=dtype),
        )
        self.ffn_down = operations.Linear(ffn_dim, dim, device=device, dtype=dtype)

    def forward(self, x, rope):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(_attention(q, k, v, rope, self.heads))
        y = self.norm2(x)
        return x + self.ffn_down(_situ(self.ffn_up(y), self.ffn_gate(y)))


class _TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads, device, dtype, operations):
        super().__init__()
        self.proj = operations.Linear(text_dim, dim, device=device, dtype=dtype)
        self.blocks = nn.ModuleList(
            [_TextBlock(dim, ffn_dim, heads, device, dtype, operations) for _ in range(2)]
        )

    def forward(self, x, rope):
        x = self.proj(x)
        for block in self.blocks:
            x = block(x, rope)
        return x


class _Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads, device, dtype, operations):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = (
            operations.RMSNorm(dim, eps=1e-6, device=device, dtype=dtype),
            operations.RMSNorm(dim, eps=1e-6, device=device, dtype=dtype),
        )
        self.q_norm, self.k_norm = (
            operations.RMSNorm(dim // heads, eps=1e-6, device=device, dtype=dtype),
            operations.RMSNorm(dim // heads, eps=1e-6, device=device, dtype=dtype),
        )
        self.proj = nn.ModuleDict(
            {
                name: operations.Linear(dim, dim, bias=False, device=device, dtype=dtype)
                for name in ("q", "k", "v", "o")
            }
        )
        self.proj["ffn_up_gate"] = operations.Linear(dim, 2 * ffn_dim, bias=False, device=device, dtype=dtype)
        self.proj["ffn_down"] = operations.Linear(ffn_dim, dim, bias=False, device=device, dtype=dtype)

    def forward(self, x, rope, modulation):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        dtype = x.dtype
        y = y.to(dtype)
        q = self.q_norm(self.proj["q"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj["k"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj["o"](_attention(q, k, self.proj["v"](y), rope, self.heads)) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
        return x + self.proj["ffn_down"](_situ(hidden, gate)) * gate_f[:, None]


class _RoPE(nn.Module):
    def __init__(self, head_dim, base):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        self.register_buffer("bands", self._default_bands(torch.device("cpu")))

    def _default_bands(self, device):
        # Match training's CPU fp32 construction, including under meta init.
        exponent = torch.arange(self.num_bands, device="cpu", dtype=torch.float32) / self.num_bands
        return (1 / (self.base**exponent)).to(device)

    def forward(self, height, width, text_length, device):
        bands = self._default_bands(device)
        # Coordinates: (r, f, h, w); sequence order is image followed by text.
        h, w = torch.meshgrid(
            torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij"
        )
        image = (
            torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1)
            .flatten(0, 1)
            .float()
        )
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        phase = torch.cat((image, text), 0).unsqueeze(-1) * bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


class MOSSImageTransformer(nn.Module):
    """MOSSImage DiT with training-compatible keys and Forge attention.

    Each sample is evaluated at its true text length, including text refinement;
    padded tokens cannot influence either text or joint attention.
    """

    def __init__(
        self,
        dim=3072,
        ffn_dim=8192,
        num_heads=24,
        num_layers=42,
        text_dim=4096,
        freq_dim=256,
        latent_channels=128,
        rope_base=10000.0,
        device=None,
        dtype=None,
        operations=nn,
        uncond_skip_layers=(22,),
        **kwargs,
    ):
        super().__init__()
        if dim % num_heads or (dim // num_heads) % 8 or freq_dim % 2:
            raise ValueError("Require dim divisible by heads, head_dim divisible by 8, and even freq_dim.")
        self.uncond_skip_layers = frozenset(uncond_skip_layers)
        self.latent_channels = latent_channels
        self.freq_dim = freq_dim
        self.video_embedder = operations.Linear(latent_channels, dim, device=device, dtype=dtype)
        self.text_embedder = _TextProjection(text_dim, dim, ffn_dim, num_heads, device, dtype, operations)
        self.rope = _RoPE(dim // num_heads, rope_base)
        self.time_embedding = nn.Sequential(
            operations.Linear(freq_dim, dim, device=device, dtype=dtype),
            nn.SiLU(),
            operations.Linear(dim, dim, device=device, dtype=dtype),
        )
        self.shared_block_modulation = operations.Linear(dim, 6 * dim, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [_Block(dim, ffn_dim, num_heads, device, dtype, operations) for _ in range(num_layers)]
        )
        self.final_norm_video = operations.RMSNorm(dim, eps=1e-6, device=device, dtype=dtype)
        self.final_linear_video = operations.Linear(
            dim, latent_channels, bias=False, device=device, dtype=dtype
        )
        self.video_head_modulation = operations.Linear(dim, 2 * dim, device=device, dtype=dtype)

    def forward(self, x, timestep, context, attention_mask=None, transformer_options={}, **kwargs):
        transformer_options = transformer_options.copy()
        blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
        hidden_states, encoder_hidden_states = x, context
        encoder_attention_mask = attention_mask
        if hidden_states.ndim != 4 or hidden_states.shape[1] != self.latent_channels:
            raise ValueError("hidden_states must be [B, latent_channels, H, W].")
        batch, _, height, width = hidden_states.shape
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch:
            raise ValueError("encoder_hidden_states must be [B, S, text_dim].")
        mask = encoder_attention_mask
        if mask is None:
            mask = torch.ones(encoder_hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        if mask.shape != encoder_hidden_states.shape[:2] or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("encoder_attention_mask must be a binary [B, S] mask.")
        times = torch.as_tensor(timestep, device=hidden_states.device, dtype=torch.float32).reshape(-1)
        times = times.expand(batch)
        # Training computes the sinusoid in float64 and feeds float32 into its MLP.
        half = self.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(hidden_states.dtype)).float()
        mod = self.shared_block_modulation(time.to(hidden_states.dtype)).view(
            batch, 6, -1
        )
        head = self.video_head_modulation(time.to(hidden_states.dtype)).view(batch, 2, -1)
        uncond_indices = transformer_options.get("uncond_indices", ())
        outputs = []
        for index in range(batch):
            context = encoder_hidden_states[index : index + 1, mask[index].bool()]
            if context.shape[1] == 0:
                raise ValueError("Each sample needs at least one valid text token.")
            video = hidden_states[index : index + 1].flatten(2).transpose(1, 2)
            video = self.video_embedder(video.to(hidden_states.dtype).contiguous())
            # Match the reference activation dtype under fp32 weights + autocast.
            rope = self.rope(height, width, context.shape[1], hidden_states.device).to(video.dtype)
            text = self.text_embedder(context, rope[:, height * width :])
            x = torch.cat((video, text), 1)
            transformer_options["total_blocks"] = len(self.layers)
            transformer_options["block_type"] = "single"
            for layer_index, layer in enumerate(self.layers):
                if index in uncond_indices and layer_index in self.uncond_skip_layers:
                    continue
                transformer_options["block_index"] = layer_index
                if ("single_block", layer_index) in blocks_replace:

                    def block_wrap(args):
                        return {"img": layer(args["img"], args["pe"], args["vec"])}

                    out = blocks_replace[("single_block", layer_index)](
                        {
                            "img": x,
                            "pe": rope,
                            "vec": mod[index : index + 1],
                            "transformer_options": transformer_options,
                        },
                        {"original_block": block_wrap},
                    )
                    x = out["img"]
                else:
                    x = layer(x, rope, mod[index : index + 1])
            x = self.final_norm_video(x[:, : height * width].to(hidden_states.dtype))
            x = x * (1 + head[index : index + 1, 1, None]) + head[index : index + 1, 0, None]
            x = self.final_linear_video(x.to(hidden_states.dtype))
            outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))
        sample = torch.cat(outputs)
        return sample
