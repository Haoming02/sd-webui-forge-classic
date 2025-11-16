# implementation of Chroma for Forge
# Credit:    croquelois
# Reference: https://github.com/lodestone-rock/ComfyUI_FluxMod

from dataclasses import dataclass

import torch
from einops import rearrange, repeat
from torch import nn

from backend.nn.flux import (
    EmbedND,
    MLPEmbedder,
    QKNorm,
    RMSNorm,
    SelfAttention,
    attention,
    timestep_embedding,
)
from backend.utils import fp16_fix


class Approximator(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers=4):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim, bias=True)
        self.layers = nn.ModuleList([MLPEmbedder(hidden_dim, hidden_dim) for x in range(n_layers)])
        self.norms = nn.ModuleList([RMSNorm(hidden_dim) for x in range(n_layers)])
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.in_proj(x)
        for layer, norms in zip(self.layers, self.norms):
            x = x + layer(norms(x))
        x = self.out_proj(x)
        return x


@dataclass
class ModulationOut:
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio, qkv_bias=False):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img: torch.Tensor, txt: torch.Tensor, mod: torch.Tensor, pe: torch.Tensor, attn_mask=None, transformer_options={}):
        (img_mod1, img_mod2), (txt_mod1, txt_mod2) = mod

        img_modulated = torch.addcmul(img_mod1.shift, 1 + img_mod1.scale, self.img_norm1(img))
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = torch.addcmul(txt_mod1.shift, 1 + txt_mod1.scale, self.txt_norm1(txt))
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        attn = attention(
            torch.cat((txt_q, img_q), dim=2),
            torch.cat((txt_k, img_k), dim=2),
            torch.cat((txt_v, img_v), dim=2),
            pe=pe,
            mask=attn_mask,
            transformer_options=transformer_options,
        )

        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        img.addcmul_(img_mod1.gate, self.img_attn.proj(img_attn))
        img.addcmul_(img_mod2.gate, self.img_mlp(torch.addcmul(img_mod2.shift, 1 + img_mod2.scale, self.img_norm2(img))))

        txt.addcmul_(txt_mod1.gate, self.txt_attn.proj(txt_attn))
        txt.addcmul_(txt_mod2.gate, self.txt_mlp(torch.addcmul(txt_mod2.shift, 1 + txt_mod2.scale, self.txt_norm2(txt))))

        return img, fp16_fix(txt)


class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qk_scale=None):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
        self.norm = QKNorm(head_dim)
        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor, mod: torch.Tensor, pe: torch.Tensor, attn_mask=None, transformer_options={}) -> torch.Tensor:
        x_mod = torch.addcmul(mod.shift, 1 + mod.scale, self.pre_norm(x))
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k = self.norm(q, k, v)

        attn = attention(q, k, v, pe=pe, mask=attn_mask, transformer_options=transformer_options)
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        x.addcmul_(mod.gate, output)
        return fp16_fix(x)


class LastLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    def forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        shift, scale = mod
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        x = torch.addcmul(shift[:, None, :], 1 + scale[:, None, :], self.norm_final(x))
        x = self.linear(x)
        return x


class IntegratedChromaTransformer2DModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, context_in_dim: int, hidden_size: int, mlp_ratio: float, num_heads: int, depth: int, depth_single_blocks: int, axes_dim: list[int], theta: int, patch_size: int, qkv_bias: bool, in_dim: int, out_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        if hidden_size % num_heads != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}")

        pe_dim = hidden_size // num_heads
        if sum(axes_dim) != pe_dim:
            raise ValueError(f"Got {axes_dim} but expected positional dim {pe_dim}")

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.distilled_guidance_layer = Approximator(in_dim, out_dim, hidden_dim, n_layers)
        self.txt_in = nn.Linear(context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    @staticmethod
    def distribute_modulations(tensor, single_block_count: int = 38, double_blocks_count: int = 19):
        """
        Distributes slices of the tensor into the block_dict as ModulationOut objects.

        Args:
            tensor (torch.Tensor): Input tensor with shape [batch_size, vectors, dim].
        """
        block_dict = {}
        for i in range(single_block_count):
            key = f"single_blocks.{i}.modulation.lin"
            block_dict[key] = None
        for i in range(double_blocks_count):
            key = f"double_blocks.{i}.img_mod.lin"
            block_dict[key] = None
        for i in range(double_blocks_count):
            key = f"double_blocks.{i}.txt_mod.lin"
            block_dict[key] = None
        block_dict["final_layer.adaLN_modulation.1"] = None
        idx = 0  # Index to keep track of the vector slices
        for key in block_dict.keys():
            if "single_blocks" in key:
                # Single block: 1 ModulationOut
                block_dict[key] = ModulationOut(shift=tensor[:, idx : idx + 1, :], scale=tensor[:, idx + 1 : idx + 2, :], gate=tensor[:, idx + 2 : idx + 3, :])
                idx += 3  # Advance by 3 vectors
            elif "img_mod" in key:
                # Double block: List of 2 ModulationOut
                double_block = []
                for _ in range(2):  # Create 2 ModulationOut objects
                    double_block.append(ModulationOut(shift=tensor[:, idx : idx + 1, :], scale=tensor[:, idx + 1 : idx + 2, :], gate=tensor[:, idx + 2 : idx + 3, :]))
                    idx += 3  # Advance by 3 vectors per ModulationOut
                block_dict[key] = double_block
            elif "txt_mod" in key:
                # Double block: List of 2 ModulationOut
                double_block = []
                for _ in range(2):  # Create 2 ModulationOut objects
                    double_block.append(ModulationOut(shift=tensor[:, idx : idx + 1, :], scale=tensor[:, idx + 1 : idx + 2, :], gate=tensor[:, idx + 2 : idx + 3, :]))
                    idx += 3  # Advance by 3 vectors per ModulationOut
                block_dict[key] = double_block
            elif "final_layer" in key:
                # Final layer: 1 ModulationOut
                block_dict[key] = [
                    tensor[:, idx : idx + 1, :],
                    tensor[:, idx + 1 : idx + 2, :],
                ]
                idx += 2  # Advance by 2 vectors
        return block_dict

    def inner_forward(self, img, img_ids, txt, txt_ids, timesteps):
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")
        img = self.img_in(img)
        device = img.device
        dtype = img.dtype  # torch.bfloat16
        nb_double_block = len(self.double_blocks)
        nb_single_block = len(self.single_blocks)

        mod_index_length = nb_double_block * 12 + nb_single_block * 3 + 2
        distill_timestep = timestep_embedding(timesteps.detach().clone(), 16).to(device=device, dtype=dtype)
        distil_guidance = timestep_embedding(torch.zeros_like(timesteps), 16).to(device=device, dtype=dtype)
        modulation_index = timestep_embedding(torch.arange(mod_index_length), 32).to(device=device, dtype=dtype)
        modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1)
        timestep_guidance = torch.cat([distill_timestep, distil_guidance], dim=1).unsqueeze(1).repeat(1, mod_index_length, 1)
        input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1)
        mod_vectors = self.distilled_guidance_layer(input_vec)
        mod_vectors_dict = self.distribute_modulations(mod_vectors, nb_single_block, nb_double_block)

        txt = self.txt_in(txt)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        del txt_ids, img_ids
        pe = self.pe_embedder(ids)
        del ids
        for i, block in enumerate(self.double_blocks):
            img_mod = mod_vectors_dict[f"double_blocks.{i}.img_mod.lin"]
            txt_mod = mod_vectors_dict[f"double_blocks.{i}.txt_mod.lin"]
            double_mod = [img_mod, txt_mod]
            img, txt = block(img=img, txt=txt, mod=double_mod, pe=pe)
        img = torch.cat((txt, img), 1)
        for i, block in enumerate(self.single_blocks):
            single_mod = mod_vectors_dict[f"single_blocks.{i}.modulation.lin"]
            img = block(img, mod=single_mod, pe=pe)
        del pe
        img = img[:, txt.shape[1] :, ...]
        final_mod = mod_vectors_dict["final_layer.adaLN_modulation.1"]
        img = self.final_layer(img, final_mod)
        return img

    def forward(self, x: torch.Tensor, timestep, context, **kwargs):
        bs, c, h, w = x.shape
        input_device = x.device
        input_dtype = x.dtype
        pad_h = (self.patch_size - x.shape[-2] % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - x.shape[-1] % self.patch_size) % self.patch_size
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="circular")
        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=self.patch_size, pw=self.patch_size)
        del x, pad_h, pad_w
        h_len = (h + (self.patch_size // 2)) // self.patch_size
        w_len = (w + (self.patch_size // 2)) // self.patch_size
        img_ids = torch.zeros((h_len, w_len, 3), device=input_device, dtype=input_dtype)
        img_ids[..., 1] = img_ids[..., 1] + torch.linspace(0, h_len - 1, steps=h_len, device=input_device, dtype=input_dtype)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.linspace(0, w_len - 1, steps=w_len, device=input_device, dtype=input_dtype)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
        txt_ids = torch.zeros((bs, context.shape[1], 3), device=input_device, dtype=input_dtype)
        del input_device, input_dtype
        out = self.inner_forward(img, img_ids, context, txt_ids, timestep)
        del img, img_ids, txt_ids, timestep, context
        out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:, :, :h, :w]
        del h_len, w_len, bs
        return out
