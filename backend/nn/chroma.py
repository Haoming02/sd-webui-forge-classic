# https://github.com/Comfy-Org/ComfyUI/tree/v0.10.0/comfy/ldm/chroma

from dataclasses import dataclass

import torch
from einops import rearrange, repeat
from torch import nn

from backend.nn.flux import (
    DoubleStreamBlock,
    EmbedND,
    MLPEmbedder,
    ModulationOut,
    SingleStreamBlock,
    timestep_embedding,
)
from backend.utils import fp16_fix, pad_to_patch_size


class ChromaModulationOut(ModulationOut):
    @classmethod
    def from_offset(cls, tensor: torch.Tensor, offset: int = 0) -> ModulationOut:
        return cls(
            shift=tensor[:, offset : offset + 1, :],
            scale=tensor[:, offset + 1 : offset + 2, :],
            gate=tensor[:, offset + 2 : offset + 3, :],
        )


class Approximator(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers=5):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim, bias=True)
        self.layers = nn.ModuleList([MLPEmbedder(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.RMSNorm(hidden_dim) for _ in range(n_layers)])
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        for layer, norms in zip(self.layers, self.norms):
            x = x + layer(norms(x))
        x = self.out_proj(x)
        return x


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        shift, scale = vec
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        x = torch.addcmul(shift[:, None, :], 1 + scale[:, None, :], self.norm_final(x))
        x = self.linear(x)
        return x


@dataclass
class ChromaParams:
    in_channels: int
    out_channels: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list
    theta: int
    patch_size: int
    qkv_bias: bool
    in_dim: int
    out_dim: int
    hidden_dim: int
    n_layers: int
    txt_ids_dims: list
    vec_in_dim: int


class IntegratedChromaTransformer2DModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        params = ChromaParams(**kwargs)
        self.params = params

        self.patch_size = params.patch_size
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}")
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.in_dim = params.in_dim
        self.out_dim = params.out_dim
        self.hidden_dim = params.hidden_dim
        self.n_layers = params.n_layers
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.distilled_guidance_layer = Approximator(in_dim=self.in_dim, hidden_dim=self.hidden_dim, out_dim=self.out_dim, n_layers=self.n_layers)

        self.double_blocks = nn.ModuleList([DoubleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio, qkv_bias=params.qkv_bias, modulation=False) for _ in range(params.depth)])

        self.single_blocks = nn.ModuleList([SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio, modulation=False) for _ in range(params.depth_single_blocks)])

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

        self.skip_mmdit = []
        self.skip_dit = []
        self.lite = False

    def get_modulations(self, tensor: torch.Tensor, block_type: str, *, idx: int = 0):
        #  This function slices up the modulations tensor which has the following layout:
        #    single     : num_single_blocks * 3 elements
        #    double_img : num_double_blocks * 6 elements
        #    double_txt : num_double_blocks * 6 elements
        #    final      : 2 elements
        if block_type == "final":
            return (tensor[:, -2:-1, :], tensor[:, -1:, :])
        single_block_count = self.params.depth_single_blocks
        double_block_count = self.params.depth
        offset = 3 * idx
        if block_type == "single":
            return ChromaModulationOut.from_offset(tensor, offset)
        offset *= 2
        if block_type in {"double_img", "double_txt"}:
            offset += 3 * single_block_count
            if block_type == "double_txt":
                offset += 6 * double_block_count
            return (
                ChromaModulationOut.from_offset(tensor, offset),
                ChromaModulationOut.from_offset(tensor, offset + 3),
            )
        raise ValueError("block_type")

    def forward_orig(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: torch.Tensor,
        guidance: torch.Tensor = None,
        control=None,
        transformer_options={},
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        patches_replace = transformer_options.get("patches_replace", {})

        img = self.img_in(img)

        mod_index_length = 344
        distill_timestep = timestep_embedding(timesteps.detach().clone(), 16).to(img.device, img.dtype)
        distil_guidance = timestep_embedding(guidance.detach().clone(), 16).to(img.device, img.dtype)

        modulation_index = timestep_embedding(torch.arange(mod_index_length, device=img.device), 32).to(img.device, img.dtype)
        modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1).to(img.device, img.dtype)
        timestep_guidance = torch.cat([distill_timestep, distil_guidance], dim=1).unsqueeze(1).repeat(1, mod_index_length, 1).to(img.dtype).to(img.device, img.dtype)
        input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1).to(img.device, img.dtype)

        mod_vectors = self.distilled_guidance_layer(input_vec)

        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        blocks_replace = patches_replace.get("dit", {})
        transformer_options["total_blocks"] = len(self.double_blocks)
        transformer_options["block_type"] = "double"
        for i, block in enumerate(self.double_blocks):
            transformer_options["block_index"] = i
            if i not in self.skip_mmdit:
                double_mod = (
                    self.get_modulations(mod_vectors, "double_img", idx=i),
                    self.get_modulations(mod_vectors, "double_txt", idx=i),
                )
                if ("double_block", i) in blocks_replace:

                    def block_wrap(args):
                        out = {}
                        out["img"], out["txt"] = block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], attn_mask=args.get("attn_mask"), transformer_options=args.get("transformer_options"))
                        return out

                    out = blocks_replace[("double_block", i)]({"img": img, "txt": txt, "vec": double_mod, "pe": pe, "attn_mask": attn_mask, "transformer_options": transformer_options}, {"original_block": block_wrap})
                    txt = out["txt"]
                    img = out["img"]
                else:
                    img, txt = block(img=img, txt=txt, vec=double_mod, pe=pe, attn_mask=attn_mask, transformer_options=transformer_options)

                if control is not None:  # ControlNet
                    control_i = control.get("input")
                    if i < len(control_i):
                        add = control_i[i]
                        if add is not None:
                            img += add

        img = fp16_fix(img)
        img = torch.cat((txt, img), 1)

        transformer_options["total_blocks"] = len(self.single_blocks)
        transformer_options["block_type"] = "single"
        for i, block in enumerate(self.single_blocks):
            transformer_options["block_index"] = i
            if i not in self.skip_dit:
                single_mod = self.get_modulations(mod_vectors, "single", idx=i)
                if ("single_block", i) in blocks_replace:

                    def block_wrap(args):
                        out = {}
                        out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"], attn_mask=args.get("attn_mask"), transformer_options=args.get("transformer_options"))
                        return out

                    out = blocks_replace[("single_block", i)]({"img": img, "vec": single_mod, "pe": pe, "attn_mask": attn_mask, "transformer_options": transformer_options}, {"original_block": block_wrap})
                    img = out["img"]
                else:
                    img = block(img, vec=single_mod, pe=pe, attn_mask=attn_mask, transformer_options=transformer_options)

                if control is not None:  # ControlNet
                    control_o = control.get("output")
                    if i < len(control_o):
                        add = control_o[i]
                        if add is not None:
                            img[:, txt.shape[1] :, ...] += add

        img = img[:, txt.shape[1] :, ...]
        if hasattr(self, "final_layer"):
            final_mod = self.get_modulations(mod_vectors, "final")
            img = self.final_layer(img, vec=final_mod)

        return img

    def forward(self, x, timestep, context, guidance=None, control=None, transformer_options={}, **kwargs):
        bs, c, h, w = x.shape
        x = pad_to_patch_size(x, (self.patch_size, self.patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=self.patch_size, pw=self.patch_size)

        if img.ndim != 3 or context.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        h_len = (h + (self.patch_size // 2)) // self.patch_size
        w_len = (w + (self.patch_size // 2)) // self.patch_size
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).unsqueeze(0)
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        out = self.forward_orig(img, img_ids, context, txt_ids, timestep, guidance or torch.zeros_like(timestep), control, transformer_options, attn_mask=kwargs.get("attention_mask", None))
        return rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=self.patch_size, pw=self.patch_size)[:, :, :h, :w]
