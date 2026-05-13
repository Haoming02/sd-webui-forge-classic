# https://github.com/kohya-ss/ComfyUI-Anima-LLLite/blob/main/control_net_lllite_anima.py

import logging
import math
from typing import Final, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.state_dict import load_state_dict

logger = logging.getLogger("ControlNet")


# region Consts


TARGET_ATTENTION_CLASS: Final[str] = "SelfCrossAttention"
TARGET_MLP_CLASS: Final[str] = "GPT2FeedForward"

ATOMIC_SPECIFIERS: Final[tuple[str]] = (
    "self_attn_q_pre",
    "self_attn_kv_pre",
    "cross_attn_q_pre",
    "mlp_fc1_pre",
)

PRESETS: Final[dict[str, tuple[str]]] = {
    "self_attn_q": ("self_attn_q_pre",),
    "self_attn_qkv": ("self_attn_q_pre", "self_attn_kv_pre"),
    "self_attn_qkv_cross_q": ("self_attn_q_pre", "self_attn_kv_pre", "cross_attn_q_pre"),
}

ASPP_DEFAULT_DILATIONS: Final[tuple[int]] = (1, 2, 4, 8)


_INTERNAL_MODULES_PREFIX = "lllite_modules."
_INTERNAL_COND_PREFIX = "conditioning1."
_INTERNAL_DEPTH_KEY = "depth_embeds"
_SAVED_COND_PREFIX = "lllite_conditioning1."
_SAVED_DEPTH_SUFFIX = ".depth_embed"


def parse_target_layers(spec: str) -> tuple[str]:
    spec = spec.strip()
    if spec in PRESETS:
        return PRESETS[spec]

    parts = [p.strip() for p in spec.split(",") if p.strip()]
    assert not any([p not in ATOMIC_SPECIFIERS for p in parts])

    return tuple(a for a in ATOMIC_SPECIFIERS if a in parts)


# region Conditioning


def _gn(channels: int) -> nn.GroupNorm:
    g = 8
    while g > 1 and channels % g != 0:
        g //= 2
    return nn.GroupNorm(g, channels)


class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.norm1 = _gn(ch)
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.norm2 = _gn(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class _ASPP(nn.Module):
    def __init__(self, ch: int, dilations: tuple[int] = ASPP_DEFAULT_DILATIONS):
        super().__init__()
        branches = []
        for d in dilations:
            conv = nn.Conv2d(ch, ch, kernel_size=1) if d == 1 else nn.Conv2d(ch, ch, kernel_size=3, padding=d, dilation=d)
            branches.append(nn.Sequential(conv, _gn(ch), nn.SiLU()))
        self.branches = nn.ModuleList(branches)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=1), _gn(ch), nn.SiLU())
        n = len(dilations) + 1
        self.proj = nn.Sequential(nn.Conv2d(ch * n, ch, kernel_size=1), _gn(ch), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        outs = [b(x) for b in self.branches]
        g = self.global_conv(self.global_pool(x))
        g = F.interpolate(g, size=(h, w), mode="bilinear", align_corners=False)
        outs.append(g)
        return self.proj(torch.cat(outs, dim=1))


class _Conditioning(nn.Module):
    def __init__(self, cond_dim: int, cond_emb_dim: int, n_resblocks: int, use_aspp: bool = False, aspp_dilations: tuple[int] = ASPP_DEFAULT_DILATIONS, cond_in_ch: int = 3):
        super().__init__()
        ch_half = cond_dim // 2
        self.conv1 = nn.Conv2d(cond_in_ch, ch_half, kernel_size=4, stride=4, padding=0)
        self.norm1 = _gn(ch_half)
        self.conv2 = nn.Conv2d(ch_half, ch_half, kernel_size=3, stride=1, padding=1)
        self.norm2 = _gn(ch_half)
        self.conv3 = nn.Conv2d(ch_half, cond_dim, kernel_size=4, stride=4, padding=0)
        self.norm3 = _gn(cond_dim)
        self.resblocks = nn.ModuleList([_ResBlock(cond_dim) for _ in range(n_resblocks)])
        self.aspp = _ASPP(cond_dim, aspp_dilations) if use_aspp else None
        self.proj = nn.Conv2d(cond_dim, cond_emb_dim, kernel_size=1)
        self.out_norm = nn.LayerNorm(cond_emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = F.silu(self.norm2(self.conv2(h)))
        h = F.silu(self.norm3(self.conv3(h)))
        for rb in self.resblocks:
            h = rb(h)
        if self.aspp is not None:
            h = self.aspp(h)
        h = self.proj(h)
        b, c, hh, ww = h.shape
        h = h.view(b, c, hh * ww).permute(0, 2, 1).contiguous()
        return self.out_norm(h)


# region Per-Linear LLLite Module


class LLLiteModuleDiT(nn.Module):
    def __init__(self, name: str, org_module: nn.Linear, cond_emb_dim: int, mlp_dim: int, dropout: Optional[float] = None, multiplier: float = 1.0):
        super().__init__()
        self.lllite_name = name

        self.org_module: list[nn.Linear] = [org_module]
        self.multiplier = multiplier
        self.dropout = dropout

        in_dim = org_module.in_features
        self.down = nn.Linear(in_dim, mlp_dim)
        self.mid = nn.Linear(mlp_dim + cond_emb_dim, mlp_dim)

        self.cond_to_film = nn.Linear(cond_emb_dim, 2 * mlp_dim)
        nn.init.zeros_(self.cond_to_film.weight)
        nn.init.zeros_(self.cond_to_film.bias)

        self.up = nn.Linear(mlp_dim, in_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

        self.cond_emb: Optional[torch.Tensor] = None
        self.org_forward = None
        self.layer_idx: int = -1
        self._depth_embeds_ref: list[nn.Parameter] = []

        self.num_steps: int = 0
        self.start_step: int = 0
        self.end_step: int = 0
        self.current_step: int = 0
        self.is_first: bool = False

    def apply_to(self):
        if self.org_forward is None:
            self.org_forward = self.org_module[0].forward
            self.org_module[0].forward = self.forward

    def restore(self):
        if self.org_forward is not None:
            self.org_module[0].forward = self.org_forward
            self.org_forward = None

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        is_5d = x.dim() == 5

        def _pass():
            return self.org_forward(x.reshape(orig_shape) if is_5d else x)

        if self.multiplier == 0.0 or self.cond_emb is None:
            return _pass()

        if self.num_steps > 0:
            step = self.current_step
            self.current_step += 1
            if self.current_step >= self.num_steps:
                self.current_step = 0
            if step < self.start_step or step >= self.end_step:
                return _pass()

        if is_5d:
            B, T, H, W, D = orig_shape
            x = x.reshape(B, T * H * W, D)

        cx = self.cond_emb

        if x.shape[0] != cx.shape[0]:
            if x.shape[0] % cx.shape[0] != 0:
                return self.org_forward(x.reshape(orig_shape) if is_5d else x)
            cx = cx.repeat(x.shape[0] // cx.shape[0], 1, 1)

        if x.shape[1] != cx.shape[1]:
            return self.org_forward(x.reshape(orig_shape) if is_5d else x)

        param_dtype = self.down.weight.dtype
        x_proc = x if x.dtype == param_dtype else x.to(param_dtype)
        if cx.dtype != param_dtype or cx.device != x.device:
            cx = cx.to(device=x.device, dtype=param_dtype)

        if self._depth_embeds_ref:
            depth_e = self._depth_embeds_ref[0][self.layer_idx]
            if depth_e.dtype != param_dtype or depth_e.device != x.device:
                depth_e = depth_e.to(device=x.device, dtype=param_dtype)
            cond_local = cx + depth_e
        else:
            cond_local = cx

        h = F.silu(self.down(x_proc))
        gamma, beta = self.cond_to_film(cond_local).chunk(2, dim=-1)
        m = self.mid(torch.cat([cond_local, h], dim=-1))
        m = F.silu(m * (1 + gamma) + beta)
        if self.dropout is not None and self.training:
            m = F.dropout(m, p=self.dropout)

        out = self.up(m) * self.multiplier

        if out.dtype != x.dtype:
            out = out.to(x.dtype)

        y = self.org_forward(x + out)
        if is_5d:
            y = y.reshape(orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3], -1)

        return y


# region ControlNetLLLiteDiT


class ControlNetLLLiteDiT(nn.Module):
    def __init__(self, dit: nn.Module, cond_emb_dim: int = 32, mlp_dim: int = 64, target_layers: str = "self_attn_q", dropout: Optional[float] = None, multiplier: float = 1.0, cond_dim: int = 64, cond_resblocks: int = 1, use_aspp: bool = False, aspp_dilations: tuple[int] = ASPP_DEFAULT_DILATIONS, cond_in_ch: int = 3):
        super().__init__()
        atomics = parse_target_layers(target_layers)
        self.multiplier = multiplier
        self.target_atomics = atomics

        self.conditioning1 = _Conditioning(
            cond_dim,
            cond_emb_dim,
            cond_resblocks,
            use_aspp=use_aspp,
            aspp_dilations=aspp_dilations,
            cond_in_ch=cond_in_ch,
        )
        modules = self._create_modules(dit, cond_emb_dim, mlp_dim, atomics, dropout, multiplier)
        self.lllite_modules: list[LLLiteModuleDiT] = nn.ModuleList(modules)

        n = len(self.lllite_modules)
        self.depth_embeds = nn.Parameter(torch.zeros(n, cond_emb_dim))
        for i, m in enumerate(self.lllite_modules):
            m.layer_idx = i
            m._depth_embeds_ref = [self.depth_embeds]

        logger.info(f"Loaded Control-LLLite (Anima) ({n} modules)")

    @staticmethod
    def _attn_atomic_match(is_self_attn: bool, child_name: str, atomics: tuple[str]) -> bool:
        if "output_proj" in child_name:
            return False
        if is_self_attn:
            if child_name == "q_proj":
                return "self_attn_q_pre" in atomics
            if child_name in ("k_proj", "v_proj"):
                return "self_attn_kv_pre" in atomics
        else:
            if child_name == "q_proj":
                return "cross_attn_q_pre" in atomics
        return False

    def _create_modules(self, dit, cond_emb_dim, mlp_dim, atomics, dropout, multiplier):
        modules = []
        want_mlp = "mlp_fc1_pre" in atomics
        any_attn = any(a in atomics for a in ("self_attn_q_pre", "self_attn_kv_pre", "cross_attn_q_pre"))

        for name, module in dit.named_modules():
            cls = module.__class__.__name__

            if any_attn and cls == TARGET_ATTENTION_CLASS:
                if not hasattr(module, "is_SelfAttn"):
                    continue
                is_self_attn = bool(module.is_SelfAttn)
                for child_name, child in module.named_children():
                    if not isinstance(child, nn.Linear):
                        continue
                    if not self._attn_atomic_match(is_self_attn, child_name, atomics):
                        continue
                    full_name = f"lllite_dit.{name}.{child_name}".replace(".", "_")
                    modules.append(LLLiteModuleDiT(full_name, child, cond_emb_dim, mlp_dim, dropout, multiplier))

            elif want_mlp and cls == TARGET_MLP_CLASS:
                child = getattr(module, "layer1", None)
                if not isinstance(child, nn.Linear):
                    continue
                full_name = f"lllite_dit.{name}.layer1".replace(".", "_")
                modules.append(LLLiteModuleDiT(full_name, child, cond_emb_dim, mlp_dim, dropout, multiplier))

        return modules

    def set_cond_image(self, cond_image: Optional[torch.Tensor]):
        """cond_image: (B, 3, H, W) in [-1, 1]. None clears."""
        if cond_image is None:
            for m in self.lllite_modules:
                m.cond_emb = None
            return
        cx = self.conditioning1(cond_image)
        for m in self.lllite_modules:
            m.cond_emb = cx

    def set_multiplier(self, multiplier: float):
        self.multiplier = multiplier
        for m in self.lllite_modules:
            m.multiplier = multiplier

    def set_step_range(self, num_steps: int, start_percent: float, end_percent: float):
        start_step = math.floor(num_steps * start_percent) if start_percent > 0 else 0
        end_step = math.floor(num_steps * end_percent) if end_percent > 0 else num_steps

        for i, m in enumerate(self.lllite_modules):
            m.num_steps = num_steps
            m.start_step = start_step
            m.end_step = end_step
            m.current_step = 0
            m.is_first = i == 0

    def apply_to(self):
        for m in self.lllite_modules:
            m.apply_to()

    def restore(self):
        for m in self.lllite_modules:
            m.restore()
        self.set_cond_image(None)


# region Weight Loading (v2)


def _from_saved_state_dict(lllite: ControlNetLLLiteDiT, weights_sd: dict[str, torch.Tensor]) -> dict:
    name_to_idx = {m.lllite_name: i for i, m in enumerate(lllite.lllite_modules)}
    n_modules = len(name_to_idx)
    out: dict = {}
    depth_slices: dict = {}

    for k, v in weights_sd.items():
        if k.startswith(_SAVED_COND_PREFIX):
            out[_INTERNAL_COND_PREFIX + k[len(_SAVED_COND_PREFIX) :]] = v
            continue
        if k.endswith(_SAVED_DEPTH_SUFFIX):
            name = k[: -len(_SAVED_DEPTH_SUFFIX)]
            if name in name_to_idx:
                depth_slices[name_to_idx[name]] = v
                continue
        head, dot, tail = k.partition(".")
        if dot and head in name_to_idx:
            out[f"{_INTERNAL_MODULES_PREFIX}{name_to_idx[head]}.{tail}"] = v
            continue
        out[k] = v

    if depth_slices:
        missing = [i for i in range(n_modules) if i not in depth_slices]
        if missing:
            raise RuntimeError(f"depth_embed slices missing for module indices: {missing}")
        out[_INTERNAL_DEPTH_KEY] = torch.stack([depth_slices[i] for i in range(n_modules)], dim=0)
    return out


def load_lllite_weights_from_dict(lllite: ControlNetLLLiteDiT, state_dict: dict[str, torch.Tensor]):
    assert not any(k.startswith(_INTERNAL_MODULES_PREFIX) for k in state_dict)
    converted = _from_saved_state_dict(lllite, state_dict)
    load_state_dict(lllite, converted)


def infer_anima_config(state_dict: dict[str, torch.Tensor]) -> dict:
    """Reconstruct ControlNetLLLiteDiT constructor kwargs from a saved state dict."""
    cond_emb_dim = 32
    cond_dim = 64
    cond_in_ch = 3
    mlp_dim = 64
    cond_resblocks = 0
    use_aspp = False
    aspp_dilations = ASPP_DEFAULT_DILATIONS

    for k, v in state_dict.items():
        if k == "lllite_conditioning1.proj.weight":
            cond_emb_dim = v.shape[0]
        elif k == "lllite_conditioning1.conv1.weight":
            cond_dim = v.shape[0] * 2
            cond_in_ch = v.shape[1]

    for k, v in state_dict.items():
        if k.endswith(".down.weight") and "conditioning1" not in k:
            mlp_dim = v.shape[0]
            break

    rb_indices = set()
    for k in state_dict:
        if "lllite_conditioning1.resblocks." in k:
            parts = k.split(".")
            try:
                idx = parts.index("resblocks")
                rb_indices.add(int(parts[idx + 1]))
            except (ValueError, IndexError):
                pass

    if rb_indices:
        cond_resblocks = max(rb_indices) + 1

    use_aspp = any("lllite_conditioning1.aspp" in k for k in state_dict)

    has_self_q = any("self_attn_q_proj.down.weight" in k for k in state_dict)
    has_self_kv = any(("self_attn_k_proj" in k or "self_attn_v_proj" in k) and k.endswith(".down.weight") for k in state_dict)
    has_cross_q = any("cross_attn_q_proj.down.weight" in k for k in state_dict)
    has_mlp = any("_mlp_layer1.down.weight" in k for k in state_dict)

    parts = []
    if has_self_q:
        parts.append("self_attn_q_pre")
    if has_self_kv:
        parts.append("self_attn_kv_pre")
    if has_cross_q:
        parts.append("cross_attn_q_pre")
    if has_mlp:
        parts.append("mlp_fc1_pre")
    target_layers = ",".join(parts) if parts else "self_attn_q"

    return dict(
        cond_emb_dim=cond_emb_dim,
        mlp_dim=mlp_dim,
        target_layers=target_layers,
        cond_dim=cond_dim,
        cond_in_ch=cond_in_ch,
        cond_resblocks=cond_resblocks,
        use_aspp=use_aspp,
        aspp_dilations=aspp_dilations,
    )
