# https://github.com/Comfy-Org/ComfyUI/blob/v0.28.0/comfy/ldm/pixeldit/pid.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.args import dynamic_args

from .model import PixDiT_T2I
from .modules import precompute_freqs_cis_2d


class SigmaAwareGate(nn.Module):
    def __init__(self, dim: int, per_token: bool = False):
        super().__init__()

        self.content_proj = nn.Linear(dim * 2, 1 if per_token else dim)
        self.log_alpha = nn.Parameter(torch.empty(()))

    def forward(self, x: torch.Tensor, lq: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        content_logit = self.content_proj(torch.cat([x, lq], dim=-1))
        log_alpha = self.log_alpha.to(device=x.device, dtype=torch.float32)
        sigma_offset = -log_alpha.exp() * sigma.float().view(-1, 1, 1)
        gate = torch.sigmoid(content_logit + sigma_offset)
        return x + (gate * lq).to(x.dtype)


class ResBlock(nn.Module):
    def __init__(self, channels: int, num_groups: int = 4, conv_padding_mode: str = "zeros"):
        super().__init__()

        self.block = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode=conv_padding_mode),
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode=conv_padding_mode),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class LQProjection2D(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        hidden_dim: int = 512,
        out_dim: int = 1536,
        patch_size: int = 16,
        sr_scale: int = 4,
        latent_spatial_down_factor: int = 8,
        latent_unpatchify_factor: int = 1,
        num_res_blocks: int = 4,
        num_outputs: int = 7,
        interval: int = 2,
        conv_padding_mode: str = "zeros",
        gate_per_token: bool = False,
        pit_output: bool = False,
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.sr_scale = sr_scale
        self.latent_spatial_down_factor = latent_spatial_down_factor
        self.latent_unpatchify_factor = latent_unpatchify_factor
        self.num_outputs = num_outputs
        self.interval = interval

        effective_latent_channels = latent_channels // (latent_unpatchify_factor * latent_unpatchify_factor)
        effective_spatial_down_factor = latent_spatial_down_factor // latent_unpatchify_factor
        z_to_patch_ratio = (sr_scale * effective_spatial_down_factor) / patch_size
        self.z_to_patch_ratio = z_to_patch_ratio
        if z_to_patch_ratio >= 1:
            self.latent_fold_factor = 0
            latent_proj_in_ch = effective_latent_channels
        else:
            fold_factor = int(1 / z_to_patch_ratio)
            assert fold_factor * z_to_patch_ratio == 1.0
            self.latent_fold_factor = fold_factor
            latent_proj_in_ch = effective_latent_channels * fold_factor * fold_factor

        layers = [
            nn.Conv2d(latent_proj_in_ch, hidden_dim, kernel_size=3, padding=1, padding_mode=conv_padding_mode),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, padding_mode=conv_padding_mode),
        ]
        for _ in range(num_res_blocks):
            layers.append(ResBlock(hidden_dim, conv_padding_mode=conv_padding_mode))
        self.latent_proj = nn.Sequential(*layers)

        self.output_heads = nn.ModuleList([nn.Linear(hidden_dim, out_dim) for _ in range(num_outputs)])
        self.pit_head = nn.Linear(hidden_dim, out_dim) if pit_output else None
        self.gate_modules = nn.ModuleList([SigmaAwareGate(out_dim, per_token=gate_per_token) for _ in range(num_outputs)])

    def is_gate_active(self, block_idx: int) -> bool:
        return block_idx % self.interval == 0

    def output_index(self, block_idx: int) -> int:
        return block_idx // self.interval

    def gate(self, x: torch.Tensor, lq_feature: torch.Tensor, sigma: torch.Tensor, out_idx: int) -> torch.Tensor:
        return self.gate_modules[out_idx](x, lq_feature, sigma)

    def _align_latent_to_patch_grid(self, lq_latent: torch.Tensor, pH: int, pW: int) -> torch.Tensor:
        f = self.latent_unpatchify_factor
        if f > 1:
            B, C, H, W = lq_latent.shape
            lq_latent = lq_latent.reshape(B, C // (f * f), f, f, H, W)
            lq_latent = lq_latent.permute(0, 1, 4, 2, 5, 3).reshape(B, C // (f * f), H * f, W * f)
        B, z_dim = lq_latent.shape[:2]
        if self.z_to_patch_ratio >= 1:
            if lq_latent.shape[2] != pH or lq_latent.shape[3] != pW:
                z_aligned = F.interpolate(lq_latent, size=(pH, pW), mode="nearest")
            else:
                z_aligned = lq_latent
        else:
            f = self.latent_fold_factor
            zH_expected, zW_expected = pH * f, pW * f
            if lq_latent.shape[2] != zH_expected or lq_latent.shape[3] != zW_expected:
                lq_latent = F.interpolate(lq_latent, size=(zH_expected, zW_expected), mode="nearest")
            z_aligned = lq_latent.reshape(B, z_dim, pH, f, pW, f).permute(0, 1, 3, 5, 2, 4)
            z_aligned = z_aligned.reshape(B, z_dim * f * f, pH, pW)
        return self.latent_proj(z_aligned)

    def forward(self, lq_latent: torch.Tensor, target_pH: int, target_pW: int) -> list[torch.Tensor]:
        feat = self._align_latent_to_patch_grid(lq_latent, target_pH, target_pW)
        B, C, H, W = feat.shape
        tokens = feat.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        outputs = [head(tokens) for head in self.output_heads]
        if self.pit_head is not None:
            outputs.append(self.pit_head(tokens))
        return outputs


class PidNet(PixDiT_T2I):
    def __init__(
        self,
        lq_latent_channels: int = 16,
        lq_hidden_dim: int = 512,
        lq_num_res_blocks: int = 4,
        lq_interval: int = 2,
        sr_scale: int = 4,
        latent_spatial_down_factor: int = 8,
        lq_latent_unpatchify_factor: int = 1,
        lq_conv_padding_mode: str = "zeros",
        lq_gate_per_token: bool = False,
        pit_lq_inject: bool = False,
        rope_ref_h: int = 1024,
        rope_ref_w: int = 1024,
        **pixdit_kwargs,
    ):
        super().__init__(**pixdit_kwargs)

        self.rope_ref_grid_h = rope_ref_h // self.patch_size
        self.rope_ref_grid_w = rope_ref_w // self.patch_size

        def _pit_rope_fn(head_dim, h, w, device=None, dtype=torch.float32, **rope_opts):
            return precompute_freqs_cis_2d(head_dim, h, w, ref_grid_h=self.rope_ref_grid_h, ref_grid_w=self.rope_ref_grid_w, device=device, dtype=dtype, **rope_opts)

        for blk in self.pixel_blocks:
            blk._rope_fn = _pit_rope_fn

        self.pit_lq_inject = pit_lq_inject

        num_lq_outputs = (self.patch_depth + lq_interval - 1) // lq_interval
        self.lq_proj = LQProjection2D(
            latent_channels=lq_latent_channels,
            hidden_dim=lq_hidden_dim,
            out_dim=self.hidden_size,
            patch_size=self.patch_size,
            sr_scale=sr_scale,
            latent_spatial_down_factor=latent_spatial_down_factor,
            latent_unpatchify_factor=lq_latent_unpatchify_factor,
            num_res_blocks=lq_num_res_blocks,
            num_outputs=num_lq_outputs,
            interval=lq_interval,
            conv_padding_mode=lq_conv_padding_mode,
            gate_per_token=lq_gate_per_token,
            pit_output=pit_lq_inject,
        )
        self.pit_lq_gate = SigmaAwareGate(self.hidden_size, per_token=lq_gate_per_token) if pit_lq_inject else None

    def _fetch_patch_pos(self, height, width, device, dtype, **rope_opts):
        return precompute_freqs_cis_2d(
            self.hidden_size // self.num_groups,
            height,
            width,
            ref_grid_h=self.rope_ref_grid_h,
            ref_grid_w=self.rope_ref_grid_w,
            device=device,
            dtype=dtype,
            **rope_opts,
        )

    def _pre_patch_block(self, s, i, pid_lq_features, pid_degrade_sigma, **kwargs):
        if not self.lq_proj.is_gate_active(i):
            return s
        out_idx = self.lq_proj.output_index(i)
        if out_idx >= len(pid_lq_features):
            return s
        return self.lq_proj.gate(s, pid_lq_features[out_idx], pid_degrade_sigma, out_idx)

    def _pre_pixel_blocks(self, s, pid_pit_lq_feature=None, pid_degrade_sigma=None, **kwargs):
        if pid_pit_lq_feature is None:
            return s
        return self.pit_lq_gate(s, pid_pit_lq_feature, pid_degrade_sigma)

    def forward(self, x, timesteps, context=None, attention_mask=None, transformer_options={}, **kwargs):
        lq_latent, degrade_sigma = dynamic_args.lq_latent
        assert lq_latent is not None

        expected_c = self.lq_proj.latent_channels
        assert lq_latent.shape[1] == expected_c

        B = x.shape[0]
        Hs = -(-x.shape[2] // self.patch_size)
        Ws = -(-x.shape[3] // self.patch_size)

        degrade_sigma = degrade_sigma.to(device=x.device).reshape(-1)
        if degrade_sigma.numel() == 1 and B > 1:
            degrade_sigma = degrade_sigma.expand(B).contiguous()

        lq_features = self.lq_proj(lq_latent=lq_latent.to(x), target_pH=Hs, target_pW=Ws)
        pit_lq_feature = lq_features.pop() if self.pit_lq_inject else None

        return super().forward(
            x,
            timesteps,
            context=context,
            attention_mask=attention_mask,
            transformer_options=transformer_options,
            pid_lq_features=lq_features,
            pid_pit_lq_feature=pit_lq_feature,
            pid_degrade_sigma=degrade_sigma,
            **kwargs,
        )
