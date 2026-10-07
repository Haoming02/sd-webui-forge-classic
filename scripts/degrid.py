import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from modules import scripts_postprocessing
from modules.ui_components import InputAccordion


class ScriptPostprocessingDegrid(scripts_postprocessing.ScriptPostprocessing):
    name = "DeGrid"
    order = 900

    def ui(self):
        with InputAccordion(True, label="DeGrid", elem_id="extras_degrid") as degrid_enabled:
            degrid_threshold = gr.Slider(value=0.0, minimum=0.0, maximum=1.0, step=0.05, label="Threshold", info="0.0 for Auto", elem_id="extras_degrid_threshold")

        return {
            "degrid_enabled": degrid_enabled,
            "degrid_threshold": degrid_threshold,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, degrid_enabled: bool = True, degrid_threshold: float = 0.0):
        if not degrid_enabled:
            return

        image = np.asarray(pp.image, dtype=np.float32)
        image = torch.from_numpy(np.clip(image / 255.0, 0.0, 1.0))

        image = degrid(image.cuda(), degrid_threshold)

        image = image.detach().clone().cpu().numpy()
        pp.image = Image.fromarray(np.clip((image * 255.0).round(), 0, 255).astype(np.uint8))


# https://github.com/lunaaispace-eng/ComfyUI-DeGrid


_KERNEL = [1.0, -8.0, 28.0, -56.0, 70.0, -56.0, 28.0, -8.0, 1.0]
_NORM = 256.0
_PAD = 4

NEGLIGIBLE_AMP = 0.5 / 255.0


def extract_grid(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    if h <= 2 * _PAD or w <= 2 * _PAD:
        return torch.zeros_like(x)

    k = torch.tensor(_KERNEL, dtype=x.dtype, device=x.device) / _NORM
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    bx = F.conv2d(F.pad(x, (_PAD, _PAD, 0, 0), mode="reflect"), kx, groups=c)
    by = F.conv2d(F.pad(x, (0, 0, _PAD, _PAD), mode="reflect"), ky, groups=c)

    bxy = F.conv2d(F.pad(bx, (0, 0, _PAD, _PAD), mode="reflect"), ky, groups=c)
    return bx + by - bxy


def lattice_amp(corr: torch.Tensor) -> torch.Tensor:
    h = corr.shape[2] // 2 * 2
    w = corr.shape[3] // 2 * 2
    if h < 2 or w < 2:
        z = torch.zeros(corr.shape[0], dtype=corr.dtype, device=corr.device)
        return z, z, z, z
    c = corr[:, :, :h, :w]
    m = torch.stack(
        [c[:, :, i::2, j::2].mean(dim=(2, 3)) for i in (0, 1) for j in (0, 1)],
        dim=-1,
    )
    p2p = (m.amax(-1) - m.amin(-1)).amax(-1)
    return p2p


def _subsample(flat: torch.Tensor, max_samples: int = 1_000_000) -> torch.Tensor:
    n = flat.shape[-1]
    if n > max_samples:
        return flat[..., :: n // max_samples + 1]
    return flat


def auto_limit(corr: torch.Tensor, floor: float = 0.004, ceil: float = 0.05, mult: float = 3.0) -> torch.Tensor:
    flat = _subsample(corr.abs().reshape(corr.shape[0], -1))
    q = torch.quantile(flat.float(), 0.75, dim=1)
    return (q * mult).clamp(floor, ceil).to(corr.dtype)


def degrid(image: torch.Tensor, threshold: float) -> torch.Tensor:

    x = image.permute(2, 0, 1).unsqueeze(0).contiguous()

    corr = extract_grid(x)

    amp = lattice_amp(corr)

    if threshold < 0.05:
        lim = auto_limit(corr)
    else:
        lim = torch.full((x.shape[0],), threshold / 10.0, dtype=corr.dtype, device=corr.device)

    lim_b = lim.view(-1, 1, 1, 1)
    clipped = (corr.abs() > lim_b).float().mean(dim=(1, 2, 3)) * 100.0
    corr = corr.clamp(-lim_b, lim_b)

    skipped = amp < NEGLIGIBLE_AMP
    corr = torch.where(skipped.view(-1, 1, 1, 1), torch.zeros_like(corr), corr)
    clipped = torch.where(skipped, torch.zeros_like(clipped), clipped)

    cleaned = (x - corr).clamp(0.0, 1.0).permute(0, 2, 3, 1)

    return cleaned.squeeze(0)
