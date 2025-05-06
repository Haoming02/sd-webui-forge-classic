import dataclasses
from typing import Callable

import k_diffusion
import numpy as np
import torch
from modules import shared
from scipy import stats


def to_d(x: torch.Tensor, sigma: float, denoised: torch.Tensor):
    """Converts a denoiser output to a Karras ODE derivative"""
    return (x - denoised) / sigma


k_diffusion.sampling.to_d = to_d


@dataclasses.dataclass
class Scheduler:
    name: str
    label: str
    function: Callable

    default_rho: float = -1.0
    need_inner_model: bool = False
    aliases: list[str] = None


def normal_scheduler(n, sigma_min, sigma_max, inner_model, device, sgm=False, floor=False):
    start = inner_model.sigma_to_t(torch.tensor(sigma_max))
    end = inner_model.sigma_to_t(torch.tensor(sigma_min))

    if sgm:
        timesteps = torch.linspace(start, end, n + 1)[:-1]
    else:
        timesteps = torch.linspace(start, end, n)

    sigs = []
    for x in range(len(timesteps)):
        ts = timesteps[x]
        sigs.append(inner_model.t_to_sigma(ts))
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


def simple_scheduler(n, sigma_min, sigma_max, inner_model, device):
    sigs = []
    ss = len(inner_model.sigmas) / n
    for x in range(n):
        sigs += [float(inner_model.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


def uniform(n, sigma_min, sigma_max, inner_model, device):
    return inner_model.get_sigmas(n).to(device)


def sgm_uniform(n, sigma_min, sigma_max, inner_model, device):
    start = inner_model.sigma_to_t(torch.tensor(sigma_max))
    end = inner_model.sigma_to_t(torch.tensor(sigma_min))
    sigs = [inner_model.t_to_sigma(ts) for ts in torch.linspace(start, end, n + 1)[:-1]]
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


def _loglinear_interp(t_steps, num_steps):
    """Performs log-linear interpolation of a given array of decreasing numbers"""
    xs = np.linspace(0, 1, len(t_steps))
    ys = np.log(t_steps[::-1])

    new_xs = np.linspace(0, 1, num_steps)
    new_ys = np.interp(new_xs, xs, ys)

    interped_ys = np.exp(new_ys)[::-1].copy()
    return interped_ys


def get_align_your_steps_sigmas(n, sigma_min, sigma_max, device):
    """https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/howto.html"""

    if shared.sd_model.is_sdxl:
        sigmas = [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.029]
    else:
        sigmas = [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029]

    if n != len(sigmas):
        sigmas = np.append(_loglinear_interp(sigmas, n), [0.0])
    else:
        sigmas.append(0.0)

    return torch.FloatTensor(sigmas).to(device)


def get_align_your_steps_sigmas_GITS(n, sigma_min, sigma_max, device):
    if shared.sd_model.is_sdxl:
        sigmas = [14.615, 4.734, 2.567, 1.529, 0.987, 0.652, 0.418, 0.268, 0.179, 0.127, 0.029]
    else:
        sigmas = [14.615, 4.617, 2.507, 1.236, 0.702, 0.402, 0.240, 0.156, 0.104, 0.094, 0.029]

    if n != len(sigmas):
        sigmas = np.append(_loglinear_interp(sigmas, n), [0.0])
    else:
        sigmas.append(0.0)

    return torch.FloatTensor(sigmas).to(device)


def kl_optimal(n, sigma_min, sigma_max, device):
    alpha_min = torch.arctan(torch.tensor(sigma_min, device=device))
    alpha_max = torch.arctan(torch.tensor(sigma_max, device=device))
    step_indices = torch.arange(n + 1, device=device)
    sigmas = torch.tan(step_indices / n * alpha_min + (1.0 - step_indices / n) * alpha_max)
    return sigmas


def ddim_scheduler(n, sigma_min, sigma_max, inner_model, device):
    sigs = []
    ss = max(len(inner_model.sigmas) // n, 1)
    x = 1
    while x < len(inner_model.sigmas):
        sigs += [float(inner_model.sigmas[x])]
        x += ss
    sigs = sigs[::-1]
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


def beta_scheduler(n, sigma_min, sigma_max, inner_model, device):
    """
    Beta scheduler
    Based on "Beta Sampling is All You Need" [arXiv:2407.12173] (Lee et. al, 2024)
    """
    alpha = shared.opts.beta_dist_alpha
    beta = shared.opts.beta_dist_beta

    total_timesteps = len(inner_model.sigmas) - 1
    ts = 1 - np.linspace(0, 1, n, endpoint=False)
    ts = np.rint(stats.beta.ppf(ts, alpha, beta) * total_timesteps)

    sigs = []
    last_t = -1
    for t in ts:
        if t != last_t:
            sigs += [float(inner_model.sigmas[int(t)])]
        last_t = t
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


def turbo_scheduler(n, sigma_min, sigma_max, inner_model, device):
    unet = inner_model.inner_model.forge_objects.unet
    timesteps = torch.flip(torch.arange(1, n + 1) * float(1000.0 / n) - 1, (0,)).round().long().clip(0, 999)
    sigmas = unet.model.model_sampling.sigma(timesteps)
    sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
    return sigmas.to(device)


schedulers = [
    Scheduler("automatic", "Automatic", None),
    Scheduler("karras", "Karras", k_diffusion.sampling.get_sigmas_karras, default_rho=7.0),
    Scheduler("exponential", "Exponential", k_diffusion.sampling.get_sigmas_exponential),
    Scheduler("polyexponential", "Polyexponential", k_diffusion.sampling.get_sigmas_polyexponential, default_rho=1.0),
    Scheduler("normal", "Normal", normal_scheduler, need_inner_model=True),
    Scheduler("simple", "Simple", simple_scheduler, need_inner_model=True),
    Scheduler("uniform", "Uniform", uniform, need_inner_model=True),
    Scheduler("sgm_uniform", "SGM Uniform", sgm_uniform, need_inner_model=True, aliases=["SGMUniform"]),
    Scheduler("kl_optimal", "KL Optimal", kl_optimal),
    Scheduler("ddim", "DDIM", ddim_scheduler, need_inner_model=True),
    Scheduler("align_your_steps", "Align Your Steps", get_align_your_steps_sigmas),
    Scheduler("align_your_steps_GITS", "Align Your Steps GITS", get_align_your_steps_sigmas_GITS),
    Scheduler("beta", "Beta", beta_scheduler, need_inner_model=True),
    Scheduler("turbo", "Turbo", turbo_scheduler, need_inner_model=True),
]

schedulers_map = {**{x.name: x for x in schedulers}, **{x.label: x for x in schedulers}}
