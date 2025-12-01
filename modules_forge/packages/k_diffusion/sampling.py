# https://github.com/comfyanonymous/ComfyUI/blob/v0.3.75/comfy/k_diffusion/sampling.py

import math
from functools import partial

import torch
import torchsde
from scipy import integrate
from tqdm.auto import trange

from backend.patcher.base import set_model_options_post_cfg_function

from . import utils


def _is_const(sampling) -> bool:
    return sampling.prediction_type == "const"


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)"""
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def get_sigmas_exponential(n, sigma_min, sigma_max, device="cpu"):
    """Constructs an exponential noise schedule"""
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
    return append_zero(sigmas)


def get_sigmas_polyexponential(n, sigma_min, sigma_max, rho=1.0, device="cpu"):
    """Constructs an polynomial in log sigma noise schedule"""
    ramp = torch.linspace(1, 0, n, device=device) ** rho
    sigmas = torch.exp(ramp * (math.log(sigma_max) - math.log(sigma_min)) + math.log(sigma_min))
    return append_zero(sigmas)


def get_sigmas_vp(n, beta_d=19.9, beta_min=0.1, eps_s=1e-3, device="cpu"):
    """Constructs a continuous VP noise schedule"""
    t = torch.linspace(1, eps_s, n, device=device)
    sigmas = torch.sqrt(torch.special.expm1(beta_d * t**2 / 2 + beta_min * t))
    return append_zero(sigmas)


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative"""
    return (x - denoised) / utils.append_dims(sigma, x.ndim)


def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step"""
    if not eta:
        return sigma_to, 0.0
    sigma_up = min(sigma_to, eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5)
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)


class BatchedBrownianTree:
    """A wrapper around torchsde.BrownianTree that enables batches of entropy"""

    def __init__(self, x, t0, t1, seed=None, **kwargs):
        self.cpu_tree = kwargs.pop("cpu", True)
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.pop("w0", None)
        if w0 is None:
            w0 = torch.zeros_like(x)
        self.batched = False
        if seed is None:
            seed = (torch.randint(0, 2**63 - 1, ()).item(),)
        elif isinstance(seed, (tuple, list)):
            if len(seed) != x.shape[0]:
                raise ValueError("Passing a list or tuple of seeds to BatchedBrownianTree requires a length matching the batch size.")
            self.batched = True
            w0 = w0[0]
        else:
            seed = (seed,)
        if self.cpu_tree:
            t0, w0, t1 = t0.detach().cpu(), w0.detach().cpu(), t1.detach().cpu()
        self.trees = tuple(torchsde.BrownianTree(t0, w0, t1, entropy=s, **kwargs) for s in seed)

    @staticmethod
    def sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0, t1, sign = self.sort(t0, t1)
        device, dtype = t0.device, t0.dtype
        if self.cpu_tree:
            t0, t1 = t0.detach().cpu().float(), t1.detach().cpu().float()
        w = torch.stack([tree(t0, t1) for tree in self.trees]).to(device=device, dtype=dtype) * (self.sign * sign)
        return w if self.batched else w[0]


class BrownianTreeNoiseSampler:
    """A noise sampler backed by a torchsde.BrownianTree.

    Args:
        x (Tensor): The tensor whose shape, device and dtype to use to generate
            random samples.
        sigma_min (float): The low end of the valid interval.
        sigma_max (float): The high end of the valid interval.
        seed (int or List[int]): The random seed. If a list of seeds is
            supplied instead of a single integer, then the noise sampler will
            use one BrownianTree per batch item, each with its own seed.
        transform (callable): A function that maps sigma to the sampler's
            internal timestep.
    """

    def __init__(self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x, cpu=False):
        self.transform = transform
        t0, t1 = self.transform(torch.as_tensor(sigma_min)), self.transform(torch.as_tensor(sigma_max))
        self.tree = BatchedBrownianTree(x, t0, t1, seed, cpu=cpu)

    def __call__(self, sigma, sigma_next):
        t0, t1 = self.transform(torch.as_tensor(sigma)), self.transform(torch.as_tensor(sigma_next))
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()


def sigma_to_half_log_snr(sigma, model_sampling):
    """Convert sigma to half-logSNR log(alpha_t / sigma_t)"""
    if _is_const(model_sampling):
        # log((1 - t) / t) = log((1 - sigma) / sigma)
        return sigma.logit().neg()
    return sigma.log().neg()


def half_log_snr_to_sigma(half_log_snr, model_sampling):
    """Convert half-logSNR log(alpha_t / sigma_t) to sigma"""
    if _is_const(model_sampling):
        # 1 / (1 + exp(half_log_snr))
        return half_log_snr.neg().sigmoid()
    return half_log_snr.neg().exp()


def offset_first_sigma_for_snr(sigmas, model_sampling, percent_offset=1e-4):
    """Adjust the first sigma to avoid invalid logSNR"""
    if len(sigmas) <= 1:
        return sigmas
    if _is_const(model_sampling):
        if sigmas[0] >= 1:
            sigmas = sigmas.clone()
            sigmas[0] = model_sampling.percent_to_sigma(percent_offset)
    return sigmas


@torch.no_grad()
def sample_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)"""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        if s_churn > 0:
            gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
            sigma_hat = sigmas[i] * (gamma + 1)
        else:
            gamma = 0
            sigma_hat = sigmas[i]

        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
    return x


@torch.no_grad()
def sample_euler_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None):
    if _is_const(model.inner_model.predictor):
        return sample_euler_ancestral_RF(model, x, sigmas, extra_args, callback, disable, eta, s_noise, noise_sampler)
    """Ancestral sampling with Euler method steps"""
    extra_args = {} if extra_args is None else extra_args

    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})

        if sigma_down == 0:
            x = denoised
        else:
            d = to_d(x, sigmas[i], denoised)
            # Euler method
            dt = sigma_down - sigmas[i]
            x = x + d * dt + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x


@torch.no_grad()
def sample_euler_ancestral_RF(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None):
    """Ancestral sampling with Euler method steps"""
    extra_args = {} if extra_args is None else extra_args

    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        # sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})

        if sigmas[i + 1] == 0:
            x = denoised
        else:
            downstep_ratio = 1 + (sigmas[i + 1] / sigmas[i] - 1) * eta
            sigma_down = sigmas[i + 1] * downstep_ratio
            alpha_ip1 = 1 - sigmas[i + 1]
            alpha_down = 1 - sigma_down
            renoise_coeff = (sigmas[i + 1] ** 2 - sigma_down**2 * alpha_ip1**2 / alpha_down**2) ** 0.5
            # Euler method
            sigma_down_i_ratio = sigma_down / sigmas[i]
            x = sigma_down_i_ratio * x + (1 - sigma_down_i_ratio) * denoised
            if eta > 0:
                x = (alpha_ip1 / alpha_down) * x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * renoise_coeff
    return x


@torch.no_grad()
def sample_heun(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)"""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        if s_churn > 0:
            gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
            sigma_hat = sigmas[i] * (gamma + 1)
        else:
            gamma = 0
            sigma_hat = sigmas[i]

        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    return x


@torch.no_grad()
def sample_dpm_2(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0):
    """A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022)"""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        if s_churn > 0:
            gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
            sigma_hat = sigmas[i] * (gamma + 1)
        else:
            gamma = 0
            sigma_hat = sigmas[i]

        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        if sigmas[i + 1] == 0:
            # Euler method
            dt = sigmas[i + 1] - sigma_hat
            x = x + d * dt
        else:
            # DPM-Solver-2
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigmas[i + 1] - sigma_hat
            x_2 = x + d * dt_1
            denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
    return x


def linear_multistep_coeff(order, t, i, j):
    if order - 1 > i:
        raise ValueError(f"Order {order} too high for step {i}")

    def fn(tau):
        prod = 1.0
        for k in range(order):
            if j == k:
                continue
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod

    return integrate.quad(fn, t[i], t[i + 1], epsrel=1e-4)[0]


@torch.no_grad()
def sample_lms(model, x, sigmas, extra_args=None, callback=None, disable=None, order=4):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigmas_cpu = sigmas.detach().cpu().numpy()
    ds = []
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = to_d(x, sigmas[i], denoised)
        ds.append(d)
        if len(ds) > order:
            ds.pop(0)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            cur_order = min(i + 1, order)
            coeffs = [linear_multistep_coeff(cur_order, sigmas_cpu, i, j) for j in range(cur_order)]
            x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))
    return x


@torch.no_grad()
def sample_dpmpp_2s_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None):
    if _is_const(model.inner_model.predictor):
        return sample_dpmpp_2s_ancestral_RF(model, x, sigmas, extra_args, callback, disable, eta, s_noise, noise_sampler)

    """Ancestral sampling with DPM-Solver++(2S) second-order steps"""
    extra_args = {} if extra_args is None else extra_args

    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigma_down == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++(2S)
            t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * denoised
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_2
        # Noise addition
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x


@torch.no_grad()
def sample_dpmpp_2s_ancestral_RF(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None):
    """Ancestral sampling with DPM-Solver++(2S) second-order steps"""
    extra_args = {} if extra_args is None else extra_args

    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda lbda: (lbda.exp() + 1) ** -1
    lambda_fn = lambda sigma: ((1 - sigma) / sigma).log()

    # logged_x = x.unsqueeze(0)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        downstep_ratio = 1 + (sigmas[i + 1] / sigmas[i] - 1) * eta
        sigma_down = sigmas[i + 1] * downstep_ratio
        alpha_ip1 = 1 - sigmas[i + 1]
        alpha_down = 1 - sigma_down
        renoise_coeff = (sigmas[i + 1] ** 2 - sigma_down**2 * alpha_ip1**2 / alpha_down**2) ** 0.5
        # sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigmas[i + 1] == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++(2S)
            if sigmas[i] == 1.0:
                sigma_s = 0.9999
            else:
                t_i, t_down = lambda_fn(sigmas[i]), lambda_fn(sigma_down)
                r = 1 / 2
                h = t_down - t_i
                s = t_i + r * h
                sigma_s = sigma_fn(s)
            # sigma_s = sigmas[i+1]
            sigma_s_i_ratio = sigma_s / sigmas[i]
            u = sigma_s_i_ratio * x + (1 - sigma_s_i_ratio) * denoised
            D_i = model(u, sigma_s * s_in, **extra_args)
            sigma_down_i_ratio = sigma_down / sigmas[i]
            x = sigma_down_i_ratio * x + (1 - sigma_down_i_ratio) * D_i
            # print("sigma_i", sigmas[i], "sigma_ip1", sigmas[i+1],"sigma_down", sigma_down, "sigma_down_i_ratio", sigma_down_i_ratio, "sigma_s_i_ratio", sigma_s_i_ratio, "renoise_coeff", renoise_coeff)
        # Noise addition
        if sigmas[i + 1] > 0 and eta > 0:
            x = (alpha_ip1 / alpha_down) * x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * renoise_coeff
        # logged_x = torch.cat((logged_x, x.unsqueeze(0)), dim=0)
    return x


@torch.no_grad()
def sample_dpmpp_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None, r=1 / 2):
    """DPM-Solver++ (stochastic)"""
    if len(sigmas) <= 1:
        return x

    extra_args = {} if extra_args is None else extra_args
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None)
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    model_sampling = model.inner_model.predictor
    sigma_fn = partial(half_log_snr_to_sigma, model_sampling=model_sampling)
    lambda_fn = partial(sigma_to_half_log_snr, model_sampling=model_sampling)
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # DPM-Solver++
            lambda_s, lambda_t = lambda_fn(sigmas[i]), lambda_fn(sigmas[i + 1])
            h = lambda_t - lambda_s
            lambda_s_1 = lambda_s + r * h
            fac = 1 / (2 * r)

            sigma_s_1 = sigma_fn(lambda_s_1)

            alpha_s = sigmas[i] * lambda_s.exp()
            alpha_s_1 = sigma_s_1 * lambda_s_1.exp()
            alpha_t = sigmas[i + 1] * lambda_t.exp()

            # Step 1
            sd, su = get_ancestral_step(lambda_s.neg().exp(), lambda_s_1.neg().exp(), eta)
            lambda_s_1_ = sd.log().neg()
            h_ = lambda_s_1_ - lambda_s
            x_2 = (alpha_s_1 / alpha_s) * (-h_).exp() * x - alpha_s_1 * (-h_).expm1() * denoised
            if eta > 0 and s_noise > 0:
                x_2 = x_2 + alpha_s_1 * noise_sampler(sigmas[i], sigma_s_1) * s_noise * su
            denoised_2 = model(x_2, sigma_s_1 * s_in, **extra_args)

            # Step 2
            sd, su = get_ancestral_step(lambda_s.neg().exp(), lambda_t.neg().exp(), eta)
            lambda_t_ = sd.log().neg()
            h_ = lambda_t_ - lambda_s
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            x = (alpha_t / alpha_s) * (-h_).exp() * x - alpha_t * (-h_).expm1() * denoised_d
            if eta > 0 and s_noise > 0:
                x = x + alpha_t * noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * su
    return x


@torch.no_grad()
def sample_dpmpp_2m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """DPM-Solver++(2M)"""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x


@torch.no_grad()
def sample_dpmpp_2m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None, solver_type="midpoint"):
    """DPM-Solver++(2M) SDE"""
    if len(sigmas) <= 1:
        return x

    if solver_type not in {"heun", "midpoint"}:
        raise ValueError("solver_type must be 'heun' or 'midpoint'")

    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    model_sampling = model.inner_model.predictor
    lambda_fn = partial(sigma_to_half_log_snr, model_sampling=model_sampling)
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)

    old_denoised = None
    h, h_last = None, None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # DPM-Solver++(2M) SDE
            lambda_s, lambda_t = lambda_fn(sigmas[i]), lambda_fn(sigmas[i + 1])
            h = lambda_t - lambda_s
            h_eta = h * (eta + 1)

            alpha_t = sigmas[i + 1] * lambda_t.exp()

            x = sigmas[i + 1] / sigmas[i] * (-h * eta).exp() * x + alpha_t * (-h_eta).expm1().neg() * denoised

            if old_denoised is not None:
                r = h_last / h
                if solver_type == "heun":
                    x = x + alpha_t * ((-h_eta).expm1().neg() / (-h_eta) + 1) * (1 / r) * (denoised - old_denoised)
                elif solver_type == "midpoint":
                    x = x + 0.5 * alpha_t * (-h_eta).expm1().neg() * (1 / r) * (denoised - old_denoised)

            if eta > 0 and s_noise > 0:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        old_denoised = denoised
        h_last = h
    return x


@torch.no_grad()
def sample_dpmpp_3m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None):
    """DPM-Solver++(3M) SDE"""

    if len(sigmas) <= 1:
        return x

    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    model_sampling = model.inner_model.predictor
    lambda_fn = partial(sigma_to_half_log_snr, model_sampling=model_sampling)
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)

    denoised_1, denoised_2 = None, None
    h, h_1, h_2 = None, None, None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            lambda_s, lambda_t = lambda_fn(sigmas[i]), lambda_fn(sigmas[i + 1])
            h = lambda_t - lambda_s
            h_eta = h * (eta + 1)

            alpha_t = sigmas[i + 1] * lambda_t.exp()

            x = sigmas[i + 1] / sigmas[i] * (-h * eta).exp() * x + alpha_t * (-h_eta).expm1().neg() * denoised

            if h_2 is not None:
                # DPM-Solver++(3M) SDE
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + (alpha_t * phi_2) * d1 - (alpha_t * phi_3) * d2
            elif h_1 is not None:
                # DPM-Solver++(2M) SDE
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + (alpha_t * phi_2) * d

            if eta > 0 and s_noise > 0:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
    return x


@torch.no_grad()
def sample_lcm(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    extra_args = {} if extra_args is None else extra_args

    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})

        x = denoised
        if sigmas[i + 1] > 0:
            x = model.inner_model.predictor.noise_scaling(sigmas[i + 1], noise_sampler(sigmas[i], sigmas[i + 1]), x)
    return x


@torch.no_grad()
def sample_euler_ancestral_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None):
    """Ancestral sampling with Euler method steps (CFG++)"""
    extra_args = {} if extra_args is None else extra_args

    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler

    model_sampling = model.inner_model.predictor
    lambda_fn = partial(sigma_to_half_log_snr, model_sampling=model_sampling)

    uncond_denoised = None

    def post_cfg_function(args):
        nonlocal uncond_denoised
        uncond_denoised = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            alpha_s = sigmas[i] * lambda_fn(sigmas[i]).exp()
            alpha_t = sigmas[i + 1] * lambda_fn(sigmas[i + 1]).exp()
            d = to_d(x, sigmas[i], alpha_s * uncond_denoised)  # to noise

            # DDIM stochastic sampling
            sigma_down, sigma_up = get_ancestral_step(sigmas[i] / alpha_s, sigmas[i + 1] / alpha_t, eta=eta)
            sigma_down = alpha_t * sigma_down

            # Euler method
            x = alpha_t * denoised + sigma_down * d
            if eta > 0 and s_noise > 0:
                x = x + alpha_t * noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x


@torch.no_grad()
def sample_euler_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """Euler method steps (CFG++)"""
    return sample_euler_ancestral_cfg_pp(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta=0.0, s_noise=0.0, noise_sampler=None)


@torch.no_grad()
def sample_dpmpp_2m_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """DPM-Solver++(2M)"""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    t_fn = lambda sigma: sigma.log().neg()

    old_uncond_denoised = None
    uncond_denoised = None

    def post_cfg_function(args):
        nonlocal uncond_denoised
        uncond_denoised = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_uncond_denoised is None or sigmas[i + 1] == 0:
            denoised_mix = -torch.exp(-h) * uncond_denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_mix = -torch.exp(-h) * uncond_denoised - torch.expm1(-h) * (1 / (2 * r)) * (denoised - old_uncond_denoised)
        x = denoised + denoised_mix + torch.exp(-h) * x
        old_uncond_denoised = uncond_denoised
    return x


@torch.no_grad()
def res_multistep(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1.0, noise_sampler=None, eta=1.0, cfg_pp=False):
    extra_args = {} if extra_args is None else extra_args

    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    phi1_fn = lambda t: torch.expm1(t) / t
    phi2_fn = lambda t: (phi1_fn(t) - 1.0) / t

    old_sigma_down = None
    old_denoised = None
    uncond_denoised = None

    def post_cfg_function(args):
        nonlocal uncond_denoised
        uncond_denoised = args["uncond_denoised"]
        return args["denoised"]

    if cfg_pp:
        model_options = extra_args.get("model_options", {}).copy()
        extra_args["model_options"] = set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigma_down == 0 or old_denoised is None:
            # Euler method
            if cfg_pp:
                d = to_d(x, sigmas[i], uncond_denoised)
                x = denoised + d * sigma_down
            else:
                d = to_d(x, sigmas[i], denoised)
                dt = sigma_down - sigmas[i]
                x = x + d * dt
        else:
            # Second order multistep method in https://arxiv.org/pdf/2308.02157
            t, t_old, t_next, t_prev = t_fn(sigmas[i]), t_fn(old_sigma_down), t_fn(sigma_down), t_fn(sigmas[i - 1])
            h = t_next - t
            c2 = (t_prev - t_old) / h

            phi1_val, phi2_val = phi1_fn(-h), phi2_fn(-h)
            b1 = torch.nan_to_num(phi1_val - phi2_val / c2, nan=0.0)
            b2 = torch.nan_to_num(phi2_val / c2, nan=0.0)

            if cfg_pp:
                x = x + (denoised - uncond_denoised)
                x = sigma_fn(h) * x + h * (b1 * uncond_denoised + b2 * old_denoised)
            else:
                x = sigma_fn(h) * x + h * (b1 * denoised + b2 * old_denoised)

        # Noise addition
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up

        if cfg_pp:
            old_denoised = uncond_denoised
        else:
            old_denoised = denoised
        old_sigma_down = sigma_down
    return x


@torch.no_grad()
def sample_res_multistep(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1.0, noise_sampler=None):
    return res_multistep(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, s_noise=s_noise, noise_sampler=noise_sampler, eta=0.0, cfg_pp=False)


@torch.no_grad()
def sample_Kohaku_LoNyu_Yog(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=None, s_tmin=None, s_tmax=float("inf"), s_noise=None, noise_sampler=None, eta=None):
    s_churn = 0.0 if s_churn is None else s_churn
    s_tmin = 0.0 if s_tmin is None else s_tmin
    s_noise = 1.0 if s_noise is None else s_noise
    eta = 1.0 if eta is None else eta

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        dt = sigma_down - sigmas[i]
        if i <= (len(sigmas) - 1) / 2:
            x2 = -x
            denoised2 = model(x2, sigma_hat * s_in, **extra_args)
            d2 = to_d(x2, sigma_hat, denoised2)
            x3 = x + ((d + d2) / 2) * dt
            denoised3 = model(x3, sigma_hat * s_in, **extra_args)
            d3 = to_d(x3, sigma_hat, denoised3)
            real_d = (d + d3) / 2
            x = x + real_d * dt
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
        else:
            x = x + d * dt
    return x
