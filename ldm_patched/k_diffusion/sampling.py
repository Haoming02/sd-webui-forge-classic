# Reference: https://github.com/comfyanonymous/ComfyUI

import math

import torch
import torchsde
from scipy import integrate
from torch import nn
from tqdm.auto import trange

from . import utils


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


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative"""
    return (x - denoised) / utils.append_dims(sigma, x.ndim)


def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    """
    Calculates the noise level (sigma_down) to step down to and the
    amount of noise to add (sigma_up) when doing an ancestral sampling step
    """
    if not eta:
        return sigma_to, 0.0
    sigma_up = min(eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5, sigma_to)
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


def default_noise_sampler(x, seed=None):
    return lambda sigma, sigma_next: torch.randn_like(x)


class BatchedBrownianTree:
    """A wrapper around torchsde.BrownianTree that enables batches of entropy"""

    def __init__(self, x, t0, t1, seed=None, **kwargs):
        self.cpu_tree = True
        if "cpu" in kwargs:
            self.cpu_tree = kwargs.pop("cpu")
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get("w0", torch.zeros_like(x))
        if seed is None:
            seed = torch.randint(0, 2**63 - 1, []).item()
        self.batched = True
        try:
            assert len(seed) == x.shape[0]
            w0 = w0[0]
        except TypeError:
            seed = [seed]
            self.batched = False
        if self.cpu_tree:
            self.trees = [torchsde.BrownianTree(t0.cpu(), w0.cpu(), t1.cpu(), entropy=s, **kwargs) for s in seed]
        else:
            self.trees = [torchsde.BrownianTree(t0, w0, t1, entropy=s, **kwargs) for s in seed]

    @staticmethod
    def sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0, t1, sign = self.sort(t0, t1)
        if self.cpu_tree:
            w = torch.stack([tree(t0.cpu().float(), t1.cpu().float()).to(t0.dtype).to(t0.device) for tree in self.trees]) * (self.sign * sign)
        else:
            w = torch.stack([tree(t0, t1) for tree in self.trees]) * (self.sign * sign)

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
        t0, t1 = (
            self.transform(torch.as_tensor(sigma_min)),
            self.transform(torch.as_tensor(sigma_max)),
        )
        self.tree = BatchedBrownianTree(x, t0, t1, seed, cpu=cpu)

    def __call__(self, sigma, sigma_next):
        t0, t1 = (
            self.transform(torch.as_tensor(sigma)),
            self.transform(torch.as_tensor(sigma_next)),
        )
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()


@torch.no_grad()
def sample_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)"""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        dt = sigmas[i + 1] - sigma_hat
        x = x + d * dt
    return x


@torch.no_grad()
def sample_euler_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None):
    """Ancestral sampling with Euler method steps"""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        d = to_d(x, sigmas[i], denoised)
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x


@torch.no_grad()
def sample_heun(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)"""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
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
            x = x + d * dt
        else:
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
        gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        if sigmas[i + 1] == 0:
            dt = sigmas[i + 1] - sigma_hat
            x = x + d * dt
        else:
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigmas[i + 1] - sigma_hat
            x_2 = x + d * dt_1
            denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
    return x


def _linear_multistep_coeff(order, t, i, j):
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
        cur_order = min(i + 1, order)
        coeffs = [_linear_multistep_coeff(cur_order, sigmas_cpu, i, j) for j in range(cur_order)]
        x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))
    return x


class PIDStepSizeController:
    """A PID controller for ODE adaptive step size control"""

    def __init__(self, h, pcoeff, icoeff, dcoeff, order=1, accept_safety=0.81, eps=1e-8):
        self.h = h
        self.b1 = (pcoeff + icoeff + dcoeff) / order
        self.b2 = -(pcoeff + 2 * dcoeff) / order
        self.b3 = dcoeff / order
        self.accept_safety = accept_safety
        self.eps = eps
        self.errs = []

    def limiter(self, x):
        return 1 + math.atan(x - 1)

    def propose_step(self, error):
        inv_error = 1 / (float(error) + self.eps)
        if not self.errs:
            self.errs = [inv_error, inv_error, inv_error]
        self.errs[0] = inv_error
        factor = self.errs[0] ** self.b1 * self.errs[1] ** self.b2 * self.errs[2] ** self.b3
        factor = self.limiter(factor)
        accept = factor >= self.accept_safety
        if accept:
            self.errs[2] = self.errs[1]
            self.errs[1] = self.errs[0]
        self.h *= factor
        return accept


class DPMSolver(nn.Module):
    """DPM-Solver. See https://arxiv.org/abs/2206.00927"""

    def __init__(self, model, extra_args=None, eps_callback=None, info_callback=None):
        super().__init__()
        self.model = model
        self.extra_args = {} if extra_args is None else extra_args
        self.eps_callback = eps_callback
        self.info_callback = info_callback

    def t(self, sigma):
        return -sigma.log()

    def sigma(self, t):
        return t.neg().exp()

    def eps(self, eps_cache, key, x, t, *args, **kwargs):
        if key in eps_cache:
            return eps_cache[key], eps_cache
        sigma = self.sigma(t) * x.new_ones([x.shape[0]])
        eps = (x - self.model(x, sigma, *args, **self.extra_args, **kwargs)) / self.sigma(t)
        if self.eps_callback is not None:
            self.eps_callback()
        return eps, {key: eps, **eps_cache}

    def dpm_solver_1_step(self, x, t, t_next, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, "eps", x, t)
        x_1 = x - self.sigma(t_next) * h.expm1() * eps
        return x_1, eps_cache

    def dpm_solver_2_step(self, x, t, t_next, r1=1 / 2, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, "eps", x, t)
        s1 = t + r1 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, "eps_r1", u1, s1)
        x_2 = x - self.sigma(t_next) * h.expm1() * eps - self.sigma(t_next) / (2 * r1) * h.expm1() * (eps_r1 - eps)
        return x_2, eps_cache

    def dpm_solver_3_step(self, x, t, t_next, r1=1 / 3, r2=2 / 3, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, "eps", x, t)
        s1 = t + r1 * h
        s2 = t + r2 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, "eps_r1", u1, s1)
        u2 = x - self.sigma(s2) * (r2 * h).expm1() * eps - self.sigma(s2) * (r2 / r1) * ((r2 * h).expm1() / (r2 * h) - 1) * (eps_r1 - eps)
        eps_r2, eps_cache = self.eps(eps_cache, "eps_r2", u2, s2)
        x_3 = x - self.sigma(t_next) * h.expm1() * eps - self.sigma(t_next) / r2 * (h.expm1() / h - 1) * (eps_r2 - eps)
        return x_3, eps_cache


@torch.no_grad()
def sample_dpmpp_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None, r=0.5):
    """DPM-Solver++ (stochastic)"""
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None)
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigmas[i + 1] == 0:
            d = to_d(x, sigmas[i], denoised)
            dt = sigmas[i + 1] - sigmas[i]
            x = x + d * dt
        else:
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            s = t + h * r
            fac = 1 / (2 * r)

            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(s), eta)
            s_ = t_fn(sd)
            x_2 = (sigma_fn(s_) / sigma_fn(t)) * x - (t - s_).expm1() * denoised
            x_2 = x_2 + noise_sampler(sigma_fn(t), sigma_fn(s)) * s_noise * su
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)

            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(t_next), eta)
            t_next_ = t_fn(sd)
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            x = (sigma_fn(t_next_) / sigma_fn(t)) * x - (t - t_next_).expm1() * denoised_d
            x = x + noise_sampler(sigma_fn(t), sigma_fn(t_next)) * s_noise * su
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
def sample_dpmpp_3m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None):
    """DPM-Solver++(3M) SDE"""

    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    denoised_1, denoised_2 = None, None
    h, h_1, h_2 = None, None, None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            h_eta = h * (eta + 1)

            x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised

            if h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + phi_2 * d1 - phi_3 * d2
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + phi_2 * d

            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
    return x


@torch.no_grad()
def sample_dpmpp_3m_sde_gpu(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None):
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=extra_args.get("seed", None), cpu=False) if noise_sampler is None else noise_sampler
    return sample_dpmpp_3m_sde(
        model,
        x,
        sigmas,
        extra_args=extra_args,
        callback=callback,
        disable=disable,
        eta=eta,
        s_noise=s_noise,
        noise_sampler=noise_sampler,
    )


@torch.no_grad()
def sample_dpmpp_sde_gpu(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None, r=0.5):
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=extra_args.get("seed", None), cpu=False) if noise_sampler is None else noise_sampler
    return sample_dpmpp_sde(
        model,
        x,
        sigmas,
        extra_args=extra_args,
        callback=callback,
        disable=disable,
        eta=eta,
        s_noise=s_noise,
        noise_sampler=noise_sampler,
        r=r,
    )


def DDPMSampler_step(x, sigma, sigma_prev, noise, noise_sampler):
    alpha_cumprod = 1 / ((sigma * sigma) + 1)
    alpha_cumprod_prev = 1 / ((sigma_prev * sigma_prev) + 1)
    alpha = alpha_cumprod / alpha_cumprod_prev

    mu = (1.0 / alpha).sqrt() * (x - (1 - alpha) * noise / (1 - alpha_cumprod).sqrt())
    if sigma_prev > 0:
        mu += ((1 - alpha) * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)).sqrt() * noise_sampler(sigma, sigma_prev)
    return mu


def generic_step_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, step_function=None):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        x = step_function(
            x / torch.sqrt(1.0 + sigmas[i] ** 2.0),
            sigmas[i],
            sigmas[i + 1],
            (x - denoised) / sigmas[i],
            noise_sampler,
        )
        if sigmas[i + 1] != 0:
            x *= torch.sqrt(1.0 + sigmas[i + 1] ** 2.0)
    return x


@torch.no_grad()
def sample_ddpm(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    return generic_step_sampler(model, x, sigmas, extra_args, callback, disable, noise_sampler, DDPMSampler_step)
