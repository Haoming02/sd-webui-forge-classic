# https://github.com/Comfy-Org/ComfyUI/blob/v0.27.1/comfy/model_sampling.py

import math

import numpy as np
import torch


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = torch.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64) ** 2
    elif schedule == "cosine":
        timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = torch.clamp(betas, min=0, max=0.999)
    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas


def flux_time_shift(mu: float, sigma: float, t):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def time_snr_shift(alpha, t):
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


def rescale_zero_terminal_snr_sigmas(sigmas):
    alphas_cumprod = 1 / ((sigmas * sigmas) + 1)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas_bar[-1] = 4.8973451890853435e-08
    return ((1 - alphas_bar) / alphas_bar) ** 0.5


class AbstractPrediction(torch.nn.Module):
    def __init__(self, sigma_data=1.0, prediction_type="epsilon"):
        super().__init__()

        self.sigma_data = sigma_data
        self.prediction_type = prediction_type
        assert self.prediction_type in ("epsilon", "v_prediction", "const")

    def calculate_input(self, sigma, noise):
        if self.prediction_type == "const":
            return noise
        else:
            sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
            return noise / (sigma**2 + self.sigma_data**2) ** 0.5

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        if self.prediction_type == "v_prediction":
            return model_input * self.sigma_data**2 / (sigma**2 + self.sigma_data**2) - model_output * sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        else:
            return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
        if self.prediction_type == "const":
            noise_scale = 1.0
            return sigma * (noise_scale * noise) + (1.0 - sigma) * latent_image
        else:
            if max_denoise:
                noise = noise * torch.sqrt(1.0 + sigma**2.0)
            else:
                noise = noise * sigma

            noise += latent_image
            return noise

    def inverse_noise_scaling(self, sigma, latent):
        if self.prediction_type == "const":
            sigma = sigma.view(sigma.shape[:1] + (1,) * (latent.ndim - 1))
            return latent / (1.0 - sigma)
        else:
            return latent


class Prediction(AbstractPrediction):
    def __init__(self, sigma_data: float, prediction_type: str, beta_schedule: str, linear_start: float, linear_end: float, timesteps: int):
        super().__init__(sigma_data=sigma_data, prediction_type=prediction_type)
        self.register_schedule(beta_schedule=beta_schedule, timesteps=timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=8e-3)

    def register_schedule(self, beta_schedule: str, timesteps: int, linear_start: float, linear_end: float, cosine_s: float):
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5

        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("sigmas", sigmas.float())
        self.register_buffer("log_sigmas", sigmas.log().float())

    def set_sigmas(self, sigmas):
        self.register_buffer("sigmas", sigmas.float())
        self.register_buffer("log_sigmas", sigmas.log().float())

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape).to(sigma.device)

    def sigma(self, timestep):
        t = torch.clamp(timestep.float().to(self.log_sigmas.device), min=0, max=(len(self.sigmas) - 1))
        low_idx = t.floor().long()
        high_idx = t.ceil().long()
        w = t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp().to(timestep.device)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 999999999.9
        if percent >= 1.0:
            return 0.0
        percent = 1.0 - percent
        return self.sigma(torch.tensor(percent * 999.0)).item()


class PredictionFlux(AbstractPrediction):
    def __init__(self, model_config):
        super().__init__(sigma_data=None, prediction_type="const")

        sampling_settings: dict = model_config.sampling_settings
        self.set_parameters(shift=sampling_settings.get("shift", 1.15))

    def set_parameters(self, *, shift=None, timesteps=10000):
        self.shift = shift or self.shift
        ts = self.sigma((torch.arange(1, timesteps + 1, 1) / timesteps))
        self.register_buffer("sigmas", ts)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma

    def sigma(self, timestep):
        return flux_time_shift(self.shift, 1.0, timestep)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return flux_time_shift(self.shift, 1.0, 1.0 - percent)


class PredictionDiscreteFlow(AbstractPrediction):
    def __init__(self, model_config):
        super().__init__(sigma_data=None, prediction_type="const")

        sampling_settings: dict = model_config.sampling_settings
        self.set_parameters(shift=sampling_settings.get("shift", 1.0), multiplier=sampling_settings.get("multiplier", 1000))

    def set_parameters(self, *, shift=None, multiplier=None, timesteps=1000):
        self.shift = shift or self.shift
        self.multiplier = multiplier or self.multiplier
        ts = self.sigma((torch.arange(1, timesteps + 1, 1) / timesteps) * self.multiplier)
        self.register_buffer("sigmas", ts)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma * self.multiplier

    def sigma(self, timestep):
        return time_snr_shift(self.shift, timestep / self.multiplier)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return time_snr_shift(self.shift, 1.0 - percent)


def k_prediction_from_diffusers_scheduler(scheduler) -> "Prediction":
    assert getattr(scheduler.config, "prediction_type", None) in ["epsilon", "v_prediction"]
    assert scheduler.config.beta_schedule == "scaled_linear"
    return Prediction(sigma_data=1.0, prediction_type=scheduler.config.prediction_type, beta_schedule="linear", linear_start=scheduler.config.beta_start, linear_end=scheduler.config.beta_end, timesteps=scheduler.config.num_train_timesteps)
