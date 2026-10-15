import logging

from backend.logging import setup_logger

logger = logging.getLogger("Spectrum")
setup_logger(logger)

import math
import torch
from .forecaster import ChebyshevForecaster, Spectrum


class SpectrumSDXL:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model to patch with Spectrum forecasting logic."}),
                "w": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Blending weight between predicted (Chebyshev) and local (Taylor) features. Lower (0.3-0.5) preserves sharpness, higher relies on global smoothing."}),
                "m": ("INT", {"default": 3, "min": 1, "max": 8, "tooltip": "Number of Chebyshev basis functions (forecast complexity). Lower values (3-4) are more stable for SDXL."}),
                "lam": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Ridge regularization strength. Prevents latent explosions and rainbow artifacts in low-precision modes."}),
                "window_size": ("INT", {"default": 2, "min": 1, "max": 10, "tooltip": "Initial forecasting window size (number of skipped steps)."}),
                "flex_window": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Increment added to window size after each actual UNet pass. Higher = aggressive acceleration."}),
                "warmup_steps": ("INT", {"default": 5, "min": 0, "max": 20, "tooltip": "Initial full-model steps before forecasting begins. Gives the model time to establish composition."}),
                "stop_caching_step": ("INT", {"default": -1, "min": -1, "max": 100, "step": 1, "tooltip": "The exact step where Spectrum stops and returns to native UNet. Essential for final detail recovery. Set to Total Steps - 3."}),
                "steps": ("INT", {"default": 30, "min": 10, "max": 500, "step": 1, "tooltip": "Match this value with your KSampler total steps for stable forecast accuracy and drift reduction."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "sampling"

    def patch(self, model, w, m, lam, window_size, flex_window, warmup_steps, stop_caching_step, steps=30):
        self.total_steps = steps

        state = getattr(model, "spectrum_state", {})
        state["total_steps"] = steps
        model.spectrum_state = state

        state = {
            "forecasters": None,
            "cnt": 0,
            "num_cached": [0],
            "curr_ws": float(window_size),
            "last_t": -1,
            "total_runs": 0,
            "estimated_total_steps": steps,
        }

        # Remove any lingering hooks from previously bypassed models to clear global memory leaks
        diffusion_model = model.model.diffusion_model
        if hasattr(diffusion_model, "_sp_hooks"):
            for h in diffusion_model._sp_hooks:
                h.remove()
            diffusion_model._sp_hooks = []
        if hasattr(diffusion_model, "spectrum_hook_handles"):
            for h in diffusion_model.spectrum_hook_handles:
                h.remove()
            diffusion_model.spectrum_hook_handles = []

        forecast_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        def spectrum_unet_wrapper(model_function, kwargs):
            x, timestep, c = kwargs["input"], kwargs["timestep"], kwargs["c"]
            batch_size = x.shape[0]
            t_scalar = timestep[0].item() if isinstance(timestep, torch.Tensor) and timestep.numel() > 0 else float(timestep)

            if t_scalar > state["last_t"]:
                state["forecasters"] = None
                state["cnt"] = 0
                state["num_cached"] = [0] * batch_size
                state["curr_ws"] = float(window_size)
                state["total_runs"] += 1
                print(f"[Spectrum] Detected new pass ({state['total_runs']}) - Reset state")

            state["last_t"] = t_scalar

            if state["forecasters"] is None:
                state["forecasters"] = [Spectrum(cheb_like=ChebyshevForecaster(M=m, K=100, lam=lam, t_max=float(steps)), w=w) for _ in range(batch_size)]

            if len(state["num_cached"]) != batch_size:
                state["num_cached"] = [0] * batch_size

            do_actual = torch.ones(batch_size, dtype=torch.bool, device=x.device)
            for i in range(batch_size):
                is_micro_final = False
                if stop_caching_step == -1:
                    auto_stop = int(state["estimated_total_steps"] * 0.8)
                    if state["cnt"] >= auto_stop:
                        is_micro_final = True
                elif stop_caching_step > 0 and state["cnt"] >= stop_caching_step:
                    is_micro_final = True
                if state["cnt"] >= warmup_steps and not is_micro_final:
                    # Match harness_port.py: FORCE actual forward if not enough cached points yet
                    if state["forecasters"][i].ready():
                        do_actual[i] = (state["num_cached"][i] + 1) % math.floor(state["curr_ws"]) == 0
                    else:
                        do_actual[i] = True

            real_mask = do_actual
            forecast_mask = ~do_actual

            out = torch.empty_like(x)

            # ====================== REAL STEP: Run full diffusion_model → capture RAW 4D tensor (post final_layer/unpatchify) ======================
            if real_mask.any():
                x_real = x[real_mask]
                timestep_real = timestep[real_mask.to(timestep.device)] if isinstance(timestep, torch.Tensor) and timestep.shape[0] == batch_size else timestep
                c_real = {k: v[real_mask.to(v.device)] if isinstance(v, torch.Tensor) and v.shape[0] == batch_size else v for k, v in c.items()}

                with torch.cuda.stream(torch.cuda.default_stream()):
                    raw_real = model_function(x_real, timestep_real, **c_real)  # ← RAW 4D tensor from diffusion_model

                # ComfyUI's Sampler automatically applies calculate_denoised externally on the UNet's return value
                out[real_mask] = raw_real

                # Update forecaster with the RAW tensor (Spatial Feature)
                real_indices = real_mask.nonzero().squeeze()
                if real_indices.dim() == 0:
                    real_indices = [real_indices.item()]
                else:
                    real_indices = real_indices.tolist()

                for i, idx in enumerate(real_indices):
                    state["forecasters"][idx].update(state["cnt"], raw_real[i])
                    state["num_cached"][idx] = 0

                print(f"[Spectrum] Step {state['cnt']}: Real forward (RAW captured) for {real_mask.sum().item()} items")

            # ====================== SKIP STEP: Forecast RAW tensor ======================
            if forecast_mask.any():
                forecast_indices = forecast_mask.nonzero().squeeze()
                if forecast_indices.dim() == 0:
                    forecast_indices = [forecast_indices.item()]
                else:
                    forecast_indices = forecast_indices.tolist()

                out_forecast = torch.empty((len(forecast_indices), *x.shape[1:]), device=x.device, dtype=x.dtype)

                if forecast_stream:
                    with torch.cuda.stream(forecast_stream):
                        for j, i in enumerate(forecast_indices):
                            raw_pred = state["forecasters"][i].predict(state["cnt"])  # forecasted RAW 4D tensor
                            out_forecast[j] = raw_pred

                        out[forecast_mask] = out_forecast
                        for i in forecast_indices:
                            state["num_cached"][i] += 1
                    torch.cuda.current_stream().wait_stream(forecast_stream)
                else:
                    # CPU fallback
                    for j, i in enumerate(forecast_indices):
                        raw_pred = state["forecasters"][i].predict(state["cnt"])
                        out_forecast[j] = raw_pred

                    out[forecast_mask] = out_forecast
                    for i in forecast_indices:
                        state["num_cached"][i] += 1

                print(f"[Spectrum] Step {state['cnt']}: Forecast (RAW predicted) for {forecast_mask.sum().item()} items")

            if state["cnt"] >= warmup_steps:
                state["curr_ws"] += flex_window

            state["cnt"] += 1
            return out

        new_model = model.clone()

        # SAFEGUARD: Deepcopy model_options to prevent the wrapper from permanently
        # mutating the globally cached CheckpointLoader model in memory.
        import copy

        if hasattr(model, "model_options"):
            new_model.model_options = copy.deepcopy(model.model_options)

        new_model.set_model_unet_function_wrapper(spectrum_unet_wrapper)
        return (new_model,)
