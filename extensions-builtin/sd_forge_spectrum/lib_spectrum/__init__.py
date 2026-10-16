# https://github.com/ruwwww/ComfyUI-Spectrum-sdxl/blob/main/src/spectrum_node.py

import logging
import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.patcher.unet import UnetPatcher

import torch

from backend import stream
from backend.logging import setup_logger
from backend.memory_management import sync_stream

from .forecaster import ChebyshevForecaster, Spectrum

logger = logging.getLogger("Spectrum")
setup_logger(logger)


@dataclass
class State:
    forecasters: list[ChebyshevForecaster]
    cnt: int
    num_cached: list[int]
    curr_ws: float
    last_t: float
    total_runs: int
    estimated_total_steps: int


if stream.should_use_stream():
    main_stream = stream.get_current_stream()
    forecast_stream = stream.get_new_stream()
    context = lambda arg: stream.stream_context()(arg)
else:
    main_stream = None
    forecast_stream = None
    context = lambda _: nullcontext()


class SpectrumNode:

    @staticmethod
    def patch(model: "UnetPatcher", steps: int, w: float, m: int, lam: float, window_size: int, flex_window: float, warmup_steps: int, stop_caching_step: float):

        state = State(
            forecasters=None,
            cnt=0,
            num_cached=[0],
            curr_ws=float(window_size),
            last_t=-1,
            total_runs=0,
            estimated_total_steps=steps,
        )

        def spectrum_unet_wrapper(model_function, kwargs):
            x, timestep, c = kwargs["input"], kwargs["timestep"], kwargs["c"]
            t_scalar = timestep[0].item() if isinstance(timestep, torch.Tensor) and timestep.numel() > 0 else float(timestep)

            cond_or_uncond: list[int] = c["transformer_options"]["cond_or_uncond"]
            batch_count: int = x.shape[0]
            batch_size: int = batch_count // len(cond_or_uncond)
            num_forecasters: int = batch_size * 2

            if t_scalar > state.last_t:
                state.forecasters = None
                state.cnt = 0
                state.num_cached = [0] * num_forecasters
                state.curr_ws = float(window_size)
                state.total_runs += 1
                logger.info(f"Detected New Pass ({state.total_runs}) - Reset State")

            state.last_t = t_scalar

            if state.forecasters is None:
                state.forecasters = [Spectrum(cheb=ChebyshevForecaster(M=m, K=100, lam=lam, t_max=float(steps)), w=w) for _ in range(num_forecasters)]

            assert len(state.forecasters) == len(state.num_cached) == num_forecasters

            forecaster_indices: list[int] = []
            for i in range(batch_count):
                chunk = i // batch_size
                b = i % batch_size
                flag = int(cond_or_uncond[chunk])
                forecaster_indices.append(flag * batch_size + b)

            do_actual = torch.ones(batch_count, dtype=torch.bool, device=x.device)

            for i in range(batch_count):
                f_idx = forecaster_indices[i]
                is_micro_final = state.cnt >= int(state.estimated_total_steps * stop_caching_step)
                if state.cnt >= warmup_steps and not is_micro_final:
                    if state.forecasters[f_idx].ready():
                        do_actual[i] = (state.num_cached[f_idx] + 1) % math.floor(state.curr_ws) == 0
                    else:
                        do_actual[i] = True

            real_mask = do_actual
            forecast_mask = ~do_actual

            out = torch.empty_like(x)

            def _mask(v: torch.Tensor | Any, mask: torch.Tensor) -> torch.Tensor | Any:
                return v[mask.to(v.device)] if isinstance(v, torch.Tensor) and v.shape[0] == batch_count else v

            if real_mask.any():
                x_real = x[real_mask]
                timestep_real = _mask(timestep, real_mask)
                c_real = {k: _mask(v, real_mask) for k, v in c.items()}

                with context(main_stream):
                    raw_real = model_function(x_real, timestep_real, **c_real)

                out[real_mask] = raw_real

                real_indices = real_mask.nonzero().squeeze()
                if real_indices.dim() == 0:
                    real_indices = [real_indices.item()]
                else:
                    real_indices = real_indices.tolist()

                for i, idx in enumerate(real_indices):
                    f_idx = forecaster_indices[idx]
                    state.forecasters[f_idx].update(state.cnt, raw_real[i])
                    state.num_cached[f_idx] = 0

                logger.debug(f"Step {state.cnt}: Forward ({real_mask.sum().item()} items)")

            if forecast_mask.any():
                forecast_indices = forecast_mask.nonzero().squeeze()
                if forecast_indices.dim() == 0:
                    forecast_indices = [forecast_indices.item()]
                else:
                    forecast_indices = forecast_indices.tolist()

                out_forecast = torch.empty((len(forecast_indices), *x.shape[1:]), device=x.device, dtype=x.dtype)

                with context(forecast_stream):
                    for j, i in enumerate(forecast_indices):
                        f_idx = forecaster_indices[i]
                        raw_pred = state.forecasters[f_idx].predict(state.cnt)
                        out_forecast[j] = raw_pred

                    out[forecast_mask] = out_forecast
                    for i in forecast_indices:
                        state.num_cached[forecaster_indices[i]] += 1

                logger.debug(f"Step {state.cnt}: Forecast ({forecast_mask.sum().item()} items)")

            if stream.should_use_stream():
                sync_stream(x.device, main_stream)
                sync_stream(x.device, forecast_stream)

            if state.cnt >= warmup_steps:
                state.curr_ws += flex_window

            state.cnt += 1
            return out

        new_model = model.clone()
        new_model.set_model_unet_function_wrapper(spectrum_unet_wrapper)
        return new_model
