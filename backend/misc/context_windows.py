# https://github.com/Comfy-Org/ComfyUI/blob/v0.24.1/comfy/context_windows.py
# https://github.com/Comfy-Org/ComfyUI/blob/v0.24.1/comfy_extras/nodes_context_windows.py

import collections
import logging
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from backend.patcher.controlnet import ControlBase
    from backend.patcher.unet import KModel

import torch

from backend.args import dynamic_args
from backend.logging import setup_logger

logger = logging.getLogger("ContextWindow")
setup_logger(logger)


ContextResults = collections.namedtuple("ContextResults", ["window_idx", "sub_conds_out", "sub_conds", "window"])


class IndexListContextWindow:
    def __init__(self, index_list: list[int], dim: int = 0, total_frames: int = 0):
        self.index_list = index_list
        self.context_length = len(index_list)
        self.dim = dim
        self.total_frames = total_frames
        self.center_ratio = (min(index_list) + max(index_list)) / (2 * total_frames)

    def get_tensor(self, full: torch.Tensor, device=None, dim=None, retain_index_list=[]) -> torch.Tensor:
        if dim is None:
            dim = self.dim
        if dim == 0 and full.shape[dim] == 1:
            return full
        indices = self.index_list
        anchor_idx = getattr(self, "causal_anchor_index", None)
        if anchor_idx is not None and anchor_idx >= 0:
            indices = [anchor_idx] + list(indices)
        idx = tuple([slice(None)] * dim + [indices])
        window = full[idx]
        if retain_index_list:
            idx = tuple([slice(None)] * dim + [retain_index_list])
            window[idx] = full[idx]
        return window.to(device)

    def add_window(self, full: torch.Tensor, to_add: torch.Tensor, dim=None) -> torch.Tensor:
        if dim is None:
            dim = self.dim
        idx = tuple([slice(None)] * dim + [self.index_list])
        full[idx] += to_add
        return full

    def get_region_index(self, num_regions: int) -> int:
        region_idx = int(self.center_ratio * num_regions)
        return min(max(region_idx, 0), num_regions - 1)


class IndexListContextHandler:
    def __init__(self, context_length: int = 2048, context_overlap: int = 512, context_stride: int = 1, closed_loop: bool = False, dim: int = 2, freenoise: bool = False, cond_retain_index_list: str = None, split_conds_to_windows: bool = False, causal_window_fix: bool = False):
        self.context_length = context_length
        self.context_overlap = context_overlap
        self.context_stride = context_stride
        self.closed_loop = closed_loop
        self.dim = dim
        self.freenoise = freenoise
        self.cond_retain_index_list = [int(x.strip()) for x in cond_retain_index_list.split(",")] if cond_retain_index_list else []
        self.split_conds_to_windows = split_conds_to_windows
        self.causal_window_fix = causal_window_fix

        self._step: int = 0
        self.orig_lq_latent: torch.Tensor = None

    def should_use_context(self, x_in: torch.Tensor) -> bool:
        return x_in.size(self.dim) > self.context_length

    def _prepare_control_objects(self, control: "ControlBase", device=None) -> "ControlBase":
        if control.previous_controlnet is not None:
            self._prepare_control_objects(control.previous_controlnet, device)
        return control

    def _get_resized_cond(self, cond_in: list[dict], x_in: torch.Tensor, window: IndexListContextWindow, device=None) -> list:
        if cond_in is None:
            return None

        resized_cond = []
        if self.split_conds_to_windows and len(cond_in) > 1:
            region = window.get_region_index(len(cond_in))
            logger.info(f"Splitting conds to windows; using region {region} for window {window.index_list[0]}-{window.index_list[-1]} with center ratio {window.center_ratio:.3f}")
            cond_in = [cond_in[region]]

        for actual_cond in cond_in:
            resized_actual_cond = actual_cond.copy()
            for key in actual_cond:
                try:
                    cond_item = actual_cond[key]
                    if key == "lq_latent":
                        dynamic_args.lq_latent[0] = _resize_cond_for_context_window(self._model, cond_item, window, device)
                    elif isinstance(cond_item, torch.Tensor):
                        if self.dim < cond_item.ndim and cond_item.size(self.dim) == x_in.size(self.dim):
                            actual_cond_item = window.get_tensor(cond_item)
                            resized_actual_cond[key] = actual_cond_item.to(device)
                        else:
                            resized_actual_cond[key] = cond_item.to(device)
                    elif key == "control":
                        resized_actual_cond[key] = self._prepare_control_objects(cond_item, device)
                    elif isinstance(cond_item, dict):
                        new_cond_item = cond_item.copy()
                        for cond_key, cond_value in new_cond_item.items():
                            if isinstance(cond_value, torch.Tensor):
                                if (self.dim < cond_value.ndim and cond_value.size(self.dim) == x_in.size(self.dim)) or (cond_value.ndim < self.dim and cond_value.size(0) == x_in.size(self.dim)):
                                    new_cond_item[cond_key] = window.get_tensor(cond_value, device)
                        resized_actual_cond[key] = new_cond_item
                    else:
                        resized_actual_cond[key] = cond_item
                finally:
                    del cond_item
            resized_cond.append(resized_actual_cond)

        return resized_cond

    def _set_step(self, timestep: torch.Tensor, model_options: dict[str]):
        mask = torch.isclose(model_options["transformer_options"]["sampling_sigmas"], timestep[0], rtol=0.0001)
        matches = torch.nonzero(mask)
        if torch.numel(matches) == 0:
            return
        self._step = int(matches[0].item())

    def _get_context_windows(self, x_in: torch.Tensor, model_options: dict[str]) -> list[IndexListContextWindow]:
        full_length = x_in.size(self.dim)
        context_windows = _create_windows_static_standard(full_length, self, model_options)
        context_windows = [IndexListContextWindow(window, dim=self.dim, total_frames=full_length) for window in context_windows]
        return context_windows

    def execute(self, calc_cond_batch: Callable, model: "KModel", conds: list[list[dict]], x_in: torch.Tensor, timestep: torch.Tensor, model_options: dict[str]):
        self._model = model
        self._set_step(timestep, model_options)
        context_windows = self._get_context_windows(x_in, model_options)
        enumerated_context_windows = list(enumerate(context_windows))

        conds[0][0]["lq_latent"] = self.orig_lq_latent.clone()
        conds_final = [torch.zeros_like(x_in) for _ in conds]
        counts_final = [torch.zeros(_get_shape_for_dim(x_in, self.dim), device=x_in.device) for _ in conds]

        for enum_window in enumerated_context_windows:
            results = self._evaluate_context_windows(calc_cond_batch, model, x_in, conds, timestep, [enum_window], model_options)
            for result in results:
                self._combine_context_window_results(x_in, result.sub_conds_out, result.window, conds_final, counts_final)

        for i in range(len(conds_final)):
            conds_final[i] /= counts_final[i]
        del counts_final

        return conds_final

    def _evaluate_context_windows(self, calc_cond_batch: Callable, model: "KModel", x_in: torch.Tensor, conds, timestep: torch.Tensor, enumerated_context_windows: list[tuple[int, IndexListContextWindow]], model_options, device=None):
        results: list[ContextResults] = []

        for window_idx, window in enumerated_context_windows:
            anchor_applied = False
            if self.causal_window_fix:
                anchor_idx = window.index_list[0] - 1
                if 0 <= anchor_idx < x_in.size(self.dim):
                    window.causal_anchor_index = anchor_idx
                    anchor_applied = True

            model_options["transformer_options"]["context_window"] = window
            sub_x = window.get_tensor(x_in, device)
            sub_timestep = window.get_tensor(timestep, device, dim=0)
            sub_conds = [self._get_resized_cond(cond, x_in, window, device) for cond in conds]

            sub_conds_out = calc_cond_batch(model, *sub_conds, sub_x, sub_timestep, model_options)
            if device is not None:
                for i in range(len(sub_conds_out)):
                    sub_conds_out[i] = sub_conds_out[i].to(x_in.device)

            if anchor_applied:
                for i in range(len(sub_conds_out)):
                    sub_conds_out[i] = sub_conds_out[i].narrow(self.dim, 1, sub_conds_out[i].shape[self.dim] - 1)

            results.append(ContextResults(window_idx, sub_conds_out, sub_conds, window))

        return results

    def _combine_context_window_results(self, x_in: torch.Tensor, sub_conds_out: list[torch.Tensor], window: IndexListContextWindow, conds_final: list[torch.Tensor], counts_final: list[torch.Tensor]):
        weights = _create_weights_pyramid(window.context_length)
        weights_tensor = _match_weights_to_dim(weights, x_in, self.dim, device=x_in.device)
        for i in range(len(sub_conds_out)):
            window.add_window(conds_final[i], sub_conds_out[i] * weights_tensor)
            window.add_window(counts_final[i], weights_tensor)


def _match_weights_to_dim(weights: list[float], x_in: torch.Tensor, dim: int, device=None) -> torch.Tensor:
    total_dims = len(x_in.shape)
    weights_tensor = torch.Tensor(weights).to(device=device)
    for _ in range(dim):
        weights_tensor = weights_tensor.unsqueeze(0)
    for _ in range(total_dims - dim - 1):
        weights_tensor = weights_tensor.unsqueeze(-1)
    return weights_tensor


def _get_shape_for_dim(x_in: torch.Tensor, dim: int) -> list[int]:
    total_dims = len(x_in.shape)
    shape = []
    for _ in range(dim):
        shape.append(1)
    shape.append(x_in.shape[dim])
    for _ in range(total_dims - dim - 1):
        shape.append(1)
    return shape


def _create_windows_static_standard(num_frames: int, handler: IndexListContextHandler, model_options: dict[str]):
    windows = []

    if num_frames <= handler.context_length:
        windows.append(list(range(num_frames)))
        return windows

    delta = handler.context_length - handler.context_overlap
    for start_idx in range(0, num_frames, delta):
        ending = start_idx + handler.context_length
        if ending >= num_frames:
            final_delta = ending - num_frames
            final_start_idx = start_idx - final_delta
            windows.append(list(range(final_start_idx, final_start_idx + handler.context_length)))
            break
        windows.append(list(range(start_idx, start_idx + handler.context_length)))

    return windows


def _create_weights_pyramid(length: int) -> list[float]:
    if length % 2 == 0:
        max_weight = length // 2
        weight_sequence = list(range(1, max_weight + 1, 1)) + list(range(max_weight, 0, -1))
    else:
        max_weight = (length + 1) // 2
        weight_sequence = list(range(1, max_weight, 1)) + [max_weight] + list(range(max_weight - 1, 0, -1))
    return weight_sequence


def _resize_cond_for_context_window(model: "KModel", cond_value: torch.Tensor, window: IndexListContextWindow, device):
    # https://github.com/Comfy-Org/ComfyUI/blob/v0.24.1/comfy/model_base.py#L1433

    dim = window.dim
    if dim >= cond_value.ndim:
        return None

    lq_proj = model.diffusion_model.lq_proj
    ratio = lq_proj.sr_scale * lq_proj.latent_spatial_down_factor
    lq_size = cond_value.size(dim)
    lq_indices = sorted({i // ratio for i in window.index_list if 0 <= i // ratio < lq_size})
    if not lq_indices:
        return None

    idx = tuple([slice(None)] * dim + [lq_indices])
    return cond_value[idx].to(device)
