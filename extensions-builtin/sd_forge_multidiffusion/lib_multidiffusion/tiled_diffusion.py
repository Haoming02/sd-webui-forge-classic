# 1st Edit by. https://github.com/shiimizu/ComfyUI-TiledDiffusion
# 2nd Edit by. Forge Official
# 3rd Edit by. Panchovix
# 4th Edit by. Haoming02
# - Based on: https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111

from math import pi
from typing import Callable, Dict, List, Tuple, Union
from weakref import WeakSet

import torch
from torch import Tensor

import ldm_patched.modules.model_management
from ldm_patched.modules.controlnet import ControlNet, T2IAdapter
from ldm_patched.modules.model_base import BaseModel
from ldm_patched.modules.model_management import current_loaded_models, load_models_gpu
from ldm_patched.modules.model_patcher import ModelPatcher
from ldm_patched.modules.utils import common_upscale

opt_C = 4
opt_f = 8


def processing_interrupted():
    from modules import shared

    return shared.state.interrupted or shared.state.skipped


def ceildiv(big, small):
    return -(big // -small)


from enum import Enum


class BlendMode(Enum):
    FOREGROUND = "Foreground"
    BACKGROUND = "Background"


class Processing: ...


class Device: ...


devices = Device()
devices.device = ldm_patched.modules.model_management.get_torch_device()


def null_decorator(fn):
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


keep_signature = null_decorator
controlnet = null_decorator
stablesr = null_decorator
grid_bbox = null_decorator
custom_bbox = null_decorator
noise_inverse = null_decorator


class BBox:
    """grid bbox"""

    def __init__(self, x: int, y: int, w: int, h: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.box = [x, y, x + w, y + h]
        self.slicer = slice(None), slice(None), slice(y, y + h), slice(x, x + w)

    def __getitem__(self, idx: int) -> int:
        return self.box[idx]


def repeat_to_batch_size(tensor, batch_size, dim=0):
    if dim == 0 and tensor.shape[dim] == 1:
        return tensor.expand([batch_size] + [-1] * (len(tensor.shape) - 1))
    if tensor.shape[dim] > batch_size:
        return tensor.narrow(dim, 0, batch_size)
    elif tensor.shape[dim] < batch_size:
        return tensor.repeat(dim * [1] + [ceildiv(batch_size, tensor.shape[dim])] + [1] * (len(tensor.shape) - 1 - dim)).narrow(dim, 0, batch_size)
    return tensor


def split_bboxes(w: int, h: int, tile_w: int, tile_h: int, overlap: int = 16, init_weight: Union[Tensor, float] = 1.0) -> Tuple[List[BBox], Tensor]:
    cols = ceildiv((w - overlap), (tile_w - overlap))
    rows = ceildiv((h - overlap), (tile_h - overlap))
    dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
    dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

    bbox_list: List[BBox] = []
    weight = torch.zeros((1, 1, h, w), device=devices.device, dtype=torch.float32)
    for row in range(rows):
        y = min(int(row * dy), h - tile_h)
        for col in range(cols):
            x = min(int(col * dx), w - tile_w)

            bbox = BBox(x, y, tile_w, tile_h)
            bbox_list.append(bbox)
            weight[bbox.slicer] += init_weight

    return bbox_list, weight


class CustomBBox(BBox):
    """region control bbox"""

    pass


class AbstractDiffusion:
    def __init__(self):
        self.method = self.__class__.__name__
        self.pbar = None

        self.w: int = 0
        self.h: int = 0
        self.tile_width: int = None
        self.tile_height: int = None
        self.tile_overlap: int = None
        self.tile_batch_size: int = None

        self.x_buffer: Tensor = None
        self._weights: Tensor = None
        self._init_grid_bbox = None
        self._init_done = None

        self.step_count = 0
        self.inner_loop_count = 0
        self.kdiff_step = -1

        self.enable_grid_bbox: bool = False
        self.tile_w: int = None
        self.tile_h: int = None
        self.tile_bs: int = None
        self.num_tiles: int = None
        self.num_batches: int = None
        self.batched_bboxes: List[List[BBox]] = []

        self.enable_custom_bbox: bool = False
        self.custom_bboxes: List[CustomBBox] = []

        self.enable_controlnet: bool = False
        self.control_tensor_batch_dict = {}
        self.control_tensor_batch: List[List[Tensor]] = [[]]
        self.control_params: Dict[Tuple, List[List[Tensor]]] = {}
        self.control_tensor_cpu: bool = None
        self.control_tensor_custom: List[List[Tensor]] = []

        self.draw_background: bool = True
        self.control_tensor_cpu = False
        self.weights = None
        self.uniform_distribution = None
        self.sigmas = None

    def reset(self):
        tile_width = self.tile_width
        tile_height = self.tile_height
        tile_overlap = self.tile_overlap
        tile_batch_size = self.tile_batch_size
        compression = self.compression
        width = self.width
        height = self.height
        overlap = self.overlap
        self.__init__()
        self.compression = compression
        self.width = width
        self.height = height
        self.overlap = overlap
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.tile_overlap = tile_overlap
        self.tile_batch_size = tile_batch_size

    def repeat_tensor(self, x: Tensor, n: int, concat=False, concat_to=0) -> Tensor:
        """repeat the tensor on it's first dim"""
        if n == 1:
            return x
        B = x.shape[0]
        r_dims = len(x.shape) - 1
        if B == 1:
            shape = [n] + [-1] * r_dims
            return x.expand(shape)
        else:
            if concat:
                return torch.cat([x for _ in range(n)], dim=0)[:concat_to]
            shape = [n] + [1] * r_dims
            return x.repeat(shape)

    def update_pbar(self):
        if self.pbar.n >= self.pbar.total:
            self.pbar.close()
        else:
            sampling_step = 20
            if self.step_count == sampling_step:
                self.inner_loop_count += 1
                if self.inner_loop_count < self.total_bboxes:
                    self.pbar.update()
            else:
                self.step_count = sampling_step
                self.inner_loop_count = 0

    def reset_buffer(self, x_in: Tensor):
        if self.x_buffer is None or self.x_buffer.shape != x_in.shape:
            self.x_buffer = torch.zeros_like(x_in, device=x_in.device, dtype=x_in.dtype)
        else:
            self.x_buffer.zero_()

    @grid_bbox
    def init_grid_bbox(self, tile_w: int, tile_h: int, overlap: int, tile_bs: int):
        self.weights = torch.zeros((1, 1, self.h, self.w), device=devices.device, dtype=torch.float32)
        self.enable_grid_bbox = True

        self.tile_w = min(tile_w, self.w)
        self.tile_h = min(tile_h, self.h)
        overlap = max(0, min(overlap, min(tile_w, tile_h) - 4))
        bboxes, weights = split_bboxes(self.w, self.h, self.tile_w, self.tile_h, overlap, self.get_tile_weights())
        self.weights += weights
        self.num_tiles = len(bboxes)
        self.num_batches = ceildiv(self.num_tiles, tile_bs)
        self.tile_bs = ceildiv(len(bboxes), self.num_batches)
        self.batched_bboxes = [bboxes[i * self.tile_bs : (i + 1) * self.tile_bs] for i in range(self.num_batches)]

    @grid_bbox
    def get_grid_bbox(self, tile_w: int, tile_h: int, overlap: int, tile_bs: int, w: int, h: int, device: torch.device, get_tile_weights: Callable = lambda: 1.0) -> List[List[BBox]]:
        weights = torch.zeros((1, 1, h, w), device=device, dtype=torch.float32)

        tile_w = min(tile_w, w)
        tile_h = min(tile_h, h)
        overlap = max(0, min(overlap, min(tile_w, tile_h) - 4))
        bboxes, weights_ = split_bboxes(w, h, tile_w, tile_h, overlap, get_tile_weights())
        weights += weights_
        num_tiles = len(bboxes)
        num_batches = ceildiv(num_tiles, tile_bs)
        tile_bs = ceildiv(len(bboxes), num_batches)
        batched_bboxes = [bboxes[i * tile_bs : (i + 1) * tile_bs] for i in range(num_batches)]
        return batched_bboxes

    @grid_bbox
    def get_tile_weights(self) -> Union[Tensor, float]:
        return 1.0

    @noise_inverse
    def init_noise_inverse(self, steps: int, retouch: float, get_cache_callback, set_cache_callback, renoise_strength: float, renoise_kernel: int):
        self.noise_inverse_enabled = True
        self.noise_inverse_steps = steps
        self.noise_inverse_retouch = float(retouch)
        self.noise_inverse_renoise_strength = float(renoise_strength)
        self.noise_inverse_renoise_kernel = int(renoise_kernel)
        self.noise_inverse_set_cache = set_cache_callback
        self.noise_inverse_get_cache = get_cache_callback

    def init_done(self):
        """
        Call this after all `init_*`, settings are done, now perform:
          - settings sanity check
          - pre-computations, cache init
          - anything thing needed before denoising starts
        """

        self.total_bboxes = 0
        if self.enable_grid_bbox:
            self.total_bboxes += self.num_batches
        if self.enable_custom_bbox:
            self.total_bboxes += len(self.custom_bboxes)
        assert self.total_bboxes > 0, "Nothing to paint! No background to draw and no custom bboxes were provided."

    @controlnet
    def prepare_controlnet_tensors(self, refresh: bool = False, tensor=None):
        """Crop the control tensor into tiles and cache them"""
        if not refresh:
            if self.control_tensor_batch is not None or self.control_params is not None:
                return
        tensors = [tensor]
        self.org_control_tensor_batch = tensors
        self.control_tensor_batch = []
        for i in range(len(tensors)):
            control_tile_list = []
            control_tensor = tensors[i]
            for bboxes in self.batched_bboxes:
                single_batch_tensors = []
                for bbox in bboxes:
                    if len(control_tensor.shape) == 3:
                        control_tensor.unsqueeze_(0)
                    control_tile = control_tensor[:, :, bbox[1] * opt_f : bbox[3] * opt_f, bbox[0] * opt_f : bbox[2] * opt_f]
                    single_batch_tensors.append(control_tile)
                control_tile = torch.cat(single_batch_tensors, dim=0)
                if self.control_tensor_cpu:
                    control_tile = control_tile.cpu()
                control_tile_list.append(control_tile)
            self.control_tensor_batch.append(control_tile_list)

            if len(self.custom_bboxes) > 0:
                custom_control_tile_list = []
                for bbox in self.custom_bboxes:
                    if len(control_tensor.shape) == 3:
                        control_tensor.unsqueeze_(0)
                    control_tile = control_tensor[:, :, bbox[1] * opt_f : bbox[3] * opt_f, bbox[0] * opt_f : bbox[2] * opt_f]
                    if self.control_tensor_cpu:
                        control_tile = control_tile.cpu()
                    custom_control_tile_list.append(control_tile)
                self.control_tensor_custom.append(custom_control_tile_list)

    @controlnet
    def switch_controlnet_tensors(self, batch_id: int, x_batch_size: int, tile_batch_size: int, is_denoise=False):
        if self.control_tensor_batch is None:
            return

        for param_id in range(len(self.control_tensor_batch)):
            control_tile = self.control_tensor_batch[param_id][batch_id]
            if x_batch_size > 1:
                all_control_tile = []
                for i in range(tile_batch_size):
                    this_control_tile = [control_tile[i].unsqueeze(0)] * x_batch_size
                    all_control_tile.append(torch.cat(this_control_tile, dim=0))
                control_tile = torch.cat(all_control_tile, dim=0)
                self.control_tensor_batch[param_id][batch_id] = control_tile

    def process_controlnet(self, x_noisy, c_in: dict, cond_or_uncond: List, bboxes, batch_size: int, batch_id: int, shifts=None, shift_condition=None):
        control: ControlNet = c_in["control"]
        param_id = -1
        tuple_key = tuple(cond_or_uncond) + tuple(x_noisy.shape)
        while control is not None:
            param_id += 1

            if tuple_key not in self.control_params:
                self.control_params[tuple_key] = [[None]]

            while len(self.control_params[tuple_key]) <= param_id:
                self.control_params[tuple_key].append([None])

            while len(self.control_params[tuple_key][param_id]) <= batch_id:
                self.control_params[tuple_key][param_id].append(None)

            if self.refresh or control.cond_hint is None or not isinstance(self.control_params[tuple_key][param_id][batch_id], Tensor):
                if control.cond_hint is not None:
                    del control.cond_hint
                control.cond_hint = None
                compression_ratio = control.compression_ratio
                if control.vae is not None:
                    compression_ratio *= control.vae.downscale_ratio
                else:
                    if control.latent_format is not None:
                        raise ValueError("This Controlnet needs a VAE but none was provided, please use a ControlNetApply node with a VAE input and connect it.")
                PH, PW = self.h * compression_ratio, self.w * compression_ratio

                device = getattr(control, "device", x_noisy.device)
                dtype = getattr(control, "manual_cast_dtype", None)
                if dtype is None:
                    dtype = getattr(getattr(control, "control_model", None), "dtype", None)
                if dtype is None:
                    dtype = x_noisy.dtype

                if isinstance(control, T2IAdapter):
                    width, height = control.scale_image_to(PW, PH)
                    cns = common_upscale(control.cond_hint_original, width, height, control.upscale_algorithm, "center").float().to(device=device)
                    if control.channels_in == 1 and control.cond_hint.shape[1] > 1:
                        cns = torch.mean(control.cond_hint, 1, keepdim=True)
                elif control.__class__.__name__ == "ControlLLLiteAdvanced":
                    if getattr(control, "sub_idxs", None) is not None and control.cond_hint_original.shape[0] >= control.full_latent_length:
                        cns = common_upscale(control.cond_hint_original[control.sub_idxs], PW, PH, control.upscale_algorithm, "center").to(dtype=dtype, device=device)
                    else:
                        cns = common_upscale(control.cond_hint_original, PW, PH, control.upscale_algorithm, "center").to(dtype=dtype, device=device)
                else:
                    cns = common_upscale(control.cond_hint_original, PW, PH, control.upscale_algorithm, "center").to(dtype=dtype, device=device)
                    if getattr(control, "vae", None) is not None:
                        loaded_models_ = current_loaded_models(only_currently_used=True)
                        cns = control.vae.encode(cns.movedim(1, -1))
                        load_models_gpu(loaded_models_)
                    if getattr(control, "latent_format", None) is not None:
                        cns = control.latent_format.process_in(cns)
                    if len(getattr(control, "extra_concat_orig", ())) > 0:
                        to_concat = []
                        for c in control.extra_concat_orig:
                            c = c.to(device=device)
                            c = common_upscale(c, cns.shape[3], cns.shape[2], control.upscale_algorithm, "center")
                            to_concat.append(repeat_to_batch_size(c, cns.shape[0]))
                        cns = torch.cat([cns] + to_concat, dim=1)

                    cns = cns.to(device=device, dtype=dtype)
                cf = control.compression_ratio
                if cns.shape[0] != batch_size:
                    cns = repeat_to_batch_size(cns, batch_size)
                if shifts is not None:
                    control.cns = cns
                    sh_h, sh_w = shifts
                    sh_h *= cf
                    sh_w *= cf
                    if (sh_h, sh_w) != (0, 0):
                        if sh_h == 0 or sh_w == 0:
                            cns = control.cns.roll(shifts=(sh_h, sh_w), dims=(-2, -1))
                        else:
                            if shift_condition:
                                cns = control.cns.roll(shifts=sh_h, dims=-2)
                            else:
                                cns = control.cns.roll(shifts=sh_w, dims=-1)
                cns_slices = [cns[:, :, bbox[1] * cf : bbox[3] * cf, bbox[0] * cf : bbox[2] * cf] for bbox in bboxes]
                control.cond_hint = torch.cat(cns_slices, dim=0).to(device=cns.device)
                del cns_slices
                del cns
                self.control_params[tuple_key][param_id][batch_id] = control.cond_hint
            else:
                if hasattr(control, "cns") and shifts is not None:
                    cf = control.compression_ratio
                    cns = control.cns
                    sh_h, sh_w = shifts
                    sh_h *= cf
                    sh_w *= cf
                    if (sh_h, sh_w) != (0, 0):
                        if sh_h == 0 or sh_w == 0:
                            cns = control.cns.roll(shifts=(sh_h, sh_w), dims=(-2, -1))
                        else:
                            if shift_condition:
                                cns = control.cns.roll(shifts=sh_h, dims=-2)
                            else:
                                cns = control.cns.roll(shifts=sh_w, dims=-1)
                    cns_slices = [cns[:, :, bbox[1] * cf : bbox[3] * cf, bbox[0] * cf : bbox[2] * cf] for bbox in bboxes]
                    control.cond_hint = torch.cat(cns_slices, dim=0).to(device=cns.device)
                    del cns_slices
                    del cns
                else:
                    control.cond_hint = self.control_params[tuple_key][param_id][batch_id]
            control = control.previous_controlnet


import numpy as np
from numpy import exp, pi, sqrt


def gaussian_weights(tile_w: int, tile_h: int) -> Tensor:
    """
    Copy from the original implementation of Mixture of Diffusers
    https://github.com/albarji/mixture-of-diffusers/blob/master/mixdiff/tiling.py
    This generates gaussian weights to smooth the noise of each tile.
    This is critical for this method to work.
    """
    f = lambda x, midpoint, var=0.01: exp(-(x - midpoint) * (x - midpoint) / (tile_w * tile_w) / (2 * var)) / sqrt(2 * pi * var)
    x_probs = [f(x, (tile_w - 1) / 2) for x in range(tile_w)]
    y_probs = [f(y, tile_h / 2) for y in range(tile_h)]

    w = np.outer(y_probs, x_probs)
    return torch.from_numpy(w).to(devices.device, dtype=torch.float32)


class CondDict: ...


class MultiDiffusion(AbstractDiffusion):

    @torch.inference_mode()
    def __call__(self, model_function: BaseModel.apply_model, args: dict):
        x_in: Tensor = args["input"]
        t_in: Tensor = args["timestep"]
        c_in: dict = args["c"]
        cond_or_uncond: List = args["cond_or_uncond"]

        N, C, H, W = x_in.shape

        self.refresh = False
        if self.weights is None or self.h != H or self.w != W:
            self.h, self.w = H, W
            self.refresh = True
            self.init_grid_bbox(self.tile_width, self.tile_height, self.tile_overlap, self.tile_batch_size)
            self.init_done()
        self.h, self.w = H, W
        self.reset_buffer(x_in)

        if self.draw_background:
            for batch_id, bboxes in enumerate(self.batched_bboxes):
                if processing_interrupted():
                    return x_in

                x_tile = torch.cat([x_in[bbox.slicer] for bbox in bboxes], dim=0)
                t_tile = repeat_to_batch_size(t_in, x_tile.shape[0])
                c_tile = {}
                for k, v in c_in.items():
                    if isinstance(v, torch.Tensor):
                        if len(v.shape) == len(x_tile.shape):
                            bboxes_ = bboxes
                            if v.shape[-2:] != x_in.shape[-2:]:
                                cf = x_in.shape[-1] * self.compression // v.shape[-1]
                                bboxes_ = self.get_grid_bbox(
                                    self.width // cf,
                                    self.height // cf,
                                    self.overlap // cf,
                                    self.tile_batch_size,
                                    v.shape[-1],
                                    v.shape[-2],
                                    x_in.device,
                                    self.get_tile_weights,
                                )
                            v = torch.cat([v[bbox_.slicer] for bbox_ in bboxes_[batch_id]])
                        if v.shape[0] != x_tile.shape[0]:
                            v = repeat_to_batch_size(v, x_tile.shape[0])
                    c_tile[k] = v

                if "control" in c_in:
                    self.process_controlnet(x_tile, c_in, cond_or_uncond, bboxes, N, batch_id)
                    c_tile["control"] = c_in["control"].get_control_orig(x_tile, t_tile, c_tile, len(cond_or_uncond))

                x_tile_out = model_function(x_tile, t_tile, **c_tile)

                for i, bbox in enumerate(bboxes):
                    self.x_buffer[bbox.slicer] += x_tile_out[i * N : (i + 1) * N, :, :, :]
                del x_tile_out, x_tile, t_tile, c_tile

        x_out = torch.where(self.weights > 1, self.x_buffer / self.weights, self.x_buffer)

        return x_out


class Store:
    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


store = Store()


def fibonacci_spacing(x):
    result = torch.zeros_like(x)
    fib = [0, 1]
    while fib[-1] < len(x):
        fib.append(fib[-1] + fib[-2])

    used_indices = set()
    for i, val in enumerate(x):
        fib_index = i % len(fib)
        target_index = fib[fib_index] % len(x)
        while target_index in used_indices:
            target_index = (target_index + 1) % len(x)
        result[target_index] = val
        used_indices.add(target_index)

    return result


def find_nearest(a, b):
    diff = (a - b).abs()

    nearest_indices = diff.argmin()

    return b[nearest_indices]


class SpotDiffusion(AbstractDiffusion):

    @torch.inference_mode()
    def __call__(self, model_function: BaseModel.apply_model, args: dict):
        x_in: Tensor = args["input"]
        t_in: Tensor = args["timestep"]
        c_in: dict = args["c"]
        cond_or_uncond: List = args["cond_or_uncond"]

        N, C, H, W = x_in.shape

        self.refresh = False
        if self.weights is None or self.h != H or self.w != W:
            self.h, self.w = H, W
            self.refresh = True
            self.init_grid_bbox(self.tile_width, self.tile_height, self.tile_overlap, self.tile_batch_size)
            self.init_done()
        self.h, self.w = H, W
        self.reset_buffer(x_in)

        if self.uniform_distribution is None:
            sigmas = self.sigmas = store.sigmas
            shift_method = store.model_options.get("tiled_diffusion_shift_method", "random")
            seed = store.model_options.get("tiled_diffusion_seed", store.extra_args.get("seed", 0))
            th = self.tile_height
            tw = self.tile_width
            cf = self.compression
            if "effnet" in c_in:
                cf = x_in.shape[-1] * self.compression // c_in["effnet"].shape[-1]
                th = self.height // cf
                tw = self.width // cf
            shift_height = torch.randint(0, th, (len(sigmas) - 1,), generator=torch.Generator(device="cpu").manual_seed(seed), device="cpu")
            shift_height = (shift_height * cf / self.compression).round().to(torch.int32)
            shift_width = torch.randint(0, tw, (len(sigmas) - 1,), generator=torch.Generator(device="cpu").manual_seed(seed), device="cpu")
            shift_width = (shift_width * cf / self.compression).round().to(torch.int32)
            if shift_method == "sorted":
                shift_height = shift_height.sort().values
                shift_width = shift_width.sort().values
            elif shift_method == "fibonacci":
                shift_height = fibonacci_spacing(shift_height.sort().values)
                shift_width = fibonacci_spacing(shift_width.sort().values)
            self.uniform_distribution = (shift_height, shift_width)

        sigmas = self.sigmas
        ts_in = find_nearest(t_in[0], sigmas)
        cur_i = ss.item() if (ss := (sigmas == ts_in).nonzero()).shape[0] != 0 else 0

        sh_h = self.uniform_distribution[0][cur_i].item()
        sh_w = self.uniform_distribution[1][cur_i].item()
        if min(self.tile_height, x_in.shape[-2]) == x_in.shape[-2]:
            sh_h = 0
        if min(self.tile_width, x_in.shape[-1]) == x_in.shape[-1]:
            sh_w = 0
        condition = cur_i % 2 == 0 if self.tile_height > self.tile_width else cur_i % 2 != 0
        if (sh_h, sh_w) != (0, 0):
            if sh_h == 0 or sh_w == 0:
                x_in = x_in.roll(shifts=(sh_h, sh_w), dims=(-2, -1))
            else:
                if condition:
                    x_in = x_in.roll(shifts=sh_h, dims=-2)
                else:
                    x_in = x_in.roll(shifts=sh_w, dims=-1)

        if self.draw_background:
            for batch_id, bboxes in enumerate(self.batched_bboxes):
                if processing_interrupted():
                    return x_in

                x_tile = torch.cat([x_in[bbox.slicer] for bbox in bboxes], dim=0)
                t_tile = repeat_to_batch_size(t_in, x_tile.shape[0])
                c_tile = {}
                for k, v in c_in.items():
                    if isinstance(v, torch.Tensor):
                        if len(v.shape) == len(x_tile.shape):
                            bboxes_ = bboxes
                            sh_h_new, sh_w_new = sh_h, sh_w
                            if v.shape[-2:] != x_in.shape[-2:]:
                                cf = x_in.shape[-1] * self.compression // v.shape[-1]
                                bboxes_ = self.get_grid_bbox(
                                    self.width // cf,
                                    self.height // cf,
                                    self.overlap // cf,
                                    self.tile_batch_size,
                                    v.shape[-1],
                                    v.shape[-2],
                                    x_in.device,
                                    self.get_tile_weights,
                                )
                                sh_h_new, sh_w_new = round(sh_h * self.compression / cf), round(sh_w * self.compression / cf)
                            v = v.roll(shifts=(sh_h_new, sh_w_new), dims=(-2, -1))
                            v = torch.cat([v[bbox_.slicer] for bbox_ in bboxes_[batch_id]])
                        if v.shape[0] != x_tile.shape[0]:
                            v = repeat_to_batch_size(v, x_tile.shape[0])
                    c_tile[k] = v

                if "control" in c_in:
                    self.process_controlnet(x_tile, c_in, cond_or_uncond, bboxes, N, batch_id, (sh_h, sh_w), condition)
                    c_tile["control"] = c_in["control"].get_control_orig(x_tile, t_tile, c_tile, len(cond_or_uncond))

                x_tile_out = model_function(x_tile, t_tile, **c_tile)

                for i, bbox in enumerate(bboxes):
                    self.x_buffer[bbox.slicer] = x_tile_out[i * N : (i + 1) * N, :, :, :]

                del x_tile_out, x_tile, t_tile, c_tile

        if (sh_h, sh_w) != (0, 0):
            if sh_h == 0 or sh_w == 0:
                self.x_buffer = self.x_buffer.roll(shifts=(-sh_h, -sh_w), dims=(-2, -1))
            else:
                if condition:
                    self.x_buffer = self.x_buffer.roll(shifts=-sh_h, dims=-2)
                else:
                    self.x_buffer = self.x_buffer.roll(shifts=-sh_w, dims=-1)

        return self.x_buffer


class MixtureOfDiffusers(AbstractDiffusion):
    """
    Mixture-of-Diffusers Implementation
    https://github.com/albarji/mixture-of-diffusers
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.custom_weights: List[Tensor] = []
        self.get_weight = gaussian_weights

    def init_done(self):
        super().init_done()
        self.rescale_factor = 1 / self.weights
        for bbox_id, bbox in enumerate(self.custom_bboxes):
            if bbox.blend_mode == BlendMode.BACKGROUND:
                self.custom_weights[bbox_id] *= self.rescale_factor[bbox.slicer]

    @grid_bbox
    def get_tile_weights(self) -> Tensor:
        self.tile_weights = self.get_weight(self.tile_w, self.tile_h)
        return self.tile_weights

    @torch.inference_mode()
    def __call__(self, model_function: BaseModel.apply_model, args: dict):
        x_in: Tensor = args["input"]
        t_in: Tensor = args["timestep"]
        c_in: dict = args["c"]
        cond_or_uncond: List = args["cond_or_uncond"]

        N, C, H, W = x_in.shape

        self.refresh = False
        if self.weights is None or self.h != H or self.w != W:
            self.h, self.w = H, W
            self.refresh = True
            self.init_grid_bbox(self.tile_width, self.tile_height, self.tile_overlap, self.tile_batch_size)
            self.init_done()
        self.h, self.w = H, W
        self.reset_buffer(x_in)

        if self.draw_background:
            for batch_id, bboxes in enumerate(self.batched_bboxes):
                if processing_interrupted():
                    return x_in
                x_tile_list = []
                for bbox in bboxes:
                    x_tile_list.append(x_in[bbox.slicer])

                x_tile = torch.cat(x_tile_list, dim=0)
                t_tile = repeat_to_batch_size(t_in, x_tile.shape[0])
                c_tile = {}
                for k, v in c_in.items():
                    if isinstance(v, torch.Tensor):
                        if len(v.shape) == len(x_tile.shape):
                            bboxes_ = bboxes
                            if v.shape[-2:] != x_in.shape[-2:]:
                                cf = x_in.shape[-1] * self.compression // v.shape[-1]
                                bboxes_ = self.get_grid_bbox(
                                    (tile_w := self.width // cf),
                                    (tile_h := self.height // cf),
                                    self.overlap // cf,
                                    self.tile_batch_size,
                                    v.shape[-1],
                                    v.shape[-2],
                                    x_in.device,
                                    lambda: self.get_weight(tile_w, tile_h),
                                )
                            v = torch.cat([v[bbox_.slicer] for bbox_ in bboxes_[batch_id]])
                        if v.shape[0] != x_tile.shape[0]:
                            v = repeat_to_batch_size(v, x_tile.shape[0])
                    c_tile[k] = v

                if "control" in c_in:
                    self.process_controlnet(x_tile, c_in, cond_or_uncond, bboxes, N, batch_id)
                    c_tile["control"] = c_in["control"].get_control_orig(x_tile, t_tile, c_tile, len(cond_or_uncond))

                x_tile_out = model_function(x_tile, t_tile, **c_tile)

                for i, bbox in enumerate(bboxes):
                    w = self.tile_weights * self.rescale_factor[bbox.slicer]
                    self.x_buffer[bbox.slicer] += x_tile_out[i * N : (i + 1) * N, :, :, :] * w
                del x_tile_out, x_tile, t_tile, c_tile

        x_out = self.x_buffer

        return x_out


MAX_RESOLUTION = 8192


class TiledDiffusion:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "method": (["MultiDiffusion", "Mixture of Diffusers", "SpotDiffusion"], {"default": "Mixture of Diffusers"}),
                "tile_width": ("INT", {"default": 96 * opt_f, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                "tile_height": ("INT", {"default": 96 * opt_f, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                "tile_overlap": ("INT", {"default": 8 * opt_f, "min": 0, "max": 256 * opt_f, "step": 4 * opt_f}),
                "tile_batch_size": ("INT", {"default": 4, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "_for_testing"
    instances = WeakSet()

    @classmethod
    def IS_CHANGED(s, *args, **kwargs):
        for o in s.instances:
            o.impl.reset()
        return ""

    def __init__(self) -> None:
        self.__class__.instances.add(self)

    def apply(self, model: ModelPatcher, method, tile_width, tile_height, tile_overlap, tile_batch_size):
        if method == "Mixture of Diffusers":
            self.impl = MixtureOfDiffusers()
        elif method == "MultiDiffusion":
            self.impl = MultiDiffusion()
        else:
            self.impl = SpotDiffusion()

        compression = 4 if "CASCADE" in str(model.model.model_type) else 8
        self.impl.tile_width = tile_width // compression
        self.impl.tile_height = tile_height // compression
        self.impl.tile_overlap = tile_overlap // compression
        self.impl.tile_batch_size = tile_batch_size

        self.impl.compression = compression
        self.impl.width = tile_width
        self.impl.height = tile_height
        self.impl.overlap = tile_overlap
        model = model.clone()
        model.set_model_unet_function_wrapper(self.impl)
        model.model_options["tiled_diffusion"] = True
        return (model,)
