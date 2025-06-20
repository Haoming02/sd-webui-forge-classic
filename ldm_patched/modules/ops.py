"""
Credit: ComfyUI
https://github.com/comfyanonymous/ComfyUI

- Edited by. Forge Official
- Edited by. Haoming02
"""

import contextlib

import torch
from ldm_patched.modules.model_management import cast_to_device, device_supports_non_blocking
from modules_forge import stream

stash = {}


@contextlib.contextmanager
def use_patched_ops(operations):
    names = ("Linear", "Conv2d", "Conv3d", "GroupNorm", "LayerNorm")
    backups = {name: getattr(torch.nn, name) for name in names}

    try:
        for name in names:
            setattr(torch.nn, name, getattr(operations, name))
        yield
    finally:
        for name in names:
            setattr(torch.nn, name, backups[name])
        return


def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None):
    if input is not None:
        if dtype is None:
            dtype = input.dtype
        if bias_dtype is None:
            bias_dtype = dtype
        if device is None:
            device = input.device

    bias, signal = None, None

    with stream.stream_context()(stream.mover_stream) if stream.using_stream else contextlib.nullcontext():
        if s.bias is not None:
            bias = cast_to_device(
                s.bias,
                device=device,
                dtype=bias_dtype,
            )

        weight = cast_to_device(
            s.weight,
            device=device,
            dtype=dtype,
        )

        if stream.using_stream:
            signal = stream.mover_stream.record_event()

    return weight, bias, signal


@contextlib.contextmanager
def main_stream_worker(weight, bias, signal):
    if not stream.using_stream or signal is None:
        yield
        return

    with stream.stream_context()(stream.current_stream):
        stream.current_stream.wait_event(signal)
        yield
        finished_signal = stream.current_stream.record_event()
        stash[id(finished_signal)] = (weight, bias, finished_signal)

    garbage = []
    for k, (_, _, s) in stash.items():
        if s.query():
            garbage.append(k)

    for k in garbage:
        del stash[k]


def cleanup_cache():
    if not stream.using_stream:
        return

    stream.current_stream.synchronize()
    stream.mover_stream.synchronize()
    stash.clear()


class disable_weight_init:
    class Linear(torch.nn.Linear):
        ldm_patched_cast_weights = False

        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias, signal = cast_bias_weight(self, input)
            with main_stream_worker(weight, bias, signal):
                return torch.nn.functional.linear(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv2d(torch.nn.Conv2d):
        ldm_patched_cast_weights = False

        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias, signal = cast_bias_weight(self, input)
            with main_stream_worker(weight, bias, signal):
                return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv3d(torch.nn.Conv3d):
        ldm_patched_cast_weights = False

        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias, signal = cast_bias_weight(self, input)
            with main_stream_worker(weight, bias, signal):
                return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class GroupNorm(torch.nn.GroupNorm):
        ldm_patched_cast_weights = False

        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias, signal = cast_bias_weight(self, input)
            with main_stream_worker(weight, bias, signal):
                return torch.nn.functional.group_norm(
                    input,
                    self.num_groups,
                    weight,
                    bias,
                    self.eps,
                )

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class LayerNorm(torch.nn.LayerNorm):
        ldm_patched_cast_weights = False

        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias, signal = cast_bias_weight(self, input)
            with main_stream_worker(weight, bias, signal):
                return torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    @classmethod
    def conv_nd(s, dims, *args, **kwargs):
        match dims:
            case 2:
                return s.Conv2d(*args, **kwargs)
            case 3:
                return s.Conv3d(*args, **kwargs)
            case _:
                raise ValueError(f"unsupported dimensions: {dims}")


class manual_cast(disable_weight_init):
    class Linear(disable_weight_init.Linear):
        ldm_patched_cast_weights = True

    class Conv2d(disable_weight_init.Conv2d):
        ldm_patched_cast_weights = True

    class Conv3d(disable_weight_init.Conv3d):
        ldm_patched_cast_weights = True

    class GroupNorm(disable_weight_init.GroupNorm):
        ldm_patched_cast_weights = True

    class LayerNorm(disable_weight_init.LayerNorm):
        ldm_patched_cast_weights = True


def fp8_linear(self, input):
    dtype = self.weight.dtype
    if dtype is not torch.float8_e4m3fn:
        return None

    tensor_2d = False
    if len(input.shape) == 2:
        tensor_2d = True
        input = input.unsqueeze(1)

    if len(input.shape) != 3:
        return None

    input_shape = input.shape
    input_dtype = input.dtype
    input_device = input.device

    w, bias, signal = cast_bias_weight(self, input, dtype=dtype, bias_dtype=input_dtype)
    w = w.t()

    scale_weight = self.scale_weight
    scale_input = self.scale_input
    if scale_weight is None:
        scale_weight = torch.ones((), device=input_device, dtype=torch.float32)
    else:
        scale_weight = scale_weight.to(input_device)

    if scale_input is None:
        scale_input = torch.ones((), device=input_device, dtype=torch.float32)
        inn = torch.clamp(input, min=-448, max=448).view(-1, input_shape[2]).to(dtype)
    else:
        scale_input = scale_input.to(input_device)
        inn = (input * (1.0 / scale_input).to(input_dtype)).view(-1, input_shape[2]).to(dtype)

    with main_stream_worker(w, bias, signal):
        o = torch._scaled_mm(
            input=inn,
            mat2=w,
            bias=bias,
            out_dtype=input_dtype,
            scale_a=scale_input,
            scale_b=scale_weight,
        )

    if isinstance(o, tuple):
        o = o[0]

    if tensor_2d:
        return o.view(input_shape[0], -1)
    else:
        return o.view((-1, input_shape[1], self.weight.shape[0]))


class fp8_ops(manual_cast):
    class Linear(manual_cast.Linear):
        def reset_parameters(self):
            self.scale_weight = None
            self.scale_input = None
            return None

        def forward_ldm_patched_cast_weights(self, input):
            if (out := fp8_linear(self, input)) is not None:
                return out

            weight, bias, signal = cast_bias_weight(self, input)
            with main_stream_worker(weight, bias, signal):
                return torch.nn.functional.linear(input, weight, bias)


try:
    from cublas_ops import CublasLinear
except ImportError:
    pass
else:

    class cublas_ops(disable_weight_init):
        class Linear(CublasLinear, disable_weight_init.Linear):
            def reset_parameters(self):
                return None

            def forward_ldm_patched_cast_weights(self, input):
                return super().forward(input)

            def forward(self, *args, **kwargs):
                return super().forward(*args, **kwargs)


class tiled_ops(disable_weight_init):
    class Conv2d(disable_weight_init.Conv2d):
        tile_size: int

        def __init__(self, *arg, **kwargs):
            from modules.shared import opts

            super().__init__(*arg, **kwargs)
            self._3x1x1: bool = self.kernel_size == (3, 3) and self.stride == (1, 1) and self.padding == (1, 1)
            self.tile_size = opts.sd_vae_tiled_size

        @torch.inference_mode()
        def forward(self, x: torch.Tensor):
            if not self._3x1x1:
                return super().forward(x)

            B, C, H, W = x.shape

            if H <= self.tile_size and W <= self.tile_size:
                return super().forward(x)

            out = torch.empty((B, C if self.out_channels is None else self.out_channels, H, W), device=x.device, dtype=x.dtype, memory_format=torch.contiguous_format)
            non_blocking = device_supports_non_blocking(x.device)

            for i in range(0, H, self.tile_size):
                for j in range(0, W, self.tile_size):
                    i0 = max(i - 1, 0)
                    j0 = max(j - 1, 0)
                    i1 = min(i + self.tile_size + 1, H)
                    j1 = min(j + self.tile_size + 1, W)

                    tile = x[:, :, i0:i1, j0:j1]
                    tile_conv = super().forward(tile)

                    pi = i - i0
                    pj = j - j0
                    ph = min(self.tile_size, H - i)
                    pw = min(self.tile_size, W - j)

                    out[:, :, i : i + ph, j : j + pw].copy_(tile_conv[:, :, pi : pi + ph, pj : pj + pw], non_blocking=non_blocking)
                    del tile_conv

            return out

    class Upsample(torch.nn.Module):
        tile_size: int

        def __init__(self, in_channels, with_conv):
            from modules.shared import opts

            super().__init__()
            self.with_conv = with_conv
            if self.with_conv:
                self.conv = tiled_ops.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

            self.tile_size = opts.sd_vae_tiled_size

        @torch.inference_mode()
        def forward(self, x: torch.Tensor):
            B, C, H, W = x.shape
            out_H, out_W = (H * 2, W * 2)

            if out_H <= self.tile_size and out_W <= self.tile_size:
                x_up = torch.nn.functional.interpolate(x, size=(out_H, out_W), mode="nearest")
                return x_up if not self.with_conv else self.conv(x_up)

            scale_h = out_H / H
            scale_w = out_W / W

            out = torch.empty((B, C, out_H, out_W), device=x.device, dtype=x.dtype, memory_format=torch.contiguous_format)
            non_blocking = device_supports_non_blocking(x.device)

            for i in range(0, H, self.tile_size):
                for j in range(0, W, self.tile_size):
                    i0 = max(i - 1, 0)
                    j0 = max(j - 1, 0)
                    i1 = min(i + self.tile_size + 1, H)
                    j1 = min(j + self.tile_size + 1, W)
                    tile = x[:, :, i0:i1, j0:j1]

                    tile_up = torch.nn.functional.interpolate(tile, scale_factor=(scale_h, scale_w), mode="nearest")

                    if self.with_conv:
                        tile_up = self.conv(tile_up)

                    pi = int((i - i0) * scale_h)
                    pj = int((j - j0) * scale_w)
                    ph = int(min(self.tile_size, H - i) * scale_h)
                    pw = int(min(self.tile_size, W - j) * scale_w)

                    oi = int(i * scale_h)
                    oj = int(j * scale_w)

                    # Clip output patch to not exceed requested output size
                    ph = min(ph, out_H - oi)
                    pw = min(pw, out_W - oj)

                    out[:, :, oi : oi + ph, oj : oj + pw].copy_(tile_up[:, :, pi : pi + ph, pj : pj + pw], non_blocking=non_blocking)
                    del tile_up

            return out
