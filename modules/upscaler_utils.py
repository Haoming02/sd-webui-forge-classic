import logging
from functools import wraps
from typing import Callable

import numpy as np
import torch
import tqdm
from PIL import Image

from modules import devices, images, shared, torch_utils

logger = logging.getLogger(__name__)


def try_patch_spandrel():
    try:
        from spandrel.architectures.__arch_helpers.block import RRDB, ResidualDenseBlock_5C

        _orig_init: Callable = ResidualDenseBlock_5C.__init__
        _orig_5c_forward: Callable = ResidualDenseBlock_5C.forward
        _orig_forward: Callable = RRDB.forward

        @wraps(_orig_init)
        def RDB5C_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            self.nf, self.gc = kwargs.get("nf", 64), kwargs.get("gc", 32)

        @wraps(_orig_5c_forward)
        def RDB5C_forward(self, x: torch.Tensor):
            B, _, H, W = x.shape
            nf, gc = self.nf, self.gc

            buf = torch.empty((B, nf + 4 * gc, H, W), dtype=x.dtype, device=x.device)
            buf[:, :nf].copy_(x)

            x1 = self.conv1(x)
            buf[:, nf : nf + gc].copy_(x1)

            x2 = self.conv2(buf[:, : nf + gc])
            if self.conv1x1:
                x2.add_(self.conv1x1(x))
            buf[:, nf + gc : nf + 2 * gc].copy_(x2)

            x3 = self.conv3(buf[:, : nf + 2 * gc])
            buf[:, nf + 2 * gc : nf + 3 * gc].copy_(x3)

            x4 = self.conv4(buf[:, : nf + 3 * gc])
            if self.conv1x1:
                x4.add_(x2)
            buf[:, nf + 3 * gc : nf + 4 * gc].copy_(x4)

            x5 = self.conv5(buf)
            return x5.mul_(0.2).add_(x)

        @wraps(_orig_forward)
        def RRDB_forward(self, x):
            return self.RDB3(self.RDB2(self.RDB1(x))).mul_(0.2).add_(x)

        ResidualDenseBlock_5C.__init__ = RDB5C_init
        ResidualDenseBlock_5C.forward = RDB5C_forward
        RRDB.forward = RRDB_forward

        logger.info("Successfully patched Spandrel blocks")
    except Exception as e:
        logger.info(f"Failed to patch Spandrel blocks\n{type(e).__name__}: {e}")


try_patch_spandrel()


def _model(model: Callable, x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.float32 or model.architecture.name not in ("ATD", "DAT", "DRCT"):
        return model(x)

    # Spandrel does not correctly handle non-FP32 for certain models
    try:
        # Force the upscaler to use the dtype it should for new tensors
        torch.set_default_dtype(x.dtype)
        # Using torch.device incurs a small amount of overhead, but makes sure we don't
        # get errors when unsupported dtype tensors would be made on the CPU.
        with torch.device(x.device):
            return model(x)
    finally:
        torch.set_default_dtype(torch.float32)


def pil_rgb_to_tensor_bgr(img: Image.Image, param: torch.Tensor) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(img)).to(param.device)
    tensor = tensor.to(param.dtype).mul_(1.0 / 255.0).permute(2, 0, 1)
    return tensor[[2, 1, 0], ...].unsqueeze(0).contiguous()


def tensor_bgr_to_pil_rgb(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor[:, [2, 1, 0], ...]
    tensor = tensor.squeeze(0).permute(1, 2, 0).mul_(255.0).round_().clamp_(0.0, 255.0)
    return Image.fromarray(tensor.to(torch.uint8).cpu().numpy())


def pil_image_to_torch_bgr(img: Image.Image) -> torch.Tensor:
    img = np.array(img.convert("RGB"))
    img = img[:, :, ::-1]
    img = np.transpose(img, (2, 0, 1))
    img = np.ascontiguousarray(img) / 255
    return torch.from_numpy(img)


def torch_bgr_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"{tensor.shape} does not describe a BCHW tensor")
        tensor = tensor.squeeze(0)
    assert tensor.ndim == 3, f"{tensor.shape} does not describe a CHW tensor"
    arr = tensor.detach().float().cpu().numpy()
    arr = 255.0 * np.moveaxis(arr, 0, 2)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    arr = arr[:, :, ::-1]
    return Image.fromarray(arr, "RGB")


@torch.inference_mode()
def upscale_tensor_tiles(model: Callable, tensor: torch.Tensor, tile_size: int, overlap: int, desc: str) -> torch.Tensor:
    _, _, H_in, W_in = tensor.shape
    stride = tile_size - overlap
    n_tiles_x, n_tiles_y = (W_in + stride - 1) // stride, (H_in + stride - 1) // stride
    total_tiles = n_tiles_x * n_tiles_y

    if tile_size <= 0 or total_tiles <= 4:
        return _model(model, tensor)

    device = tensor.device
    dtype = tensor.dtype  # Accumulate in native model dtype

    accum = None
    model_scale = None
    H_out = W_out = None

    last_mask = None
    last_mask_key = None

    def get_weight_mask(h, w, y, x):
        """Generate feathered mask for tile overlap"""
        top, bottom, left, right = y > 0, y + h < H_out, x > 0, x + w < W_out
        key = (h, w, top, bottom, left, right)

        if key == last_mask_key:
            return key, last_mask
        elif overlap == 0:
            mask = torch.ones((1, 1, h, w), device=device, dtype=dtype)
        else:
            ov_h, ov_w = min(overlap, h), min(overlap, w)

            ramp_x, ramp_y = torch.ones(w, device=device, dtype=dtype), torch.ones(h, device=device, dtype=dtype)
            fade_x, fade_y = torch.linspace(0, 1, ov_w, device=device, dtype=dtype), torch.linspace(0, 1, ov_h, device=device, dtype=dtype)

            ramp_x[:ov_w].lerp_(fade_x, float(left))
            ramp_x[-ov_w:].lerp_(fade_x.flip(0), float(right))
            ramp_y[:ov_h].lerp_(fade_y, float(top))
            ramp_y[-ov_h:].lerp_(fade_y.flip(0), float(bottom))

            mask = (ramp_y[:, None] * ramp_x[None, :]).expand(1, 1, h, w)
        return key, mask

    with tqdm.tqdm(desc=desc, total=total_tiles) as pbar:
        for tile_idx in range(total_tiles):
            if shared.state.interrupted:
                return None

            # Loop in row-major or column-major, depending on aspect ratio to maximise hit-rate on cached mask
            x_idx, y_idx = (tile_idx % n_tiles_x, tile_idx // n_tiles_x) if W_in >= H_in else (tile_idx // n_tiles_y, tile_idx % n_tiles_y)
            x, y = x_idx * stride, y_idx * stride

            tile = tensor[:, :, y : y + tile_size, x : x + tile_size]
            out = _model(model, tile)

            if model_scale is None:
                model_scale = out.shape[-2] / tile.shape[-2]
                H_out, W_out = int(H_in * model_scale), int(W_in * model_scale)
                accum = torch.zeros((1, 4, H_out, W_out), dtype=dtype, device=device)

            h_out, w_out = out.shape[-2:]
            y_out, x_out = int(y * model_scale), int(x * model_scale)
            ys, ye = y_out, y_out + h_out
            xs, xe = x_out, x_out + w_out

            last_mask_key, last_mask = get_weight_mask(h_out, w_out, y_out, x_out)
            accum_slice = accum[:, :, ys:ye, xs:xe]
            accum_slice[:, :3].addcmul_(out, last_mask)
            accum_slice[:, 3:].add_(last_mask)

            del tile, out
            pbar.update(1)

    del last_mask
    return accum[:, :3].div_(accum[:, 3:].clamp_min_(1e-6))


def upscale_with_model_gpu(
    model: Callable[[torch.Tensor], torch.Tensor],
    img: Image.Image,
    *,
    tile_size: int,
    tile_overlap: int = 0,
    desc="tiled upscale",
) -> Image.Image:

    tensor = pil_rgb_to_tensor_bgr(img, torch_utils.get_param(model))
    out = upscale_tensor_tiles(model, tensor, tile_size, tile_overlap, desc)
    return img if out is None else tensor_bgr_to_pil_rgb(out)


def upscale_pil_patch(model, img: Image.Image) -> Image.Image:
    """Upscale a given PIL image using the given model"""
    param = torch_utils.get_param(model)

    with torch.inference_mode():
        tensor = pil_image_to_torch_bgr(img).unsqueeze(0)
        tensor = tensor.to(device=param.device, dtype=param.dtype)
        with devices.without_autocast():
            return torch_bgr_to_pil_image(_model(model, tensor))


def upscale_with_model_cpu(
    model: Callable[[torch.Tensor], torch.Tensor],
    img: Image.Image,
    *,
    tile_size: int,
    tile_overlap: int = 0,
    desc="tiled upscale",
) -> Image.Image:
    if tile_size <= 0:
        logger.debug("Upscaling %s without tiling", img)
        output = upscale_pil_patch(model, img)
        logger.debug("=> %s", output)
        return output

    grid = images.split_grid(img, tile_size, tile_size, tile_overlap)
    newtiles = []

    with tqdm.tqdm(
        total=grid.tile_count,
        desc=desc,
        disable=not shared.opts.enable_upscale_progressbar,
    ) as p:
        for y, h, row in grid.tiles:
            newrow = []
            for x, w, tile in row:
                if shared.state.interrupted:
                    break
                logger.debug("Tile (%d, %d) %s...", x, y, tile)
                output = upscale_pil_patch(model, tile)
                scale_factor = output.width // tile.width
                logger.debug("=> %s (scale factor %s)", output, scale_factor)
                newrow.append([x * scale_factor, w * scale_factor, output])
                p.update(1)
            newtiles.append([y * scale_factor, h * scale_factor, newrow])

    newgrid = images.Grid(
        newtiles,
        tile_w=grid.tile_w * scale_factor,
        tile_h=grid.tile_h * scale_factor,
        image_w=grid.image_w * scale_factor,
        image_h=grid.image_h * scale_factor,
        overlap=grid.overlap * scale_factor,
    )
    return images.combine_grid(newgrid)


def upscale_with_model(
    model: Callable[[torch.Tensor], torch.Tensor],
    img: Image.Image,
    *,
    tile_size: int,
    tile_overlap: int = 0,
    desc="tiled upscale",
) -> Image.Image:
    if shared.opts.composite_tiles_on_gpu:
        return upscale_with_model_gpu(model, img, tile_size=tile_size, tile_overlap=tile_overlap, desc=f"{desc} (GPU Composite)")
    else:
        return upscale_with_model_cpu(model, img, tile_size=tile_size, tile_overlap=tile_overlap, desc=f"{desc} (CPU Composite)")
