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


def pil_rgb_to_tensor_bgr(img: Image.Image, param: torch.Tensor) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(img)).to(param.device)
    tensor = tensor.to(param.dtype).mul_(1.0 / 255.0).permute(2, 0, 1)
    return tensor[[2, 1, 0], ...].unsqueeze(0).contiguous()


def tensor_bgr_to_pil_rgb(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor[:, [2, 1, 0], ...]
    tensor = tensor.squeeze(0).permute(1, 2, 0).mul_(255.0).round_().clamp_(0.0, 255.0)
    return Image.fromarray(tensor.to(torch.uint8).cpu().numpy())


@torch.inference_mode()
def upscale_tensor_tiles(model, tensor: torch.Tensor, tile_size: int, overlap: int, desc: str) -> torch.Tensor:
    _, _, H_in, W_in = tensor.shape
    stride = tile_size - overlap
    n_tiles_x, n_tiles_y = (W_in + stride - 1) // stride, (H_in + stride - 1) // stride
    total_tiles = n_tiles_x * n_tiles_y

    if tile_size <= 0 or total_tiles <= 4:
        return model(tensor)

    device = tensor.device
    dtype = tensor.dtype  # Accumulate in native model dtype.

    accum = None
    model_scale = None
    H_out = W_out = None

    last_mask = None
    last_mask_key = None

    # Generate feathered mask for tile overlap.
    def get_weight_mask(h, w, y, x):
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

            # Loop in row-major or column-major, depending on aspect ratio to maximise hit-rate on cached mask.
            x_idx, y_idx = (tile_idx % n_tiles_x, tile_idx // n_tiles_x) if W_in >= H_in else (tile_idx // n_tiles_y, tile_idx % n_tiles_y)
            x, y = x_idx * stride, y_idx * stride

            tile = tensor[:, :, y : y + tile_size, x : x + tile_size]
            out = model(tile)

            if model_scale is None:
                model_scale = out.shape[-2] / tile.shape[-2]
                H_out, W_out = int(H_in * model_scale), int(W_in * model_scale)
                accum = torch.zeros((1, 4, H_out, W_out), dtype=dtype, device=device)

            h_out, w_out = out.shape[-2:]
            y_out, x_out = int(y * model_scale), int(x * model_scale)
            ys, ye = y_out, y_out + h_out
            xs, xe = x_out, x_out + w_out

            # With correct traversal order, mask hit-rate is about 40%.
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


def pil_image_to_torch_bgr(img: Image.Image) -> torch.Tensor:
    img = np.array(img.convert("RGB"))
    img = img[:, :, ::-1]  # flip RGB to BGR
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.ascontiguousarray(img) / 255  # Rescale to [0, 1]
    return torch.from_numpy(img)


def torch_bgr_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim == 4:
        # If we're given a tensor with a batch dimension, squeeze it out
        # (but only if it's a batch of size 1).
        if tensor.shape[0] != 1:
            raise ValueError(f"{tensor.shape} does not describe a BCHW tensor")
        tensor = tensor.squeeze(0)
    assert tensor.ndim == 3, f"{tensor.shape} does not describe a CHW tensor"
    arr = tensor.detach().float().cpu().numpy()
    arr = 255.0 * np.moveaxis(arr, 0, 2)  # CHW to HWC, rescale
    arr = np.clip(arr, 0, 255).astype(np.uint8)  # clamp
    arr = arr[:, :, ::-1]  # flip BGR to RGB
    return Image.fromarray(arr, "RGB")


def upscale_pil_patch(model, img: Image.Image) -> Image.Image:
    """
    Upscale a given PIL image using the given model.
    """
    param = torch_utils.get_param(model)

    with torch.inference_mode():
        tensor = pil_image_to_torch_bgr(img).unsqueeze(0)  # add batch dimension
        tensor = tensor.to(device=param.device, dtype=param.dtype)
        with devices.without_autocast():
            return torch_bgr_to_pil_image(model(tensor))


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
        return upscale_with_model_gpu(model, img, tile_size=tile_size, tile_overlap=tile_overlap, desc=f"{desc} (GPU composite)")
    else:
        return upscale_with_model_cpu(model, img, tile_size=tile_size, tile_overlap=tile_overlap, desc=f"{desc} (CPU composite)")


def tiled_upscale_2(
    img: torch.Tensor,
    model,
    *,
    tile_size: int,
    tile_overlap: int,
    scale: int,
    device: torch.device,
    desc="Tiled upscale",
):
    """
    Alternative implementation of `upscale_with_model` originally used by
    SwinIR and ScuNET. It differs from `upscale_with_model` in that tiling and
    weighting is done in PyTorch space, as opposed to `images.Grid` doing it in
    Pillow space without weighting.
    """

    b, c, h, w = img.size()
    tile_size = min(tile_size, h, w)

    if tile_size <= 0:
        logger.debug("Upscaling %s without tiling", img.shape)
        return model(img)

    stride = tile_size - tile_overlap
    h_idx_list = list(range(0, h - tile_size, stride)) + [h - tile_size]
    w_idx_list = list(range(0, w - tile_size, stride)) + [w - tile_size]
    result = torch.zeros(
        b,
        c,
        h * scale,
        w * scale,
        device=device,
        dtype=img.dtype,
    )
    weights = torch.zeros_like(result)
    logger.debug("Upscaling %s to %s with tiles", img.shape, result.shape)
    with tqdm.tqdm(
        total=len(h_idx_list) * len(w_idx_list),
        desc=desc,
        disable=not shared.opts.enable_upscale_progressbar,
    ) as pbar:
        for h_idx in h_idx_list:
            if shared.state.interrupted or shared.state.skipped:
                break

            for w_idx in w_idx_list:
                if shared.state.interrupted or shared.state.skipped:
                    break

                # Only move this patch to the device if it's not already there.
                in_patch = img[
                    ...,
                    h_idx : h_idx + tile_size,
                    w_idx : w_idx + tile_size,
                ].to(device=device)

                out_patch = model(in_patch)

                result[
                    ...,
                    h_idx * scale : (h_idx + tile_size) * scale,
                    w_idx * scale : (w_idx + tile_size) * scale,
                ].add_(out_patch)

                out_patch_mask = torch.ones_like(out_patch)

                weights[
                    ...,
                    h_idx * scale : (h_idx + tile_size) * scale,
                    w_idx * scale : (w_idx + tile_size) * scale,
                ].add_(out_patch_mask)

                pbar.update(1)

    output = result.div_(weights)

    return output


def upscale_2(
    img: Image.Image,
    model,
    *,
    tile_size: int,
    tile_overlap: int,
    scale: int,
    desc: str,
):
    """
    Convenience wrapper around `tiled_upscale_2` that handles PIL images.
    """
    param = torch_utils.get_param(model)
    tensor = pil_image_to_torch_bgr(img).to(dtype=param.dtype).unsqueeze(0)

    with torch.inference_mode():
        output = tiled_upscale_2(
            tensor,
            model,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            scale=scale,
            desc=desc,
            device=param.device,
        )
    return torch_bgr_to_pil_image(output)
