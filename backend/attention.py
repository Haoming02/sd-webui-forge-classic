import math

import einops
import torch

from backend import memory_management
from backend.args import args
from modules.errors import display_once

if memory_management.xformers_enabled():
    import xformers
    import xformers.ops

    try:
        x_vers = xformers.__version__
    except Exception:
        BROKEN_XFORMERS = True
    else:
        BROKEN_XFORMERS = x_vers.startswith("0.0.2") and not x_vers.startswith("0.0.20")

IS_SAGE_2 = False
"""SageAttention 2 has looser restrictions, allowing it to work on more models (e.g. SD1)"""

if memory_management.sage_enabled():
    import importlib.metadata

    from sageattention import sageattn

    IS_SAGE_2 = importlib.metadata.version("sageattention").startswith("2")

    if IS_SAGE_2 and args.sage2_fp16_triton:
        from functools import partial
        import sageattention

        _function = getattr(sageattention, f"sageattn_qk_int8_pv_fp16_triton")
        sageattn = partial(_function, quantization_backend="triton")


if memory_management.flash_enabled():
    from flash_attn import flash_attn_func

    @torch.library.custom_op("flash_attention::flash_attn", mutates_args=())
    def flash_attn_wrapper(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout_p: float = 0.0, causal: bool = False) -> torch.Tensor:
        return flash_attn_func(q, k, v, dropout_p=dropout_p, causal=causal)

    @flash_attn_wrapper.register_fake
    def flash_attn_fake(q, k, v, dropout_p=0.0, causal=False):
        return q.new_empty(q.shape)


FORCE_UPCAST_ATTENTION_DTYPE = memory_management.force_upcast_attention_dtype()


def get_xformers_flash_attention_op(q, k, v):
    try:
        flash_attention_op = xformers.ops.MemoryEfficientAttentionFlashAttentionOp
        fw, bw = flash_attention_op
        if fw.supports(xformers.ops.fmha.Inputs(query=q, key=k, value=v, attn_bias=None)):
            return flash_attention_op
    except Exception as e:
        display_once(e, "get_xformers_flash_attention_op")

    return None


def get_attn_precision(attn_precision=torch.float32):
    if args.disable_attention_upcast:
        return None
    if FORCE_UPCAST_ATTENTION_DTYPE is not None:
        return FORCE_UPCAST_ATTENTION_DTYPE
    return attn_precision


def exists(val):
    return val is not None


# ========== Diffusion ========== #


def attention_basic(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
    attn_precision = get_attn_precision(attn_precision)

    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads

    scale = dim_head**-0.5

    h = heads
    if skip_reshape:
        q, k, v = map(
            lambda t: t.reshape(b * heads, -1, dim_head),
            (q, k, v),
        )
    else:
        q, k, v = map(
            lambda t: t.unsqueeze(3).reshape(b, -1, heads, dim_head).permute(0, 2, 1, 3).reshape(b * heads, -1, dim_head).contiguous(),
            (q, k, v),
        )

    if attn_precision == torch.float32:
        sim = torch.einsum("b i d, b j d -> b i j", q.float(), k.float()) * scale
    else:
        sim = torch.einsum("b i d, b j d -> b i j", q, k) * scale

    del q, k

    if exists(mask):
        if mask.dtype == torch.bool:
            mask = einops.rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = einops.repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)
        else:
            if len(mask.shape) == 2:
                bs = 1
            else:
                bs = mask.shape[0]
            mask = mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1]).expand(b, heads, -1, -1).reshape(-1, mask.shape[-2], mask.shape[-1])
            sim.add_(mask)

    sim = sim.softmax(dim=-1)
    out = torch.einsum("b i j, b j d -> b i d", sim.to(v.dtype), v)
    out = out.unsqueeze(0).reshape(b, heads, -1, dim_head).permute(0, 2, 1, 3).reshape(b, -1, heads * dim_head)
    return out


def attention_split(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
    attn_precision = get_attn_precision(attn_precision)

    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads

    scale = dim_head**-0.5

    h = heads
    if skip_reshape:
        q, k, v = map(
            lambda t: t.reshape(b * heads, -1, dim_head),
            (q, k, v),
        )
    else:
        q, k, v = map(
            lambda t: t.unsqueeze(3).reshape(b, -1, heads, dim_head).permute(0, 2, 1, 3).reshape(b * heads, -1, dim_head).contiguous(),
            (q, k, v),
        )

    r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)

    mem_free_total = memory_management.get_free_memory(q.device)

    if attn_precision == torch.float32:
        element_size = 4
        upcast = True
    else:
        element_size = q.element_size()
        upcast = False

    gb = 1024**3
    tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * element_size
    modifier = 3
    mem_required = tensor_size * modifier
    steps = 1

    if mem_required > mem_free_total:
        steps = 2 ** (math.ceil(math.log(mem_required / mem_free_total, 2)))
        # print(f"Expected tensor size:{tensor_size/gb:0.1f}GB, cuda free:{mem_free_cuda/gb:0.1f}GB "
        #      f"torch free:{mem_free_torch/gb:0.1f} total:{mem_free_total/gb:0.1f} steps:{steps}")

    if steps > 64:
        max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
        raise RuntimeError(f"Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). " f"Need: {mem_required / 64 / gb:0.1f}GB free, Have:{mem_free_total / gb:0.1f}GB free")

    if mask is not None:
        if len(mask.shape) == 2:
            bs = 1
        else:
            bs = mask.shape[0]
        mask = mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1]).expand(b, heads, -1, -1).reshape(-1, mask.shape[-2], mask.shape[-1])

    # print("steps", steps, mem_required, mem_free_total, modifier, q.element_size(), tensor_size)
    first_op_done = False
    cleared_cache = False
    while True:
        try:
            slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
            for i in range(0, q.shape[1], slice_size):
                end = i + slice_size
                if upcast:
                    with torch.autocast(enabled=False, device_type="cuda"):
                        s1 = torch.einsum("b i d, b j d -> b i j", q[:, i:end].float(), k.float()) * scale
                else:
                    s1 = torch.einsum("b i d, b j d -> b i j", q[:, i:end], k) * scale

                if mask is not None:
                    if len(mask.shape) == 2:
                        s1 += mask[i:end]
                    else:
                        s1 += mask[:, i:end]

                s2 = s1.softmax(dim=-1).to(v.dtype)
                del s1
                first_op_done = True

                r1[:, i:end] = torch.einsum("b i j, b j d -> b i d", s2, v)
                del s2
            break
        except memory_management.OOM_EXCEPTION as e:
            if first_op_done == False:
                memory_management.soft_empty_cache(True)
                if cleared_cache == False:
                    cleared_cache = True
                    print("out of memory error, emptying cache and trying again")
                    continue
                steps *= 2
                if steps > 64:
                    raise e
                print("out of memory error, increasing steps and trying again {}".format(steps))
            else:
                raise e

    del q, k, v

    r1 = r1.unsqueeze(0).reshape(b, heads, -1, dim_head).permute(0, 2, 1, 3).reshape(b, -1, heads * dim_head)
    return r1


def attention_xformers(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads

    if BROKEN_XFORMERS and b * heads > 65535:
        return attention_pytorch(q, k, v, heads, mask, skip_reshape=skip_reshape)

    if skip_reshape:
        q, k, v = map(
            lambda t: t.reshape(b * heads, -1, dim_head),
            (q, k, v),
        )
    else:
        q, k, v = map(
            lambda t: t.reshape(b, -1, heads, dim_head),
            (q, k, v),
        )

    if mask is not None:
        pad = 8 - q.shape[1] % 8
        mask_out = torch.empty([q.shape[0], q.shape[1], q.shape[1] + pad], dtype=q.dtype, device=q.device)
        mask_out[:, :, : mask.shape[-1]] = mask
        mask = mask_out[:, :, : mask.shape[-1]]

    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask)

    if skip_reshape:
        out = out.unsqueeze(0).reshape(b, heads, -1, dim_head).permute(0, 2, 1, 3).reshape(b, -1, heads * dim_head)
    else:
        out = out.reshape(b, -1, heads * dim_head)

    return out


def attention_pytorch(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    return out


def attention_sage(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
    if skip_reshape:
        b, _, _, dim_head = q.shape
        tensor_layout = "HND"
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        tensor_layout = "NHD"

    if (IS_SAGE_2 and dim_head > 128) or ((not IS_SAGE_2) and (dim_head not in (64, 96, 128))):
        if memory_management.xformers_enabled():
            return attention_xformers(q, k, v, heads, mask, attn_precision, skip_reshape)
        else:
            return attention_pytorch(q, k, v, heads, mask, attn_precision, skip_reshape)

    if not skip_reshape:
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head),
            (q, k, v),
        )

    if mask is not None:
        # add a batch dimension if there isn't already one
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        # add a heads dimension if there isn't already one
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    try:
        out = sageattn(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout)
    except Exception as e:
        display_once(e, "attention_sage")
        if tensor_layout == "NHD":
            q, k, v = map(
                lambda t: t.transpose(1, 2),
                (q, k, v),
            )
        if memory_management.xformers_enabled():
            return attention_xformers(q, k, v, heads, mask=mask, skip_reshape=True)
        else:
            return attention_pytorch(q, k, v, heads, mask=mask, skip_reshape=True)

    if tensor_layout == "HND":
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    else:
        out = out.reshape(b, -1, heads * dim_head)

    return out


def attention_flash(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )

    if mask is not None:
        # add a batch dimension if there isn't already one
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        # add a heads dimension if there isn't already one
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    try:
        assert mask is None
        out = flash_attn_wrapper(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=0.0,
            causal=False,
        ).transpose(1, 2)
    except Exception as e:
        display_once(e, "attention_flash")
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)

    out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    return out


if memory_management.sage_enabled():
    print(f"Using SageAttention{' 2' if IS_SAGE_2 else ''}{' with fp16_triton function' if args.sage2_fp16_triton else ''}")
    attention_function = attention_sage
elif memory_management.flash_enabled():
    print("Using FlashAttention")
    attention_function = attention_flash
elif memory_management.xformers_enabled():
    print("Using xformers Cross Attention")
    attention_function = attention_xformers
elif memory_management.pytorch_attention_enabled():
    print("Using PyTorch Cross Attention")
    attention_function = attention_pytorch
elif args.attention_split:
    print("Using Split Optimization for Cross Attention")
    attention_function = attention_split
else:
    print("Using Basic Cross Attention")
    attention_function = attention_basic


# ========== VAE ========== #


def slice_attention_single_head_spatial(q, k, v):
    r1 = torch.zeros_like(k, device=q.device)
    scale = int(q.shape[-1]) ** (-0.5)

    mem_free_total = memory_management.get_free_memory(q.device)

    tensor_size = q.shape[0] * q.shape[1] * k.shape[2] * q.element_size()
    modifier = 3 if q.element_size() == 2 else 2.5
    mem_required = tensor_size * modifier
    steps = 1

    if mem_required > mem_free_total:
        steps = 2 ** (math.ceil(math.log(mem_required / mem_free_total, 2)))

    while True:
        try:
            slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
            for i in range(0, q.shape[1], slice_size):
                end = i + slice_size
                s1 = torch.bmm(q[:, i:end], k) * scale

                s2 = torch.nn.functional.softmax(s1, dim=2).permute(0, 2, 1)
                del s1

                r1[:, :, i:end] = torch.bmm(v, s2)
                del s2
            break
        except memory_management.OOM_EXCEPTION as e:
            memory_management.soft_empty_cache(True)
            steps *= 2
            if steps > 128:
                raise e
            print("out of memory error, increasing steps and trying again {}".format(steps))

    return r1


def normal_attention_single_head_spatial(q, k, v):
    # compute attention
    b, c, h, w = q.shape

    q = q.reshape(b, c, h * w)
    q = q.permute(0, 2, 1)  # b,hw,c
    k = k.reshape(b, c, h * w)  # b,c,hw
    v = v.reshape(b, c, h * w)

    r1 = slice_attention_single_head_spatial(q, k, v)
    h_ = r1.reshape(b, c, h, w)
    del r1
    return h_


def xformers_attention_single_head_spatial(q, k, v):
    # compute attention
    B, C, H, W = q.shape
    q, k, v = map(
        lambda t: t.view(B, C, -1).transpose(1, 2).contiguous(),
        (q, k, v),
    )

    try:
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=get_xformers_flash_attention_op(q, k, v))
        out = out.transpose(1, 2).reshape(B, C, H, W)
    except NotImplementedError:
        out = slice_attention_single_head_spatial(q.view(B, -1, C), k.view(B, -1, C).transpose(1, 2), v.view(B, -1, C).transpose(1, 2)).reshape(B, C, H, W)
    return out


def pytorch_attention_single_head_spatial(q, k, v):
    # compute attention
    B, C, H, W = q.shape
    q, k, v = map(
        lambda t: t.view(B, 1, C, -1).transpose(2, 3).contiguous(),
        (q, k, v),
    )

    try:
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        out = out.transpose(2, 3).reshape(B, C, H, W)
    except memory_management.OOM_EXCEPTION as e:
        display_once(e, "pytorch_attention_single_head_spatial")
        out = slice_attention_single_head_spatial(q.view(B, -1, C), k.view(B, -1, C).transpose(1, 2), v.view(B, -1, C).transpose(1, 2)).reshape(B, C, H, W)
    return out


if memory_management.xformers_enabled_vae():
    print("Using xformers Attention for VAE")
    attention_function_single_head_spatial = xformers_attention_single_head_spatial
elif memory_management.pytorch_attention_enabled():
    print("Using PyTorch Attention for VAE")
    attention_function_single_head_spatial = pytorch_attention_single_head_spatial
else:
    print("Using Split Attention for VAE")
    attention_function_single_head_spatial = normal_attention_single_head_spatial
