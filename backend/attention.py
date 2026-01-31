# https://github.com/Comfy-Org/ComfyUI/blob/v0.7.0/comfy/ldm/modules/attention.py

import logging
import math

import torch
from einops import rearrange, repeat
from torch import einsum

from backend import memory_management, operations
from backend.args import args
from backend.logging import setup_logger

logger = logging.getLogger("attention")
setup_logger(logger)


if memory_management.xformers_enabled() or memory_management.xformers_enabled_vae():
    import xformers
    import xformers.ops


if memory_management.sage_enabled():
    import importlib.metadata

    if importlib.metadata.version("sageattention").startswith("1"):
        IS_SAGE_2 = False
        from sageattention import sageattn
    else:
        IS_SAGE_2 = True
        from backend.args import SageAttentionFuncs

        if args.sage2_function is SageAttentionFuncs.auto:
            from sageattention import sageattn
        else:

            from functools import partial

            import sageattention

            _function = getattr(sageattention, f"sageattn_qk_int8_pv_{args.sage2_function.value}")
            if args.sage2_function is SageAttentionFuncs.fp16_triton:
                sageattn = partial(_function, quantization_backend=args.sage_quantization_backend.value)
            else:
                sageattn = partial(_function, qk_quant_gran=args.sage_quant_gran.value, pv_accum_dtype=args.sage_accum_dtype.value)


if memory_management.flash_enabled():
    from flash_attn import flash_attn_func

    @torch.library.custom_op("flash_attention::flash_attn", mutates_args=())
    def flash_attn_wrapper(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout_p: float = 0.0, causal: bool = False) -> torch.Tensor:
        return flash_attn_func(q, k, v, dropout_p=dropout_p, causal=causal)

    @flash_attn_wrapper.register_fake
    def flash_attn_fake(q, k, v, dropout_p=0.0, causal=False):
        return q.new_empty(q.shape)


def get_attn_precision(attn_precision: torch.dtype, current_dtype: torch.dtype) -> torch.dtype:
    memory_management.force_upcast_attention_dtype().get(current_dtype, attn_precision)


def exists(val) -> bool:
    return val is not None


if memory_management.is_nvidia():
    SDP_BATCH_LIMIT = 2**15
else:
    SDP_BATCH_LIMIT = 2**31


# region Attentions


def attention_basic(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    attn_precision = get_attn_precision(attn_precision, q.dtype)

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
        sim = einsum("b i d, b j d -> b i j", q.float(), k.float()) * scale
    else:
        sim = einsum("b i d, b j d -> b i j", q, k) * scale

    del q, k

    if exists(mask):
        if mask.dtype == torch.bool:
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)
        else:
            if len(mask.shape) == 2:
                bs = 1
            else:
                bs = mask.shape[0]
            mask = mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1]).expand(b, heads, -1, -1).reshape(-1, mask.shape[-2], mask.shape[-1])
            sim.add_(mask)

    sim = sim.softmax(dim=-1)

    out = einsum("b i j, b j d -> b i d", sim.to(v.dtype), v)

    if skip_output_reshape:
        out = out.unsqueeze(0).reshape(b, heads, -1, dim_head)
    else:
        out = out.unsqueeze(0).reshape(b, heads, -1, dim_head).permute(0, 2, 1, 3).reshape(b, -1, heads * dim_head)

    return out


def attention_xformers(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    b = q.shape[0]
    dim_head = q.shape[-1]

    if torch.jit.is_tracing() or torch.jit.is_scripting():
        return attention_pytorch(q, k, v, heads, mask, skip_reshape=skip_reshape, **kwargs)

    if skip_reshape:
        q, k, v = map(
            lambda t: t.permute(0, 2, 1, 3),
            (q, k, v),
        )
    else:
        dim_head //= heads
        q, k, v = map(
            lambda t: t.reshape(b, -1, heads, dim_head),
            (q, k, v),
        )

    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        pad = 8 - mask.shape[-1] % 8
        mask_out = torch.empty([mask.shape[0], mask.shape[1], q.shape[1], mask.shape[-1] + pad], dtype=q.dtype, device=q.device)
        mask_out[..., : mask.shape[-1]] = mask
        mask = mask_out[..., : mask.shape[-1]]
        mask = mask.expand(b, heads, -1, -1)

    try:
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask)
        _fallback = False
    except Exception as e:
        if "(too new)" in str(e):
            logger.warning("xformers does not work on RTX 50s")
        else:
            logger.error(f"Error running xformers: {e}")
        _fallback = True

    if _fallback:
        if not skip_reshape:
            q, k, v = map(
                lambda t: t.transpose(1, 2),
                (q, k, v),
            )
        return attention_pytorch(q, k, v, heads, mask=mask, skip_reshape=True, **kwargs)

    if skip_output_reshape:
        out = out.permute(0, 2, 1, 3)
    else:
        out = out.reshape(b, -1, heads * dim_head)

    return out


def attention_pytorch(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
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
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    if SDP_BATCH_LIMIT >= b:
        out = operations.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        if not skip_output_reshape:
            out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    else:
        out = torch.empty((b, q.shape[2], heads * dim_head), dtype=q.dtype, layout=q.layout, device=q.device)
        for i in range(0, b, SDP_BATCH_LIMIT):
            m = mask
            if mask is not None:
                if mask.shape[0] > 1:
                    m = mask[i : i + SDP_BATCH_LIMIT]

            out[i : i + SDP_BATCH_LIMIT] = operations.scaled_dot_product_attention(q[i : i + SDP_BATCH_LIMIT], k[i : i + SDP_BATCH_LIMIT], v[i : i + SDP_BATCH_LIMIT], attn_mask=m, dropout_p=0.0, is_causal=False).transpose(1, 2).reshape(-1, q.shape[2], heads * dim_head)

    return out


def attention_sage(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    if skip_reshape:
        b, _, _, dim_head = q.shape
        tensor_layout = "HND"
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head),
            (q, k, v),
        )
        tensor_layout = "NHD"

    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    _fallback: bool = (IS_SAGE_2 and dim_head > 128) or ((not IS_SAGE_2) and (dim_head not in (64, 96, 128)))

    try:
        if not _fallback:
            out = sageattn(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout)
    except Exception as e:
        logger.error(f"Error running sageattn: {e}")
        _fallback = True

    if _fallback:
        if tensor_layout == "NHD":
            q, k, v = map(
                lambda t: t.transpose(1, 2),
                (q, k, v),
            )
        return attention_pytorch(q, k, v, heads, mask=mask, skip_reshape=True, skip_output_reshape=skip_output_reshape, **kwargs)

    if tensor_layout == "HND":
        if not skip_output_reshape:
            out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    else:
        if skip_output_reshape:
            out = out.transpose(1, 2)
        else:
            out = out.reshape(b, -1, heads * dim_head)

    return out


def attention_flash(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
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
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
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
        _fallback = False
    except Exception as e:
        logger.error(f"Error running flash_attn: {e}")
        _fallback = True

    if _fallback:
        out = operations.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)

    if not skip_output_reshape:
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)

    return out


if memory_management.sage_enabled():
    attention_function = attention_sage
    if not IS_SAGE_2:
        logger.info("Using SageAttention")
    else:
        match args.sage2_function:
            case SageAttentionFuncs.auto:
                logger.info("Using SageAttention 2")
            case SageAttentionFuncs.fp16_triton:
                logger.info("Using SageAttention (fp16 Triton)")
            case SageAttentionFuncs.fp16_cuda:
                logger.info("Using SageAttention (fp16 CUDA)")
            case SageAttentionFuncs.fp8_cuda:
                logger.info("Using SageAttention (fp8 CUDA)")

elif memory_management.flash_enabled():
    logger.info("Using FlashAttention")
    attention_function = attention_flash
elif memory_management.xformers_enabled():
    logger.info("Using xformers Cross Attention")
    attention_function = attention_xformers
elif memory_management.pytorch_attention_enabled():
    logger.info("Using PyTorch Cross Attention")
    attention_function = attention_pytorch
else:
    logger.info("Using Basic Cross Attention")
    attention_function = attention_basic


# region VAE


def slice_attention_vae(q, k, v):
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
            logger.warning(f"Out of Memory Error; trying again... ({steps})")

    return r1


def normal_attention_vae(q, k, v):
    orig_shape = q.shape
    b = orig_shape[0]
    c = orig_shape[1]

    q = q.reshape(b, c, -1)
    q = q.permute(0, 2, 1)
    k = k.reshape(b, c, -1)
    v = v.reshape(b, c, -1)

    r1 = slice_attention_vae(q, k, v)
    h_ = r1.reshape(orig_shape)
    del r1
    return h_


def xformers_attention_vae(q, k, v):
    orig_shape = q.shape
    B = orig_shape[0]
    C = orig_shape[1]
    q, k, v = map(
        lambda t: t.view(B, C, -1).transpose(1, 2).contiguous(),
        (q, k, v),
    )

    try:
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)
        out = out.transpose(1, 2).reshape(orig_shape)
        _fallback = False
    except Exception:
        _fallback = True

    if _fallback:
        out = slice_attention_vae(q.view(B, -1, C), k.view(B, -1, C).transpose(1, 2), v.view(B, -1, C).transpose(1, 2)).reshape(orig_shape)

    return out


def pytorch_attention_vae(q, k, v):
    orig_shape = q.shape
    B = orig_shape[0]
    C = orig_shape[1]
    q, k, v = map(
        lambda t: t.view(B, 1, C, -1).transpose(2, 3).contiguous(),
        (q, k, v),
    )

    try:
        out = operations.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        out = out.transpose(2, 3).reshape(orig_shape)
        _fallback = False
    except memory_management.OOM_EXCEPTION:
        logger.warning("Out of Memory Error; retrying with Slice Attention")
        _fallback = True

    if _fallback:
        out = slice_attention_vae(q.view(B, -1, C), k.view(B, -1, C).transpose(1, 2), v.view(B, -1, C).transpose(1, 2)).reshape(orig_shape)

    return out


if memory_management.xformers_enabled_vae():
    logger.info("Using xformers Attention for VAE")
    attention_function_vae = xformers_attention_vae
elif memory_management.pytorch_attention_enabled():
    logger.info("Using PyTorch Attention for VAE")
    attention_function_vae = pytorch_attention_vae
else:
    logger.info("Using Slice Attention for VAE")
    attention_function_vae = normal_attention_vae
