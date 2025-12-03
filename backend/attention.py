import math

import einops
import torch

from backend import memory_management
from backend.args import SageAttentionFuncs, args
from modules.errors import display_once

if memory_management.xformers_enabled() or args.force_xformers_vae:
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

if memory_management.flash_enabled():
    from flash_attn import flash_attn_func

    @torch.library.custom_op("flash_attention::flash_attn", mutates_args=())
    def flash_attn_wrapper(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout_p: float = 0.0, causal: bool = False) -> torch.Tensor:
        return flash_attn_func(q, k, v, dropout_p=dropout_p, causal=causal)

    @flash_attn_wrapper.register_fake
    def flash_attn_fake(q, k, v, dropout_p=0.0, causal=False):
        return q.new_empty(q.shape)


def get_xformers_flash_attention_op(q, k, v):
    try:
        flash_attention_op = xformers.ops.MemoryEfficientAttentionFlashAttentionOp
        fw, bw = flash_attention_op
        if fw.supports(xformers.ops.fmha.Inputs(query=q, key=k, value=v, attn_bias=None)):
            return flash_attention_op
    except Exception as e:
        display_once(e, "get_xformers_flash_attention_op")

    return None


FORCE_UPCAST_ATTENTION_DTYPE = memory_management.force_upcast_attention_dtype()


def get_attn_precision(attn_precision, current_dtype):
    if args.disable_attention_upcast:
        return None
    if FORCE_UPCAST_ATTENTION_DTYPE is not None:
        return FORCE_UPCAST_ATTENTION_DTYPE.get(current_dtype, attn_precision)
    return attn_precision


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d


if memory_management.is_nvidia():
    SDP_BATCH_LIMIT = 2**15
else:
    SDP_BATCH_LIMIT = 2**31


# ========== Diffusion ========== #


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

    if skip_output_reshape:
        return out.unsqueeze(0).reshape(b, heads, -1, dim_head)
    else:
        return out.unsqueeze(0).reshape(b, heads, -1, dim_head).permute(0, 2, 1, 3).reshape(b, -1, heads * dim_head)


def attention_split(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    attn_precision = get_attn_precision(attn_precision, q.dtype)

    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads

    scale = dim_head**-0.5

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

    if steps > 64:
        max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
        raise RuntimeError(f"Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). " f"Need: {mem_required / 64 / gb:0.1f}GB free, Have:{mem_free_total / gb:0.1f}GB free")

    if mask is not None:
        if len(mask.shape) == 2:
            bs = 1
        else:
            bs = mask.shape[0]
        mask = mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1]).expand(b, heads, -1, -1).reshape(-1, mask.shape[-2], mask.shape[-1])

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
                        if mask.shape[1] == 1:
                            s1 += mask
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
                    print(f"[Out of Memory Error] emptying cache and trying again...")
                    continue
                steps *= 2
                if steps > 64:
                    raise e
                print(f"[Out of Memory Error] increasing steps and trying again {steps}...")
            else:
                raise e

    del q, k, v

    if skip_output_reshape:
        return r1.unsqueeze(0).reshape(b, heads, -1, dim_head)
    else:
        return r1.unsqueeze(0).reshape(b, heads, -1, dim_head).permute(0, 2, 1, 3).reshape(b, -1, heads * dim_head)


def attention_xformers(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    b = q.shape[0]
    dim_head = q.shape[-1]
    disabled_xformers = False

    if BROKEN_XFORMERS and b * heads > 65535:
        disabled_xformers = True

    if not disabled_xformers:
        disabled_xformers = torch.jit.is_tracing() or torch.jit.is_scripting()

    if disabled_xformers:
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

    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask)

    if skip_output_reshape:
        return out.permute(0, 2, 1, 3)
    else:
        return out.reshape(b, -1, heads * dim_head)


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
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        if skip_output_reshape:
            return out
        else:
            return out.transpose(1, 2).reshape(b, -1, heads * dim_head)

    out = torch.empty((b, q.shape[2], heads * dim_head), dtype=q.dtype, layout=q.layout, device=q.device)

    for i in range(0, b, SDP_BATCH_LIMIT):
        m = mask
        if mask is not None:
            if mask.shape[0] > 1:
                m = mask[i : i + SDP_BATCH_LIMIT]

        out[i : i + SDP_BATCH_LIMIT] = (
            torch.nn.functional.scaled_dot_product_attention(
                q[i : i + SDP_BATCH_LIMIT],
                k[i : i + SDP_BATCH_LIMIT],
                v[i : i + SDP_BATCH_LIMIT],
                attn_mask=m,
                dropout_p=0.0,
                is_causal=False,
            )
            .transpose(1, 2)
            .reshape(-1, q.shape[2], heads * dim_head)
        )

    return out


if IS_SAGE_2 and args.sage2_function is not SageAttentionFuncs.auto:
    from functools import partial

    import sageattention

    _function = getattr(sageattention, f"sageattn_qk_int8_pv_{args.sage2_function.value}")
    if args.sage2_function is SageAttentionFuncs.fp16_triton:
        sageattn = partial(_function, quantization_backend=args.sage_quantization_backend.value)
    else:
        sageattn = partial(_function, qk_quant_gran=args.sage_quant_gran.value, pv_accum_dtype=args.sage_accum_dtype.value)


def attention_sage(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    if skip_reshape:
        b, _, _, dim_head = q.shape
        tensor_layout = "HND"
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        tensor_layout = "NHD"

    if (IS_SAGE_2 and dim_head > 128) or ((not IS_SAGE_2) and (dim_head not in (64, 96, 128))):
        if memory_management.xformers_enabled():
            return attention_xformers(q, k, v, heads, mask, attn_precision, skip_reshape, skip_output_reshape, **kwargs)
        else:
            return attention_pytorch(q, k, v, heads, mask, attn_precision, skip_reshape, skip_output_reshape, **kwargs)

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
            return attention_xformers(q, k, v, heads, mask=mask, skip_reshape=True, skip_output_reshape=skip_output_reshape, **kwargs)
        else:
            return attention_pytorch(q, k, v, heads, mask=mask, skip_reshape=True, skip_output_reshape=skip_output_reshape, **kwargs)

    if tensor_layout == "HND":
        if skip_output_reshape:
            return out
        else:
            return out.transpose(1, 2).reshape(b, -1, heads * dim_head)

    else:
        if skip_output_reshape:
            return out.transpose(1, 2)
        else:
            return out.reshape(b, -1, heads * dim_head)


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

    if skip_output_reshape:
        return out
    else:
        return out.transpose(1, 2).reshape(b, -1, heads * dim_head)


if memory_management.sage_enabled():
    attention_function = attention_sage
    match args.sage2_function:
        case SageAttentionFuncs.auto:
            print(f"Using SageAttention {'2' if IS_SAGE_2 else ''}")
        case SageAttentionFuncs.fp16_triton:
            print("Using SageAttention (fp16 Triton)")
        case SageAttentionFuncs.fp16_cuda:
            print("Using SageAttention (fp16 CUDA)")
        case SageAttentionFuncs.fp8_cuda:
            print("Using SageAttention (fp8 CUDA)")

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
    orig_shape = q.shape
    b = orig_shape[0]
    c = orig_shape[1]

    q = q.reshape(b, c, -1)
    q = q.permute(0, 2, 1)  # b,hw,c
    k = k.reshape(b, c, -1)  # b,c,hw
    v = v.reshape(b, c, -1)

    r1 = slice_attention_single_head_spatial(q, k, v)
    h_ = r1.reshape(orig_shape)
    del r1
    return h_


def xformers_attention_single_head_spatial(q, k, v):
    # compute attention
    orig_shape = q.shape
    B = orig_shape[0]
    C = orig_shape[1]
    q, k, v = map(
        lambda t: t.view(B, C, -1).transpose(1, 2).contiguous(),
        (q, k, v),
    )

    try:
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=get_xformers_flash_attention_op(q, k, v))
        out = out.transpose(1, 2).reshape(orig_shape)
    except NotImplementedError:
        out = slice_attention_single_head_spatial(q.view(B, -1, C), k.view(B, -1, C).transpose(1, 2), v.view(B, -1, C).transpose(1, 2)).reshape(orig_shape)
    return out


def pytorch_attention_single_head_spatial(q, k, v):
    # compute attention
    orig_shape = q.shape
    B = orig_shape[0]
    C = orig_shape[1]
    q, k, v = map(
        lambda t: t.view(B, 1, C, -1).transpose(2, 3).contiguous(),
        (q, k, v),
    )

    try:
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        out = out.transpose(2, 3).reshape(orig_shape)
    except memory_management.OOM_EXCEPTION as e:
        display_once(e, "pytorch_attention_single_head_spatial")
        out = slice_attention_single_head_spatial(q.view(B, -1, C), k.view(B, -1, C).transpose(1, 2), v.view(B, -1, C).transpose(1, 2)).reshape(orig_shape)
    return out


if memory_management.xformers_enabled_vae() or args.force_xformers_vae:
    print("Using xformers Attention for VAE")
    attention_function_single_head_spatial = xformers_attention_single_head_spatial
elif memory_management.pytorch_attention_enabled():
    print("Using PyTorch Attention for VAE")
    attention_function_single_head_spatial = pytorch_attention_single_head_spatial
else:
    print("Using Split Attention for VAE")
    attention_function_single_head_spatial = normal_attention_single_head_spatial
