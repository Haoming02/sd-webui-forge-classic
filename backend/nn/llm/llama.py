# https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/text_encoders/llama.py

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.memory_management import pytorch_attention_enabled

if pytorch_attention_enabled():
    from backend.attention import attention_pytorch as attention_function
else:
    from backend.attention import attention_basic as attention_function

from backend.nn.anima import LLMAdapter
from backend.nn.llm import qwen_vl
from backend.quant_ops import ck


@dataclass
class Qwen3_06BConfig:
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = False
    rope_dims = None
    q_norm = "gemma3"
    k_norm = "gemma3"
    rope_scale = None
    final_norm: bool = True


@dataclass
class Qwen3_4BConfig:
    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = False
    rope_dims = None
    q_norm = "gemma3"
    k_norm = "gemma3"
    rope_scale = None
    final_norm: bool = True


@dataclass
class Qwen3_8BConfig:
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = False
    rope_dims = None
    q_norm = "gemma3"
    k_norm = "gemma3"
    rope_scale = None
    final_norm: bool = True


@dataclass
class Qwen3VL_8BConfig(Qwen3_8BConfig):
    max_position_embeddings: int = 262144
    rope_theta: float = 5000000.0
    rope_dims = [24, 20, 20]
    interleaved_mrope: bool = True
    lm_head: bool = True


@dataclass
class Qwen3VL_4BConfig(Qwen3VL_8BConfig):
    hidden_size: int = 2560
    intermediate_size: int = 9728
    lm_head: bool = False


@dataclass
class Qwen25_7BVLI_Config:
    vocab_size: int = 152064
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    max_position_embeddings: int = 128000
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = True
    rope_dims = [16, 24, 24]
    q_norm = None
    k_norm = None
    rope_scale = None
    final_norm: bool = True


@dataclass
class Gemma2_2B_Config:
    vocab_size: int = 256000
    hidden_size: int = 2304
    intermediate_size: int = 9216
    num_hidden_layers: int = 26
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    transformer_type: str = "gemma2"
    head_dim = 256
    rms_norm_add = True
    mlp_activation = "gelu_pytorch_tanh"
    qkv_bias = False
    rope_dims = None
    q_norm = None
    k_norm = None
    sliding_attention = None
    rope_scale = None
    final_norm: bool = True


@dataclass
class Ministral3_3BConfig:
    vocab_size: int = 131072
    hidden_size: int = 3072
    intermediate_size: int = 9216
    num_hidden_layers: int = 26
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = False
    rope_dims = None
    q_norm = None
    k_norm = None
    rope_scale = None
    final_norm: bool = True


def precompute_freqs_cis(head_dim, position_ids, theta, rope_scale=None, rope_dims=None, device=None, interleaved_mrope=False):
    if not isinstance(theta, list):
        theta = [theta]

    out = []
    for index, t in enumerate(theta):
        theta_numerator = torch.arange(0, head_dim, 2, device=device).float()
        inv_freq = 1.0 / (t ** (theta_numerator / head_dim))

        if rope_scale is not None:
            if isinstance(rope_scale, list):
                inv_freq /= rope_scale[index]
            else:
                inv_freq /= rope_scale

        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        if rope_dims is not None and position_ids.shape[0] > 1 and interleaved_mrope:
            freqs_inter = freqs[0].clone()
            for axis_idx, offset in ((1, 1), (2, 2)):
                length = rope_dims[axis_idx] * 3
                idx = slice(offset, length, 3)
                freqs_inter[..., idx] = freqs[axis_idx, ..., idx]
            emb = torch.cat((freqs_inter, freqs_inter), dim=-1)
            cos = emb.cos().unsqueeze(0)
            sin = emb.sin().unsqueeze(0)
        else:
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            if rope_dims is not None and position_ids.shape[0] > 1:
                mrope_section = rope_dims * 2
                cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(0)
                sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(0)
            else:
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
        sin_split = sin.shape[-1] // 2
        out.append((cos, sin[..., :sin_split], -sin[..., sin_split:]))

    if len(out) == 1:
        return out[0]

    return out


def rope_matrix(freqs_cis):
    if torch.is_tensor(freqs_cis):
        return freqs_cis
    cos, sin, neg_sin = freqs_cis
    half = sin.shape[-1]
    matrix = torch.stack((cos[..., :half], neg_sin, sin, cos[..., half:]), dim=-1)
    return matrix.reshape(*matrix.shape[:-1], 2, 2)


def apply_rope(xq, xk, freqs_cis):
    matrix = rope_matrix(freqs_cis)
    if matrix.ndim == 5:
        matrix = matrix.unsqueeze(0)

    q_ndim, k_ndim = xq.ndim, xk.ndim
    if q_ndim == 3:
        xq = xq.unsqueeze(0)
    if k_ndim == 3:
        xk = xk.unsqueeze(0)

    xq, xk = ck.apply_rope_split_half(xq, xk, matrix)
    if q_ndim == 3:
        xq = xq.squeeze(0)
    if k_ndim == 3:
        xk = xk.squeeze(0)

    return xq, xk


class Attention(nn.Module):
    def __init__(self, config: Qwen3_06BConfig | Qwen3_4BConfig | Qwen3_8BConfig | Qwen25_7BVLI_Config | Gemma2_2B_Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size

        self.head_dim = config.head_dim
        self.inner_size = self.num_heads * self.head_dim

        self.q_proj = nn.Linear(config.hidden_size, self.inner_size, bias=config.qkv_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias)
        self.o_proj = nn.Linear(self.inner_size, config.hidden_size, bias=False)

        if config.q_norm == "gemma3":
            self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps, add=config.rms_norm_add)
        else:
            self.q_norm = None

        if config.k_norm == "gemma3":
            self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps, add=config.rms_norm_add)
        else:
            self.k_norm = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        optimized_attention=None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        batch_size, seq_length, _ = hidden_states.shape
        xq = self.q_proj(hidden_states)
        xk = self.k_proj(hidden_states)
        xv = self.v_proj(hidden_states)

        xq = xq.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if self.k_norm is not None:
            xk = self.k_norm(xk)

        xq, xk = apply_rope(xq, xk, freqs_cis=freqs_cis)

        present_key_value = None
        if past_key_value is not None:
            index = 0
            num_tokens = xk.shape[2]
            if len(past_key_value) > 0:
                past_key, past_value, index = past_key_value
                if past_key.shape[2] >= (index + num_tokens):
                    past_key[:, :, index : index + xk.shape[2]] = xk
                    past_value[:, :, index : index + xv.shape[2]] = xv
                    xk = past_key[:, :, : index + xk.shape[2]]
                    xv = past_value[:, :, : index + xv.shape[2]]
                    present_key_value = (past_key, past_value, index + num_tokens)
                else:
                    xk = torch.cat((past_key[:, :, :index], xk), dim=2)
                    xv = torch.cat((past_value[:, :, :index], xv), dim=2)
                    present_key_value = (xk, xv, index + num_tokens)
            else:
                present_key_value = (xk, xv, index + num_tokens)

        gqa_kwargs = {"enable_gqa": True} if self.num_heads != self.num_kv_heads else {}
        output = optimized_attention(xq, xk, xv, self.num_heads, mask=attention_mask, skip_reshape=True, **gqa_kwargs)
        return self.o_proj(output), present_key_value


class MLP(nn.Module):
    def __init__(self, config: Qwen3_06BConfig | Qwen3_4BConfig | Qwen3_8BConfig | Qwen25_7BVLI_Config | Gemma2_2B_Config):
        super().__init__()

        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        if config.mlp_activation == "silu":
            self.activation = nn.functional.silu
        elif config.mlp_activation == "gelu_pytorch_tanh":
            self.activation = lambda a: nn.functional.gelu(a, approximate="tanh")

    def forward(self, x):
        return self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: Qwen3_06BConfig | Qwen3_4BConfig | Qwen3_8BConfig | Qwen25_7BVLI_Config | Gemma2_2B_Config, index):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        optimized_attention=None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        # Self Attention
        residual = x
        x = self.input_layernorm(x)
        x, present_key_value = self.self_attn(
            hidden_states=x,
            attention_mask=attention_mask,
            freqs_cis=freqs_cis,
            optimized_attention=optimized_attention,
            past_key_value=past_key_value,
        )
        x = residual + x

        # MLP
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x, present_key_value


class TransformerBlockGemma2(nn.Module):
    def __init__(self, config: Qwen3_06BConfig | Qwen3_4BConfig | Qwen3_8BConfig | Qwen25_7BVLI_Config | Gemma2_2B_Config, index):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add)
        self.pre_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add)
        self.post_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add)

        if config.sliding_attention is not None:
            self.sliding_attention = config.sliding_attention[index % len(config.sliding_attention)]
        else:
            self.sliding_attention = False

        self.transformer_type = config.transformer_type

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        optimized_attention=None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        if self.transformer_type == "gemma3":
            if self.sliding_attention:
                if x.shape[1] > self.sliding_attention:
                    sliding_mask = torch.full((x.shape[1], x.shape[1]), float("-inf"), device=x.device, dtype=x.dtype)
                    sliding_mask.tril_(diagonal=-self.sliding_attention)
                    if attention_mask is not None:
                        attention_mask = attention_mask + sliding_mask
                    else:
                        attention_mask = sliding_mask
                freqs_cis = freqs_cis[1]
            else:
                freqs_cis = freqs_cis[0]

        # Self Attention
        residual = x
        x = self.input_layernorm(x)
        x, present_key_value = self.self_attn(
            hidden_states=x,
            attention_mask=attention_mask,
            freqs_cis=freqs_cis,
            optimized_attention=optimized_attention,
            past_key_value=past_key_value,
        )

        x = self.post_attention_layernorm(x)
        x = residual + x

        # MLP
        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x

        return x, present_key_value


class Llama2_(nn.Module):
    def __init__(self, config: Qwen3_06BConfig | Qwen3_4BConfig | Qwen3_8BConfig | Qwen25_7BVLI_Config | Gemma2_2B_Config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        if self.config.transformer_type == "gemma2" or self.config.transformer_type == "gemma3":
            transformer = TransformerBlockGemma2
            self.normalize_in = True
        else:
            transformer = TransformerBlock
            self.normalize_in = False

        self.layers = nn.ModuleList([transformer(config, index=i) for i in range(config.num_hidden_layers)])

        if config.final_norm:
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add)
        else:
            self.norm = None

        if getattr(config, "lm_head", False):
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, x, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None, final_layer_norm_intermediate=True, dtype=None, position_ids=None, embeds_info=[], input_ids=None, deepstack_embeds=None, visual_pos_masks=None, **kwargs):
        if embeds is not None:
            x = embeds
        else:
            x = self.embed_tokens(x).to(dtype=dtype)

        if self.normalize_in:
            x *= self.config.hidden_size**0.5

        seq_len = x.shape[1]
        past_len = 0

        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + seq_len, device=x.device).unsqueeze(0)

        freqs_cis = precompute_freqs_cis(self.config.head_dim, position_ids, self.config.rope_theta, self.config.rope_scale, self.config.rope_dims, interleaved_mrope=getattr(self.config, "interleaved_mrope", False), device=x.device)

        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])).expand(attention_mask.shape[0], 1, seq_len, attention_mask.shape[-1])
            mask = mask.masked_fill(mask.to(torch.bool), torch.finfo(x.dtype).min / 4)

        if seq_len > 1:
            causal_mask = torch.empty(past_len + seq_len, past_len + seq_len, dtype=x.dtype, device=x.device).fill_(torch.finfo(x.dtype).min / 4).triu_(1)
            if mask is not None:
                mask += causal_mask
            else:
                mask = causal_mask

        intermediate = None
        all_intermediate = None
        only_layers = None
        if intermediate_output is not None:
            if isinstance(intermediate_output, list):
                all_intermediate = []
                only_layers = set(intermediate_output)
            elif intermediate_output == "all":
                all_intermediate = []
                intermediate_output = None
            elif intermediate_output < 0:
                intermediate_output = len(self.layers) + intermediate_output

        for i, layer in enumerate(self.layers):
            if all_intermediate is not None:
                if only_layers is None or (i in only_layers):
                    all_intermediate.append(x.unsqueeze(1).clone())

            x, _ = layer(
                x=x,
                attention_mask=mask,
                freqs_cis=freqs_cis,
                optimized_attention=attention_function,
                past_key_value=None,
            )

            if deepstack_embeds is not None and i < len(deepstack_embeds):
                x[visual_pos_masks] = x[visual_pos_masks] + deepstack_embeds[i].to(x)

            if i == intermediate_output:
                intermediate = x.clone()

        if self.norm is not None:
            x = self.norm(x)

        if all_intermediate is not None:
            if only_layers is None or ((i + 1) in only_layers):
                all_intermediate.append(x.unsqueeze(1).clone())

        if all_intermediate is not None:
            intermediate = torch.cat(all_intermediate, dim=1)

        if intermediate is not None and final_layer_norm_intermediate and self.norm is not None:
            intermediate = self.norm(intermediate)

        return x, intermediate


class BaseLlama:
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, embeddings):
        self.model.embed_tokens = embeddings

    def forward(self, input_ids, *args, **kwargs):
        return self.model(input_ids, *args, **kwargs)


class Qwen3_06B(BaseLlama, nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        config = Qwen3_06BConfig()

        _config_dict = asdict(config)
        for key, value in _config_dict.items():
            if key in config_dict:
                assert value == config_dict[key]

        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config)

        self.llm_adapter = LLMAdapter()

    def preprocess_text_embeds(self, text_embeds, text_ids):
        if text_ids is not None:
            return self.llm_adapter(text_embeds, text_ids)
        else:
            return text_embeds


class Qwen3_4B(BaseLlama, nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        config = Qwen3_4BConfig()

        _config_dict = asdict(config)
        for key, value in _config_dict.items():
            if key in config_dict:
                assert value == config_dict[key]

        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config)


class Qwen3_8B(BaseLlama, nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        config = Qwen3_8BConfig()

        _config_dict = asdict(config)
        for key, value in _config_dict.items():
            if key in config_dict:
                assert value == config_dict[key]

        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config)


class Qwen25_7BVLI(BaseLlama, nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        config = Qwen25_7BVLI_Config()

        _config_dict = asdict(config)
        for key, value in _config_dict.items():
            if key in config_dict:
                assert value == config_dict[key]

        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config)
        self.visual = qwen_vl.Qwen2VLVisionTransformer(hidden_size=1280, output_hidden_size=config.hidden_size)

    def preprocess_embed(self, embed, device):
        if embed["type"] == "image":
            image, grid = qwen_vl.process_qwen2vl_images(embed["data"])
            return self.visual(image.to(device, dtype=torch.float32), grid), grid
        return None, None

    def forward(self, x, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None, final_layer_norm_intermediate=True, dtype=None, embeds_info=[]):
        grid = None
        position_ids = None
        offset = 0
        for e in embeds_info:
            if e.get("type") == "image":
                grid = e.get("extra", None)
                start = e.get("index")
                if position_ids is None:
                    position_ids = torch.zeros((3, embeds.shape[1]), device=embeds.device)
                    position_ids[:, :start] = torch.arange(0, start, device=embeds.device)
                end = e.get("size") + start
                len_max = int(grid.max()) // 2
                start_next = len_max + start
                position_ids[:, end:] = torch.arange(start_next + offset, start_next + (embeds.shape[1] - end) + offset, device=embeds.device)
                position_ids[0, start:end] = start + offset
                max_d = int(grid[0][1]) // 2
                position_ids[1, start:end] = torch.arange(start + offset, start + max_d + offset, device=embeds.device).unsqueeze(1).repeat(1, math.ceil((end - start) / max_d)).flatten(0)[: end - start]
                max_d = int(grid[0][2]) // 2
                position_ids[2, start:end] = torch.arange(start + offset, start + max_d + offset, device=embeds.device).unsqueeze(0).repeat(math.ceil((end - start) / max_d), 1).flatten(0)[: end - start]
                offset += len_max - (end - start)

        if grid is None:
            position_ids = None

        return super().forward(x, attention_mask=attention_mask, embeds=embeds, num_tokens=num_tokens, intermediate_output=intermediate_output, final_layer_norm_intermediate=final_layer_norm_intermediate, dtype=dtype, position_ids=position_ids)


class Gemma2_2B(BaseLlama, nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        config = Gemma2_2B_Config()

        _config_dict = asdict(config)
        for key, value in _config_dict.items():
            if key in config_dict:
                assert value == config_dict[key]

        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config)


class Ministral3_3B(BaseLlama, nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        config = Ministral3_3BConfig()

        _config_dict = asdict(config)
        for key, value in _config_dict.items():
            if key in config_dict:
                assert value == config_dict[key]

        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config)


from backend.nn.llm.qwen35 import (
    QWEN3VL_VISION,
    QWEN3VL_VISION_COMMON,
    Qwen3VLVisionModel,
)


class Qwen3VL(BaseLlama, nn.Module):
    def __init__(self, config_dict, *, model_type: str):
        super().__init__()
        config = {"qwen3vl_4b": Qwen3VL_4BConfig, "qwen3vl_8b": Qwen3VL_8BConfig}[model_type]()

        _config_dict = asdict(config)
        for key, value in _config_dict.items():
            if key in config_dict:
                assert value == config_dict[key]

        self.num_layers = config.num_hidden_layers
        self.model = Llama2_(config)
        vision_config = {**QWEN3VL_VISION_COMMON, **QWEN3VL_VISION[model_type], "out_hidden_size": config.hidden_size}
        self.visual = Qwen3VLVisionModel(vision_config)

    def preprocess_embed(self, embed, device):
        if embed["type"] == "image":
            image, grid = qwen_vl.process_qwen2vl_images(embed["data"], patch_size=16, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
            merged, deepstack = self.visual(image.to(device, dtype=torch.float32), grid)
            return merged, {"grid": grid, "deepstack": deepstack}
        return None, None

    def build_image_inputs(self, embeds, embeds_info):
        images = sorted([e for e in embeds_info if e.get("type") == "image"], key=lambda e: e["index"])
        if len(images) == 0:
            return None, None, None

        device = embeds.device
        seq = embeds.shape[1]
        position_ids = qwen_vl.qwen2vl_mrope_position_ids(embeds_info, seq, device)

        visual_pos_masks = torch.zeros((1, seq), dtype=torch.bool, device=device)
        deepstack = None
        for e in images:
            start = e["index"]
            end = e["size"] + start
            visual_pos_masks[0, start:end] = True
            ds = e["extra"]["deepstack"]
            if deepstack is None:
                deepstack = [d for d in ds]
            else:
                deepstack = [torch.cat([deepstack[i], ds[i]], dim=0) for i in range(len(ds))]
        return position_ids, visual_pos_masks, deepstack

    def forward(self, input_ids, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None, final_layer_norm_intermediate=True, dtype=None, embeds_info=[], **kwargs):
        position_ids = kwargs.pop("position_ids", None)
        visual_pos_masks = kwargs.pop("visual_pos_masks", None)
        deepstack_embeds = kwargs.pop("deepstack_embeds", None)
        if embeds is not None and position_ids is None:
            position_ids, visual_pos_masks, deepstack_embeds = self.build_image_inputs(embeds, embeds_info)
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            embeds=embeds,
            num_tokens=num_tokens,
            intermediate_output=intermediate_output,
            final_layer_norm_intermediate=final_layer_norm_intermediate,
            dtype=dtype,
            position_ids=position_ids,
            embeds_info=embeds_info,
            visual_pos_masks=visual_pos_masks,
            deepstack_embeds=deepstack_embeds,
            **kwargs,
        )


def _qwen35_layer_types(n):
    return [("full_attention" if (i + 1) % 4 == 0 else "linear_attention") for i in range(n)]


@dataclass
class Qwen35_9BConfig:
    vocab_size: int = 248320
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    partial_rotary_factor: float = 0.25
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    conv_kernel_size: int = 4
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    mrope_section: list = field(default_factory=lambda: [11, 11, 10])
    layer_types: list = field(default_factory=lambda: _qwen35_layer_types(32))
    rms_norm_add: bool = True
    mlp_activation: str = "silu"
    qkv_bias: bool = False
    final_norm: bool = True
    lm_head: bool = False
    transformer_type: str = "qwen35_9b"
    head_dim = 256
    rope_dims = None
    rope_scale = None


def torch_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False):
    initial_dtype = query.dtype
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen35_9BConfig):
        super().__init__()

        hidden = config.hidden_size
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.conv_kernel_size = config.conv_kernel_size

        key_dim = self.num_key_heads * self.key_head_dim
        value_dim = self.num_value_heads * self.value_head_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        conv_dim = key_dim * 2 + value_dim

        self.in_proj_qkv = nn.Linear(hidden, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden, value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden, self.num_value_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden, self.num_value_heads, bias=False)
        self.out_proj = nn.Linear(value_dim, hidden, bias=False)

        self.dt_bias = nn.Parameter(torch.empty(self.num_value_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_value_heads))

        self.conv1d = nn.Conv1d(in_channels=conv_dim, out_channels=conv_dim, bias=False, kernel_size=self.conv_kernel_size, groups=conv_dim, padding=self.conv_kernel_size - 1)

        self.norm = nn.RMSNorm(self.value_head_dim, eps=config.rms_norm_eps)

    def forward(self, x, **kwargs):
        batch_size, seq_len, _ = x.shape

        proj = self.in_proj_qkv(x)
        z = self.in_proj_z(x)
        mixed_qkv = proj.transpose(1, 2)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = mixed_qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        g_decay = -self.A_log.to(x.device, torch.float32).exp()
        dt_bias = self.dt_bias.to(x.device, torch.float32)

        beta = b.sigmoid()
        g = g_decay * F.softplus(a.float() + dt_bias)
        query = query.reshape(batch_size, seq_len, self.num_key_heads, self.key_head_dim)
        key = key.reshape(batch_size, seq_len, self.num_key_heads, self.key_head_dim)
        value = value.reshape(batch_size, seq_len, self.num_value_heads, self.value_head_dim)
        if self.num_value_heads != self.num_key_heads:
            rep = self.num_value_heads // self.num_key_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        core_attn_out, _ = torch_chunk_gated_delta_rule(query, key, value, g=g, beta=beta, initial_state=None, output_final_state=False)

        core_attn_out = self.norm(core_attn_out.reshape(-1, self.value_head_dim)) * F.silu(z.reshape(-1, self.value_head_dim))
        output = self.out_proj(core_attn_out.reshape(batch_size, seq_len, -1))
        return output, None


def apply_partial_rope(xq, xk, freqs_cis, rotary_dim):
    xq_rot = xq[..., :rotary_dim]
    xq_pass = xq[..., rotary_dim:]
    xk_rot = xk[..., :rotary_dim]
    xk_pass = xk[..., rotary_dim:]

    xq_rot, xk_rot = apply_rope(xq_rot, xk_rot, freqs_cis)

    xq = torch.cat([xq_rot, xq_pass], dim=-1)
    xk = torch.cat([xk_rot, xk_pass], dim=-1)
    return xq, xk


class GatedAttention(nn.Module):
    def __init__(self, config: Qwen35_9BConfig):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.inner_size = self.num_heads * self.head_dim
        self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)

        self.q_proj = nn.Linear(config.hidden_size, self.inner_size * 2, bias=config.qkv_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias)
        self.o_proj = nn.Linear(self.inner_size, config.hidden_size, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps, add=config.rms_norm_add)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps, add=config.rms_norm_add)

    def forward(self, x, attention_mask=None, freqs_cis=None, optimized_attention=None, **kwargs):
        batch_size, seq_length, _ = x.shape

        qg = self.q_proj(x)
        qg = qg.view(batch_size, seq_length, self.num_heads, self.head_dim * 2)
        xq, gate = qg[..., :self.head_dim], qg[..., self.head_dim:]
        gate = gate.reshape(batch_size, seq_length, -1)

        xk = self.k_proj(x)
        xv = self.v_proj(x)

        xq = self.q_norm(xq).transpose(1, 2)
        xk = self.k_norm(xk.view(batch_size, seq_length, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        xv = xv.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)

        xq, xk = apply_partial_rope(xq, xk, freqs_cis, self.rotary_dim)

        gqa_kwargs = {"enable_gqa": True} if self.num_heads != self.num_kv_heads else {}
        output = optimized_attention(xq, xk, xv, self.num_heads, mask=attention_mask, skip_reshape=True, **gqa_kwargs)

        output = output * gate.sigmoid()
        return self.o_proj(output), None


class Qwen35TransformerBlock(nn.Module):
    def __init__(self, config: Qwen35_9BConfig, index):
        super().__init__()
        self.layer_type = config.layer_types[index]
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config)
        else:
            self.self_attn = GatedAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add)

    def forward(self, x, attention_mask=None, freqs_cis=None, optimized_attention=None, **kwargs):
        if self.layer_type == "linear_attention":
            h, _ = self.linear_attn(self.input_layernorm(x), attention_mask=attention_mask)
        else:
            h, _ = self.self_attn(self.input_layernorm(x), attention_mask=attention_mask, freqs_cis=freqs_cis, optimized_attention=optimized_attention)

        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, None


class Qwen35Transformer(nn.Module):
    def __init__(self, config: Qwen35_9BConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen35TransformerBlock(config, index=i) for i in range(config.num_hidden_layers)])

        if config.final_norm:
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add)
        else:
            self.norm = None

        if config.lm_head:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, x, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None, final_layer_norm_intermediate=True, dtype=None, position_ids=None, embeds_info=[], **kwargs):
        if embeds is not None:
            x = embeds
        else:
            x = self.embed_tokens(x).to(dtype=dtype)

        seq_len = x.shape[1]

        if position_ids is None:
            position_ids = torch.arange(0, seq_len, device=x.device).unsqueeze(0)

        rotary_dim = int(self.config.head_dim * self.config.partial_rotary_factor)
        freqs_cis = precompute_freqs_cis(rotary_dim, position_ids, self.config.rope_theta, self.config.rope_scale, self.config.mrope_section, interleaved_mrope=True, device=x.device)

        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])).expand(attention_mask.shape[0], 1, seq_len, attention_mask.shape[-1])
            mask = mask.masked_fill(mask.to(torch.bool), torch.finfo(x.dtype).min / 4)

        if seq_len > 1:
            causal_mask = torch.empty(seq_len, seq_len, dtype=x.dtype, device=x.device).fill_(torch.finfo(x.dtype).min / 4).triu_(1)
            if mask is not None:
                mask += causal_mask
            else:
                mask = causal_mask

        intermediate = None
        all_intermediate = None
        only_layers = None
        if intermediate_output is not None:
            if isinstance(intermediate_output, list):
                all_intermediate = []
                only_layers = set(intermediate_output)
            elif intermediate_output == "all":
                all_intermediate = []
                intermediate_output = None
            elif intermediate_output < 0:
                intermediate_output = len(self.layers) + intermediate_output

        for i, layer in enumerate(self.layers):
            if all_intermediate is not None:
                if only_layers is None or (i in only_layers):
                    all_intermediate.append(x.unsqueeze(1).clone())

            x, _ = layer(x=x, attention_mask=mask, freqs_cis=freqs_cis, optimized_attention=attention_function)

            if i == intermediate_output:
                intermediate = x.clone()

        if self.norm is not None:
            x = self.norm(x)

        if all_intermediate is not None:
            all_intermediate.append(x.unsqueeze(1).clone())
            intermediate = torch.cat(all_intermediate, dim=1)

        if intermediate is not None and final_layer_norm_intermediate and self.norm is not None:
            intermediate = self.norm(intermediate)

        return x, intermediate


class Qwen35_9B(BaseLlama, nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        config = Qwen35_9BConfig()

        _config_dict = asdict(config)
        for key, value in _config_dict.items():
            if key in config_dict:
                assert value == config_dict[key]

        self.num_layers = config.num_hidden_layers
        self.model = Qwen35Transformer(config)
