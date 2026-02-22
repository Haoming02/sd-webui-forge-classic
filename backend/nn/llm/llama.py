# https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/text_encoders/llama.py

import math
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn

from backend.memory_management import pytorch_attention_enabled

if pytorch_attention_enabled:
    from backend.attention import attention_pytorch as attention_function
else:
    from backend.attention import attention_basic as attention_function

from backend.nn.anima import LLMAdapter
from backend.nn.llm import qwen_vl


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


def precompute_freqs_cis(head_dim, position_ids, theta, rope_scale=None, rope_dims=None, device=None):
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


def apply_rope(xq, xk, freqs_cis):
    org_dtype = xq.dtype
    cos = freqs_cis[0]
    sin = freqs_cis[1]
    nsin = freqs_cis[2]

    q_embed = xq * cos
    q_split = q_embed.shape[-1] // 2
    q_embed[..., :q_split].addcmul_(xq[..., q_split:], nsin)
    q_embed[..., q_split:].addcmul_(xq[..., :q_split], sin)

    k_embed = xk * cos
    k_split = k_embed.shape[-1] // 2
    k_embed[..., :k_split].addcmul_(xk[..., k_split:], nsin)
    k_embed[..., k_split:].addcmul_(xk[..., :k_split], sin)

    return q_embed.to(org_dtype), k_embed.to(org_dtype)


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

        xk = xk.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        xv = xv.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        output = optimized_attention(xq, xk, xv, self.num_heads, mask=attention_mask, skip_reshape=True)
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

    def forward(self, x, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None, final_layer_norm_intermediate=True, dtype=None, position_ids=None, embeds_info=[], past_key_values=None):
        if embeds is not None:
            x = embeds
        else:
            x = self.embed_tokens(x, out_dtype=dtype)

        if self.normalize_in:
            x *= self.config.hidden_size**0.5

        seq_len = x.shape[1]
        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0:
            past_len = past_key_values[0][2]

        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + seq_len, device=x.device).unsqueeze(0)

        freqs_cis = precompute_freqs_cis(self.config.head_dim, position_ids, self.config.rope_theta, self.config.rope_scale, self.config.rope_dims, device=x.device)

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

        next_key_values = []
        for i, layer in enumerate(self.layers):
            if all_intermediate is not None:
                if only_layers is None or (i in only_layers):
                    all_intermediate.append(x.unsqueeze(1).clone())

            past_kv = None
            if past_key_values is not None:
                past_kv = past_key_values[i] if len(past_key_values) > 0 else []

            x, current_kv = layer(
                x=x,
                attention_mask=mask,
                freqs_cis=freqs_cis,
                optimized_attention=attention_function,
                past_key_value=past_kv,
            )

            if current_kv is not None:
                next_key_values.append(current_kv)

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

        if len(next_key_values) > 0:
            return x, intermediate, next_key_values
        else:
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
