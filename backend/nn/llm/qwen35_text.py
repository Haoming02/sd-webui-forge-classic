# Adapted from ComfyUI's comfy/text_encoders/qwen35.py (GPL-3.0).
# Text-only Qwen3.5 backbone for MOSS-Image; no generation or vision weights.

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend import memory_management
from backend.nn.llm.llama import MLP, apply_rope, attention_function, precompute_freqs_cis


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
    def __init__(self, config):
        super().__init__()
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.key_dim = self.num_key_heads * self.key_head_dim
        self.value_dim = self.num_value_heads * self.value_head_dim
        conv_dim = self.key_dim * 2 + self.value_dim

        self.in_proj_qkv = nn.Linear(config.hidden_size, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(config.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(config.hidden_size, self.num_value_heads, bias=False)
        self.in_proj_a = nn.Linear(config.hidden_size, self.num_value_heads, bias=False)
        self.out_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(self.num_value_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_value_heads))
        kernel = config.linear_conv_kernel_dim
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, kernel, groups=conv_dim, padding=kernel - 1, bias=False)
        self.norm = nn.RMSNorm(self.value_head_dim, eps=config.rms_norm_eps)

    def forward(self, x):
        batch, length, _ = x.shape
        mixed = F.silu(self.conv1d(self.in_proj_qkv(x).transpose(1, 2))[:, :, :length]).transpose(1, 2)
        query, key, value = mixed.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch, length, self.num_key_heads, self.key_head_dim)
        key = key.reshape(batch, length, self.num_key_heads, self.key_head_dim)
        value = value.reshape(batch, length, self.num_value_heads, self.value_head_dim)
        repeats = self.num_value_heads // self.num_key_heads
        query = query.repeat_interleave(repeats, dim=2)
        key = key.repeat_interleave(repeats, dim=2)
        decay = -memory_management.cast_to(self.A_log, device=x.device, dtype=torch.float32).exp()
        bias = memory_management.cast_to(self.dt_bias, device=x.device, dtype=torch.float32)
        g = decay * F.softplus(self.in_proj_a(x).float() + bias)
        beta = self.in_proj_b(x).sigmoid()
        out, _ = torch_chunk_gated_delta_rule(query, key, value, g, beta)
        out = self.norm(out) * F.silu(self.in_proj_z(x).reshape(batch, length, self.num_value_heads, self.value_head_dim))
        return self.out_proj(out.flatten(2))


class GatedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        inner = self.num_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, inner * 2, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(inner, config.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps, add=True)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps, add=True)

    def forward(self, x, mask, freqs):
        batch, length, _ = x.shape
        query, gate = self.q_proj(x).view(batch, length, self.num_heads, 2 * self.head_dim).chunk(2, dim=-1)
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(self.k_proj(x).view(batch, length, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        value = self.v_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        query_rot, key_rot = apply_rope(query[..., : self.rotary_dim], key[..., : self.rotary_dim], freqs)
        query = torch.cat((query_rot, query[..., self.rotary_dim :]), dim=-1)
        key = torch.cat((key_rot, key[..., self.rotary_dim :]), dim=-1)
        out = attention_function(query, key, value, self.num_heads, mask=mask, skip_reshape=True, enable_gqa=self.num_heads != self.num_kv_heads)
        return self.o_proj(out * gate.reshape(batch, length, -1).sigmoid())


class Qwen35Block(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.layer_type = config.layer_types[index]
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config)
        else:
            self.self_attn = GatedAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=True)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=True)

    def forward(self, x, mask, freqs):
        normalized = self.input_layernorm(x)
        if self.layer_type == "linear_attention":
            x = x + self.linear_attn(normalized)
        else:
            x = x + self.self_attn(normalized, mask, freqs)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen35Backbone(nn.Module):
    def __init__(self, config, computation_dtype):
        super().__init__()
        self.config = config
        self.computation_dtype = computation_dtype
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=computation_dtype)
        self.layers = nn.ModuleList([Qwen35Block(config, index) for index in range(config.num_hidden_layers)])
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=True)

    def forward(self, ids):
        x = self.embed_tokens(ids).to(self.computation_dtype)
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        freqs = precompute_freqs_cis(int(self.config.head_dim * self.config.partial_rotary_factor), positions, self.config.rope_parameters["rope_theta"], device=x.device)
        mask = torch.full((x.shape[1], x.shape[1]), torch.finfo(x.dtype).min / 4, device=x.device, dtype=x.dtype).triu_(1)
        for layer in self.layers:
            x = layer(x, mask, freqs)
        return self.norm(x), None


class Qwen35Text(nn.Module):
    def __init__(self, config_dict, computation_dtype=torch.float32):
        super().__init__()
        text_config = config_dict.get("text_config", config_dict).copy()
        text_config["mlp_activation"] = text_config.get("hidden_act", "silu")
        config = SimpleNamespace(**text_config)
        self.num_layers = config.num_hidden_layers
        self.model = Qwen35Backbone(config, computation_dtype)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(self, input_ids):
        return self.model(input_ids)
