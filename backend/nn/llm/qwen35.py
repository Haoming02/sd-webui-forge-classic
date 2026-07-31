# https://github.com/Comfy-Org/ComfyUI/blob/v0.26.1/comfy/text_encoders/qwen35.py
# https://github.com/Comfy-Org/ComfyUI/blob/v0.26.1/comfy/text_encoders/qwen3vl.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.attention import attention_function
from backend.nn.llm.llama import apply_rope

QWEN3VL_VISION = dict(num_heads=16, patch_size=16, temporal_patch_size=2, in_channels=3, spatial_merge_size=2, num_position_embeddings=2304, hidden_size=1024, intermediate_size=4096, depth=24, deepstack_visual_indexes=[5, 11, 17])


class Qwen35VisionPatchEmbed(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.patch_size = config["patch_size"]
        self.temporal_patch_size = config["temporal_patch_size"]
        self.in_channels = config["in_channels"]
        self.embed_dim = config["hidden_size"]
        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, x):
        x = x.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        return self.proj(x).view(-1, self.embed_dim)


class Qwen35VisionMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()

        self.linear_fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, hidden_state):
        return self.linear_fc2(F.gelu(self.linear_fc1(hidden_state), approximate="tanh"))


class Qwen35VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()

        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen):
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen35VisionAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()

        self.dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)

    def forward(self, x, cu_seqlens, position_embeddings):
        seq_length = x.shape[0]
        query_states, key_states, value_states = self.qkv(x).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        query_states, key_states = apply_rope(query_states, key_states, position_embeddings)

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        q_splits = torch.split(query_states, lengths, dim=0)
        k_splits = torch.split(key_states, lengths, dim=0)
        v_splits = torch.split(value_states, lengths, dim=0)

        attn_outputs = []
        for q, k, v in zip(q_splits, k_splits, v_splits):
            q = q.transpose(0, 1).unsqueeze(0)
            k = k.transpose(0, 1).unsqueeze(0)
            v = v.transpose(0, 1).unsqueeze(0)
            attn_outputs.append(attention_function(q, k, v, self.num_heads, skip_reshape=True))

        attn_output = torch.cat(attn_outputs, dim=1)
        attn_output = attn_output.reshape(seq_length, -1)
        return self.proj(attn_output)


class Qwen35VisionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Qwen35VisionAttention(hidden_size, num_heads)
        self.mlp = Qwen35VisionMLP(hidden_size, intermediate_size)

    def forward(self, x, cu_seqlens, position_embeddings):
        x = x + self.attn(self.norm1(x), cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
        return x + self.mlp(self.norm2(x))


class Qwen35VisionPatchMerger(nn.Module):
    def __init__(self, hidden_size: int, spatial_merge_size: int, out_hidden_size: int):
        super().__init__()

        merge_dim = hidden_size * (spatial_merge_size**2)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(merge_dim, merge_dim)
        self.linear_fc2 = nn.Linear(merge_dim, out_hidden_size)
        self.merge_dim = merge_dim

    def forward(self, x):
        x = self.norm(x).view(-1, self.merge_dim)
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


class Qwen35VisionModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.spatial_merge_size = config["spatial_merge_size"]
        self.patch_size = config["patch_size"]
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_heads"]
        self.num_position_embeddings = config["num_position_embeddings"]

        self.patch_embed = Qwen35VisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(self.num_position_embeddings, self.hidden_size)
        self.num_grid_per_side = int(self.num_position_embeddings**0.5)
        self.rotary_pos_emb = Qwen35VisionRotaryEmbedding(self.hidden_size // self.num_heads // 2)
        self.blocks = nn.ModuleList([Qwen35VisionBlock(self.hidden_size, self.num_heads, config["intermediate_size"]) for _ in range(config["depth"])])
        self.merger = Qwen35VisionPatchMerger(self.hidden_size, self.spatial_merge_size, config["out_hidden_size"])
        self.deepstack_visual_indexes = []
        self.deepstack_merger_list = None

    def rot_pos_emb(self, grid_thw):
        merge_size = self.spatial_merge_size
        grid_thw_list = grid_thw.tolist()
        max_hw = max(max(h, w) for _, h, w in grid_thw_list)
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device
        total_tokens = sum(int(t * h * w) for t, h, w in grid_thw_list)
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)
        offset = 0
        for num_frames, height, width in grid_thw_list:
            num_frames, height, width = int(num_frames), int(height), int(width)
            merged_h, merged_w = height // merge_size, width // merge_size
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)
            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens
        embeddings = freq_table[pos_ids]
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_thw_list = grid_thw.tolist()
        grid_ts = [int(row[0]) for row in grid_thw_list]
        grid_hs = [int(row[1]) for row in grid_thw_list]
        grid_ws = [int(row[2]) for row in grid_thw_list]
        device = self.pos_embed.weight.device
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]
        for t, h, w in grid_thw_list:
            h, w = int(h), int(w)
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h, device=device)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w, device=device)
            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor
            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side
            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for j in range(4):
                idx_list[j].extend(indices[j].tolist())
                weight_list[j].extend(weights[j].tolist())
        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])
        patch_pos_embeds_permute = []
        merge_size = self.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1).permute(0, 1, 3, 2, 4, 5).flatten(0, 4)
            patch_pos_embeds_permute.append(pos_embed)
        return torch.cat(patch_pos_embeds_permute)

    def forward(self, x, grid_thw):
        x = self.patch_embed(x)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw).to(x.device)
        x = x + pos_embeds
        rotary_pos_emb = self.rot_pos_emb(grid_thw).to(x.device)
        seq_len = x.shape[0]
        x = x.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos().unsqueeze(-2)
        sin = emb.sin().unsqueeze(-2)
        sin_half = sin.shape[-1] // 2
        position_embeddings = (cos, sin[..., :sin_half], -sin[..., sin_half:])
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_features = []
        for layer_num, blk in enumerate(self.blocks):
            x = blk(x, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
            if self.deepstack_merger_list is not None and layer_num in self.deepstack_visual_indexes:
                deepstack_features.append(self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](x))
        merged = self.merger(x)
        if self.deepstack_merger_list is not None:
            return merged, deepstack_features
        return merged


class Qwen3VLDeepstackMerger(nn.Module):
    def __init__(self, hidden_size: int, spatial_merge_size: int, out_hidden_size: int):
        super().__init__()

        self.merge_dim = hidden_size * (spatial_merge_size**2)
        self.norm = nn.LayerNorm(self.merge_dim, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.merge_dim, self.merge_dim)
        self.linear_fc2 = nn.Linear(self.merge_dim, out_hidden_size)

    def forward(self, x):
        x = self.norm(x.view(-1, self.merge_dim))
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


class Qwen3VLVisionModel(Qwen35VisionModel):
    def __init__(self, config: dict):
        super().__init__(config)

        self.deepstack_visual_indexes = config["deepstack_visual_indexes"]
        self.deepstack_merger_list = nn.ModuleList([Qwen3VLDeepstackMerger(self.hidden_size, self.spatial_merge_size, config["out_hidden_size"]) for _ in self.deepstack_visual_indexes])
