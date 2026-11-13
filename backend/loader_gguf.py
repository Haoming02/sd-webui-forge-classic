# https://github.com/city96/ComfyUI-GGUF/blob/main/loader.py
# (c) City96


import gguf
import torch

TORCH_COMPATIBLE_QTYPES = (None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)


def is_quantized(tensor) -> bool:
    return tensor is not None and getattr(tensor, "tensor_type", None) not in TORCH_COMPATIBLE_QTYPES


def dequantize(p: torch.nn.Parameter, dtype: torch.dtype) -> torch.Tensor:
    from backend.operations_gguf import dequantize_tensor

    gguf_cls = getattr(p, "gguf_cls", None)
    if gguf_cls is not None:
        gguf_cls.bake(p)

    return dequantize_tensor(p).to(dtype=dtype)


def _reverse_qwen35(sd: dict[str, torch.Tensor], num_k_heads: int):
    plain = lambda x: dequantize(x, torch.float32).as_subclass(torch.Tensor)

    for key in list(sd.keys()):
        if key.endswith("norm.weight") and not key.endswith("linear_attn.norm.weight"):
            sd[key] = plain(sd[key]) - 1.0
        elif key.endswith("linear_attn.A_log"):
            sd[key] = torch.log(-plain(sd[key]))
        elif key.endswith("linear_attn.conv1d.weight"):
            sd[key] = plain(sd[key]).unsqueeze(1)

    for key in [k for k in sd.keys() if k.endswith("linear_attn.A_log")]:
        prefix = key[: -len("A_log")]
        num_v_heads = sd[key].shape[0]
        head_v_dim = sd[f"{prefix}norm.weight"].shape[0]
        perm = lambda d: torch.arange(num_v_heads * d).reshape(num_v_heads // num_k_heads, num_k_heads, d).transpose(0, 1).flatten()
        v_perm = perm(head_v_dim)

        for name in ("in_proj_qkv.weight", "conv1d.weight", "in_proj_z.weight"):
            t = sd[f"{prefix}{name}"]
            offset = t.shape[0] - v_perm.shape[0]
            t.data = t.data[torch.cat([torch.arange(offset), offset + v_perm])]
        for name in ("in_proj_a.weight", "in_proj_b.weight", "A_log", "dt_bias"):
            t = sd[f"{prefix}{name}"]
            t.data = t.data[perm(1)]

        sd[f"{prefix}out_proj.weight"] = dequantize(sd[f"{prefix}out_proj.weight"], torch.float16)[:, v_perm]

    return sd


def gguf_remapping(state_dict: dict[str, torch.Tensor], metadata: dict) -> dict[str, torch.Tensor]:

    if metadata.get("general.architecture", None) == "qwen35":
        qwen_35_format = {
            "blk.": "model.language_model.layers.",
            "attn_norm": "input_layernorm",
            "post_attention_norm": "post_attention_layernorm",
            "attn_q_norm.": "self_attn.q_norm.",
            "attn_k_norm.": "self_attn.k_norm.",
            "attn_qkv": "linear_attn.in_proj_qkv",
            "attn_gate": "linear_attn.in_proj_z",
            "attn_q.": "self_attn.q_proj.",
            "attn_k.": "self_attn.k_proj.",
            "attn_v.": "self_attn.v_proj.",
            "attn_output": "self_attn.o_proj",
            "ssm_alpha": "linear_attn.in_proj_a",
            "ssm_beta": "linear_attn.in_proj_b",
            "ssm_out": "linear_attn.out_proj",
            "ssm_norm": "linear_attn.norm",
            "ssm_conv1d": "linear_attn.conv1d",
            "ssm_dt.bias": "linear_attn.dt_bias",
            "ssm_a": "linear_attn.A_log",
            "ffn_up": "mlp.up_proj",
            "ffn_down": "mlp.down_proj",
            "ffn_gate": "mlp.gate_proj",
            "token_embd": "model.language_model.embed_tokens",
            "output_norm": "model.language_model.norm",
            "output.weight": "lm_head.weight",
        }
        new_sd = {}
        for k, v in state_dict.items():
            for s, d in qwen_35_format.items():
                k = k.replace(s, d)
            new_sd[k] = v
        new_sd["model.language_model.embed_tokens.weight"] = new_sd["model.language_model.embed_tokens.weight"].dequantize_as_pytorch_parameter()
        state_dict.clear()
        state_dict = _reverse_qwen35(new_sd, int(metadata["qwen35.ssm.group_count"]))

    if "enc.blk.0.attn_k.weight" in state_dict:
        gguf_t5_format = {
            "enc.": "encoder.",
            ".blk.": ".block.",
            "token_embd": "shared",
            "output_norm": "final_layer_norm",
            "attn_q": "layer.0.SelfAttention.q",
            "attn_k": "layer.0.SelfAttention.k",
            "attn_v": "layer.0.SelfAttention.v",
            "attn_o": "layer.0.SelfAttention.o",
            "attn_norm": "layer.0.layer_norm",
            "attn_rel_b": "layer.0.SelfAttention.relative_attention_bias",
            "ffn_up": "layer.1.DenseReluDense.wi_1",
            "ffn_down": "layer.1.DenseReluDense.wo",
            "ffn_gate": "layer.1.DenseReluDense.wi_0",
            "ffn_norm": "layer.1.layer_norm",
        }
        new_sd = {}
        for k, v in state_dict.items():
            for s, d in gguf_t5_format.items():
                k = k.replace(s, d)
            new_sd[k] = v
        new_sd["shared.weight"] = new_sd["shared.weight"].dequantize_as_pytorch_parameter()
        state_dict.clear()
        state_dict = new_sd

    if "blk.0.attn_norm.weight" in state_dict:
        gguf_llm_format = {
            "blk.": "model.layers.",
            "attn_norm": "input_layernorm",
            "attn_q_norm.": "self_attn.q_norm.",
            "attn_k_norm.": "self_attn.k_norm.",
            "attn_v_norm.": "self_attn.v_norm.",
            "attn_q": "self_attn.q_proj",
            "attn_k": "self_attn.k_proj",
            "attn_v": "self_attn.v_proj",
            "attn_output": "self_attn.o_proj",
            "ffn_up": "mlp.up_proj",
            "ffn_down": "mlp.down_proj",
            "ffn_gate": "mlp.gate_proj",
            "ffn_norm": "post_attention_layernorm",
            "token_embd": "model.embed_tokens",
            "output_norm": "model.norm",
            "output.weight": "lm_head.weight",
        }
        new_sd = {}
        for k, v in state_dict.items():
            for s, d in gguf_llm_format.items():
                k = k.replace(s, d)
            new_sd[k] = v
        new_sd["model.embed_tokens.weight"] = new_sd["model.embed_tokens.weight"].dequantize_as_pytorch_parameter()
        state_dict.clear()
        state_dict = new_sd

    if "v.patch_embd.weight.1" in state_dict:
        w1 = dequantize(state_dict.pop("v.patch_embd.weight"), torch.float32)
        w2 = dequantize(state_dict.pop("v.patch_embd.weight.1"), torch.float32)
        state_dict["v.patch_embd.weight"] = torch.stack([w1, w2], dim=2)

    if "mm.0.weight" in state_dict:
        gguf_clip_vision_format = {
            "v.patch_embd": "visual.patch_embed.proj",
            "v.blk.": "visual.blocks.",
            "attn_out.": "attn.proj.",
            "ln1.": "norm1.",
            "ln2.": "norm2.",
        }
        if metadata.get("clip.projector_type") == "qwen3vl_merger":
            gguf_clip_vision_format.update(
                {
                    "mm.0.": "visual.merger.linear_fc1.",
                    "mm.2.": "visual.merger.linear_fc2.",
                    "v.post_ln.": "visual.merger.norm.",
                    "v.position_embd.": "visual.pos_embed.",
                    "attn_qkv.": "attn.qkv.",
                    "ffn_up.": "mlp.linear_fc1.",
                    "ffn_down.": "mlp.linear_fc2.",
                }
            )
        else:
            gguf_clip_vision_format.update(
                {
                    "mm.": "visual.merger.mlp.",
                    "v.post_ln.": "visual.merger.ln_q.",
                    "ffn_up": "mlp.up_proj",
                    "ffn_down": "mlp.down_proj",
                    "ffn_gate": "mlp.gate_proj",
                }
            )
        new_sd = {}
        for k, v in state_dict.items():
            for s, d in gguf_clip_vision_format.items():
                k = k.replace(s, d)
            new_sd[k] = v
        state_dict.clear()
        state_dict = new_sd

    if "visual.blocks.0.attn_q.weight" in state_dict:
        attns = {}

        _keys = list(state_dict.keys())
        _sd = {}

        for k in _keys:
            if any(x in k for x in ["attn_q", "attn_k", "attn_v"]):
                k_attn, k_name = k.rsplit(".attn_", 1)
                k_attn += ".attn.qkv." + k_name.split(".")[-1]
                if k_attn not in attns:
                    attns[k_attn] = {}
                attns[k_attn][k_name] = dequantize(state_dict.pop(k), torch.float16)
            else:
                _sd[k] = state_dict.pop(k)

        del state_dict
        state_dict = _sd

        for k, v in attns.items():
            suffix = k.split(".")[-1]
            state_dict[k] = torch.cat(
                [
                    v[f"q.{suffix}"],
                    v[f"k.{suffix}"],
                    v[f"v.{suffix}"],
                ],
                dim=0,
            )

        del attns

    return state_dict


def get_orig_shape(reader: gguf.GGUFReader, tensor_name: str) -> torch.Size | None:
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    if len(field.types) == 2 and field.types[0] == gguf.GGUFValueType.ARRAY and field.types[1] == gguf.GGUFValueType.INT32:
        return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))
    return None
