# https://github.com/city96/ComfyUI-GGUF/blob/main/loader.py
# (c) City96


import torch


def dequantize(p: torch.nn.Parameter, dtype: torch.dtype) -> torch.Tensor:
    from backend.operations_gguf import dequantize_tensor

    gguf_cls = getattr(p, "gguf_cls", None)
    if gguf_cls is not None:
        gguf_cls.bake(p)

    return dequantize_tensor(p).to(dtype=dtype)


def gguf_remapping(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

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
            "mm.": "visual.merger.mlp.",
            "v.post_ln.": "visual.merger.ln_q.",
            "v.patch_embd": "visual.patch_embed.proj",
            "v.blk.": "visual.blocks.",
            "ffn_up": "mlp.up_proj",
            "ffn_down": "mlp.down_proj",
            "ffn_gate": "mlp.gate_proj",
            "attn_out.": "attn.proj.",
            "ln1.": "norm1.",
            "ln2.": "norm2.",
        }
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
