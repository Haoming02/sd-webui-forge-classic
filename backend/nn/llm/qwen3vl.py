import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2Tokenizer

from comfy import sd1_clip
import comfy.text_encoders.qwen_vl
from .qwen35 import Qwen35VisionModel
from .llama import BaseLlama, BaseQwen3, BaseGenerate, Llama2_, Qwen3VL_4BConfig, Qwen3VL_8BConfig

QWEN3VL_VISION = dict(num_heads=16, patch_size=16, temporal_patch_size=2, in_channels=3, spatial_merge_size=2, num_position_embeddings=2304, hidden_size=1024, intermediate_size=4096, depth=24, deepstack_visual_indexes=[5, 11, 17])

QWEN3VL_CONFIGS = {"qwen3vl_4b": Qwen3VL_4BConfig, "qwen3vl_8b": Qwen3VL_8BConfig}


class Qwen3VLDeepstackMerger(nn.Module):
    # DeepStack merger: postshuffle LayerNorm (applied after spatial merge), unlike the main merger.
    def __init__(self, hidden_size, spatial_merge_size, out_hidden_size, device=None, dtype=None, ops=None):
        super().__init__()
        self.merge_dim = hidden_size * (spatial_merge_size ** 2)
        self.norm = ops.LayerNorm(self.merge_dim, eps=1e-6, device=device, dtype=dtype)
        self.linear_fc1 = ops.Linear(self.merge_dim, self.merge_dim, device=device, dtype=dtype)
        self.linear_fc2 = ops.Linear(self.merge_dim, out_hidden_size, device=device, dtype=dtype)

    def forward(self, x):
        x = self.norm(x.view(-1, self.merge_dim))
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


class Qwen3VLVisionModel(Qwen35VisionModel):
    # Qwen3.5 vision + DeepStack
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__(config, device=device, dtype=dtype, ops=ops)
        self.deepstack_visual_indexes = config["deepstack_visual_indexes"]
        self.deepstack_merger_list = nn.ModuleList([
            Qwen3VLDeepstackMerger(self.hidden_size, self.spatial_merge_size, config["out_hidden_size"], device=device, dtype=dtype, ops=ops)
            for _ in self.deepstack_visual_indexes
        ])


class Qwen3VLClipModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="hidden", layer_idx=-1, dtype=None, attention_mask=True, model_options={}, model_type="qwen3vl_8b"):
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={},
                         dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False,
                         model_class=_make_qwen3vl_model(model_type), enable_attention_masks=attention_mask,
                         return_attention_masks=attention_mask, model_options=model_options)

    def generate(self, tokens, do_sample, max_length, temperature, top_k, top_p, min_p, repetition_penalty, seed, presence_penalty=0.0):
        if isinstance(tokens, dict):
            tokens = next(iter(tokens.values()))
        tokens_only = [[t[0] for t in b] for b in tokens]
        embeds, _, _, embeds_info = self.process_tokens(tokens_only, self.execution_device)
        position_ids, visual_pos_masks, deepstack = self.transformer.build_image_inputs(embeds, embeds_info)
        return self.transformer.generate(embeds, do_sample, max_length, temperature, top_k, top_p, min_p, repetition_penalty, seed,
                                         presence_penalty=presence_penalty, position_ids=position_ids,
                                         visual_pos_masks=visual_pos_masks, deepstack_embeds=deepstack)


class Qwen3VLTEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}, model_type="qwen3vl_8b"):
        clip_model = lambda **kw: Qwen3VLClipModel(**kw, model_type=model_type)
        super().__init__(device=device, dtype=dtype, name=model_type, clip_model=clip_model, model_options=model_options)


class Qwen3VLSDTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}, embedding_size=4096, embedding_key="qwen3vl_8b"):
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "qwen25_tokenizer")
        super().__init__(tokenizer_path, pad_with_end=False, embedding_directory=embedding_directory, embedding_size=embedding_size, embedding_key=embedding_key, tokenizer_class=Qwen2Tokenizer,
                         has_start_token=False, has_end_token=False, pad_to_max_length=False, max_length=99999999, min_length=1, pad_token=151643, tokenizer_data=tokenizer_data)


class Qwen3VLTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}, model_type="qwen3vl_8b"):
        embedding_size = 2560 if model_type == "qwen3vl_4b" else 4096
        tokenizer = lambda *a, **kw: Qwen3VLSDTokenizer(*a, **kw, embedding_size=embedding_size, embedding_key=model_type)
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name=model_type, tokenizer=tokenizer)
        self.llama_template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.llama_template_images = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, images=[], prevent_empty_text=False, thinking=False, **kwargs):
        image = kwargs.get("image", None)
        if image is not None and len(images) == 0:
            images = [image[i:i + 1] for i in range(image.shape[0])]

        skip_template = text.startswith('<|im_start|>')
        if prevent_empty_text and text == '':
            text = ' '

        if skip_template:
            llama_text = text
        else:
            if llama_template is not None:
                template = llama_template
            elif len(images) == 0:
                template = self.llama_template
            else:
                template = self.llama_template_images
                if len(images) > 1:
                    vision_block = "<|vision_start|><|image_pad|><|vision_end|>"
                    template = template.replace(vision_block, vision_block * len(images), 1)
            llama_text = template.format(text)
            if not thinking:  # Qwen3 convention: empty think block suppresses reasoning
                llama_text += "<think>\n\n</think>\n\n"

        tokens = super().tokenize_with_weights(llama_text, return_word_ids=return_word_ids, disable_weights=True, **kwargs)
        key_name = next(iter(tokens))
        embed_count = 0
        for r in tokens[key_name]:
            for i in range(len(r)):
                if r[i][0] == 151655:  # <|image_pad|>
                    if len(images) > embed_count:
                        r[i] = ({"type": "image", "data": images[embed_count], "original_type": "image"},) + r[i][1:]
                        embed_count += 1
        return tokens


def tokenizer(model_type="qwen3vl_8b"):
    class Qwen3VLTokenizer_(Qwen3VLTokenizer):
        def __init__(self, embedding_directory=None, tokenizer_data={}):
            super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, model_type=model_type)
    return Qwen3VLTokenizer_


def te(dtype_llama=None, llama_quantization_metadata=None, model_type="qwen3vl_8b"):
    class Qwen3VLTEModel_(Qwen3VLTEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            super().__init__(device=device, dtype=dtype, model_options=model_options, model_type=model_type)
    return Qwen3VLTEModel_
