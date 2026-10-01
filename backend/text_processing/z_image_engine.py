# https://github.com/Comfy-Org/ComfyUI/blob/v0.36.0/comfy/text_encoders/z_image.py

import torch

from backend.args import dynamic_args
from backend.text_processing import emphasis

from ._comfy import EMBEDDINGS, INF, SDClipModel, SDTokenizer


class Qwen34BEngine:
    def __init__(self, text_encoder, tokenizer):
        self.text_encoder = SDClipModel(text_encoder, layer="hidden", layer_idx=-2, special_tokens={"pad": 151643}, layer_norm_hidden_state=False, enable_attention_masks=True, return_attention_masks=True)
        self.tokenizer = SDTokenizer(tokenizer, pad_with_end=False, has_start_token=False, has_end_token=False, pad_to_max_length=False, max_length=INF, min_length=1, pad_token=151643)

        self.llama_template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

    @property
    def emphasis(self) -> "emphasis.Emphasis":
        return emphasis.EmphasisNone()

    def tokenize(self, texts: str | list[str]) -> EMBEDDINGS | list[EMBEDDINGS]:
        if isinstance(texts, str):
            return self.tokenizer.tokenizer(self.llama_template.format(texts))["input_ids"]
        else:
            return [self.tokenizer.tokenizer(self.llama_template.format(t))["input_ids"] for t in texts]

    def __call__(self, texts: list[str]) -> list[torch.Tensor]:
        if any(emphasis.uses_emphasis(text) for text in texts):
            dynamic_args.last_extra_generation_params["Emphasis"] = "None"

        zs: list[torch.Tensor] = []
        cache: dict[str, torch.Tensor] = {}

        for line in texts:
            line = self.llama_template.format(line)

            if line in cache:
                cond = cache[line]
            else:
                chunk = self.tokenizer.tokenize_with_weights(line, disable_weights=True)
                cond = self.text_encoder.encode_token_weights(chunk)[0]
                cache[line] = cond

            zs.extend(cond)

        return zs
