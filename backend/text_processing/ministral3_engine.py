# https://github.com/Comfy-Org/ComfyUI/blob/v0.19.1/comfy/sd1_clip.py
# https://github.com/Comfy-Org/ComfyUI/blob/v0.19.1/comfy/text_encoders/ernie.py

import torch

from backend.args import dynamic_args
from backend.text_processing import emphasis

from ._comfy import INF, SDClipModel, SDTokenizer


class Wrap:

    def __init__(self, tokenizer):
        self.model = tokenizer

    def __call__(self, text: str, *args, **kwargs):
        z: torch.Tensor = self.model(text=text)["input_ids"]
        return {"input_ids": z.squeeze(0).tolist()}

    def __getattr__(self, name):
        return getattr(self.model, name)


class Ministral3TextProcessingEngine:
    def __init__(self, text_encoder, tokenizer):
        self.text_encoder = SDClipModel(text_encoder, layer="hidden", layer_idx=-2, special_tokens={"start": 1, "pad": 0}, layer_norm_hidden_state=False, enable_attention_masks=True, return_attention_masks=True)
        self.tokenizer = SDTokenizer(Wrap(tokenizer), pad_with_end=False, has_end_token=False, pad_to_max_length=False, pad_token=11, start_token=1, max_length=INF, min_length=1, pad_left=True, disable_weights=True)

    @property
    def emphasis(self) -> "emphasis.Emphasis":
        return emphasis.EmphasisNone()

    def tokenize(self, texts: list[str]) -> tuple[list[int], list[int]]:
        if isinstance(texts, str):
            return self.tokenizer.tokenizer(texts)["input_ids"]
        else:
            return [self.tokenizer.tokenizer(t)["input_ids"] for t in texts]

    def __call__(self, texts: list[str]) -> list[torch.Tensor]:
        if any(emphasis.uses_emphasis(text) for text in texts):
            dynamic_args.last_extra_generation_params["Emphasis"] = "None"

        zs = []
        cache: dict[str, torch.Tensor] = {}

        for line in texts:
            if line in cache:
                cond = cache[line]
            else:
                chunk = self.tokenizer.tokenize_with_weights(line, disable_weights=True)
                cond = self.text_encoder.encode_token_weights(chunk)[0]
                cache[line] = cond

            zs.extend(cond)

        return zs
