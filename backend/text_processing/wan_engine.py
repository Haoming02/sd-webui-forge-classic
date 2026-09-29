# https://github.com/Comfy-Org/ComfyUI/blob/v0.36.0/comfy/text_encoders/wan.py

import torch

from backend.args import dynamic_args
from backend.text_processing import emphasis
from modules.shared import opts

from ._comfy import EMBEDDINGS, INF, SDClipModel, SDTokenizer


class UMT5XXLEngine:
    def __init__(self, text_encoder, tokenizer):
        self.text_encoder = SDClipModel(text_encoder.transformer, layer="last", layer_idx=None, special_tokens={"end": 1, "pad": 0}, enable_attention_masks=True, zero_out_masked=True)
        self.tokenizer = SDTokenizer(tokenizer, pad_with_end=False, has_start_token=False, pad_to_max_length=False, max_length=INF, min_length=512, pad_token=0)

    @property
    def emphasis(self) -> "emphasis.Emphasis":
        return emphasis.get_current_option(opts.emphasis)()

    def tokenize(self, texts: str | list[str]) -> EMBEDDINGS | list[EMBEDDINGS]:
        return self.tokenizer.tokenizer(texts)["input_ids"]

    def __call__(self, texts: list[str]) -> list[torch.Tensor]:
        if any(emphasis.uses_emphasis(text) for text in texts) and self.emphasis.name in ("None", "Ignore"):
            dynamic_args.last_extra_generation_params["Emphasis"] = self.emphasis.name

        zs: list[torch.Tensor] = []
        cache: dict[str, torch.Tensor] = {}

        for line in texts:
            if line in cache:
                cond = cache[line]
            else:
                chunk = self.tokenizer.tokenize_with_weights(line, disable_weights=self.emphasis.name == "None")

                if self.emphasis.name == "Ignore":
                    chunk = [[(x[0], 1.0) for x in inner] for inner in chunk]

                cond = self.text_encoder.encode_token_weights(chunk)[0]
                cache[line] = cond

            zs.extend(cond)

        return zs
