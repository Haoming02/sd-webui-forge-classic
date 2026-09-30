# https://github.com/Comfy-Org/ComfyUI/blob/v0.36.0/comfy/text_encoders/flux.py

from functools import wraps

import torch

from backend.args import dynamic_args
from backend.text_processing import emphasis
from modules.shared import opts

from ._comfy import EMBEDDINGS, INF, SDClipModel, SDTokenizer, gen_empty_tokens


class T5XXLEngine:
    def __init__(self, text_encoder, tokenizer, *, is_chroma: bool = False):
        self.text_encoder = SDClipModel(text_encoder.transformer, layer="last", layer_idx=None, special_tokens={"end": 1, "pad": 0}, enable_attention_masks=False, return_attention_masks=False)
        self.tokenizer = SDTokenizer(tokenizer, pad_with_end=False, has_start_token=False, pad_to_max_length=False, max_length=INF, min_length=1 if is_chroma else 256)

        if is_chroma:
            import types

            @wraps(gen_empty_tokens)
            def _gen_empty_tokens(self, special_tokens: dict, *args, **kwargs):
                special_tokens = special_tokens.copy()
                special_tokens.pop("end")
                return gen_empty_tokens(special_tokens, *args, **kwargs)

            self.text_encoder.gen_empty_tokens = types.MethodType(_gen_empty_tokens, self.text_encoder)

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
