# https://github.com/Comfy-Org/ComfyUI/blob/v0.38.0/comfy/text_encoders/qwen_image21.py

import numbers
import types
from functools import wraps

import torch

from backend.args import dynamic_args
from backend.text_processing import emphasis

from ._comfy import EMBEDDINGS, INF, TOKEN_WEIGHTS, SDClipModel, SDTokenizer


class Qwen3VL8BEngine:
    def __init__(self, text_encoder, tokenizer):
        self.text_encoder = SDClipModel(text_encoder, layer="hidden", layer_idx=-1, special_tokens={"pad": 151643}, layer_norm_hidden_state=False, enable_attention_masks=True, return_attention_masks=True)
        self.tokenizer = SDTokenizer(tokenizer, pad_with_end=False, has_start_token=False, has_end_token=False, pad_to_max_length=False, max_length=INF, min_length=1, pad_token=151643)

        self.llama_template = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

        self.vision_block = "<|vision_start|><|image_pad|><|vision_end|>"

        self.image_spans = []

        orig = self.text_encoder.process_tokens

        @wraps(orig)
        def _process_tokens(_self, *args, **kwargs):
            embeds, attention_mask, num_tokens, embeds_info = orig(*args, **kwargs)
            self.image_spans = [(e["index"], e["size"]) for e in embeds_info if e["type"] == "image"]
            return embeds, attention_mask, num_tokens, embeds_info

        self.text_encoder.process_tokens = types.MethodType(_process_tokens, self.text_encoder)

    @property
    def emphasis(self) -> "emphasis.Emphasis":
        return emphasis.EmphasisNone()

    def tokenize(self, texts: str | list[str]) -> EMBEDDINGS | list[EMBEDDINGS]:
        if isinstance(texts, str):
            return self.tokenizer.tokenizer(self.llama_template.format(texts))["input_ids"]
        else:
            return [self.tokenizer.tokenizer(self.llama_template.format(t))["input_ids"] for t in texts]

    def __call__(self, texts: list[str], images: list[torch.Tensor] = []) -> list[torch.Tensor]:
        if any(emphasis.uses_emphasis(text) for text in texts):
            dynamic_args.last_extra_generation_params["Emphasis"] = "None"

        zs: list[torch.Tensor] = []
        cache: dict[str, torch.Tensor] = {}

        for line in texts:
            line: str = line.strip() or " "

            if line in cache:
                cond = cache[line]
            else:
                chunk = self._tokenize_with_weights(line, images)
                cond = self.text_encoder.encode_token_weights(chunk)[0]
                tokens = [t[0] for t in chunk[0]]

                im_starts, offset, spans = [], 0, iter(self.image_spans)
                for i, t in enumerate(tokens):
                    if isinstance(t, numbers.Integral):
                        if t == 151644:
                            im_starts.append(i + offset)
                    elif isinstance(t, dict) and t.get("type") == "image":
                        offset += next(spans, (0, 1))[1] - 1
                keep = torch.ones(cond.shape[1], dtype=torch.bool)
                keep[: im_starts[1] if len(im_starts) > 1 else 0] = False

                if len(images) > 0:
                    slots = []

                    for start, size in self.image_spans:
                        keep[start : start + size] = False
                        slots.append(int(keep[:start].sum()))

                    dynamic_args.image_slots = slots.copy()

                cond = cond[:, keep.to(cond.device)]
                cache[line] = cond

            zs.extend(cond)

        return zs

    def _tokenize_with_weights(self, text: str, images: list[torch.Tensor]) -> TOKEN_WEIGHTS:
        if len(images) > 0:
            refs = " ".join(f"<image{i + 1}>{self.vision_block}" for i in range(len(images)))
            llama_text = self.llama_template.format(refs + text)
        else:
            llama_text = self.llama_template.format(text)

        tokens = self.tokenizer.tokenize_with_weights(llama_text, disable_weights=True)

        embed_count = 0

        for r in tokens:
            for i in range(len(r)):
                if isinstance(r[i][0], (int, float)) and r[i][0] == 151655:
                    if len(images) > embed_count:
                        r[i] = ({"type": "image", "data": images[embed_count], "original_type": "image"},) + r[i][1:]
                        embed_count += 1

        return tokens
