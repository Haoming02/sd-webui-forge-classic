# https://github.com/comfyanonymous/ComfyUI/blob/v0.3.64/comfy/sd1_clip.py
# https://github.com/comfyanonymous/ComfyUI/blob/v0.3.64/comfy/text_encoders/lumina2.py

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.prompt_parser import SdConditioning

import torch

from backend.args import dynamic_args
from backend.text_processing import emphasis
from modules.shared import opts

from ._comfy import INF, SDClipModel, SDTokenizer


class GemmaTextProcessingEngine:
    def __init__(self, text_encoder, tokenizer):
        self.text_encoder = SDClipModel(text_encoder, layer="hidden", layer_idx=-2, special_tokens={"start": 2, "pad": 0}, layer_norm_hidden_state=False, enable_attention_masks=True, return_attention_masks=True)
        self.tokenizer = SDTokenizer(tokenizer, pad_with_end=False, has_end_token=False, pad_to_max_length=False, max_length=INF, min_length=1)

    @property
    def emphasis(self) -> "emphasis.Emphasis":
        return emphasis.get_current_option(opts.emphasis)()

    def tokenize(self, texts: list[str]) -> tuple[list[int], list[int]]:
        return self.tokenizer.tokenizer(texts)["input_ids"]

    @staticmethod
    def process_template(text: str, is_negative: bool) -> str:
        if "<Prompt Start>" in text:
            return text

        return "\n".join([opts.neta_template_negative if is_negative else opts.neta_template_positive, text])

    def __call__(self, texts: "SdConditioning") -> torch.Tensor:
        if any(emphasis.uses_emphasis(text) for text in texts) and self.emphasis.name in ("None", "Ignore"):
            dynamic_args.last_extra_generation_params["Emphasis"] = self.emphasis.name

        zs = []
        cache: dict[str, torch.Tensor] = {}

        for line in texts:
            line = self.process_template(line, texts.is_negative_prompt)

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
