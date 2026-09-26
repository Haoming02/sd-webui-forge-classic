# https://github.com/Comfy-Org/ComfyUI/blob/v0.9.0/comfy/sd1_clip.py
# https://github.com/Comfy-Org/ComfyUI/blob/v0.9.0/comfy/text_encoders/flux.py

import torch

from backend.args import dynamic_args
from backend.text_processing import emphasis

from ._comfy import INF, SDClipModel, SDTokenizer


class KleinTextProcessingEngine:
    def __init__(self, text_encoder, tokenizer):
        self.text_encoder = SDClipModel(text_encoder, layer=[9, 18, 27], layer_idx=None, special_tokens={"pad": 151643}, layer_norm_hidden_state=False, enable_attention_masks=True, return_attention_masks=True)
        self.tokenizer = SDTokenizer(tokenizer, pad_with_end=False, has_start_token=False, has_end_token=False, pad_to_max_length=False, max_length=INF, min_length=512, pad_token=151643)

        self.llama_template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    @property
    def emphasis(self) -> "emphasis.Emphasis":
        return emphasis.EmphasisNone()

    def tokenize(self, texts: list[str]) -> tuple[list[int], list[int]]:
        llama_texts = [self.llama_template.format(text) for text in texts]
        return self.tokenizer.tokenizer(llama_texts)["input_ids"]

    def __call__(self, texts: list[str]) -> torch.Tensor:
        if any(emphasis.uses_emphasis(text) for text in texts):
            dynamic_args.last_extra_generation_params["Emphasis"] = "None"

        zs = []
        cache: dict[str, torch.Tensor] = {}

        for line in texts:
            line = self.llama_template.format(line)

            if line in cache:
                cond = cache[line]
            else:
                chunk = self.tokenizer.tokenize_with_weights(line, disable_weights=True)
                cond = self.text_encoder.encode_token_weights(chunk)[0]

                cond = torch.stack((cond[:, 0], cond[:, 1], cond[:, 2]), dim=1)
                cond = cond.movedim(1, 2)
                cond = cond.reshape(cond.shape[0], cond.shape[1], -1)

                cache[line] = cond

            zs.extend(cond)

        return zs
