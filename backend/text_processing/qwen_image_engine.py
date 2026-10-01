# https://github.com/Comfy-Org/ComfyUI/blob/v0.36.0/comfy/text_encoders/qwen_image.py

import numbers

import torch

from backend.args import dynamic_args
from backend.text_processing import emphasis

from ._comfy import EMBEDDINGS, INF, TOKEN_WEIGHTS, SDClipModel, SDTokenizer


class Qwen25VL7BEngine:
    def __init__(self, text_encoder, tokenizer):
        self.text_encoder = SDClipModel(text_encoder, layer="last", layer_idx=None, special_tokens={"pad": 151643}, layer_norm_hidden_state=False, enable_attention_masks=True, return_attention_masks=True)
        self.tokenizer = SDTokenizer(tokenizer, pad_with_end=False, has_start_token=False, has_end_token=False, pad_to_max_length=False, max_length=INF, min_length=1, pad_token=151643)

        self.llama_template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.image_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"

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
            if line in cache:
                cond = cache[line]
            else:
                chunk = self._tokenize_with_weights(line, images)
                cond = self.text_encoder.encode_token_weights(chunk)[0]
                tok_pairs = chunk[0]

                count_im_start = 0
                template_end = -1

                if template_end == -1:
                    for i, v in enumerate(tok_pairs):
                        elem = v[0]
                        if not torch.is_tensor(elem):
                            if isinstance(elem, numbers.Integral):
                                if elem == 151644 and count_im_start < 2:
                                    template_end = i
                                    count_im_start += 1

                    if cond.shape[1] > (template_end + 3):
                        if tok_pairs[template_end + 1][0] == 872:
                            if tok_pairs[template_end + 2][0] == 198:
                                template_end += 3

                cond = cond[:, template_end:]
                cache[line] = cond

            zs.extend(cond)

        return zs

    def _tokenize_with_weights(self, text: str, images: list[torch.Tensor]) -> TOKEN_WEIGHTS:
        llama_text = (self.image_template if len(images) > 0 else self.llama_template).format(text)
        tokens = self.tokenizer.tokenize_with_weights(llama_text, disable_weights=True)

        embed_count = 0

        for r in tokens:
            for i in range(len(r)):
                if r[i][0] == 151655:
                    if len(images) > embed_count:
                        r[i] = ({"type": "image", "data": images[embed_count], "original_type": "image"},) + r[i][1:]
                        embed_count += 1

        return tokens
