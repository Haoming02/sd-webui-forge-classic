# https://github.com/Comfy-Org/ComfyUI/blob/v0.24.1/comfy/sd1_clip.py
# https://github.com/Comfy-Org/ComfyUI/blob/v0.24.1/comfy/text_encoders/pixeldit.py

import torch

from backend.args import dynamic_args
from backend.text_processing import emphasis

from ._comfy import INF, SDClipModel, SDTokenizer

PIXELDIT_MAX_LENGTH = 300

PIXELDIT_CHI_PROMPT = (
    'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions '
    "suitable for image generation. Evaluate the level of detail in the user prompt:\n"
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, "
    "and spatial relationships to create vivid and concrete scenes.\n"
    "- If the prompt is already detailed, refine and enhance the existing details slightly without "
    "overcomplicating.\n"
    "Here are examples of how to transform or refine prompts:\n"
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, "
    "sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.\n"
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring "
    "glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus "
    "passing by towering glass skyscrapers.\n"
    "Please generate only the enhanced description for the prompt below and avoid including any "
    "additional commentary or evaluations:\n"
    "User Prompt: "
)


class GemmaItTextProcessingEngine:
    def __init__(self, text_encoder, tokenizer):
        self.text_encoder = SDClipModel(text_encoder, layer="last", layer_idx=None, special_tokens={"start": 2, "pad": 0}, layer_norm_hidden_state=False, enable_attention_masks=True, return_attention_masks=True)
        self.tokenizer = SDTokenizer(tokenizer, pad_with_end=False, has_end_token=False, pad_to_max_length=False, max_length=INF, min_length=1)

        self.max_length_all = len(self.tokenize(PIXELDIT_CHI_PROMPT)) + PIXELDIT_MAX_LENGTH - 2

    @property
    def emphasis(self) -> "emphasis.Emphasis":
        return emphasis.EmphasisNone()

    def tokenize(self, texts: list[str]) -> tuple[list[int], list[int]]:
        return self.tokenizer.tokenizer(texts)["input_ids"]

    def _tokenize_with_weights(self, text: str):
        if not text.strip():
            return self.tokenizer.tokenize_with_weights("", disable_weights=True, min_length=PIXELDIT_MAX_LENGTH)

        out = self.tokenizer.tokenize_with_weights(PIXELDIT_CHI_PROMPT + text, disable_weights=True, min_length=self.max_length_all)

        return [out[0][: self.max_length_all]]

    def __call__(self, texts: list[str]) -> list[torch.Tensor]:
        if any(emphasis.uses_emphasis(text) for text in texts):
            dynamic_args.last_extra_generation_params["Emphasis"] = "None"

        zs = []
        cache: dict[str, torch.Tensor] = {}

        for line in texts:
            if line in cache:
                cond = cache[line]
            else:
                chunk = self._tokenize_with_weights(line)
                cond = self.text_encoder.encode_token_weights(chunk)[0]
                cache[line] = cond

            zs.extend(cond)

        return zs
