"""
Anima text processing path for Forge.

Copyright and license notice:
- This file contains code derived from ComfyUI.
- Upstream project: https://github.com/comfyanonymous/ComfyUI
- Upstream files:
  - https://github.com/comfyanonymous/ComfyUI/blob/f350a842611f4d75da7104c2d2965f45989089b9/comfy/text_encoders/anima.py
  - https://github.com/comfyanonymous/ComfyUI/blob/f350a842611f4d75da7104c2d2965f45989089b9/comfy/sd1_clip.py
- ComfyUI license: GNU General Public License v3.0.
- The upstream license text is available at:
  https://github.com/comfyanonymous/ComfyUI/blob/f350a842611f4d75da7104c2d2965f45989089b9/LICENSE

Design intent:
- Qwen3 encoder produces `crossattn`.
- T5 tokenizer provides `t5xxl_ids` / `t5xxl_weights` for LLM adapter.
- Keep token-weight behavior explicit and documented when diverging.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.prompt_parser import SdConditioning

import torch
from transformers import T5TokenizerFast

from backend import memory_management
from backend.text_processing import emphasis, parsing
from modules.shared import opts


class AnimaTextProcessingEngine:
    def __init__(self, text_encoder, qwen_tokenizer, t5_tokenizer_path: str):
        self.text_encoder = text_encoder
        self.qwen_tokenizer = qwen_tokenizer
        self.max_length = 99999999
        self.min_length = 1
        self.id_pad = 151643
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_tokenizer_path, local_files_only=True)
        # Comfy SDTokenizer defaults for T5 path:
        # - has_start_token=False
        # - has_end_token=True
        # - pad_with_end=False (pad token falls back to 0 unless explicitly set)
        self.t5_end_id = int(self.t5_tokenizer("")["input_ids"][0])
        self.t5_pad_id = self.t5_tokenizer.pad_token_id or 0

    def _tokenize_qwen(self, text: str) -> list[int]:
        # Comfy SDTokenizer uses tokenizer(word)["input_ids"] (default kwargs).
        ids = self.qwen_tokenizer(text)["input_ids"]
        if len(ids) == 0:
            ids = [self.id_pad]
        return ids

    def tokenize(self, texts):
        # Compatibility helper used by UI prompt length checks.
        return [self._tokenize_qwen(text) for text in texts]

    def _tokenize_t5(self, text: str) -> list[int]:
        # Mirror Comfy SDTokenizer behavior:
        # tokenizer(word)["input_ids"] with tokenizer_adds_end_token=True,
        # then strip the tokenizer-added end token from each segment,
        # and append a single end token after concatenation.
        ids = self.t5_tokenizer(text)["input_ids"][:-1]
        if len(ids) == 0:
            ids = []
        return ids

    def _build_weighted_tokens(self, line: str):
        parsed = parsing.parse_prompt_attention(line, self.emphasis.name)
        qwen_tokens = []
        qwen_weights = []
        t5_tokens = []
        t5_weights = []
        for text, weight in parsed:
            qt = self._tokenize_qwen(text)
            tt = self._tokenize_t5(text)
            qwen_tokens.extend(qt)
            # PORT_NOTE:
            # Comfy `AnimaTokenizer.tokenize_with_weights` rewrites Qwen token weights
            # to 1.0, while T5 keeps prompt-emphasis weights.
            qwen_weights.extend([1.0] * len(qt))
            t5_tokens.extend(tt)
            t5_weights.extend([weight] * len(tt))

        if len(qwen_tokens) == 0:
            qwen_tokens = [self.id_pad]
            qwen_weights = [1.0]
        if len(t5_tokens) == 0:
            t5_tokens = [self.t5_end_id]
            t5_weights = [1.0]
        else:
            t5_tokens.append(self.t5_end_id)
            t5_weights.append(1.0)

        if self.min_length > 0 and len(qwen_tokens) < self.min_length:
            pad_n = self.min_length - len(qwen_tokens)
            qwen_tokens.extend([self.id_pad] * pad_n)
            qwen_weights.extend([1.0] * pad_n)

        return qwen_tokens, qwen_weights, t5_tokens, t5_weights

    def process_embeds(self, batch_tokens):
        device = memory_management.text_encoder_device()
        embeds_out = []
        attention_masks = []
        num_tokens = []

        for tokens in batch_tokens:
            attention_mask = []
            token_ids = []
            eos = False
            for token in tokens:
                token_ids.append(int(token))
                attention_mask.append(0 if eos else 1)
                if not eos and int(token) == self.id_pad:
                    eos = True

            token_tensor = torch.tensor([token_ids], device=device, dtype=torch.long)
            token_embed = self.text_encoder.get_input_embeddings()(token_tensor)
            embeds_out.append(token_embed)
            attention_masks.append(attention_mask)
            num_tokens.append(sum(attention_mask))

        return torch.cat(embeds_out), torch.tensor(attention_masks, device=device, dtype=torch.long), num_tokens

    def process_tokens(self, batch_tokens, batch_multipliers):
        embeds, mask, count = self.process_embeds(batch_tokens)
        if embeds.size(1) == len(batch_multipliers[0]):
            self.emphasis.tokens = batch_tokens
            self.emphasis.multipliers = torch.asarray(batch_multipliers).to(embeds)
            self.emphasis.z = embeds
            self.emphasis.after_transformers()
            embeds = self.emphasis.z

        z, _ = self.text_encoder(
            None,
            attention_mask=mask,
            embeds=embeds,
            num_tokens=count,
            intermediate_output=None,
            final_layer_norm_intermediate=False,
        )
        return z

    def __call__(self, texts: "SdConditioning"):
        self.emphasis = emphasis.get_current_option(opts.emphasis)()

        qwen_tokens = []
        qwen_weights = []
        t5_ids = []
        t5_weights = []
        for line in texts:
            qt, qw, tt, tw = self._build_weighted_tokens(line)
            qwen_tokens.append(qt)
            qwen_weights.append(qw)
            t5_ids.append(tt)
            t5_weights.append(tw)

        crossattn = self.process_tokens(qwen_tokens, qwen_weights)
        max_t5_len = max(len(x) for x in t5_ids)
        text_ids = torch.full((len(t5_ids), max_t5_len), self.t5_pad_id, dtype=torch.long, device=crossattn.device)
        text_weights = crossattn.new_ones((len(t5_ids), max_t5_len))
        for i, ids in enumerate(t5_ids):
            text_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=crossattn.device)
            text_weights[i, : len(ids)] = torch.tensor(t5_weights[i], dtype=crossattn.dtype, device=crossattn.device)

        vector = crossattn.new_zeros((crossattn.shape[0], 1))
        return {
            "crossattn": crossattn,
            "vector": vector,
            "t5xxl_ids": text_ids,
            "t5xxl_weights": text_weights,
            "text_ids": text_ids,
        }
