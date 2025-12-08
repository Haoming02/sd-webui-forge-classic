# https://github.com/comfyanonymous/ComfyUI/blob/v0.3.64/comfy/sd1_clip.py
# https://github.com/comfyanonymous/ComfyUI/blob/v0.3.64/comfy/text_encoders/wan.py

import torch

from backend import memory_management
from backend.text_processing import emphasis, parsing
from modules.shared import opts


class PromptChunk:
    def __init__(self):
        self.tokens = []
        self.multipliers = []


class UMT5TextProcessingEngine:
    def __init__(self, text_encoder, tokenizer, min_length=512):
        super().__init__()

        self.text_encoder = text_encoder.transformer
        self.tokenizer = tokenizer
        self.device = memory_management.text_encoder_device()

        self.max_length = 99999999
        self.min_length = min_length

        empty = self.tokenizer("")["input_ids"]
        self.tokens_start = 0
        self.tokens_end = -1
        self.end_token = empty[0]
        self.pad_token = 0

    def tokenize(self, texts):
        return self.tokenizer(texts)["input_ids"]

    def process_attn_mask(self, tokens):
        attention_masks = []

        for x in tokens:
            attention_mask = []
            eos = False

            for y in x:
                if isinstance(y, int):
                    attention_mask.append(0 if eos else 1)
                    if not eos and int(y) == self.end_token:
                        eos = True

            attention_masks.append(attention_mask)

        return torch.tensor(attention_masks, dtype=torch.long, device=self.device)

    def encode_with_transformers(self, tokens, attention_mask):
        tokens = tokens.to(self.device)
        self.text_encoder.shared.to(device=self.device, dtype=torch.float32)
        return self.text_encoder(input_ids=tokens, attention_mask=attention_mask)

    def tokenize_line(self, line):
        parsed = parsing.parse_prompt_attention(line, self.emphasis.name)

        tokenized = self.tokenize([text[self.tokens_start : self.tokens_end] for text, _ in parsed])

        chunks = []
        chunk = PromptChunk()
        token_count = 0

        def next_chunk():
            nonlocal token_count
            nonlocal chunk

            chunk.tokens.append(self.end_token)
            chunk.multipliers.append(1.0)

            current_chunk_length = len(chunk.tokens)
            token_count += current_chunk_length

            if current_chunk_length < self.min_length:
                chunk.tokens.extend([self.pad_token] * (self.min_length - current_chunk_length))
                chunk.multipliers.extend([1.0] * (self.min_length - current_chunk_length))

            chunks.append(chunk)
            chunk = PromptChunk()

        for tokens, (text, weight) in zip(tokenized, parsed):
            if text == "BREAK" and weight == -1:
                next_chunk()
                continue

            position = 0
            while position < len(tokens):
                token = tokens[position]
                chunk.tokens.append(token)
                chunk.multipliers.append(weight)
                position += 1

        if chunk.tokens or not chunks:
            next_chunk()

        return chunks, token_count

    def __call__(self, texts):
        zs = []
        cache = {}

        self.emphasis = emphasis.get_current_option(opts.emphasis)()

        for line in texts:
            if line in cache:
                line_z_values = cache[line]
            else:
                chunks, _ = self.tokenize_line(line)
                line_z_values = []

                # pad all chunks to length of longest chunk
                max_tokens = 0
                for chunk in chunks:
                    max_tokens = max(len(chunk.tokens), max_tokens)

                for chunk in chunks:
                    tokens = chunk.tokens
                    multipliers = chunk.multipliers

                    remaining_count = max_tokens - len(tokens)
                    if remaining_count > 0:
                        tokens += [self.id_pad] * remaining_count
                        multipliers += [1.0] * remaining_count

                    z = self.process_tokens([tokens], [multipliers])[0]
                    line_z_values.append(z)
                cache[line] = line_z_values

            zs.extend(line_z_values)

        return torch.stack(zs)

    def process_tokens(self, batch_tokens, batch_multipliers):
        tokens = torch.asarray(batch_tokens)

        attention_mask = self.process_attn_mask(batch_tokens)
        z = self.encode_with_transformers(tokens, attention_mask)
        z *= attention_mask.unsqueeze(-1).float()

        self.emphasis.tokens = batch_tokens
        self.emphasis.multipliers = torch.asarray(batch_multipliers).to(z)
        self.emphasis.z = z
        self.emphasis.after_transformers()
        z = self.emphasis.z

        return z
