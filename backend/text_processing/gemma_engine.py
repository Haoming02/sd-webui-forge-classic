# https://github.com/comfyanonymous/ComfyUI/blob/v0.3.64/comfy/sd1_clip.py
# https://github.com/comfyanonymous/ComfyUI/blob/v0.3.64/comfy/text_encoders/lumina2.py

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.prompt_parser import SdConditioning

import torch

from backend import memory_management
from backend.text_processing import emphasis, parsing
from modules.shared import opts


class PromptChunk:
    def __init__(self):
        self.tokens = []
        self.multipliers = []


class GemmaTextProcessingEngine:
    def __init__(self, text_encoder, tokenizer):
        super().__init__()

        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

        self.id_start = 2
        self.id_pad = 0

        self.intermediate_output = -2
        self.layer_norm_hidden_state = False

    def tokenize(self, texts):
        tokenized = self.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]
        return tokenized

    def tokenize_line(self, line):
        parsed = parsing.parse_prompt_attention(line, self.emphasis.name)

        tokenized = self.tokenize([text for text, _ in parsed])

        chunks = []
        chunk = PromptChunk()
        token_count = 0

        def next_chunk():
            nonlocal token_count
            nonlocal chunk

            chunk.tokens = [self.id_start] + chunk.tokens
            chunk.multipliers = [1.0] + chunk.multipliers

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

    @staticmethod
    def process_template(text: str, negative: bool) -> str:
        if "<Prompt Start>" in text:
            return text

        from modules.shared import opts

        if negative:
            return "\n".join([opts.neta_template_negative, text])
        else:
            return "\n".join([opts.neta_template_positive, text])

    def __call__(self, texts: "SdConditioning"):
        zs = []
        cache = {}

        self.emphasis = emphasis.get_current_option(opts.emphasis)()

        for line in texts:
            line = self.process_template(line, texts.is_negative_prompt)

            if line in cache:
                line_z_values = cache[line]
            else:
                chunks, token_count = self.tokenize_line(line)
                line_z_values = []

                # pad all chunks to length of longest chunk
                # max_tokens = 0
                # for chunk in chunks:
                #     max_tokens = max(len(chunk.tokens), max_tokens)

                for chunk in chunks:
                    tokens = chunk.tokens
                    multipliers = chunk.multipliers

                    # remaining_count = max_tokens - len(tokens)
                    # if remaining_count > 0:
                    #     tokens += [self.id_pad] * remaining_count
                    #     multipliers += [1.0] * remaining_count

                    z = self.process_tokens([tokens], [multipliers])[0]
                    line_z_values.append(z)
                cache[line] = line_z_values

            zs.extend(line_z_values)

        return torch.stack(zs)

    def process_embeds(self, batch_tokens):
        device = memory_management.text_encoder_device()
        self.text_encoder.to(device)

        embeds_out = []
        attention_masks = []
        num_tokens = []

        for tokens in batch_tokens:
            attention_mask = []
            tokens_temp = []
            eos = False
            index = 0

            for t in tokens:
                token = int(t)
                attention_mask.append(0 if eos else 1)
                tokens_temp += [token]
                if not eos and token == self.id_pad:
                    eos = True
                index += 1

            tokens_embed = torch.tensor([tokens_temp], device=device, dtype=torch.long)
            tokens_embed = self.text_encoder.get_input_embeddings()(tokens_embed)

            index = 0

            embeds_out.append(tokens_embed)
            attention_masks.append(attention_mask)
            num_tokens.append(sum(attention_mask))

        return torch.cat(embeds_out), torch.tensor(attention_masks, device=device, dtype=torch.long), num_tokens

    def process_tokens(self, batch_tokens, batch_multipliers):
        embeds, mask, count = self.process_embeds(batch_tokens)
        _, z = self.text_encoder(
            None,
            attention_mask=mask,
            embeds=embeds,
            num_tokens=count,
            intermediate_output=self.intermediate_output,
            final_layer_norm_intermediate=self.layer_norm_hidden_state,
        )
        return z
