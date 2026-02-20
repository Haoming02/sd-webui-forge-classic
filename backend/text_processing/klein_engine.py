# https://github.com/Comfy-Org/ComfyUI/blob/v0.9.0/comfy/sd1_clip.py
# https://github.com/Comfy-Org/ComfyUI/blob/v0.9.0/comfy/text_encoders/flux.py

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


class KleinTextProcessingEngine:
    def __init__(self, text_encoder, tokenizer):
        super().__init__()

        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

        self.id_pad = 151643
        self.min_length = 512
        self.llama_template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.intermediate_output = [9, 18, 27]
        self.layer_norm_hidden_state = False

    def tokenize(self, texts):
        llama_texts = [self.llama_template.format(text) for text in texts]
        return self.tokenizer(llama_texts)["input_ids"]

    def tokenize_line(self, line: str):
        parsed = parsing.parse_prompt_attention(line, self.emphasis.name)
        tokenized = self.tokenize([text for text, _ in parsed])

        chunks = []
        chunk = PromptChunk()

        def next_chunk():
            nonlocal chunk

            current_chunk_length = len(chunk.tokens)
            remaining_count = self.min_length - current_chunk_length

            if remaining_count > 0:
                chunk.tokens += [self.id_pad] * remaining_count
                chunk.multipliers += [1.0] * remaining_count

            chunks.append(chunk)
            chunk = PromptChunk()

        for tokens, (text, weight) in zip(tokenized, parsed):
            position = 0
            while position < len(tokens):
                token = tokens[position]
                chunk.tokens.append(token)
                chunk.multipliers.append(weight)
                position += 1

        if chunk.tokens or not chunks:
            next_chunk()

        return chunks

    def __call__(self, texts: "SdConditioning"):
        zs = []
        cache = {}

        self.emphasis = emphasis.get_current_option(opts.emphasis)()

        for line in texts:
            if line in cache:
                line_z_values = cache[line]
            else:
                chunks = self.tokenize_line(line)
                line_z_values = []

                for chunk in chunks:
                    tokens = chunk.tokens
                    multipliers = chunk.multipliers

                    z = self.process_tokens([tokens], [multipliers])[0]
                    line_z_values.append(z)
                cache[line] = line_z_values

            zs.extend(line_z_values)

        return zs

    def process_embeds(self, batch_tokens):
        device = memory_management.text_encoder_device()

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
                    attention_mask[-1] = 0
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

        self.emphasis.tokens = batch_tokens
        self.emphasis.multipliers = torch.asarray(batch_multipliers).to(embeds)
        self.emphasis.z = embeds
        self.emphasis.after_transformers()
        embeds = self.emphasis.z

        _, out = self.text_encoder(
            None,
            attention_mask=mask,
            embeds=embeds,
            num_tokens=count,
            intermediate_output=self.intermediate_output,
            final_layer_norm_intermediate=self.layer_norm_hidden_state,
        )

        out = torch.stack((out[:, 0], out[:, 1], out[:, 2]), dim=1)
        out = out.movedim(1, 2)
        out = out.reshape(out.shape[0], out.shape[1], -1)

        return out
