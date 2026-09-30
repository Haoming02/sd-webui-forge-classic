import math
from collections import namedtuple
from dataclasses import dataclass, field

import torch

from backend import memory_management
from backend.args import dynamic_args
from backend.text_processing import emphasis, parsing
from backend.text_processing.textual_inversion import EmbeddingDatabase
from modules.shared import opts

from ._comfy import EMBEDDINGS

PromptChunkFix = namedtuple("PromptChunkFix", ["offset", "embedding"])


@dataclass
class PromptChunk:
    tokens: list[int] = field(default_factory=list)
    multipliers: list[float] = field(default_factory=list)
    fixes: list[PromptChunkFix] = field(default_factory=list)


class CLIPEmbeddingForTextualInversion(torch.nn.Module):
    def __init__(self, wrapped, embeddings, textual_inversion_key="clip_l"):
        super().__init__()

        self.wrapped = wrapped
        self.embeddings = embeddings
        self.textual_inversion_key = textual_inversion_key
        self.weight = self.wrapped.weight

    def forward(self, input_ids):
        batch_fixes = self.embeddings.fixes
        self.embeddings.fixes = None

        inputs_embeds = self.wrapped(input_ids)

        if not batch_fixes or max(len(x) for x in batch_fixes) == 0:
            return inputs_embeds

        vecs = []
        for fixes, tensor in zip(batch_fixes, inputs_embeds):
            for offset, embedding in fixes:
                emb = (embedding.vec[self.textual_inversion_key] if isinstance(embedding.vec, dict) else embedding.vec).to(inputs_embeds)
                emb_len = min(tensor.shape[0] - offset - 1, emb.shape[0])
                tensor = torch.cat([tensor[: offset + 1], emb[:emb_len], tensor[offset + 1 + emb_len :]]).to(dtype=inputs_embeds.dtype)

            vecs.append(tensor)

        return torch.stack(vecs)


class ClipEngine:
    def __init__(self, text_encoder, tokenizer, chunk_length=75, embedding_dir=None, embedding_key="clip_l", embedding_expected_shape=768, text_projection=False, minimal_clip_skip=1, clip_skip=1, return_pooled=False, final_layer_norm=True):

        self.embeddings = EmbeddingDatabase(tokenizer, embedding_expected_shape)
        self.embedding_key = embedding_key
        if isinstance(embedding_dir, str):
            self.embeddings.add_embedding_dir(embedding_dir)
            self.embeddings.load_textual_inversion_embeddings()

        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

        self.minimal_clip_skip = minimal_clip_skip
        self.clip_skip = clip_skip

        self.chunk_length = chunk_length
        self.text_projection = text_projection
        self.return_pooled = return_pooled
        self.final_layer_norm = final_layer_norm

        self.id_start = self.tokenizer.bos_token_id
        self.id_end = self.tokenizer.eos_token_id
        self.id_pad = self.tokenizer.pad_token_id

        model_embeddings = text_encoder.transformer.text_model.embeddings
        model_embeddings.token_embedding = CLIPEmbeddingForTextualInversion(model_embeddings.token_embedding, self.embeddings, textual_inversion_key=embedding_key)

        self.comma_token = self.tokenizer.get_vocab()[",</w>"]

    @property
    def emphasis(self) -> "emphasis.Emphasis":
        return emphasis.get_current_option(opts.emphasis)()

    def tokenize(self, texts: str | list[str]) -> EMBEDDINGS | list[EMBEDDINGS]:
        return self.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]

    def empty_chunk(self):
        chunk = PromptChunk()
        chunk.tokens = [self.id_start] + [self.id_end] * (self.chunk_length + 1)
        chunk.multipliers = [1.0] * (self.chunk_length + 2)
        return chunk

    def get_target_prompt_token_count(self, token_count: int) -> int:
        return math.ceil(max(token_count, 1) / self.chunk_length) * self.chunk_length

    def encode_with_transformers(self, tokens: torch.Tensor) -> torch.Tensor:
        device = memory_management.text_encoder_device()
        tokens = tokens.to(device)

        embeddings = self.text_encoder.transformer.text_model.embeddings
        embeddings.position_ids = embeddings.position_ids.to(device=device)
        embeddings.position_embedding = embeddings.position_embedding.to(dtype=torch.float32)
        embeddings.token_embedding = embeddings.token_embedding.to(dtype=torch.float32)

        outputs = self.text_encoder.transformer(tokens, output_hidden_states=True)

        layer_id = -max(self.clip_skip, self.minimal_clip_skip)
        z = outputs.hidden_states[layer_id]

        if self.final_layer_norm:
            z = self.text_encoder.transformer.text_model.final_layer_norm(z)

        if self.return_pooled:
            pooled_output = outputs.pooler_output
            if self.text_projection and self.embedding_key != "clip_l":
                pooled_output = self.text_encoder.transformer.text_projection(pooled_output)
            z.pooled = pooled_output

        return z

    def tokenize_line(self, line: str) -> tuple[list[PromptChunk], int]:
        parsed = parsing.parse_prompt_attention(line, self.emphasis.name)
        tokenized = self.tokenize([text for text, _ in parsed])

        chunks = []
        chunk = PromptChunk()
        token_count = 0
        last_comma = -1

        def next_chunk(is_last=False):
            nonlocal token_count, last_comma, chunk

            token_count += len(chunk.tokens) if is_last else self.chunk_length

            to_add = self.chunk_length - len(chunk.tokens)
            if to_add > 0:
                chunk.tokens += [self.id_end] * to_add
                chunk.multipliers += [1.0] * to_add

            chunk.tokens = [self.id_start] + chunk.tokens + [self.id_end]
            chunk.multipliers = [1.0] + chunk.multipliers + [1.0]

            last_comma = -1
            chunks.append(chunk)
            chunk = PromptChunk()

        for tokens, (text, weight) in zip(tokenized, parsed):
            if text == "BREAK" and weight == -1:
                next_chunk()
                continue

            position = 0
            while position < len(tokens):
                token = tokens[position]

                if token == self.comma_token:
                    last_comma = len(chunk.tokens)

                elif opts.comma_padding_backtrack != 0 and len(chunk.tokens) == self.chunk_length and last_comma != -1 and len(chunk.tokens) - last_comma <= opts.comma_padding_backtrack:
                    break_location = last_comma + 1

                    reloc_tokens = chunk.tokens[break_location:]
                    reloc_mults = chunk.multipliers[break_location:]

                    chunk.tokens = chunk.tokens[:break_location]
                    chunk.multipliers = chunk.multipliers[:break_location]

                    next_chunk()
                    chunk.tokens = reloc_tokens
                    chunk.multipliers = reloc_mults

                if len(chunk.tokens) == self.chunk_length:
                    next_chunk()

                embedding, embedding_length_in_tokens = self.embeddings.find_embedding_at_position(tokens, position)
                if embedding is None:
                    chunk.tokens.append(token)
                    chunk.multipliers.append(weight)
                    position += 1
                    continue

                emb_len = int(embedding.vectors)
                if len(chunk.tokens) + emb_len > self.chunk_length:
                    next_chunk()

                chunk.fixes.append(PromptChunkFix(len(chunk.tokens), embedding))

                chunk.tokens += [0] * emb_len
                chunk.multipliers += [weight] * emb_len
                position += embedding_length_in_tokens

        if chunk.tokens or not chunks:
            next_chunk(is_last=True)

        return chunks, token_count

    def process_texts(self, texts: list[str]) -> tuple[list[PromptChunk], int]:
        token_count = 0
        cache = {}
        batch_chunks = []
        for line in texts:
            if line in cache:
                chunks = cache[line]
            else:
                chunks, current_token_count = self.tokenize_line(line)
                token_count = max(current_token_count, token_count)
                cache[line] = chunks
            batch_chunks.append(chunks)

        return batch_chunks, token_count

    def __call__(self, texts: list[str]) -> torch.Tensor:
        if any(emphasis.uses_emphasis(x) for x in texts):
            dynamic_args.last_extra_generation_params["Emphasis"] = self.emphasis.name

        batch_chunks, _ = self.process_texts(texts)
        chunk_count = max([len(x) for x in batch_chunks])

        zs: list[torch.Tensor] = []
        used_embeddings = {}

        for i in range(chunk_count):
            batch_chunk = [chunks[i] if i < len(chunks) else self.empty_chunk() for chunks in batch_chunks]

            tokens = [x.tokens for x in batch_chunk]
            multipliers = [x.multipliers for x in batch_chunk]
            self.embeddings.fixes = [x.fixes for x in batch_chunk]

            for fixes in self.embeddings.fixes:
                for _, embedding in fixes:
                    used_embeddings[embedding.name] = embedding

            z = self.process_tokens(tokens, multipliers)
            zs.append(z)

        if used_embeddings:
            names = []

            for name, embedding in used_embeddings.items():
                print(f"[Textual Inversion] Used Embedding [{name}] in CLIP of [{self.embedding_key}]")
                names.append(name.replace(":", "").replace(",", ""))

            prev = dynamic_args.last_extra_generation_params.get("TI", "")
            dynamic_args.last_extra_generation_params["TI"] = ", ".join([prev] + names).strip(", ")

        if self.return_pooled:
            return torch.hstack(zs), zs[0].pooled
        else:
            return torch.hstack(zs)

    def process_tokens(self, batch_tokens: list[int], batch_multipliers: list[float]) -> torch.Tensor:
        tokens = torch.asarray(batch_tokens)

        if self.id_end != self.id_pad:
            for batch_pos in range(len(batch_tokens)):
                index = batch_tokens[batch_pos].index(self.id_end)
                tokens[batch_pos, index + 1 : tokens.shape[1]] = self.id_pad

        z = self.encode_with_transformers(tokens)
        pooled = getattr(z, "pooled", None)

        z = self.emphasis(z, torch.asarray(batch_multipliers).to(z))

        if pooled is not None:
            z.pooled = pooled

        return z
