# https://github.com/Comfy-Org/ComfyUI/blob/v0.26.1/comfy/sd1_clip.py
# https://github.com/Comfy-Org/ComfyUI/blob/v0.26.1/comfy/text_encoders/krea2.py
# https://github.com/Comfy-Org/ComfyUI/blob/v0.26.1/comfy/text_encoders/qwen35.py
# https://github.com/Comfy-Org/ComfyUI/blob/v0.26.1/comfy/text_encoders/qwen3vl.py

import torch

from backend import memory_management
from backend.args import dynamic_args
from backend.text_processing import emphasis, parsing
from modules.shared import opts

KREA2_TAP_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]


class PromptChunk:
    def __init__(self):
        self.tokens = []
        self.multipliers = []


class Qwen3VLTextProcessingEngine:
    def __init__(self, text_encoder, tokenizer):
        self.emphasis = emphasis.get_current_option(opts.emphasis)()

        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

        self.max_length = 99999999
        self.min_length = 1
        self.id_pad = 151643
        self.id_template = 151644
        self.id_image = 151655

        self.llama_template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

        self.vision_block = "<|vision_start|><|image_pad|><|vision_end|>"

    def tokenize(self, texts: list[str], images: int = 0):
        if images:
            prompts = [(self.vision_block * images + text.strip()) for text in texts]
        else:
            prompts = [text.strip() for text in texts]

        llama_texts = [self.llama_template.format(p) for p in prompts]
        return self.tokenizer(llama_texts)["input_ids"]

    def tokenize_line(self, line: str, images: list[torch.Tensor] = []):
        parsed = parsing.parse_prompt_attention(line, self.emphasis.name)
        tokenized = self.tokenize([text for text, _ in parsed], len(images))

        chunks = []
        chunk = PromptChunk()

        def next_chunk():
            nonlocal chunk

            chunks.append(chunk)
            chunk = PromptChunk()

        for tokens, (text, weight) in zip(tokenized, parsed):
            embed_count = 0
            position = 0
            while position < len(tokens):
                token = tokens[position]

                if token == self.id_image:
                    token = {"type": "image", "data": images[embed_count], "original_type": "image"}
                    embed_count += 1

                chunk.tokens.append(token)
                chunk.multipliers.append(weight)
                position += 1

        if chunk.tokens or not chunks:
            next_chunk()

        return chunks

    def __call__(self, texts, images: list[torch.Tensor] = []):
        if images:
            self.emphasis = emphasis.EmphasisNone()
        else:
            self.emphasis = emphasis.get_current_option(opts.emphasis)()

        if any(emphasis.uses_emphasis(x) for x in texts):
            dynamic_args.last_extra_generation_params["Emphasis"] = self.emphasis.name

        zs = []
        cache = {}

        for line in texts:
            if line in cache:
                line_z_values = cache[line]
            else:
                chunks = self.tokenize_line(line, images)
                line_z_values = []

                for chunk in chunks:
                    tokens = chunk.tokens
                    multipliers = chunk.multipliers

                    z = self.process_tokens([tokens], [multipliers])
                    z = self.strip_template(z, tokens)

                    b, seq, fuse = z.shape
                    assert b == 1 and fuse == 12 * 2560
                    z = z.reshape(b * seq, 12, 2560)

                    line_z_values.append(z)
                cache[line] = line_z_values

            zs.extend(line_z_values)

        return zs

    def strip_template(self, out, tokens):
        template_end = 0
        count_im_start = 0

        for i, v in enumerate(tokens):
            try:
                elem = int(v)
                if elem == self.id_template and count_im_start < 2:
                    template_end = i
                    count_im_start += 1
            except TypeError:
                continue

        if out.shape[2] > (template_end + 3):
            if int(tokens[template_end + 1]) == 872:
                if int(tokens[template_end + 2]) == 198:
                    template_end += 3

        out = out[:, :, template_end:]

        b, n, seq, h = out.shape
        out = out.permute(0, 2, 1, 3).reshape(b, seq, n * h)

        return out

    def process_embeds(self, batch_tokens):
        device = memory_management.text_encoder_device()

        embeds_out = []
        attention_masks = []
        num_tokens = []

        for tokens in batch_tokens:
            attention_mask = []
            tokens_temp = []
            other_embeds = []
            eos = False
            index = 0

            for t in tokens:
                try:
                    token = int(t)
                    attention_mask.append(0 if eos else 1)
                    tokens_temp += [token]
                    if not eos and token == self.id_pad:
                        eos = True
                except TypeError:
                    other_embeds.append((index, t))
                index += 1

            tokens_embed = torch.tensor([tokens_temp], device=device, dtype=torch.long)
            tokens_embed = self.text_encoder.get_input_embeddings()(tokens_embed)

            index = 0
            embeds_info = []

            for o in other_embeds:
                emb, extra = self.text_encoder.preprocess_embed(o[1], device=device)
                if emb is None:
                    index += -1
                    continue

                ind = index + o[0]
                emb = emb.view(1, -1, emb.shape[-1]).to(device=device, dtype=torch.float32)
                emb_shape = emb.shape[1]

                assert emb.shape[-1] == tokens_embed.shape[-1]
                tokens_embed = torch.cat([tokens_embed[:, :ind], emb, tokens_embed[:, ind:]], dim=1)
                attention_mask = attention_mask[:ind] + [1] * emb_shape + attention_mask[ind:]
                index += emb_shape - 1
                emb_type = o[1].get("type", None)
                embeds_info.append({"type": emb_type, "index": ind, "size": emb_shape, "extra": extra})

            embeds_out.append(tokens_embed)
            attention_masks.append(attention_mask)
            num_tokens.append(sum(attention_mask))

        return torch.cat(embeds_out), torch.tensor(attention_masks, device=device, dtype=torch.long), num_tokens, embeds_info

    def process_tokens(self, batch_tokens, batch_multipliers):
        embeds, mask, count, info = self.process_embeds(batch_tokens)

        if embeds.size(1) == len(batch_multipliers[0]):
            # images would cause the length to be different...
            self.emphasis.tokens = batch_tokens
            self.emphasis.multipliers = torch.asarray(batch_multipliers).to(embeds)
            self.emphasis.z = embeds
            self.emphasis.after_transformers()
            embeds = self.emphasis.z

        _, z = self.text_encoder(None, embeds=embeds, attention_mask=mask, num_tokens=count, embeds_info=info, intermediate_output=KREA2_TAP_LAYERS, final_layer_norm_intermediate=False)
        return z
