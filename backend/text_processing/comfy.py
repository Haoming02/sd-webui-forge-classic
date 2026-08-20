import numbers

import torch

from backend import memory_management


def gen_empty_tokens(special_tokens, length):
    start_token = special_tokens.get("start", None)
    end_token = special_tokens.get("end", None)
    pad_token = special_tokens.get("pad")
    output = []
    if start_token is not None:
        output.append(start_token)
    if end_token is not None:
        output.append(end_token)
    output += [pad_token] * (length - len(output))
    return output


class ClipTokenWeightEncoder:
    def encode_token_weights(self: "SDClipModel", token_weight_pairs):
        to_encode = list()
        max_token_len = 0
        has_weights = False
        for x in token_weight_pairs:
            tokens = list(map(lambda a: a[0], x))
            max_token_len = max(len(tokens), max_token_len)
            has_weights = has_weights or not all(map(lambda a: a[1] == 1.0, x))
            to_encode.append(tokens)

        sections = len(to_encode)
        if has_weights or sections == 0:
            if hasattr(self, "gen_empty_tokens"):
                to_encode.append(self.gen_empty_tokens(self.special_tokens, max_token_len))
            else:
                to_encode.append(gen_empty_tokens(self.special_tokens, max_token_len))

        o = self.forward(to_encode)
        out, pooled = o[:2]

        if pooled is not None:
            first_pooled = pooled[0:1].to(device=memory_management.intermediate_device())
        else:
            first_pooled = pooled

        output = []
        for k in range(0, sections):
            z = out[k : k + 1]
            if has_weights:
                z_empty = out[-1]
                for i in range(len(z)):
                    for j in range(len(z[i])):
                        weight = token_weight_pairs[k][j][1]
                        if weight != 1.0:
                            z[i][j] = (z[i][j] - z_empty[j]) * weight + z_empty[j]
            output.append(z)

        if len(output) == 0:
            r = (out[-1:].to(device=memory_management.intermediate_device()), first_pooled)
        else:
            r = (torch.cat(output, dim=-2).to(device=memory_management.intermediate_device()), first_pooled)

        if len(o) > 2:
            extra = {}
            for k in o[2]:
                v = o[2][k]
                if k == "attention_mask":
                    v = v[:sections].flatten().unsqueeze(dim=0).to(device=memory_management.intermediate_device())
                extra[k] = v

            r = r + (extra,)
        return r


class SDClipModel(torch.nn.Module, ClipTokenWeightEncoder):
    def __init__(self, model: torch.nn.Module, max_length=77, layer="last", layer_idx=None, special_tokens={"start": 49406, "end": 49407, "pad": 49407}, layer_norm_hidden_state=True, enable_attention_masks=False, zero_out_masked=False, return_projected_pooled=True, return_attention_masks=False):
        super().__init__()

        self.transformer = model
        self.freeze()

        self.num_layers = self.transformer.num_layers
        self.max_length = max_length
        self.layer = layer
        self.layer_idx = None
        self.special_tokens = special_tokens

        self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))
        self.enable_attention_masks = enable_attention_masks
        self.zero_out_masked = zero_out_masked

        self.layer_norm_hidden_state = layer_norm_hidden_state
        self.return_projected_pooled = return_projected_pooled
        self.return_attention_masks = return_attention_masks

        if layer == "hidden":
            assert layer_idx is not None
            assert abs(layer_idx) < self.num_layers

            if isinstance(self.layer, list) or self.layer == "all":
                pass
            elif isinstance(layer_idx, list):
                self.layer = layer_idx
            elif layer_idx is None or abs(layer_idx) > self.num_layers:
                self.layer = "last"
            else:
                self.layer = "hidden"
                self.layer_idx = layer_idx

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def process_tokens(self, tokens, device):
        end_token = self.special_tokens.get("end", None)
        pad_token = self.special_tokens.get("pad", -1)
        if end_token is None:
            cmp_token = pad_token
        else:
            cmp_token = end_token

        embeds_out = []
        attention_masks = []
        num_tokens = []

        for x in tokens:
            attention_mask = []
            tokens_temp = []
            other_embeds = []
            eos = False
            index = 0
            left_pad = False
            for y in x:
                if isinstance(y, numbers.Integral):
                    token = int(y)
                    if index == 0 and token == pad_token:
                        left_pad = True

                    if eos or (left_pad and token == pad_token):
                        attention_mask.append(0)
                    else:
                        attention_mask.append(1)
                        left_pad = False

                    tokens_temp += [token]
                    if not eos and token == cmp_token and not left_pad:
                        if end_token is None:
                            attention_mask[-1] = 0
                        eos = True
                else:
                    other_embeds.append((index, y))
                index += 1

            tokens_embed = torch.tensor([tokens_temp], device=device, dtype=torch.long)
            tokens_embed = self.transformer.get_input_embeddings()(tokens_embed).to(dtype=torch.float32)
            index = 0
            pad_extra = 0
            embeds_info = []
            for o in other_embeds:
                emb = o[1]
                if torch.is_tensor(emb):
                    emb = {"type": "embedding", "data": emb}

                extra = None
                emb_type = emb.get("type", None)
                if emb_type == "embedding":
                    emb = emb.get("data", None)
                else:
                    if hasattr(self.transformer, "preprocess_embed"):
                        emb, extra = self.transformer.preprocess_embed(emb, device=device)
                    else:
                        emb = None

                if emb is None:
                    index += -1
                    continue

                ind = index + o[0]
                emb = emb.view(1, -1, emb.shape[-1]).to(device=device, dtype=torch.float32)
                emb_shape = emb.shape[1]
                if emb.shape[-1] == tokens_embed.shape[-1]:
                    tokens_embed = torch.cat([tokens_embed[:, :ind], emb, tokens_embed[:, ind:]], dim=1)
                    attention_mask = attention_mask[:ind] + [1] * emb_shape + attention_mask[ind:]
                    index += emb_shape - 1
                    embeds_info.append({"type": emb_type, "index": ind, "size": emb_shape, "extra": extra})
                else:
                    index += -1
                    pad_extra += emb_shape
                    memory_management.logger.warning("Shape mismatch when applying embedding, embedding will be ignored ({} != {})".format(emb.shape[-1], tokens_embed.shape[-1]))

            if pad_extra > 0:
                padd_embed = self.transformer.get_input_embeddings()(torch.tensor([[self.special_tokens["pad"]] * pad_extra], device=device, dtype=torch.long)).to(dtype=torch.float32)
                tokens_embed = torch.cat([tokens_embed, padd_embed], dim=1)
                attention_mask = attention_mask + [0] * pad_extra

            embeds_out.append(tokens_embed)
            attention_masks.append(attention_mask)
            num_tokens.append(sum(attention_mask))

        return torch.cat(embeds_out), torch.tensor(attention_masks, device=device, dtype=torch.long), num_tokens, embeds_info

    def forward(self, tokens):
        device = self.transformer.get_input_embeddings().weight.device

        embeds, attention_mask, num_tokens, embeds_info = self.process_tokens(tokens, device)

        attention_mask_model = None
        if self.enable_attention_masks:
            attention_mask_model = attention_mask

        if isinstance(self.layer, list):
            intermediate_output = self.layer
        elif self.layer == "all":
            intermediate_output = "all"
        else:
            intermediate_output = self.layer_idx

        outputs = self.transformer(None, attention_mask_model, embeds=embeds, num_tokens=num_tokens, intermediate_output=intermediate_output, final_layer_norm_intermediate=self.layer_norm_hidden_state, dtype=torch.float32, embeds_info=embeds_info)

        if self.layer == "last":
            z = outputs[0].float()
        else:
            z = outputs[1].float()

        if self.zero_out_masked:
            z *= attention_mask.unsqueeze(-1).float()

        pooled_output = None
        if len(outputs) >= 3:
            if not self.return_projected_pooled and len(outputs) >= 4 and outputs[3] is not None:
                pooled_output = outputs[3].float()
            elif outputs[2] is not None:
                pooled_output = outputs[2].float()

        extra = {}
        if self.return_attention_masks:
            extra["attention_mask"] = attention_mask

        if len(extra) > 0:
            return z, pooled_output, extra

        return z, pooled_output


def parse_parentheses(string):
    result = []
    current_item = ""
    nesting_level = 0
    for char in string:
        if char == "(":
            if nesting_level == 0:
                if current_item:
                    result.append(current_item)
                    current_item = "("
                else:
                    current_item = "("
            else:
                current_item += char
            nesting_level += 1
        elif char == ")":
            nesting_level -= 1
            if nesting_level == 0:
                result.append(current_item + ")")
                current_item = ""
            else:
                current_item += char
        else:
            current_item += char
    if current_item:
        result.append(current_item)
    return result


def token_weights(string, current_weight):
    a = parse_parentheses(string)
    out = []
    for x in a:
        weight = current_weight
        if len(x) >= 2 and x[-1] == ")" and x[0] == "(":
            x = x[1:-1]
            xx = x.rfind(":")
            weight *= 1.1
            if xx > 0:
                try:
                    weight = float(x[xx + 1 :])
                    x = x[:xx]
                except:
                    pass
            out += token_weights(x, weight)
        else:
            out += [(x, current_weight)]
    return out


def escape_important(text):
    text = text.replace("\\)", "\0\1")
    text = text.replace("\\(", "\0\2")
    return text


def unescape_important(text):
    text = text.replace("\0\1", ")")
    text = text.replace("\0\2", "(")
    return text


class SDTokenizer:
    def __init__(self, model: torch.nn.Module, max_length=77, pad_with_end=True, has_start_token=True, has_end_token=True, pad_to_max_length=True, min_length=None, pad_token=None, end_token=None, start_token=None, min_padding=None, pad_left=False, disable_weights=False):

        self.tokenizer = model

        self.max_length = max_length
        self.min_length = min_length
        self.end_token = None
        self.min_padding = min_padding
        self.pad_left = pad_left

        empty = self.tokenizer("")["input_ids"]
        self.tokenizer_adds_end_token = has_end_token
        if has_start_token:
            assert len(empty) > 0
            self.tokens_start = 1
            self.start_token = empty[0]

            if end_token is not None:
                self.end_token = end_token
            else:
                if has_end_token:
                    self.end_token = empty[1]
        else:
            self.tokens_start = 0
            self.start_token = start_token
            if end_token is not None:
                self.end_token = end_token
            else:
                if has_end_token:
                    self.end_token = empty[0]

        if pad_token is not None:
            self.pad_token = pad_token
        elif pad_with_end:
            self.pad_token = self.end_token
        else:
            self.pad_token = 0

        self.pad_with_end = pad_with_end
        self.pad_to_max_length = pad_to_max_length
        self.max_word_length = 8

        self.disable_weights = disable_weights

    def pad_tokens(self, tokens, amount):
        if self.pad_left:
            for i in range(amount):
                tokens.insert(0, (self.pad_token, 1.0, 0))
        else:
            tokens.extend([(self.pad_token, 1.0, 0)] * amount)

    def tokenize_with_weights(self, text: str, return_word_ids=False, **kwargs):
        """
        Takes a prompt and converts it to a list of (token, weight, word id) elements.
        Tokens can both be integer tokens and pre computed CLIP tensors.
        Word id values are unique per word and embedding, where the id 0 is reserved for non word tokens.
        Returned list has the dimensions NxM where M is the input size of CLIP
        """
        min_length = kwargs.get("min_length", self.min_length)
        min_padding = self.min_padding

        text = escape_important(text)
        if kwargs.get("disable_weights", self.disable_weights):
            parsed_weights = [(text, 1.0)]
        else:
            parsed_weights = token_weights(text, 1.0)

        # tokenize words
        tokens = []
        for weighted_segment, weight in parsed_weights:
            if (word := unescape_important(weighted_segment)) == "":
                continue

            # parse word
            end = -1 if self.tokenizer_adds_end_token else 999999999999
            tokens.append([(t, weight) for t in self.tokenizer(word)["input_ids"][self.tokens_start : end]])

        # reshape token array to CLIP input size
        batched_tokens = []
        batch = []
        if self.start_token is not None:
            batch.append((self.start_token, 1.0, 0))
        batched_tokens.append(batch)
        for i, t_group in enumerate(tokens):
            # determine if we're going to try and keep the tokens in a single batch
            is_large = len(t_group) >= self.max_word_length
            if self.end_token is not None:
                has_end_token = 1
            else:
                has_end_token = 0

            while len(t_group) > 0:
                if len(t_group) + len(batch) > self.max_length - has_end_token:
                    remaining_length = self.max_length - len(batch) - has_end_token
                    # break word in two and add end token
                    if is_large:
                        batch.extend([(t, w, i + 1) for t, w in t_group[:remaining_length]])
                        if self.end_token is not None:
                            batch.append((self.end_token, 1.0, 0))
                        t_group = t_group[remaining_length:]
                    # add end token and pad
                    else:
                        if self.end_token is not None:
                            batch.append((self.end_token, 1.0, 0))
                        if self.pad_to_max_length:
                            self.pad_tokens(batch, remaining_length)
                    # start new batch
                    batch = []
                    if self.start_token is not None:
                        batch.append((self.start_token, 1.0, 0))
                    batched_tokens.append(batch)
                else:
                    batch.extend([(t, w, i + 1) for t, w in t_group])
                    t_group = []

        # fill last batch
        if self.end_token is not None:
            batch.append((self.end_token, 1.0, 0))
        if min_padding is not None:
            self.pad_tokens(batch, min_padding)
        if self.pad_to_max_length and len(batch) < self.max_length:
            self.pad_tokens(batch, self.max_length - len(batch))
        if min_length is not None and len(batch) < min_length:
            self.pad_tokens(batch, min_length - len(batch))

        if not return_word_ids:
            batched_tokens = [[(t, w) for t, w, _ in x] for x in batched_tokens]

        return batched_tokens
