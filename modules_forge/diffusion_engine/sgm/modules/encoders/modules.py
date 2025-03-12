import torch
import torch.nn as nn
from einops import rearrange
from ldm_patched.ldm.modules.diffusionmodules.openaimodel import Timestep
from ldm_patched.ldm.util import instantiate_from_config
from omegaconf import ListConfig
from open_clip import tokenize
from torch.utils.checkpoint import checkpoint

from . import AbstractEmbModel


class GeneralConditioner(nn.Module):
    OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat", 5: "concat"}
    KEY2CATDIM = {"vector": 1, "crossattn": 2, "concat": 1}

    def __init__(self, emb_models: ListConfig):
        super().__init__()
        embedders = []

        for embconfig in emb_models:
            embedder = instantiate_from_config(embconfig)
            assert isinstance(embedder, AbstractEmbModel)
            embedder.is_trainable = False
            embedder.ucg_rate = 0.0

            if "input_key" in embconfig:
                embedder.input_key = embconfig["input_key"]
            elif "input_keys" in embconfig:
                embedder.input_keys = embconfig["input_keys"]
            else:
                raise KeyError('either "input_key" or "input_keys" is needed for embedder')

            embedder.legacy_ucg_val = embconfig.get("legacy_ucg_value", None)

            embedders.append(embedder)

        self.embedders = nn.ModuleList(embedders)

    def forward(self, batch: dict, force_zero_embeddings: list = None) -> dict:
        output = dict()
        if force_zero_embeddings is None:
            force_zero_embeddings = []
        for embedder in self.embedders:
            with torch.no_grad():
                if hasattr(embedder, "input_key") and (embedder.input_key is not None):
                    if embedder.legacy_ucg_val is not None:
                        batch = self.possibly_get_ucg_val(embedder, batch)
                    emb_out = embedder(batch[embedder.input_key])
                elif hasattr(embedder, "input_keys"):
                    emb_out = embedder(*[batch[k] for k in embedder.input_keys])
            assert isinstance(emb_out, (torch.Tensor, list, tuple))
            if not isinstance(emb_out, (list, tuple)):
                emb_out = [emb_out]
            for emb in emb_out:
                out_key = self.OUTPUT_DIM2KEYS[emb.dim()]
                if hasattr(embedder, "input_key") and embedder.input_key in force_zero_embeddings:
                    emb = torch.zeros_like(emb)
                if out_key in output:
                    output[out_key] = torch.cat((output[out_key], emb), self.KEY2CATDIM[out_key])
                else:
                    output[out_key] = emb
        return output

    # ========== sd_models_xl.py ========== #

    def encode_embedding_init_text(self, init_text, nvpt):
        res = []

        for embedder in [embedder for embedder in self.embedders if hasattr(embedder, "encode_embedding_init_text")]:
            encoded = embedder.encode_embedding_init_text(init_text, nvpt)
            res.append(encoded)

        return torch.cat(res, dim=1)

    def tokenize(self, texts):
        for embedder in (embedder for embedder in self.embedders if hasattr(embedder, "tokenize")):
            return embedder.tokenize(texts)
        raise SystemError

    def process_texts(self, texts):
        for embedder in (embedder for embedder in self.embedders if hasattr(embedder, "process_texts")):
            return embedder.process_texts(texts)
        raise SystemError

    def get_target_prompt_token_count(self, token_count):
        for embedder in (embedder for embedder in self.embedders if hasattr(embedder, "get_target_prompt_token_count")):
            return embedder.get_target_prompt_token_count(token_count)
        raise SystemError


class FrozenCLIPEmbedder(AbstractEmbModel):
    """Uses the CLIP transformer encoder for text (from huggingface)"""

    LAYERS = ["last", "pooled", "hidden"]

    def __init__(
        self,
        device="cuda",
        max_length=77,
        layer="last",
        layer_idx=None,
        always_return_pooled=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = None
        self.transformer = None
        self.device = device
        self.max_length = max_length
        self.layer = layer
        self.layer_idx = layer_idx
        self.return_pooled = always_return_pooled
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer == "hidden")
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        if self.return_pooled:
            return z, outputs.pooler_output
        return z

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder2(AbstractEmbModel):
    """Uses the OpenCLIP transformer encoder for text"""

    LAYERS = ["pooled", "last", "penultimate"]

    def __init__(
        self,
        device="cuda",
        max_length=77,
        layer="last",
        always_return_pooled=False,
        legacy=True,
        *args,
        **kwargs,
    ):
        super().__init__()
        assert layer in self.LAYERS
        self.model = None
        self.device = device
        self.max_length = max_length
        self.return_pooled = always_return_pooled
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError
        self.legacy = legacy

    def forward(self, text):
        tokens = tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        if not self.return_pooled and self.legacy:
            return z
        if self.return_pooled:
            assert not self.legacy
            return z[self.layer], z["pooled"]
        return z[self.layer]

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        if self.legacy:
            x = x[self.layer]
            x = self.model.ln_final(x)
            return x
        else:
            o = x["last"]
            o = self.model.ln_final(o)
            pooled = self.pool(o, text)
            x["pooled"] = pooled
            return x

    def pool(self, x, text):
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.model.text_projection
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        outputs = {}
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - 1:
                outputs["penultimate"] = x.permute(1, 0, 2)
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        outputs["last"] = x.permute(1, 0, 2)
        return outputs

    def encode(self, text):
        return self(text)


class ConcatTimestepEmbedderND(AbstractEmbModel):
    """Embeds each dimension independently and concatenates them"""

    def __init__(self, outdim):
        super().__init__()
        self.timestep = Timestep(outdim)
        self.outdim = outdim

    def forward(self, x):
        if x.ndim == 1:
            x = x[:, None]
        assert len(x.shape) == 2
        b, dims = x.shape[0], x.shape[1]
        x = rearrange(x, "b d -> (b d)")
        emb = self.timestep(x)
        emb = rearrange(emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        return emb
