import torch

from backend import memory_management
from backend.args import dynamic_args
from backend.text_processing import emphasis
from modules.shared import opts

from ._comfy import INF, SDClipModel, SDTokenizer


class AnimaTextProcessingEngine:
    def __init__(self, text_encoder, qwen_tokenizer, t5_tokenizer):
        self.text_encoder = SDClipModel(text_encoder, layer="last", layer_idx=None, special_tokens={"pad": 151643}, layer_norm_hidden_state=False, enable_attention_masks=True, return_attention_masks=True)
        self.qwen_tokenizer = SDTokenizer(qwen_tokenizer, pad_with_end=False, has_start_token=False, has_end_token=False, pad_to_max_length=False, max_length=INF, min_length=1, pad_token=151643)
        self.t5_tokenizer = SDTokenizer(t5_tokenizer, pad_with_end=False, has_start_token=False, pad_to_max_length=False, max_length=INF, min_length=1)

    @property
    def emphasis(self) -> "emphasis.Emphasis":
        return emphasis.get_current_option(opts.emphasis)()

    def tokenize(self, texts: list[str]) -> tuple[list[int], list[int]]:
        return (
            self.qwen_tokenizer.tokenizer(texts)["input_ids"],
            self.t5_tokenizer.tokenizer(texts)["input_ids"],
        )

    def __call__(self, texts: list[str]) -> list[torch.Tensor]:
        if any(emphasis.uses_emphasis(text) for text in texts) and self.emphasis.name in ("None", "Ignore"):
            dynamic_args.last_extra_generation_params["Emphasis"] = self.emphasis.name

        n: bool = self.emphasis.name == "None"
        i: bool = self.emphasis.name == "Ignore"

        zs = []
        cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        for line in texts:
            if line in cache:
                cond, ids, weights = cache[line]
            else:
                qwen_chunk = self.qwen_tokenizer.tokenize_with_weights(line, disable_weights=n)
                t5_chunk = self.t5_tokenizer.tokenize_with_weights(line, disable_weights=n)

                qwen_chunk = [[(x[0], 1.0) for x in inner] for inner in qwen_chunk]

                cond = self.text_encoder.encode_token_weights(qwen_chunk)[0]
                ids = torch.tensor(list(map(lambda x: x[0], t5_chunk[0])), dtype=torch.int).unsqueeze(0)
                weights = torch.tensor(list(map(lambda x: (1.0 if i else x[1]), t5_chunk[0]))).unsqueeze(0).unsqueeze(-1)

                cache[line] = (cond, ids, weights)

            zs.append(self._preprocess(cond, ids, weights))

        return zs

    def _preprocess(self, cross_attn: torch.Tensor, t5xxl_ids: torch.Tensor, t5xxl_weights: torch.Tensor) -> torch.Tensor:
        device = memory_management.text_encoder_device()

        out: torch.Tensor = self.text_encoder.transformer.preprocess_text_embeds(cross_attn.to(device), t5xxl_ids.to(device))
        out.mul_(t5xxl_weights.to(device))

        if out.shape[1] < 512:
            out = torch.nn.functional.pad(out, (0, 0, 0, 512 - out.shape[1]))

        return out
