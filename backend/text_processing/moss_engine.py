import torch
import torch.nn.functional as F

from backend import memory_management
from backend.args import dynamic_args
from backend.text_processing import emphasis


class Qwen35MOSSImageEngine:
    max_length = 4096
    template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n"

    def __init__(self, text_encoder, tokenizer):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

    @property
    def emphasis(self) -> "emphasis.Emphasis":
        return emphasis.EmphasisNone()

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer(self.template.format(text), add_special_tokens=False)["input_ids"]

    def __call__(self, texts: list[str]) -> dict[str, list[torch.Tensor]]:
        if any(emphasis.uses_emphasis(text) for text in texts):
            dynamic_args.last_extra_generation_params["Emphasis"] = "None"

        zs, masks, cache = [], [], {}
        device = self.text_encoder.get_input_embeddings().weight.device
        for text in texts:
            if text not in cache:
                tokens = self.tokenize(text)[: self.max_length]
                ids = torch.tensor([tokens], dtype=torch.long, device=device)
                cond = self.text_encoder(ids)[0][0].float().to(memory_management.intermediate_device())
                # The parser repeats the final row when stacking variable lengths.
                # A masked sentinel keeps that padding out of MOSS joint attention.
                mask = F.pad(torch.ones(cond.shape[0], dtype=torch.bool, device=cond.device), (0, 1))
                cache[text] = (F.pad(cond, (0, 0, 0, 1)), mask)
            cond, mask = cache[text]
            zs.append(cond)
            masks.append(mask)
        return {"crossattn": zs, "attention_mask": masks}
