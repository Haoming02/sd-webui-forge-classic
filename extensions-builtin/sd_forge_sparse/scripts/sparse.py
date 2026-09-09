# https://github.com/Comfy-Org/ComfyUI/blob/v0.35.0/comfy_extras/nodes_sparse_attention.py

import logging
import re
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from backend.patcher.base import ModelPatcher

import gradio as gr
import torch
from gradio_rangeslider import RangeSlider

from backend.logging import setup_logger
from backend.quant_ops import ck
from modules import scripts
from modules.ui_components import InputAccordion

logger = logging.getLogger("SolAttn")
setup_logger(logger)

HEAD_DIM = 128


class SparseAttnPatch:

    def __init__(self, tau, sigma_start, sigma_end, min_tokens, extra_tokens, dense_blocks, verbose):
        self.tau = tau
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.min_tokens = min_tokens
        self.extra_tokens = extra_tokens
        self.dense_blocks = dense_blocks
        self.verbose = verbose

        self._logged: set[str] = set()

    def log_once(self, key: str, message: str):
        if self.verbose and key not in self._logged:
            self._logged.add(key)
            logger.info(message)

    def dense_reason(self, transformer_options: dict, tokens: int, block_index: int) -> str | None:
        if (sigmas := transformer_options.get("sigmas")) is not None:
            sigma = float(sigmas[0])
            if sigma > self.sigma_start or sigma < self.sigma_end:
                return 'sigma outside the "Timestep Range"'
        if tokens < self.min_tokens:
            return f"tokens ({tokens}) < min_tokens ({self.min_tokens})"
        if self.dense_blocks:
            if block_index is None:
                self.log_once("no_block_index", 'this model does not include block indices; "Dense Blocks" is ignored')
            elif block_index in self.dense_blocks:
                return f"block {block_index} is in dense_blocks"
        return None

    @staticmethod
    def ineligible(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dim_head: int) -> str | None:
        if q.device.type != "cuda":
            return "CUDA device is required"
        if not ck.sol_attn_is_available(q.device):
            return "sol_attn is not available for this GPU"
        if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            return f'dtype "{q.dtype}" is not supported'
        if dim_head != HEAD_DIM:
            return f"head_dim ({dim_head}) != {HEAD_DIM}"
        if q.shape != k.shape or q.shape != v.shape:
            return "cross-attention"
        if k.dtype != q.dtype or v.dtype != q.dtype:
            return f"mixed dtypes: {q.dtype}/{k.dtype}/{v.dtype}"
        return None


def make_attention_override(patch: "SparseAttnPatch", prev_attn: Callable) -> Callable:

    def override(func: Callable, q, k, v, heads, **kwargs):

        def dense():
            args = (q, k, v, heads)
            return func(*args, **kwargs) if prev_attn is None else prev_attn(func, *args, **kwargs)

        if kwargs.get("mask", None) is not None:
            return dense()

        transformer_options: dict = kwargs.get("transformer_options") or {}
        tokens: int = q.shape[2 if kwargs.get("skip_reshape", False) else 1]

        reason = patch.dense_reason(transformer_options, tokens, transformer_options.get("block_index", None))
        if reason is not None:
            patch.log_once(("dense", tokens, reason), f"dense: {reason}")
            return dense()

        if kwargs.get("skip_reshape", False):
            b, _, _, dim_head = q.shape
            qs, ks, vs = (t.transpose(1, 2) for t in (q, k, v))
        else:
            b, _, dim_head = q.shape
            dim_head //= heads
            qs, ks, vs = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        reason = patch.ineligible(qs, ks, vs, dim_head)
        if reason is not None:
            patch.log_once(("ineligible", tuple(qs.shape), reason), f"dense: {reason}")
            return dense()

        if q.dtype == torch.float32:
            qs, ks, vs = (t.to(dtype=torch.bfloat16) for t in (qs, ks, vs))

        out = ck.sol_attn(qs, ks, vs, tau=patch.tau, scale=kwargs.get("scale"), token_aug=patch.extra_tokens).to(q.dtype)
        patch.log_once(("sparse", tuple(qs.shape)), "sparse")

        if kwargs.get("skip_output_reshape", False):
            return out.transpose(1, 2)
        else:
            return out.reshape(b, -1, heads * dim_head)

    return override


class SparseAttentionForForge(scripts.Script):

    def title(self):
        return "Sparse Attention Integrated"

    def show(self, *args, **kwargs):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enable:
            with gr.Row():
                tau = gr.Slider(value=1.25, minimum=0.0, maximum=4.0, step=0.05, label="Tau", info="higher = faster & worse quality")
                percent = RangeSlider(minimum=0.0, maximum=1.0, step=0.05, value=(0.15, 0.85), label="Timestep Range")
            with gr.Row():
                min_tokens = gr.Slider(value=4096, minimum=0, maximum=32768, step=512, label="Tokens Threshold", info="model & resolution dependent ; higher = slower & better quality")
                extra_tokens = gr.Slider(value=0, minimum=0, maximum=256, step=64, label="Extra Tokens", info="higher = slower & better quality")
            dense_blocks = gr.Textbox(value="", lines=1, max_lines=1, placeholder="0, 1, 25-27", label="Dense Blocks", info="blocks to keep dense (i.e. slower / better quality)")
            verbose = gr.Checkbox(False, label="Verbose (Debug)")

        return [enable, tau, percent, min_tokens, extra_tokens, dense_blocks, verbose]

    def process_before_every_sampling(self, p, enable: bool, tau: float = 1.25, percent: tuple[float, float] = (0.15, 0.85), min_tokens: int = 4096, extra_tokens: int = 0, dense_blocks: str = "", verbose: bool = False, **kwargs):
        if not enable:
            return

        unet = p.sd_model.forge_objects.unet

        patched_unet = self._apply_block_sparse_attention(
            model=unet,
            tau=tau,
            start_percent=percent[0],
            end_percent=percent[1],
            min_tokens=min_tokens,
            extra_tokens=extra_tokens,
            dense_blocks=self._parse_block_list(dense_blocks),
            verbose=verbose,
        )

        p.sd_model.forge_objects.unet = patched_unet

    @staticmethod
    def _apply_block_sparse_attention(model: "ModelPatcher", tau: float, start_percent: float, end_percent: float, min_tokens: int, extra_tokens: int, dense_blocks: set[int], verbose: bool) -> "ModelPatcher":
        model_sampling = model.model.predictor

        patch = SparseAttnPatch(
            tau=tau,
            sigma_start=float(model_sampling.percent_to_sigma(start_percent)),
            sigma_end=float(model_sampling.percent_to_sigma(end_percent)),
            min_tokens=min_tokens,
            extra_tokens=extra_tokens,
            dense_blocks=dense_blocks,
            verbose=verbose,
        )

        m = model.clone()

        prev = m.model_options["transformer_options"].get("optimized_attention_override", None)
        override = make_attention_override(patch, prev)
        m.model_options["transformer_options"]["optimized_attention_override"] = override

        return m

    @staticmethod
    def _parse_block_list(text: str) -> set[int]:
        blocks = set()

        for part in re.findall(r"\d+\s*-\s*\d+|\d+", text or ""):
            if "-" in part:
                a, b = (int(x) for x in part.split("-"))
                blocks.update(range(min(a, b), max(a, b) + 1))
            else:
                blocks.add(int(part))

        return blocks
