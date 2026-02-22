import logging
from typing import Iterable, Optional

from rich.console import ConsoleRenderable, RenderableType
from rich.containers import Renderables
from rich.logging import RichHandler
from rich.table import Table
from rich.text import Text

from backend.args import args


class ForgeRender:

    def __call__(self, console, renderables: Iterable["ConsoleRenderable"], log_time=None, time_format=None, level: str = "", path: Optional[str] = None, line_no=None, link_path=None) -> Table:
        output = Table.grid(padding=(0, 1))
        output.expand = True

        output.add_column(ratio=1, style="log.message", overflow="fold")
        if path:
            output.add_column(style="log.path")
            output.add_column()
        output.add_column(style="log.level")

        row: list["RenderableType"] = [Renderables(renderables)]

        if path:
            path_text = Text()
            path_text.append(path)

            row.append(path_text)
            row.append("::")

        row.append(level)
        output.add_row(*row)

        return output


_DTYPE: list[str] = ["float16", "float32", "bfloat16", "float8_e4m3fn", "float8_e5m2", "int8", "gguf", "nf4", "fp4", "int4"]
_MODELS: list[str] = ["Gemma2", "Qwen2.5", "Qwen3", "T5XXL", "Model"]

KEYWORDS: dict[str, list[str]] = {
    "attention": ["SageAttention", "FlashAttention", "PyTorch", "xformers", "sage", "flash"],
    "loader": _DTYPE + _MODELS,
    "memory_management": ["cpu", "cuda", "rocm", "PyTorch"],
    "lora": ["LORA", "UNet", "CLIP"],
}


def setup_logger(logger: logging.Logger):
    logger.propagate = False
    if not logger.handlers:
        handler = RichHandler(keywords=KEYWORDS.get(logger.name, None))
        handler._log_render = ForgeRender()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    logger.setLevel(args.loglevel or "INFO")
