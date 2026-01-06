import copy
import logging
import os
import sys
import textwrap
from typing import Final

from backend.args import args

MAX_CHAR: int = os.get_terminal_size().columns


class ForgeFormatter(logging.Formatter):
    RESET: Final[str] = "\033[0m"

    COLORS: Final[dict[str, str]] = {
        "DEBUG": "\033[0;90m",  # Gray
        "INFO": "\033[0;96m",  # Cyan
        "WARNING": "\033[0;93m",  # Yellow
        "ERROR": "\033[0;91m",  # Red
        "CRITICAL": "\033[0;37;41m",  # White on Red
    }

    @classmethod
    def _message(cls, color: str, msg: str, file: str) -> str:
        file_len: int = len(file)
        msg: str = textwrap.fill(msg, width=MAX_CHAR - file_len - 4)
        line_len: str = len(msg.rsplit("\n", 1)[-1])
        gap: int = MAX_CHAR - line_len - file_len
        return f"{color}{msg}{cls.RESET}{' ' * gap}{cls.COLORS['DEBUG']}{file}{cls.RESET}"

    def format(self, record):
        new_record = copy.copy(record)
        color = ForgeFormatter.COLORS[record.levelname]
        new_record.msg = ForgeFormatter._message(color, record.msg, record.name)
        return super().format(new_record)


def setup_logger(logger: logging.Logger):
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ForgeFormatter("%(message)s"))
        logger.addHandler(handler)

    logger.setLevel(args.loglevel or "INFO")
