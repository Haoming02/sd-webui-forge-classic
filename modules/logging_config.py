import logging
import os

from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, fallback_handler: logging.Handler):
        super().__init__()
        self.fallback_handler = fallback_handler

    def emit(self, record: logging.LogRecord):
        try:
            if tqdm._instances:
                tqdm.write(self.format(record))
            else:
                self.fallback_handler.emit(record)
        except Exception:
            self.fallback_handler.emit(record)


def setup_logging(loglevel: str = None):
    loglevel: str = loglevel or os.environ.get("SD_WEBUI_LOG_LEVEL") or logging.ERROR

    if os.environ.get("SD_WEBUI_RICH_LOG"):
        from rich.logging import RichHandler

        handler = RichHandler(show_path=False)
    else:
        formatter = logging.Formatter("%(message)s")

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        handler = TqdmLoggingHandler(handler)
        handler.setFormatter(formatter)

    logging.basicConfig(level=loglevel, force=True, handlers=[handler])

    if os.environ.get("SD_WEBUI_USE_PRINT"):
        return

    import builtins
    from functools import wraps

    logger = logging.getLogger("system")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False

    _orig = builtins.print

    @wraps(_orig)
    def log(*args, **kwargs):
        if kwargs:
            return _orig(*args, **kwargs)

        try:
            return logger.info(" ".join(str(arg) for arg in args))
        except Exception:
            return _orig(*args, **kwargs)

    builtins.print = log
