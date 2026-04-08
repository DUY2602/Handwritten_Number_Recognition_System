import logging
from typing import Any


LOGGER_NAME = "segmentation"
_logger = logging.getLogger(LOGGER_NAME)
_logger.addHandler(logging.NullHandler())


def get_logger(name: str | None = None) -> logging.Logger:
    if not name:
        return _logger
    return _logger.getChild(name)


def log_info_print(*args: Any, logger: logging.Logger | None = None, sep: str = " ", end: str = "\n") -> None:
    target = logger or _logger
    message = sep.join(str(arg) for arg in args)
    if end and end != "\n":
        message += end
    target.info(message)
