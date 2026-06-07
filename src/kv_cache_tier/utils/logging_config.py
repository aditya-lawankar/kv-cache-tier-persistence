"""
Structured logging configuration.
"""

import logging
import sys
from typing import Optional

try:
    from rich.logging import RichHandler
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """Configure and return the root logger."""
    logger = logging.getLogger("kv_cache_tier")
    logger.setLevel(level)

    # Avoid adding duplicate handlers if setup_logging is called multiple times
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(module)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    if HAS_RICH:
        console_handler = RichHandler(rich_tracebacks=True, show_time=False, show_path=False)
        # RichHandler does its own formatting, but we can set a simpler formatter
        console_formatter = logging.Formatter("%(module)s:%(lineno)d | %(message)s")
        console_handler.setFormatter(console_formatter)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
