"""
Shared, verbose logging setup for all collector/processing scripts.

Why this exists: scripts used to call their own logging.basicConfig(),
which writes to STDERR by default. agent_controller.py only read
STDOUT, so most of a child script's error/warning output was invisible
to it. get_logger() fixes this by binding the console handler to
STDOUT explicitly, and also writes a separate DEBUG-level file per run
(console shows INFO, the file has full detail).
"""
import sys
import logging
from datetime import datetime

from config.settings import LOGS_DIR


def get_logger(name: str, console_level: int = logging.INFO, file_level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        # Avoid duplicate handlers if get_logger is called again with the same name.
        return logger

    logger.setLevel(min(console_level, file_level))
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"{name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.debug(f"Log file: {log_file}")
    return logger
