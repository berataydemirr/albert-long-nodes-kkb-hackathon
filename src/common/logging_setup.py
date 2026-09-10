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

# Windows' console defaults to a locale codepage (e.g. cp1254) that can't
# encode every character a log line or print() might contain - hit live:
# a model response with '->' (U+2192) crashed with UnicodeEncodeError,
# losing the whole output, when called via a raw script that never
# imported agent.py (whose own fix only covers its own entry point).
# Fixed here instead, at the one import every module in this project
# already goes through, so it applies regardless of entry point - the
# console handler created below binds to this same (now-reconfigured)
# sys.stdout object.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


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


# Human-readable labels for the stats dict keys every silver script uses
# (some use all three, some only 'parsed'/'empty_or_missing' - see
# log_stats_summary below).
_STATS_LABELS = {
    "parsed": "Newly parsed",
    "skipped_exists": "Already existed",
    "empty_or_missing": "Empty/missing",
}


def log_stats_summary(logger: logging.Logger, title: str, stats: dict) -> None:
    """Logs the "===\\nSILVER PARSE SUMMARY (...)\\n...\\n===" block every
    src/processing/*_silver.py script ends its run() with. Extracted
    here because all 5 of them had this block copy-pasted with only the
    title and (sometimes) one stat key differing - a change to the
    format previously had to be repeated in 5 places by hand."""
    logger.info("=" * 50)
    logger.info(f"SILVER PARSE SUMMARY ({title})")
    label_width = max((len(_STATS_LABELS.get(k, k)) for k in stats), default=0)
    for key, value in stats.items():
        label = _STATS_LABELS.get(key, key)
        logger.info(f"{label.ljust(label_width)} : {value}")
    logger.info("=" * 50)
