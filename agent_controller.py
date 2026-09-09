"""
Top-level orchestrator: runs collector scripts in sequence, with a
simple keyword-based retry on failure.

HONEST NAMING NOTE: this script is NOT an "autonomous ReAct agent" - it
has no logic that sees an error and fixes its own code. What it does:
runs a subprocess, and if ERROR/EXCEPTION/WARNING/500 shows up in its
output, restarts it up to MAX_RETRIES times. Real self-healing logic is
deliberately out of scope here (a later phase).

PREVIOUS BUG (fixed): Python's logging module writes to STDERR by
default. This script used to only read the child process's STDOUT - so
almost all log output produced by child scripts was invisible. It also
piped stderr without ever draining it, which risked filling the OS pipe
buffer and deadlocking the child process on longer runs.
Fix: (1) collectors now bind their console handler explicitly to STDOUT
via src/common/logging_setup.py, (2) here stderr is merged into stdout
(stderr=subprocess.STDOUT) instead of being read separately - this
removes both the blindness and the deadlock risk.

Usage:
    python agent_controller.py --mode test   (default, safe/fast)
    python agent_controller.py --mode full   (can take HOURS, use with care)
"""
import sys
import subprocess
import time
import argparse

from config.settings import PROJECT_ROOT
from src.common.logging_setup import get_logger

logger = get_logger("agent_controller")

MAX_RETRIES = 3


def run_module(module_path: str, extra_args: list) -> bool:
    """Runs a collector module as 'python -m <module_path> <extra_args>'.
    Returns True on success, False if it still fails after MAX_RETRIES."""
    retries = 0
    while retries < MAX_RETRIES:
        cmd = [sys.executable, "-m", module_path, *extra_args]
        logger.info(f"Running '{' '.join(cmd)}'. (Attempt: {retries + 1}/{MAX_RETRIES})")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # merge stderr into stdout: fixes both blindness and deadlock risk
                text=True,
                cwd=PROJECT_ROOT,
            )

            error_found = False
            error_summary = ""

            for line in iter(process.stdout.readline, ""):
                line = line.strip()
                if line:
                    print(line)
                    upper_line = line.upper()
                    if any(keyword in upper_line for keyword in ["ERROR", "EXCEPTION", "WARNING", " 500 "]):
                        error_found = True
                        error_summary = line
                        logger.warning(f"Detected error/warning in subprocess: {error_summary}")

            process.wait()

            if error_found or process.returncode != 0:
                logger.error(f"'{module_path}' failed or produced an error. Retrying...")
                retries += 1
                time.sleep(2)
            else:
                logger.info(f"'{module_path}' completed successfully.")
                return True

        except Exception as e:
            logger.error(f"System error while running '{module_path}': {e}")
            retries += 1
            time.sleep(2)

    logger.error(f"Reached MAX_RETRIES. '{module_path}' failed permanently.")
    return False


def parse_args():
    parser = argparse.ArgumentParser(description="Top-level orchestrator that runs collectors in sequence")
    parser.add_argument("--mode", choices=["test", "full"], default="test",
                         help="Forwarded to child collectors. Defaults to 'test' to avoid accidentally triggering a multi-hour full scan.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    modules = [
        ("src.collectors.bddk_monthly", ["--mode", args.mode]),
        ("src.processing.bddk_monthly_silver", ["--mode", args.mode]),
    ]

    logger.info(f"Starting orchestrator (mode={args.mode})...")
    for module_path, extra_args in modules:
        run_module(module_path, extra_args)
    logger.info("All tasks finished.")
