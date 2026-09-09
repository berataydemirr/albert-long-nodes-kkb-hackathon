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
                    # Match the logger's own level marker (see logging_setup.py's
                    # "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"),
                    # not a bare keyword search - a keyword search flags summary
                    # lines that merely MENTION errors, e.g. bddk_finturk's own
                    # "Errors / failures : 0" (0 failures, still contains
                    # "Errors"), which was previously causing pointless retries
                    # of runs that had already succeeded. "Traceback" is kept as
                    # a separate signal since an unhandled Python exception never
                    # goes through the logger format at all.
                    if " - WARNING - " in line or " - ERROR - " in line or line.startswith("Traceback"):
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
                         help="Forwarded to child collectors that support --mode. Defaults to 'test' to avoid accidentally triggering a multi-hour full scan.")
    return parser.parse_args()


def build_pipeline(mode: str) -> list[tuple[str, list]]:
    """Full bronze -> silver -> gold pipeline, in dependency order. Not
    every module takes the same CLI - EVDS/TUIK collectors and every
    silver/gold module either take no --mode (they always process
    whatever is configured/already in bronze) or no args at all, so
    --mode is forwarded only to the modules that actually accept it."""
    mode_args = ["--mode", mode]
    return [
        # Bronze (network) then its matching silver (network-free) step,
        # source by source - keeps a source's own data self-contained if
        # only part of the pipeline needs re-running.
        ("src.collectors.bddk_monthly", mode_args),
        ("src.processing.bddk_monthly_silver", mode_args),
        ("src.collectors.bddk_weekly", mode_args),
        ("src.processing.bddk_weekly_silver", mode_args),
        ("src.collectors.bddk_finturk", mode_args),
        ("src.processing.bddk_finturk_silver", mode_args),
        ("src.collectors.evds", []),          # no --mode: always (re)fetches every series in config/evds_series.py
        ("src.processing.evds_silver", []),
        ("src.collectors.tuik", []),          # no --mode: always (re)fetches every series in config/tuik_series.py
        ("src.processing.tuik_silver", []),
        # Gold depends on every silver source above being present; each
        # gold_* builder degrades gracefully (returns an empty frame) if
        # a given source is missing, rather than crashing - see build_gold.py.
        ("src.gold.scan_cumulative", []),
        ("src.gold.build_gold", []),
    ]


if __name__ == "__main__":
    args = parse_args()
    pipeline = build_pipeline(args.mode)

    logger.info(f"Starting orchestrator (mode={args.mode}, {len(pipeline)} steps)...")
    results = []
    for module_path, extra_args in pipeline:
        results.append((module_path, run_module(module_path, extra_args)))

    failed = [name for name, ok in results if not ok]
    logger.info("=" * 50)
    logger.info("PIPELINE SUMMARY")
    for name, ok in results:
        logger.info(f"  {'OK  ' if ok else 'FAIL'} - {name}")
    logger.info(f"{len(results) - len(failed)}/{len(results)} steps succeeded.")
    logger.info("=" * 50)
    if failed:
        raise SystemExit(1)
