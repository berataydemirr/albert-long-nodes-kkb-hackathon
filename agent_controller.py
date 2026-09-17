"""
Top-level orchestrator: runs collector scripts in sequence, retrying a
step that exits non-zero.

HONEST NAMING NOTE: this script is NOT an "autonomous ReAct agent" - it
has no logic that sees an error and fixes its own code. What it does:
runs a subprocess and restarts it up to MAX_RETRIES times if it exits
with a non-zero code. WARNING/ERROR log lines are counted and reported,
but they do not trigger a retry: a warning used to re-run a multi-hour
--mode full collector that had actually succeeded. A step whose
dependency failed is skipped, so gold never builds on stale silver.
Real self-healing logic is deliberately out of scope here (a later phase).

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

            warnings = 0
            last_problem = ""
            for line in iter(process.stdout.readline, ""):
                line = line.strip()
                if line:
                    print(line)
                    # Match the logger's own level marker (see logging_setup.py's
                    # "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"),
                    # not a bare keyword search - a summary line like
                    # bddk_finturk's "Errors / failures : 0" merely mentions
                    # errors. "Traceback" is kept since an unhandled exception
                    # never goes through the logger format at all.
                    if " - WARNING - " in line or " - ERROR - " in line or line.startswith("Traceback"):
                        warnings += 1
                        last_problem = line

            process.wait()

            if warnings:
                logger.warning(f"'{module_path}' logged {warnings} warning/error line(s); last: {last_problem}")
            if process.returncode != 0:
                logger.error(f"'{module_path}' exited with code {process.returncode}. Retrying...")
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


def build_pipeline(mode: str) -> list[tuple[str, list, tuple[str, ...]]]:
    """Full bronze -> silver -> gold pipeline, in dependency order, as
    (module, args, depends_on). Not every module takes the same CLI -
    EVDS/TUIK collectors and every silver/gold module either take no
    --mode (they always process whatever is configured/already in bronze)
    or no args at all, so --mode is forwarded only to the modules that
    actually accept it.

    A silver step depends on its own collector and gold depends on every
    silver step: building on top of a failed step would publish stale data
    as if it were fresh."""
    mode_args = ["--mode", mode]
    silver = (
        "src.processing.bddk_monthly_silver",
        "src.processing.bddk_weekly_silver",
        "src.processing.bddk_finturk_silver",
        "src.processing.evds_silver",
        "src.processing.tuik_silver",
    )
    return [
        # Bronze (network) then its matching silver (network-free) step,
        # source by source - keeps a source's own data self-contained if
        # only part of the pipeline needs re-running.
        ("src.collectors.bddk_monthly", mode_args, ()),
        ("src.processing.bddk_monthly_silver", mode_args, ("src.collectors.bddk_monthly",)),
        ("src.collectors.bddk_weekly", mode_args, ()),
        ("src.processing.bddk_weekly_silver", mode_args, ("src.collectors.bddk_weekly",)),
        ("src.collectors.bddk_finturk", mode_args, ()),
        ("src.processing.bddk_finturk_silver", mode_args, ("src.collectors.bddk_finturk",)),
        ("src.collectors.evds", [], ()),          # no --mode: always (re)fetches every series in config/evds_series.py
        ("src.processing.evds_silver", [], ("src.collectors.evds",)),
        ("src.collectors.tuik", [], ()),          # no --mode: always (re)fetches every series in config/tuik_series.py
        ("src.processing.tuik_silver", [], ("src.collectors.tuik",)),
        ("src.gold.scan_cumulative", [], silver),
        ("src.gold.build_gold", [], (*silver, "src.gold.scan_cumulative")),
    ]


def run_pipeline(pipeline: list[tuple[str, list, tuple[str, ...]]], runner=run_module) -> list[tuple[str, str]]:
    """Returns (module, status) with status OK, FAIL or SKIP. A step is
    skipped when any dependency did not end OK."""
    status: dict[str, str] = {}
    for module_path, extra_args, depends_on in pipeline:
        blocked = [dep for dep in depends_on if status.get(dep) != "OK"]
        if blocked:
            logger.error(f"Skipping '{module_path}': dependency failed or skipped: {blocked}")
            status[module_path] = "SKIP"
            continue
        status[module_path] = "OK" if runner(module_path, extra_args) else "FAIL"
    return list(status.items())


if __name__ == "__main__":
    args = parse_args()
    pipeline = build_pipeline(args.mode)

    logger.info(f"Starting orchestrator (mode={args.mode}, {len(pipeline)} steps)...")
    results = run_pipeline(pipeline)

    failed = [name for name, state in results if state != "OK"]
    logger.info("=" * 50)
    logger.info("PIPELINE SUMMARY")
    for name, state in results:
        logger.info(f"  {state:<4} - {name}")
    logger.info(f"{len(results) - len(failed)}/{len(results)} steps succeeded.")
    logger.info("=" * 50)
    if failed:
        raise SystemExit(1)
