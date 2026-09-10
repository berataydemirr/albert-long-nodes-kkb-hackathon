"""CLI entry point: python agent.py "sorunuz" """

import argparse
import sys

from src.agent.orchestrator import ask

# Windows console defaults to a locale codepage (e.g. cp1254) that can't
# encode every character the model might return (seen live: '->' as U+2192
# crashed print() with UnicodeEncodeError, losing the whole answer). Same
# class of bug as the CSV encoding issue elsewhere in this repo - always
# force UTF-8 at the boundary instead of trusting the OS default.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="KKB MIA Agent")
    parser.add_argument("question", nargs="+", help="Natural-language analytics question")
    args = parser.parse_args()
    print(ask(" ".join(args.question)))


if __name__ == "__main__":
    main()
