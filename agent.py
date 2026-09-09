"""CLI entry point: python agent.py "sorunuz""" 

import argparse

from src.agent.orchestrator import ask


def main() -> None:
    parser = argparse.ArgumentParser(description="KKB MIA Agent")
    parser.add_argument("question", nargs="+", help="Natural-language analytics question")
    args = parser.parse_args()
    print(ask(" ".join(args.question)))


if __name__ == "__main__":
    main()
