"""Command-line entry point for the adaptation study."""
from __future__ import annotations
import argparse
from pretrained_llm_adaptation import __version__

def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrained LLM Adaptation")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("version", help="Print project version")
    args = parser.parse_args()
    if args.command == "version":
        print(__version__)

if __name__ == "__main__":
    main()
