""""Command-line entry point for the adaptation study."""

from __future__ import annotations

import argparse

from pretrained_llm_adaptation import __version__


def main() -> None:
    """Run the minimal project CLI."""
    parser = argparse.ArgumentParser(description="Pretrained LLM Adaptation")
    parser.add_argument("--version", action="version", version=__version__)
    parser.parse_args()


if __name__ == "__main__":
    main()
