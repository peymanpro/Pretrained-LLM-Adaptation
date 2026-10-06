"""Command-line entry point for the adaptation study."""

from __future__ import annotations

import argparse
import json

from pretrained_llm_adaptation import __version__


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrained LLM Adaptation")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    version_parser = sub.add_parser("version", help="Print project version")
    version_parser.set_defaults(handler=lambda _: print(__version__))

    predict_parser = sub.add_parser("predict", help="Classify one text with a saved adapter")
    predict_parser.add_argument("--model-dir", required=True)
    predict_parser.add_argument("--text", required=True)
    predict_parser.add_argument("--model-name", default="microsoft/deberta-v3-small")
    predict_parser.add_argument(
        "--revision",
        default="a59be8aa63396e73dbb45a1487e4cde4be98bfa4",
    )
    predict_parser.add_argument("--max-length", type=int, default=48)
    predict_parser.set_defaults(handler=_predict)

    args = parser.parse_args()
    handler = getattr(args, "handler", None)
    if handler is not None:
        handler(args)


def _predict(args: argparse.Namespace) -> None:
    from pretrained_llm_adaptation.inference import load_adapter_model, predict

    model, tokenizer = load_adapter_model(
        args.model_name,
        args.revision,
        args.model_dir,
    )
    result = predict(model, tokenizer, args.text, max_length=args.max_length)
    print(
        json.dumps(
            {
                "model": args.model_name,
                "revision": args.revision,
                "adapter": args.model_dir,
                "intent": result.intent,
                "confidence": result.confidence,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
