"""Read-only shared likelihood, generation and model comparison."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("test", "tokenizer"):
        parser.add_argument("--" + name, required=True)
    for name in (
        "trigram-model",
        "transformer-model",
        "config",
        "output",
        "prompts",
        "dataset-manifest",
    ):
        parser.add_argument("--" + name)
    parser.add_argument("--model", choices=("both", "trigram", "transformer"), default="both")
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument("--limit-documents", type=int)
    limits.add_argument(
        "--full-test", action="store_true", help="Explicitly override a profile development limit"
    )
    args = parser.parse_args(argv)
    from src.config.settings import PROJECT_ROOT, load_settings
    from src.evaluation.service import EvaluationFailure, evaluate_models

    def path(value):
        return (PROJECT_ROOT / value).resolve() if value else None

    try:
        result = evaluate_models(
            load_settings(args.config),
            path(args.test),
            path(args.tokenizer),
            trigram_path=path(args.trigram_model),
            transformer_path=path(args.transformer_model),
            output=path(args.output),
            model=args.model,
            limit_documents=args.limit_documents,
            full_test=args.full_test,
            prompts_path=path(args.prompts),
            dataset_manifest=path(args.dataset_manifest),
        )
        print(json.dumps(result, indent=2))
        return 130 if result["status"] == "INTERRUPTED" else 0
    except EvaluationFailure as exc:
        print(json.dumps(exc.summary, indent=2), file=sys.stderr)
        return 1
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
