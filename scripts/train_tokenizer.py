"""Train WordPiece only on the Phase 3 canonical training split."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import PROJECT_ROOT, ConfigurationError, load_settings  # noqa: E402
from src.tokenizer.service import train_wordpiece  # noqa: E402


def resolved(value: str | None) -> Path | None:
    """Resolve optional CLI paths relative to a checkout."""
    return (
        PROJECT_ROOT / value
        if value and not Path(value).is_absolute()
        else Path(value)
        if value
        else None
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a genuine WordPiece tokenizer from train JSONL only."
    )
    parser.add_argument("--train", required=True, help="Canonical Phase 3 train.jsonl only")
    parser.add_argument("--output", help="Tokenizer artifact directory")
    parser.add_argument("--validation", help="Optional validation JSONL for analysis only")
    parser.add_argument("--test", help="Optional test JSONL for analysis only")
    parser.add_argument("--dataset-manifest", help="Optional Phase 3 preprocessing manifest")
    parser.add_argument("--config", help="YAML profile; otherwise CONFIG_PATH/APP_ENV")
    parser.add_argument(
        "--overwrite", action="store_true", help="Explicitly replace tokenizer artifacts"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        settings = load_settings(args.config)
        fingerprint = None
        manifest_path = resolved(args.dataset_manifest)
        if manifest_path:
            fingerprint = json.loads(manifest_path.read_text(encoding="utf-8")).get(
                "dataset_fingerprint"
            )
        result = train_wordpiece(
            resolved(args.train),
            settings,
            output_dir=resolved(args.output),
            validation_path=resolved(args.validation),
            test_path=resolved(args.test),
            dataset_fingerprint=fingerprint,
            overwrite=args.overwrite,
        )
        print(
            json.dumps(
                {
                    "output": str(result.output_dir),
                    "manifest": str(result.manifest_path),
                    **result.manifest,
                },
                indent=2,
            )
        )
        return 0
    except (
        ConfigurationError,
        ValueError,
        FileExistsError,
        FileNotFoundError,
        json.JSONDecodeError,
    ) as exc:
        logging.error("Invalid tokenizer request: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
