"""Train the CPU-only trigram baseline from canonical artifacts."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config.settings import PROJECT_ROOT, ConfigurationError, load_settings  # noqa: E402


def path(value):
    return (
        (PROJECT_ROOT / value).resolve()
        if value and not Path(value).is_absolute()
        else Path(value).resolve()
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a WordPiece-ID trigram baseline.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", default="models/trigram")
    parser.add_argument("--validation")
    parser.add_argument("--config")
    parser.add_argument("--dataset-fingerprint")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        from src.trigram.trainer import train

        print(
            json.dumps(
                train(
                    path(args.train),
                    path(args.tokenizer),
                    load_settings(args.config),
                    output_dir=path(args.output),
                    validation_path=path(args.validation) if args.validation else None,
                    dataset_fingerprint=args.dataset_fingerprint,
                    overwrite=args.overwrite,
                ),
                indent=2,
            )
        )
        return 0
    except (ConfigurationError, OSError, ValueError, FileExistsError) as exc:
        logging.error("Trigram training failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
