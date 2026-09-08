"""Create a clean canonical corpus and deterministic splits from Phase 2 JSONL."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import PROJECT_ROOT, ConfigurationError, load_settings  # noqa: E402
from src.data.preprocessing.pipeline import prepare_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Run preparation; output replacement always requires explicit --overwrite."""
    parser = argparse.ArgumentParser(
        description="Normalize, filter, deduplicate and split Phase 2 JSONL."
    )
    parser.add_argument("--input", required=True, help="Phase 2 JSONL input")
    parser.add_argument("--clean-output", help="New clean JSONL path")
    parser.add_argument("--split-dir", help="Directory for canonical split files")
    parser.add_argument("--config", help="YAML profile; otherwise CONFIG_PATH/APP_ENV")
    parser.add_argument("--limit", type=int, help="Lower the configured input record cap")
    parser.add_argument("--seed", type=int, help="Override global random_seed for this run")
    parser.add_argument("--no-split", action="store_true", help="Write clean JSONL only")
    parser.add_argument(
        "--overwrite", action="store_true", help="Explicitly replace existing output files"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        settings = load_settings(args.config)
        input_path = PROJECT_ROOT / args.input if not Path(args.input).is_absolute() else args.input
        clean = (
            PROJECT_ROOT / args.clean_output
            if args.clean_output and not Path(args.clean_output).is_absolute()
            else args.clean_output
        )
        split = (
            PROJECT_ROOT / args.split_dir
            if args.split_dir and not Path(args.split_dir).is_absolute()
            else args.split_dir
        )
        result = prepare_dataset(
            input_path,
            settings,
            clean_output=clean,
            split_dir=split,
            limit=args.limit,
            seed=args.seed,
            make_splits=not args.no_split,
            overwrite=args.overwrite,
        )
        print(json.dumps({"manifest": str(result.manifest_path), **result.summary}, indent=2))
        return 0
    except (ConfigurationError, ValueError, FileExistsError) as exc:
        logging.error("Invalid preparation request: %s", exc)
        return 2
    except OSError as exc:
        logging.error("Preparation I/O failure (%s)", type(exc).__name__)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
