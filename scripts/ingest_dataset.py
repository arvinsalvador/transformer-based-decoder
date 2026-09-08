"""Portable batch entry point; ingestion itself is independent of Streamlit and CUDA."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Support the requested `python scripts/ingest_dataset.py` invocation from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import PROJECT_ROOT, ConfigurationError, load_settings  # noqa: E402
from src.data.ingestion import ingest_directory  # noqa: E402
from src.data.models import IngestionOptions  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Return 0 for completed runs, 1 for partial/failed ingestion, 2 for invalid inputs."""
    parser = argparse.ArgumentParser(
        description="Extract raw TXT/CSV/PDF/DOCX into streaming JSONL."
    )
    parser.add_argument("--source", help="Corpus directory; default DATA_DIR/raw")
    parser.add_argument("--output", help="New JSONL path; existing output is never overwritten")
    parser.add_argument("--config", help="YAML profile; otherwise CONFIG_PATH/APP_ENV")
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Recurse through folders (default from profile)",
    )
    parser.add_argument(
        "--limit", type=int, help="Lower the configured working limit; never above 100000"
    )
    parser.add_argument("--csv-mode", choices=["rows", "file"])
    parser.add_argument("--text-columns", nargs="+", help="CSV text columns in concatenation order")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        settings = load_settings(args.config)
        options = IngestionOptions.from_settings(
            settings,
            limit=args.limit,
            recursive=args.recursive,
            csv_mode=args.csv_mode,
            text_columns=args.text_columns,
        )
        source = PROJECT_ROOT / args.source if args.source else settings.paths["DATA_DIR"] / "raw"
        output = PROJECT_ROOT / args.output if args.output else options.output_path
        result = ingest_directory(source, options, output=output)
        print(json.dumps({"manifest": str(result.manifest_path), **result.summary}, indent=2))
        return 1 if result.summary["failed"] else 0
    except (ConfigurationError, ValueError, FileExistsError) as exc:
        logging.error("Invalid ingestion request: %s", exc)
        return 2
    except OSError as exc:
        logging.error("Ingestion I/O failure (%s)", type(exc).__name__)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
