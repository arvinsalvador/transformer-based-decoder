"""CLI for dry runs and single-device custom Transformer training."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--config")
    parser.add_argument("--output", help="Experiment parent beneath EXPERIMENT_DIR")
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-name", help="Display label; never used as a path")
    parser.add_argument("--overwrite-export", action="store_true")
    parser.add_argument("--dataset-manifest", help="Optional completed Phase 3 manifest")
    args = parser.parse_args(argv)
    from src.config.settings import PROJECT_ROOT, load_settings
    from src.training.trainer import TrainingFailure, run_training

    def path(value):
        return (PROJECT_ROOT / value).resolve() if value else None

    try:
        result = run_training(
            load_settings(args.config),
            path(args.train),
            path(args.validation),
            path(args.tokenizer),
            output=path(args.output),
            resume=path(args.resume),
            max_steps=args.max_steps,
            epochs=args.epochs,
            dry_run=args.dry_run,
            run_name=args.run_name,
            overwrite_export=args.overwrite_export,
            dataset_manifest=path(args.dataset_manifest),
        )
        print(json.dumps(result, indent=2))
        return 130 if result["status"] == "INTERRUPTED" else 0
    except TrainingFailure as exc:
        print(json.dumps(exc.summary, indent=2), file=sys.stderr)
        return 1
    except (ValueError, OSError) as exc:
        print(f"Invalid training request: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
