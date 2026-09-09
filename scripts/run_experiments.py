"""Plan-only by default; explicit, sequential and resumable experiment stages."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--experiment-config", default="config/experiments.yaml")
    parser.add_argument("--dataset-manifest")
    parser.add_argument("--output")
    parser.add_argument("--resume", help="Existing experiment ID beneath output")
    parser.add_argument("--confirm-large-run", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    for name in ("plan", "preflight", "dry-run-only", "execute"):
        modes.add_argument("--" + name, action="store_true")
    scales = parser.add_mutually_exclusive_group()
    scales.add_argument("--scale", action="append", help="Repeat for selected configured scales")
    scales.add_argument("--full-only", action="store_true")
    scales.add_argument("--all-scales", action="store_true")
    args = parser.parse_args(argv)
    from src.config.settings import PROJECT_ROOT, load_settings
    from src.experiments.plan import load_plan
    from src.experiments.runner import run_experiment

    def path(value):
        return (PROJECT_ROOT / value).resolve() if value else None

    try:
        mode = (
            "execute"
            if args.execute
            else "preflight"
            if args.preflight
            else "dry-run-only"
            if args.dry_run_only
            else "plan"
        )
        result = run_experiment(
            load_settings(args.config),
            load_plan(path(args.experiment_config)),
            mode=mode,
            selected=["full"] if args.full_only else args.scale,
            output=path(args.output),
            resume=args.resume,
            confirm_large_run=args.confirm_large_run,
            dataset_manifest=path(args.dataset_manifest),
        )
        print(json.dumps(result, indent=2))
        return (
            130
            if result["status"] == "INTERRUPTED"
            else 1
            if result["status"] in ("FAILED", "PREFLIGHT_FAILED") or result.get("failed_scale")
            else 0
        )
    except (ValueError, OSError, KeyError) as exc:
        print(f"Experiment request rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
