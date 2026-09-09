"""Read-only audit and final FULL report; missing results produce readiness output."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--output", help="Report directory beneath REPORT_DIR")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--experiment", help="Explicit Phase 9 experiment ID")
    modes.add_argument("--latest-complete", action="store_true")
    modes.add_argument("--readiness-only", action="store_true")
    args = parser.parse_args(argv)
    from src.config.settings import PROJECT_ROOT, load_settings
    from src.reporting.report import export_report

    try:
        settings = load_settings(args.config)
        result = export_report(
            settings,
            (PROJECT_ROOT / args.output).resolve() if args.output else settings.paths["REPORT_DIR"],
            experiment=args.experiment,
            latest_complete=args.latest_complete,
            readiness_only=args.readiness_only,
        )
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "project_status",
                        "audit_status",
                        "compliance",
                        "recommendation",
                        "report_paths",
                    )
                },
                indent=2,
            )
        )
        return 1 if result["audit_status"] == "FAIL" else 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"Reporting failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
