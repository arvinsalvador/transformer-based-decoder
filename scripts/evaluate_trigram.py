"""Read-only trigram split scoring."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config.settings import PROJECT_ROOT, load_settings  # noqa: E402


def path(value):
    return (
        (PROJECT_ROOT / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    )


parser = argparse.ArgumentParser(description="Score a split without updating trigram counts.")
parser.add_argument("--model", required=True)
parser.add_argument("--test", required=True)
parser.add_argument("--tokenizer", required=True)
parser.add_argument("--config")
args = parser.parse_args()
from src.tokenizer.service import load_tokenizer  # noqa: E402
from src.trigram.serialization import load_model  # noqa: E402
from src.trigram.trainer import score  # noqa: E402

with load_model(path(args.model) / "trigram_counts.sqlite") as model:
    print(
        json.dumps(
            score(
                model,
                load_tokenizer(path(args.tokenizer)),
                path(args.test),
                load_settings(args.config),
            ),
            indent=2,
        )
    )
