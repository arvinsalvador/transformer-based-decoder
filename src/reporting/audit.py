"""Static source inventory and bounded runtime readiness; no Git or training calls."""

import ast
import json
import re

from src.config.settings import PROJECT_ROOT
from src.evaluation.identity import read_json
from src.experiments.plan import load_plan
from src.experiments.preflight import audit_data
from src.training.reproducibility import file_hash, verify_inputs
from src.utils.device import DeviceUnavailableError, detect_device

EVIDENCE = {
    "decoder_only_transformer": [
        "src/transformer/model.py",
        "src/transformer/attention.py",
        "src/training/trainer.py",
    ],
    "total_corpus_cap": ["src/data/ingestion.py", "src/experiments/preflight.py"],
    "wordpiece": ["src/tokenizer/service.py", "src/tokenizer/corpus.py"],
    "trigram_comparison": [
        "src/trigram/model.py",
        "src/trigram/sqlite_model.py",
        "src/evaluation/identity.py",
    ],
    "performance_training_time": ["src/evaluation/metrics.py", "src/experiments/registry.py"],
}
POLICIES = {
    "deduplicate_before_split": "src/data/preprocessing/pipeline.py",
    "tokenizer_train_only": "src/tokenizer/corpus.py",
    "trigram_train_only": "src/trigram/trainer.py",
    "transformer_train_only": "src/training/reproducibility.py",
    "validation_selects_best": "src/training/trainer.py",
    "fixed_validation_test_and_technical_gates": "src/experiments/runner.py",
    "test_readonly_scoring": "src/evaluation/service.py",
}


def source_audit(root=PROJECT_ROOT):
    inventory, findings = [], []
    intended = [root / name for name in ("src", "app", "scripts", "config", "docs", "tests")]
    for directory in intended:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = str(path.relative_to(root))
            inventory.append(
                {"path": relative, "bytes": path.stat().st_size, "sha256": file_hash(path)}
            )
            if (
                path.stat().st_size > 5 * 2**20
                or path.suffix in (".pt", ".pth", ".sqlite", ".pdf", ".docx")
                or path.name == "tokenizer.json"
            ):
                findings.append(
                    {
                        "status": "WARNING",
                        "path": relative,
                        "issue": "Possible runtime/large file in source area; review manually",
                    }
                )
            if path.suffix == ".py":
                text = path.read_text(encoding="utf-8")
                try:
                    ast.parse(text)
                except SyntaxError:
                    findings.append(
                        {"status": "FAIL", "path": relative, "issue": "Invalid Python syntax"}
                    )
                if directory.name in ("src", "app", "scripts"):
                    for number, line in enumerate(text.splitlines(), 1):
                        if re.search(r"/home/[A-Za-z0-9_-]+/|/mnt/" r"c/|[A-Z]:\\\\", line):
                            findings.append(
                                {
                                    "status": "WARNING",
                                    "path": relative,
                                    "line": number,
                                    "issue": "Possible workstation-specific path",
                                }
                            )
                        if re.search(
                            r"(?:api_key|password|secret_key|access_token)\s*=\s*['\"][^'\"]{8,}",
                            line,
                            re.I,
                        ):
                            findings.append(
                                {
                                    "status": "WARNING",
                                    "path": relative,
                                    "line": number,
                                    "issue": "Possible embedded credential; value suppressed",
                                }
                            )
    implementation = {
        name: {
            "status": "PASS" if all((root / p).is_file() for p in paths) else "FAIL",
            "evidence": paths,
            "assurance": "Source presence/syntax and code policy; not proof of unseen execution",
        }
        for name, paths in EVIDENCE.items()
    }
    ignore = (root / ".gitignore").read_text() if (root / ".gitignore").exists() else ""
    required = [
        "/data/**",
        "/models/**",
        "/checkpoints/**",
        "/experiments/**",
        "/reports/**",
        ".env",
        "*.pt",
        "*.pth",
        "*.jsonl",
        "*.sqlite",
    ]
    missing = [pattern for pattern in required if pattern not in ignore.splitlines()]
    if missing:
        findings.append(
            {
                "status": "FAIL",
                "path": ".gitignore",
                "issue": f"Missing runtime patterns: {missing}",
            }
        )
    files = [
        "README.md",
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        "Dockerfile",
        "docker-compose.yml",
        "docker-compose.gpu.yml",
        ".gitignore",
        ".dockerignore",
        "Makefile",
    ]
    for name in files:
        if not (root / name).is_file():
            findings.append(
                {"status": "FAIL", "path": name, "issue": "Required reproducibility file missing"}
            )
        else:
            inventory.append(
                {
                    "path": name,
                    "bytes": (root / name).stat().st_size,
                    "sha256": file_hash(root / name),
                }
            )
    return {
        "status": "FAIL"
        if any(f["status"] == "FAIL" for f in findings)
        or any(i["status"] == "FAIL" for i in implementation.values())
        else "PASS",
        "implementation": implementation,
        "inventory": inventory,
        "findings": findings,
        "git_operations": "NONE",
        "ignore_assurance": "Conceptual pattern audit; no Git tracked-file inspection",
        "tests": "NOT_RUN_BY_AUDITOR; execute full regression separately",
        "leakage_policy": {
            key: {
                "status": "PASS" if (root / path).exists() else "FAIL",
                "evidence": path,
                "assurance": "CODE_POLICY_ONLY",
            }
            for key, path in POLICIES.items()
        },
        "unobservable_policy": {
            "test_not_used_for_manual_tuning": (
                "WARNING: requires experimenter attestation; "
                "artifacts cannot prove unseen decisions"
            )
        },
    }


def runtime_audit(settings):
    data = {"status": "NOT_RUN", "counts": None, "message": "Canonical splits not generated"}
    token = {"status": "NOT_RUN", "message": "Canonical tokenizer not generated"}
    splits = settings.paths["DATA_DIR"] / "splits"
    if all((splits / f"{name}.jsonl").exists() for name in ("train", "validation", "test")):
        try:
            data = {
                "status": "PASS",
                **audit_data(settings, load_plan(PROJECT_ROOT / "config/experiments.yaml")),
            }
        except (OSError, ValueError, KeyError) as exc:
            data = {"status": "FAIL", "counts": None, "message": str(exc)}
    elif any(splits.glob("*.jsonl")):
        data["status"] = "WARNING"
        data["message"] = "Only some canonical splits exist"
    directory = settings.paths["MODEL_DIR"] / "tokenizer"
    if (directory / "tokenizer.json").exists():
        try:
            manifest = read_json(directory / "tokenizer_manifest.json")
            from src.tokenizer.service import load_tokenizer

            tokenizer = load_tokenizer(directory)
            if (
                manifest.get("tokenizer_type") != "WordPiece"
                or json.loads(tokenizer.to_str())["model"]["type"] != "WordPiece"
                or manifest.get("tokenizer_fingerprint") != file_hash(directory / "tokenizer.json")
                or tokenizer.get_vocab_size() != manifest["actual_vocabulary_size"]
            ):
                raise ValueError("WordPiece type/vocabulary/fingerprint mismatch")
            if data["status"] == "PASS":
                verify_inputs(
                    splits / "train.jsonl",
                    splits / "validation.jsonl",
                    directory,
                    settings,
                    data["dataset_manifest"],
                )
            token = {
                "status": "PASS" if data["status"] == "PASS" else "WARNING",
                "provenance": "Verified train-only"
                if data["status"] == "PASS"
                else "Canonical data unavailable; training provenance not verified",
                "vocabulary_size": tokenizer.get_vocab_size(),
                "fingerprint": manifest["tokenizer_fingerprint"],
            }
        except (OSError, ValueError, KeyError) as exc:
            token = {"status": "FAIL", "message": str(exc)}
    try:
        from dataclasses import asdict

        device = {"status": "PASS", **asdict(detect_device(settings.values["device"]))}
    except DeviceUnavailableError as exc:
        device = {"status": "NOT_RUN", "message": str(exc)}
    return {"data": data, "tokenizer": token, "device": device}
