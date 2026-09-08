import csv
import json
import math

import pytest
import torch

from src.config.evaluation import validate_evaluation
from src.evaluation import service
from src.evaluation.adapters import score_transformer, synchronize
from src.evaluation.data import documents
from src.evaluation.generation import transformer_generate
from src.evaluation.identity import read_json, training_metadata
from src.evaluation.metrics import ModelEvaluationResult, compare, ratio
from src.evaluation.service import EvaluationFailure, evaluate_models
from src.tokenizer.service import load_tokenizer
from src.training.data import sequence_windows
from src.training.reproducibility import file_hash
from src.transformer.config import TransformerConfig
from src.transformer.model import DecoderOnlyTransformer
from src.trigram.serialization import load_model


def execute(case, **kwargs):
    settings, test, tokenizer, manifest = case
    return evaluate_models(settings, test, tokenizer, dataset_manifest=manifest, **kwargs)


def test_shared_integration_readonly_exports_and_limits(evaluation_case):
    settings, test, token_dir, _ = evaluation_case
    paths = list(settings.paths["MODEL_DIR"].rglob("*")) + [test]
    before = {p: file_hash(p) for p in paths if p.is_file()}
    result = execute(evaluation_case)
    assert result["status"] == "COMPLETED" and result["scope"] == "FULL TEST SET"
    from pathlib import Path

    directory = Path(result["run_directory"])
    comparison = read_json(directory / "comparison.json")
    tri, transformer = (comparison["results"][name] for name in ("trigram", "transformer"))
    tokenizer = load_tokenizer(token_dir)
    expected = sum(len(ids) + 1 for ids in documents(test, tokenizer))
    assert tri["prediction_events"] == transformer["prediction_events"] == expected
    assert tri["documents"] == transformer["documents"] == 3
    assert (
        tri["model_size_bytes"]
        == (settings.paths["MODEL_DIR"] / "trigram/trigram_counts.sqlite").stat().st_size
    )
    assert (
        transformer["model_size_bytes"]
        == (settings.paths["MODEL_DIR"] / "transformer/best_model.pt").stat().st_size
    )
    assert all(file_hash(p) == value for p, value in before.items())
    generated = read_json(directory / "generation_results.json")["results"]
    assert generated[0]["prompt"] == generated[1]["prompt"]
    assert generated[0]["prompt_token_count"] == generated[1]["prompt_token_count"]
    for item in generated:
        assert item["duration_seconds"] >= 0 and item["generated_token_count"] <= 3
        assert "[PAD]" not in item["generated_text"] and "[BOS]" not in item["generated_text"]
        assert all(r["rating"] is None for r in item["human_review"].values())
    with (directory / "comparison.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert all(row["evaluation_id"] == result["evaluation_id"] for row in rows)
    limited = execute(evaluation_case, limit_documents=1)
    assert limited["scope"] == "DEVELOPMENT SUBSET" and limited["test_documents"] == 1
    assert limited["document_limit"] == 1


def test_exact_window_targets_once():
    ids = list(range(10, 47))
    targets = [
        target for window in sequence_windows(ids, 2, 3, 8, 8) for target in window["labels"]
    ]
    assert targets == [*ids, 3]
    assert list(sequence_windows([10, 11, 12], 2, 3, 8, 8))[0] == {
        "input_ids": [2, 10, 11, 12],
        "labels": [10, 11, 12, 3],
    }


def test_hand_metrics_and_overflow():
    result = ModelEvaluationResult("synthetic", 1, 2, math.log(8), 2, "cpu", "fp32", {}).to_dict()
    assert result["average_nll"] == pytest.approx(math.log(8) / 2)
    assert result["perplexity"] == pytest.approx(math.sqrt(8))
    assert result["bits_per_token"] == pytest.approx(1.5)
    assert result["total_log_likelihood"] == -math.log(8)
    assert math.isinf(
        ModelEvaluationResult("x", 1, 1, 1000, 1, "cpu", "fp32", {}).to_dict()["perplexity"]
    )
    assert ratio(1, 0) is None and ratio(None, 1) is None and ratio(2, 1) == 2
    assert training_metadata({})["training_duration_seconds"] is None
    assert training_metadata({"training_duration_seconds": 3})["training_duration_seconds"] == 3
    with pytest.raises(ValueError):
        training_metadata({"training_duration_seconds": "bad"})


@pytest.mark.parametrize(
    "field,value",
    [
        ("tokenizer_fingerprint", "bad"),
        ("dataset_fingerprint", "other"),
        ("architecture_fingerprint", "bad"),
        ("vocabulary_size", 999),
    ],
)
def test_mismatch_rejected(evaluation_case, field, value):
    settings = evaluation_case[0]
    path = settings.paths["MODEL_DIR"] / "transformer/model_manifest.json"
    data = read_json(path)
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(EvaluationFailure) as caught:
        execute(evaluation_case)
    assert caught.value.summary["status"] == "FAILED_FINGERPRINT"


def test_test_replacement_rejected(evaluation_case):
    evaluation_case[1].write_text('{"text":"changed"}\n')
    with pytest.raises(EvaluationFailure, match="Test split fingerprint"):
        execute(evaluation_case)


def test_derived_dataset_fingerprint_rejected(evaluation_case):
    path = evaluation_case[0].paths["MODEL_DIR"] / "transformer/model_manifest.json"
    manifest = read_json(path)
    manifest.update(dataset_fingerprint_source="train_validation_pair", dataset_fingerprint="bad")
    path.write_text(json.dumps(manifest))
    with pytest.raises(EvaluationFailure, match="pair fingerprint"):
        execute(evaluation_case)


def test_missing_model_is_load_failure(evaluation_case):
    with pytest.raises(EvaluationFailure) as caught:
        execute(evaluation_case, transformer_path=evaluation_case[1].parent / "missing")
    assert caught.value.summary["status"] == "FAILED_MODEL_LOAD"


def test_midrun_test_mutation_rejected(evaluation_case, monkeypatch):
    original = service.score_trigram

    def change(*args, **kwargs):
        result = original(*args, **kwargs)
        evaluation_case[1].write_text('{"text":"replacement"}\n')
        return result

    monkeypatch.setattr(service, "score_trigram", change)
    with pytest.raises(EvaluationFailure, match="Test split changed"):
        execute(evaluation_case)


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_new_tokens", 0),
        ("top_k", 0),
        ("strategy", "bad"),
        ("temperature", float("nan")),
        ("temperature", 0),
    ],
)
def test_invalid_generation_config(key, value):
    from src.config.settings import load_settings

    cfg = load_settings("config/local.yaml").values["evaluation"]
    cfg["generation"][key] = value
    with pytest.raises(ValueError):
        validate_evaluation(cfg)


def test_transformer_readonly_batch_invariance(evaluation_case):
    settings, test, token_dir, _ = evaluation_case
    manifest = read_json(settings.paths["MODEL_DIR"] / "transformer/model_manifest.json")
    model = DecoderOnlyTransformer(TransformerConfig(**manifest["architecture"]))
    before = {key: value.clone() for key, value in model.state_dict().items()}
    tokenizer = load_tokenizer(token_dir)
    special = read_json(token_dir / "tokenizer_manifest.json")["special_tokens"]
    cfg = settings.values["evaluation"]
    first = score_transformer(model, tokenizer, test, None, special, {}, cfg, "cpu", "fp32")
    second = score_transformer(
        model,
        tokenizer,
        test,
        None,
        special,
        {},
        {**cfg, "transformer_batch_size": 1},
        "cpu",
        "fp32",
    )
    assert first["average_nll"] == pytest.approx(second["average_nll"], abs=1e-6)
    assert not model.training and all(p.grad is None for p in model.parameters())
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    with load_model(settings.paths["MODEL_DIR"] / "trigram/trigram_counts.sqlite") as trigram:
        with pytest.raises(Exception, match="readonly"):
            trigram.connection.execute("DELETE FROM trigram")


def test_generation_rollover_eos_and_greedy():
    class Model:
        config = type("Config", (), {"context_length": 4})()

        def eval(self):
            return self

        def __call__(self, ids):
            assert ids.shape[1] <= 4 and not torch.is_grad_enabled()
            logits = torch.zeros(1, ids.shape[1], 6)
            logits[..., self.target] = 10
            return logits

    model = Model()
    model.target = 4
    special = {"bos_token_id": 2, "eos_token_id": 3, "pad_token_id": 0}
    cfg = {"max_new_tokens": 8, "strategy": "greedy", "temperature": 1, "top_k": 2}
    first, timing = transformer_generate(model, [4] * 10, special, cfg, "cpu", "fp32", 42)
    assert first == [4] * 8 and not timing["stopped_on_eos"]
    assert first == transformer_generate(model, [4] * 10, special, cfg, "cpu", "fp32", 42)[0]
    model.target = 3
    assert transformer_generate(model, [], special, cfg, "cpu", "fp32", 42)[1]["stopped_on_eos"]


def test_sync_mocked(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(device))
    synchronize("cpu")
    synchronize("cuda")
    assert calls == ["cuda"]


def test_incremental_stream(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text('{"text":"first"}\nINVALID\n')

    class Tokenizer:
        def encode(self, text, **kwargs):
            return type("Encoding", (), {"ids": [1]})()

    source = documents(path, Tokenizer())
    assert next(source) == [1]
    with pytest.raises(ValueError):
        next(source)
    assert list(documents(path, Tokenizer(), 1)) == [[1]]


@pytest.mark.parametrize(
    "error,status",
    [
        (RuntimeError("failure"), "FAILED_EVALUATION"),
        (torch.cuda.OutOfMemoryError("oom"), "FAILED_EVALUATION"),
        (KeyboardInterrupt(), "INTERRUPTED"),
    ],
)
def test_failures_not_completed(evaluation_case, monkeypatch, error, status):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(service, "score_transformer", fail)
    if status == "INTERRUPTED":
        result = execute(evaluation_case)
    else:
        with pytest.raises(EvaluationFailure) as caught:
            execute(evaluation_case)
        result = caught.value.summary
    assert result["status"] == status
    from pathlib import Path

    assert not (Path(result["run_directory"]) / "comparison.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("document_limit", 0),
        ("document_limit", True),
        ("transformer_batch_size", 0),
        ("gpu_warmup_batches", -1),
        ("precision", "bad"),
    ],
)
def test_invalid_config(field, value):
    from src.config.settings import load_settings

    cfg = load_settings("config/local.yaml").values["evaluation"]
    cfg[field] = value
    with pytest.raises(ValueError):
        validate_evaluation(cfg)


def test_relative_comparisons():
    base = dict(
        documents=1,
        prediction_events=4,
        test_fingerprint="a",
        tokenizer_fingerprint="b",
        perplexity=4,
        training_duration_seconds=2,
        model_size_bytes=10,
        tokens_per_second=5,
    )
    relative, _ = compare(
        base,
        {
            **base,
            "perplexity": 2,
            "training_duration_seconds": 6,
            "model_size_bytes": 20,
            "tokens_per_second": 10,
        },
    )
    assert relative == {
        "perplexity_reduction_percent": 50,
        "training_time_multiplier": 3,
        "model_size_multiplier": 2,
        "evaluation_throughput_ratio": 2,
    }
