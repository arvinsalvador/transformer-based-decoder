"""Train/save/load/inspect a tiny genuine WordPiece artifact in temporary storage."""

import json

import pytest
from tokenizers import models

from src.tokenizer.service import encode, inspect_vocabulary, load_tokenizer, train_wordpiece


def test_train_artifacts_encode_statistics(tokenizer_settings, splits):
    train, validation, test = splits
    output = tokenizer_settings.paths["MODEL_DIR"] / "tokenizer"
    result = train_wordpiece(
        train,
        tokenizer_settings,
        output_dir=output,
        validation_path=validation,
        test_path=test,
        dataset_fingerprint="dataset-fingerprint",
    )
    assert isinstance(load_tokenizer(output).model, models.WordPiece)
    assert (output / "tokenizer.json").is_file() and (output / "vocab.txt").is_file()
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["training_corpus_documents"] == 6
    assert manifest["dataset_fingerprint"] == "dataset-fingerprint"
    assert set(manifest["analysis"]) == {"train", "validation", "test"}
    assert all(
        key in manifest["special_tokens"]
        for key in ("pad_token_id", "unk_token_id", "bos_token_id", "eos_token_id")
    )
    loaded = load_tokenizer(output)
    plain = encode(loaded, "Cybersecurity protects APIs.")
    special = encode(loaded, "Cybersecurity protects APIs.", add_special_tokens=True)
    assert plain.ids and len(special.ids) == len(plain.ids) + 2
    assert loaded.decode(plain.ids)
    assert inspect_vocabulary(loaded, 5)[0][1] == 0
    assert (
        manifest["tokenizer_fingerprint"]
        == __import__("hashlib").sha256((output / "tokenizer.json").read_bytes()).hexdigest()
    )
    assert (
        load_tokenizer(output).encode("Software systems.").ids
        == loaded.encode("Software systems.").ids
    )


def test_train_only_and_overwrite(tokenizer_settings, splits, monkeypatch):
    train, validation, test = splits
    observed = []
    from src.tokenizer import service

    original = service.text_batches

    def tracked(path, batch):
        observed.append(path)
        yield from original(path, batch)

    monkeypatch.setattr(service, "text_batches", tracked)
    output = tokenizer_settings.paths["MODEL_DIR"] / "tokenizer"
    train_wordpiece(
        train, tokenizer_settings, output_dir=output, validation_path=validation, test_path=test
    )
    assert observed[0] == train
    with pytest.raises(FileExistsError):
        train_wordpiece(train, tokenizer_settings, output_dir=output)


def test_empty_training_set(tokenizer_settings, tmp_path):
    source = tmp_path / "train.jsonl"
    source.write_text("")
    with pytest.raises(Exception, match="no valid|no valid documents|contains no"):
        train_wordpiece(
            source, tokenizer_settings, output_dir=tokenizer_settings.paths["MODEL_DIR"] / "out"
        )


def test_rejects_noncanonical_training_split_and_external_output(
    tokenizer_settings, splits, tmp_path
):
    train, validation, _ = splits
    with pytest.raises(ValueError, match="canonical train.jsonl"):
        train_wordpiece(
            validation,
            tokenizer_settings,
            output_dir=tokenizer_settings.paths["MODEL_DIR"] / "tokenizer",
        )
    with pytest.raises(ValueError, match="MODEL_DIR"):
        train_wordpiece(train, tokenizer_settings, output_dir=tmp_path / "outside")


def test_optional_lowercasing_changes_encoding(tokenizer_settings, splits):
    train, _, _ = splits
    tokenizer_settings.values["tokenizer"]["lowercase"] = True
    output = tokenizer_settings.paths["MODEL_DIR"] / "tokenizer"
    tokenizer = load_tokenizer(
        train_wordpiece(train, tokenizer_settings, output_dir=output).output_dir
    )
    assert tokenizer.encode("CYBERSECURITY").tokens == tokenizer.encode("cybersecurity").tokens
