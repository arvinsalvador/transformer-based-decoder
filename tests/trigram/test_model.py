"""Hand-computable Phase 5 count, smoothing, scoring, and persistence checks."""

import math

from src.trigram.model import TrigramModel
from src.trigram.serialization import load_model, save_model


def test_document_boundaries_counts_smoothing_and_eos(tmp_path):
    model = TrigramModel(10, 1, 2, 0.1)
    model.update([3, 4, 5])
    model.update([3, 4, 6])
    assert model.trigrams[(1, 1, 3)] == 2
    assert model.trigrams[(3, 4, 5)] == 1 and model.trigrams[(3, 4, 6)] == 1
    assert model.trigrams[(5, 1, 1)] == 0
    assert model.trigrams[(4, 5, 2)] == 1
    assert model.probability(3, 4, 5) == (1.1 / 3)
    assert model.probability(9, 9, 9) > 0
    score = model.score_sequences([[3, 4, 5]])
    assert score.events == 4 and score.average_nll == -score.log_likelihood / 4
    assert score.perplexity == math.exp(score.average_nll)
    path = save_model(model, tmp_path / "counts.sqlite")
    loaded = load_model(path)
    assert (
        loaded.trigrams == model.trigrams
        and loaded.score_sequences([[3, 4]]).perplexity
        == model.score_sequences([[3, 4]]).perplexity
    )


def test_short_prompt_and_seeded_sampling():
    model = TrigramModel(10, 1, 2, 0.1)
    model.update([3, 4, 5])
    model.update([3, 4, 6])
    assert model.generate([], max_new_tokens=2)[0]
    first = model.generate([3], max_new_tokens=3, strategy="sample", seed=42)
    second = model.generate([3], max_new_tokens=3, strategy="sample", seed=42)
    assert first[0] == second[0]
