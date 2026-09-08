"""Hash buckets are stable and validate different seeds without retaining records."""

from src.data.preprocessing.splitter import assign_split


def test_same_seed_is_stable(prep_settings):
    digest = "a" * 64
    ratios = prep_settings.values["split"]
    assert assign_split(digest, 42, ratios) == assign_split(digest, 42, ratios)


def test_different_seed_can_change_assignment(prep_settings):
    ratios = prep_settings.values["split"]
    assignments = {
        assign_split(f"{number:064x}", seed, ratios) for number in range(100) for seed in (1, 2)
    }
    assert len(assignments) > 1
