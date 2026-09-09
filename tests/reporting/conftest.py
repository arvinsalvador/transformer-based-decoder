import pytest

from src.experiments.runner import run_experiment
from tests.experiments import conftest as experiment_fixtures

training_case = experiment_fixtures.training_case
experiment_case = experiment_fixtures.experiment_case


@pytest.fixture
def completed_case(experiment_case):
    settings, plan, manifest = experiment_case
    result = run_experiment(
        settings,
        plan,
        mode="execute",
        selected=["full"],
        confirm_large_run=True,
        dataset_manifest=manifest,
    )
    assert result["scales"][-1]["status"] == "COMPLETED", result
    return settings, result
