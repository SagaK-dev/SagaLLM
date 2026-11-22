"""Early stopping tests: the test split cannot be used for selection."""
import json
import torch
import pytest

from sagallm.java_generated_data import build_java_dataset
from scripts.train_java_original import train


def test_best_checkpoint_and_test_separation(tmp_path):
    dataset = tmp_path / "dataset"
    build_java_dataset(dataset)
    out = tmp_path / "trained"
    result = train(dataset, out, steps=6, max_seconds=22, hidden_size=96, layers=2,
                   progress_every=2, compile_examples=0, patience_evals=1)
    assert result["steps_completed"] <= 6
    assert result["test_loss_before"] is None
    assert result["test_used_for_model_selection"] is False
    assert result["test_loss_evaluation_count"] == 1
    assert set(result["history"]) == {"step", "train_loss", "validation_loss"}
    assert result["selected_checkpoint_step"] <= result["steps_completed"]
    assert result["best_validation_loss"] <= min(result["history"]["validation_loss"]) + result["min_delta"] + 1e-9
    best = torch.load(out / "java-trained.pt", weights_only=True, map_location="cpu")
    last = torch.load(out / "java-last.pt", weights_only=True, map_location="cpu")
    assert best["step"] == result["selected_checkpoint_step"]
    assert last["step"] == result["steps_completed"]
    assert best["train_state"]["selection_rule"] == "lowest_validation_loss_only"
    assert last["train_state"]["selection_rule"] == "last_step_not_used_for_model_selection"
    assert result["test_loss_after"] >= 0
    assert result["compile_only_heldout"]["status"] == "NOT_RUN"
    assert json.loads((out / "train-report.json").read_text())["test_loss_evaluation_count"] == 1


def test_invalid_early_stop_budget_rejected(tmp_path):
    for kwargs in ({"patience_evals": -1}, {"patience_evals": 31},
                   {"min_delta": -0.1}, {"min_delta": 1.0}, {"progress_every": 0}):
        with pytest.raises(ValueError):
            train(tmp_path / "not-used", tmp_path / "out", steps=1, **kwargs)