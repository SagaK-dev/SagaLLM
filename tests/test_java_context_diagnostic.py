"""Unit tests for the v0.18 report-only regression analyzer."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_java_context_regression.py"
spec = importlib.util.spec_from_file_location("java_context_diagnostic", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def sample():
    return {
        "status": "TRAINED_CPU",
        "model_context_length": 384,
        "best_checkpoint_step": 50,
        "compile_only_base": {
            "evaluated": 2, "compiled": 1,
            "cases": [
                {"id": "a", "family": "one", "compile": "PASS", "generated_sha256": "1"},
                {"id": "b", "family": "two", "compile": "FAIL", "generated_sha256": "2"},
            ],
        },
        "compile_only_best": {
            "evaluated": 2, "compiled": 0,
            "cases": [
                {"id": "a", "family": "one", "compile": "FAIL", "generated_sha256": "3"},
                {"id": "b", "family": "two", "compile": "FAIL", "generated_sha256": "4"},
            ],
        },
        "history": [
            {"step": 50, "validation_response_loss": 1.7},
            {"step": 100, "validation_response_loss": 1.9},
        ],
        "heldout_test_response_loss_last_sft": 2.2,
    }


def test_regression_summary():
    result = module.analyze(sample())
    assert result["regressed"] == 1
    assert result["improved"] == 0
    assert result["generation_changed"] == 2
    assert result["root_cause"] == "NOT_ESTABLISHED"
    assert result["best_validation_step"] == 50
    assert result["generated_code_execution"] == "NOT_RUN"


@pytest.mark.parametrize("mutation", [
    lambda r: r["compile_only_best"]["cases"][0].update(id="b"),
    lambda r: r["compile_only_best"]["cases"][0].update(family="wrong"),
    lambda r: r["compile_only_base"].update(compiled=2),
    lambda r: r["history"][0].update(validation_response_loss=2.0),
    lambda r: r["compile_only_best"]["cases"][0].update(compile="UNKNOWN"),
])
def test_rejects_inconsistent_reports(mutation):
    report = copy.deepcopy(sample())
    mutation(report)
    with pytest.raises(ValueError):
        module.analyze(report)
