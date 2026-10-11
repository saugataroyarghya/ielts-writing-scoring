"""Safety checks for augmentation inputs and conditional checkpoint promotion."""

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from ielts_scoring.synthetic_finetune import make_mixture, public_gates, synthetic_frame

ROOT = Path(__file__).resolve().parents[1]


def test_quarantine_cannot_enter_training(tmp_path):
    row = {"id": "q", "synthetic": True, "human_verified": False,
           "TR": None, "CC": None, "LR": None, "GRA": None}
    path = tmp_path / "quarantine.parquet"
    pd.DataFrame([row]).to_parquet(path)
    with pytest.raises(ValueError, match="accepted"):
        synthetic_frame(path)


def test_mixture_excludes_judgments_and_holdout_rows():
    def frame(ids, source):
        return pd.DataFrame([{"id": i, "source": source, "group_id": i, "prompt_group": i,
                              "prompt": "question", "essay": "response", "TR": 4, "CC": 5, "LR": 6, "GRA": 7,
                              "evidence": "target leak", "flags": "target leak", "intended_band": 9} for i in ids])
    original = frame(["a", "b", "c", "d"], "original")
    synthetic = frame(["s1", "s2"], "synthetic")
    mixed, replay = make_mixture(original, synthetic, 2, 42)
    assert len(mixed) == 4 and len(replay) == 2
    assert set(mixed.id) == set(replay) | {"s1", "s2"}
    assert not {"evidence", "flags", "intended_band"} & set(mixed.columns)
    assert set(mixed.essay) == {"response"}
    assert make_mixture(original, synthetic, 2, 42)[1] == replay


def test_better_synthetic_average_cannot_hide_original_criterion_regression():
    config = json.loads((ROOT / "configs/synthetic_moe.json").read_text())
    metrics = {"macro_mae": 1.0, **{key: {"mae": 1.0} for key in ["TR", "CC", "LR", "GRA"]}}
    baseline = {name: copy.deepcopy(metrics) for name in
                ["synthetic_diagnostic", "original_test", "original_validation"]}
    candidate = copy.deepcopy(baseline)
    candidate["synthetic_diagnostic"]["macro_mae"] = 0.5
    difference = {"synthetic_diagnostic": {"paired_prompt_group_95ci": [-0.7, -0.2]}}
    assert public_gates(baseline, candidate, difference, config)["public_evaluation_passed"]
    assert not public_gates(baseline, candidate, difference, config)["publication_ready"]
    candidate["original_test"]["TR"]["mae"] = 1.3
    assert not public_gates(baseline, candidate, difference, config)["public_evaluation_passed"]


def test_uncertain_synthetic_gain_does_not_pass():
    config = json.loads((ROOT / "configs/synthetic_moe.json").read_text())
    metrics = {"macro_mae": 1.0, **{key: {"mae": 1.0} for key in ["TR", "CC", "LR", "GRA"]}}
    baseline = {name: copy.deepcopy(metrics) for name in
                ["synthetic_diagnostic", "original_test", "original_validation"]}
    candidate = copy.deepcopy(baseline)
    candidate["synthetic_diagnostic"]["macro_mae"] = 0.7
    difference = {"synthetic_diagnostic": {"paired_prompt_group_95ci": [-0.7, 0.01]}}
    assert not public_gates(baseline, candidate, difference, config)["public_evaluation_passed"]
