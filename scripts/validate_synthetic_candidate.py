"""Recompute public metrics and run local-only sanity and portable-loader checks."""

import argparse
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from ielts_scoring.evaluate import regression_metrics
from ielts_scoring.predict import Scorer
from ielts_scoring.synthetic_finetune import TARGETS, load_hub_model, public_gates, read, sha, write
from ielts_scoring.train import Collator, encode_examples
from ielts_scoring.transfer import paired_group_difference

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "artifacts/staging/huggingface-transfer/ielts-task2-modernbert-base-moe"


def predict(model, tokenizer, question, essay):
    batch = Collator(tokenizer)(encode_examples(tokenizer, [question], [essay], "ielts", 1024))
    with torch.inference_mode():
        return model(**batch)["scores"][0].float().tolist()


def main(run, package, private):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    config, result = read(run / "experiment_config.json"), read(run / "result.json")
    if result["status"] != "evaluated":
        write(run / "release_gate.json", {"passed": False, "reason": "No selected candidate"})
        return
    assert sha(PARENT / "model.safetensors") == config["parent_weights_sha256"]
    assert sha(run / "best/model.safetensors") == result["weights_sha256"]
    metrics, differences = {}, {}
    for label in ["baseline", "candidate"]:
        metrics[label] = {}
        for split in result[label]:
            frame = pd.read_csv(run / f"{label}_{split}_predictions.csv")
            y = frame[[f"actual_{t}" for t in TARGETS]].to_numpy()
            p = frame[[f"predicted_{t}" for t in TARGETS]].to_numpy()
            computed = regression_metrics(y, p, TARGETS, ielts=True)
            assert abs(computed["macro_mae"] - result[label][split]["macro_mae"]) < 1e-6
            for target in TARGETS:
                for key in ["mae", "rmse", "bias"]:
                    assert abs(computed[target][key] - result[label][split][target][key]) < 1e-6
            metrics[label][split] = computed
    for split in result["candidate"]:
        differences[split] = paired_group_difference(
            pd.read_csv(run / f"baseline_{split}_predictions.csv"),
            pd.read_csv(run / f"candidate_{split}_predictions.csv"), TARGETS)
    public = public_gates(metrics["baseline"], metrics["candidate"], differences, config)
    private.mkdir(parents=True, exist_ok=True)
    cases = read(ROOT / "artifacts/private-sanity-20261010/cases.json")
    private_predictions, private_summary = [], {}
    tokenizer = AutoTokenizer.from_pretrained(PARENT, local_files_only=True)
    for label in ["baseline", "candidate"]:
        model = (load_hub_model(PARENT, torch.device("cpu")) if label == "baseline" else
                 Scorer(run / "best").model).eval()
        selected = []
        for case in cases:
            if not case["essay"].strip():
                continue
            values = predict(model, tokenizer, case["question"], case["essay"])
            assert np.isfinite(values).all()
            row = {"model": label, "case_id": case["case_id"], "kind": case["kind"],
                   "parent": case["parent"], "scores": dict(zip(TARGETS, values)),
                   "mean": sum(values) / 4, "reference_task_band": case["official_task_band"]}
            private_predictions.append(row)
            selected.append(row)
        originals = [r for r in selected if r["kind"] == "original"]
        assert len(originals) == 7
        by_id = {r["case_id"]: r for r in originals}
        wrong = [r for r in selected if r["kind"] == "wrong_question"]
        shifts = [r["scores"]["TR"] - by_id[r["parent"]]["scores"]["TR"] for r in wrong]
        errors = [r["mean"] - r["reference_task_band"] for r in originals]
        private_summary[label] = {
            "n": 7, "task_mean_mae": float(np.abs(errors).mean()),
            "task_mean_bias": float(np.mean(errors)), "within_half_count": sum(abs(e) <= 0.5 for e in errors),
            "wrong_question_mean_TR_change": float(np.mean(shifts)),
            "wrong_question_TR_increased_count": sum(e > 0 for e in shifts)}
        del model
        gc.collect()
    private_pass = private_summary["candidate"]["task_mean_mae"] <= (
        private_summary["baseline"]["task_mean_mae"] + config["promotion"]["maximum_private_sanity_mae_regression"])
    write(private / "predictions.json", private_predictions)
    write(private / "summary.json", {"heads": private_summary, "passed": private_pass,
                                    "scope": "Private seven-sample sanity check; not a representative benchmark.",
                                    "samples_sha256": sha(ROOT / "artifacts/private-sanity-20261010/samples.json")})
    # Build an isolated package with unchanged, reviewed public inference code.
    package.mkdir(parents=True, exist_ok=True)
    for name in ["modeling_ielts.py", "config.json", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "LICENSE.md", "BASE_MODEL_LICENSE"]:
        shutil.copy2(PARENT / name, package / name)
    for name in ["model.safetensors", "scorer_config.json"]:
        shutil.copy2(run / "best" / name, package / name)
    hub_model, loading = AutoModel.from_pretrained(package, local_files_only=True, trust_remote_code=True,
                                                  output_loading_info=True)
    hub_model.eval()
    assert all(not loading[k] for k in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"])
    original = Scorer(run / "best")
    assert all(torch.equal(value, original.model.state_dict()[key]) for key, value in hub_model.state_dict().items())
    frame = pd.read_parquet(ROOT / "artifacts/kaggle-output/preparation/ielts/validation.parquet")
    rows = frame.iloc[[0, 50, 150]].to_dict("records")
    rows.append({"prompt": rows[0]["prompt"], "essay": rows[0]["essay"] * 10})
    expected = [original.predict(r["essay"], r["prompt"]) for r in rows]
    actual = hub_model.score(tokenizer, [r["prompt"] for r in rows], [r["essay"] for r in rows])
    delta = max(abs(a["scores"][t] - b["scores"][t]) for a, b in zip(actual, expected) for t in TARGETS)
    assert delta < 1e-4
    assert [r["truncated"] for r in actual] == [r["truncated"] for r in expected]
    assert actual[-1]["truncated"]
    for question, essay in [("", "An essay."), ("A question?", " ")]:
        try:
            hub_model.score(tokenizer, [question], [essay])
        except ValueError:
            pass
        else:
            raise AssertionError("Blank input accepted.")
    verification = {"passed": True, "all_tensors_identical": True, "strict_loading": loading,
                    "maximum_prediction_difference": delta, "cases": 4,
                    "truncation_verified": True, "blank_input_rejection_verified": True}
    write(package / "inference_verification.json", verification)
    gate = {"passed": bool(public["public_evaluation_passed"] and private_pass),
            "public": public, "private_sanity_passed": bool(private_pass), "portable_loader_passed": True,
            "weights_sha256": result["weights_sha256"], "parent_revision": config["parent_revision"],
            "dataset_revision": config["dataset_revision"]}
    write(run / "release_gate.json", gate)
    # This gate receipt stays local; no private samples, scores, or aggregate results enter the Hub package.
    print(json.dumps({"public_evaluation_passed": public["public_evaluation_passed"],
                      "private_sanity_passed": bool(private_pass), "loader_passed": True,
                      "promotion_passed": gate["passed"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--package", type=Path, default=ROOT / "artifacts/staging/huggingface-synthetic-moe")
    parser.add_argument("--private", type=Path, default=ROOT / "artifacts/private-sanity-synthetic-v1")
    args = parser.parse_args()
    main(args.run, args.package, args.private)
