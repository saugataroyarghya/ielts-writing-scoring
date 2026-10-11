"""Frozen-parent discovery, cross-corpus leakage checks, and validation-only comparisons."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .train import find_data


def find_checkpoint(root, track, head):
    matches = []
    for path in Path(root).rglob("best/scorer_config.json"):
        config = json.loads(path.read_text())
        if config.get("track") == track and config.get("head") == head and not config.get("initialization"):
            matches.append(path.parent)
    if len(matches) != 1:
        raise ValueError(f"Expected one frozen {track}/{head} parent, found {len(matches)}.")
    return matches[0]


def audit_overlap(root):
    pinned = json.loads(Path("configs/prepared_datasets.json").read_text())
    manifests = {}
    for track in ("ellipse", "ielts"):
        folder, metadata = find_data(root, track)
        if metadata["prepared_sha256"] != pinned[track]["prepared_sha256"]:
            raise ValueError(f"Unexpected {track} dataset version.")
        # Read the frozen parquet hashes, not an unchecked manifest; no test labels are used.
        manifests[track] = {split: set(pd.read_parquet(folder / f"{split}.parquet",
                                                     columns=["essay_hash"])["essay_hash"])
                            for split in ("train", "validation", "test")}
    used = manifests["ellipse"]["train"] | manifests["ellipse"]["validation"]
    overlap = {split: len(used & hashes) for split, hashes in manifests["ielts"].items()}
    if overlap["validation"] or overlap["test"]:
        raise ValueError(f"ELLIPSE development data overlaps IELTS holdouts: {overlap}")
    return {"normalized_exact_essay_overlap": overlap, "passed": True,
            "note": "ELLIPSE train and validation checked against all IELTS splits; not a semantic near-duplicate audit."}


def paired_group_difference(first, second, targets, draws=5000):
    if first["id"].duplicated().any() or second["id"].duplicated().any():
        raise ValueError("Duplicate validation IDs.")
    first, second = first.set_index("id").sort_index(), second.set_index("id").sort_index()
    if not first.index.equals(second.index) or not first["group_id"].equals(second["group_id"]):
        raise ValueError("Validation rows or groups differ.")
    actual = [f"actual_{target}" for target in targets]
    predicted = [f"predicted_{target}" for target in targets]
    labels = first[actual].to_numpy()
    np.testing.assert_allclose(labels, second[actual].to_numpy(), equal_nan=True)
    difference = np.abs(second[predicted].to_numpy() - labels) - np.abs(first[predicted].to_numpy() - labels)
    groups, indices = np.unique(first["group_id"], return_inverse=True)
    sums = np.zeros((len(groups), len(targets)))
    counts = np.zeros_like(sums)
    np.add.at(sums, indices, np.nan_to_num(difference))
    np.add.at(counts, indices, np.isfinite(labels))
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(draws):
        selected = rng.integers(0, len(groups), len(groups))
        total, observed = sums[selected].sum(0), counts[selected].sum(0)
        samples.append(float(np.nanmean(np.divide(total, observed, out=np.full_like(total, np.nan),
                                                  where=observed > 0))))
    return {"macro_mae_difference_second_minus_first": float(np.nanmean(np.nanmean(difference, axis=0))),
            "paired_prompt_group_95ci": np.quantile(samples, [0.025, 0.975]).tolist(),
            "groups": len(groups), "rows": len(first), "draws": draws,
            "interpretation": "Negative favors the second model; resamples validation prompt groups, not training seeds."
                              " Checkpoints were selected on this validation set; this is exploratory uncertainty."}


def compare(baseline_root, runs):
    runs = Path(runs)
    folders = {f"direct_{head}": find_checkpoint(baseline_root, "ielts", head).parent for head in ("mlp", "moe")}
    folders.update({f"transfer_{head}": runs / head for head in ("mlp", "moe")})
    metrics = {name: json.loads((folder / "best_validation.json").read_text()) for name, folder in folders.items()}
    frames = {name: pd.read_csv(folder / "validation_predictions.csv") for name, folder in folders.items()}
    targets = json.loads((runs / "mlp/run_config.json").read_text())["targets"]
    pairs = [("transfer_mlp", "transfer_moe"), ("direct_mlp", "transfer_mlp"),
             ("direct_moe", "transfer_moe"), ("direct_mlp", "direct_moe")]
    return {"validation": metrics,
            "comparisons": {f"{first}_vs_{second}": paired_group_difference(frames[first], frames[second], targets)
                            for first, second in pairs},
            "test_policy": "Test results excluded from this comparison and all model selection."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/kaggle/input")
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = compare(args.input, args.runs)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
