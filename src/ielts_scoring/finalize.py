"""Select the head on validation, evaluate its held-out test set, and stage a release."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .evaluate import error_slices, prediction_frame, regression_metrics
from .models import EssayScorer
from .predict import Scorer
from .train import Collator, environment, find_data, predict_loader, tokenize_frame


def group_bootstrap_mae(labels, predictions, groups, seed=42, draws=1000):
    errors = np.abs(predictions - labels)
    unique, indices = np.unique(groups, return_inverse=True)
    sums = np.zeros((len(unique), labels.shape[1]))
    counts = np.zeros_like(sums)
    np.add.at(sums, indices, np.nan_to_num(errors))
    np.add.at(counts, indices, np.isfinite(labels))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        selected = rng.integers(len(unique), size=len(unique))
        totals, observed = sums[selected].sum(0), counts[selected].sum(0)
        averages = np.divide(totals, observed, out=np.full_like(totals, np.nan), where=observed > 0)
        samples.append(float(np.nanmean(averages)))
    return {"lower": float(np.quantile(samples, 0.025)), "upper": float(np.quantile(samples, 0.975)),
            "groups": len(unique), "draws": draws,
            "interpretation": "Test-group sampling uncertainty for this checkpoint; not training-seed uncertainty."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["ellipse", "ielts"], required=True)
    parser.add_argument("--input", default="/kaggle/input")
    parser.add_argument("--runs", help="Completed run directories; defaults to --input.")
    parser.add_argument("--output", default="/kaggle/working/evaluation")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    data_dir, metadata = find_data(args.input, args.track)
    expected = json.loads(Path("configs/prepared_datasets.json").read_text())[args.track]
    if expected["prepared_sha256"] != metadata["prepared_sha256"]:
        raise ValueError("Dataset differs from the frozen comparison.")
    candidates = {}
    for metric_path in Path(args.runs or args.input).rglob("best_validation.json"):
        run_dir = metric_path.parent
        specification = json.loads((run_dir / "run_config.json").read_text())
        if specification["track"] == args.track:
            if specification["head"] in candidates:
                raise ValueError("Multiple runs for one head; attach only the intended comparison versions.")
            candidates[specification["head"]] = {
                "path": run_dir, "config": specification, "metrics": json.loads(metric_path.read_text()),
            }
    if set(candidates) != {"mlp", "moe"}:
        raise ValueError("Attach completed plain and MoE notebook outputs before final evaluation.")
    for key in ["model_id", "revision", "dataset_checksums", "max_length", "seed", "encoder_lr", "head_lr"]:
        if candidates["mlp"]["config"][key] != candidates["moe"]["config"][key]:
            raise ValueError(f"Comparison differs in {key}; review before test evaluation.")
    if candidates["mlp"]["config"].get("initialization") != candidates["moe"]["config"].get("initialization"):
        raise ValueError("Candidates must use the same encoder initialization.")
    chosen = min(candidates, key=lambda name: candidates[name]["metrics"]["macro_mae"])
    candidate = candidates[chosen]
    specification = candidate["config"]
    selection = {"selected_head": chosen, "criterion": "lowest macro validation MAE",
                 "validation": {name: run["metrics"] for name, run in candidates.items()},
                 "seed": specification["seed"]}
    if specification.get("initialization"):
        selection["initialization"] = specification["initialization"]
        selection["test_note"] = (
            "The same frozen test split was reported for a prior IELTS-only baseline. "
            "This is a comparative held-out evaluation, not a newly unseen test set. "
            "The transfer experiment and checkpoint selection use validation data only."
        )
    (output / "selection.json").write_text(json.dumps(selection, indent=2))
    print("Selected using validation only:", chosen, flush=True)
    test = pd.read_parquet(data_dir / "test.parquet")
    tokenizer = AutoTokenizer.from_pretrained(candidate["path"] / "best", local_files_only=True)
    examples, lengths = tokenize_frame(test, tokenizer, specification)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EssayScorer.load(candidate["path"] / "best", device)
    loader = DataLoader(examples, batch_size=4 if device.type == "cuda" else 2, collate_fn=Collator(tokenizer))
    predictions, labels, gates = predict_loader(model, loader, device)
    metrics = regression_metrics(labels, predictions, specification["targets"], args.track == "ielts")
    metrics["macro_mae_95ci"] = group_bootstrap_mae(labels, predictions, test["group_id"].to_numpy())
    low, high = specification["label_range"]
    metrics["out_of_range_predictions"] = int(((predictions < low) | (predictions > high)).sum())
    metrics["qwk"] = None
    metrics["qwk_note"] = "Not reported until a source-verified discrete rating grid is recorded; no arbitrary rounding."
    (output / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    (output / "test_slices.json").write_text(json.dumps(error_slices(test, labels, predictions, specification["targets"]), indent=2))
    (output / "tokenization.json").write_text(json.dumps(lengths, indent=2))
    (output / "gates.json").write_text(json.dumps(gates, indent=2))
    (output / "environment.json").write_text(json.dumps(environment(), indent=2))
    # Controlled perturbations have no invented gold scores and do not affect selection.
    validation = pd.read_parquet(data_dir / "validation.parquet")
    probe = validation.iloc[0]
    paragraphs = [p for p in probe["essay"].split("\n") if p.strip()]
    cases = [
        ("original", probe["prompt"], probe["essay"]),
        ("reversed_paragraphs", probe["prompt"], "\n\n".join(reversed(paragraphs))),
        ("repeated_first_paragraph", probe["prompt"], probe["essay"] + "\n\n" + paragraphs[0] * 2),
        ("short_answer", probe["prompt"], " ".join(probe["essay"].split()[:40])),
    ]
    if args.track == "ielts":
        alternatives = validation.loc[validation["prompt_group"].ne(probe["prompt_group"]), "prompt"]
        if len(alternatives):
            cases.append(("different_question", alternatives.iloc[0], probe["essay"]))
    from .train import encode_examples

    encoded = encode_examples(tokenizer, [p for _, p, _ in cases], [e for _, _, e in cases],
                               args.track, specification["max_length"])
    diagnostic_scores, _, _ = predict_loader(model, DataLoader(encoded, batch_size=2,
                                                              collate_fn=Collator(tokenizer)), device)
    diagnostics = {name: dict(zip(specification["targets"], scores.tolist()))
                   for (name, _, _), scores in zip(cases, diagnostic_scores)}
    diagnostics["note"] = "Controlled cases only; these are not human-scored test examples."
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    table = prediction_frame(test, labels, predictions, specification["targets"])
    table.to_csv(output / "test_predictions.csv", index=False)
    table["mean_absolute_error"] = np.nanmean(np.abs(predictions - labels), axis=1)
    table.nlargest(30, "mean_absolute_error").to_csv(output / "largest_errors.csv", index=False)
    release = output / "release"
    shutil.copytree(candidate["path"] / "best", release, dirs_exist_ok=True)
    shutil.copytree(Path("src/ielts_scoring"), release / "src/ielts_scoring", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy("pyproject.toml", release / "pyproject.toml")
    shutil.copy(candidate["path"] / "baselines_validation.json", release / "baselines_validation.json")
    for name in ["selection.json", "test_metrics.json", "tokenization.json"]:
        shutil.copy(output / name, release / name)
    cpu_scorer = Scorer(release, device="cpu")
    cpu_result = cpu_scorer.predict(probe["essay"], probe["prompt"] if args.track == "ielts" else None)
    cpu_scores = np.array([cpu_result["scores"][name] for name in specification["targets"]])
    np.testing.assert_allclose(cpu_scores, diagnostic_scores[0], atol=0.03, rtol=0.005)
    (output / "cpu_inference_check.json").write_text(json.dumps({
        "passed": True, "max_difference_from_gpu": float(np.max(np.abs(cpu_scores - diagnostic_scores[0]))),
        "source": "fresh staged checkpoint loaded locally without network",
    }, indent=2))
    lines = ["---", "library_name: pytorch", f"base_model: {specification['model_id']}",
             "language: en", "tags:", "- essay-scoring", "- regression", "- research", "---", "",
             f"# {args.track.upper()} ModernBERT {chosen} writing scorer", "",
             "Private research checkpoint; public release and weight licensing remain pending review.", "",
             f"Targets (in output order): {', '.join(specification['targets'])}.", "",
             f"Selected by validation MAE. Test macro MAE: {metrics['macro_mae']:.4f}.", "",
             "| Target | MAE | RMSE | Bias |", "| --- | ---: | ---: | ---: |"]
    for name in specification["targets"]:
        row = metrics[name]
        lines.append(f"| {name} | {row['mae']:.4f} | {row['rmse']:.4f} | {row['bias']:.4f} |")
    lines += ["", "## Usage", "", "Install this directory with `pip install .`, then:", "", "```python",
              "from ielts_scoring.predict import Scorer", 'model = Scorer(".")',
              'model.predict(essay="Your essay", question="Original Task 2 question")' if args.track == "ielts"
              else 'model.predict(essay="Your essay")', "```", "",
              "## Training and limitations", "",
              ("The encoder and head were jointly fine-tuned. See scorer_config.json for model revision, exact data checksums, "
              "length policy, target scales, and hyperparameters. Original source attribution and license: "
              f"{metadata['source']['source']} — {metadata['source']['license']}."), "",
              ("Only one training seed has been evaluated. Confidence intervals resample test groups, not training seeds. "
              "The 1,024-token limit can truncate essays. Predictions are raw continuous estimates. "
              "A default classification pipeline is not supported."), "",
              "IELTS scores are Writing9 weak-label estimates, not validated examiner scores or a complete IELTS Writing result."
              if args.track == "ielts" else "ELLIPSE proficiency scores are not IELTS bands and must not be rescaled into them.", ""]
    if specification.get("initialization"):
        lines += ["", "## Encoder transfer", "",
                  ("Both candidate heads started from the same ELLIPSE-trained encoder. "
                  "The six-score ELLIPSE head was discarded; this four-score IELTS head was initialized afresh. "
                  "See scorer_config.json initialization for the parent checkpoint, hash, and data provenance."),
                  "", selection["test_note"], ""]
    (release / "README.md").write_text("\n".join(lines))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
