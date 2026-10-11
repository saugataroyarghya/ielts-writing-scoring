"""Continue the published MoE checkpoint with synthetic augmentation and frozen gates."""

import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from .evaluate import error_slices, prediction_frame, regression_metrics
from .models import EssayScorer
from .train import (
    Collator,
    environment,
    find_data,
    make_optimizer,
    predict_loader,
    seed_everything,
    tokenize_frame,
    train_epoch,
)
from .transfer import paired_group_difference

TARGETS = ["TR", "CC", "LR", "GRA"]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def synthetic_frame(path):
    frame = pd.read_parquet(path).copy()
    if not frame.synthetic.all() or frame.human_verified.any() or frame[TARGETS].isna().any().any():
        raise ValueError("Expected accepted synthetic LLM labels only.")
    if not np.isfinite(frame[TARGETS].to_numpy()).all() or not frame[TARGETS].isin(range(1, 10)).all().all():
        raise ValueError("Invalid synthetic score.")
    if frame.id.duplicated().any():
        raise ValueError("Duplicate synthetic IDs.")
    frame["prompt"] = frame.question
    frame["source"] = "synthetic-llm-v1"
    frame["group_id"] = "synthetic:" + frame.prompt_group_id
    frame["prompt_group"] = frame.group_id
    # No evidence, scores, intended ability, or metadata are concatenated into input text.
    return frame


def make_mixture(real_train, synthetic_train, count, seed):
    if count > len(real_train):
        raise ValueError("Replay count exceeds original training set.")
    replay = real_train.sample(n=count, random_state=seed).copy()
    columns = ["id", "source", "group_id", "prompt_group", "prompt", "essay", *TARGETS]
    replay["group_id"] = "original:" + replay.group_id.astype(str)
    replay["prompt_group"] = replay.group_id
    mixture = pd.concat([replay[columns], synthetic_train[columns]], ignore_index=True)
    if mixture.id.duplicated().any():
        raise ValueError("Training IDs overlap.")
    return mixture, replay.id.tolist()


def no_regression(baseline, candidate, macro_tolerance, criterion_tolerance):
    return bool(candidate["macro_mae"] <= baseline["macro_mae"] + macro_tolerance and
                all(candidate[t]["mae"] <= baseline[t]["mae"] + criterion_tolerance for t in TARGETS))


def public_gates(baseline, candidate, differences, config):
    limits = config["promotion"]
    syn_delta = candidate["synthetic_diagnostic"]["macro_mae"] - baseline["synthetic_diagnostic"]["macro_mae"]
    checks = {
        "synthetic_diagnostic_gain": syn_delta <= -limits["minimum_synthetic_macro_gain"],
        "synthetic_group_interval_below_zero":
            differences["synthetic_diagnostic"]["paired_prompt_group_95ci"][1] < 0,
        "original_test_no_material_regression": no_regression(
            baseline["original_test"], candidate["original_test"],
            limits["maximum_original_macro_regression"], limits["maximum_original_criterion_regression"]),
        "original_validation_no_material_regression": no_regression(
            baseline["original_validation"], candidate["original_validation"],
            limits["maximum_original_macro_regression"], limits["maximum_original_criterion_regression"]),
    }
    return {"checks": checks, "public_evaluation_passed": all(checks.values()),
            "synthetic_macro_mae_change": syn_delta,
            "publication_ready": False,
            "remaining_gates": ["local private sanity comparison", "portable checkpoint loading parity"],
            "interpretation": "Synthetic agreement and website-label comparison do not establish examiner accuracy."}


def load_hub_model(folder, device):
    specification = read(folder / "scorer_config.json")
    hub_config = read(folder / "config.json")
    settings = dict(hub_config["encoder_config"])
    model_type = settings.pop("model_type")
    settings["reference_compile"] = False
    encoder = AutoModel.from_config(AutoConfig.for_model(model_type, **settings), attn_implementation="sdpa")
    model = EssayScorer(encoder, specification)
    model.load_state_dict(load_file(str(folder / "model.safetensors")), strict=True)
    return model.to(device)


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = read(args.config)
    write(output / "experiment_config.json", config)
    if config["dataset_revision"] in [None, "main", "PENDING"]:
        raise ValueError("Pin the public dataset commit before running.")
    if not torch.cuda.is_available() and not args.smoke_cpu:
        raise RuntimeError("A CUDA GPU is required for this training run.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(config["seed"])
    write(output / "environment.json", environment())
    hf_root = Path(snapshot_download(config["parent_repo"], revision=config["parent_revision"],
                                    allow_patterns=["*.json", "*.safetensors", "*.py"]))
    if sha(hf_root / "model.safetensors") != config["parent_weights_sha256"]:
        raise ValueError("Parent checkpoint hash mismatch.")
    data_root = Path(snapshot_download(config["dataset_repo"], repo_type="dataset",
                                      revision=config["dataset_revision"],
                                      allow_patterns=["data/*.parquet", "checksums.json", "manifest.json"]))
    checksums = read(data_root / "checksums.json")
    for name in ["train", "synthetic_validation", "synthetic_diagnostic"]:
        key = f"data/{name}.parquet"
        if sha(data_root / key) != checksums[key]:
            raise ValueError("Synthetic dataset hash mismatch.")
    real_folder, metadata = find_data(args.data_root, "ielts")
    if metadata["prepared_sha256"] != config["original_dataset_checksums"]:
        raise ValueError("Original split versions changed.")
    real_train = pd.read_parquet(real_folder / "train.parquet")
    syn_train = synthetic_frame(data_root / "data/train.parquet")
    mixture, replay_ids = make_mixture(real_train, syn_train, config["replay_rows"], config["seed"])
    write(output / "training_manifest.json", {
        "synthetic_train_rows": len(syn_train), "original_replay_rows": len(replay_ids),
        "total_rows_per_epoch": len(mixture), "replay_ids": replay_ids,
        "synthetic_ids": syn_train.id.tolist(), "dataset_revision": config["dataset_revision"],
        "input_columns": ["prompt", "essay"], "target_columns": TARGETS,
        "source_note": "Original labels originate from Writing9, not qualified examiners."})
    frames = {"original_validation": pd.read_parquet(real_folder / "validation.parquet"),
              "synthetic_validation": synthetic_frame(data_root / "data/synthetic_validation.parquet")}
    for name, frame in frames.items():
        if set(frame.id) & set(mixture.id):
            raise ValueError(f"Training overlap with {name}.")
    if set(syn_train.group_id) & set(frames["synthetic_validation"].group_id):
        raise ValueError("Synthetic training/validation group overlap.")
    tokenizer = AutoTokenizer.from_pretrained(hf_root, local_files_only=True)
    model = load_hub_model(hf_root, device)
    model.specification = {**model.specification, "continued_finetuning": config}
    if device.type == "cuda":
        model.encoder.gradient_checkpointing_enable()
    loaders, lengths = {}, {}

    def loader_for(name, frame, shuffle=False):
        examples, report = tokenize_frame(frame, tokenizer, model.specification)
        lengths[name] = report
        return DataLoader(examples, batch_size=config["batch_size"], shuffle=shuffle,
                          collate_fn=Collator(tokenizer), num_workers=0)

    train_loader = loader_for("train", mixture, True)
    for name, frame in frames.items():
        loaders[name] = loader_for(name, frame)

    def evaluate(name, label):
        predictions, actual, gates = predict_loader(model, loaders[name], device)
        metrics = regression_metrics(actual, predictions, TARGETS, ielts=True)
        report = {"metrics": metrics, "gates": gates,
                  "out_of_range": int(((predictions < 0) | (predictions > 9)).sum()),
                  "slices": error_slices(frames[name], actual, predictions, TARGETS)}
        write(output / f"{label}_{name}.json", report)
        prediction_frame(frames[name], actual, predictions, TARGETS).to_csv(
            output / f"{label}_{name}_predictions.csv", index=False)
        return metrics

    baseline = {name: evaluate(name, "baseline") for name in frames}
    write(output / "baseline_validation.json", baseline)
    print(json.dumps({"phase": "baseline", "metrics": baseline}), flush=True)
    if args.smoke_cpu:
        return
    optimizer = make_optimizer(model, config["encoder_lr"], config["head_lr"])
    steps = math.ceil(len(train_loader) / config["accumulation"]) * config["epoch_cap"]
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(steps * 0.1)), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=not torch.cuda.is_bf16_supported())
    best_objective = sum(x["macro_mae"] for x in baseline.values()) / 2
    selected_epoch, patience, history = 0, 0, []
    started = time.perf_counter()
    for epoch in range(1, config["epoch_cap"] + 1):
        trained = train_epoch(model, train_loader, optimizer, scheduler, scaler, device,
                              accumulation=config["accumulation"])
        validation = {name: evaluate(name, f"epoch_{epoch}") for name in frames}
        objective = sum(x["macro_mae"] for x in validation.values()) / 2
        allowed = no_regression(baseline["original_validation"], validation["original_validation"],
                                config["promotion"]["maximum_original_macro_regression"],
                                config["promotion"]["maximum_original_criterion_regression"])
        improved = allowed and objective < best_objective - config["minimum_selection_gain"]
        if improved:
            best_objective, selected_epoch, patience = objective, epoch, 0
            model.save(output / "best", tokenizer)
        else:
            patience += 1
        row = {"epoch": epoch, "training": trained, "validation": validation,
               "selection_objective": objective, "original_guard_passed": allowed,
               "selected": improved, "elapsed_seconds": time.perf_counter() - started}
        history.append(row)
        write(output / "history.json", history)
        print(json.dumps(row), flush=True)
        if patience >= config["patience"]:
            break
    write(output / "selection.json", {"epoch": selected_epoch, "objective": best_objective,
                                     "rule": config["selection"], "test_labels_used": False})
    # Freeze selection before opening either final evaluation split.
    frames["original_test"] = pd.read_parquet(real_folder / "test.parquet")
    frames["synthetic_diagnostic"] = synthetic_frame(data_root / "data/synthetic_diagnostic.parquet")
    for name in ["original_test", "synthetic_diagnostic"]:
        if set(frames[name].id) & set(mixture.id):
            raise ValueError("Training/final-evaluation row overlap.")
        loaders[name] = loader_for(name, frames[name])
    groups = [set(frame.group_id) for frame in [syn_train, frames["synthetic_validation"],
                                               frames["synthetic_diagnostic"]]]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("Synthetic group split leakage.")
    del optimizer, scheduler, model
    gc.collect()
    torch.cuda.empty_cache()
    model = load_hub_model(hf_root, device)
    for name in ["original_test", "synthetic_diagnostic"]:
        baseline[name] = evaluate(name, "baseline")
    if selected_epoch == 0:
        write(output / "result.json", {"status": "no_validation_improvement", "baseline": baseline,
                                      "promotion": {"publication_ready": False}, "history": history})
        print("No candidate passed validation selection; parent remains unchanged.", flush=True)
        return
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = EssayScorer.load(output / "best", device)
    candidate = {name: evaluate(name, "candidate") for name in frames}
    differences = {}
    for name in frames:
        differences[name] = paired_group_difference(
            pd.read_csv(output / f"baseline_{name}_predictions.csv"),
            pd.read_csv(output / f"candidate_{name}_predictions.csv"), TARGETS)
        differences[name]["interpretation"] = (
            "Negative favors fine-tuning. Paired prompt-group bootstrap, not training-seed uncertainty. "
            + ("Development set used for selection." if "validation" in name else
               "Frozen before this comparison; original test was previously reported for the parent model."))
    result = {"status": "evaluated", "selected_epoch": selected_epoch, "baseline": baseline,
              "candidate": candidate, "paired_differences": differences,
              "promotion": public_gates(baseline, candidate, differences, config),
              "weights_sha256": sha(output / "best/model.safetensors"),
              "elapsed_seconds": time.perf_counter() - started}
    write(output / "tokenization.json", lengths)
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synthetic_moe.json")
    parser.add_argument("--data-root", default="/kaggle/input")
    parser.add_argument("--output", default="/kaggle/working")
    parser.add_argument("--smoke-cpu", action="store_true")
    main(parser.parse_args())
