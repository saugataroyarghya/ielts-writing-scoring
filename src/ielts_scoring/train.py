"""Single-GPU training with explicit regression loss and portable checkpoints."""

import argparse
import importlib.metadata
import json
import math
import os
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .baselines import run_baselines
from .data import assert_split_integrity, sha256
from .evaluate import error_slices, prediction_frame, regression_metrics
from .models import EssayScorer, masked_mse_sum


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_examples(tokenizer, prompts, essays, track, max_length):
    if track == "ielts":
        encoded = tokenizer(prompts, essays, truncation="only_second", max_length=max_length)
    else:
        encoded = tokenizer(essays, truncation=True, max_length=max_length)
    examples = []
    for i in range(len(essays)):
        examples.append({
            "input_ids": encoded["input_ids"][i],
            "attention_mask": encoded["attention_mask"][i],
            "essay_mask": [int(sequence == (1 if track == "ielts" else 0))
                           for sequence in encoded.sequence_ids(i)],
        })
    return examples


def tokenize_frame(frame, tokenizer, specification):
    examples = encode_examples(tokenizer, frame["prompt"].tolist(), frame["essay"].tolist(),
                               specification["track"], specification["max_length"])
    labels = frame[specification["targets"]].to_numpy(dtype=np.float32)
    for i, example in enumerate(examples):
        example["labels"] = labels[i].tolist()
    lengths = [len(tokenizer.encode(text, add_special_tokens=False)) for text in frame["essay"]]
    if specification["track"] == "ielts":
        question_lengths = [len(tokenizer.encode(text, add_special_tokens=False)) for text in frame["prompt"]]
        lengths = np.array(lengths) + np.array(question_lengths) + tokenizer.num_special_tokens_to_add(pair=True)
    else:
        lengths = np.array(lengths) + tokenizer.num_special_tokens_to_add(pair=False)
    truncated = lengths > specification["max_length"]
    report = {"n": len(frame), "truncated": int(truncated.sum()),
              "fraction_truncated": float(truncated.mean()),
              "token_length_percentiles": dict(zip(["p50", "p90", "p95", "p99", "max"],
                                                    np.percentile(lengths, [50, 90, 95, 99, 100]).tolist())),
              "truncated_by_mean_band": {str(band): int((truncated & (np.floor(np.nanmean(labels, axis=1)) == band)).sum())
                                         for band in np.unique(np.floor(np.nanmean(labels, axis=1)))}}
    return examples, report


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, items):
        batch = self.tokenizer.pad(
            [{"input_ids": x["input_ids"], "attention_mask": x["attention_mask"]} for x in items],
            padding=True, pad_to_multiple_of=8, return_tensors="pt",
        )
        size = batch["input_ids"].shape[1]
        batch["essay_mask"] = torch.tensor([x["essay_mask"] + [0] * (size - len(x["essay_mask"])) for x in items])
        if "labels" in items[0]:
            batch["labels"] = torch.tensor([x["labels"] for x in items], dtype=torch.float32)
        return batch


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def autocast_context(device):
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    predictions, labels, gate_values = [], [], []
    for batch in loader:
        batch = to_device(batch, device)
        actual = batch.pop("labels", None)
        with autocast_context(device):
            result = model(**batch)
        predictions.append(result["scores"].float().cpu().numpy())
        if actual is not None:
            labels.append(actual.cpu().numpy())
        if result["gates"] is not None:
            gate_values.append(result["gates"].float().cpu().numpy())
    gate_report = None
    if gate_values:
        gates = np.concatenate(gate_values)
        gate_report = {"mean_weights_by_criterion": gates.mean(axis=0).tolist(),
                       "mean_entropy_by_criterion": (-(gates * np.log(gates.clip(1e-8))).sum(-1)).mean(0).tolist()}
    return np.concatenate(predictions), np.concatenate(labels) if labels else None, gate_report


def find_data(root, track):
    for path in sorted(Path(root).rglob("metadata.json")):
        try:
            metadata = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        if metadata.get("track") == track and "prepared_sha256" in metadata:
            for name, expected in metadata["prepared_sha256"].items():
                if sha256((path.parent / name).read_bytes()) != expected:
                    raise ValueError(f"Prepared dataset checksum mismatch: {name}")
            return path.parent, metadata
    raise FileNotFoundError(f"Prepared {track} dataset was not attached under {root}.")


def environment():
    packages = ["torch", "transformers", "huggingface-hub", "safetensors", "pandas", "numpy", "scikit-learn"]
    return {"packages": {n: importlib.metadata.version(n) for n in packages},
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count(),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 2 ** 30
            if torch.cuda.is_available() else 0}


def make_optimizer(model, encoder_lr=2e-5, head_lr=1e-4):
    groups = []
    for module, rate in [(model.encoder, encoder_lr), (model.head, head_lr)]:
        decay, no_decay = [], []
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                (no_decay if parameter.ndim < 2 or name.endswith("bias") else decay).append(parameter)
        groups.extend([{"params": decay, "lr": rate, "weight_decay": 0.01},
                       {"params": no_decay, "lr": rate, "weight_decay": 0.0}])
    return torch.optim.AdamW(groups)


def train_epoch(model, loader, optimizer, scheduler, scaler, device, accumulation=8, max_batches=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_total = 0.0
    observed_total = 0
    group_observed = 0
    examples = 0
    count = min(len(loader), max_batches) if max_batches else len(loader)
    started = time.perf_counter()
    for step, batch in enumerate(loader):
        if step >= count:
            break
        batch = to_device(batch, device)
        labels = batch.pop("labels")
        with autocast_context(device):
            result = model(**batch)
            loss_sum, observed = masked_mse_sum(result["scores"], labels)
        if not torch.isfinite(loss_sum):
            raise FloatingPointError(f"Nonfinite loss at batch {step}.")
        # Backprop sums; normalize gradients once by ALL observed targets in this accumulation group.
        scaler.scale(loss_sum).backward()
        group_observed += int(observed)
        observed_total += int(observed)
        loss_total += float(loss_sum.detach())
        examples += labels.shape[0]
        if (step + 1) % accumulation == 0 or step + 1 == count:
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(max(group_observed, 1))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and scaler.get_scale() >= previous_scale:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            group_observed = 0
        if (step + 1) % 200 == 0:
            print(json.dumps({"batch": step + 1, "of": count, "mse": loss_total / max(observed_total, 1),
                              "examples_per_second": examples / (time.perf_counter() - started)}), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"mse": loss_total / max(observed_total, 1), "seconds": time.perf_counter() - started,
            "examples": examples, "batches": count}


def smoke(model, tokenizer, train, specification, device, output):
    """Check real full-length backward passes, tiny-set learning, and checkpoint parity."""
    examples, lengths = tokenize_frame(train.iloc[:64], tokenizer, specification)
    loader = DataLoader(examples, batch_size=2, collate_fn=Collator(tokenizer), shuffle=False)
    optimizer = make_optimizer(model)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not torch.cuda.is_bf16_supported())
    warmup = train_epoch(model, loader, optimizer, None, scaler, device, max_batches=4)
    timing = train_epoch(model, loader, optimizer, None, scaler, device, max_batches=32)
    # Tiny overfit uses short examples for speed; main benchmark above keeps the full configured length.
    tiny_spec = dict(specification, max_length=128)
    tiny, _ = tokenize_frame(train.iloc[:8], tokenizer, tiny_spec)
    tiny_loader = DataLoader(tiny, batch_size=2, collate_fn=Collator(tokenizer))
    before, actual, _ = predict_loader(model, tiny_loader, device)
    before_loss = float(np.nanmean((before - actual) ** 2))
    for _ in range(15):
        train_epoch(model, tiny_loader, optimizer, None, scaler, device, accumulation=4)
    after, _, gate_report = predict_loader(model, tiny_loader, device)
    after_loss = float(np.nanmean((after - actual) ** 2))
    if after_loss >= before_loss:
        raise AssertionError(f"Tiny-set loss did not decrease: {before_loss} -> {after_loss}")
    checkpoint = Path("/kaggle/temp") / f"parity-{specification['head']}"
    model.save(checkpoint, tokenizer)
    restored = EssayScorer.load(checkpoint, device)
    reloaded, _, _ = predict_loader(restored, tiny_loader, device)
    np.testing.assert_allclose(after, reloaded, atol=0.02, rtol=0.005)
    del restored
    shutil.rmtree(checkpoint)
    report = {"warmup": warmup, "benchmark": timing, "token_lengths": lengths,
              "tiny_mse_before": before_loss, "tiny_mse_after": after_loss,
              "save_load_max_difference": float(np.abs(after - reloaded).max()),
              "head_parameters": sum(p.numel() for p in model.head.parameters()),
              "total_parameters": sum(p.numel() for p in model.parameters()), "gates": gate_report,
              "estimated_training_hours_5_epochs": len(train) * 5 * timing["seconds"] / timing["examples"] / 3600,
              "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0}
    (output / "smoke_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def fit(model, tokenizer, train, validation, specification, device, output, epochs=5):
    train_data, train_lengths = tokenize_frame(train, tokenizer, specification)
    val_data, val_lengths = tokenize_frame(validation, tokenizer, specification)
    (output / "tokenization.json").write_text(json.dumps({"train": train_lengths, "validation": val_lengths}, indent=2))
    generator = torch.Generator().manual_seed(specification["seed"])
    collator = Collator(tokenizer)
    loader = DataLoader(train_data, batch_size=2, shuffle=True, generator=generator, collate_fn=collator)
    val_loader = DataLoader(val_data, batch_size=4, collate_fn=collator)
    optimizer = make_optimizer(model)
    steps = math.ceil(len(loader) / 8) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(steps * 0.1), num_training_steps=steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not torch.cuda.is_bf16_supported())
    best, stale = float("inf"), 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_stats = train_epoch(model, loader, optimizer, scheduler, scaler, device)
        predictions, labels, gates = predict_loader(model, val_loader, device)
        metrics = regression_metrics(labels, predictions, specification["targets"], specification["track"] == "ielts")
        row = {"epoch": epoch, "train": epoch_stats, "validation": metrics, "gates": gates}
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(row), flush=True)
        if metrics["macro_mae"] < best:
            best, stale = metrics["macro_mae"], 0
            model.save(output / "best", tokenizer)
            (output / "best_validation.json").write_text(json.dumps({"epoch": epoch, **metrics}, indent=2))
            prediction_frame(validation, labels, predictions, specification["targets"]).to_csv(
                output / "validation_predictions.csv", index=False,
            )
            (output / "validation_slices.json").write_text(json.dumps(
                error_slices(validation, labels, predictions, specification["targets"]), indent=2,
            ))
        else:
            stale += 1
        if stale >= 2:
            print("Early stopping after two epochs without validation improvement.", flush=True)
            break
    (output / "runtime.json").write_text(json.dumps({
        "seconds": time.perf_counter() - started, "epochs": len(history),
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0,
    }, indent=2))
    # The test set stays unopened until architecture selection is finished.
    print("Training complete. Test evaluation is a separate, explicit step.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["ellipse", "ielts"], required=True)
    parser.add_argument("--head", choices=["mlp", "moe"], required=True)
    parser.add_argument("--mode", choices=["smoke", "train"], default="train")
    parser.add_argument("--encoder", choices=["modernbert", "ettin"], default="modernbert")
    parser.add_argument("--data-root", default="/kaggle/input")
    parser.add_argument("--output", default="artifacts/runs/latest")
    parser.add_argument("--source-hash", default="local-development")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    runtime = environment()
    (output / "environment.json").write_text(json.dumps(runtime, indent=2))
    print(json.dumps(runtime, indent=2), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("This training job requires a GPU. Preparation and unit tests can run on CPU.")
    device = torch.device("cuda:0")
    data_dir, metadata = find_data(args.data_root, args.track)
    pinned_dataset = json.loads(Path("configs/prepared_datasets.json").read_text())[args.track]
    if metadata["prepared_sha256"] != pinned_dataset["prepared_sha256"]:
        raise ValueError("Attached dataset differs from the frozen experiment version.")
    train = pd.read_parquet(data_dir / "train.parquet")
    validation = pd.read_parquet(data_dir / "validation.parquet")
    manifest = pd.read_csv(data_dir / "split_manifest.csv")
    assert_split_integrity(manifest)
    sources = json.loads(Path("configs/sources.json").read_text())
    specification = {**sources[args.encoder], "head": args.head, "track": args.track,
                     "targets": metadata["targets"], "label_range": metadata["label_range"],
                     "max_length": 1024, "seed": args.seed, "source_hash": args.source_hash,
                     "dataset_revision": metadata["source"]["revision"],
                     "kaggle_dataset": pinned_dataset["kaggle_dataset"],
                     "kaggle_dataset_version": pinned_dataset["version"],
                     "dataset_checksums": metadata["prepared_sha256"], "pooling": "mean_essay_tokens",
                     "encoder_lr": 2e-5, "head_lr": 1e-4, "batch_size": 2, "accumulation": 8}
    (output / "run_config.json").write_text(json.dumps(specification, indent=2))
    if args.mode == "train":
        run_baselines(train, validation, metadata["targets"], args.track, output / "baselines_validation.json")
    tokenizer = AutoTokenizer.from_pretrained(specification["model_id"], revision=specification["revision"])
    model = EssayScorer.from_pretrained_encoder(specification).to(device)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    torch.cuda.reset_peak_memory_stats()
    if args.mode == "smoke":
        smoke(model, tokenizer, train, specification, device, output)
    else:
        fit(model, tokenizer, train, validation, specification, device, output, epochs=args.epochs)


if __name__ == "__main__":
    main()
