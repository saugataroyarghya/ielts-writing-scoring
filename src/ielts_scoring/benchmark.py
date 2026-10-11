"""Paired, warmed batch-one latency measurements with explicit CUDA synchronization."""

import argparse
import gc
import hashlib
import json
import platform
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .predict import Scorer
from .train import Collator, encode_examples, environment


def summarize(milliseconds):
    values = np.asarray(milliseconds)
    return {"n": len(values), "p50_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)), "mean_ms": float(values.mean())}


def paired_summary(plain, moe):
    differences = np.asarray(moe) - np.asarray(plain)
    rng = np.random.default_rng(42)
    boot = np.median(differences[rng.integers(0, len(differences), (2000, len(differences)))], axis=1)
    return {"median_difference_ms": float(np.median(differences)),
            "median_ratio_moe_over_mlp": float(np.median(np.asarray(moe) / np.asarray(plain))),
            "paired_median_difference_95ci_ms": np.quantile(boot, [0.025, 0.975]).tolist(),
            "note": "Within-session timing uncertainty only; excludes variation across machines or sessions."}


def synthetic_input(tokenizer, target):
    question = "Should governments invest more in public education? Discuss your opinion."
    count = target - len(tokenizer.encode(question, add_special_tokens=False)) - tokenizer.num_special_tokens_to_add(pair=True)
    for _ in range(20):
        essay = "Education " * count
        length = len(tokenizer(question, essay)["input_ids"])
        if length == target:
            return question, essay
        count += target - length
    raise ValueError(f"Could not construct exactly {target} tokenizer tokens.")


def measure_pair(functions, device, warmup, repetitions):
    def call(name):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter_ns()
        functions[name]()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return (time.perf_counter_ns() - started) / 1e6

    for _ in range(warmup):
        for name in ("mlp", "moe"):
            call(name)
    samples = {"mlp": [], "moe": []}
    for repeat in range(repetitions):
        order = ("mlp", "moe") if repeat % 2 == 0 else ("moe", "mlp")
        for name in order:
            samples[name].append(call(name))
    return {**{name: summarize(values) for name, values in samples.items()},
            "comparison": paired_summary(samples["mlp"], samples["moe"]), "samples_ms": samples}


@torch.inference_mode()
def benchmark(checkpoints, device, precisions, lengths, warmup=5, repetitions=50):
    device = torch.device(device)
    scorers = {name: Scorer(folder, device=device) for name, folder in checkpoints.items()}
    if any(s.specification["track"] != "ielts" for s in scorers.values()):
        raise ValueError("Benchmark requires two IELTS checkpoints.")
    if scorers["mlp"].tokenizer.get_vocab() != scorers["moe"].tokenizer.get_vocab():
        raise ValueError("Tokenizers differ.")
    for name, scorer in scorers.items():
        if scorer.specification["head"] != name:
            raise ValueError("Checkpoint head does not match its benchmark label.")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    report = {"environment": environment(), "platform": platform.platform(),
              "processor": platform.processor(), "torch_threads": torch.get_num_threads(),
              "device": str(device), "batch_size": 1, "warmup_per_head_per_case": warmup,
              "paired_repetitions": repetitions, "order": "alternating MLP/MoE and MoE/MLP",
              "timing": "perf_counter_ns with CUDA synchronization before and after each measured call",
              "input": "Synthetic repeated-word text with exact joint token length; latency only, not quality.",
              "end_to_end_includes": ["tokenization", "padding", "device transfer", "model forward",
                                      "score transfer to CPU", "output formatting", "truncation check"],
              "excludes": ["cold model loading", "network transport", "server queueing"],
              "checkpoints": {}, "cases": []}
    for name, folder in checkpoints.items():
        with (Path(folder) / "model.safetensors").open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        report["checkpoints"][name] = {"sha256": digest, "specification": scorers[name].specification,
                                       "parameters": sum(p.numel() for p in scorers[name].model.parameters())}
    for precision in precisions:
        if precision != "float32" and device.type != "cuda":
            raise ValueError("This benchmark uses mixed precision only on CUDA.")
        context = (torch.autocast("cuda", dtype=getattr(torch, precision))
                   if precision != "float32" else nullcontext())
        with context:
            for length in lengths:
                question, essay = synthetic_input(scorers["mlp"].tokenizer, length)
                batches = {}
                for name, scorer in scorers.items():
                    examples = encode_examples(scorer.tokenizer, [question], [essay], "ielts", 1024)
                    batches[name] = {k: v.to(device) for k, v in Collator(scorer.tokenizer)(examples).items()}
                    if batches[name]["input_ids"].shape != (1, length):
                        raise ValueError("Input shape differs from requested benchmark length.")
                end_to_end = {name: lambda s=s, e=essay, q=question: s.predict(e, q)
                              for name, s in scorers.items()}
                forward = {name: lambda s=s, b=batches[name]: s.model(**b)["scores"].float().cpu().tolist()
                           for name, s in scorers.items()}
                case = {"tokens": length, "precision": precision,
                        "end_to_end": measure_pair(end_to_end, device, warmup, repetitions),
                        "pretokenized_forward": measure_pair(forward, device, warmup, repetitions)}
                report["cases"].append(case)
                print(json.dumps({"benchmark_tokens": length, "precision": precision,
                                  "end_to_end": {name: case["end_to_end"][name] for name in scorers}}), flush=True)
            representation = torch.zeros(1, scorers["mlp"].model.encoder.config.hidden_size, device=device)
            functions = {name: lambda s=s, r=representation: s.model.head(r) for name, s in scorers.items()}
            report.setdefault("head_only", {})[precision] = measure_pair(functions, device, warmup, 200)
    del scorers
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp", required=True)
    parser.add_argument("--moe", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precisions", nargs="+", default=["float32"], choices=["float32", "float16"])
    parser.add_argument("--lengths", nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    if args.repetitions < 2 or args.warmup < 1:
        raise ValueError("Use at least one warmup and two paired repetitions.")
    report = benchmark({"mlp": args.mlp, "moe": args.moe}, args.device, args.precisions,
                       args.lengths, args.warmup, args.repetitions)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
