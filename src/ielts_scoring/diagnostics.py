"""Validation-only perturbation probes; no invented labels or benchmark-accuracy claims."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .models import EssayScorer
from .train import Collator, encode_examples, find_data, predict_loader
from .transfer import find_checkpoint


def diagnose(root, runs):
    folder, _ = find_data(root, "ielts")
    validation = pd.read_parquet(folder / "validation.parquet")
    probes = validation.sample(frac=1, random_state=42).drop_duplicates("group_id").head(32)
    if len(probes) < 2:
        raise ValueError("Need at least two distinct validation prompt groups.")
    questions = probes["prompt"].tolist()
    essays = probes["essay"].tolist()
    cases = {"original": (questions, essays),
             "different_question": (questions[1:] + questions[:1], essays),
             "short_40_words": (questions, [" ".join(essay.split()[:40]) for essay in essays])}
    paths = {f"direct_{head}": find_checkpoint(root, "ielts", head) for head in ("mlp", "moe")}
    paths.update({f"transfer_{head}": Path(runs) / head / "best" for head in ("mlp", "moe")})
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    report = {"n": len(probes), "ids": probes["id"].tolist(), "source": "IELTS validation only",
              "note": "Perturbed examples have no human scores. Directional sensitivity probes do not establish accuracy.",
              "models": {}}
    for name, path in paths.items():
        model = EssayScorer.load(path, device)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        outputs = {}
        for case, (prompts, answers) in cases.items():
            examples = encode_examples(tokenizer, prompts, answers, "ielts", 1024)
            outputs[case], _, _ = predict_loader(model, DataLoader(examples, batch_size=4,
                                                                  collate_fn=Collator(tokenizer)), device)
        targets = model.specification["targets"]
        report["models"][name] = {case: {
            target: {"mean_score": float(values[:, i].mean()),
                     "mean_change_from_original": float((values - outputs["original"])[:, i].mean()),
                     "median_change_from_original": float(np.median((values - outputs["original"])[:, i])),
                     "fraction_lower_than_original": float((values[:, i] < outputs["original"][:, i]).mean())}
            for i, target in enumerate(targets)} for case, values in outputs.items()}
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/kaggle/input")
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = diagnose(args.input, args.runs)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
