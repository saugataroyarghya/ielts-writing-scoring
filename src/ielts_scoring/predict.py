"""Offline-capable inference for a downloaded release directory."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .models import EssayScorer
from .train import Collator, encode_examples


class Scorer:
    def __init__(self, checkpoint, device="cpu"):
        self.model = EssayScorer.load(checkpoint, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        self.specification = self.model.specification
        self.device = device

    @torch.no_grad()
    def predict(self, essay, question=None):
        if not essay or not essay.strip():
            raise ValueError("Provide a nonempty essay.")
        if self.specification["track"] == "ielts" and (not question or not question.strip()):
            raise ValueError("An IELTS Task 2 estimate requires the question.")
        examples = encode_examples(self.tokenizer, [question or ""], [essay], self.specification["track"],
                                   self.specification["max_length"])
        batch = {k: v.to(self.device) for k, v in Collator(self.tokenizer)(examples).items()}
        scores = self.model(**batch)["scores"][0].float().cpu().tolist()
        result = {"scores": dict(zip(self.specification["targets"], scores)),
                  "track": self.specification["track"], "max_tokens": self.specification["max_length"]}
        full_length = len(self.tokenizer.encode(essay, add_special_tokens=False))
        if self.specification["track"] == "ielts":
            full_length += len(self.tokenizer.encode(question, add_special_tokens=False))
            result["task2_mean_estimate"] = sum(scores) / len(scores)
        full_length += self.tokenizer.num_special_tokens_to_add(pair=self.specification["track"] == "ielts")
        result["truncated"] = full_length > self.specification["max_length"]
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("essay_file")
    parser.add_argument("--question")
    args = parser.parse_args()
    print(json.dumps(Scorer(args.checkpoint).predict(Path(args.essay_file).read_text(), args.question), indent=2))
