"""Verify standalone Transformers releases against the original trained checkpoints."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from ielts_scoring.predict import Scorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", choices=["mlp", "moe"], required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    root = Path(__file__).resolve().parents[1]
    frame = pd.read_parquet(root / "artifacts/kaggle-output/preparation/ielts/validation.parquet")
    rows = frame.iloc[[0, 50, 150]].to_dict("records")
    rows.append({"prompt": rows[0]["prompt"], "essay": rows[0]["essay"] * 10})
    questions = [r["prompt"] for r in rows]
    essays = [r["essay"] for r in rows]
    tokenizer = AutoTokenizer.from_pretrained(args.package, local_files_only=True)
    model, loading = AutoModel.from_pretrained(args.package, trust_remote_code=True,
                                              local_files_only=True, output_loading_info=True)
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        assert not loading[key], (key, loading[key])
    model.eval()
    original = Scorer(root / "artifacts/kaggle-output/transfer-v1" / args.head / "best")
    assert model.state_dict().keys() == original.model.state_dict().keys()
    assert all(torch.equal(value, original.model.state_dict()[key])
               for key, value in model.state_dict().items())
    expected = [original.predict(e, q) for q, e in zip(questions, essays)]
    actual = model.score(tokenizer, questions, essays)
    delta = max(abs(a["scores"][t] - b["scores"][t])
                for a, b in zip(actual, expected) for t in model.config.targets)
    assert delta < 1e-4, delta
    assert [r["truncated"] for r in actual] == [r["truncated"] for r in expected]
    assert actual[-1]["truncated"]
    for q, e in (("", "Some essay."), ("Question", "")):
        try:
            model.score(tokenizer, [q], [e])
        except ValueError:
            pass
        else:
            raise AssertionError("Blank inputs were accepted.")
    report = {"head": args.head, "passed": True, "all_weight_tensors_identical": True,
              "strict_loading": loading, "batch_vs_original_max_score_difference": delta,
              "cases": len(rows), "truncation_verified": True, "blank_inputs_rejected": True,
              "offline": True, "torch": torch.__version__}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
