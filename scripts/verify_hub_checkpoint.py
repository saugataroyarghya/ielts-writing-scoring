"""Verify a packaged/downloaded checkpoint in a fresh, offline CPU process."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from pathlib import Path


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    folder = args.checkpoint.resolve()
    for name, expected in json.loads((folder / "checksums.json").read_text()).items():
        assert checksum(folder / name) == expected, f"Checksum mismatch: {name}"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path.insert(0, str(folder / "src"))
    import torch

    from ielts_scoring import predict

    assert Path(predict.__file__).resolve().is_relative_to(folder / "src")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    scorer = predict.Scorer(folder, device="cpu")
    question = "Should cities make public transport free? Discuss both views."
    essay = (
        "Making public transport free could help residents reach work and education without worrying "
        "about the price of a ticket. It could also encourage some drivers to leave their cars at home. "
        "However, removing fares would not automatically make buses more frequent or reliable.\n\n"
        "The money would need to come from taxes or another source. A city with crowded buses might "
        "benefit more from additional services than from free tickets. In my view, local authorities "
        "should first improve reliability and offer discounted travel to people who need it most. "
        "They can then evaluate whether a wider free service would be affordable and useful."
    )
    result = scorer.predict(essay, question if scorer.specification["track"] == "ielts" else None)
    assert list(result["scores"]) == scorer.specification["targets"]
    assert all(math.isfinite(value) for value in result["scores"].values())
    assert result["truncated"] is False
    if scorer.specification["track"] == "ielts":
        assert math.isclose(result["task2_mean_estimate"], sum(result["scores"].values()) / 4)
    maximum_difference = None
    if args.reference:
        reference = json.loads(args.reference.read_text())["result"]
        maximum_difference = max(abs(result["scores"][name] - reference["scores"][name])
                                 for name in result["scores"])
        assert maximum_difference <= 1e-6, f"Inference changed: {maximum_difference}"
    report = {
        "passed": True, "offline": True, "device": "cpu", "fresh_process": True,
        "input_provenance": "Synthetic smoke-test text, not a scored or benchmark example",
        "question": question if scorer.specification["track"] == "ielts" else None,
        "essay": essay, "result": result, "max_difference_from_reference": maximum_difference,
        "packages": {name: importlib.metadata.version(name)
                     for name in ["torch", "transformers", "huggingface-hub", "safetensors"]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"checkpoint": folder.name, "passed": True,
                      "max_difference_from_reference": maximum_difference}), flush=True)


if __name__ == "__main__":
    main()
