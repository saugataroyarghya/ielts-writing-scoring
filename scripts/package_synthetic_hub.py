"""Build a public, allowlisted dataset package from the frozen synthetic release."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml
from huggingface_hub import DatasetCard

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/synthetic-ielts-v1"
DEST = ROOT / "artifacts/staging/ielts-task2-synthetic-v1"
REPO = "saugataroy/ielts-task2-synthetic-v1"
SPLITS = {"train": 976, "synthetic_validation": 98, "synthetic_diagnostic": 120}
TARGETS = ["TR", "CC", "LR", "GRA"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(license_id):
    receipt = json.loads((SOURCE / "release/verification.json").read_text())
    assert receipt["status"] == "passed" and receipt["split_counts"] == SPLITS
    for folder in ["data", "annotations", "audit", "protocol"]:
        (DEST / folder).mkdir(parents=True, exist_ok=True)
    seen, groups = set(), {}
    for split, count in SPLITS.items():
        source = SOURCE / f"release/{split}.jsonl"
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        assert len(rows) == count
        flat = []
        for row in rows:
            assert row["id"] not in seen
            seen.add(row["id"])
            assert groups.setdefault(row["prompt_group_id"], split) == split
            assert row["synthetic"] and not row["human_verified"]
            assert all(isinstance(row["scores"][key], int) for key in TARGETS)
            assert row["task_mean"] == sum(row["scores"].values()) / 4
            fields = ["id", "family_id", "prompt_group_id", "split", "task", "question", "essay",
                      "topic", "question_type", "word_count", "task_mean", "flags", "label_origin",
                      "human_verified", "synthetic", "input_sha256", "generator"]
            flat.append({**{key: row[key] for key in fields}, **row["scores"]})
        frame = pd.DataFrame(flat)
        frame.to_parquet(DEST / f"data/{split}.parquet", index=False)
        restored = pd.read_parquet(DEST / f"data/{split}.parquet")
        assert restored["essay"].tolist() == frame["essay"].tolist()
        assert restored[TARGETS].equals(frame[TARGETS])
        shutil.copy2(source, DEST / f"annotations/{split}.jsonl")
    # Quarantine has a deliberately separate config and no usable numeric labels.
    quarantine = [json.loads(line) for line in (SOURCE / "release/quarantine.jsonl").read_text().splitlines()]
    assert len(quarantine) == 6 and all(set(row["scores"].values()) == {None} for row in quarantine)
    qflat = [{**{key: row[key] for key in ["id", "family_id", "prompt_group_id", "intended_split", "task",
                                        "question", "essay", "input_sha256", "synthetic", "human_verified",
                                        "label_origin", "task_mean", "status", "reason"]},
              **row["scores"]} for row in quarantine]
    qframe = pd.DataFrame(qflat)
    for key in [*TARGETS, "task_mean"]:
        qframe[key] = qframe[key].astype("Float64")
    qframe.to_parquet(DEST / "data/quarantine.parquet", index=False)
    shutil.copy2(SOURCE / "release/quarantine.jsonl", DEST / "annotations/quarantine.jsonl")
    for name in ["GENERATOR.md", "JUDGE.md", "ADJUDICATOR.md"]:
        shutil.copy2(SOURCE / "protocol" / name, DEST / "protocol" / name)
    # Publish only self-contained aggregate audit facts, never reference essays or local paths.
    summary = json.loads((SOURCE / "work/quality_summary.json").read_text())
    stats = {key: summary[key] for key in ["judged_score_distribution", "judged_criterion_means", "review_flags"]}
    stats["verification"] = receipt
    (DEST / "audit/quality.json").write_text(json.dumps(stats, indent=2) + "\n")
    metadata = {
        "pretty_name": "IELTS Task 2 Synthetic Rubric Scores v1", "language": ["en"],
        "size_categories": ["1K<n<10K"],
        "tags": ["synthetic", "ielts", "essay-scoring", "regression", "text", "llm-annotations"],
        "configs": [
            {"config_name": "default", "default": True,
             "data_files": [{"split": split, "path": f"data/{split}.parquet"} for split in SPLITS]},
            {"config_name": "quarantine", "data_files": [{"split": "quarantine", "path": "data/quarantine.parquet"}]},
        ],
    }
    if license_id != "unspecified":
        metadata["license"] = license_id
    license_text = {
        "cc-by-4.0": "The original synthetic questions, responses, annotations, and documentation are released under "
                     "[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). "
                     "Attribute this dataset and indicate changes. Third-party rubric documents and trademarks are "
                     "not covered by this grant and are not redistributed.",
        "cc0-1.0": "The original synthetic questions, responses, annotations, and documentation are released under "
                    "[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). Third-party rubric documents and "
                    "trademarks are not covered and are not redistributed.",
        "unspecified": "No distribution license has been selected. Public availability does not itself grant an "
                       "open reuse license. Contact the publisher through this repository before relying on reuse rights.",
    }[license_id]
    template = (ROOT / "release_assets/synthetic_dataset_card.md").read_text()
    card = "---\n" + yaml.safe_dump(metadata, sort_keys=False) + "---\n\n" + template.replace("{{LICENSE}}", license_text)
    (DEST / "README.md").write_text(card)
    (DEST / "LICENSE.md").write_text("# Dataset license\n\n" + license_text + "\n")
    DatasetCard.load(DEST / "README.md")
    manifest = {"version": "1.0", "repo_id": REPO, "rows": 1200, "accepted_rows": 1194,
                "quarantined_rows": 6, "split_counts": SPLITS, "human_verified_rows": 0,
                "model_id": None, "generation_model_id_note": "Not exposed by the agent tool.",
                "input_set_sha256": receipt["input_set_sha256"], "license": license_id}
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    files = {p.relative_to(DEST).as_posix(): digest(p) for p in sorted(DEST.rglob("*"))
             if p.is_file() and p.name != "checksums.json"}
    (DEST / "checksums.json").write_text(json.dumps(files, indent=2) + "\n")
    print(json.dumps({"repo_id": REPO, "folder": str(DEST), "files": len(files) + 1,
                      "accepted_rows": len(seen), "license": license_id}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--license", choices=["cc-by-4.0", "cc0-1.0", "unspecified"], required=True)
    main(parser.parse_args().license)
