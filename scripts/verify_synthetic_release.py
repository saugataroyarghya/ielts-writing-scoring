#!/usr/bin/env python3
"""Reconcile the exported dataset against current reviews, CSVs and input whitelists."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict

from synthetic_dataset import (
    CRITERIA,
    ROOT,
    anonymous_id_for,
    audit,
    digest,
    load,
    payload,
    read_candidates,
    save,
)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    report, accepted = audit(ROOT)
    assert report["quarantine_release_ready"], "Underlying assessment work is incomplete"
    output = ROOT / "release"
    exported_audit = load(output / "audit.json")
    assert exported_audit["input_set_sha256"] == report["input_set_sha256"]
    assert exported_audit["accepted"] == report["accepted"]
    for name, expected_hash in load(output / "checksums.json").items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected_hash, name

    current = {row["id"]: row for row in accepted}
    seen, groups, split_counts = set(), defaultdict(set), {}
    for split in ("train", "synthetic_validation", "synthetic_diagnostic"):
        rows = read_jsonl(output / f"{split}.jsonl")
        inputs = read_jsonl(output / f"{split}_inputs.jsonl")
        with (output / f"{split}.csv").open(newline="", encoding="utf-8") as handle:
            csv_rows = list(csv.DictReader(handle))
        assert len(rows) == len(inputs) == len(csv_rows)
        split_counts[split] = len(rows)
        for row, model_input, csv_row in zip(rows, inputs, csv_rows, strict=True):
            identifier = row["id"]
            assert identifier not in seen
            seen.add(identifier)
            assert row == current[identifier], f"Stale or modified export: {identifier}"
            assert row["split"] == split
            assert row["synthetic"] is True and row["human_verified"] is False
            assert model_input == {key: row[key] for key in ("id", "task", "question", "essay")}
            assert row["input_sha256"] == digest(payload(row))
            for key in ("id", "question", "essay", "family_id", "prompt_group_id", "split"):
                assert csv_row[key] == row[key], (identifier, key)
            for criterion in CRITERIA:
                value = row["scores"][criterion]
                assert csv_row[criterion] == ("" if value is None else str(value))
            if row["assessable"]:
                assert row["task_mean"] == sum(row["scores"].values()) / 4
                assert float(csv_row["task_mean"]) == row["task_mean"]
            else:
                assert row["task_mean"] is None and csv_row["task_mean"] == ""
            groups[row["prompt_group_id"]].add(split)

    quarantined = read_jsonl(output / "quarantine.jsonl")
    assert len(quarantined) == len(report["held_for_review"])
    for row in quarantined:
        assert row["id"] not in seen
        seen.add(row["id"])
        assert row["status"] == "quarantined"
        assert all(value is None for value in row["scores"].values())
        assert row["task_mean"] is None and row["adjudication"]["confidence"] == "low"
        assert row["input_sha256"] == digest(payload(row))
        groups[row["prompt_group_id"]].add(row["intended_split"])
    assert all(len(splits) == 1 for splits in groups.values()), "Prompt group crosses splits"
    candidates, errors = read_candidates(ROOT)
    assert not errors and seen == {anonymous_id_for(row) for row in candidates}
    receipt = {
        "status": "passed", "authored_rows": len(candidates), "accepted_rows": len(current),
        "quarantined_rows": len(quarantined), "split_counts": split_counts,
        "prompt_groups": len(groups), "input_set_sha256": report["input_set_sha256"],
        "checks": ["export checksums", "current review equality", "CSV/JSONL parity",
                   "model input whitelist", "mean arithmetic", "quarantine separation",
                   "all candidate IDs covered once", "prompt group split integrity"],
    }
    save(output / "verification.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
