#!/usr/bin/env python3
"""Check current synthetic text against local corpora and the serving tokenizer."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from synthetic_dataset import ROOT, digest, payload, read_candidates, save, words
from tokenizers import Tokenizer

PROJECT = Path(__file__).resolve().parents[1]


def grams(text, n=5):
    tokens = words(text.casefold())
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def main():
    rows, errors = read_candidates(ROOT)
    if errors:
        raise ValueError(f"Invalid source rows: {errors[:2]}")
    # Index only generated n-grams: memory use is bounded by the synthetic set.
    index, denominators, exact_index = defaultdict(list), {}, defaultdict(list)
    matches = {}
    for row in rows:
        identifier = row["candidate_id"]
        distinct = grams(row["essay"])
        denominators[identifier] = len(distinct)
        for gram in distinct:
            index[gram].append(identifier)
        exact_index[digest(words(row["essay"].casefold()))].append(identifier)
        matches[identifier] = {"exact_matches": [], "max_fivegram_containment": 0.0, "nearest": None}
    sources = []

    def compare(text, identity):
        for identifier in exact_index.get(digest(words(text.casefold())), []):
            matches[identifier]["exact_matches"].append(identity)
        shared = Counter()
        for gram in grams(text):
            shared.update(index.get(gram, ()))
        for identifier, count in shared.items():
            containment = count / denominators[identifier]
            if containment > matches[identifier]["max_fivegram_containment"]:
                matches[identifier].update(max_fivegram_containment=containment, nearest=identity)

    for track in ("ielts", "ellipse"):
        for split in ("train", "validation", "test"):
            path = PROJECT / f"artifacts/kaggle-output/preparation/{track}/{split}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Required local comparison corpus missing: {path}")
            table = pq.read_table(path, columns=["id", "essay"])
            sources.append({"track": track, "split": split, "rows": table.num_rows,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            for reference in table.to_pylist():
                compare(reference["essay"], {"track": track, "split": split, "id": str(reference["id"])})
    sanity_path = PROJECT / "artifacts/private-sanity-20261010/samples.json"
    if sanity_path.exists():
        samples = json.loads(sanity_path.read_text())
        for reference in samples:
            compare(reference["essay"], {"track": "private_official_sanity", "id": reference["id"]})
        sources.append({"track": "private_official_sanity", "rows": len(samples),
                        "sha256": hashlib.sha256(sanity_path.read_bytes()).hexdigest()})
    tokenizer_path = PROJECT / "artifacts/staging/huggingface-transfer/ielts-task2-modernbert-base-mlp/tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    lengths = {row["candidate_id"]: len(tokenizer.encode(row["question"], row["essay"]).ids) for row in rows}
    flags = [{"candidate_id": identifier, **result} for identifier, result in matches.items()
             if result["exact_matches"] or (denominators[identifier] >= 40
                                            and result["max_fivegram_containment"] >= 0.20)]
    output = {
        "rows_checked": len(rows),
        "input_set_sha256": digest(sorted((row["candidate_id"], digest(payload(row))) for row in rows)),
        "method": "Exact lowercase word sequences and distinct five-word overlap / synthetic five-word count.",
        "limits": "Lexical checks cannot rule out semantic overlap, shared arguments or pretraining exposure.",
        "sources": sources, "overlap_flags": flags, "row_matches": matches,
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "token_counts": lengths, "max_tokens": max(lengths.values(), default=0),
        "over_1024_tokens": [identifier for identifier, length in lengths.items() if length > 1024],
    }
    save(ROOT / "work/extended_audit.json", output)
    print(json.dumps({"rows": len(rows), "reference_rows": sum(s["rows"] for s in sources),
                      "overlap_flags": len(flags), "max_tokens": output["max_tokens"],
                      "over_1024_tokens": len(output["over_1024_tokens"])}))


if __name__ == "__main__":
    main()
