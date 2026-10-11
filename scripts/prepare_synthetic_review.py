#!/usr/bin/env python3
"""Prepare a disjoint, immutable, blind review batch from completed families."""
from __future__ import annotations

import argparse
import json
import re

from synthetic_dataset import ROOT, anonymous_id_for, digest, load, packetize, read_candidates, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9_]*", args.name) or args.limit < 1:
        raise ValueError("Use a simple lowercase batch name and a positive limit")
    folder = ROOT / "work/review-batches"
    path = folder / f"{args.name}.jsonl"
    if path.exists():
        raise FileExistsError(f"Batch already frozen: {path}")
    packetize(ROOT)
    previously_assigned = set()
    for metadata_path in folder.glob("*.json"):
        previously_assigned.update(load(metadata_path)["ids"])
    judged = {p.stem for p in (ROOT / "jury").glob("*/*.json")}
    rows, errors = read_candidates(ROOT)
    if errors:
        raise ValueError(errors[:3])
    pending = []
    for row in rows:
        identifier = anonymous_id_for(row)
        packet_path = ROOT / "work/packets" / f"{identifier}.json"
        if identifier in previously_assigned or identifier in judged or not packet_path.exists():
            continue
        pending.append(load(packet_path))
    pending.sort(key=lambda packet: packet["id"])
    selected = pending[:args.limit]
    if not selected:
        raise ValueError("No unassigned, unjudged, completed rows available")
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(packet, ensure_ascii=False) + "\n" for packet in selected),
                    encoding="utf-8")
    save(folder / f"{args.name}.json", {
        "name": args.name, "status": "prepared", "rows": len(selected),
        "ids": [packet["id"] for packet in selected], "packet_set_sha256": digest(selected),
        "notes": "Prepared for independent LLM review; this file is not evidence of completed judgment."})
    print(json.dumps({"path": str(path), "rows": len(selected), "remaining_unassigned_ready":
                      len(pending) - len(selected)}))


if __name__ == "__main__":
    main()
