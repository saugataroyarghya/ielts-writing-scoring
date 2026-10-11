#!/usr/bin/env python3
"""Validate, blind, audit and package individually authored synthetic IELTS records.

No text generation or score estimation happens here. Labels must come from actual
separate LLM assessments. Candidate intentions are excluded from judge packets and
model inputs. Run with Python 3.11+; audit and export require only the standard library.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data/synthetic-ielts-v1"
CRITERIA = ("TR", "CC", "LR", "GRA")
TYPES = {"opinion", "discussion", "advantages_disadvantages", "causes_solutions", "two_part"}
FLAGS = {"off_topic", "underlength", "repetitive", "partial_coverage", "contradictory",
         "unrateable", "copied_prompt", "language_errors", "weak_development"}
BOUNDS = {1: (120, 240), 2: (230, 320), 3: (250, 380), 4: (250, 380),
          5: (250, 400), 6: (280, 420), 7: (250, 350), 8: (250, 350), 9: (250, 350)}
NAMESPACE = uuid.UUID("863a7433-5820-478c-a12d-6876559b45d0")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def words(text):
    return re.findall(r"\b[\w]+(?:['’\-][\w]+)*\b", text, flags=re.UNICODE)


def norm(text):
    return " ".join(words(text.casefold()))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def payload(row):
    return {"task": "task2", "question": row["question"], "essay": row["essay"]}


def anonymous_id_for(row):
    revision = row.get("generation", {}).get("revision", 1)
    name = row["candidate_id"] if revision == 1 else f"{row['candidate_id']}:v{revision}"
    return uuid.uuid5(NAMESPACE, name).hex


def candidate_errors(row):
    errors = []
    for field in ("candidate_id", "family_id", "task", "question", "essay", "topic", "question_type"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            errors.append(f"missing/non-string {field}")
    if errors:
        return errors
    match = re.fullmatch(r"([abc])(\d{2})-(\d{2})", row["candidate_id"])
    if not match or not 1 <= int(match[2]) <= 40 or not 1 <= int(match[3]) <= 10:
        errors.append("invalid candidate_id")
        return errors
    if row["family_id"] != row["candidate_id"].split("-")[0]:
        errors.append("family/id mismatch")
    if row["task"] != "task2" or row["question_type"] not in TYPES:
        errors.append("invalid task/question type")
    if not 12 <= len(words(row["question"])) <= 160:
        errors.append("question outside 12–160 word bounds")
    slot, family_num = int(match[3]), int(match[2])
    lower, upper = BOUNDS.get(slot, [(5, 20), (50, 110), (150, 260), (200, 320)][(family_num - 1) % 4])
    count = len(words(row["essay"]))
    if not lower <= count <= upper:
        errors.append(f"word count {count} outside [{lower}, {upper}]")
    if any(marker in row["essay"].casefold() for marker in
           ("as an ai language model", "target band", "intended band", "[insert", "lorem ipsum")):
        errors.append("generation instruction/placeholder in essay")
    generation = row.get("generation", {})
    if type(generation.get("revision", 1)) is not int or generation.get("revision", 1) < 1:
        errors.append("invalid candidate revision")
    if (generation.get("origin") != "llm_synthetic" or generation.get("human_rating") is not False
            or generation.get("method") != "individually_authored"
            or generation.get("slot") != slot or not generation.get("author_agent")):
        errors.append("invalid generation provenance")
    return errors


def read_candidates(root):
    rows, errors = [], []
    for path in sorted((root / "candidates").glob("*/*.json")):
        try:
            row = load(path)
            issue = candidate_errors(row)
            if path.stem != row.get("candidate_id"):
                issue.append("filename/id mismatch")
            if issue:
                errors.append({"path": str(path), "issues": issue})
            else:
                rows.append(row)
        except (ValueError, TypeError, KeyError) as exc:
            errors.append({"path": str(path), "issues": [str(exc)]})
    return rows, errors


def judge_errors(judge, packet):
    errors = []
    if judge.get("id") != packet["id"]:
        errors.append("judge/packet id mismatch")
    if judge.get("input_sha256") != packet["input_sha256"]:
        errors.append("judge not bound to current input hash")
    if not judge.get("judge_id") or judge.get("label_origin") != "llm_judgment":
        errors.append("missing judge identity/origin")
    if type(judge.get("assessable")) is not bool:
        errors.append("assessable must be boolean")
    if judge.get("confidence") not in {"low", "medium", "high"}:
        errors.append("invalid confidence")
    scores = judge.get("scores", {})
    if set(scores) != set(CRITERIA):
        errors.append("scores must contain precisely four criterion keys")
    for criterion in CRITERIA:
        score = scores.get(criterion)
        if judge.get("assessable") is True:
            if type(score) is not int or not 0 <= score <= 9:
                errors.append(f"invalid {criterion} band")
            elif len(words(packet["essay"])) <= 20 and score != 1:
                errors.append(f"{criterion}: <=20 words requires band 1 under public descriptors")
        elif score is not None:
            errors.append(f"unassessable {criterion} must be null")
        evidence = judge.get("evidence", {}).get(criterion, {})
        quote, reason = evidence.get("quote"), evidence.get("reason")
        if not isinstance(quote, str) or not quote or quote not in packet["essay"]:
            errors.append(f"{criterion} evidence quote is not an exact essay substring")
        if not isinstance(reason, str) or len(reason.strip()) < 15:
            errors.append(f"{criterion} lacks substantive reason")
    if not isinstance(judge.get("flags"), list) or any(flag not in FLAGS for flag in judge["flags"]):
        errors.append("invalid flags")
    if not isinstance(judge.get("concerns"), str):
        errors.append("concerns must be string")
    return errors


def family_groups(root=None):
    groups = {f"{shard}{i:02d}": f"{shard}{i:02d}" for shard in "abc" for i in range(1, 41)}
    path = root / "generation/linked_families.json" if root else None
    if path and path.exists():
        seen = set()
        for canonical, members in load(path).items():
            if canonical not in members or any(member not in groups or member in seen for member in members):
                raise ValueError("Invalid or overlapping linked-family group")
            for member in members:
                groups[member] = canonical
                seen.add(member)
    return groups


def family_splits(root=None):
    families = [f"{shard}{i:02d}" for shard in "abc" for i in range(1, 41)]
    random.Random(20261010).shuffle(families)
    initial = {family: "train" if i < 96 else "synthetic_validation" if i < 108
               else "synthetic_diagnostic" for i, family in enumerate(families)}
    return {family: initial[group] for family, group in family_groups(root).items()}


def near_duplicates(rows):
    """Five-word overlap is a review signal, not a semantic originality guarantee."""
    postings, sizes, overlap = defaultdict(list), {}, Counter()
    for row in rows:
        tokens = words(row["essay"].casefold())
        grams = {tuple(tokens[i:i + 5]) for i in range(len(tokens) - 4)}
        if len(grams) < 40:
            continue
        identifier = row["candidate_id"]
        sizes[identifier] = len(grams)
        for gram in grams:
            for other in postings[gram]:
                overlap[tuple(sorted((identifier, other)))] += 1
            postings[gram].append(identifier)
    return [{"ids": list(pair), "fivegram_containment": count / min(sizes[pair[0]], sizes[pair[1]])}
            for pair, count in sorted(overlap.items())
            if count / min(sizes[pair[0]], sizes[pair[1]]) >= 0.30]


def packetize(root, shard=None, family=None):
    rows, errors = read_candidates(root)
    if errors:
        raise ValueError(f"Fix invalid candidates first: {errors[:5]}")
    if shard:
        rows = [row for row in rows if row["candidate_id"].startswith(shard)]
    if family:
        rows = [row for row in rows if row["family_id"] == family]
    # A successful syntax check is not the writer's final handoff.
    ready_rows = []
    for row in rows:
        handoff = root / "work" / f"ready-{row['family_id']}.json"
        if not handoff.exists():
            continue
        marker = load(handoff)
        is_ready = marker.get("ready") is True or marker.get("status") == "final_generator_checked"
        if (not is_ready or marker.get("family_id") != row["family_id"]
                or marker.get("author_agent") != row["generation"]["author_agent"]):
            raise ValueError(f"Invalid writer handoff for {row['family_id']}")
        ready_rows.append(row)
    rows = ready_rows
    packets = []
    for row in rows:
        data = payload(row)
        hash_value = digest(data)
        anonymous_id = anonymous_id_for(row)
        packet = {"id": anonymous_id, "input_sha256": hash_value, **data}
        path = root / "work" / "packets" / f"{anonymous_id}.json"
        if path.exists() and load(path) != packet:
            raise ValueError(f"Sealed input changed: {row['candidate_id']}; explicitly version it before rejudging")
        save(path, packet)
        save(root / "generation" / f"{anonymous_id}.json", {
            "candidate_id": row["candidate_id"], "id": anonymous_id,
            "family_id": row["family_id"], "input_sha256": hash_value,
            "split": family_splits(root)[row["family_id"]], **row["generation"]})
        packets.append(packet)
    # Anonymous order cannot reveal target proficiency slots or families.
    packets.sort(key=lambda packet: packet["id"])
    manifest = root / "work" / (f"packets-{shard}.jsonl" if shard else "packets.jsonl")
    manifest.write_text("".join(json.dumps(packet, ensure_ascii=False) + "\n" for packet in packets))
    return {"packets": len(packets), "path": str(manifest)}


def audit(root, expected=1200):
    rows, structural_errors = read_candidates(root)
    errors = list(structural_errors)
    ids = [row["candidate_id"] for row in rows]
    if len(set(ids)) != len(ids):
        errors.append({"issues": ["duplicate candidate ids"]})
    families, essay_hashes, questions = defaultdict(list), defaultdict(list), defaultdict(set)
    for row in rows:
        families[row["family_id"]].append(row)
        essay_hashes[digest(norm(row["essay"]))].append(row["candidate_id"])
        questions[row["family_id"]].add(row["question"])
    for family, values in questions.items():
        if len(values) != 1:
            errors.append({"family": family, "issues": ["prompt differs within family"]})
    prompt_owners = defaultdict(set)
    for row in rows:
        prompt_owners[norm(row["question"])].add(family_groups(root)[row["family_id"]])
    for owners in prompt_owners.values():
        if len(owners) > 1:
            errors.append({"issues": ["identical prompts assigned to different split groups"],
                           "groups": sorted(owners)})
    duplicate_groups = [values for values in essay_hashes.values() if len(values) > 1]
    if duplicate_groups:
        errors.append({"issues": ["normalised duplicate essays"], "groups": duplicate_groups})
    if len(rows) != expected:
        errors.append({"issues": [f"expected {expected} valid candidates, found {len(rows)}"]})
    near_duplicate_flags = near_duplicates(rows)
    if near_duplicate_flags:
        errors.append({"issues": ["near-duplicate response pairs require explicit review"],
                       "pairs": near_duplicate_flags})
    judgments = defaultdict(list)
    for path in sorted((root / "jury").glob("*/*.json")):
        try:
            judgment = load(path)
            packet_path = root / "work" / "packets" / f"{judgment['id']}.json"
            packet = load(packet_path)
            issue = judge_errors(judgment, packet)
            if issue:
                errors.append({"path": str(path), "issues": issue})
            else:
                judgments[judgment["id"]].append(judgment)
        except (ValueError, TypeError, KeyError, FileNotFoundError) as exc:
            errors.append({"path": str(path), "issues": [str(exc)]})
    requirements_path = root / "work/adjudication_requirements.json"
    requirements = load(requirements_path) if requirements_path.exists() else {}
    accepted, pending, held = [], [], []
    for row in rows:
        anonymous_id = anonymous_id_for(row)
        ratings = judgments.get(anonymous_id, [])
        current_hash = digest(payload(row))
        if not ratings:
            pending.append(row["candidate_id"])
            continue
        if any(r["input_sha256"] != current_hash for r in ratings):
            errors.append({"candidate_id": row["candidate_id"], "issues": ["judgment is stale"]})
            continue
        if any(r["judge_id"] == row["generation"]["author_agent"] for r in ratings):
            errors.append({"candidate_id": row["candidate_id"], "issues": ["self-judging is not independent review"]})
            continue
        if len({r["judge_id"] for r in ratings}) != len(ratings):
            errors.append({"candidate_id": row["candidate_id"], "issues": ["repeated judge identity"]})
            continue
        disagreement = any(r["scores"] != ratings[0]["scores"]
                           or r["assessable"] != ratings[0]["assessable"] for r in ratings[1:])
        adjudication = None
        adjudication_path = root / "adjudications" / f"{anonymous_id}.json"
        if adjudication_path.exists():
            adjudication = load(adjudication_path)
            packet = {"id": anonymous_id, "input_sha256": current_hash, **payload(row)}
            issue = judge_errors(adjudication, packet)
            if adjudication.get("reviewed_judgment_hashes") != sorted(digest(r) for r in ratings):
                issue.append("adjudication does not cover current immutable judgments")
            if adjudication.get("judge_id") in {row["generation"]["author_agent"],
                                                *(r["judge_id"] for r in ratings)}:
                issue.append("adjudicator must be a separate assessor")
            if len(adjudication.get("resolution", "")) < 30:
                issue.append("adjudication lacks a resolution explanation")
            if issue:
                errors.append({"candidate_id": row["candidate_id"], "issues": issue})
                continue
        required = requirements.get(anonymous_id)
        if required and (not adjudication or adjudication.get("rubric_protocol") != required["protocol"]):
            held.append({"candidate_id": row["candidate_id"], "reason": required["reason"]})
            continue
        if not adjudication and (disagreement or any(r["confidence"] == "low" for r in ratings)):
            held.append({"candidate_id": row["candidate_id"], "reason": "disagreement/low confidence"})
            continue
        label = adjudication or ratings[0]
        if label["confidence"] == "low":
            held.append({"candidate_id": row["candidate_id"], "reason": "adjudication remains uncertain"})
            continue
        score_values = list(label["scores"].values())
        accepted.append({
            "id": anonymous_id, "family_id": row["family_id"],
            "prompt_group_id": family_groups(root)[row["family_id"]],
            "split": family_splits(root)[row["family_id"]], **payload(row),
            "topic": row["topic"], "question_type": row["question_type"],
            "word_count": len(words(row["essay"])), "scores": label["scores"],
            "task_mean": sum(score_values) / 4 if label["assessable"] else None,
            "assessable": label["assessable"], "flags": label["flags"],
            "label_origin": "llm_judgment", "human_verified": False,
            "synthetic": True, "input_sha256": current_hash,
            "ratings": ratings, "adjudication": adjudication,
            "generator": row["generation"]["author_agent"],
        })
    input_set_hash = digest(sorted((row["candidate_id"], digest(payload(row))) for row in rows))
    extended_path = root / "work/extended_audit.json"
    extended = load(extended_path) if extended_path.exists() else None
    extended_ready = bool(extended and extended.get("input_set_sha256") == input_set_hash
                          and not extended.get("overlap_flags") and not extended.get("over_1024_tokens"))
    report = {
        "created_at": datetime.now(UTC).isoformat(), "expected": expected,
        "candidate_files": len(list((root / "candidates").glob("*/*.json"))),
        "valid_candidates": len(rows), "families": len(families),
        "structural_errors": errors, "pending_judgment": pending,
        "held_for_review": held, "accepted": len(accepted),
        "human_verified_rows": 0, "exact_duplicate_groups": duplicate_groups,
        "near_duplicate_flags": near_duplicate_flags,
        "observed_prompt_groups": len({family_groups(root)[family] for family in families}),
        "split_counts": dict(Counter(row["split"] for row in accepted)),
        "question_types": dict(Counter(row["question_type"] for row in rows)),
        "score_distribution": {c: dict(Counter(str(r["scores"][c]) for r in accepted)) for c in CRITERIA},
        "input_set_sha256": input_set_hash, "extended_audit_current_and_clear": extended_ready,
        "release_ready": (not errors and not pending and not held and len(accepted) == expected
                          and (expected < 1000 or extended_ready)),
        "quarantine_release_ready": (not errors and not pending
                                      and len(accepted) + len(held) == expected
                                      and all(h["reason"] == "adjudication remains uncertain" for h in held)
                                      and (expected < 1000 or extended_ready)),
        "limits": ["Same model family may generate and judge; errors are correlated.",
                   "LLM scores are not examiner ground truth.",
                   "Synthetic validation/diagnostic splits cannot establish real-learner accuracy.",
                   "Structural checks cannot establish linguistic or scoring quality."]}
    save(root / "work" / "audit.json", report)
    return report, accepted


def release(root, expected, allow_quarantine=False):
    report, accepted = audit(root, expected)
    if not report["release_ready"] and not (allow_quarantine and report["quarantine_release_ready"]):
        raise ValueError(f"Release blocked: {len(report['structural_errors'])} errors; "
                         f"{len(report['pending_judgment'])} unjudged; "
                         f"{len(report['held_for_review'])} held; "
                         f"extended audit clear/current={report['extended_audit_current_and_clear']}. "
                         "Inspect work/audit.json")
    output = root / "release"
    output.mkdir(exist_ok=True)
    held_ids = {item["candidate_id"]: item["reason"] for item in report["held_for_review"]}
    quarantined = []
    if held_ids:
        candidates, _ = read_candidates(root)
        for row in candidates:
            if row["candidate_id"] not in held_ids:
                continue
            identifier = anonymous_id_for(row)
            ratings = [load(p) for p in sorted((root / "jury").glob(f"*/{identifier}.json"))]
            quarantined.append({
                "id": identifier, "family_id": row["family_id"],
                "prompt_group_id": family_groups(root)[row["family_id"]],
                "intended_split": family_splits(root)[row["family_id"]], **payload(row),
                "input_sha256": digest(payload(row)), "synthetic": True,
                "human_verified": False, "label_origin": "llm_judgment",
                "scores": dict.fromkeys(CRITERIA), "task_mean": None,
                "status": "quarantined", "reason": held_ids[row["candidate_id"]],
                "ratings": ratings,
                "adjudication": load(root / "adjudications" / f"{identifier}.json"),
            })
    (output / "quarantine.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in quarantined), encoding="utf-8")
    accepted.sort(key=lambda row: row["id"])
    for split in ("train", "synthetic_validation", "synthetic_diagnostic"):
        selected = [row for row in accepted if row["split"] == split]
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
        # Explicit feature whitelist keeps labels, intended ability and rationales out of inference input.
        (output / f"{split}_inputs.jsonl").write_text("".join(json.dumps(
            {k: row[k] for k in ("id", "task", "question", "essay")}, ensure_ascii=False) + "\n"
            for row in selected))
        with (output / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["id", "family_id", "prompt_group_id", "split", "task", "question", "essay",
                      "word_count", *CRITERIA, "task_mean", "assessable", "label_origin",
                      "human_verified", "synthetic", "input_sha256"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in selected:
                writer.writerow({field: row["scores"].get(field) if field in CRITERIA else row[field]
                                 for field in fields})
    save(output / "audit.json", {**report, "quarantined_rows": len(quarantined),
                                "export_mode": "quarantine_separated" if quarantined else "all_accepted"})
    save(output / "checksums.json", {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in sorted(output.iterdir())
                                     if path.suffix in {".jsonl", ".csv"}})
    return {"release": str(output), "rows": len(accepted), "quarantined_rows": len(quarantined)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check-row", "check-judge", "packetize", "audit", "release"))
    parser.add_argument("path", nargs="?", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--shard", choices=list("abc"))
    parser.add_argument("--family", help="Only packetize this explicitly completed family")
    parser.add_argument("--expected", type=int, default=1200)
    parser.add_argument("--allow-quarantine", action="store_true",
                        help="Export unresolved, fully adjudicated rows separately without training targets")
    args = parser.parse_args()
    try:
        if args.command == "check-row":
            row = load(args.path)
            issues = candidate_errors(row)
            result = {"id": row.get("candidate_id"), "words": len(words(row.get("essay", ""))),
                      "ok": not issues, "issues": issues}
        elif args.command == "check-judge":
            judgment = load(args.path)
            packet = load(args.root / "work" / "packets" / f"{judgment['id']}.json")
            issues = judge_errors(judgment, packet)
            result = {"id": judgment["id"], "ok": not issues, "issues": issues}
        elif args.command == "packetize":
            result = packetize(args.root, args.shard, args.family)
        elif args.command == "release":
            result = release(args.root, args.expected, args.allow_quarantine)
        else:
            report, _ = audit(args.root, args.expected)
            result = {key: report[key] for key in ("candidate_files", "valid_candidates", "families",
                                                  "accepted", "release_ready")}
            result.update(errors=len(report["structural_errors"]),
                          pending=len(report["pending_judgment"]), held=len(report["held_for_review"]))
        print(json.dumps(result, ensure_ascii=False))
        return 1 if result.get("ok") is False else 0
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
