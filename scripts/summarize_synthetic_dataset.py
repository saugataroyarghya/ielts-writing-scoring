#!/usr/bin/env python3
"""Summarize actual authored text and assessed labels; never estimate missing labels."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from statistics import mean, median

from synthetic_dataset import CRITERIA, ROOT, audit, digest, read_candidates, save, words


def main():
    report, accepted = audit(ROOT)
    candidates, errors = read_candidates(ROOT)
    if errors:
        raise ValueError(errors[:3])
    lengths = sorted(len(words(row["essay"])) for row in candidates)
    sentences = defaultdict(set)
    for row in candidates:
        for sentence in re.split(r"(?<=[.!?])\s+", row["essay"]):
            tokens = words(sentence.casefold())
            if len(tokens) >= 12:
                sentences[" ".join(tokens)].add(row["family_id"])
    recurring = [{"sentence": sentence, "families": sorted(families)}
                 for sentence, families in sentences.items() if len(families) >= 3]
    pairs = []
    for row in accepted:
        for left, right in combinations(row["ratings"], 2):
            if left["assessable"] and right["assessable"]:
                pairs.append({"id": row["id"], "judges": [left["judge_id"], right["judge_id"]],
                              "full_agreement": left["scores"] == right["scores"],
                              "absolute_difference": {
                                  c: abs(left["scores"][c] - right["scores"][c]) for c in CRITERIA}})
    labels = [row for row in accepted if row["assessable"]]
    result = {
        "input_set_sha256": report["input_set_sha256"],
        "review_set_sha256": digest([(row["id"], row["ratings"], row["adjudication"])
                                     for row in accepted]),
        "status": ("complete" if report["release_ready"] else "complete_with_quarantine"
                   if report["quarantine_release_ready"] else "in_progress"),
        "counts": {key: report[key] for key in ("expected", "candidate_files", "valid_candidates",
                                                "families", "observed_prompt_groups", "accepted",
                                                "human_verified_rows", "split_counts")},
        "pending_judgments": len(report["pending_judgment"]),
        "unresolved_holds": len(report["held_for_review"]),
        "quarantined_cases": report["held_for_review"],
        "structural_or_provenance_errors": len(report["structural_errors"]),
        "authored_by": dict(Counter(row["generation"]["author_agent"] for row in candidates)),
        "word_count": {"min": min(lengths, default=0), "median": median(lengths) if lengths else 0,
                       "max": max(lengths, default=0), "under_250": sum(x < 250 for x in lengths),
                       "at_most_20": sum(x <= 20 for x in lengths)},
        "paragraph_counts": dict(Counter(len([p for p in row["essay"].split("\n\n") if p.strip()])
                                          for row in candidates)),
        "long_sentences_reused_across_three_or_more_families": recurring,
        "question_types": report["question_types"],
        "judged_score_distribution": report["score_distribution"],
        "judged_criterion_means": {c: mean(row["scores"][c] for row in labels) if labels else None
                                   for c in CRITERIA},
        "review_flags": dict(Counter(flag for row in accepted for flag in row["flags"])),
        "adjudicated_rows": sum(row["adjudication"] is not None for row in accepted),
        "pairwise_review_comparisons": pairs,
        "pairwise_review_note": (
            "Selected multiply reviewed rows, with correlated LLM assessors and protocol revisions. "
            "Not an unbiased reliability estimate, calibration score or evidence of examiner accuracy."),
        "limits": report["limits"] + [
            "No literal sentence reuse does not establish diversity of voice, argument or structure.",
            "Artificial errors and deliberately extreme stress cases are not a sample of real learners.",
            "Criterion labels depend on LLM rubric interpretation, especially on wholly unrelated answers.",
        ],
    }
    save(ROOT / "work/quality_summary.json", result)
    print(json.dumps({"path": str(ROOT / "work/quality_summary.json"), "status": result["status"],
                      "rows": len(candidates), "accepted": len(accepted),
                      "reused_long_sentences": len(recurring)}))


if __name__ == "__main__":
    main()
