"""Checks for provenance, leakage and stale judgments in synthetic data preparation."""
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "synthetic_dataset", Path(__file__).resolve().parents[1] / "scripts/synthetic_dataset.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def example():
    return {
        "candidate_id": "a01-10", "family_id": "a01", "task": "task2",
        "question": "Some people want city libraries open on Sundays. Do you agree or disagree with this proposal?",
        "essay": "I agree because working people need books too.", "topic": "library hours",
        "question_type": "opinion", "generation": {
            "author_agent": "/root/writer", "origin": "llm_synthetic", "slot": 10,
            "intent": "short attempt", "method": "individually_authored", "human_rating": False,
        },
    }


def rating(packet):
    return {
        "id": packet["id"], "input_sha256": packet["input_sha256"],
        "judge_id": "/root/judge", "label_origin": "llm_judgment", "assessable": True,
        "scores": dict.fromkeys(module.CRITERIA, 1), "confidence": "medium",
        "evidence": {c: {"quote": "working people", "reason": "This very short response gives insufficient evidence."}
                     for c in module.CRITERIA}, "flags": ["underlength"], "concerns": "",
    }


def setup_packet(tmp_path):
    row = example()
    module.save(tmp_path / "candidates/a/a01-10.json", row)
    module.save(tmp_path / "work/ready-a01.json", {
        "ready": True, "family_id": "a01", "author_agent": "/root/writer"})
    module.packetize(tmp_path)
    packet = module.load(next((tmp_path / "work/packets").glob("*.json")))
    return row, packet


def test_blind_payload_and_sealed_inputs(tmp_path):
    row, packet = setup_packet(tmp_path)
    assert set(packet) == {"id", "input_sha256", "task", "question", "essay"}
    assert not module.candidate_errors(row)
    row["essay"] += " Please."
    module.save(tmp_path / "candidates/a/a01-10.json", row)
    import pytest
    with pytest.raises(ValueError, match="Sealed input changed"):
        module.packetize(tmp_path)


def test_scores_evidence_and_hash_are_checked(tmp_path):
    _, packet = setup_packet(tmp_path)
    judge = rating(packet)
    assert not module.judge_errors(judge, packet)
    judge["scores"]["TR"] = 5
    assert any("<=20 words" in e for e in module.judge_errors(judge, packet))
    judge["scores"]["TR"] = 1
    judge["evidence"]["LR"]["quote"] = "invented quotation"
    assert any("exact essay substring" in e for e in module.judge_errors(judge, packet))
    judge["input_sha256"] = "stale"
    assert any("current input hash" in e for e in module.judge_errors(judge, packet))


def test_no_self_judging_and_no_unjudged_release(tmp_path):
    row, packet = setup_packet(tmp_path)
    report, _ = module.audit(tmp_path, expected=1)
    assert not report["release_ready"]
    assert report["pending_judgment"] == [row["candidate_id"]]
    judge = rating(packet)
    judge["judge_id"] = row["generation"]["author_agent"]
    module.save(tmp_path / "jury/judge/rating.json", judge)
    report, _ = module.audit(tmp_path, expected=1)
    assert not report["release_ready"]
    assert any("self-judging" in str(e) for e in report["structural_errors"])


def test_family_split_and_input_export(tmp_path):
    _, packet = setup_packet(tmp_path)
    module.save(tmp_path / "jury/judge/rating.json", rating(packet))
    splits = module.family_splits()
    assert len(splits) == 120
    assert list(splits.values()).count("train") == 96
    module.release(tmp_path, expected=1)
    import json
    rows = [json.loads(line) for path in (tmp_path / "release").glob("*_inputs.jsonl")
            for line in path.read_text().splitlines()]
    assert len(rows) == 1
    assert set(rows[0]) == {"id", "task", "question", "essay"}


def test_adjudication_preserves_original_ratings_and_requires_current_hashes(tmp_path):
    _, packet = setup_packet(tmp_path)
    first = rating(packet)
    first["confidence"] = "low"
    second = rating(packet)
    second["judge_id"] = "/root/judge2"
    module.save(tmp_path / "jury/one/one.json", first)
    module.save(tmp_path / "jury/two/two.json", second)
    report, _ = module.audit(tmp_path, expected=1)
    assert len(report["held_for_review"]) == 1
    decision = rating(packet)
    decision.update(judge_id="/root/adjudicator",
                    resolution="The explicit public short-response rule resolves the confidence concern.",
                    reviewed_judgment_hashes=sorted(module.digest(r) for r in (first, second)))
    path = tmp_path / "adjudications" / f"{packet['id']}.json"
    module.save(path, decision)
    report, accepted = module.audit(tmp_path, expected=1)
    assert report["release_ready"]
    assert len(accepted[0]["ratings"]) == 2
    assert accepted[0]["adjudication"]["resolution"] == decision["resolution"]
    second["concerns"] = "A new concern after adjudication."
    module.save(tmp_path / "jury/two/two.json", second)
    report, _ = module.audit(tmp_path, expected=1)
    assert not report["release_ready"]


def test_explicit_rubric_reassessment_blocks_old_adjudication(tmp_path):
    _, packet = setup_packet(tmp_path)
    first = rating(packet)
    module.save(tmp_path / "jury/one/one.json", first)
    module.save(tmp_path / "work/adjudication_requirements.json", {
        packet["id"]: {"protocol": "JUDGE-v3.md", "reason": "Rubric reassessment required"}})
    decision = rating(packet)
    decision.update(judge_id="/root/adjudicator",
                    resolution="The short-response rule is consistent with the updated rubric instructions.",
                    reviewed_judgment_hashes=[module.digest(first)])
    path = tmp_path / "adjudications" / f"{packet['id']}.json"
    module.save(path, decision)
    report, _ = module.audit(tmp_path, expected=1)
    assert not report["release_ready"]
    assert report["held_for_review"][0]["reason"] == "Rubric reassessment required"
    decision["rubric_protocol"] = "JUDGE-v3.md"
    module.save(path, decision)
    report, _ = module.audit(tmp_path, expected=1)
    assert report["release_ready"]


def test_quarantine_only_exports_fully_adjudicated_uncertainty(tmp_path):
    import pytest
    _, packet = setup_packet(tmp_path)
    with pytest.raises(ValueError, match="Release blocked"):
        module.release(tmp_path, 1, allow_quarantine=True)
    first = rating(packet)
    first["confidence"] = "low"
    module.save(tmp_path / "jury/one" / f"{packet['id']}.json", first)
    with pytest.raises(ValueError, match="Release blocked"):
        module.release(tmp_path, 1, allow_quarantine=True)
    decision = rating(packet)
    decision.update(judge_id="/root/adjudicator", confidence="low",
                    resolution="The linguistic evidence remains uncertain after separate adjudication.",
                    reviewed_judgment_hashes=[module.digest(first)])
    module.save(tmp_path / "adjudications" / f"{packet['id']}.json", decision)
    result = module.release(tmp_path, 1, allow_quarantine=True)
    assert result["rows"] == 0 and result["quarantined_rows"] == 1
    import json
    held = json.loads((tmp_path / "release/quarantine.jsonl").read_text())
    assert all(score is None for score in held["scores"].values())
    assert held["ratings"] == [first]
    assert not any(p.read_text() for p in (tmp_path / "release").glob("*_inputs.jsonl"))
