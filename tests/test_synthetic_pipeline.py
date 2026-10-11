"""The unattended continuation must never publish an unvalidated candidate."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_runner(tmp_path, monkeypatch, result, gate=None):
    spec = importlib.util.spec_from_file_location("synthetic_pipeline", ROOT / "scripts/run_synthetic_pipeline.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    (tmp_path / "configs").mkdir()
    (tmp_path / "reports").mkdir()
    run = tmp_path / "run"
    run.mkdir()
    config = {"parent_repo": runner.MODEL, "parent_revision": "frozen-parent"}
    for path, content in [(tmp_path / "configs/synthetic_moe.json", config),
                           (run / "experiment_config.json", config), (run / "result.json", result)]:
        path.write_text(json.dumps(content))
    if gate is not None:
        (run / "release_gate.json").write_text(json.dumps(gate))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "RUN", run)
    monkeypatch.setattr(runner, "STATUS", tmp_path / "reports/status.json")
    return runner


def test_failed_private_gate_prevents_any_hub_write(tmp_path, monkeypatch):
    runner = load_runner(tmp_path, monkeypatch, {"status": "evaluated"},
                         {"passed": False, "public": {"checks": {"synthetic_gain": True}},
                          "private_sanity_passed": False})
    calls = []

    def command(args, required=True):
        calls.append(args)
        return "KernelWorkerStatus.COMPLETE" if args[:3] == ["kaggle", "kernels", "status"] else ""

    monkeypatch.setattr(runner, "command", command)
    runner.main(0)
    assert not any(args[0] == "hf" for args in calls)
    assert json.loads(runner.STATUS.read_text())["phase"] == "completed_candidate_failed_promotion"


def test_changed_public_parent_prevents_overwrite(tmp_path, monkeypatch):
    runner = load_runner(tmp_path, monkeypatch, {"status": "evaluated"}, {"passed": True})
    calls = []

    def command(args, required=True):
        calls.append(args)
        if args[:3] == ["kaggle", "kernels", "status"]:
            return "KernelWorkerStatus.COMPLETE"
        if args[:3] == ["hf", "models", "info"]:
            return json.dumps({"sha": "some-new-revision"})
        return ""

    monkeypatch.setattr(runner, "command", command)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        runner.main(0)
    assert not any(args[:2] == ["hf", "upload"] or args[:3] == ["hf", "repos", "tag"] for args in calls)
