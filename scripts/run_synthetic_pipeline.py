"""Wait for Kaggle capacity, train, validate locally, and conditionally publish the MoE update.

No existing Kaggle job is stopped. Only the authorized MoE repository can be updated.
All transitions are recorded in reports/synthetic_pipeline_status.json.
"""

import argparse
import fcntl
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT / ".venv/bin/python")
KERNEL = "saugataroyarghya/essay-scoring-moe-synthetic-v1"
MODEL = "saugataroy/ielts-task2-modernbert-base-moe"
RUN = ROOT / "artifacts/kaggle-output/synthetic-moe-v1"
PACKAGE = ROOT / "artifacts/staging/huggingface-synthetic-moe"
STATUS = ROOT / "reports/synthetic_pipeline_status.json"


def read(path):
    return json.loads(path.read_text())


def state(phase, **values):
    previous = read(STATUS) if STATUS.exists() else {}
    previous.update(phase=phase, updated_at=datetime.now(UTC).isoformat(), **values)
    STATUS.write_text(json.dumps(previous, indent=2) + "\n")
    print(json.dumps({"phase": phase, **values}), flush=True)


def command(args, required=True):
    result = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if required and result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {args[0]} {args[1]}\n{result.stdout[-2500:]}")
    return result.stdout


def main(wait_hours):
    config = read(ROOT / "configs/synthetic_moe.json")
    if config["parent_repo"] != MODEL:
        raise ValueError("Unexpected model repository.")
    existing = command(["kaggle", "kernels", "status", KERNEL], required=False)
    if "RUNNING" in existing or "QUEUED" in existing:
        state("training", kernel=KERNEL, resumed_existing_run=True)
    elif "COMPLETE" in existing:
        state("retrieving_completed_run", kernel=KERNEL)
    elif "ERROR" in existing or "CANCEL" in existing:
        raise RuntimeError("The previous run failed or was canceled; inspect it before restarting.")
    else:
        deadline = time.monotonic() + wait_hours * 3600
        while True:
            message = command(["kaggle", "kernels", "push", "-p", "notebooks/essay-scoring-moe-synthetic-v1",
                               "--accelerator", "NvidiaTeslaT4", "--timeout", "10800"], required=False)
            if "Maximum batch GPU session count" in message:
                state("waiting_for_gpu_slot", kernel=KERNEL,
                      reason="Both Kaggle batch GPU slots are occupied; existing jobs left running.")
                if time.monotonic() >= deadline:
                    state("capacity_wait_expired", training_started=False)
                    return
                # Each individual sleep is short so termination remains responsive.
                for _ in range(6):
                    time.sleep(30)
                continue
            if "successfully pushed" not in message.lower():
                raise RuntimeError("Kaggle did not confirm launch: " + message[-2000:])
            state("training", kernel=KERNEL, training_started=True, launch_response=message.strip())
            break
    deadline = time.monotonic() + 4 * 3600
    while True:
        message = command(["kaggle", "kernels", "status", KERNEL], required=False)
        if "COMPLETE" in message:
            break
        if "ERROR" in message or "CANCEL" in message:
            raise RuntimeError("Kaggle run did not complete successfully: " + message)
        if time.monotonic() > deadline:
            state("training_monitor_timeout", message=message.strip())
            return
        for _ in range(2):
            time.sleep(30)
    state("downloading_results")
    command(["kaggle", "kernels", "output", KERNEL, "-p", str(RUN), "--force", "--quiet"])
    result = read(RUN / "result.json")
    if read(RUN / "experiment_config.json") != config:
        raise RuntimeError("Completed run does not match the frozen local experiment configuration.")
    if result["status"] != "evaluated":
        state("completed_without_promotable_checkpoint", reason=result["status"], model_updated=False)
        return
    state("validating_candidate")
    command([PYTHON, "scripts/validate_synthetic_candidate.py", "--run", str(RUN), "--package", str(PACKAGE)])
    gate = read(RUN / "release_gate.json")
    if not gate["passed"]:
        state("completed_candidate_failed_promotion", model_updated=False,
              public_checks=gate["public"]["checks"], private_sanity_passed=gate["private_sanity_passed"])
        return
    command([PYTHON, "scripts/package_synthetic_model.py", "--run", str(RUN), "--package", str(PACKAGE)])
    info = json.loads(command(["hf", "models", "info", MODEL, "--format", "json"]))
    if info["sha"] != config["parent_revision"]:
        raise RuntimeError("Public model changed while training; refusing to overwrite a newer revision.")
    state("publishing_validated_checkpoint")
    tag = "pre-synthetic-v1"
    tag_message = command(["hf", "repos", "tag", "create", MODEL, tag,
                           "--revision", config["parent_revision"]], required=False)
    if "error" in tag_message.lower() and "already exist" not in tag_message.lower():
        raise RuntimeError("Could not preserve parent tag: " + tag_message[-1500:])
    message = command(["hf", "upload", MODEL, str(PACKAGE), ".", "--type", "model",
                       "--delete", "benchmark.json", "--commit-message",
                       "Update validated MoE checkpoint with synthetic augmentation and evaluation"])
    state("verifying_public_checkpoint", upload_response=message.strip())
    info = json.loads(command(["hf", "models", "info", MODEL, "--format", "json"]))
    verification = ROOT / "artifacts/staging/synthetic-model-public-verification"
    command(["hf", "download", MODEL, "--revision", info["sha"], "--local-dir", str(verification)])
    for name, expected in read(PACKAGE / "checksums.json").items():
        with (verification / name).open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                raise RuntimeError("Published file differs: " + name)
    state("complete", model_updated=True, model_revision=info["sha"], public=not info["private"],
          weights_sha256=gate["weights_sha256"], selected_epoch=result["selected_epoch"],
          evaluation={split: {"baseline_macro_mae": result["baseline"][split]["macro_mae"],
                              "candidate_macro_mae": row["macro_mae"]}
                      for split, row in result["candidate"].items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-hours", type=float, default=6)
    args = parser.parse_args()
    lock = (ROOT / "reports/synthetic_pipeline.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        main(args.wait_hours)
    except Exception as exc:
        state("failed", error=str(exc))
        raise
