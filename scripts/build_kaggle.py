"""Build private Kaggle notebooks with a reviewable, self-contained code snapshot."""

import argparse
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER = "saugataroyarghya"


def cell(kind, source):
    item = {"cell_type": kind, "id": hashlib.sha256(source.encode()).hexdigest()[:12],
            "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


def build(mode, track="ellipse", head="mlp"):
    files = sorted([*ROOT.glob("src/**/*.py"), *ROOT.glob("configs/*.json"), ROOT / "pyproject.toml"])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            # Fixed ZIP timestamps make the source bundle hash reproducible.
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), (2026, 10, 9, 0, 0, 0))
            archive.writestr(info, path.read_bytes())
    payload = base64.b64encode(buf.getvalue()).decode()
    digest = hashlib.sha256(buf.getvalue()).hexdigest()
    if mode == "prepare":
        slug, title = "essay-scoring-data-preparation", "Essay Scoring Data Preparation"
    elif mode in {"smoke", "evaluate"}:
        slug = f"essay-scoring-{track}-smoke"
        if mode == "evaluate":
            slug = f"essay-scoring-{track}-evaluate"
        title = f"Essay Scoring {track.title()} {mode.title()}"
    else:
        slug = f"essay-scoring-{track}-{mode}-{head}"
        title = f"Essay Scoring {track.title()} {mode.title()} {head.upper()}"
        if head == "both":
            slug = f"essay-scoring-{track}-train"
            title = f"Essay Scoring {track.title()} Train"
    notebook_dir = ROOT / "notebooks" / slug
    notebook_dir.mkdir(parents=True, exist_ok=True)
    bootstrap = f'''import base64, hashlib, io, json, os, pathlib, subprocess, sys, zipfile
project = pathlib.Path("/kaggle/temp/essay-scoring-code")
project.mkdir(parents=True, exist_ok=True)
payload = base64.b64decode({payload!r})
assert hashlib.sha256(payload).hexdigest() == {digest!r}
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    archive.extractall(project)
os.environ["PYTHONPATH"] = str(project / "src")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HOME"] = "/kaggle/temp/huggingface"
os.environ["WANDB_DISABLED"] = "true"
os.chdir(project)
print("Source bundle SHA256:", {digest!r})
pathlib.Path("/kaggle/working/source_bundle_sha256.txt").write_text({digest!r})
'''
    if mode == "prepare":
        run = '''subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyzipper==0.3.6"], check=True)
subprocess.run([sys.executable, "-m", "ielts_scoring.data", "--output", "/kaggle/temp/essay-data"], check=True)
import shutil, importlib.metadata
for track_name in ["ellipse", "ielts"]:
    source = pathlib.Path("/kaggle/temp/essay-data/prepared") / track_name
    shutil.copytree(source, pathlib.Path("/kaggle/working") / track_name, dirs_exist_ok=True)
packages = {n: importlib.metadata.version(n) for n in ["pandas", "numpy", "scikit-learn", "pyarrow", "requests", "pyzipper"]}
pathlib.Path("/kaggle/working/preparation_environment.json").write_text(json.dumps(packages, indent=2))
print("Prepared datasets saved. No GPU was used.")
'''
    else:
        heads = ["mlp", "moe"] if mode == "smoke" or head == "both" else [head]
        run = '''dependencies = "/kaggle/temp/essay-scoring-dependencies"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", dependencies, "transformers==4.57.6", "huggingface-hub==0.36.0", "tokenizers==0.22.2", "safetensors==0.7.0"], check=True)
os.environ["PYTHONPATH"] = dependencies + os.pathsep + os.environ["PYTHONPATH"]
'''
        if mode == "evaluate":
            run += f'''subprocess.run([sys.executable, "-m", "ielts_scoring.finalize", "--track", {track!r}], check=True)
'''
        elif mode == "train" and head == "both":
            run += f'''import torch
gpu_count = torch.cuda.device_count()
if gpu_count == 0:
    raise RuntimeError("Kaggle did not allocate a GPU.")
processes = []
for index, head_name in enumerate({heads!r}):
    command = [sys.executable, "-m", "ielts_scoring.train", "--track", {track!r}, "--head", head_name, "--mode", "train", "--data-root", "/kaggle/input", "--output", "/kaggle/working/" + head_name, "--source-hash", {digest!r}]
    worker_env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(index % gpu_count))
    if gpu_count > 1:
        processes.append((head_name, subprocess.Popen(command, env=worker_env)))
    else:
        subprocess.run(command, env=worker_env, check=True)
failures = [(name, process.wait()) for name, process in processes]
if any(code != 0 for _, code in failures):
    raise RuntimeError("A training run failed: " + repr(failures))
subprocess.run([sys.executable, "-m", "ielts_scoring.finalize", "--track", {track!r}, "--runs", "/kaggle/working"], check=True)
'''
        else:
            run += f'''for head_name in {heads!r}:
    subprocess.run([sys.executable, "-m", "ielts_scoring.train", "--track", {track!r}, "--head", head_name, "--mode", {mode!r}, "--data-root", "/kaggle/input", "--output", "/kaggle/working/" + head_name, "--source-hash", {digest!r}], check=True)
'''
    notebook = {
        "cells": [cell("markdown", f"# {title}\n\nPrivate research run. Source snapshot: `{digest}`.\n"
                      "ELLIPSE and IELTS labels are separate. Website labels do not establish examiner agreement.\n"),
                  cell("code", bootstrap), cell("code", run)],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    (notebook_dir / "run.ipynb").write_text(json.dumps(notebook, indent=2) + "\n")
    metadata = {
        "id": f"{OWNER}/{slug}", "title": title, "code_file": "run.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": mode != "prepare", "enable_internet": True,
        "dataset_sources": [] if mode == "prepare" else [f"{OWNER}/{track}-writing-scoring-private"],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }
    if mode == "evaluate":
        metadata["kernel_sources"] = [f"{OWNER}/essay-scoring-{track}-train"]
    if mode != "prepare":
        metadata["machine_shape"] = "NvidiaTeslaT4"
    (notebook_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(notebook_dir.relative_to(ROOT), digest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "smoke", "train", "evaluate"])
    parser.add_argument("--track", default="ellipse", choices=["ellipse", "ielts"])
    parser.add_argument("--head", default="mlp", choices=["mlp", "moe", "both"])
    args = parser.parse_args()
    build(args.mode, args.track, args.head)
