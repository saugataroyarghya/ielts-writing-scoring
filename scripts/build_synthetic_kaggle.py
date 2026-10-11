"""Create the private, pinned synthetic-augmentation experiment notebook."""

import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "notebooks/essay-scoring-moe-synthetic-v1"
DEST.mkdir(parents=True, exist_ok=True)
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted([*ROOT.glob("src/**/*.py"), *ROOT.glob("configs/*.json")]):
        info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), (2026, 10, 10, 0, 0, 0))
        archive.writestr(info, path.read_bytes())
payload = base64.b64encode(buffer.getvalue()).decode()
digest = hashlib.sha256(buffer.getvalue()).hexdigest()
nb = nbformat.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.cells = [
    nbformat.v4.new_markdown_cell("""# MoE synthetic augmentation experiment

Continue the published ModernBERT MoE checkpoint with 976 synthetic training responses
and 1,952 fixed original-training replay examples. The original labels are from Writing9;
the synthetic labels are LLM judgments. Neither is examiner ground truth.

## Method

Train all encoder and head parameters for at most five epochs, using separate learning
rates of 5e-6 and 5e-5. Select using original and synthetic validation only, with an
original-data regression guard. Freeze selection before the final comparison.
Both model and dataset downloads use immutable Hugging Face revisions. The attached
private original dataset must match its pinned checksums. No public upload occurs here.
"""),
    nbformat.v4.new_markdown_cell("## Restore the reviewed code and pinned dependencies"),
    nbformat.v4.new_code_cell(f'''import base64, hashlib, io, json, os, pathlib, subprocess, sys, zipfile
project = pathlib.Path("/kaggle/temp/synthetic-moe-code")
project.mkdir(parents=True, exist_ok=True)
payload = base64.b64decode({payload!r})
assert hashlib.sha256(payload).hexdigest() == {digest!r}
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    archive.extractall(project)
os.chdir(project)
dependencies = "/kaggle/temp/synthetic-moe-dependencies"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", dependencies,
                "transformers==4.57.6", "huggingface-hub==0.36.0", "tokenizers==0.22.2", "safetensors==0.7.0"], check=True)
os.environ.update(PYTHONPATH=dependencies + os.pathsep + str(project / "src"),
                  HF_HOME="/kaggle/temp/huggingface", HF_HUB_DISABLE_TELEMETRY="1",
                  TOKENIZERS_PARALLELISM="false", WANDB_DISABLED="true", CUDA_VISIBLE_DEVICES="0")
pathlib.Path("/kaggle/working/source_bundle_sha256.txt").write_text({digest!r})
print("Code SHA256:", {digest!r})
print(json.dumps(json.loads((project / "configs/synthetic_moe.json").read_text()), indent=2))
'''),
    nbformat.v4.new_markdown_cell("## Train and evaluate\n\nDetailed per-epoch logs and prediction tables are saved as private outputs."),
    nbformat.v4.new_code_cell('''command = [sys.executable, "-m", "ielts_scoring.synthetic_finetune",
           "--config", "configs/synthetic_moe.json", "--data-root", "/kaggle/input",
           "--output", "/kaggle/working"]
with open("/kaggle/working/training.log", "w") as log:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        log.write(line)
        log.flush()
        print(line, end="", flush=True)
    if process.wait() != 0:
        raise RuntimeError("Training failed. See training.log.")
'''),
    nbformat.v4.new_markdown_cell("## Results\n\nLower macro MAE is better. These are label-source-specific comparisons, not examiner accuracy."),
    nbformat.v4.new_code_cell('''import pandas as pd
result = json.loads(pathlib.Path("/kaggle/working/result.json").read_text())
if result["status"] == "evaluated":
    display(pd.DataFrame([
        {"Evaluation": name, "Parent macro MAE": result["baseline"][name]["macro_mae"],
         "Candidate macro MAE": values["macro_mae"],
         "Change": values["macro_mae"] - result["baseline"][name]["macro_mae"]}
        for name, values in result["candidate"].items()
    ]).round(4))
print(json.dumps(result["promotion"], indent=2))
'''),
]
nbformat.validate(nb)
nbformat.write(nb, DEST / "run.ipynb")
metadata = {"id": "saugataroyarghya/essay-scoring-moe-synthetic-v1",
            "title": "Essay Scoring MoE Synthetic v1", "code_file": "run.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": ["saugataroyarghya/ielts-writing-scoring-private"],
            "competition_sources": [], "kernel_sources": [], "model_sources": [],
            "machine_shape": "NvidiaTeslaT4"}
(DEST / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
for cell in nb.cells:
    if cell.cell_type == "code":
        compile(cell.source, "notebook-cell", "exec")
print(json.dumps({"notebook": str(DEST), "source_sha256": digest, "format_and_syntax": "passed"}))
