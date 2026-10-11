"""Write a complete MoE model card only after the local promotion gates pass."""

import argparse
import json
import shutil
from pathlib import Path

import yaml
from huggingface_hub import ModelCard

from ielts_scoring.synthetic_finetune import TARGETS, read, sha, write

ROOT = Path(__file__).resolve().parents[1]
REPO = "saugataroy/ielts-task2-modernbert-base-moe"


def table(baseline, candidate):
    lines = ["| Criterion | Parent MAE | Updated MAE | Updated RMSE | Updated bias |",
             "|---|---:|---:|---:|---:|"]
    for key in TARGETS:
        b, c = baseline[key], candidate[key]
        lines.append(f"| {key} | {b['mae']:.4f} | {c['mae']:.4f} | {c['rmse']:.4f} | {c['bias']:+.4f} |")
    lines.append(f"| Macro | {baseline['macro_mae']:.4f} | **{candidate['macro_mae']:.4f}** | "
                 f"{candidate['macro_rmse']:.4f} | — |")
    return "\n".join(lines)


def main(run, package):
    gate, result, config = [read(run / name) for name in
                            ["release_gate.json", "result.json", "experiment_config.json"]]
    assert gate["passed"] and gate["private_sanity_passed"] and gate["portable_loader_passed"]
    assert gate["public"]["public_evaluation_passed"]
    assert sha(package / "model.safetensors") == gate["weights_sha256"] == result["weights_sha256"]
    assert read(package / "inference_verification.json")["passed"]
    metadata = {"language": ["en"], "library_name": "transformers",
                "base_model": REPO, "base_model_relation": "finetune",
                "license": "other", "license_name": "licensing-status-notice", "license_link": "LICENSE.md",
                "datasets": ["chillies/ielts-writing-task2-essays", config["dataset_repo"]],
                "tags": ["modernbert", "ielts", "essay-scoring", "regression", "custom-code",
                         "mixture-of-experts", "synthetic-data"]}
    # A repository cannot use itself as its unversioned parent in discovery metadata.
    metadata["base_model"] = "answerdotai/ModernBERT-base"
    sections = []
    names = {
        "original_validation": ("Original validation", "529 responses; 23 prompt groups; Writing9 labels. Used for selection."),
        "synthetic_validation": ("Synthetic validation", "98 responses; LLM labels. Used for selection."),
        "original_test": ("Original comparison holdout", "450 responses; 23 prompt groups; Writing9 labels. Previously evaluated for the parent model."),
        "synthetic_diagnostic": ("Synthetic diagnostic holdout", "120 responses; LLM labels. Excluded from training and epoch selection."),
    }
    for key, (title, note) in names.items():
        d = result["paired_differences"][key]
        lower, upper = d["paired_prompt_group_95ci"]
        sections.append(f"### {title}\n\n{note}\n\n" + table(result["baseline"][key], result["candidate"][key]) +
                        f"\n\nChange in macro MAE (updated − parent): **{d['macro_mae_difference_second_minus_first']:+.4f}**; "
                        f"paired prompt-group bootstrap 95% interval **[{lower:+.4f}, {upper:+.4f}]** "
                        f"({d['groups']} groups; {d['draws']:,} resamples).\n")
    evaluation_text = "\n".join(sections)
    epochs = len(read(run / "history.json"))
    card = f"""# IELTS Task 2 ModernBERT-base — Multi-gate MoE

An English essay-scoring research model that estimates **Task Response (TR), Coherence
and Cohesion (CC), Lexical Resource (LR), and Grammatical Range and Accuracy (GRA)**
from an IELTS-style Task 2 question and typed response.

This checkpoint continues the ELLIPSE-to-IELTS MoE model with synthetic augmentation.
Its labels come from Writing9 and LLM judgments. **Reliable agreement with qualified
IELTS examiners has not been established.** It is not an official IELTS grading service.

## Model details

| Property | Value |
|---|---|
| Publisher | [saugataroy](https://huggingface.co/saugataroy) |
| Architecture | ModernBERT-base encoder, essay-token mean pooling, dense multi-gate MoE head |
| Parameters | 150,078,228 |
| Language and scope | English; Writing Task 2 only |
| Input limit | 1,024 tokens including question, response, and special tokens |
| Weight format | Full FP32 Safetensors; encoder and head |
| Parent checkpoint | [`{config['parent_revision']}`](https://huggingface.co/{REPO}/tree/{config['parent_revision']}) |
| Synthetic dataset revision | [`{config['dataset_revision']}`](https://huggingface.co/datasets/{config['dataset_repo']}/tree/{config['dataset_revision']}) |
| Related plain-head model | [saugataroy/ielts-task2-modernbert-base-mlp](https://huggingface.co/saugataroy/ielts-task2-modernbert-base-mlp) |

## Uses and limitations

Intended for rubric-regression research, writing-practice prototypes, and controlled
studies of synthetic augmentation. Human review and independent examiner-rated
evaluation are needed before relying on its estimates for consequential decisions.
Do not use it to determine official examination bands, admissions, or employment.

Task 1 Academic, General Training letters, handwriting, speaking, and a complete IELTS
Writing band are unsupported. The model supplies estimates, not generated feedback,
sentence-level corrections, or calibrated confidence. No subgroup fairness evaluation
or representative real-world examiner study has been completed.

Synthetic responses can introduce artificial errors, model-specific style, and rubric
bias. Their GRA labels are strongly skewed high: 533 of 1,194 accepted rows are band 9,
while only three are band 3. Synthetic holdout gains measure agreement with LLM labels.
Original-corpus results measure agreement with Writing9, whose scoring reliability is
not independently established. Neither result should be described as official IELTS accuracy.

## Inputs and outputs

Provide the original question and a nonempty essay. Preserve spelling, grammar, and
paragraphs. The paired tokenizer preserves the question and truncates only the essay
when necessary; overly long questions can fail tokenization.

`model.score` returns one dictionary per input pair:

- `scores`: continuous TR, CC, LR, and GRA estimates on the training labels' 0–9 scale.
- `task2_mean_estimate`: their unrounded arithmetic mean.
- `truncated`: whether the full input exceeded 1,024 tokens.
- `track`: `ielts`; `max_tokens`: `1024`.

Outputs are not clipped or rounded; values outside the label range are possible.
The mean is not an official rounded Task 2 band or the overall Writing score.

## Usage

Use Python 3.11–3.13 and a PyTorch build appropriate for your device:

```bash
pip install 'torch>=2.6,<2.12' 'transformers==4.57.6' 'huggingface-hub==0.36.0' 'safetensors>=0.6,<0.8'
```

```python
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

model_id = "{REPO}"
revision = "main"  # Pin the repository commit SHA for reproducible inference.
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
model = AutoModel.from_pretrained(
    model_id, revision=revision, trust_remote_code=True,
).eval().to("cpu")  # Use "cuda" for NVIDIA or "mps" for a supported Apple GPU.
result = model.score(
    tokenizer,
    questions=["Should cities make public transport free? Discuss both views."],
    essays=[Path("essay.txt").read_text(encoding="utf-8")],
)[0]
print(result)
```

Review and pin [modeling_ielts.py](modeling_ielts.py), which `trust_remote_code=True`
loads. Batch inference accepts lists of questions and essays. A generic text-classification
pipeline does not create the required essay-token mask; use `score` or reproduce the
paired preprocessing. Low-level `forward` takes `input_ids`, `attention_mask`, and
`essay_mask`, and returns `logits` shaped `[batch, 4]` containing regression estimates.

## Architecture

```text
Question + essay → paired tokenizer → 22-layer ModernBERT-base
  → mean pooling over contextualized essay tokens
  → four shared experts + four criterion-specific softmax gates
  → criterion-specific linear outputs + fixed 4.5 offset
  → TR, CC, LR, GRA
```

The encoder uses hidden size 768 and 12 attention heads per layer. Each expert is
`Linear(768,256) → GELU → Dropout(0.1) → Linear(256,256) → GELU`. Each criterion
has a `Linear(768,4)` softmax gate and a `Linear(256,1)` output. **All four experts
execute for every essay.** This is a dense soft mixture, not sparse top-k MoE routing.
The output scores themselves use no softmax or sigmoid. Dropout is disabled in evaluation.

## Training data and procedure

The parent encoder was first supervised on [ELLIPSE](https://github.com/scrosseye/ELLIPSE-Corpus)
(3,324 training and 587 validation responses, six human-rated 1–5 proficiency traits).
A fresh IELTS head and the encoder were then trained on
[chillies/ielts-writing-task2-essays](https://huggingface.co/datasets/chillies/ielts-writing-task2-essays)
(7,069 training, 529 validation, 450 test; four Writing9 criteria). ELLIPSE traits
were not mapped to IELTS bands. Original splits group related prompts and lexical
near-duplicate essays; they cannot guarantee semantic topic isolation.

This continuation initializes **all existing encoder and MoE head weights** from the
pinned parent, without resetting the head. Each epoch uses **976 synthetic training
responses plus 1,952 fixed original-training replay responses** sampled with seed 42.
Replay examples come only from the original training split. Synthetic validation,
diagnostic, and quarantine rows are excluded from training. Only question and essay
text enter the model; rationales, flags, metadata, and scores are never input text.

Training uses masked mean squared error, AdamW, encoder/head learning rates
`5e-6`/`5e-5`, weight decay 0.01 on matrix parameters, gradient clipping at 1.0,
10% linear warm-up, and an effective batch size of 16 (microbatch 2 × accumulation 8).
All encoder and head parameters are updated. The run uses seed 42, gradient checkpointing,
and CUDA mixed precision. Full environment details are in [environment.json](environment.json).

The run completed {epochs} epochs within a five-epoch cap; the released checkpoint is
**continuation epoch {result['selected_epoch']}**. Selection minimizes the equal-weight
mean of original and synthetic validation macro MAE, subject to an original-validation
guard of at most +0.10 macro MAE and +0.20 MAE for any criterion relative to the parent.
Early stopping uses two epochs without an eligible gain of at least 0.01.
The parent remains eligible as epoch zero. Final diagnostic and test labels are not
used for epoch selection. See [training_config.json](training_config.json).

## Evaluation

All errors below use raw, unrounded predictions in band units. Lower MAE and RMSE are
better; bias is prediction minus reference. Macro MAE averages the four criterion MAEs.
The parent and updated model were evaluated on identical rows with identical preprocessing.

{evaluation_text}

Intervals resample prompt groups; they do not capture variability across training
seeds or the uncertainty of the labels. Validation intervals are exploratory because
those splits selected the checkpoint. The original comparison holdout has already
been reported for the parent, so it is not a newly unseen evaluation set.

Public release requires at least 0.15 lower synthetic-diagnostic macro MAE, a paired
group interval entirely below zero, and no original-validation or original-test
regression beyond +0.10 macro MAE or +0.20 for any criterion. These thresholds were
fixed before training. Passing them does not establish reliable examiner grading.
Machine-readable results are in [evaluation.json](evaluation.json).

## Technical validation and performance

The exported Transformers package was strictly loaded offline, compared tensor by
tensor with the selected training checkpoint, and checked for prediction agreement,
truncation reporting, and blank-input rejection. See
[inference_verification.json](inference_verification.json).

Production throughput, p95 HTTP latency, and concurrent-serving memory have not been
benchmarked for this updated checkpoint. Earlier checkpoint timings should not be
presented as measurements of this release. Architecture-level MoE naming does not
imply an inference-speed advantage over the plain head.

## Licensing and provenance

See [LICENSE.md](LICENSE.md) for the checkpoint's licensing-status notice. The base
encoder is Apache 2.0; the ELLIPSE corpus declares CC BY-NC-SA 4.0; the IELTS dataset
uploader declares CC BY 4.0; the synthetic dataset is CC BY 4.0. These dataset terms do
not automatically determine the legal status of trained weights. Public download
availability does not establish unrestricted commercial-use rights.

The model is not affiliated with or endorsed by IELTS, Cambridge, British Council, or
IDP. Pin the repository commit and report the model and dataset revisions when citing
results. [provenance.json](provenance.json) records the parent, selected epoch, hashes,
and source snapshot. Report issues through this repository's Community tab.
"""
    (package / "README.md").write_text("---\n" + yaml.safe_dump(metadata, sort_keys=False) + "---\n\n" + card)
    ModelCard.load(package / "README.md")
    # Explicit allowlist: no private sanity values, original essays, or per-example predictions are published.
    evaluation = {key: result[key] for key in ["baseline", "candidate", "paired_differences", "selected_epoch"]}
    evaluation["label_sources"] = {"original": "Writing9", "synthetic": "LLM judgments"}
    write(package / "evaluation.json", evaluation)
    write(package / "training_config.json", config)
    shutil.copy2(run / "environment.json", package / "environment.json")
    write(package / "provenance.json", {"repo_id": REPO, "parent_revision": config["parent_revision"],
                                        "parent_weights_sha256": config["parent_weights_sha256"],
                                        "dataset_repo": config["dataset_repo"],
                                        "dataset_revision": config["dataset_revision"],
                                        "weights_sha256": result["weights_sha256"],
                                        "selected_epoch": result["selected_epoch"],
                                        "source_bundle_sha256": (run / "source_bundle_sha256.txt").read_text().strip()})
    license_path = package / "LICENSE.md"
    original_license = license_path.read_text()
    if "Synthetic augmentation" not in original_license:
        license_path.write_text(original_license + "\n- Synthetic augmentation: CC BY 4.0, "
                                "https://huggingface.co/datasets/saugataroy/ielts-task2-synthetic-v1.\n")
    write(package / "checksums.json", {p.name: sha(p) for p in sorted(package.iterdir())
                                       if p.is_file() and p.name != "checksums.json"})
    print(json.dumps({"package": str(package), "weights_sha256": result["weights_sha256"],
                      "ready_for_upload": True}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--package", type=Path, default=ROOT / "artifacts/staging/huggingface-synthetic-moe")
    args = parser.parse_args()
    main(args.run, args.package)
