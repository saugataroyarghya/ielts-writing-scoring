"""Build standalone Hub packages for the two final Task 2 transfer checkpoints."""

import hashlib
import json
import shutil
from pathlib import Path

import yaml
from huggingface_hub import ModelCard

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = "saugataroy"
HEADS = ("mlp", "moe")


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def repo_id(head):
    return f"{NAMESPACE}/ielts-task2-modernbert-base-{head}"


def metric_table(metrics):
    lines = ["| Criterion | MAE ↓ | RMSE ↓ | Bias (prediction − label) |",
             "|---|---:|---:|---:|"]
    for target in ("TR", "CC", "LR", "GRA"):
        row = metrics[target]
        lines.append(f"| {target} | {row['mae']:.4f} | {row['rmse']:.4f} | {row['bias']:+.4f} |")
    lines.append(f"| Macro average | **{metrics['macro_mae']:.4f}** | **{metrics['macro_rmse']:.4f}** | — |")
    return "\n".join(lines)


def model_card(head, results, evaluation, benchmark):
    name = "Plain MLP" if head == "mlp" else "Multi-gate MoE"
    other = "moe" if head == "mlp" else "mlp"
    params = results["gpu_latency"]["checkpoints"][head]["parameters"]
    metadata = {
        "language": ["en"], "library_name": "transformers",
        "base_model": "answerdotai/ModernBERT-base", "base_model_relation": "finetune",
        "license": "other", "license_name": "licensing-status-notice",
        "license_link": "LICENSE.md",
        "datasets": ["chillies/ielts-writing-task2-essays"],
        "tags": ["modernbert", "ielts", "essay-scoring", "regression", "custom-code",
                 "mixture-of-experts" if head == "moe" else "mlp"],
    }
    architecture = (
        "The plain head is `Linear(768,1376) → GELU → Dropout(0.1) → Linear(1376,4)`."
        if head == "mlp" else
        "The head contains four shared experts, each `Linear(768,256) → GELU → Dropout(0.1) → "
        "Linear(256,256) → GELU`. Each rubric has a `Linear(768,4) → softmax` gate, a weighted "
        "sum of the expert vectors, and a separate `Linear(256,1)` output. All four experts "
        "execute for every essay; this is a dense soft mixture, not sparse top-k routing."
    )
    test_section = (
        metric_table(evaluation["test"]) + "\n\n"
        "Test macro MAE: **1.1289**, with a prompt-group bootstrap 95% interval of "
        "**1.0253–1.2864** (23 groups, 1,000 resamples). This measures test-sample uncertainty, "
        "not variability across training seeds. No out-of-range predictions were observed in this test."
        if head == "moe" else
        "This checkpoint has **not been evaluated on the held-out test split**. Only the model "
        "selected by validation macro MAE (the MoE variant) received the held-out evaluation. "
        "The validation metrics above must not be presented as test results."
    )
    latency_rows = "\n".join(f"| {c['tokens']} | {c['p50_ms']:.2f} | {c['p95_ms']:.2f} |"
                             for c in benchmark["cases"] if c["precision"] == "float16")
    config = results["models"][head]["configuration"]
    return "---\n" + yaml.safe_dump(metadata, sort_keys=False) + f"""---

# IELTS Task 2 ModernBERT-base — {name}

An English essay-scoring model that estimates four IELTS Task 2 writing criteria from a question and its essay response: **Task Response (TR), Coherence and Cohesion (CC), Lexical Resource (LR), and Grammatical Range and Accuracy (GRA)**.

The model uses a fine-tuned ModernBERT-base encoder and a {name.lower()} regression head. It is intended for research, writing-practice prototypes, and evaluation of automated scoring systems. Its IELTS labels originate from Writing9; agreement with qualified IELTS examiners has not been established.

## Model details

| Property | Value |
|---|---|
| Model type | Encoder with four continuous regression outputs |
| Base encoder | [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) |
| Parameters | {params:,} |
| Language | English |
| Supported task | IELTS Writing Task 2 |
| Input limit | 1,024 tokens across question, essay, and special tokens |
| Pooling | Mean over essay tokens only |
| Weight format | FP32 Safetensors, full encoder and scoring head |
| Related model | [{repo_id(other)}](https://huggingface.co/{repo_id(other)}) |

## Inputs and outputs

Supply the **original question** and a **nonempty typed essay**. The tokenizer preserves the question and truncates only the essay when necessary. A question too long to fit is rejected by the tokenizer. Spelling, grammar, capitalization, and paragraph structure should be preserved.

The `score` method returns one result per question–essay pair:

- `scores`: continuous estimates keyed by `TR`, `CC`, `LR`, and `GRA`.
- `task2_mean_estimate`: arithmetic mean of the four estimates.
- `truncated`: whether the complete input exceeded the token limit.
- `track`: `"ielts"`; `max_tokens`: `1024`.

Labels use the 0–9 scale. Outputs are **not clipped or half-band rounded**, so out-of-range values are possible. The arithmetic mean is not an official IELTS score or the full Writing band. The model does not generate feedback or identify sentence-level errors. Task 1 Academic and General Training letters are outside its supported scope.

## Usage

Use Python 3.11–3.13 and install a PyTorch build suitable for your device:

```bash
pip install 'torch>=2.6,<2.12' 'transformers==4.57.6' 'huggingface-hub==0.36.0' 'safetensors>=0.6,<0.8'
```

```python
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

model_id = "{repo_id(head)}"
# Set revision to the repository commit SHA for reproducible deployments.
revision = "main"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
model = AutoModel.from_pretrained(
    model_id, revision=revision, trust_remote_code=True,
).eval().to("cpu")  # "cuda" for NVIDIA; "mps" for a supported Apple GPU

result = model.score(
    tokenizer,
    questions=["Should cities make public transport free? Discuss both views."],
    essays=[Path("essay.txt").read_text(encoding="utf-8")],
)[0]
print(result["scores"])
print(result["task2_mean_estimate"])
print(result["truncated"])
```

`trust_remote_code=True` loads the small custom architecture implementation in [modeling_ielts.py](modeling_ielts.py). Review and pin this file together with the weights. Pass lists of multiple questions and essays for batch inference. Download the repository first and use its local path with `local_files_only=True` for offline loading. Private repositories require authentication.

The low-level forward method accepts `input_ids`, `attention_mask`, and `essay_mask`, and returns `logits` shaped `[batch, 4]`. These logits are regression scores, not class probabilities. `essay_mask` marks sequence-1 tokens and excludes question, special, and padding tokens. A generic text-classification pipeline does not supply this mask; use `score` or reproduce that preprocessing explicitly.

## Architecture

```text
Question + essay → paired tokenizer → ModernBERT-base
  → essay-token mean pooling → {name} head → TR / CC / LR / GRA
```

The encoder has 22 layers, hidden size 768, and 12 attention heads per layer. Question and essay tokens are jointly contextualized, although only essay tokens contribute directly to pooling. {architecture} A fixed offset of 4.5 is added to each head output. The prediction scores use no sigmoid or softmax. Dropout is disabled during evaluation.

## Training data and method

The encoder first received supervised writing-proficiency training on [ELLIPSE](https://github.com/scrosseye/ELLIPSE-Corpus), then was fine-tuned with a fresh IELTS head on [chillies/ielts-writing-task2-essays](https://huggingface.co/datasets/chillies/ielts-writing-task2-essays). ELLIPSE's six 1–5 trait scores were not mapped to IELTS bands. Both released variants started from the same ELLIPSE-trained encoder; their final encoder weights differ after IELTS fine-tuning.

| Dataset | Training | Validation | Test | Labels |
|---|---:|---:|---:|---|
| ELLIPSE | 3,324 | 587 | 2,571, unused by these IELTS models | Six human-rated proficiency traits |
| IELTS Task 2 | 7,069 | 529 | 450 | Four Writing9 rubric scores |

The IELTS source contained 8,049 records; one exact duplicate was removed. Prompt variants and lexical near-duplicate essays were grouped before splitting. Validation and test each contain 23 groups. Grouping reduces leakage but does not prove separation of all semantic paraphrases. No normalized exact essay overlap was found between ELLIPSE training/validation and the IELTS splits. Source feedback and overall scores are not model inputs; overall scores are not training targets. Raw essays are not included in this repository.

IELTS fine-tuning used masked mean squared error, AdamW, encoder/head learning rates of `2e-5`/`1e-4`, effective batch size 16 (microbatch 2, accumulation 8), 10% warm-up, weight decay 0.01, and seed 42. All encoder and head parameters were trained. The released checkpoint is epoch 4, selected by validation macro MAE within a five-epoch run. Exact dataset revisions, configuration, and checkpoint checksums are in [provenance.json](provenance.json).

## Evaluation

MAE and RMSE are measured in rubric-band units; lower is better. Macro MAE is the unweighted mean of the four rubric MAEs. Bias is prediction minus reference label. Evaluation compares raw continuous predictions against the source labels, without rounding. These results measure agreement with Writing9 labels, not official examiner agreement.

### Validation — 529 essays

{metric_table(evaluation['validation'])}

Under the same split and initialization, plain MLP validation macro MAE is **1.0619** and MoE is **1.0468**. The paired prompt-group bootstrap interval for MoE minus MLP is **−0.0350 to +0.0100**. It includes zero; this one-seed comparison does not establish a robust MoE advantage. The interval is exploratory because checkpoint selection used the same validation set.

### Held-out test — 450 essays

{test_section}

The frozen test split had also been used to report an earlier IELTS-only baseline. It was excluded from transfer-model tuning and selection, but is not a newly unseen benchmark. The selected transfer MoE's test MAE of 1.1289 did not improve on that baseline's 1.1001. See [evaluation.json](evaluation.json) for the exact reported metrics and evaluation policy.

## Inference performance

Measured on one Tesla T4 (`cuda:0`) using FP16 autocast, batch size 1, five warm-up calls, and 50 measured calls per model and length. Inputs were synthetic repeated-word sequences of the specified joint token length. Timing includes tokenization, padding, device transfer, inference, and result formatting; it excludes loading, HTTP transport, and server queueing.

| Joint tokens | Median request latency (ms) | p95 request latency (ms) |
|---:|---:|---:|
{latency_rows}

These are warm single-request latency measurements, **not a sustained-throughput or production-service benchmark**. The benchmark host had two T4 devices; timed inference used only one. Environment details and FP32 measurements are in [benchmark.json](benchmark.json).

## Limitations and appropriate use

- Use for research and practice estimates with human review; not for official grading, certification, admissions, or other consequential decisions.
- Generalization beyond the source website, demographic fairness, and agreement with independent examiners have not been established.
- In 23 validation probes, replacing the question did not lower average TR scores. Reliable question relevance detection is not established.
- Truncating essays to 40 words lowered TR/CC but could leave LR/GRA implausibly high. These unlabeled probes are sensitivity checks, not an accuracy benchmark.
- Only one training seed was evaluated. The MoE head provides no demonstrated large accuracy or inference-speed advantage.
- Truncated responses are scored only from the visible portion. No uncertainty interval or calibrated confidence is returned for individual predictions.

## Licensing and attribution

The base model is published under Apache 2.0; its license is retained in [BASE_MODEL_LICENSE](BASE_MODEL_LICENSE). ELLIPSE is distributed under CC BY-NC-SA 4.0. The IELTS dataset uploader declares CC BY 4.0, while original Writing9 source permissions have not been independently resolved. Dataset licenses and weight licenses are separate questions. This repository does not claim a blanket permissive or commercial-use license for the complete checkpoint; see [LICENSE.md](LICENSE.md).

Base model: Answer.AI, LightOn, and ModernBERT contributors. ELLIPSE: Crossley and collaborators. IELTS dataset: `chillies`, sourced from Writing9. Base revision: `{config['revision']}`. IELTS dataset revision: `{config['dataset_revision']}`.
"""


def main():
    results = read(ROOT / "reports/final_transfer_results.json")
    manifest = []
    for head in HEADS:
        source = ROOT / "artifacts/kaggle-output/transfer-v1" / head / "best"
        destination = ROOT / "artifacts/staging/huggingface-transfer" / repo_id(head).split("/")[1]
        destination.mkdir(parents=True, exist_ok=True)
        for path in source.iterdir():
            if path.is_file():
                shutil.copy2(path, destination / path.name)
        config = read(source / "scorer_config.json")
        encoder = read(source / "encoder_config/config.json")
        write(destination / "config.json", {
            "model_type": "ielts_scoring", "architectures": ["IELTSScoringModel"],
            "auto_map": {"AutoConfig": "modeling_ielts.IELTSScoringConfig",
                         "AutoModel": "modeling_ielts.IELTSScoringModel"},
            "encoder_config": encoder, "head": head, "targets": config["targets"],
            "max_length": 1024, "label_range": [0, 9], "torch_dtype": "float32",
        })
        shutil.copy2(ROOT / "release_assets/modeling_ielts.py", destination / "modeling_ielts.py")
        shutil.copy2(ROOT / "artifacts/staging/base-model-license/LICENSE", destination / "BASE_MODEL_LICENSE")
        evaluation = {
            "validation": results["selection"]["validation"][head],
            "test": results["test_metrics_selected_moe"] if head == "moe" else None,
            "test_policy": results["selection"]["test_note"],
            "paired_validation_comparison": results["validation_comparison"]["comparisons"]
                ["transfer_mlp_vs_transfer_moe"],
        }
        benchmark = {k: v for k, v in results["gpu_latency"].items()
                     if k in ("environment", "device", "batch_size", "warmup_per_head_per_case",
                              "paired_repetitions", "timing", "input", "end_to_end_includes", "excludes")}
        benchmark["cases"] = [{"tokens": c["tokens"], "precision": c["precision"],
                               **c["end_to_end"][head]} for c in results["gpu_latency"]["cases"]]
        write(destination / "evaluation.json", evaluation)
        write(destination / "benchmark.json", benchmark)
        (destination / "README.md").write_text(model_card(head, results, evaluation, benchmark))
        (destination / "LICENSE.md").write_text(
            "# Licensing status\n\n"
            "This notice records provenance; it does not grant an additional blanket license for the "
            "complete fine-tuned checkpoint. Existing third-party permissions remain applicable.\n\n"
            "- ModernBERT-base: Apache 2.0; see BASE_MODEL_LICENSE.\n"
            "- ELLIPSE training data: CC BY-NC-SA 4.0, https://github.com/scrosseye/ELLIPSE-Corpus.\n"
            "- IELTS training data: uploader declares CC BY 4.0 at "
            "https://huggingface.co/datasets/chillies/ielts-writing-task2-essays. "
            "Original Writing9 permissions and commercial weight-use rights are not independently resolved.\n"
            "- Custom scoring code is supplied with the checkpoint; no separate open-source license "
            "has been assigned to project-authored code.\n\n"
            "Dataset terms do not automatically determine the legal status of trained weights. "
            "Public availability, if enabled, is not a representation of unrestricted commercial-use rights.\n"
        )
        weights_sha = sha256(destination / "model.safetensors")
        assert weights_sha == results["local_checkpoints"][head]["sha256"]
        provenance = {"repo_id": repo_id(head), "head": head, "best_epoch": 4,
                      "parameters": results["gpu_latency"]["checkpoints"][head]["parameters"],
                      "weights_sha256": weights_sha, "training_configuration": config,
                      "ellipse_revision": "dc3b8f0b3b4332fc9f64302c4ccfc4ed582f4b43",
                      "training_environment": results["models"][head]["environment"]}
        write(destination / "provenance.json", provenance)
        ModelCard.load(destination / "README.md")
        write(destination / "checksums.json", {
            p.name: sha256(p) for p in sorted(destination.iterdir()) if p.is_file() and p.name != "checksums.json"
        })
        manifest.append({"repo_id": repo_id(head), "head": head, "local_dir": str(destination),
                         "weights_sha256": weights_sha})
    write(ROOT / "reports/huggingface_transfer_packages.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
