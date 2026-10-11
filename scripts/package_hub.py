"""Package the four frozen Kaggle checkpoints and evidence for private Hub upload."""

import argparse
import hashlib
import io
import json
import math
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import ModelCard
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hash():
    files = sorted([*ROOT.glob("src/**/*.py"), *ROOT.glob("configs/*.json"), ROOT / "pyproject.toml"])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), (2026, 10, 9, 0, 0, 0))
            archive.writestr(info, path.read_bytes())
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def slug(track, head):
    prefix = "ielts-task2" if track == "ielts" else "ellipse"
    suffix = "practice-scorer" if track == "ielts" else "writing-scorer"
    return f"{prefix}-modernbert-base-{head}-{suffix}"


def table(metrics, targets):
    lines = ["| Criterion | MAE | RMSE | Signed bias |", "| --- | ---: | ---: | ---: |"]
    for target in targets:
        row = metrics[target]
        lines.append(f"| {target} | {row['mae']:.4f} | {row['rmse']:.4f} | {row['bias']:.4f} |")
    lines.append(f"| **Macro average** | **{metrics['macro_mae']:.4f}** | "
                 f"**{metrics['macro_rmse']:.4f}** | — |")
    return "\n".join(lines)


def card(repo_id, track, head, config, result, best, counts):
    selected = result["selected_head"] == head
    title = "IELTS Task 2" if track == "ielts" else "ELLIPSE"
    head_name = "multi-gate MoE" if head == "moe" else "plain MLP"
    targets = config["targets"]
    ranges = "0–9 IELTS-style bands" if track == "ielts" else "1–5 ELLIPSE proficiency scores"
    source = ("https://huggingface.co/datasets/chillies/ielts-writing-task2-essays" if track == "ielts"
              else "https://github.com/scrosseye/ELLIPSE-Corpus")
    siblings = " · ".join(f"[{name.upper()}](https://huggingface.co/{repo_id.split('/')[0]}/{slug(track, name)})"
                          for name in ["mlp", "moe"])
    intro = ("Estimates Task Response (TR), Coherence and Cohesion (CC), Lexical Resource (LR), and "
             "Grammatical Range and Accuracy (GRA) from the original question and a typed Task 2 essay. "
             "Labels come from Writing9; this model has not been validated against official IELTS examiners."
             if track == "ielts" else
             "Estimates six analytic English-writing proficiency scores from a typed essay: Cohesion, "
             "Syntax, Vocabulary, Phraseology, Grammar, and Conventions. It was fine-tuned on human-rated "
             "ELLIPSE data. Its outputs are ELLIPSE scores and must not be rescaled into IELTS bands.")
    architecture = (
        "Four shared experts each apply `Linear(768,256) → GELU → Dropout(0.1) → Linear(256,256) → GELU`. "
        "Each criterion has a separate `Linear(768,4) → softmax` gate. The gate forms a weighted sum of "
        "the four expert vectors, followed by that criterion's `Linear(256,1)` scorer. All four experts "
        "are evaluated on every response; routing is a dense soft mixture. Expert specializations are learned."
        if head == "moe" else
        f"The head is `Linear(768,1376) → GELU → Dropout(0.1) → Linear(1376,{len(targets)})`.")
    midpoint = sum(config["label_range"]) / 2
    checkpoint_note = ("**Selected model for this dataset** in the seed-42 comparison."
                       if selected else "**Comparison model; not selected** in the seed-42 comparison.")
    dataset_details = (
        "8,049 source records; 8,048 retained after one exact duplicate was removed. The frozen splits "
        "contain 7,069 training, 529 validation, and 450 test responses. Prompt variants and lexical "
        "near-duplicate essays were grouped before splitting, producing 227 connected groups. The "
        "unequal row proportions preserve groups. This does not establish perfect separation of semantic "
        "paraphrases. Test data comes from the same website source, not an independent examiner benchmark. "
        "The source field `task_achievement_band` maps to TR because this track contains Task 2 essays. "
        "Overall bands, feedback, topic IDs, and source identifiers are not model inputs. The overall "
        "label is not a target; the returned task mean is computed from the four predictions."
        if track == "ielts" else
        "6,482 human-rated essays were retained. The authors' supplied test split has 2,571 essays. "
        "Their 3,911 training essays were divided into 3,324 training and 587 validation examples. "
        "Exact/lexical near-duplicate checks precede splitting. Prompts can overlap across splits, and "
        "the selected schema has no writer IDs: writer-independent and unseen-prompt performance are "
        "not established. Overall proficiency and demographic metadata are not model targets or inputs.")
    training = result["heads"][head]
    history_rows = "\n".join(f"| {row['epoch']} | {row['training_mse']:.4f} | {row['validation_mae']:.4f} |"
                             for row in training["history"])
    inputs = ("Pass `question` and `essay` as strings. The complete question is preserved; only the essay "
              "is truncated if the combined sequence exceeds 1,024 tokens. A missing/blank question or "
              "essay is rejected. Special tokens count toward the limit."
              if track == "ielts" else
              "Pass `essay` as a nonempty string and omit `question`. The essay is truncated if its "
              "sequence exceeds 1,024 tokens, including special tokens.")
    example = ('result = scorer.predict(\n'
               '    question="Should cities make public transport free? Discuss both views.",\n'
               '    essay=Path("essay.txt").read_text(encoding="utf-8"),\n'
               ')\nprint(result["scores"])  # TR, CC, LR, GRA\n'
               'print(result["task2_mean_estimate"])  # Arithmetic mean of four estimates'
               if track == "ielts" else
               'result = scorer.predict(essay=Path("essay.txt").read_text(encoding="utf-8"))\n'
               'print(result["scores"])  # Six named ELLIPSE criterion estimates')
    metadata = {
        "library_name": "pytorch", "base_model": config["model_id"],
        "base_model_relation": "finetune", "language": ["en"],
        "license": "other", "license_name": "private-research-release-pending",
        "license_link": f"https://huggingface.co/{repo_id}/blob/main/LICENSE.md",
        "tags": ["modernbert", "essay-scoring", "regression", "research", "custom-code",
                 "mixture-of-experts" if head == "moe" else "mlp", "ielts" if track == "ielts" else "ellipse"],
    }
    if track == "ielts":
        metadata["datasets"] = ["chillies/ielts-writing-task2-essays"]
    # JSON is valid YAML, so this preserves metadata values without hand-written escaping.
    frontmatter = "---\n" + json.dumps(metadata, indent=2) + "\n---\n\n"
    text = f"""# {title} ModernBERT-base with {head_name}

{intro}

{checkpoint_note} This repository contains the **best validation checkpoint from epoch {best['epoch']}**, not necessarily the final training epoch. All four project models started independently from the same pretrained ModernBERT-base revision. No model was trained on both datasets, and ELLIPSE-to-IELTS transfer was not run.

This is a private research release. Public redistribution and a final weight license are pending; see [LICENSE.md](LICENSE.md). Model family: {siblings}.

## Input and output

{inputs}

Output is a dictionary containing `scores`, `track`, `max_tokens`, and `truncated`{' plus `task2_mean_estimate`' if track == 'ielts' else ''}. Target order is `{', '.join(targets)}`. Training labels use {ranges}. Predictions are raw continuous regression estimates and are not clipped or half-band rounded; out-of-range outputs remain possible. `truncated=true` means the model did not see the complete response.

{'Task 1 Academic, Task 1 General Training, the full IELTS Writing band, generated feedback, and sentence-level error detection are not supported.' if track == 'ielts' else 'IELTS scoring, an overall ELLIPSE score, generated feedback, and sentence-level error detection are not supported.'}

## Run locally

Use Python 3.11–3.13 and a compatible PyTorch installation. Authenticate with `hf auth login` for access to this private repository. Download and install the included package:

```bash
hf download {repo_id} --local-dir ./writing-scorer
python -m pip install ./writing-scorer
```

```python
from pathlib import Path
from ielts_scoring.predict import Scorer

scorer = Scorer("./writing-scorer", device="cpu")
{example}
print(result["truncated"])
```

Use `device="cuda"` for an available CUDA GPU. This custom regression checkpoint is loaded through the supplied `Scorer`, not a default classification pipeline. The loader reads weights, encoder configuration, and tokenizer locally and supports offline inference after download. Pin `hf download --revision <commit-sha>` for a repeatable deployment. Dependency constraints are in `pyproject.toml`; the recorded training environment is in `training_environment.json`.

## Architecture

`{'Question + essay' if track == 'ielts' else 'Essay'} → tokenizer → ModernBERT-base → mean pool essay tokens → {head_name} → {len(targets)} scores`.

The pretrained encoder has 22 Transformer layers, 12 attention heads per layer, and hidden size 768. Our experiments use 1,024 tokens. Essay tokens are contextualized by the question in the same encoder pass for IELTS, while question, padding, and special tokens are excluded from mean pooling. {architecture} A fixed offset of {midpoint:g} is added to each raw head output. Scores do not use softmax; MoE routing weights do.

This checkpoint has **{counts['total']:,} parameters**, including **{counts['head']:,} head parameters**. The encoder and prediction head were fine-tuned together, without LoRA or encoder freezing. Code is included under `src/ielts_scoring/`.

## Dataset and split

Source: [{source}]({source}); frozen revision `{config['dataset_revision']}`.

{dataset_details}

Preparation preserves spelling, grammar, casing, and paragraph structure, with only basic text/label validity checks and transport cleanup. Missing labels are masked in the loss. Dataset files and pretrained revisions are pinned in `run_config.json`, `scorer_config.json`, and `configs/`. Raw essays and row-level predictions are not distributed with this checkpoint.

## Training

- Pretrained model: [`{config['model_id']}`](https://huggingface.co/{config['model_id']}), revision `{config['revision']}`.
- Seed: 42; epoch cap: 5; patience: 2 epochs without lower validation macro MAE.
- Loss: mean squared error across observed criterion labels, with gradient accumulation normalized over observed targets.
- AdamW: encoder learning rate `2e-5`, head learning rate `1e-4`, weight decay `0.01` excluding biases/one-dimensional parameters.
- Linear schedule with 10% warmup; microbatch 2; accumulation 8; effective batch size 16; gradient clipping 1.0.
- FP16 on one Tesla T4 per model, with gradient checkpointing. The plain and MoE jobs for a dataset ran concurrently on two T4s.
- This model's training/validation loop: {training['runtime']['seconds'] / 60:.1f} minutes; peak allocated GPU memory: {training['runtime']['peak_gpu_memory_gb']:.2f} GiB.
- Source bundle SHA-256: `{config['source_hash']}`.

## Validation results

Best checkpoint: epoch **{best['epoch']}**; macro MAE **{best['macro_mae']:.4f}**. Lower MAE/RMSE is better; units are the dataset's native score scale. Signed bias is prediction minus target.

{table(best, targets)}

| Epoch | Training MSE | Validation macro MAE |
| --- | ---: | ---: |
{history_rows}

Comparison on identical splits: plain MLP **{result['heads']['mlp']['best_validation_mae']:.4f}**, MoE **{result['heads']['moe']['best_validation_mae']:.4f}**, TF-IDF + Ridge **{result['baseline_validation_mae']:.4f}**. Only one training seed was run; these differences do not establish statistical superiority.
"""
    if selected:
        test = result["test_metrics"]
        ci = test["macro_mae_95ci"]
        text += f"""
## Held-out test results

This checkpoint was selected using validation only, then evaluated once on the frozen test set.

{table(test, targets)}

The 95% test-group bootstrap interval for macro MAE is **{ci['lower']:.4f}–{ci['upper']:.4f}**, based on {ci['draws']} resamples of {ci['groups']} groups. It measures test-group sampling uncertainty for this checkpoint, not uncertainty across training seeds. QWK is not reported because a source-verified discrete score grid was not established. No test prediction was outside the label range in this run.

The staged checkpoint passed fresh offline CPU inference, with maximum CPU/GPU score difference {result['cpu_inference_check']['max_difference_from_gpu']:.6f}. See `test_metrics.json`, `cpu_inference_check.json`, and `diagnostics.json`.
"""
        if track == "ielts":
            text += """
In one validation diagnostic, changing the question changed Task Response only from 6.0000 to 5.9375. A 40-word excerpt also received higher LR/GRA estimates than the full response. These are unlabeled probes on one example, not aggregate benchmark results, but they expose concerns requiring broader investigation. The test error of about 1.10 bands per criterion is not adequate evidence of dependable grading.
"""
    else:
        text += """
## Test evaluation status

**This checkpoint was not evaluated on the test set.** The pipeline reserved test evaluation for the head selected on validation. Its sibling's test scores must not be attributed to this model. This repository therefore reports validation metrics only and does not include a `test_metrics.json` file.
"""
    text += f"""
## Intended use and limitations

Use for reproducible research, comparing regression heads, and examining automated writing-score estimates. Human review is needed for consequential judgments. Performance outside the corpus's prompt, proficiency, and language distributions is unestablished. Scores are not explanations, and the models may rely on superficial writing features. Truncation can hide evidence; check the returned flag.

{'Website labels are weak supervision, not certified examiner ratings. Task 1 and complete IELTS Writing results are outside this checkpoint’s scope.' if track == 'ielts' else 'Human-rated ELLIPSE proficiency is a different construct from IELTS Writing bands. No conversion between the scales is established.'}

## Attribution and release status

The encoder is [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base), developed by Answer.AI, LightOn, and collaborators. Its published license is Apache 2.0; the standard license text is retained in `BASE_MODEL_LICENSE`. Our changes are the scoring head and full encoder fine-tuning.

{'Dataset attribution: chillies/ielts-writing-task2-essays, derived from Writing9 (https://writing9.com/). The uploader lists CC BY 4.0; original-source rights and public weight redistribution have not been independently cleared.' if track == 'ielts' else 'Dataset attribution: Crossley, Tian, Baffour, Franklin, Kim, Morris, Benner, Picou, and Boser (2023), “Measuring second language proficiency using the English Language Learner Insight, Proficiency and Skills Evaluation (ELLIPSE) Corpus,” International Journal of Learner Corpus Research 9(2), 248–269. The corpus repository specifies CC BY-NC-SA 4.0.'}

Base-model, dataset, and project-code permissions are recorded separately in `LICENSE.md` and `NOTICE.md`. This private upload does not assert that the fine-tuned weights inherit an unrestricted base-model license.
"""
    return frontmatter + text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", required=True)
    args = parser.parse_args()
    results = read(ROOT / "reports/final_training_results.json")
    expected_bundle = source_hash()
    manifest = []
    base_license = ROOT / "artifacts/staging/base-model-license/LICENSE"
    assert "Apache License" in base_license.read_text()
    for track, result in results["tracks"].items():
        run_root = ROOT / "artifacts/kaggle-output" / f"{track}-train-v1"
        for head in ["mlp", "moe"]:
            run = run_root / head
            config = read(run / "run_config.json")
            assert config["source_hash"] == expected_bundle, "Local source differs from trained bundle"
            assert read(run / "best/scorer_config.json") == config
            name = slug(track, head)
            repo_id = f"{args.namespace}/{name}"
            destination = ROOT / "artifacts/staging/huggingface" / name
            if destination.exists():
                raise FileExistsError(f"Refusing to reuse a nonempty staging directory: {destination}")
            shutil.copytree(run / "best", destination)
            snapshot = run_root / "evaluation/release"
            shutil.copytree(snapshot / "src", destination / "src", ignore=shutil.ignore_patterns("__pycache__"))
            for path in (destination / "src").rglob("*.py"):
                assert path.read_bytes() == (ROOT / path.relative_to(destination)).read_bytes()
            shutil.copy(snapshot / "pyproject.toml", destination / "pyproject.toml")
            shutil.copytree(ROOT / "configs", destination / "configs")
            shutil.copy(base_license, destination / "BASE_MODEL_LICENSE")
            for filename in ["run_config.json", "best_validation.json", "history.json", "runtime.json"]:
                shutil.copy(run / filename, destination / filename)
            shutil.copy(run / "environment.json", destination / "training_environment.json")
            shutil.copy(run / "tokenization.json", destination / "training_tokenization.json")
            shutil.copy(ROOT / f"reports/{track}_baselines_validation.json",
                        destination / "baselines_validation.json")
            shutil.copy(run_root / "evaluation/selection.json", destination / "selection.json")
            selected = result["selected_head"] == head
            if selected:
                for filename in ["test_metrics.json", "cpu_inference_check.json", "diagnostics.json"]:
                    shutil.copy(run_root / "evaluation" / filename, destination / filename)
                shutil.copy(run_root / "evaluation/tokenization.json", destination / "test_tokenization.json")
            counts = {"total": 0, "head": 0}
            with safe_open(destination / "model.safetensors", framework="pt", device="cpu") as weights:
                for key in weights.keys():  # noqa: SIM118 -- safe_open is not an iterable dict.
                    parameters = math.prod(weights.get_slice(key).get_shape())
                    counts["total"] += parameters
                    if key.startswith("head."):
                        counts["head"] += parameters
            best = read(run / "best_validation.json")
            (destination / "README.md").write_text(card(repo_id, track, head, config, result, best, counts))
            (destination / "LICENSE.md").write_text(
                "# Private research release: licensing status\n\n"
                "These fine-tuned checkpoints are staged privately. No new public redistribution or commercial-use "
                "license for the complete checkpoint is granted by this upload. A public weight license remains pending.\n\n"
                "- Base model: ModernBERT-base, published under Apache 2.0; see BASE_MODEL_LICENSE and "
                "https://huggingface.co/answerdotai/ModernBERT-base. Existing third-party rights are unchanged.\n"
                "- ELLIPSE data: CC BY-NC-SA 4.0 according to https://github.com/scrosseye/ELLIPSE-Corpus.\n"
                "- IELTS Task 2 data: the uploader lists CC BY 4.0 at "
                "https://huggingface.co/datasets/chillies/ielts-writing-task2-essays; original Writing9 source "
                "permissions and public weight-release conditions remain unresolved.\n"
                "- Project-authored scoring code: included for running this private research checkpoint; "
                "no separate public open-source license has been assigned. Dependencies retain their own licenses.\n\n"
                "Dataset terms and model weight licensing are separate questions. This notice records current "
                "provenance and release status; it does not assert that a dataset license automatically applies to weights.\n"
            )
            (destination / "NOTICE.md").write_text(
                f"# Attribution\n\nBase: {config['model_id']} at {config['revision']}, by Answer.AI, LightOn, "
                "and collaborators. This checkpoint adds a regression head and fine-tunes the encoder.\n\n"
                f"Dataset: {'chillies/ielts-writing-task2-essays (Writing9)' if track == 'ielts' else 'ELLIPSE-Corpus (Crossley et al., 2023)'}, "
                f"revision {config['dataset_revision']}. Full attribution and evaluation details are in README.md.\n"
            )
            parsed = ModelCard.load(destination / "README.md")
            assert parsed.data.base_model == config["model_id"]
            assert not (destination / "test_metrics.json").exists() or selected
            provenance = {"repo_id": repo_id, "track": track, "head": head, "selected_on_validation": selected,
                          "checkpoint_epoch": best["epoch"], "parameters": counts,
                          "kaggle_notebook": result["notebook"], "kaggle_version": result["version"],
                          "source_bundle_sha256": expected_bundle, "dataset_checksums": config["dataset_checksums"],
                          "pretrained_revision": config["revision"], "visibility": "private"}
            write(destination / "provenance.json", provenance)
            hashes = {path.relative_to(destination).as_posix(): checksum(path)
                      for path in sorted(destination.rglob("*")) if path.is_file()}
            write(destination / "checksums.json", hashes)
            manifest.append({**provenance, "local_dir": str(destination), "file_count": len(hashes) + 1,
                             "weights_sha256": hashes["model.safetensors"]})
            print(json.dumps({"repo_id": repo_id, "best_epoch": best["epoch"], "parameters": counts,
                              "selected": selected}), flush=True)
    write(ROOT / "reports/huggingface_packages.json", manifest)


if __name__ == "__main__":
    main()
