# Writing scoring experiments

Fine-tune ModernBERT-base with either a plain regression MLP or four shared experts and one gate per scoring criterion. Training runs on Kaggle. ELLIPSE proficiency scores and IELTS Task 2 website scores stay in separate datasets and models.

## Current runs and data

- [Preparation notebook](https://www.kaggle.com/code/saugataroyarghya/essay-scoring-data-preparation): completed.
- [Private ELLIPSE dataset](https://www.kaggle.com/datasets/saugataroyarghya/ellipse-writing-scoring-private): version 1; 3,324 train / 587 validation / 2,571 official test essays.
- [Private IELTS dataset](https://www.kaggle.com/datasets/saugataroyarghya/ielts-writing-scoring-private): version 1; 7,069 train / 529 validation / 450 test essays.
- [ELLIPSE GPU smoke test](https://www.kaggle.com/code/saugataroyarghya/essay-scoring-ellipse-smoke): passed for both heads, with exact GPU save/load parity. See `reports/gpu_smoke_ellipse.json`.
- [ELLIPSE training comparison](https://www.kaggle.com/code/saugataroyarghya/essay-scoring-ellipse-train) and [IELTS training comparison](https://www.kaggle.com/code/saugataroyarghya/essay-scoring-ielts-train): see `reports/run_status.json` for the last verified states.

The links require the owner's Kaggle access. No model performance is claimed until a run completes. See [the light data audit](reports/data_audit.md) and [the plan](PLAN.md).

## Local development

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check src tests scripts
```

`uv.lock` pins dependencies; local unit tests use a tiny randomly initialized ModernBERT and do not download pretrained weights. The Kaggle image supplies CUDA PyTorch. Each remote run records its actual package versions, GPU, source bundle hash, dataset checksums, and pretrained model revision.

## Kaggle execution

```bash
python scripts/build_kaggle.py prepare
kaggle kernels push -p notebooks/essay-scoring-data-preparation -t 3600
kaggle kernels status saugataroyarghya/essay-scoring-data-preparation

python scripts/build_kaggle.py smoke --track ellipse
kaggle kernels push -p notebooks/essay-scoring-ellipse-smoke --accelerator NvidiaTeslaT4 -t 2400

python scripts/build_kaggle.py train --track ellipse --head both
kaggle kernels push -p notebooks/essay-scoring-ellipse-train --accelerator NvidiaTeslaT4 -t 21600

python scripts/build_kaggle.py train --track ielts --head both
kaggle kernels push -p notebooks/essay-scoring-ielts-train --accelerator NvidiaTeslaT4 -t 21600
```

Choose an accelerator available to the account; the notebook checks CUDA before training. Each model uses one GPU. When Kaggle allocates two T4s, `--head both` runs the two models in separate processes on separate GPUs; with one GPU it runs them sequentially. The first L4 request actually received two T4s, so subsequent jobs explicitly request T4. Building a notebook only writes local files; `kernels push` starts the remote run. All generated notebooks are private.

Preparation fetches pinned sources on Kaggle, preserves text and score scales, validates rows, groups lexical near-duplicates, and freezes splits. Source revisions are in `configs/sources.json`; immutable prepared-file hashes are in `configs/prepared_datasets.json`. A later dataset update fails the training checksum check until the experiment configuration is intentionally updated.

Training uses essay-token mean pooling, a 1,024-token cap, explicit masked MSE, microbatch 2, gradient accumulation 8, encoder LR `2e-5`, head LR `1e-4`, and at most five epochs. The best checkpoint is selected on macro validation MAE, with patience 2. The plain head has width 1,376; the MoE has four width-256 experts. At hidden size 768 their head parameter counts are within 1% for both tracks. MoE gates use softmax; the four/six regression scores do not.

The question and essay share one encoder pass for IELTS; pooling uses only essay tokens. Spelling, grammar, casing, and paragraph boundaries are preserved. A single encoder checkpoint is fine-tuned end to end. No tokens or credentials are embedded in the notebook.

## Outputs and inference

Each run saves `best/model.safetensors`, tokenizer files, `scorer_config.json`, the encoder configuration, validation predictions, per-target metrics, slices, token-length statistics, gate usage for MoE, and runtime metadata. Test labels are not used for checkpoint or architecture selection.

The combined training notebook automatically selects the head with the lowest validation MAE, evaluates only that selected checkpoint on the frozen test set, calculates confidence intervals by resampling test groups, runs controlled text/question diagnostics, stages an inference bundle, and verifies CPU inference after a fresh offline load. Both candidate checkpoints remain in the private notebook outputs. Public publication is a separate step. If final evaluation needs to be rerun after a completed training job, build the `evaluate` notebook; it attaches that job's saved outputs.

Download a completed run with `kaggle kernels output <owner>/<notebook> -p artifacts/kaggle-output/<run>`. Then:

```python
from ielts_scoring.predict import Scorer

scorer = Scorer("path/to/best", device="cpu")
result = scorer.predict(essay="Your essay", question="The original IELTS Task 2 question")
```

For ELLIPSE, omit `question`. Outputs are continuous native-scale estimates; IELTS also returns their mean as a Task 2 estimate. Empty essays are rejected and truncation is reported. This is a custom regression interface: use the package above, not a default classification pipeline.

## Scope and provenance

ELLIPSE uses its authors' CC BY-NC-SA 4.0 corpus and supplied test partition. IELTS uses Writing9-derived weak labels from `chillies/ielts-writing-task2-essays`; the uploader's CC BY 4.0 claim is recorded separately from unresolved original-source permissions. IELTS metrics measure agreement with those website labels. Public model release requires a checkpoint-specific provenance/license decision. The current Kaggle artifacts are private.
