# Training and reproducibility

This repository contains the implementation and saved source snapshots for the
plain MLP and dense multi-gate MoE IELTS Task 2 scorers. Published weights, model
cards, evaluation tables, and inference examples live on Hugging Face:

- [Plain MLP](https://huggingface.co/saugataroy/ielts-task2-modernbert-base-mlp)
- [MoE](https://huggingface.co/saugataroy/ielts-task2-modernbert-base-moe)
- [Synthetic dataset](https://huggingface.co/datasets/saugataroy/ielts-task2-synthetic-v1)

## Verify the actual training source

The completed Kaggle notebooks embed ZIP archives of their Python source and
configuration. The public source files match those archives byte for byte.
[training_sources.json](training_sources.json) records archive and member hashes,
the relevant notebook, and the published checkpoint revisions. The archive hashes
were checked against the saved output of each completed run.

```bash
python scripts/verify_training_sources.py
```

This check only reads the notebook; it does not execute embedded code or train a
model. It deliberately fails if a recorded source file changes. Future experiments
should retain these snapshots and record a new source version.

## Environment

Use Python 3.11–3.13; the local development environment used Python 3.12. Install
a PyTorch build compatible with your CUDA hardware, then install this project:

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/test_data.py tests/test_models.py tests/test_training.py \
  tests/test_finalize.py tests/test_transfer.py tests/test_synthetic_finetune.py
```

`pyproject.toml` specifies supported dependencies. `uv.lock` captures the local
development environment, not the Kaggle GPU image. Both recorded GPU runs used
Tesla T4, CUDA 12.8, torch 2.11.0+cu128, transformers 4.57.6,
huggingface-hub 0.36.0, safetensors 0.7.0, pandas 2.3.3, numpy 2.1.3,
and scikit-learn 1.6.1. Their model repositories contain the recorded environment.
The notebooks additionally pin tokenizers 0.22.2.

Training requires CUDA in the recorded scripts. CPU unit tests do not need model
downloads. The benchmark module can run on CPU or CUDA; it is not a production
HTTP load test.

## Data access and exact reproduction

The original prepared IELTS and ELLIPSE splits were attached as private Kaggle
datasets. Their rows and per-response prediction files are not redistributed here.
Source repository revisions are pinned in `configs/sources.json`, and the exact
prepared Parquet hashes are pinned in `configs/prepared_datasets.json`.

To prepare data independently from the upstream sources:

```bash
python -m ielts_scoring.data --sources configs/sources.json --output data
```

This downloads the pinned upstream data and writes `data/prepared/ellipse/` and
`data/prepared/ielts/`, including `metadata.json` and `split_manifest.csv`.
Review upstream access and licensing terms before use. The authors' documented
ELLIPSE archive password is in the preparation code.

**Exact reruns require the recorded prepared files and, for the transfer stage,
the original ELLIPSE checkpoint.** Repreparation with different library versions
can change Parquet bytes; the scripts reject hashes that differ from the frozen
experiment. Do not disable those checks and describe the result as an exact
reproduction. Record new split hashes and a separate configuration for a new run.
The private attachments are not made public by sharing these notebooks.

## Plain MLP and original MoE training

1. Train an ELLIPSE scorer with six proficiency targets using `ielts_scoring.train`.
2. Select its encoder checkpoint using ELLIPSE validation macro MAE.
3. Initialize two IELTS models with that encoder and fresh MLP or MoE heads.
4. Fine-tune on four IELTS website-label targets and select using validation only.

The original ELLIPSE MLP checkpoint SHA-256 is pinned in `configs/transfer.json`.
It is an intermediate private training artifact, not one of the public IELTS models.
ELLIPSE labels are not mapped numerically to IELTS bands.

With the recorded prepared data and ELLIPSE checkpoint available locally:

```bash
python -m ielts_scoring.train --track ielts --head mlp --mode train \
  --data-root data --init-checkpoint /path/to/ellipse/mlp/best \
  --output artifacts/runs/transfer/mlp
python -m ielts_scoring.train --track ielts --head moe --mode train \
  --data-root data --init-checkpoint /path/to/ellipse/mlp/best \
  --output artifacts/runs/transfer/moe
python -m ielts_scoring.finalize --track ielts --input data \
  --runs artifacts/runs/transfer --output artifacts/runs/transfer/evaluation
```

The last command selects between heads by validation MAE before evaluating the
selected model on the held-out test. The published MLP checkpoint has validation
results, not a reported test result. `train.py` contains the hyperparameters;
`configs/transfer.json` records initialization and experiment policy.

The exact orchestration is saved in
[`notebooks/essay-scoring-ellipse-to-ielts-transfer/run.ipynb`](../notebooks/essay-scoring-ellipse-to-ielts-transfer/run.ipynb).
It also compares inference timing and validation diagnostics. Its private
attachments must be available to the executing Kaggle account.

## MoE synthetic continuation

[`synthetic_finetune.py`](../src/ielts_scoring/synthetic_finetune.py) downloads the
parent checkpoint and synthetic dataset directly from immutable Hugging Face
revisions in [`configs/synthetic_moe.json`](../configs/synthetic_moe.json).
It requires the original prepared IELTS splits for replay and evaluation:

```bash
python -m ielts_scoring.synthetic_finetune --config configs/synthetic_moe.json \
  --data-root data --output artifacts/runs/synthetic-moe-v1
```

Each epoch uses 976 synthetic training rows and 1,952 fixed original-training
replay rows. All encoder and head parameters continue from the parent checkpoint;
no head is reset. Learning rates are 5e-6 for the encoder and 5e-5 for the head,
microbatch 2, accumulation 8, seed 42, and a five-epoch cap.

Selection uses the equal mean of original and synthetic validation macro MAE,
with the parent eligible as epoch zero. Final diagnostic and test evaluation occur
after selection is frozen. Outputs include `best/`, `history.json`, `selection.json`,
`experiment_config.json`, `environment.json`, metrics, and prediction tables.
The training program does not upload checkpoints. Its public evaluation gates
are only part of the release decision; private sanity checking and portable-loader
verification were separate local steps. Private sanity materials and results are
not included in this repository.

The exact run is saved in
[`notebooks/essay-scoring-moe-synthetic-v1/run.ipynb`](../notebooks/essay-scoring-moe-synthetic-v1/run.ipynb).
`scripts/build_kaggle.py` and `scripts/build_synthetic_kaggle.py` are notebook
builders; the latter needs `nbformat`. They target the original publisher's Kaggle
account and private attachments. Adapt these identifiers for your own experiment.
Rebuilding a notebook creates a new source snapshot; keep the recorded notebooks
unchanged when auditing the published models.

## What the results establish

The continuation reduced macro MAE on 120 synthetic diagnostic rows from 1.8788
to 0.8422, while increasing original-test macro MAE on 450 rows from 1.1289 to
1.1968. This is improved agreement with LLM labels and worse agreement with the
original website labels. Neither label source establishes examiner accuracy.
See the MoE model card and `evaluation.json` for complete results and uncertainty.

The previous MoE checkpoint remains available at the Hugging Face tag
[`pre-synthetic-v1`](https://huggingface.co/saugataroy/ielts-task2-modernbert-base-moe/tree/pre-synthetic-v1).
The plain MLP weights were not changed by synthetic continuation.

## Licensing

No separate code license has been selected for this repository. Public visibility
does not assign the dataset's license to the code or weights. The synthetic
dataset is CC BY 4.0. The model repositories contain their own licensing-status
notices covering the base model and training sources. Dependencies retain their
respective licenses.
