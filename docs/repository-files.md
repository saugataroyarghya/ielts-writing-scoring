# What belongs in Git

Commit reusable source, tests, configuration, notebook source, release templates,
project documentation, and final aggregate results. Keep mutable run state and
machine-specific artifacts local. This policy applies to new files as well as the
files from the initial experiments.

| Files | Policy | Purpose |
|---|---|---|
| `src/`, `scripts/`, `tests/`, `configs/` | Commit | Training, preparation, evaluation, release tooling, and checks |
| `notebooks/`, `release_assets/` | Commit | Recorded source snapshots, inference wrapper, and card templates |
| `README.md`, `PLAN.md`, `docs/` | Commit | Current scope, reproduction instructions, and repository conventions |
| Final results and verification receipts in `reports/` | Commit | Aggregate evidence associated with recorded runs or revisions |
| `reports/run_status.json`, `live_*_progress.json`, `*_pending_completion.json` | Ignore; preserve locally | Mutable monitoring snapshots |
| `reports/synthetic_pipeline_status.json`, `synthetic_moe_preflight.json` | Ignore; preserve locally | Queue/launch state and operational checks |
| `reports/*_packages.json`, `*_local_checkpoint.json`, `synthetic_pipeline.lock` | Ignore; preserve locally | Local staging manifests, machine paths, and process locks |
| `data/`, `artifacts/` run/staging/private directories, `release/` | Ignore | Datasets, downloads, generated packages, and private sanity material |
| Model weights and credentials | Ignore | Weights are distributed through Hugging Face; credentials stay local |

The rules intentionally do not ignore the whole `reports/` directory. New final
reports should be reviewed before committing. Publish aggregate measurements,
source revisions, and hashes; exclude private official samples, their results,
raw reference essays, per-response predictions, tokens, and absolute local paths.
Use project-relative paths for retained checkpoint locations.

`reports/run_status.json` was tracked in the initial experiment. It is now removed
from the current Git tree while its local contents are preserved. Older commits
retain the historical version.

## Supporting scripts

These scripts supplement the core training entry points in `src/ielts_scoring/`.
They preserve the workflow used for the recorded experiments; many require local
artifacts that are intentionally not distributed in the Git repository.

| Scripts | Role and prerequisites |
|---|---|
| `synthetic_dataset.py`, `prepare_synthetic_review.py` | Validate authored rows, create blind review packets, audit judgments/adjudications, and export splits; require the local authoring workspace |
| `summarize_synthetic_dataset.py`, `audit_synthetic_overlap.py` | Summarize assessed labels and check lexical overlap; the overlap audit requires locally prepared corpora and a tokenizer |
| `verify_synthetic_release.py` | Reconcile exported files with the current local review records |
| `package_synthetic_hub.py` | Build the dataset package and card from the frozen local export; takes an explicit dataset license |
| `package_transfer_hub.py`, `package_synthetic_model.py` | Build the original transfer and continued-MoE model packages from recorded outputs |
| `verify_transfer_hub.py`, `validate_synthetic_candidate.py` | Check portable loading, prediction parity, and release gates; the latter additionally requires private sanity inputs and writes private results locally |
| `package_hub.py`, `verify_hub_checkpoint.py` | Earlier baseline packaging format; retained for provenance, not the current Transformers release interface |
| `run_synthetic_pipeline.py` | Publisher-specific orchestration that can launch Kaggle work, download results, validate, and update the configured Hugging Face repository |

The pipeline script uses the original account, experiment paths, and frozen parent
revision. It is operational code, not a generic quick-start command. Its network
and publication actions are not invoked by unit tests or by importing the module.
The completed release has already advanced beyond its configured parent, so its
overwrite guard prevents rerunning it against the newer public model.

For the recorded training workflow use [training.md](training.md). To inspect
supporting scripts, install this project and use their CLI help where available.
Card packaging imports `yaml` (PyYAML); the synthetic notebook builder imports
`nbformat`. The orchestration also requires separately authenticated `kaggle` and
`hf` command-line tools. Most script paths are relative to this repository; the
pipeline expects the local interpreter at `.venv/bin/python`.

The model-card templates record their packaging-stage content. The subsequent
GitHub source-link additions are recorded in `reports/github_source_published.json`
and the live Hugging Face cards. Rebuilding a package does not reproduce those
later documentation edits automatically. Review staged cards and checksums before
any new publication.

Run local tests without contacting Kaggle or Hugging Face:

```bash
python -m pytest -q
python scripts/verify_training_sources.py
```
