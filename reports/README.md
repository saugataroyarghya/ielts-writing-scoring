# Recorded results and verification

These are immutable observations from specific experiments or publication steps,
not live status. Later publication receipts supersede earlier repository revisions;
older receipts are retained to identify the artifacts originally verified.

| Report | Scope |
|---|---|
| `final_training_results.json` | Initial separate ELLIPSE and IELTS baseline experiments |
| `final_transfer_results.json` | Shared ELLIPSE encoder transfer to MLP and MoE IELTS heads, including validation, selected-model test, and direct inference timing |
| `huggingface_synthetic_moe_published.json` | Synthetic continuation: released checkpoint, selection, and original/synthetic evaluation comparison |
| `huggingface_transfer_published.json` | Original two-model publication and loader verification |
| `huggingface_transfer_mlp_verification.json`, `huggingface_transfer_moe_verification.json` | Strict loading and prediction parity for the original transfer releases |
| `huggingface_synthetic_published.json`, `huggingface_synthetic_loader_verification.json` | Public synthetic dataset revisions, counts, and actual loader verification |
| `github_source_published.json` | Verified training-source commit and subsequent model-card documentation revisions |
| `transfer_initialization_check.json`, `synthetic_moe_parent_smoke.json` | Initialization identity and strict parent-load checks |
| `ielts_training_input_lengths.json` | Measured token lengths and truncation scope |
| `inference_latency_cpu_baseline.json` | Earlier IELTS-only model timing; not a benchmark of the latest MoE checkpoint or a production HTTP service |
| `data_audit.md`, `*_baselines_validation.json`, `baseline_environment.json`, `gpu_smoke_ellipse.json` | Preparation, simple baselines, and early environment/smoke evidence |

The transfer report's absolute checkpoint paths were made project-relative before
publication; its measurements and weight hashes were preserved. It contains
validation sensitivity probes, not the separate private official-sample sanity
check. No private official-sample texts or results are included here.

The latest MoE synthetic diagnostic MAE improved from 1.8788 to 0.8422, while
original-test MAE worsened from 1.1289 to 1.1968. The plain model was unchanged.
The original labels come from Writing9 and the synthetic labels from LLM judgments;
these results do not establish examiner accuracy.

Full model cards and evaluation artifacts are on
[Hugging Face](https://huggingface.co/saugataroy/ielts-task2-modernbert-base-moe).
See [the repository file policy](../docs/repository-files.md) for the exact classes
of mutable monitoring and local staging files excluded from Git.
