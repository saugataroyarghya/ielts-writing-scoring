# Writing Score Prediction: Training and Hugging Face Release Plan

Created: 2026-10-09

**Goal:** Learn automated essay scoring by training reproducible models, comparing their results, and publishing usable checkpoints on Hugging Face.

**Project directory:** `~/Codebase/personal/learning/ielts`

**Current status:** Implementation started on 2026-10-09. Both datasets were prepared in a private Kaggle CPU notebook and published as private version-1 Kaggle datasets. ELLIPSE has 3,324/587/2,571 train/validation/test rows; IELTS has 7,069/529/450. The project implements ModernBERT with plain and MoE regression heads; local correctness tests passed. GPU smoke testing is next. See `reports/run_status.json` for current remote run states and `reports/data_audit.md` for the light preparation report. No Hugging Face model has been published.

**Agreed scope update:** Focus on model training with light automated data cleaning. Use Kaggle GPU notebooks, ModernBERT-base first, a four-expert multi-gate MoE comparison, and optional Ettin-150m follow-up. DistilBERT is no longer required. The two tracks remain separate. Prepared outputs were briefly staged locally to create private Kaggle Dataset versions; raw source downloads occurred in the remote preparation job.

## 1. What we will build

Work toward two distinct releases:

1. **English writing proficiency model:** Train on ELLIPSE to predict its six analytic writing scores. This provides practice with human-rated data and a reproducible evaluation.
2. **IELTS Task 2 practice model:** Train on question–essay pairs to estimate Task Response (TR), Coherence and Cohesion (CC), Lexical Resource (LR), and Grammatical Range and Accuracy (GRA).

Keep the datasets, targets, metrics, and release names separate. ELLIPSE scores are not IELTS bands and must not be converted through simple numerical rescaling.

For the IELTS model, return four continuous estimates and their mean. A rounded half-band value can be a display option with a documented rounding rule. Call this a **Task 2 estimate**, not the complete IELTS Writing result, which also requires Task 1. If the source's overall score disagrees with its four subscores, audit the discrepancy; do not silently rewrite the labels.

Initial scope is English, typed essays, and numerical scoring. Add Academic Task 1, handwriting recognition, generated feedback, and multilingual support only after the first releases are evaluated. The model must receive the actual question to assess Task Response.

## 2. Dataset decisions

| Source | Reported size and labels | Planned use | Conditions |
| --- | --- | --- | --- |
| [ELLIPSE](https://github.com/scrosseye/ELLIPSE-Corpus) | Approximately 6,500 essays; human-rated overall proficiency and six analytic traits | First training and evaluation track | Use the final reliability-filtered corpus. Repository lists CC BY-NC-SA 4.0. |
| [chillies/ielts-writing-task2-essays](https://huggingface.co/datasets/chillies/ielts-writing-task2-essays) | 8,049 rows; 128 prompts; overall and four criterion scores | First IELTS experiment | Writing9 source scores; treat as weak labels. Uploader lists CC BY 4.0, but original-source permissions need checking before public weight release. |
| [EWCCE-DATA](https://data.mendeley.com/datasets/cd874w6g5k/1) | 200 scanned essays; IELTS-style criterion scores assigned by two English instructors, according to the authors | Candidate independent human-rated evaluation set | CC BY 4.0. First verify downloadable contents, original prompts, task types, score mapping, and transcription feasibility. Instructors are not established as official IELTS examiners. |
| [ndtran0101/writing9-ielts-essays](https://huggingface.co/datasets/ndtran0101/writing9-ielts-essays) | 166,075 rows; overall and criterion scores | Optional later experiment on scaling noisy data | Unknown licence; author warns about score quality. Check task mixing and overlap with the smaller Writing9 dataset. |
| [ASAP 2.0](https://github.com/scrosseye/ASAP_2.0) | Approximately 25,000 source-based school essays; human quality scores | Optional general essay-scoring track | CC BY 4.0; requires its own rubric and source material. Never present it as IELTS scoring. |

The older [10,324-row IELTS dataset](https://huggingface.co/datasets/chillies/IELTS-writing-task-2-evaluation) is not the default. Its owner clarified that bands came from tutors or experienced learners and feedback was model-generated. Criterion scores embedded in that feedback are not verified human annotations. See the [owner's clarification](https://huggingface.co/datasets/chillies/IELTS-writing-task-2-evaluation/discussions/7).

Use [official IELTS scored examples](https://ielts.org/take-a-test/preparation-resources/sample-test-questions/academic-test) for qualitative sanity checks. Their small number and possible presence in pretrained models' training data make them unsuitable as the main benchmark.

**Publication decision:** Record the original source, uploader's licence, and allowed uses separately. Keep uncertain-source experiments local until reuse and weight-release conditions are established. Do not assume that an Apache-licensed base model makes its fine-tuned weights unrestricted. Keep code licensing separate from data and weight licensing. ELLIPSE's noncommercial/share-alike terms must be considered explicitly. If an IELTS source cannot support public release, publish the methodology and permitted models while resolving that source; do not rename a general writing model as IELTS-trained.

## 3. Project structure

Create these during implementation; this plan does not imply they already exist.

| Path | Purpose |
| --- | --- |
| `README.md` | Project overview, results, setup, and links to released models |
| `pyproject.toml` and lockfile | Reproducible dependencies |
| `configs/` | Dataset, split, model, and training configurations |
| `src/ielts_scoring/data.py` | Download adapters, cleaning, validation, and splits |
| `src/ielts_scoring/models.py` | Baselines and regression models |
| `src/ielts_scoring/train.py` | Training entry point |
| `src/ielts_scoring/evaluate.py` | Metrics, slices, and prediction exports |
| `src/ielts_scoring/predict.py` | Stable inference interface |
| `notebooks/` | Exploration and a Colab training launcher |
| `data/raw/`, `data/processed/` | Local data; excluded from Git by default |
| `artifacts/splits/` | Frozen record IDs, group IDs, revisions, and hashes |
| `artifacts/runs/` | Configs, logs, checkpoints, and predictions |
| `reports/` | Data audit, comparison table, and error analysis |
| `release/` | Selected checkpoint, tokenizer, metadata, and model card |
| `tests/` | Small tests for leakage checks, loss shapes, and save/load parity |

Suggested stack: Python, PyTorch, Transformers, Datasets, scikit-learn, pandas, and huggingface_hub. Use local JSON/CSV logs and optionally TensorBoard. Pin tested versions after the first successful smoke run. Record the code commit, dataset revision, seed, and package versions for every experiment.

## 4. Phase A — Prepare and audit the data

- [ ] Download exact, pinned dataset revisions and retain source checksums.
- [ ] Inspect the actual ELLIPSE columns, rubric, score ranges, and supplied splits; do not assume a Kaggle mirror has the same schema.
- [ ] Create a common record format with `id`, `source`, `source_revision`, `task_type`, `prompt`, `prompt_group`, `essay`, `labels`, and `label_provenance`. Preserve writer IDs only where available and useful for splitting.
- [ ] For IELTS, map source fields to `TR`, `CC`, `LR`, `GRA`, and optional `overall`. Document why the field named `task_achievement_band` is mapped to TR for Task 2.
- [ ] Validate numeric scores, missing labels, empty text, duplicate IDs, text lengths, and mismatched question–essay pairs. Keep invalid rows and reasons in a quarantine manifest.
- [ ] Preserve spelling, grammar, casing, and paragraph breaks. Clean transport artefacts such as HTML and line endings without rewriting the essay.
- [ ] Keep genuinely weak, short, or off-topic attempts when their labels are usable. A blanket 250-word filter would remove part of the scoring problem.
- [ ] Exclude feedback, existing scores, and source identifiers from model inputs. They are labels or metadata, not evidence available at prediction time.
- [ ] Audit overall/subscore disagreement and the distribution of each score. Missing labels remain missing; never infer four human scores from one overall label.
- [ ] Keep manual review optional for now; prioritize model training after automated validity, duplicate, and split checks.

**Deliverable:** `reports/data_audit.md` with retained counts, exclusions, label provenance, lengths, score distributions, and a dataset choice for each track.

## 5. Phase B — Freeze evaluation splits

**ELLIPSE:** Preserve the supplied test split. Create validation data from the supplied training portion, grouping repeated writers where IDs exist. Check duplicate essays across all partitions. Document any repairs. Do not claim unseen-prompt performance unless prompt separation is actually verified.

**IELTS:** Target an approximately 80/10/10 train/validation/test division by prompt family. Cluster minor wording variants of the same question together. Group exact and near-duplicate essays, including rewritten versions, before splitting. Related prompts or duplicate essays must not cross partitions; merge conflicting groups when necessary. Accept imperfect percentages to preserve independence.

Try to retain useful score coverage without breaking group boundaries. Record band counts and any unsupported tails. Freeze the split manifest before training. Fit TF-IDF vocabularies, scaling, feature selection, and calibration using training/validation data only.

EWCCE can serve as an external check only after its prompts and tasks are verified as compatible. If prompts are missing, it cannot validate TR properly. Preserve handwriting errors during transcription and report OCR/transcription uncertainty. Exclude overlap with training data across sources. Do not tune on this external set.

**Deliverable:** Fixed manifests and automated checks for duplicate, writer, and prompt leakage where those fields are available.

## 6. Phase C — Train models in increasing complexity

| Experiment | Model | Learning objective |
| --- | --- | --- |
| B0 | Training-set mean and median score per target | Establish minimum baselines for squared and absolute error |
| B1 | Length and simple surface features with Ridge | Detect whether length alone explains apparent performance |
| B2 | Word/character TF-IDF plus Ridge | Establish a competitive CPU baseline |
| N1 | [ModernBERT base](https://huggingface.co/answerdotai/ModernBERT-base), 149M parameters, plain MLP head | Establish the main encoder fine-tuning baseline |
| N2 | Same ModernBERT, four shared MLP experts and one gate per target | Test whether multi-gate routing improves scoring |
| N3/N4, optional | [Ettin-150m](https://huggingface.co/jhu-clsp/ettin-encoder-150m), same plain/MoE heads | Compare the starting checkpoint independently of the head architecture |
| T1, optional | ELLIPSE-trained encoder, new IELTS head, then IELTS fine-tuning | Test whether proficiency training improves IELTS adaptation |

Run B0–B2 and N1/N2 on ELLIPSE first. Reuse the pipeline for the IELTS track. Compare plain and MoE heads on identical frozen splits and seeds. The MoE uses four width-256 experts, soft routing, and one criterion gate; the plain width-1,376 MLP has a comparable parameter count. Fine-tune the encoder and head jointly. Compare T1 against an identical base encoder trained directly on the same IELTS split; transferring six ELLIPSE scores into four IELTS scores is not a valid label mapping.

### Inputs and outputs

- ELLIPSE: essay text to six analytic scores, using the verified label names and original scale. Overall proficiency is optional and remains a separate target.
- IELTS: question and essay, separated consistently, to four criterion estimates. Keep the source's overall label for a separate optional one-output experiment if the subscore audit fails.
- Use a pretrained encoder with an explicit custom regression head/loss and floating-point labels. Multiple outputs must not accidentally activate classification or multilabel BCE loss. Return raw regression outputs; never softmax the criterion scores. Softmax is used only for the MoE gates.
- Start with mean squared error; compare Smooth L1 only if outliers justify it. If labels are missing, use an explicit mask and normalise over observed targets. Report errors in the original score units.
- Start ModernBERT and Ettin at 1,024 tokens; consider 2,048 only if measured truncation supports it. Preserve the complete IELTS question and truncate only the essay when needed. Report truncation and compare on the same held-out essays and a fully covered subset.

### Starting training settings

These are initial experiment choices, not promised optimal settings:

| Setting | Starting value |
| --- | --- |
| Optimiser | AdamW |
| Learning rate | Encoder `2e-5`, new head `1e-4`; tune only after the baseline |
| Epoch cap | 5, selecting the best validation checkpoint |
| Effective batch size | 16, initially physical batch 2 with 8 accumulation steps |
| Weight decay / warmup | `0.01` / 10% of training steps |
| Selection metric | Mean per-target validation MAE |
| Seeds | One during debugging; three for the final model comparison |
| Precision | fp16 on compatible CUDA hardware; bf16 only when supported; fp32 fallback |

Use CPU for data preparation and classical models. Prefer an available Colab/Kaggle GPU for neural training; actual memory needs depend on sequence length and batch size. Local Apple MPS is an optional path after a small compatibility run. Do not make Flash Attention mandatory. Record peak memory, elapsed time, and hardware rather than guessing runtime costs.

Before a full run, verify loss/output shapes, overfit a tiny sample, save/reload a checkpoint, and benchmark warmed-up GPU batches. These checks catch broken training plumbing and produce measured runtime estimates. Record per-criterion gate use and entropy to identify expert collapse.

## 7. Phase D — Evaluate and diagnose

Report the following per criterion and as a macro average where appropriate:

- **MAE:** main selection metric, in the dataset's native score units.
- **RMSE:** sensitivity to larger mistakes.
- **Quadratic weighted kappa:** agreement on the dataset's documented discrete score grid. Document the prediction rounding/binning rule and use it consistently. Do not round averaged human targets arbitrarily.
- **Within 0.5 and within 1.0 band:** IELTS only; show counts and denominators.
- **Signed error:** identify systematic over- or under-scoring.

Show errors by score band, prompt/topic, essay length, and data source. Analyse truncation explicitly. Compare against B0, B1, and B2; correlation alone is not enough. For finalists, report variation across the three seeds and, where feasible, confidence intervals that respect the grouping of the test data.

Check a small set of controlled cases: an unrelated essay attached to a question, duplicated paragraphs, shuffled paragraph order, an empty answer, and a short answer. Use these as diagnostics, not invented human-scored test data. Do not impose an unsupported fixed penalty for a short response.

Inspect approximately 30 of the largest errors. Distinguish model failures from dubious source labels. Record observed limitations instead of retroactively cleaning the test set to improve results.

Select models with validation data, then evaluate the frozen test set. Further tuning after seeing test results requires a new untouched holdout. Official example essays remain qualitative checks because they may have been seen in pretraining.

**Success criterion:** A reproducible model with clearly reported results and limitations. Prefer a neural model only when its accuracy/latency tradeoff justifies it; honest negative results are still publishable learning work. Agreement with Writing9 labels alone does not establish agreement with IELTS examiners.

## 8. Phase E — Package and publish on Hugging Face

Use a placeholder until the actual Hugging Face account is known. A GitHub username is not proof of the Hugging Face namespace.

Possible repositories:

- `<hf_username>/ellipse-modernbert-base-writing-scorer`
- `<hf_username>/ellipse-modernbert-base-moe-writing-scorer`
- `<hf_username>/ielts-task2-modernbert-base-practice-scorer`

Publish the selected checkpoints, with baseline results in their model cards. An optional comparison Space can follow after local inference is reliable.

### Release bundle

- Weights in `safetensors`, model configuration, and tokenizer files.
- Explicit score names and order, label ranges, input formatting, length policy, and any scaling/calibration parameters.
- A minimal `predict(question, essay)` example for IELTS and the corresponding essay-only interface for ELLIPSE.
- Tested dependency versions and a CPU inference example.
- Model card with dataset provenance, licence decisions, split method, training settings, baseline comparisons, per-trait metrics, and limitations.
- Links to training code, configs, and reproducible split identifiers where sharing them is permitted. Keep raw essays and per-example text out of the release unless redistribution is permitted.

Follow the [Hugging Face model-card guide](https://huggingface.co/docs/hub/model-cards). Set `library_name`, `base_model`, language, relevant tags, and the actual weight licence. Use a Hub task tag only if its behaviour matches the multi-output regression interface; a default classification widget must not turn regression outputs into probabilities.

### Publication sequence

1. Review source permissions and document which datasets trained this exact checkpoint. Resolve the release licence before public upload.
2. Authenticate locally through `hf auth login` or an interactive notebook login. Keep tokens out of notebooks, source files, and logs.
3. Create a private staging model repository and upload only the curated `release/` directory using `huggingface_hub`. Follow the current [upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload).
4. Download the staged model into a clean environment and reproduce predictions, score order, and reported evaluation. Avoid reliance on unpublished local code or file paths.
5. Check the model card and release contents, make the repository public, tag the release, and link it from the code repository.
6. Optionally add a small Gradio Space with question/essay inputs and four score outputs. Clearly label estimates and avoid retaining visitors' essays by default.

A model trained on website scores should say so prominently. Public availability and a polished demo do not establish examination-level validity. Keep the general ELLIPSE models named and described as general writing models.

## 9. Suggested milestones

This is a sequence for spare-time work, not a fixed deadline or a scheduled automation.

| Milestone | Work | Completion evidence |
| --- | --- | --- |
| 1: Data and baselines | Environment, provenance, audits, frozen splits, B0–B2 | Audit report, split manifest, baseline table |
| 2: First trained model | ELLIPSE N1, regression tests, error analysis | Reloadable checkpoint and reproducible evaluation |
| 3: IELTS experiments | Audit labels, train N1/N2, compare direct training with optional transfer | Task 2 results with label-quality and external-evaluation limitations |
| 4: Public release | Resolve licences, select checkpoints, build cards, verify fresh downloads | Public Hugging Face model links and reproducible code |

If time is limited, complete one good ELLIPSE model and one IELTS experiment before scaling data or adding feedback generation. A large model sweep is optional.

## 10. First implementation session

- [x] Initialise the local project and Git repository if one does not already exist.
- [x] Create the environment and ignore raw data, checkpoints, caches, and credentials.
- [x] Implement remote ELLIPSE/IELTS preparation and inspect the source schema.
- [x] Produce a light preparation report and freeze private Kaggle dataset versions.
- [x] Train the mean, median, length, and TF-IDF baselines (validation results in `reports/`).
- [x] Verify ModernBERT plain/MoE GPU smoke tests before full training (both passed on a T4).

## References

- [IELTS Writing band descriptors](https://ielts.org/cdn/ielts-guides/ielts-writing-band-descriptors.pdf)
- [IELTS Writing assessment criteria](https://ielts.org/cdn/Guides/ielts-writing-key-assessment-criteria.pdf)
- Dataset and model sources linked in Sections 2 and 6.
- [Transformers ModernBERT documentation](https://huggingface.co/docs/transformers/model_doc/modernbert)
- [Hugging Face CLI documentation](https://huggingface.co/docs/huggingface_hub/guides/cli)

Dataset counts above come from the completed preparation job. Model performance and GPU runtime remain unmeasured until the corresponding runs complete. Training settings are initial experimental choices.
