# IELTS Task 2 Synthetic Rubric Scores v1

## Dataset summary

An English, fully synthetic collection of **1,200 IELTS-style Task 2 responses** with
separate LLM assessments of Task Response (TR), Coherence and Cohesion (CC), Lexical
Resource (LR), and Grammatical Range and Accuracy (GRA). **1,194 responses have accepted
LLM labels; six uncertain cases have null targets and are excluded from the default
configuration. There are no human-verified labels.**

The dataset supports research on rubric regression, synthetic-data augmentation, and
failure analysis for short, irrelevant, incomplete, or inconsistent answers. It does
not establish agreement with IELTS examiners and is not an official IELTS dataset.
Task 1 Academic and General Training letters are outside its scope.

| Property | Value |
|---|---|
| Publisher | [saugataroy](https://huggingface.co/saugataroy) |
| Version | 1.0 |
| Language | English |
| Authorship | Individually composed by LLM agents |
| Annotation | Separate LLM judge sessions; selected cases adjudicated |
| Questions | 120 original questions, ten attempted responses each |
| Split groups | 105 groups of related question families |
| Human-verified scores | 0 |
| Format | Parquet data; JSONL annotation records |

## Uses

### Intended uses

- Supplement authentic learner writing when training or studying essay-scoring models.
- Inspect criterion-level errors and response-to-question relevance.
- Run controlled comparisons with and without synthetic augmentation.

Use an independently collected, human-rated holdout to assess real-world scoring
accuracy. Keep this collection's synthetic validation and diagnostic results separate
from human or website-label evaluation results.

### Out-of-scope uses

These labels should not determine official examination results, admissions, employment,
or other consequential decisions. This dataset is unsuitable as the sole evidence for
a claim of reliable IELTS grading. It does not cover handwriting, speaking, Task 1,
full IELTS Writing scores, or representative demographic and first-language groups.

## Dataset structure

| Configuration | Split | Rows | Purpose |
|---|---|---:|---|
| default | train | 976 | Training augmentation |
| default | synthetic_validation | 98 | Synthetic development evaluation |
| default | synthetic_diagnostic | 120 | Held-out synthetic diagnostics |
| quarantine | quarantine | 6 | Review only; all training targets null |

Related prompt families stay together. Initial assignments contained 980/100/120 rows;
six uncertain rows were removed from their assigned training or validation partitions.
Group isolation reduces prompt leakage but does not establish fully disjoint topics.
Do not randomly re-split individual essays from the same prompt.

### Data fields

| Field | Meaning |
|---|---|
| `id` | Opaque response identifier |
| `family_id`, `prompt_group_id` | Question family and conservative split group |
| `split`, `task` | Assigned partition and `task2` |
| `question`, `essay` | Original model input texts; learner errors preserved |
| `TR`, `CC`, `LR`, `GRA` | Accepted integer criterion labels, 1–9 in this release |
| `task_mean` | Unrounded arithmetic mean of the four labels |
| `topic`, `question_type` | Prompt metadata |
| `word_count` | Whitespace-delimited response word count |
| `flags` | LLM-assigned diagnostic flags; can overlap |
| `synthetic`, `human_verified`, `label_origin` | `true`, `false`, and `llm_judgment` |
| `input_sha256` | Frozen question/response digest used to bind assessments |
| `generator` | Agent-session provenance, not a learner identity |

The separate quarantine configuration uses `intended_split`, `status`, and `reason`;
its four criteria and `task_mean` are null. Never convert these missing labels to zero.

`annotations/<split>.jsonl` retains the original nested `scores`, individual ratings,
criterion evidence quotations and explanations, qualitative confidence, judge identity,
and any adjudication. These files preserve disagreements. The accepted target is an
explicit selected assessment or adjudication, not an average that conceals disagreement.
Confidence is not a calibrated probability.

`task_mean` is a research target summary, not a rounded official Task 2 band or the
overall Writing band. Train with **question and essay only** as textual inputs.
Scores, reasons, flags, filenames, authors, and question-type metadata can leak labels
or constructed profiles and should not be included in ordinary scoring-model inputs.

### Loading

```python
from datasets import load_dataset

dataset_id = "saugataroy/ielts-task2-synthetic-v1"
# Pin a repository commit SHA as revision for reproducible experiments.
data = load_dataset(dataset_id, "default", revision="main")
row = data["train"][0]
inputs = {"question": row["question"], "essay": row["essay"]}
targets = [row[name] for name in ["TR", "CC", "LR", "GRA"]]

# Explicit opt-in; do not add these rows to supervised training.
uncertain = load_dataset(dataset_id, "quarantine", split="quarantine")
```

## Dataset creation

### Source and generation

Questions and responses were authored for this collection, rather than scraped or
adapted from examination sample answers. The collection does not contain the private
reference essays used for overlap checks. Responses were individually composed;
no sentence bank, substitution template, or programmatic text expansion generated them.
Code serialized records and performed integrity checks.

Generation used Codex agent sessions. The tool did not expose the exact backend model
identifier or sampling parameters, so these are unknown. Agent identifiers and the
final author/judge/adjudicator instructions are included. This is not a reproducible
seeded LLM generation run: the split seed does not make generated prose reproducible.

Each question was used for ten deliberately varied attempts, including strong and weak
writing, partial task coverage, fluent unrelated answers, and short or inconsistent
responses. Intended ability profiles were authoring instructions, never ground-truth
labels. The distribution reflects this construction, not a sample of real candidates.

### Annotation process

1. Author a response and validate its structure and intended word bounds.
2. Freeze question and response with a hash and opaque identifier.
3. Give a separate judge only the question and response, hiding author identity,
   intended ability, and desired outcome.
4. Assess four criteria with exact textual evidence, reasoning, confidence, and flags.
5. Check numeric bounds, evidence substrings, immutable hashes, and reviewer separation.
6. Adjudicate conflicts and low-confidence cases. Retain unresolved cases separately.

The first 30 responses received two blind assessments. Across the full set, 134 current
adjudications were completed: 128 accepted and six unresolved. Separate sessions may
share the same underlying model; reviewer separation does **not** imply independent
model families or freedom from correlated biases. No IELTS examiner calibrated the labels.

The rubric reference is the [official public Writing Band Descriptors, updated May 2023](https://ielts.org/cdn/ielts-guides/ielts-writing-band-descriptors.pdf),
accessed 2026-10-10. Visual review of limiting descriptor features prompted explicit
reassessment of 126 earlier affected or uncertain cases. The annotation records retain
prior ratings. Difficult distinctions between partial, tangential, and wholly unrelated
answers especially require human calibration. The descriptor PDF is not redistributed.

### Integrity and overlap checks

All 1,200 authored records passed structural checks; no assessments or required
adjudications remain pending. Verified checks cover export checksums, current-review
equality, input/score separation, score-mean arithmetic, exact evidence quotations,
candidate coverage, and prompt-group isolation.

No exact or configured lexical near-duplicate essay flags were found within the set.
No exact or threshold five-word overlap was detected against 14,537 local reference
rows, and no full normalized question matched 1,400 distinct reference prompts.
The private reference corpus is not distributed, so that cross-corpus audit is not
independently reproducible from this repository alone. These checks cannot rule out
paraphrases, shared arguments, or pretraining exposure.

With the paired ModernBERT tokenizer used in the accompanying experiments, the longest
question/response pair is 434 tokens; none exceeds the 1,024-token input limit.
These are tokenizer-specific observations. File hashes are provided in `checksums.json`;
aggregate annotation statistics and the export receipt are in `audit/quality.json`.

## Distribution, bias, and limitations

Responses range from **12 to 337 words**, with median **274**. There are **248 attempts
below 250 words**, including **30 fragments of at most 20 words**. Short challenge
attempts are intentional: 1,200 responses does not mean 1,200 full-length essays.

The labels are strongly skewed toward proficient language. Among accepted rows,
**533 GRA labels are band 9 and only three are band 3**. Artificial errors may be less
natural or diverse than those of actual learners, and polished arguments can remain
underneath deliberately weakened prose. Topics, discourse patterns, and model style
can become shortcuts. Do not assume broad coverage of low-proficiency grammar.

| Criterion | Mean accepted label |
|---|---:|
| Task Response | 6.1273 |
| Coherence and Cohesion | 6.6231 |
| Lexical Resource | 6.5603 |
| Grammatical Range and Accuracy | 7.5620 |

LLM judgments can misapply rubric descriptors or inflate scores. Agreement and passing
integrity checks do not establish score accuracy. Repeated use of these public synthetic
splits for development will further limit their value as independent evaluation.

No real participant recruitment, consent process, or demographic annotations were
involved. The texts were constructed without intentionally including personal records.
Synthetic data can nevertheless reproduce social stereotypes or coincidental names;
no comprehensive human privacy or fairness audit has been performed.

## License

{{LICENSE}}

IELTS and related marks belong to their respective owners. The publisher is not
affiliated with or endorsed by IELTS, Cambridge, British Council, or IDP.

## Citation and contact

```bibtex
@misc{saugataroy2026ieltssynthetic,
  author = {{saugataroy}},
  title = {IELTS Task 2 Synthetic Rubric Scores v1},
  year = {2026},
  url = {https://huggingface.co/datasets/saugataroy/ielts-task2-synthetic-v1},
  note = {Version 1.0. Synthetic responses and LLM judgments; no human-verified labels}
}
```

Include the repository commit SHA with reported experiments. No DOI is assigned in
this release. Report annotation errors or dataset questions through the repository's
Community tab, quoting the response ID.
