# IELTS Writing Scoring

Training and evaluation source for two ModernBERT-base models that estimate IELTS
Writing Task 2 criteria from a question and essay. Both use an encoder trained on
ELLIPSE and then fine-tuned on IELTS-style responses. The MoE model additionally
received synthetic-data continuation.

| Artifact | Location |
|---|---|
| Plain MLP model, card, and evaluation | [Hugging Face](https://huggingface.co/saugataroy/ielts-task2-modernbert-base-mlp) |
| Dense multi-gate MoE model, card, and evaluation | [Hugging Face](https://huggingface.co/saugataroy/ielts-task2-modernbert-base-moe) |
| Synthetic dataset and dataset card | [Hugging Face](https://huggingface.co/datasets/saugataroy/ielts-task2-synthetic-v1) |
| Training instructions and reproduction boundaries | [docs/training.md](docs/training.md) |
| Recorded training source hashes | [docs/training_sources.json](docs/training_sources.json) |
| BentoML inference and benchmarking project | [ielts-inference-api](https://github.com/saugataroyarghya/ielts-inference-api) |

## Input and output

Input is the original question and a nonempty typed response, tokenized as a pair
with a maximum of 1,024 tokens. The encoder sees both; pooling averages the
contextualized essay tokens.

Both heads return four continuous estimates: **Task Response (TR), Coherence and
Cohesion (CC), Lexical Resource (LR), and Grammatical Range and Accuracy (GRA)**.
The inference wrapper also returns their unrounded mean and a truncation flag.
Outputs are not clipped or rounded. The mean is not an official IELTS Writing band.
Task 1, generated feedback, and examiner-calibrated confidence are unsupported.

The MLP head is a two-layer regression network. The MoE head has four shared
experts and a separate soft gate for each criterion; all experts execute for every
input. Architecture details and runnable inference examples are in the model cards.

## Training source

- `src/ielts_scoring/data.py`: source loading, light cleaning, duplicate grouping, and splits.
- `src/ielts_scoring/models.py`: encoder, pooling, MLP and MoE heads, checkpoint loading.
- `src/ielts_scoring/train.py`: ELLIPSE training and IELTS encoder transfer.
- `src/ielts_scoring/synthetic_finetune.py`: pinned-parent MoE continuation and evaluation.
- `src/ielts_scoring/evaluate.py`, `finalize.py`, and `transfer.py`: metrics, model selection, and paired comparisons.
- `src/ielts_scoring/benchmark.py`: direct inference timings; excludes server queueing and network transport.
- `configs/`, `notebooks/`, and `scripts/build*.py`: frozen configurations and Kaggle orchestration.

Verify that the published source matches the archives used for the completed
transfer and synthetic-continuation runs:

```bash
python scripts/verify_training_sources.py
```

See [training instructions](docs/training.md) for environment setup, commands,
private-data prerequisites, and exact-reproduction limitations. The existing
`PLAN.md`, earlier notebooks, and reports are historical experiment records;
the linked model cards describe the published checkpoints.

## Evaluation scope

The synthetic continuation reduced diagnostic macro MAE from **1.8788 to 0.8422**
on 120 LLM-labeled responses, while original-test macro MAE increased from
**1.1289 to 1.1968** on 450 website-labeled responses. The plain MLP was unchanged.
These results show a tradeoff between label sources, not proven agreement with
qualified IELTS examiners. See the MoE card for full metrics and uncertainty.

## Data, artifacts, and licensing

Model weights and the public synthetic dataset are hosted on Hugging Face.
Original prepared datasets, per-response evaluation outputs, credentials, and
private official-sample sanity checks are excluded from this repository.

The synthetic dataset is CC BY 4.0. The model repositories contain separate
licensing-status notices. No code license has been selected; the dataset license
does not apply automatically to this repository or the model weights.
