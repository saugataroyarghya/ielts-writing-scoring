# Light preparation report — 2026-10-09

Preparation ran successfully on Kaggle in the first version of the [private preparation notebook](https://www.kaggle.com/code/saugataroyarghya/essay-scoring-data-preparation). Dataset revision identifiers and SHA256 checksums are saved in `configs/` and the dataset's `metadata.json`.

| Track | Source rows | Retained | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: | ---: |
| ELLIPSE | 6,482 | 6,482 | 3,324 | 587 | 2,571 |
| IELTS Task 2 | 8,049 | 8,048 | 7,069 | 529 | 450 |

ELLIPSE retained all rows. Its six averaged human-rated criteria stay on the original 1–5 scale. The authors' test partition is preserved; 15% of the supplied training portion was assigned to validation by essay group. No writer ID is available in the selected schema. Prompt independence is not claimed for this track.

IELTS excluded one exact duplicate with matching text, prompt, and criterion labels. All other valid examples were retained, including short essays and overall/subscore disagreement. The `task_achievement_band` field is mapped to `TR` because the selected corpus is described as Task 2; this records the source field, not a relabeling of its score.

Automated grouping found 236 lexical prompt groups, which merged into 227 split groups after essay-duplicate links. This differs from the uploader's reported 128 prompts; the lexical grouping does not prove semantic prompt independence. The deterministic group split produced 87.8% / 6.6% / 5.6% of rows. The uneven proportions were accepted to preserve group boundaries. Train/validation/test have no overlapping normalized essay hashes or identified split groups.

847 IELTS records have an absolute difference greater than 0.5 between source overall and mean criterion score. These values were not repaired, removed, or substituted. Training predicts the four supplied criterion labels; overall is metadata only.

Cleaning only normalizes transport line endings, strips outer whitespace, and removes NUL characters. It does not rewrite spelling, grammar, casing, or internal paragraph breaks. Feedback and source scores are excluded from encoder inputs. No manual annotation campaign was performed, following the user's training-first scope.

The lexical duplicate check uses character TF-IDF cosine similarity >=0.94 for essays and >=0.88 for prompts. This catches close lexical variants, not every semantic paraphrase. Full distributions and file hashes are in each private dataset's `metadata.json`; exclusions are in `quarantine.csv`.
