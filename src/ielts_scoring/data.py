"""Small automated preparation pass; text downloads happen in the Kaggle job."""

import argparse
import hashlib
import io
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors

TARGETS = {
    "ellipse": ["Cohesion", "Syntax", "Vocabulary", "Phraseology", "Grammar", "Conventions"],
    "ielts": ["TR", "CC", "LR", "GRA"],
}
IELTS_FIELDS = {
    "task_achievement_band": "TR", "coherence_cohesion_band": "CC",
    "lexical_resource_band": "LR", "grammatical_range_band": "GRA",
}


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def normalize_group(text):
    """Only used for matching, never substituted for model input."""
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.sub(r"[^\w]+", " ", text).strip()


def clean_text(value):
    if not isinstance(value, str):
        return ""
    return value.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").strip()


class UnionFind:
    def __init__(self, n):
        self.parents = list(range(n))

    def find(self, i):
        while self.parents[i] != i:
            self.parents[i] = self.parents[self.parents[i]]
            i = self.parents[i]
        return i

    def union(self, i, j):
        a, b = self.find(i), self.find(j)
        self.parents[max(a, b)] = min(a, b)


def similarity_groups(texts, threshold=0.94):
    """Group conservative lexical near-duplicates without changing their labels."""
    texts = [normalize_group(x) for x in texts]
    uf = UnionFind(len(texts))
    seen = {}
    for i, text in enumerate(texts):
        if text in seen:
            uf.union(i, seen[text])
        seen[text] = i
    if len(texts) > 1 and any(texts):
        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=1,
            max_features=60000, dtype=np.float32,
        )
        matrix = vectorizer.fit_transform(texts)
        nn = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=2).fit(matrix)
        for start in range(0, len(texts), 128):
            neighbours = nn.radius_neighbors(
                matrix[start:start + 128], radius=1 - threshold, return_distance=False,
            )
            for offset, matches in enumerate(neighbours):
                i = start + offset
                for j in matches:
                    if int(j) > i:
                        uf.union(i, int(j))
    return [uf.find(i) for i in range(len(texts))]


def merge_groups(*group_columns):
    uf = UnionFind(len(group_columns[0]))
    for column in group_columns:
        seen = {}
        for i, group in enumerate(column):
            if group in seen:
                uf.union(i, seen[group])
            seen[group] = i
    return [uf.find(i) for i in range(len(group_columns[0]))]


def grouped_split(frame, seed=42, supplied_test=False):
    frame = frame.copy()
    if supplied_test:
        frame["split"] = np.where(frame["original_split"].eq("test"), "test", "train")
        pool = frame.index[frame["split"].eq("train")].to_numpy()
        _, val = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed).split(
            pool, groups=frame.loc[pool, "group_id"],
        ))
        frame.loc[pool[val], "split"] = "validation"
    else:
        if frame["group_id"].nunique() < 5:
            raise ValueError("Not enough independent groups for train/validation/test.")
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        _, held = next(splitter.split(frame, groups=frame["group_id"]))
        val, test = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed).split(
            held, groups=frame.iloc[held]["group_id"],
        ))
        frame["split"] = "train"
        frame.iloc[held[val], frame.columns.get_loc("split")] = "validation"
        frame.iloc[held[test], frame.columns.get_loc("split")] = "test"
    assert_split_integrity(frame)
    return frame


def assert_split_integrity(frame):
    if frame["id"].duplicated().any():
        raise ValueError("Duplicate record IDs.")
    for column in ["group_id", "essay_hash"]:
        if frame.groupby(column)["split"].nunique().gt(1).any():
            raise ValueError(f"Leakage across splits: {column}")
    if frame["source"].eq("ielts").all() and frame.groupby("prompt_group")["split"].nunique().gt(1).any():
        raise ValueError("Prompt-family leakage.")
    if set(frame["split"]) != {"train", "validation", "test"}:
        raise ValueError("Missing a required partition.")


def download(url, raw_dir, filename):
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    content = response.content
    (raw_dir / filename).write_bytes(content)
    return content, {"url": url, "file": filename, "sha256": sha256(content)}


def fetch_track(track, source, raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    revision = source["revision"]
    evidence = []
    if track == "ellipse":
        import pyzipper

        base = f"https://raw.githubusercontent.com/scrosseye/ELLIPSE-Corpus/{revision}/"
        content, record = download(base + "ELLIPSE_Final_github_train.csv", raw_dir, "train.csv")
        evidence.append(record)
        train = pd.read_csv(io.BytesIO(content))
        train["original_split"] = "train"
        content, record = download(base + "ELLIPSE_Final_github_test.zip", raw_dir, "test.zip")
        evidence.append(record)
        with pyzipper.AESZipFile(io.BytesIO(content)) as archive:
            archive.setpassword(b"ellipse_test")  # Public corpus password documented by its authors.
            names = [n for n in archive.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX")]
            if len(names) != 1:
                raise ValueError(f"Unexpected ELLIPSE test archive: {names}")
            test = pd.read_csv(io.BytesIO(archive.read(names[0])))
        test["original_split"] = "test"
        frame = pd.concat([test, train], ignore_index=True)  # Prefer official test rows on overlap.
        frame = frame.rename(columns={"text_id_kaggle": "id", "full_text": "essay", "Overall": "overall"})
        frame["label_provenance"] = "ELLIPSE reliability-filtered averaged human ratings"
    else:
        url = f"{source['source']}/resolve/{revision}/data/train-00000-of-00001.parquet"
        content, record = download(url, raw_dir, "source.parquet")
        evidence.append(record)
        frame = pd.read_parquet(io.BytesIO(content)).rename(columns={
            "essay_id": "id", "question": "prompt", "essay_text": "essay", "overall_band": "overall",
            **IELTS_FIELDS,
        })
        frame["original_split"] = "unsplit"
        frame["label_provenance"] = "Writing9 website scores; weak labels, examiner agreement unverified"
    frame["source"] = track
    frame["source_revision"] = revision
    frame["task_type"] = "IELTS Task 2" if track == "ielts" else "ELLIPSE writing"
    return frame, evidence


def prepare_track(track, source, output_root, seed=42):
    raw_dir = output_root / "raw" / track
    frame, evidence = fetch_track(track, source, raw_dir)
    source_count = len(frame)
    frame["id"] = frame["id"].astype("string")
    frame["essay"] = frame["essay"].map(clean_text)
    frame["prompt"] = frame["prompt"].map(clean_text)
    targets = TARGETS[track]
    low, high = (1, 5) if track == "ellipse" else (0, 9)
    for target in targets:
        frame[target] = pd.to_numeric(frame[target], errors="coerce")
    valid_labels = frame[targets].isna() | frame[targets].ge(low) & frame[targets].le(high)
    reasons = pd.Series("", index=frame.index)
    reasons.loc[frame["id"].isna() | frame["id"].eq("")] = "missing_id"
    reasons.loc[frame["essay"].eq("")] = "empty_essay"
    if track == "ielts":
        reasons.loc[frame["prompt"].eq("")] = "missing_prompt"
    reasons.loc[frame[targets].isna().all(axis=1)] = "all_targets_missing"
    reasons.loc[~valid_labels.all(axis=1)] = "invalid_target_range"
    reasons.loc[frame["id"].duplicated(keep="first")] = "duplicate_id"
    quarantine = frame.loc[reasons.ne(""), ["id", "source"]].assign(reason=reasons[reasons.ne("")])
    frame = frame.loc[reasons.eq("")].copy().reset_index(drop=True)
    frame["essay_hash"] = frame["essay"].map(lambda x: sha256(normalize_group(x).encode()))
    # Remove only identical essay/label/prompt rows. Keep conflicting annotations grouped.
    duplicated = frame.duplicated(subset=["essay_hash", "prompt", *targets], keep="first")
    quarantine = pd.concat([quarantine, frame.loc[duplicated, ["id", "source"]].assign(reason="exact_duplicate")])
    frame = frame.loc[~duplicated].reset_index(drop=True)
    print(f"{track}: grouping {len(frame)} essays", flush=True)
    essay_groups = similarity_groups(frame["essay"], threshold=0.94)
    frame["essay_group"] = essay_groups
    if track == "ielts":
        prompts = frame["prompt"].unique().tolist()
        prompt_groups = similarity_groups(prompts, threshold=0.88)
        mapping = dict(zip(prompts, prompt_groups))
        frame["prompt_group"] = frame["prompt"].map(mapping)
        groups = merge_groups(essay_groups, frame["prompt_group"].tolist())
    else:
        frame["prompt_group"] = frame["prompt"].map(normalize_group)
        groups = essay_groups
    frame["group_id"] = [f"{track}-{group:06d}" for group in groups]
    if track == "ellipse":
        test_groups = set(frame.loc[frame["original_split"].eq("test"), "group_id"])
        overlap = frame["original_split"].eq("train") & frame["group_id"].isin(test_groups)
        quarantine = pd.concat([quarantine, frame.loc[overlap, ["id", "source"]].assign(reason="overlaps_official_test")])
        frame = frame.loc[~overlap].reset_index(drop=True)
    frame = grouped_split(frame, seed=seed, supplied_test=track == "ellipse")
    keep = ["id", "source", "source_revision", "task_type", "prompt", "prompt_group", "essay",
            "essay_hash", "group_id", "label_provenance", "original_split", "split", *targets, "overall"]
    frame = frame[keep]
    target_dir = output_root / "prepared" / track
    target_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split in ("train", "validation", "test"):
        path = target_dir / f"{split}.parquet"
        frame.loc[frame["split"].eq(split)].sort_values("id").to_parquet(path, index=False)
        hashes[path.name] = sha256(path.read_bytes())
    frame[["id", "source", "group_id", "prompt_group", "essay_hash", "split"]].to_csv(
        target_dir / "split_manifest.csv", index=False,
    )
    quarantine.to_csv(target_dir / "quarantine.csv", index=False)
    discrepancy = (frame[targets].mean(axis=1) - pd.to_numeric(frame["overall"], errors="coerce")).abs()
    metadata = {
        "track": track, "targets": targets, "label_range": [low, high], "seed": seed,
        "source": source, "source_files": evidence, "source_rows": source_count,
        "retained_rows": len(frame), "excluded_by_reason": quarantine["reason"].value_counts().to_dict(),
        "splits": frame["split"].value_counts().to_dict(), "independent_groups": frame["group_id"].nunique(),
        "prompt_groups": frame["prompt_group"].nunique(), "prepared_sha256": hashes,
        "targets_by_split": {s: frame.loc[frame["split"].eq(s), targets].describe().to_dict()
                             for s in ("train", "validation", "test")},
        "overall_diff_gt_half": int(discrepancy.gt(0.5).sum()) if track == "ielts" else None,
        "word_counts": frame["essay"].str.split().str.len().describe().to_dict(),
        "split_policy": "Preserved supplied test; grouped 15% training validation" if track == "ellipse"
                        else "80/10/10 by merged prompt families and lexical duplicate groups",
        "limitations": [
            "Lexical near-duplicate grouping is conservative; semantic paraphrase independence is not proven.",
            "Writer IDs are unavailable in the selected public schema; writer independence is unverified.",
            "No manual relabeling, spelling correction, or minimum essay length filter.",
        ],
    }
    (target_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    (target_dir / "README.md").write_text(
        f"# {track.upper()} writing-scoring splits\n\n"
        f"Source: {source['source']}\n\nRevision: `{source['revision']}`\n\n"
        f"Source license: {source['license']}\n\n"
        "Private experiment cache. Preserve source attribution and original license terms.\n\n"
        f"Targets: {', '.join(targets)}. Text and label scales are preserved.\n\n"
        "See metadata.json for counts, checksums, exclusions, and split limitations.\n"
    )
    print(json.dumps({k: metadata[k] for k in ["track", "source_rows", "retained_rows", "splits", "excluded_by_reason"]}), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="configs/sources.json")
    parser.add_argument("--output", default="/kaggle/working/essay-data")
    parser.add_argument("--tracks", nargs="+", default=["ellipse", "ielts"])
    args = parser.parse_args()
    sources = json.loads(Path(args.sources).read_text())
    for track in args.tracks:
        prepare_track(track, sources[track], Path(args.output))


if __name__ == "__main__":
    main()
