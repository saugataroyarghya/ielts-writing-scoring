import pandas as pd
import pytest

from ielts_scoring.data import assert_split_integrity, grouped_split, merge_groups, similarity_groups


def test_duplicate_links_merge_prompt_families_transitively():
    groups = merge_groups([1, 1, 2, 3], [10, 11, 11, 12])
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_near_duplicate_matching_keeps_minor_edits_together():
    original = "Public libraries provide free access to education and information for the whole community. " * 8
    revised = original.replace("free access", "open access", 1)
    unrelated = "Space exploration depends on rockets, satellites, orbital mechanics and deep space communication. " * 8
    groups = similarity_groups([original, revised, unrelated])
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]


def test_grouped_splits_are_reproducible_and_disjoint():
    frame = pd.DataFrame({"id": [str(i) for i in range(40)], "source": "ielts",
                          "group_id": [i // 2 for i in range(40)],
                          "prompt_group": [i // 2 for i in range(40)],
                          "essay_hash": [f"hash{i}" for i in range(40)]})
    first = grouped_split(frame)
    pd.testing.assert_frame_equal(first, grouped_split(frame))
    assert_split_integrity(first)
    first.loc[0, "essay_hash"] = first.loc[first["split"].ne(first.loc[0, "split"]), "essay_hash"].iloc[0]
    with pytest.raises(ValueError, match="Leakage"):
        assert_split_integrity(first)


def test_supplied_test_partition_is_preserved():
    frame = pd.DataFrame({"id": [str(i) for i in range(40)], "source": "ellipse",
                          "group_id": range(40), "essay_hash": [f"hash{i}" for i in range(40)],
                          "original_split": ["test"] * 10 + ["train"] * 30})
    result = grouped_split(frame, supplied_test=True)
    assert result.loc[:9, "split"].eq("test").all()
    assert result.loc[10:, "split"].ne("test").all()
