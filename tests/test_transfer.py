import numpy as np
import pandas as pd
import pytest

from ielts_scoring.transfer import paired_group_difference


def test_paired_comparison_aligns_ids_and_preserves_difference_direction():
    first = pd.DataFrame({"id": ["a", "b", "c"], "group_id": ["x", "x", "y"],
                          "actual_TR": [5, 6, 7], "predicted_TR": [6, 7, 8]})
    second = first.copy()
    second["predicted_TR"] -= 0.5
    result = paired_group_difference(first, second.iloc[::-1], ["TR"], draws=30)
    assert result["macro_mae_difference_second_minus_first"] == -0.5
    np.testing.assert_allclose(result["paired_prompt_group_95ci"], [-0.5, -0.5])
    assert result["groups"] == 2
    with pytest.raises(ValueError, match="rows or groups"):
        paired_group_difference(first, second.iloc[:2], ["TR"])
    with pytest.raises(ValueError, match="Duplicate"):
        paired_group_difference(first, pd.concat([second, second.iloc[:1]]), ["TR"])
