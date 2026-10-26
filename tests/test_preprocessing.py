"""Evidence and transformation tests for the dedicated preprocessing workspace."""
import json
import numpy as np
import pandas as pd
import pytest
from mlstudio.preprocessing import inspect_finding, preview_change, apply_operation, iqr_evidence, EXPLORATORY_OPERATIONS


def sample():
    return pd.DataFrame({"value": [1., 2., 3., 4., 5., 6., 100., np.inf, np.nan],
                         "label": [" A", "a", "b", "c", "d", "e", "f", "g", None]},
                        index=[8, 8, 14, 2, 22, 25, 31, 32, 40])


def finding(title, columns, count=1):
    return {"id": "f1", "title": title, "columns": columns, "count": count,
            "method": "test method", "explanation": "Explanation", "recommendation": "Review", "risk": "Caution"}


def test_outlier_evidence_has_exact_positions_and_original_index():
    frame = sample()
    result = inspect_finding(frame, finding("Potential outliers", ["value"]))
    assert result["affected_count"] == 1
    assert result["preview"]["positions"] == [7]
    assert result["preview"]["index"] == ["31"]
    assert result["preview"]["data"][0][0] == 100.
    assert result["bounds"][0]["q1"] == 2.5 and result["bounds"][0]["q3"] == 5.5
    assert result["bounds"][0]["lower"] == -2. and result["bounds"][0]["upper"] == 10.
    assert result["bounds"][0]["finite_count"] == 7
    assert result["bounds"][0]["infinite_count"] == 1
    json.dumps(result, allow_nan=False)
    pd.testing.assert_frame_equal(frame, sample())


def test_outlier_cap_remove_previews_are_aligned_truthfully_and_no_mutation():
    frame = sample()
    cap = preview_change(frame, "Cap outliers (IQR)", "value")
    assert cap["before"]["positions"] == cap["after"]["positions"] == [7]
    assert cap["before"]["data"][0][0] == 100.
    assert cap["after"]["data"][0][0] == 10.
    assert cap["summary"]["changed_cells"] == 1 and cap["summary"]["affected_rows"] == 1
    assert cap["summary"]["outliers_after_original_bounds"] == 0
    assert cap["exploratory"] and cap["aligned_rows"]
    proposed = apply_operation(frame, "Cap outliers (IQR)", "value")
    assert np.isinf(proposed.value.iloc[7]) and pd.isna(proposed.value.iloc[8])
    remove = preview_change(frame, "Remove outlier rows (IQR)", "value")
    assert remove["summary"]["removed_rows"] == 1 and not remove["aligned_rows"]
    assert len(apply_operation(frame, "Remove outlier rows (IQR)", "value")) == 8
    assert "not aligned" in remove["after_scope"]
    pd.testing.assert_frame_equal(frame, sample())
    json.dumps(cap, allow_nan=False)


def test_missing_infinity_whitespace_and_duplicate_evidence():
    frame = sample()
    missing = inspect_finding(frame, finding("Missing values", ["value"]))
    assert missing["affected_count"] == 1 and missing["preview"]["positions"] == [9]
    assert missing["automatic_training"]
    infinity = inspect_finding(frame, finding("Infinite numeric values", ["value"]))
    assert infinity["preview"]["data"][0][0] == "Infinity"
    assert infinity["preview"]["positions"] == [8]
    whitespace = inspect_finding(frame, finding("Surrounding whitespace", ["label"]))
    assert whitespace["preview"]["positions"] == [1]
    trimmed = preview_change(frame, "Trim whitespace", "label")
    assert trimmed["after"]["data"][0][0] == "A"
    assert trimmed["summary"]["changed_cells"] == 1
    duplicated = pd.DataFrame({"a": [1, 1, 2, 1]}, index=[7, 7, 2, 7])
    evidence = inspect_finding(duplicated, finding("Repeated rows", ["a"], 2))
    assert evidence["affected_count"] == 2 and evidence["preview"]["positions"] == [2, 4]
    assert preview_change(duplicated, "Remove duplicates")["summary"]["after_duplicates"] == 0


def test_nullable_integer_cap_dtype_and_zero_iqr():
    frame = pd.DataFrame({"x": pd.Series([1, 1, 1, 1, 100, pd.NA], dtype="Int64")})
    result = apply_operation(frame, "Cap outliers (IQR)", "x")
    assert result.x.iloc[4] == 1. and pd.isna(result.x.iloc[5])
    assert str(result.x.dtype) == "float64"
    assert iqr_evidence(frame, "x")[1]["iqr"] == 0.
    assert "Cap outliers (IQR)" in EXPLORATORY_OPERATIONS
    assert "Remove outlier rows (IQR)" in EXPLORATORY_OPERATIONS
    constant = pd.DataFrame({"x": pd.Series([5, 5, 5], dtype="Int64")})
    pd.testing.assert_frame_equal(apply_operation(constant, "Cap outliers (IQR)", "x"), constant)
    large = pd.DataFrame({"x": pd.Series([1, 1, 1, 1, 2**62], dtype="Int64")})
    with pytest.raises(ValueError, match="floating-point"):
        apply_operation(large, "Cap outliers (IQR)", "x")


def test_drop_missing_rows_in_one_column_or_all_columns():
    frame = pd.DataFrame({"x": [None, 1, 2], "y": [1, None, 3]})
    one = preview_change(frame, "Drop missing rows", "x")
    assert one["summary"]["removed_rows"] == 1
    assert len(apply_operation(frame, "Drop missing rows")) == 1
    assert not one["exploratory"]
    with pytest.raises(ValueError, match="every row"):
        apply_operation(pd.DataFrame({"x": [None]}), "Drop missing rows", "x")


def test_context_findings_do_not_claim_bad_rows_and_categories_are_exact():
    frame = pd.DataFrame({"label": ["Alpha", "alpha", "Beta", "gamma ", None]})
    categories = inspect_finding(frame, finding("Equivalent category labels", ["label"], 4))
    assert categories["affected_count"] == 2 and categories["reported_count"] == 4
    assert categories["preview"]["positions"] == [1, 2]
    context = inspect_finding(frame, finding("Small dataset", [], 5))
    assert context["affected_count"] is None and not context["exact_mask"]
    assert "context" in context["scope"]
    with pytest.raises(ValueError):inspect_finding(frame, finding("Missing values", ["absent"]))


def test_preview_limits_and_unsafe_or_impossible_changes():
    frame = pd.DataFrame({"x": [np.nan] * 40, "label": ["long" * 400] * 40})
    missing = inspect_finding(frame, finding("Missing values", ["x"], 40))
    assert missing["affected_count"] == 40 and len(missing["preview"]["data"]) == 30
    assert missing["preview"]["truncated_cells"] == 30
    assert len(missing["preview"]["data"][0][1]) == 1000
    with pytest.raises(ValueError, match="finite"):
        apply_operation(frame, "Cap outliers (IQR)", "x")
    with pytest.raises(ValueError, match="numeric"):
        apply_operation(frame, "Cap outliers (IQR)", "label")
    with pytest.raises(ValueError, match="every row or column"):
        apply_operation(pd.DataFrame({"x": [1]}), "Drop columns", "x")
    with pytest.raises(ValueError):apply_operation(frame, "Arbitrary operation", "x")
    with pytest.raises(ValueError):apply_operation(frame, "Trim whitespace", "absent")
