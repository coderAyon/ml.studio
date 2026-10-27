"""Exact evidence and reversible-change previews for the preprocessing workspace.

These helpers are pure: callers own applying a reviewed operation to DatasetVersions.
IQR bounds and exploratory imputation use the full active dataset, so model evaluation
must reject these fitted whole-data transformations until they have been undone.
"""
from dataclasses import asdict, is_dataclass
import math
import numpy as np
import pandas as pd

from .data import Transformation


EXPLORATORY_OPERATIONS = frozenset({"Cap outliers (IQR)", "Remove outlier rows (IQR)", "Fill missing"})
NEW_OPERATIONS = frozenset({"Cap outliers (IQR)", "Remove outlier rows (IQR)", "Drop missing rows"})
PREVIEW_LIMIT = 30
PREVIEW_COLUMNS = 20
OUTLIER_CAUTION = (
    "The 1.5 × IQR rule identifies unusual values, not confirmed mistakes. "
    "Real extremes can be useful; capping or deleting them may bias the data. "
    "Bounds are fitted on all active rows. Undo this exploratory change before "
    "model evaluation and fit learned preprocessing within training folds instead."
)


def _check_frame(frame):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Import a dataset with at least one row and column first.")
    if not frame.columns.is_unique:
        raise ValueError("Resolve duplicate column names before preprocessing.")


def _column(frame, column):
    if not isinstance(column, str) or column not in frame:
        raise ValueError("Choose a column from the active dataset.")
    return frame[column]


def _finite_numeric(frame, column):
    series = _column(frame, column)
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        raise ValueError("IQR outlier review needs a numeric measurement, not a category or Boolean.")
    return series.replace([np.inf, -np.inf], np.nan)


def _number(value):
    return float(value) if pd.notna(value) and math.isfinite(float(value)) else None


def iqr_evidence(frame, column):
    """Return a positional boolean mask and JSON-safe 1.5-IQR evidence.

    Missing and infinite entries are excluded; infinity has its own review action.
    An IQR of zero still uses the stated fences rather than inventing a fallback.
    """
    series = _finite_numeric(frame, column)
    count = int(series.notna().sum())
    if count:
        q1, q3 = series.quantile([.25, .75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        if not all(math.isfinite(float(v)) for v in [q1, q3, iqr, lower, upper]):
            raise ValueError("IQR bounds exceed the numeric range. Review or rescale these measurements first.")
        mask = ((series < lower) | (series > upper)).fillna(False).to_numpy(dtype=bool)
    else:
        q1 = q3 = iqr = lower = upper = np.nan
        mask = np.zeros(len(frame), dtype=bool)
    return mask, {"column": column, "q1": _number(q1), "q3": _number(q3), "iqr": _number(iqr),
                  "lower": _number(lower), "upper": _number(upper), "finite_count": count,
                  "missing_count": int(frame[column].isna().sum()),
                  "infinite_count": int(frame[column].isin([np.inf, -np.inf]).sum()),
                  "outlier_count": int(mask.sum()), "multiplier": 1.5,
                  "method": "Finite values outside Q1 − 1.5 × IQR or Q3 + 1.5 × IQR"}


def _json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    return str(value)[:1000]


def _preview(frame, positions, preferred=()):
    names = [c for c in preferred if c in frame]
    names += [c for c in frame if c not in names]
    names = names[:PREVIEW_COLUMNS]
    chosen = list(positions)[:PREVIEW_LIMIT]
    subset = frame.iloc[chosen][names]
    values = [[_json_value(v) for v in row] for row in subset.itertuples(index=False, name=None)]
    truncated = sum(isinstance(v, str) and len(v) > 1000 for row in subset.itertuples(index=False, name=None) for v in row)
    return {"columns": list(names), "index": [str(frame.index[i]) for i in chosen],
            "positions": [int(i) + 1 for i in chosen], "data": values,
            "shown_rows": len(chosen), "omitted_columns": max(0, len(frame.columns) - len(names)),
            "truncated_cells": int(truncated)}


def _finding_mask(frame, issue):
    title = issue.get("title", "")
    columns = issue.get("columns", [])
    if not isinstance(columns, list) or any(c not in frame for c in columns):
        raise ValueError("This finding refers to columns outside the active dataset. Refresh the findings.")
    mask = np.zeros(len(frame), dtype=bool)
    bounds = []
    exact = True
    scope = "Exact rows matching this detection rule."
    if title == "Repeated rows":
        mask = frame.duplicated(keep="first").to_numpy(dtype=bool)
        scope = "Extra exact duplicate rows, excluding the first copy. Repeated observations may be legitimate."
    elif title in {"Potential outliers", "Skewed numeric distribution"}:
        for c in columns:
            outliers, detail = iqr_evidence(frame, c)
            bounds.append(detail)
            if title == "Potential outliers":
                mask |= outliers
            else:
                mask |= _finite_numeric(frame, c).notna().to_numpy(dtype=bool)
        if title == "Skewed numeric distribution":
            scope = "Finite values used to describe the distribution; these rows are not individually identified as errors."
    elif title == "Missing values":
        mask = frame[columns].isna().any(axis=1).to_numpy(dtype=bool) if columns else frame.isna().any(axis=1).to_numpy(dtype=bool)
    elif title == "Infinite numeric values":
        for c in columns:
            mask |= frame[c].isin([np.inf, -np.inf]).to_numpy(dtype=bool)
    elif title == "Surrounding whitespace":
        for c in columns:
            mask |= frame[c].map(lambda v: isinstance(v, str) and v != v.strip()).to_numpy(dtype=bool)
    elif title == "Possibly mixed numeric values":
        for c in columns:
            mask |= (frame[c].notna() & pd.to_numeric(frame[c], errors="coerce").isna()).to_numpy(dtype=bool)
    elif title == "Equivalent category labels":
        for c in columns:
            values = frame[c].astype("string")
            normalized = values.str.strip().str.lower()
            counts = pd.DataFrame({"key": normalized, "value": values}).dropna().groupby("key")["value"].nunique()
            ambiguous = counts[counts > 1].index
            mask |= normalized.isin(ambiguous).to_numpy(dtype=bool)
        scope = "Rows belonging to normalized labels with multiple original spellings. Review their meaning before merging them."
    elif title in {"Empty or constant column", "High cardinality / possible identifier"}:
        mask[:] = True
        scope = "Column-level finding: the preview shows context, not proof that individual records are wrong."
    else:
        exact = False
        mask[:] = True
        scope = "Dataset-level/context preview. This finding does not identify an exact set of erroneous rows."
    return mask, bounds, exact, scope


def inspect_finding(frame, issue):
    """Explain an analyzer finding using exact current-data masks, not evidence samples."""
    _check_frame(frame)
    if is_dataclass(issue):
        issue = asdict(issue)
    if not isinstance(issue, dict):
        raise ValueError("Choose a data-quality finding first.")
    mask, bounds, exact, scope = _finding_mask(frame, issue)
    positions = np.flatnonzero(mask)
    title = str(issue.get("title", "Dataset review"))
    actions = {
        "Potential outliers": ["Cap outliers (IQR)", "Remove outlier rows (IQR)"],
        "Missing values": ["Drop missing rows"],
        "Repeated rows": ["Remove duplicates"],
        "Surrounding whitespace": ["Trim whitespace"],
        "Equivalent category labels": ["Trim whitespace", "Lowercase labels"],
        "Infinite numeric values": ["Replace infinity with missing"],
        "Empty or constant column": ["Drop columns"],
    }.get(title, [])
    caution = OUTLIER_CAUTION if title == "Potential outliers" else str(issue.get("risk", "Review the data meaning before changing it."))
    return {"id": str(issue.get("id", "")), "title": title, "columns": issue.get("columns", []),
            "affected_count": int(mask.sum()) if exact else None, "total_rows": len(frame),
            "reported_count": int(issue.get("count", 0)), "exact_mask": exact,
            "preview": _preview(frame, positions, issue.get("columns", [])),
            "preview_limit": PREVIEW_LIMIT, "scope": scope, "bounds": bounds,
            "method": str(issue.get("method", "")), "explanation": str(issue.get("explanation", "")),
            "recommendation": str(issue.get("recommendation", "")), "caution": caution,
            "actions": actions,
            "automatic_training": title == "Missing values",
            "training_note": "Model training already fits median/mode imputation, category encoding and scaling inside the training split. Prefer that to whole-data imputation." if title == "Missing values" else "",
            "preview_note": "At most 30 matching/context rows and 20 columns are shown. Active positions are 1-based; retained index labels may differ. Long text is truncated at 1,000 characters."}


def apply_operation(frame, operation, column="", value=""):
    """Return changed data without mutating frame; reject empty outputs and bad input."""
    _check_frame(frame)
    if not isinstance(operation, str) or not isinstance(column, str) or not isinstance(value, str):
        raise ValueError("Choose a valid preprocessing operation and text value.")
    if operation in {"Cap outliers (IQR)", "Remove outlier rows (IQR)"}:
        mask, bounds = iqr_evidence(frame, column)
        if not bounds["finite_count"]:
            raise ValueError("This column has no finite numeric values for IQR outlier treatment.")
        result = frame.copy(deep=True)
        if operation == "Cap outliers (IQR)":
            if not mask.any():
                return result
            if pd.api.types.is_integer_dtype(frame[column]) and frame[column].dropna().map(lambda v: abs(int(v)) > 2**53).any():
                raise ValueError("These integers exceed the exact floating-point range. Review the measurement or exclude this identifier instead of capping it.")
            # Float output allows noninteger quartile fences for integer measurements.
            numeric = result[column].astype("float64")
            numeric.iloc[np.flatnonzero(mask)] = numeric.iloc[np.flatnonzero(mask)].clip(bounds["lower"], bounds["upper"])
            result[column] = numeric
        else:
            result = result.iloc[np.flatnonzero(~mask)].reset_index(drop=True)
    elif operation == "Drop missing rows":
        if column:
            _column(frame, column)
        result = frame.dropna(subset=[column] if column else None).reset_index(drop=True)
    else:
        global_operation = operation in {"Remove duplicates","Select row positions"}
        if not global_operation:
            _column(frame, column)
        result = Transformation(operation, [] if global_operation else [column], value).preview(frame)
    if result.empty:
        raise ValueError("This change would remove every row or column. Keep at least one row and column.")
    return result


def _equal_cells(before, after):
    equal = before.eq(after) | (before.isna() & after.isna())
    return equal.fillna(False).to_numpy(dtype=bool)


def preview_change(frame, operation, column="", value=""):
    """JSON-safe affected-row before/after preview, with an operations replay record."""
    proposed = apply_operation(frame, operation, column, value)
    columns = [] if operation == "Remove duplicates" or not column else [column]
    bounds = []
    removed_rows = len(frame) - len(proposed)
    changed_cells = 0
    if operation in {"Cap outliers (IQR)", "Remove outlier rows (IQR)"}:
        mask, detail = iqr_evidence(frame, column)
        bounds = [detail]
    elif operation == "Drop missing rows":
        mask = (frame[[column]] if column else frame).isna().any(axis=1).to_numpy(dtype=bool)
    elif operation == "Remove duplicates":
        mask = frame.duplicated(keep="first").to_numpy(dtype=bool)
    elif operation == 'Select row positions':
        import json
        positions=json.loads(value)
        mask=~np.isin(np.arange(len(frame)),positions)
    elif operation == "Filter rows":
        mask = ~frame[column].astype("string").str.contains(value, regex=False, case=False, na=False).to_numpy(dtype=bool)
    elif operation in {"Drop columns", "Rename column"}:
        mask = np.ones(len(frame), dtype=bool)
    else:
        equal = _equal_cells(frame, proposed)
        changed_cells = int((~equal).sum())
        mask = (~equal).any(axis=1)
    if operation == "Cap outliers (IQR)":
        changed_cells = int((~_equal_cells(frame, proposed)).sum())
    affected = np.flatnonzero(mask)
    aligned = len(frame) == len(proposed) and operation!='Select row positions'
    preferred_after = [value.strip()] if operation == "Rename column" else columns
    after_positions = affected if aligned else np.arange(min(PREVIEW_LIMIT, len(proposed)))
    types = [{"column": str(c), "before": str(frame[c].dtype), "after": str(proposed[c].dtype)}
             for c in frame.columns if c in proposed and frame[c].dtype != proposed[c].dtype]
    summary = {"before_rows": len(frame), "after_rows": len(proposed), "before_columns": len(frame.columns),
               "after_columns": len(proposed.columns), "affected_rows": int(mask.sum()),
               "changed_cells": changed_cells, "removed_rows": max(0, removed_rows),
               "removed_columns": max(0, len(frame.columns) - len(proposed.columns)),
               "before_missing": int(frame.isna().sum().sum()), "after_missing": int(proposed.isna().sum().sum()),
               "before_duplicates": int(frame.duplicated().sum()), "after_duplicates": int(proposed.duplicated().sum())}
    if bounds:
        finite = proposed[column].replace([np.inf, -np.inf], np.nan)
        rule_after = ((finite < bounds[0]["lower"]) | (finite > bounds[0]["upper"])).fillna(False)
        summary["outliers_before"] = bounds[0]["outlier_count"]
        summary["outliers_after_original_bounds"] = int(rule_after.sum())
    caution = OUTLIER_CAUTION if operation in {"Cap outliers (IQR)", "Remove outlier rows (IQR)"} else (
        "Whole-data imputation is exploratory. Undo it before evaluation; model pipelines fit imputation within training splits."
        if operation == "Fill missing" else "Review these changes and column meaning before applying. The original CSV is preserved."
    )
    return {"operation": {"operation": operation, "columns": columns, "value": value},
            "operations": [{"operation": operation, "columns": columns, "value": value}],
            "column": column, "before": _preview(frame, affected, columns),
            "after": _preview(proposed, after_positions, preferred_after), "summary": summary,
            "bounds": bounds, "type_changes": types, "aligned_rows": aligned,
            "after_scope": "Same active row positions after the proposed change." if aligned else
                "First retained rows after removal; these are not aligned to the removed rows on the left.",
            "caution": caution, "exploratory": operation in EXPLORATORY_OPERATIONS,
            "note": "Preview only; no data has changed. At most 30 rows and 20 columns are displayed. "
                "IQR after-counts use the original fences; refreshing analysis recomputes fences and may flag different values."}
