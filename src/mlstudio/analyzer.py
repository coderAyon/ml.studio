"""Deterministic evidence-based quality findings; never modifies data."""
from dataclasses import asdict, dataclass, field
import pandas as pd


@dataclass
class Issue:
    id: str
    title: str
    severity: str
    status: str
    columns: list[str]
    count: int
    method: str
    explanation: str
    recommendation: str
    risk: str
    evidence: list = field(default_factory=list)


def analyze(frame: pd.DataFrame) -> list[Issue]:
    issues = []

    def add(title, severity, status, columns, count, method, explanation, recommendation, risk, evidence=None):
        issues.append(Issue(str(len(issues) + 1), title, severity, status, columns, int(count), method, explanation, recommendation, risk, evidence or []))

    n = len(frame)
    duplicate_headers = frame.attrs.get("duplicate_headers", [])
    if duplicate_headers:
        add("Duplicate source column names", "High", "Confirmed Issue", duplicate_headers, len(duplicate_headers), "Original CSV header equality", "Pandas disambiguates repeated names with suffixes. Review their meaning.", "Rename columns in the source or review the inferred schema.", "Confusing columns can lead to incorrect feature selection.", duplicate_headers)
    if len(frame.columns) >= n:
        add("Many features relative to rows", "Medium", "Recommendation", [], len(frame.columns), "Feature count >= row count", "Flexible models may overfit this dataset.", "Use regularization and validate with training-fold feature selection.", "Selecting features on all rows introduces leakage.")
    duplicates = frame.duplicated()
    if duplicates.any():
        add("Repeated rows", "Medium", "Potential Issue", list(frame.columns), duplicates.sum(), "Exact row equality", "Repeated records may be legitimate observations.", "Inspect records before removing duplicates.", "Removal can discard valid repeated observations.", frame.index[duplicates].tolist()[:10])
    if n < 100:
        add("Small dataset", "Medium", "Recommendation", [], n, "Row count < 100", "Validation estimates may have high variance.", "Use appropriate cross-validation and report uncertainty.", "Too many folds can leave insufficient class samples.")
    for column in frame:
        s = frame[column]
        missing = int(s.isna().sum())
        if missing:
            add("Missing values", "High" if missing/n > .5 else "Medium", "Confirmed Issue", [column], missing, "isna", f"{missing/n:.1%} of this column is missing.", "Review median or most-frequent imputation inside the training pipeline.", "No imputation strategy suits every dataset; whole-data imputation can leak validation information.", frame.index[s.isna()].tolist()[:10])
        unique = s.nunique()
        if unique <= 1:
            add("Empty or constant column", "Low", "Confirmed Issue", [column], n, "Distinct nonmissing values <= 1", "This column contains no observed variation.", "Consider excluding it after review.", "A constant may have domain significance.")
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            nonfinite=s.isin([float("inf"),float("-inf")])
            if nonfinite.any():
                add("Infinite numeric values", "High", "Confirmed Issue", [column], nonfinite.sum(), "Positive or negative infinity", "Some numeric entries are infinite and cannot be used directly by the training pipeline.", "Review these rows. Preview replacing infinity with missing, then let training handle missing values inside its split.", "Infinity may carry domain meaning; review the original values before changing them.", frame.index[nonfinite].tolist()[:10])
                s=s.replace([float("inf"),float("-inf")],float("nan"))
            q1, q3 = s.quantile([.25, .75])
            iqr = q3-q1
            mask = (s < q1-1.5*iqr) | (s > q3+1.5*iqr)
            if mask.any():
                add("Potential outliers", "Low", "Potential Issue", [column], mask.sum(), "1.5 × IQR fences", "Extreme observations are not necessarily errors.", "Review a box plot and original records.", "Deleting extremes can bias the data.", s[mask].head(8).tolist())
            if s.notna().sum() > 8 and abs(s.skew()) > 2:
                add("Skewed numeric distribution", "Low", "Recommendation", [column], s.notna().sum(), "Absolute sample skewness > 2", "A long-tailed distribution can affect some estimators.", "Review the histogram; consider robust scaling or a domain-appropriate transform.", "Transforms alter interpretation and must be fitted within training folds.")
        else:
            strings = s.dropna().astype(str)
            whitespace = strings.ne(strings.str.strip())
            numeric_fraction = pd.to_numeric(strings, errors="coerce").notna().mean()
            if .8 <= numeric_fraction < 1:
                add("Possibly mixed numeric values", "Medium", "Potential Issue", [column], int((1-numeric_fraction)*len(strings)), "At least 80% but not all labels parse as numbers", "Some entries differ from the dominant numeric format.", "Review nonnumeric entries and decimal settings before confirming a type conversion.", "These values may be valid categorical labels.", strings[pd.to_numeric(strings, errors="coerce").isna()].head(8).tolist())
            if whitespace.any():
                add("Surrounding whitespace", "Low", "Confirmed Issue", [column], whitespace.sum(), "Compare stripped labels", "Whitespace can split equivalent labels.", "Preview trimming whitespace.", "Whitespace can occasionally be meaningful.", strings[whitespace].head(8).tolist())
            if strings.nunique() > strings.str.strip().str.lower().nunique():
                add("Equivalent category labels", "Low", "Potential Issue", [column], len(strings), "Case-folded and trimmed distinct counts", "Some labels differ only in case or whitespace.", "Review labels before standardizing.", "Case can carry domain meaning.")
            if unique > 20 and unique/n > .9:
                add("High cardinality / possible identifier", "Medium", "Potential Issue", [column], unique, "Distinct ratio > 90%", "Nearly every record has a different label.", "Check whether this is an identifier before modeling.", "Removing a meaningful feature can reduce predictive value.")
    numeric = frame.select_dtypes(include="number")
    if len(numeric.columns) <= 150:
        correlations = numeric.corr()
        for i, first in enumerate(correlations.columns):
            for second in correlations.columns[i+1:]:
                coefficient = correlations.loc[first, second]
                if abs(coefficient) > .95:
                    add("Strong numeric correlation", "Low", "Potential Issue", [first, second], n, "Absolute Pearson correlation > 0.95", "Features may be redundant. Association does not imply causation or prove leakage.", "Investigate feature provenance, especially if one column is the target.", "Deleting correlated features can remove useful information.", [float(coefficient)])
    return issues


def profile(frame: pd.DataFrame) -> dict:
    fields = []
    for c in frame:
        s = frame[c]
        present = int(s.notna().sum())
        item = {"name": c, "type": str(s.dtype), "missing": int(s.isna().sum()), "unique": int(s.nunique()), "missing_percent": float(s.isna().mean()*100) if len(s) else 0.,
                "present": present, "unique_percent": float(s.nunique()/max(1,present)*100),
                "bytes": int(s.memory_usage(index=False, deep=True)),
                "examples": [str(v)[:180] for v in s.dropna().drop_duplicates().head(4)],
                "kind": "Boolean" if pd.api.types.is_bool_dtype(s) else "Numeric" if pd.api.types.is_numeric_dtype(s) else "Date / time" if pd.api.types.is_datetime64_any_dtype(s) else "Text / category"}
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            finite=s.replace([float("inf"),float("-inf")],float("nan"))
            item.update(finite=int(finite.notna().sum()), infinite=int(s.isin([float('inf'),float('-inf')]).sum()), zero=int((finite==0).sum()), negative=int((finite<0).sum()))
            for name, value in {"mean": finite.mean(), "median": finite.median(), "std": finite.std(), "min": finite.min(), "max": finite.max(), "q1": finite.quantile(.25), "q3": finite.quantile(.75)}.items():
                import math
                item[name] = float(value) if pd.notna(value) and math.isfinite(float(value)) else None
        item["top_values"] = [{"value": str(k), "count": int(v), "percent": float(v/max(1,len(s))*100)} for k,v in s.value_counts().head(8).items()]
        fields.append(item)
    missing_rows=int(frame.isna().any(axis=1).sum())
    return {"rows": len(frame), "columns": len(frame.columns), "bytes": int(frame.memory_usage(deep=True).sum()), "fields": fields,
            "total_cells": int(frame.size), "rows_with_missing": missing_rows,
            "complete_rows": len(frame)-missing_rows, "empty_rows": int(frame.isna().all(axis=1).sum()),
            "duplicate_members": int(frame.duplicated(keep=False).sum()),
            "constant_columns": sum(f['unique']==1 for f in fields), "empty_columns": sum(f['present']==0 for f in fields)}
