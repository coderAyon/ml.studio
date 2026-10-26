"""Full-data grouped summaries and labeled charts for introductory data labs."""
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def grouped_analysis(frame, group, columns=None, statistic="mean", layout="combined",
                     title="", xlabel="", ylabel="", top_n=None, order="original", orientation="vertical"):
    if group not in frame:
        raise ValueError("Choose a group column from your dataset.")
    if statistic not in {"mean", "count", "median", "sum", "min", "max", "std"}:
        raise ValueError("Choose a supported summary method.")
    if layout not in {"combined", "subplots"}:
        raise ValueError("Choose a chart layout.")
    if order not in {"original", "ascending", "descending"} or orientation not in {"vertical", "horizontal"}:
        raise ValueError("Choose a supported sort order and orientation.")
    if top_n is not None and (type(top_n) is not int or not 1<=top_n<=30):
        raise ValueError("Top groups must be between 1 and 30.")
    columns = columns or []
    if not isinstance(columns, list) or len(columns) != len(set(columns)):
        raise ValueError("Choose distinct numeric measurements.")
    if statistic != "count" and (not columns or len(columns) > 12 or
            any(c not in frame or c == group or not pd.api.types.is_numeric_dtype(frame[c]) for c in columns)):
        raise ValueError("Choose 1–12 numeric measurements, excluding the group column.")
    if frame.empty:
        raise ValueError("The dataset has no rows to summarize.")
    # Missing labels remain a real group, distinct from literal '(missing)' text.
    grouped = frame.groupby(group, dropna=False, observed=True, sort=False)
    if grouped.ngroups > 30 and top_n is None:
        raise ValueError("Choose a group column with at most 30 groups; IDs usually are unsuitable.")
    if statistic == "count":
        result = grouped.size().to_frame("Row count")
    else:
        clean = frame[[group, *columns]].replace({c: {np.inf: np.nan, -np.inf: np.nan} for c in columns})
        result = clean.groupby(group, dropna=False, observed=True, sort=False)[columns].agg(statistic)
        if statistic == "sum":
            result = clean.groupby(group, dropna=False, observed=True, sort=False)[columns].sum(min_count=1)
    result = result.replace([np.inf, -np.inf], np.nan)
    total_groups = len(result)
    if order != "original":result = result.sort_values(result.columns[0], ascending=order=="ascending", kind="stable", na_position="last")
    if top_n is not None:result = result.head(top_n)
    missing_label = "(missing group)"
    existing = {str(v) for v in result.index if not pd.isna(v)}
    while missing_label in existing:missing_label = "_" + missing_label
    labels = [missing_label if pd.isna(v) else str(v) for v in result.index]
    figure = Figure(figsize=(10, max(4, len(result)*2.6) if layout == "subplots" else 5), layout="constrained")
    color = ["#8966dc", "#53ae9a", "#639ae8", "#df9b50", "#c678ad"]
    heading = str(title).strip()[:160] or f"{statistic.title()} by {group}"
    if layout == "subplots":
        for i, (label, row) in enumerate(zip(labels, result.to_numpy(dtype=float))):
            axis = figure.add_subplot(len(result), 1, i+1)
            axis.bar(list(result.columns), row, color=[color[j % len(color)] for j in range(len(row))])
            axis.set_title(label)
            axis.set_xlabel(str(xlabel).strip()[:120] or "Measurement")
            axis.set_ylabel(str(ylabel).strip()[:120] or ("Number of rows" if statistic == "count" else statistic.title()))
            axis.tick_params(axis="x", labelrotation=20)
        figure.suptitle(heading)
    else:
        axis = figure.add_subplot(111)
        x = np.arange(len(result))
        width = .8 / len(result.columns)
        for i, column in enumerate(result):
            positions=x + (i-(len(result.columns)-1)/2)*width
            if orientation=="horizontal":axis.barh(positions,result[column].to_numpy(dtype=float),width,label=str(column),color=color[i % len(color)])
            else:axis.bar(positions,result[column].to_numpy(dtype=float),width,label=str(column),color=color[i % len(color)])
        if orientation=="horizontal":axis.set_yticks(x,labels);axis.invert_yaxis()
        else:axis.set_xticks(x, labels, rotation=25)
        axis.set_title(heading)
        measure="Number of rows" if statistic == "count" else statistic.title()
        axis.set_xlabel(str(xlabel).strip()[:120] or (measure if orientation=="horizontal" else str(group)))
        axis.set_ylabel(str(ylabel).strip()[:120] or (str(group) if orientation=="horizontal" else measure))
        if len(result.columns) > 1:
            axis.legend()
    table = {"columns": [str(group), *map(str, result.columns)],
             "index": list(range(len(result))),
             "data": [[label, *[None if pd.isna(v) else float(v) for v in row]]
                      for label, row in zip(labels, result.to_numpy())]}
    note = (f"All {len(frame):,} rows; showing {len(result)} of {total_groups} groups. Missing group labels are included. "
            + ("Count counts rows, including rows with missing measurements." if statistic == "count" else
               "Missing and infinite measurements are excluded per feature; empty or overflowed results remain blank. Standard deviation uses sample ddof=1."))
    return {"table": table, "note": note, "title": heading}, figure
