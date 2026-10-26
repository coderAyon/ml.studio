"""Deterministic visual summaries, preprocessing guidance and model explanations."""
import numpy as np
import pandas as pd


def visual_summary(frame, x=None, y=None, category=None):
    """Small JSON-native summaries; samples are bounded and explicitly labeled."""
    sample = frame.sample(min(len(frame), 2000), random_state=42).replace([np.inf, -np.inf], np.nan)
    numeric = [c for c in sample.select_dtypes(include="number") if sample[c].notna().any()]
    varied = [c for c in numeric if sample[c].nunique() > 1]
    x = x if x in numeric else (varied or numeric or [None])[0]
    y = y if y in numeric else next((c for c in varied if c != x), None)
    categorical = [c for c in sample if not pd.api.types.is_numeric_dtype(sample[c])]
    category = category if category in sample else (categorical or [x])[0]
    histogram = None
    if x:
        counts, edges = np.histogram(sample[x].dropna().to_numpy(dtype=float), bins=min(16, max(3, sample[x].nunique())))
        histogram = {"column": x, "counts": counts.tolist(), "edges": edges.tolist()}
    scatter = None
    if x and y:
        pairs = sample[[x, y]].dropna()
        scatter = {"x": x, "y": y, "points": pairs.to_numpy(dtype=float).tolist()}
    bar = None
    if category:
        labels = sample[category].map(lambda v: str(v) if pd.notna(v) else "(missing)")
        counts = labels.value_counts().head(10)
        bar = {"column": category, "labels": counts.index.tolist(), "counts": counts.tolist()}
    return {"rows": len(frame), "sample_rows": len(sample), "sampled": len(sample)<len(frame), "histogram": histogram, "scatter": scatter, "bar": bar, "numeric_columns": numeric, "category_columns": list(frame.columns)}


MODEL_NOTES = {
    "Logistic Regression": ("An interpretable linear classification baseline. Regularization helps limit overfitting.", "Can miss nonlinear relationships; scaling and categorical encoding are handled in the pipeline."),
    "Decision Tree": ("Captures nonlinear rules and interactions with easy-to-inspect splits.", "An unrestricted tree can overfit, particularly with few rows."),
    "Random Forest": ("Averages many trees to capture nonlinear relationships and reduce single-tree variance.", "More training time and less direct interpretability; still needs independent validation."),
    "KNN": ("A distance-based baseline when similar feature values tend to share a class.", "Sensitive to scaling, irrelevant features and high-dimensional one-hot data; fewer than five training rows is unsuitable for its default k=5."),
    "SVM": ("A nonlinear margin-based classifier worth comparing on small-to-medium tabular data.", "Needs scaling; training can be slow on many rows. Default settings may require tuning."),
    "Naive Bayes": ("A fast probabilistic baseline, useful when resources or sample sizes are limited.", "Gaussian assumptions and conditional feature independence may not match the data."),
    "Linear Regression": ("An interpretable baseline for an approximately linear numeric outcome.", "Sensitive to outliers and correlated predictors; relationships may be nonlinear."),
    "Ridge": ("Regularized linear regression can stabilize correlated or numerous features.", "Still assumes a linear relationship; regularization strength requires validation."),
    "Decision Tree Regressor": ("Learns nonlinear numeric relationships and feature interactions.", "May overfit; cannot extrapolate beyond learned leaf values."),
    "Random Forest Regressor": ("Averages nonlinear tree predictions and can reduce individual-tree variance.", "Higher cost and limited extrapolation; not guaranteed to beat a linear baseline."),
    "SVR": ("A scaled nonlinear regression baseline for small-to-medium datasets.", "Sensitive to parameters and potentially expensive with many rows."),
}


def preprocessing_plan(frame, target="", task="classification"):
    """Actionable evidence-based steps; learned preprocessing stays inside training."""
    steps, warnings = [], []
    def add(title, columns, reason, operation=None, risk="", stage="review"):
        steps.append({"title":title,"columns":columns,"reason":reason,"operation":operation,"risk":risk,"stage":stage})
    features = [c for c in frame if c != target]
    if len(frame) < 100:
        warnings.append(f"Small dataset: only {len(frame)} rows. Validation estimates may have high variance.")
    if not target:
        warnings.append("Choose the outcome you want to predict. A CSV alone cannot establish your research goal.")
    elif target not in frame:
        warnings.append("The selected target is not in this dataset.")
    else:
        s = frame[target]
        if s.isna().any():
            warnings.append(f"Target {target} has {s.isna().sum()} missing values. Review these records; do not impute target labels automatically.")
        if task == "classification":
            counts = s.value_counts()
            if len(counts)<2:
                warnings.append("Classification requires at least two target classes.")
            if not counts.empty and counts.min()<4:
                warnings.append("Some classes have fewer than four rows; a stratified 75/25 split may fail or be unreliable. Collect more examples or reconsider the task.")
            if len(counts)>20 or (len(frame)>20 and s.nunique()/len(frame)>.5):
                warnings.append("Many distinct target values: if this is a quantity, consider regression rather than treating each value as a class.")
            if not counts.empty and counts.max()/max(1,counts.sum())>.8:
                warnings.append("Imbalanced target: review per-class recall and F1 rather than accuracy alone. Current estimators use default class weights.")
        elif not pd.api.types.is_numeric_dtype(s):
            warnings.append("Regression needs a numeric target. Review the type or choose classification for category labels.")
    if frame.duplicated().any():
        add("Review repeated observations", [], f"{int(frame.duplicated().sum())} exact repeated rows. Check whether records represent the same subject before splitting.", "Remove duplicates", "Repeated measurements can be valid. Removing them changes the sample.")
    categories = []
    numeric = []
    for c in features:
        s = frame[c]
        is_numeric = pd.api.types.is_numeric_dtype(s)
        (numeric if is_numeric else categories).append(c)
        if is_numeric and s.isin([np.inf,-np.inf]).any():
            add("Review infinite values",[c],"Some entries are positive or negative infinity. Training cannot use them directly.","Replace infinity with missing","Confirm the original values and their meaning before converting them to missing.")
        if s.nunique() <= 1:
            add("Review an empty or constant feature", [c], f"{c} has {s.nunique()} distinct observed values and cannot distinguish records.", "Drop columns", "Exclude only after domain review; original data remains preserved.")
        if s.isna().any():
            strategy = "median" if is_numeric else "most-frequent"
            add(f"Handle missing values in {c}", [c], f"{int(s.isna().sum())} values are missing ({s.isna().mean():.1%}). The training pipeline fits {strategy} imputation on training rows only.", stage="training", risk="Full-dataset imputation before evaluation leaks validation information. No single strategy suits every dataset.")
        if not is_numeric:
            strings = s.dropna().astype(str)
            count = int(strings.ne(strings.str.strip()).sum())
            if count:
                add("Trim surrounding whitespace", [c], f"{count} labels have leading or trailing whitespace.", "Trim whitespace", "Review the proposed labels before applying.")
            if strings.nunique()>strings.str.strip().str.lower().nunique():
                add("Review equivalent labels", [c], "Some categories differ only by case or whitespace.", "Lowercase labels", "Case may be meaningful; do not merge labels automatically.")
        if len(frame)>10 and s.nunique()/len(frame)>.95 and (not is_numeric or str(c).lower() in {"id","index","row_id","identifier"}):
            add("Review a possible identifier", [c], "Nearly every value is unique. Check whether this feature describes an identity rather than a transferable pattern.", "Drop columns", "Uniqueness alone does not prove leakage or irrelevance.")
    if categories:
        add("Encode categorical features", categories, "Training uses one-hot encoding with unknown-category handling and a maximum of 100 categories per feature. It is fitted on training data only.", stage="training", risk="High-cardinality labels may still produce many weak features; review identifier-like columns.")
    if numeric:
        add("Scale numeric features", numeric, "Standard scaling is fitted on training rows inside each model pipeline. This is important for KNN, SVM/SVR and linear models; trees usually do not require it.", stage="training", risk="Scaling is already handled by the pipeline; avoid fitting a scaler on all rows before evaluation.")
    algorithms = ["Linear Regression","Ridge","Decision Tree Regressor","Random Forest Regressor","SVR"] if task=="regression" else ["Logistic Regression","Decision Tree","Random Forest","KNN","SVM","Naive Bayes"]
    candidates = []
    for a in algorithms:
        fit = f"This CSV has {len(frame)} rows, {len(numeric)} numeric and {len(categories)} categorical input columns. "
        if a in {"SVM","SVR","KNN"} and categories:
            fit += "One-hot categories expand the distance space; inspect comparison results before favoring a distance-based model."
        elif a in {"Random Forest","Decision Tree","Random Forest Regressor","Decision Tree Regressor"} and len(frame)<100:
            fit += "With fewer than 100 observations, flexible trees need particular care about overfitting."
        elif a in {"Logistic Regression","Ridge","Linear Regression"}:
            fit += "Use this as a simple baseline before deciding whether nonlinear complexity improves validation."
        else:
            fit += "Compare on the same split; these feature counts alone cannot establish the best model."
        candidates.append({"algorithm":a,"why":MODEL_NOTES[a][0],"tradeoff":MODEL_NOTES[a][1],"dataset_fit":fit})
    return {"task":task,"target":target,"steps":steps,"warnings":warnings,"algorithms":candidates,"next_step":"Confirm the target and task, review suggested cleaning with previews, then compare algorithms on the same split. Training-only imputation, encoding and scaling are automatic."}


def explain_comparison(records, task):
    metric, maximize = ("RMSE",False) if task=="regression" else ("F1 (weighted)",True)
    values = [(r, r.get("result",{}).get("metrics",{}).get(metric)) for r in records]
    values = [(r,v) for r,v in values if isinstance(v,(int,float)) and np.isfinite(v)]
    if not values:
        return {"metric":metric,"leaders":[],"summary":"Run a comparison to see explanations based on actual evaluation results.","rows":[],"caution":"No performance is inferred from the dataset alone."}
    ordered = sorted(values,key=lambda pair:pair[1],reverse=maximize)
    best = ordered[0][1]
    leaders = [r["algorithm"] for r,v in ordered if np.isclose(v,best,rtol=0,atol=1e-10)]
    rows = [{"algorithm":r["algorithm"],"score":v,"difference":abs(best-v),"why":MODEL_NOTES.get(r["algorithm"],("Inspect this algorithm's assumptions.",""))[0],"tradeoff":MODEL_NOTES.get(r["algorithm"],("",""))[1],"seconds":r.get("result",{}).get("metadata",{}).get("seconds")} for r,v in ordered]
    return {"metric":metric,"leaders":leaders,"summary":f"{', '.join(leaders)} {'tie for the highest' if len(leaders)>1 and maximize else 'tie for the lowest' if len(leaders)>1 else 'has the highest' if maximize else 'has the lowest'} {metric}: {best:.4f} on this split. {'Higher is better.' if maximize else 'Lower is better.'} This describes observed predictions; it does not establish the causal reason for performance.","rows":rows,"caution":"Exploratory holdout comparison, not an unbiased final result. Repeated model selection on this holdout biases the estimate. Close scores may be sampling noise; verify with cross-validation or independent final data before choosing. No algorithm is universally best."}
