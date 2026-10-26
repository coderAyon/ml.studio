"""Training-only preprocessing and deterministic holdout evaluation."""
from dataclasses import dataclass
import time
import hashlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

ALGORITHMS = {"Logistic Regression": LogisticRegression, "Random Forest": RandomForestClassifier, "Decision Tree": DecisionTreeClassifier, "KNN": KNeighborsClassifier, "SVM": SVC, "Naive Bayes": GaussianNB, "Linear Regression": LinearRegression, "Ridge": Ridge, "Random Forest Regressor": RandomForestRegressor, "Decision Tree Regressor": DecisionTreeRegressor, "SVR": SVR}
REGRESSION = {"Linear Regression", "Ridge", "Random Forest Regressor", "Decision Tree Regressor", "SVR"}


@dataclass
class TrainingResult:
    pipeline: Pipeline
    metrics: dict
    metadata: dict


def train(frame: pd.DataFrame, target: str, algorithm: str, seed: int = 42, test_size: float = .25) -> TrainingResult:
    if target not in frame or algorithm not in ALGORITHMS:
        raise ValueError("Select a valid target and algorithm.")
    if frame[target].isna().any():
        raise ValueError("Target contains missing values. Review these rows before training.")
    x, y = frame.drop(columns=[target]), frame[target]
    if not len(x.columns):
        raise ValueError("At least one feature is required.")
    regression = algorithm in REGRESSION
    if regression:
        y = pd.to_numeric(y, errors="raise")
    elif y.nunique() < 2:
        raise ValueError("Classification requires at least two classes.")
    numeric = list(x.select_dtypes(include="number").columns)
    categorical = [c for c in x if c not in numeric]
    # Conversion is stateless; learned statistics remain entirely inside the pipeline.
    x = x.copy()
    for c in categorical:
        x[c] = x[c].map(lambda v: str(v) if pd.notna(v) else np.nan).astype(object)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=seed, test_size=test_size, stratify=None if regression else y)
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scaler", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=100))]), categorical),
    ])
    model = ALGORITHMS[algorithm]()
    if "random_state" in model.get_params():
        model.set_params(random_state=seed)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    started = time.perf_counter()
    pipeline.fit(x_train, y_train)
    duration = time.perf_counter()-started
    pred = pipeline.predict(x_test)
    if regression:
        metrics = {"MAE": mean_absolute_error(y_test, pred), "MSE": mean_squared_error(y_test, pred), "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))), "R2": r2_score(y_test, pred)}
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, average="weighted", zero_division=0)
        classes = pipeline.named_steps["model"].classes_
        metrics = {"Accuracy": accuracy_score(y_test, pred), "Precision (weighted)": precision, "Recall (weighted)": recall, "F1 (weighted)": f1, "Confusion matrix": confusion_matrix(y_test, pred, labels=classes).tolist(), "Class labels": [str(c) for c in classes], "Per-class report": classification_report(y_test, pred, output_dict=True, zero_division=0)}
        try:
            if len(classes) == 2:
                score = pipeline.predict_proba(x_test)[:, 1] if hasattr(pipeline, "predict_proba") else pipeline.decision_function(x_test)
                metrics["ROC-AUC"] = float(roc_auc_score(y_test == classes[1], score))
            elif hasattr(pipeline, "predict_proba"):
                metrics["ROC-AUC (weighted OVR)"] = float(roc_auc_score(y_test, pipeline.predict_proba(x_test), multi_class="ovr", average="weighted", labels=classes))
            else:
                metrics["ROC-AUC note"] = "This estimator does not expose multiclass probabilities."
        except ValueError as exc:
            metrics["ROC-AUC note"] = str(exc)
    metrics = {k: float(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else (None if isinstance(v, (float, np.floating)) else v) for k, v in metrics.items()}
    overlapping = set(pd.util.hash_pandas_object(x_train, index=False)) & set(pd.util.hash_pandas_object(x_test, index=False))
    warnings = ["Exploratory holdout estimate. Repeated model selection on this holdout biases final performance; reserve independent data for final evaluation."]
    if overlapping:
        warnings.append(f"{len(overlapping)} feature-row fingerprints appear in both splits. Review potential duplicate/group leakage; this is not proof of invalid observations.")
    if len(frame) < 100:
        warnings.append("Small dataset: performance estimates may have high variance.")
    if not regression and y.value_counts(normalize=True).max() > .8:
        warnings.append("Target imbalance: accuracy alone may be misleading. Review per-class recall and F1.")
    return TrainingResult(pipeline, metrics, {"algorithm": algorithm, "target": target, "features": list(x.columns), "dataset_fingerprint": hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest(), "parameters": model.get_params(), "seed": seed, "test_size": test_size, "training_rows": len(x_train), "test_rows": len(x_test), "seconds": duration, "protocol": "Stratified holdout" if not regression else "Random holdout", "warnings": warnings})


def compare(frame: pd.DataFrame, target: str, algorithms: list[str], seed: int = 42) -> list[TrainingResult]:
    """Identical split and preprocessing protocol; no automatic winner selection."""
    if not algorithms:
        raise ValueError("Choose at least one algorithm.")
    if len({a in REGRESSION for a in algorithms}) != 1:
        raise ValueError("Compare models for the same task only.")
    return [train(frame, target, algorithm, seed) for algorithm in algorithms]


def predict(result: TrainingResult, frame: pd.DataFrame):
    """Validate the exact input schema and repeat only stateless type conversion."""
    expected = result.metadata["features"]
    if set(frame.columns) != set(expected):
        missing = sorted(set(expected) - set(frame.columns))
        unexpected = sorted(set(frame.columns) - set(expected))
        raise ValueError(f"Input columns do not match. Missing: {missing}; unexpected: {unexpected}")
    frame = frame[expected].copy()
    transformer = result.pipeline.named_steps["preprocess"]
    numeric = transformer.transformers_[0][2]
    categorical = transformer.transformers_[1][2]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    for column in categorical:
        frame[column] = frame[column].map(lambda v: str(v) if pd.notna(v) else np.nan).astype(object)
    return result.pipeline.predict(frame)


def cluster(frame: pd.DataFrame, algorithm: str, clusters: int = 3) -> dict:
    """Exploratory full-dataset clustering, distinct from held-out evaluation."""
    numeric = frame.select_dtypes(include="number")
    if numeric.empty or len(numeric) < 3:
        raise ValueError("Clustering needs numeric features and at least three rows.")
    if len(numeric) > 10000:
        raise ValueError("Use at most 10,000 rows for this exploratory clustering release.")
    values = Pipeline([("impute", SimpleImputer(keep_empty_features=True)), ("scale", StandardScaler())]).fit_transform(numeric)
    if algorithm == "PCA":
        model = PCA(n_components=min(2, values.shape[1]))
        embedding = model.fit_transform(values)
        return {"algorithm": algorithm, "columns": list(numeric.columns), "explained_variance_ratio": model.explained_variance_ratio_.tolist(), "embedding": embedding.tolist(), "protocol": "Exploratory full-dataset projection"}
    factory = {"K-Means": lambda: KMeans(n_clusters=clusters, random_state=42, n_init=10), "DBSCAN": lambda: DBSCAN(), "Hierarchical": lambda: AgglomerativeClustering(n_clusters=clusters)}
    if algorithm not in factory:
        raise ValueError("Unknown clustering algorithm.")
    model = factory[algorithm]()
    labels = model.fit_predict(values)
    valid = labels != -1
    metrics = {"clusters": len(set(labels) - {-1}), "noise_rows": int((~valid).sum())}
    if 1 < len(set(labels[valid])) < valid.sum():
        metrics["Silhouette"] = float(silhouette_score(values[valid], labels[valid], sample_size=min(2000, int(valid.sum())), random_state=42))
        metrics["Davies-Bouldin"] = float(davies_bouldin_score(values[valid], labels[valid]))
    else:
        metrics["metric_note"] = "Silhouette and Davies-Bouldin require at least two non-noise clusters and more observations than clusters."
    if hasattr(model, "inertia_"):
        metrics["Inertia"] = float(model.inertia_)
    return {"algorithm": algorithm, "columns": list(numeric.columns), "metrics": metrics, "labels": labels.tolist(), "protocol": "Exploratory full-dataset clustering; numeric columns only, median imputation and scaling"}
