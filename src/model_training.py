from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    EDGE_RISK_FILE,
    ML_DATASET_FILE,
    MODEL_FILE,
    METRICS_FILE,
)
from .data_pipeline import get_season


NUMERIC_FEATURES = [
    "month", "hour", "day_of_week",
    "is_weekend", "is_rush_hour", "is_night",
    "edge_length_m", "has_cycleway", "maxspeed_num",
    "accident_count", "severity_sum", "serious_fatal_count",
    "historical_risk_norm", "node_risk_raw", "combined_spatial_risk",
]

CATEGORICAL_FEATURES = [
    "season",
    "highway_simple",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_weekend"] = out["day_of_week"].isin([1, 7]).astype(int)
    out["is_rush_hour"] = (
        out["hour"].between(7, 9) | out["hour"].between(16, 18)
    ).astype(int)
    out["is_night"] = (
        out["hour"].between(20, 23) | out["hour"].between(0, 5)
    ).astype(int)
    out["season"] = out["month"].apply(get_season)
    return out


def _clean_edge_risk(edge_risk: pd.DataFrame) -> pd.DataFrame:
    out = edge_risk.copy()

    # Ensure required model columns exist.
    defaults = {
        "edge_length_m": "length",
        "has_cycleway": 0,
        "maxspeed_num": 30.0,
        "accident_count": 0.0,
        "severity_sum": 0.0,
        "serious_fatal_count": 0.0,
        "historical_risk_norm": 0.0,
        "node_risk_raw": 0.0,
        "combined_spatial_risk": 0.0,
        "highway_simple": "other",
    }

    if "edge_length_m" not in out.columns:
        if "length" in out.columns:
            out["edge_length_m"] = out["length"]
        else:
            out["edge_length_m"] = 1.0

    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default

    for col in [c for c in defaults if c != "highway_simple"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(defaults[col] if not isinstance(defaults[col], str) else 0)

    out["highway_simple"] = out["highway_simple"].fillna("other").astype(str)
    return out


def build_ml_dataset(
    accidents: pd.DataFrame,
    snapped: pd.DataFrame,
    edge_risk: pd.DataFrame,
    neg_ratio: int = 3,
    random_state: int = 42,
    output_file: str | Path = ML_DATASET_FILE,
) -> pd.DataFrame:
    """Build supervised ML dataset.

    Positive samples:
        observed Unfallatlas accident at segment/time.

    Negative samples:
        OSM segment/time combinations with no observed Unfallatlas accident.
        These are pseudo-negative samples, not guaranteed-safe road segments.
    """
    rng = np.random.default_rng(random_state)
    edge_features = _clean_edge_risk(edge_risk)

    # Merge snapped positive accidents with edge features.
    pos_cols = [
        "year", "month", "hour", "day_of_week",
        "is_weekend", "is_rush_hour", "is_night", "season",
        "edge_uid",
    ]
    pos = snapped[pos_cols].copy()
    pos = pos.merge(edge_features, on="edge_uid", how="left", suffixes=("", "_edge"))
    pos["accident_label"] = 1

    # Positive key for exclusion.
    positive_keys = set(
        zip(pos["edge_uid"].astype(str), pos["year"], pos["month"], pos["day_of_week"], pos["hour"])
    )

    # Sample times from observed accidents, edges from all OSM edges.
    n_neg = len(pos) * int(neg_ratio)
    neg_rows = []

    edge_sample_pool = edge_features.reset_index(drop=True)
    time_pool = pos[["year", "month", "hour", "day_of_week"]].reset_index(drop=True)

    max_trials = n_neg * 20
    trials = 0

    while len(neg_rows) < n_neg and trials < max_trials:
        trials += 1
        e = edge_sample_pool.iloc[rng.integers(0, len(edge_sample_pool))]
        t = time_pool.iloc[rng.integers(0, len(time_pool))]

        key = (str(e["edge_uid"]), t["year"], t["month"], t["day_of_week"], t["hour"])
        if key in positive_keys:
            continue

        row = e.to_dict()
        row.update(t.to_dict())
        row["accident_label"] = 0
        neg_rows.append(row)

    neg = pd.DataFrame(neg_rows)

    data = pd.concat([pos, neg], ignore_index=True)
    data = _add_time_features(data)

    for col in NUMERIC_FEATURES:
        if col not in data.columns:
            data[col] = 0.0
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)

    for col in CATEGORICAL_FEATURES:
        if col not in data.columns:
            data[col] = "unknown"
        data[col] = data[col].fillna("unknown").astype(str)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_file, index=False)
    print(f"Saved ML dataset: {output_file} ({data.shape[0]:,} rows × {data.shape[1]} columns)")
    print(data["accident_label"].value_counts(normalize=True).rename("class_share").round(3))
    return data


def topk_recall(y_true: pd.Series, y_score: np.ndarray, top_frac: float = 0.10) -> float:
    n = len(y_true)
    k = max(1, int(n * top_frac))
    order = np.argsort(-y_score)
    top_idx = order[:k]
    positives = np.sum(y_true)
    if positives == 0:
        return 0.0
    return float(np.sum(np.asarray(y_true)[top_idx]) / positives)


def train_risk_model(
    ml_data: pd.DataFrame,
    model_file: str | Path = MODEL_FILE,
    metrics_file: str | Path = METRICS_FILE,
    model_type: str = "random_forest",
) -> tuple[Pipeline, dict]:
    """Train accident occurrence model with a time-based split."""
    data = ml_data.copy()

    # Use last available year for testing.
    max_year = int(data["year"].max())
    train_mask = data["year"] < max_year
    test_mask = data["year"] == max_year

    if train_mask.sum() < 100 or test_mask.sum() < 50:
        # Fallback if only one year exists.
        rng = np.random.default_rng(42)
        mask = rng.random(len(data)) < 0.8
        train_mask = mask
        test_mask = ~mask

    X_train = data.loc[train_mask, MODEL_FEATURES].copy()
    X_test = data.loc[test_mask, MODEL_FEATURES].copy()
    y_train = data.loc[train_mask, "accident_label"].astype(int)
    y_test = data.loc[test_mask, "accident_label"].astype(int)

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])

    if model_type == "logistic":
        estimator = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=None,
        )
    else:
        estimator = RandomForestClassifier(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "roc_auc": float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else None,
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "top10_recall": topk_recall(y_test, y_prob, 0.10),
        "top20_recall": topk_recall(y_test, y_prob, 0.20),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "features": MODEL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }

    model_file = Path(model_file)
    metrics_file = Path(metrics_file)
    model_file.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_file)
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model: {model_file}")
    print(f"Saved metrics: {metrics_file}")
    print(f"ROC-AUC: {metrics['roc_auc']}")
    print(f"PR-AUC: {metrics['pr_auc']:.3f}")
    print(f"Top-10% recall: {metrics['top10_recall']:.3f}")
    return model, metrics


def make_edge_prediction_table(
    edge_risk: pd.DataFrame,
    model: Pipeline,
    month: int = 7,
    hour: int = 8,
    day_of_week: int = 3,
    output_file: str | Path = EDGE_RISK_FILE,
) -> pd.DataFrame:
    """Predict ML accident risk for each edge under a chosen time condition."""
    edges = _clean_edge_risk(edge_risk)
    pred = edges.copy()

    pred["month"] = month
    pred["hour"] = hour
    pred["day_of_week"] = day_of_week
    pred = _add_time_features(pred)

    for col in NUMERIC_FEATURES:
        if col not in pred.columns:
            pred[col] = 0.0
        pred[col] = pd.to_numeric(pred[col], errors="coerce").fillna(0.0)

    for col in CATEGORICAL_FEATURES:
        if col not in pred.columns:
            pred[col] = "unknown"
        pred[col] = pred[col].fillna("unknown").astype(str)

    pred["ml_accident_risk"] = model.predict_proba(pred[MODEL_FEATURES])[:, 1]
    pred["risk_score"] = pred["ml_accident_risk"]

    # Keep historical risk too.
    if "combined_spatial_risk" in pred.columns:
        pred["historical_risk"] = pred["combined_spatial_risk"]

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(output_file, index=False)
    print(f"Saved edge ML risk scores: {output_file}")
    return pred


def load_model(model_file: str | Path = MODEL_FILE) -> Pipeline:
    return joblib.load(model_file)
