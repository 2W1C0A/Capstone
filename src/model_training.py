from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    LEAKAGE_DIAGNOSTICS_FILE,
    LEAKY_FEATURES,
    ML_DATASET_FILE,
    OCCURRENCE_COMPARISON_FILE,
    OCCURRENCE_METRICS_FILE,
    OCCURRENCE_MODEL_FILE,
    RIDEABLE_CLASSES,
)
from .data_pipeline import get_season


TIME_FEATURES = ["month", "hour", "day_of_week", "is_weekend", "is_rush_hour", "is_night"]
ROAD_NUMERIC_FEATURES = ["edge_length_m", "has_cycleway", "maxspeed_num", "maxspeed_missing"]
ROAD_CATEGORICAL_FEATURES = ["highway_simple"]

# The deployed ML model intentionally uses road-only features. Time-only has
# lift ~1.00 in the leakage notebook and does not change route ranking.
DEPLOYED_MODEL_NAME = "road_only"

FEATURE_SETS = {
    "time_only": {
        "numeric": TIME_FEATURES,
        "categorical": [],
        "status": "diagnostic",
        "valid_for_app": False,
    },
    "road_only": {
        "numeric": ROAD_NUMERIC_FEATURES,
        "categorical": ROAD_CATEGORICAL_FEATURES,
        "status": "deployed leakage-safe",
        "valid_for_app": True,
    },
    "deployable_road_time": {
        "numeric": TIME_FEATURES + ROAD_NUMERIC_FEATURES,
        "categorical": ROAD_CATEGORICAL_FEATURES + ["season"],
        "status": "diagnostic; time adds little signal in current sampling design",
        "valid_for_app": False,
    },
    "leaky_diagnostic": {
        "numeric": TIME_FEATURES + ROAD_NUMERIC_FEATURES + sorted(LEAKY_FEATURES),
        "categorical": ROAD_CATEGORICAL_FEATURES + ["season"],
        "status": "invalid target-leakage diagnostic",
        "valid_for_app": False,
    },
}


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_weekend"] = out["day_of_week"].isin([1, 7]).astype(int)
    out["is_rush_hour"] = (out["hour"].between(7, 9) | out["hour"].between(16, 18)).astype(int)
    out["is_night"] = ((out["hour"] >= 22) | (out["hour"] <= 4)).astype(int)
    out["season"] = out["month"].apply(get_season)
    return out


def _clean_edge_table(edge_table: pd.DataFrame) -> pd.DataFrame:
    out = edge_table.copy()

    defaults = {
        "edge_length_m": 1.0,
        "has_cycleway": 0,
        "maxspeed_num": 30.0,
        "maxspeed_missing": 1,
        "highway_simple": "other",
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default

    for col in ["edge_length_m", "has_cycleway", "maxspeed_num", "maxspeed_missing"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(defaults[col])

    out["highway_simple"] = out["highway_simple"].fillna("other").astype(str)

    # Ensure leaky columns exist for diagnostic checks only.
    for col in LEAKY_FEATURES:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    return out


def leakage_diagnostics(
    ml_data: pd.DataFrame,
    output_file: str | Path = LEAKAGE_DIAGNOSTICS_FILE,
) -> dict:
    """Single-feature AUC diagnostics for accident-derived columns."""
    y = ml_data["accident_label"].astype(int)
    rows = []
    for col in sorted(LEAKY_FEATURES):
        if col not in ml_data.columns:
            rows.append({"feature": col, "present": False})
            continue
        try:
            auc = roc_auc_score(y, ml_data[col])
        except Exception:
            auc = None
        rows.append({
            "feature": col,
            "present": True,
            "single_feature_auc": auc,
            "nonzero_positive": float((ml_data.loc[y == 1, col] > 0).mean()),
            "nonzero_negative": float((ml_data.loc[y == 0, col] > 0).mean()),
        })

    # Time feature TVD / AUC diagnostics.
    time_rows = []
    for col in ["hour", "day_of_week", "month"]:
        if col not in ml_data.columns:
            continue
        p = ml_data.loc[y == 1, col].value_counts(normalize=True).sort_index()
        n = ml_data.loc[y == 0, col].value_counts(normalize=True).sort_index()
        a = pd.DataFrame({"pos": p, "neg": n}).fillna(0)
        tvd = 0.5 * (a["pos"] - a["neg"]).abs().sum()
        time_rows.append({
            "feature": col,
            "tvd": float(tvd),
            "single_feature_auc": float(roc_auc_score(y, ml_data[col])),
        })

    result = {
        "leaky_features": rows,
        "time_feature_diagnostics": time_rows,
        "interpretation": (
            "Leaky accident-derived columns must not be used by the deployable model. "
            "Time features are retained for diagnostics but do not carry occurrence signal "
            "under the current negative-sampling design."
        ),
    }
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_ml_dataset(
    snapped: pd.DataFrame,
    edge_risk: pd.DataFrame,
    neg_ratio: float = 1.4,
    restrict_to_rideable: bool = True,
    random_state: int = 42,
    output_file: str | Path = ML_DATASET_FILE,
) -> pd.DataFrame:
    """Build leakage-safe occurrence dataset.

    Positives are observed crashes snapped to OSM edges.
    Negatives are pseudo-negative edge/time combinations with no recorded crash.

    Important limitation:
    This still assumes uniform exposure within the sampled rideable network. Without
    bicycle-volume data, it cannot separate infrastructure risk from where cyclists ride.
    """
    rng = np.random.default_rng(random_state)

    edge_features = _clean_edge_table(edge_risk)

    if restrict_to_rideable:
        edge_pool = edge_features[edge_features["highway_simple"].isin(RIDEABLE_CLASSES)].copy()
        if edge_pool.empty:
            raise ValueError("Rideable edge pool is empty; check highway_simple classification.")
    else:
        edge_pool = edge_features.copy()

    # Keep positives only on rideable classes for the deployed ML comparison.
    pos = snapped.merge(edge_features, on="edge_uid", how="left", suffixes=("", "_edge"))
    if restrict_to_rideable:
        pos = pos[pos["highway_simple"].isin(RIDEABLE_CLASSES)].copy()

    pos = _add_time_features(pos)
    pos["accident_label"] = 1

    pos_keys = set(zip(
        pos["edge_uid"].astype(str),
        pos["year"].astype(int),
        pos["month"].astype(int),
        pos["day_of_week"].astype(int),
        pos["hour"].astype(int),
    ))

    n_neg = int(round(len(pos) * float(neg_ratio)))
    time_pool = pos[["year", "month", "hour", "day_of_week"]].reset_index(drop=True)
    edge_pool = edge_pool.reset_index(drop=True)

    neg_rows = []
    max_trials = max(10_000, n_neg * 30)
    trials = 0

    while len(neg_rows) < n_neg and trials < max_trials:
        trials += 1
        e = edge_pool.iloc[rng.integers(0, len(edge_pool))]
        t = time_pool.iloc[rng.integers(0, len(time_pool))]

        key = (
            str(e["edge_uid"]),
            int(t["year"]),
            int(t["month"]),
            int(t["day_of_week"]),
            int(t["hour"]),
        )
        if key in pos_keys:
            continue

        row = e.to_dict()
        row.update(t.to_dict())
        row["accident_label"] = 0
        neg_rows.append(row)

    neg = pd.DataFrame(neg_rows)
    neg = _add_time_features(neg)

    data = pd.concat([pos, neg], ignore_index=True)
    data = _add_time_features(data)

    # Standard cleanup for model columns.
    all_numeric = sorted(set().union(*(v["numeric"] for v in FEATURE_SETS.values())))
    all_categorical = sorted(set().union(*(v["categorical"] for v in FEATURE_SETS.values())))

    for col in all_numeric:
        if col not in data.columns:
            data[col] = 0.0
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)

    for col in all_categorical:
        if col not in data.columns:
            data[col] = "unknown"
        data[col] = data[col].fillna("unknown").astype(str)

    data["accident_label"] = data["accident_label"].astype(int)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_file, index=False)

    print(f"saved ML dataset: {output_file} | {len(data):,} rows | positive rate {data['accident_label'].mean():.3f}")
    print(data.groupby("accident_label")["highway_simple"].value_counts(normalize=True).unstack(0).fillna(0).mul(100).round(1))
    return data


def topk_recall(y_true, y_score, frac: float = 0.10) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    k = max(1, int(len(y_true) * frac))
    top = np.argsort(-y_score)[:k]
    positives = y_true.sum()
    return float(y_true[top].sum() / positives) if positives else 0.0


def _make_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    transformers = []
    if numeric:
        transformers.append(("num", StandardScaler(), numeric))
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
    pre = ColumnTransformer(transformers)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def _evaluate(y_true, y_prob) -> dict:
    base = float(np.mean(y_true))
    ap = float(average_precision_score(y_true, y_prob))
    try:
        roc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc = None
    return {
        "base": round(base, 4),
        "pr_auc": round(ap, 4),
        "lift": round(ap / base, 2) if base > 0 else None,
        "roc_auc": round(roc, 4) if roc is not None else None,
        "top10_recall": round(topk_recall(y_true, y_prob, 0.10), 3),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
    }


def train_occurrence_models(
    ml_data: pd.DataFrame,
    test_year: int | None = None,
    model_file: str | Path = OCCURRENCE_MODEL_FILE,
    metrics_file: str | Path = OCCURRENCE_METRICS_FILE,
    comparison_file: str | Path = OCCURRENCE_COMPARISON_FILE,
) -> tuple[dict, pd.DataFrame]:
    """Train diagnostic and deployed occurrence models.

    The saved deployed model is road_only. The leaky model is trained only as a
    labelled diagnostic to show how target leakage inflated the previous score.
    """
    data = ml_data.copy()
    if test_year is None:
        test_year = int(data["year"].max())

    train_mask = data["year"] < test_year
    test_mask = data["year"] == test_year
    if train_mask.sum() < 100 or test_mask.sum() < 50:
        raise ValueError("Not enough data for forward train/test split.")

    results = []
    fitted = {}

    for name, cfg in FEATURE_SETS.items():
        numeric = [c for c in cfg["numeric"] if c in data.columns]
        categorical = [c for c in cfg["categorical"] if c in data.columns]
        cols = numeric + categorical

        # Skip if no features.
        if not cols:
            continue

        model = _make_pipeline(numeric, categorical)
        Xtr, ytr = data.loc[train_mask, cols], data.loc[train_mask, "accident_label"].astype(int)
        Xte, yte = data.loc[test_mask, cols], data.loc[test_mask, "accident_label"].astype(int)

        model.fit(Xtr, ytr)
        y_prob = model.predict_proba(Xte)[:, 1]

        row = {
            "model": name,
            "n_features": len(cols),
            "numeric_features": numeric,
            "categorical_features": categorical,
            "status": cfg["status"],
            "valid_for_app": cfg["valid_for_app"],
            "test_year": int(test_year),
            "n_train": int(len(ytr)),
            "n_test": int(len(yte)),
        }
        row.update(_evaluate(yte, y_prob))
        results.append(row)
        fitted[name] = {
            "model": model,
            "numeric_features": numeric,
            "categorical_features": categorical,
            "features": cols,
            "model_name": name,
            "status": cfg["status"],
        }
        print(f"done: {name}")

    comparison = pd.DataFrame(results)
    comparison_file = Path(comparison_file)
    comparison_file.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_file, index=False)

    deployed = fitted[DEPLOYED_MODEL_NAME]
    joblib.dump(deployed, model_file)

    # Save compact metrics for app.
    diagnostics = leakage_diagnostics(data)
    metrics = {
        "deployed_model": DEPLOYED_MODEL_NAME,
        "leakage_safe": True,
        "negative_sampling": "restricted_to_rideable_classes",
        "excluded_leaky_features": sorted(LEAKY_FEATURES),
        "comparison": results,
        "deployed_features": deployed["features"],
        "deployed_numeric_features": deployed["numeric_features"],
        "deployed_categorical_features": deployed["categorical_features"],
        "leakage_diagnostics_file": str(LEAKAGE_DIAGNOSTICS_FILE),
        "model_comparison_file": str(comparison_file),
    }
    metrics_file = Path(metrics_file)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"saved deployed occurrence model: {model_file}")
    print(f"saved model comparison: {comparison_file}")
    return deployed, comparison


def predict_edge_occurrence_risk(edge_table: pd.DataFrame, model_bundle: dict | str | Path = OCCURRENCE_MODEL_FILE) -> pd.DataFrame:
    """Predict road-only occurrence risk for every edge."""
    if not isinstance(model_bundle, dict):
        model_bundle = joblib.load(model_bundle)

    model = model_bundle["model"]
    features = model_bundle["features"]

    edges = _clean_edge_table(edge_table)
    for col in features:
        if col not in edges.columns:
            edges[col] = 0.0 if col not in ROAD_CATEGORICAL_FEATURES else "unknown"

    out = edges.copy()
    out["ml_occurrence_risk"] = model.predict_proba(out[features])[:, 1]
    return out
