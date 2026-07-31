from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    SEVERITY_BY_HIGHWAY_FILE,
    SEVERITY_BY_HOUR_FILE,
    SEVERITY_METRICS_FILE,
    SEVERITY_MODEL_FILE,
)


DEPLOY_NUM = [
    "hour",
    "is_night",
    "is_rush_hour",
    "is_weekend",
    "maxspeed_num",
    "has_cycleway",
    "edge_length_m",
    "near_junction",
    "node_degree",
]
DEPLOY_CAT = ["highway_simple", "season"]
TARGET = "is_ksi"


def build_severity_dataset(snapped: pd.DataFrame, edge_features: pd.DataFrame) -> pd.DataFrame:
    """Build crash-level severity dataset.

    This estimates E[harm | crash], not P(crash | traversal).
    """
    acc = snapped.copy()

    if "is_ksi" not in acc.columns:
        acc["is_ksi"] = acc["accident_severity"].isin([1, 2]).astype(int)
    if "is_fatal" not in acc.columns:
        acc["is_fatal"] = (acc["accident_severity"] == 1).astype(int)

    keep_edges = [
        "edge_uid",
        "highway_simple",
        "maxspeed_num",
        "maxspeed_missing",
        "has_cycleway",
        "edge_length_m",
    ]
    keep_edges = [c for c in keep_edges if c in edge_features.columns]

    # Avoid duplicated columns if snapped already has some edge attributes.
    drop_existing = [c for c in keep_edges if c != "edge_uid" and c in acc.columns]
    acc = acc.drop(columns=drop_existing, errors="ignore")
    acc = acc.merge(edge_features[keep_edges], on="edge_uid", how="left")

    for col in DEPLOY_NUM:
        if col not in acc.columns:
            acc[col] = 0.0
        acc[col] = pd.to_numeric(acc[col], errors="coerce").fillna(0.0)

    for col in DEPLOY_CAT:
        if col not in acc.columns:
            acc[col] = "unknown"
        acc[col] = acc[col].fillna("unknown").astype(str)

    return acc


def _build_model(estimator) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), DEPLOY_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), DEPLOY_CAT),
    ])
    return Pipeline([("pre", pre), ("clf", estimator)])


def _evaluate(name: str, y_true, y_prob) -> dict:
    base = float(np.mean(y_true))
    ap = float(average_precision_score(y_true, y_prob))
    return {
        "model": name,
        "base": round(base, 4),
        "pr_auc": round(ap, 4),
        "lift": round(ap / base, 2) if base > 0 else None,
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
    }


def train_severity_models(
    severity_data: pd.DataFrame,
    metrics_file: str | Path = SEVERITY_METRICS_FILE,
    model_file: str | Path = SEVERITY_MODEL_FILE,
) -> dict:
    """Train crash-severity models using only route-planning-time features.

    This module is evidence, not the main route driver. The notebook shows that
    severity is weakly predictable from deployable features.
    """
    acc = severity_data.copy()
    train = acc[acc["year"] <= 2023].copy()
    test = acc[acc["year"] >= 2024].copy()

    if len(train) == 0 or len(test) == 0:
        raise ValueError("Severity model requires 2018-2023 train and 2024-2025 test data.")

    Xtr, ytr = train[DEPLOY_NUM + DEPLOY_CAT], train[TARGET].astype(int)
    Xte, yte = test[DEPLOY_NUM + DEPLOY_CAT], test[TARGET].astype(int)

    candidates = {
        "baseline_prior": DummyClassifier(strategy="prior"),
        "logistic": LogisticRegression(max_iter=2000),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        ),
        "hist_gb": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=42),
    }

    rows = []
    fitted = {}

    for name, est in candidates.items():
        if name == "baseline_prior":
            model = est.fit(Xtr[DEPLOY_NUM], ytr)
            prob = model.predict_proba(Xte[DEPLOY_NUM])[:, 1]
        else:
            model = _build_model(est).fit(Xtr, ytr)
            prob = model.predict_proba(Xte)[:, 1]
            fitted[name] = model
        rows.append(_evaluate(name, yte, prob))
        print(f"done severity: {name}")

    best_name = max([r for r in rows if r["model"] != "baseline_prior"], key=lambda r: r["pr_auc"])["model"]
    # Calibrate the best ranker. Brier matters because probabilities can be used as a cost factor.
    cal = CalibratedClassifierCV(fitted[best_name], method="isotonic", cv=3).fit(Xtr, ytr)
    prob_cal = cal.predict_proba(Xte)[:, 1]
    rows.append(_evaluate(f"{best_name}_isotonic", yte, prob_cal))

    frac_pos, mean_pred = calibration_curve(yte, prob_cal, n_bins=10, strategy="quantile")
    calibration = [
        {"predicted": float(p), "observed": float(o)}
        for p, o in zip(mean_pred, frac_pos)
    ]

    metrics = {
        "target": TARGET,
        "interpretation": (
            "Severity model estimates KSI conditional on a crash. It is not an occurrence model. "
            "In the notebook, deployable features only weakly predict KSI; severity evidence should "
            "be used for interpretation and limitations, not as a strong route-changing model."
        ),
        "train_years": "2018-2023",
        "test_years": "2024-2025",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "results": rows,
        "best_ranker": best_name,
        "features": DEPLOY_NUM + DEPLOY_CAT,
        "calibration": calibration,
    }

    metrics_file = Path(metrics_file)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    joblib.dump(
        {
            "model": cal,
            "features": DEPLOY_NUM + DEPLOY_CAT,
            "numeric_features": DEPLOY_NUM,
            "categorical_features": DEPLOY_CAT,
            "model_name": f"{best_name}_isotonic",
            "target": TARGET,
        },
        model_file,
    )
    print(f"saved severity model: {model_file}")
    print(f"saved severity metrics: {metrics_file}")
    return metrics


def export_severity_tables(
    severity_data: pd.DataFrame,
    by_hour_file: str | Path = SEVERITY_BY_HOUR_FILE,
    by_highway_file: str | Path = SEVERITY_BY_HIGHWAY_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    acc = severity_data.copy()

    by_hour = (
        acc.groupby("hour")[TARGET]
        .agg(n="size", ksi="sum", rate="mean")
        .reset_index()
    )
    by_hour["rate_pct"] = (by_hour["rate"] * 100).round(2)

    by_highway = (
        acc.groupby("highway_simple")[TARGET]
        .agg(n="size", ksi="sum", rate="mean")
        .reset_index()
        .sort_values("rate", ascending=False)
    )
    by_highway["rate_pct"] = (by_highway["rate"] * 100).round(2)

    by_hour_file = Path(by_hour_file)
    by_highway_file = Path(by_highway_file)
    by_hour_file.parent.mkdir(parents=True, exist_ok=True)

    by_hour.to_csv(by_hour_file, index=False)
    by_highway.to_csv(by_highway_file, index=False)
    print(f"saved severity by hour: {by_hour_file}")
    print(f"saved severity by highway: {by_highway_file}")
    return by_hour, by_highway
