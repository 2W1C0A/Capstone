from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd

from .config import (
    EDGE_RISK_FILE,
    NODE_RADIUS_M,
    NODE_RISK_FILE,
    OSM_EDGE_FEATURES_FILE,
    ROUTE_RISK_FILE,
    SNAP_MAX_DIST_M,
    SNAPPED_ACCIDENTS_FILE,
    TEMPORAL_VALIDATION_FILE,
)
from .osm_network import edge_uid, pair_id


# BASt-style severity-cost ratio used for the historical GIS baseline.
SEVERITY_COST = {
    "minor_injury": 1.0,
    "serious_injury": 23.0,
    "fatal": 222.0,
}


def severity_weight(label: str) -> float:
    return float(SEVERITY_COST.get(str(label), 1.0))


def ensure_accident_outcome_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure outcome columns required by GIS risk aggregation exist.

    Compatible with your cleaned CSV schema:
        year, month, hour, day_of_week, accident_severity, ..., is_ksi, severity_label, ...

    Required downstream columns:
        is_ksi
        is_fatal
        severity_label
        severity_weight
    """
    out = df.copy()

    if "accident_severity" in out.columns:
        sev = pd.to_numeric(out["accident_severity"], errors="coerce")
    else:
        sev = None

    # KSI = killed or seriously injured.
    if "is_ksi" not in out.columns:
        if "serious_or_fatal" in out.columns:
            out["is_ksi"] = (
                pd.to_numeric(out["serious_or_fatal"], errors="coerce")
                .fillna(0)
                .astype(int)
            )
        elif sev is not None:
            out["is_ksi"] = sev.isin([1, 2]).astype(int)
        else:
            raise ValueError(
                "Accident data must contain one of: is_ksi, serious_or_fatal, or accident_severity."
            )
    else:
        out["is_ksi"] = pd.to_numeric(out["is_ksi"], errors="coerce").fillna(0).astype(int)

    # Fatal = accident_severity == 1.
    if "is_fatal" not in out.columns:
        if sev is not None:
            out["is_fatal"] = (sev == 1).astype(int)
        else:
            # Cannot recover fatalities if only a binary KSI column exists.
            out["is_fatal"] = 0
            out["is_fatal_imputed_missing"] = 1
    else:
        out["is_fatal"] = pd.to_numeric(out["is_fatal"], errors="coerce").fillna(0).astype(int)

    if "severity_label" not in out.columns:
        if sev is not None:
            sev_map = {1: "fatal", 2: "serious_injury", 3: "minor_injury"}
            out["severity_label"] = sev.map(sev_map).fillna("unknown")
        else:
            out["severity_label"] = out["is_ksi"].map({1: "serious_injury", 0: "minor_injury"})
    else:
        out["severity_label"] = out["severity_label"].fillna("unknown").astype(str)

    if "severity_weight" not in out.columns:
        out["severity_weight"] = out["severity_label"].map(severity_weight).fillna(1.0)
    else:
        out["severity_weight"] = pd.to_numeric(out["severity_weight"], errors="coerce")
        fallback = out["severity_label"].map(severity_weight).fillna(1.0)
        out["severity_weight"] = out["severity_weight"].fillna(fallback)

    return out


def _normalise_nearest_edge_results(edge_ids: Any, n_expected: int) -> pd.DataFrame:
    """OSMnx compatibility wrapper for nearest_edges return formats."""
    if isinstance(edge_ids, tuple) and len(edge_ids) == 3:
        u_arr, v_arr, k_arr = list(edge_ids[0]), list(edge_ids[1]), list(edge_ids[2])
        if len(u_arr) == n_expected and len(v_arr) == n_expected and len(k_arr) == n_expected:
            return pd.DataFrame({"u": u_arr, "v": v_arr, "key": k_arr})

    edge_list = list(edge_ids)
    if len(edge_list) != n_expected:
        raise ValueError(f"nearest_edges returned {len(edge_list)} edges for {n_expected} points")

    rows = []
    for e in edge_list:
        if isinstance(e, (tuple, list)) and len(e) >= 3:
            rows.append((e[0], e[1], e[2]))
        else:
            raise ValueError(f"Unexpected nearest edge item: {e!r}")
    return pd.DataFrame(rows, columns=["u", "v", "key"])


def _normalise_distance_results(dist: Any, n_expected: int) -> list[float]:
    if hasattr(dist, "__len__") and not isinstance(dist, (str, bytes)):
        dist_list = list(dist)
    else:
        dist_list = [float(dist)]
    if len(dist_list) != n_expected:
        raise ValueError(f"nearest_edges returned {len(dist_list)} distances for {n_expected} points")
    return [float(x) for x in dist_list]


def add_junction_features(Gp, snapped: pd.DataFrame, radius_m: float = NODE_RADIUS_M) -> pd.DataFrame:
    """Add nearest node, node distance, node degree and near_junction flag."""
    pts = gpd.GeoSeries(
        gpd.points_from_xy(snapped["longitude"], snapped["latitude"]),
        crs="EPSG:4326",
    ).to_crs(Gp.graph["crs"])

    nodes, dist = ox.distance.nearest_nodes(
        Gp,
        X=pts.x.to_numpy(),
        Y=pts.y.to_numpy(),
        return_dist=True,
    )
    degree = dict(Gp.degree())

    out = snapped.copy()
    out["nearest_node"] = list(nodes)
    out["node_dist_m"] = np.asarray(dist, dtype=float)
    out["node_degree"] = [int(degree.get(n, 0)) for n in nodes]
    out["near_junction"] = (out["node_dist_m"] <= float(radius_m)).astype(int)
    return out


def snap_accidents_to_edges(
    Gp,
    accidents: pd.DataFrame,
    max_snap_dist_m: float = SNAP_MAX_DIST_M,
    output_file: str | Path = SNAPPED_ACCIDENTS_FILE,
) -> pd.DataFrame:
    """Snap Unfallatlas accident points to nearest OSM edge in projected CRS."""
    required = {"longitude", "latitude"}
    missing = required.difference(accidents.columns)
    if missing:
        raise ValueError(f"Accident data missing required columns: {sorted(missing)}")

    pts = gpd.GeoDataFrame(
        accidents.copy().reset_index(drop=True),
        geometry=gpd.points_from_xy(accidents["longitude"], accidents["latitude"]),
        crs="EPSG:4326",
    ).to_crs(Gp.graph["crs"])

    nearest = ox.distance.nearest_edges(
        Gp,
        X=pts.geometry.x.to_numpy(),
        Y=pts.geometry.y.to_numpy(),
        return_dist=True,
    )
    edge_ids, dist = nearest
    edge_df = _normalise_nearest_edge_results(edge_ids, len(pts))

    out = pts.drop(columns=["geometry"]).copy()
    out = ensure_accident_outcome_columns(out)

    out[["u", "v", "key"]] = edge_df[["u", "v", "key"]].to_numpy()
    out["snap_dist_m"] = _normalise_distance_results(dist, len(pts))
    out = out[out["snap_dist_m"] <= max_snap_dist_m].copy()

    out["edge_uid"] = [edge_uid(u, v, k) for u, v, k in zip(out["u"], out["v"], out["key"])]
    out["pair_id"] = [pair_id(u, v) for u, v in zip(out["u"], out["v"])]

    # Re-run after filtering to guarantee all derived fields are present.
    out = ensure_accident_outcome_columns(out)
    out = add_junction_features(Gp, out)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_file, index=False)

    print(
        f"snapped {len(out):,} / {len(accidents):,} accidents within {max_snap_dist_m:.0f} m "
        f"({len(out)/len(accidents):.1%}); median offset {out['snap_dist_m'].median():.1f} m"
    )
    print(f"near junction within {NODE_RADIUS_M:.0f} m: {out['near_junction'].mean():.1%}")
    return out


def _load_edge_features(edge_features: pd.DataFrame | None = None) -> pd.DataFrame:
    if edge_features is not None:
        return edge_features.copy()
    return pd.read_csv(OSM_EDGE_FEATURES_FILE)


def partition_node_edge_accidents(
    snapped: pd.DataFrame,
    node_radius_m: float = NODE_RADIUS_M,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition accidents: an accident belongs to a junction or a link, never both."""
    snapped = ensure_accident_outcome_columns(snapped)
    if "node_dist_m" not in snapped.columns:
        raise ValueError("snapped accident data must contain node_dist_m")
    near_node = snapped["node_dist_m"] <= float(node_radius_m)
    node_accidents = snapped[near_node].copy()
    edge_accidents = snapped[~near_node].copy()
    return node_accidents, edge_accidents


def build_node_risk(
    node_accidents: pd.DataFrame,
    output_file: str | Path = NODE_RISK_FILE,
) -> pd.DataFrame:
    """Build severity-weighted node/junction risk from node-assigned accidents."""
    node_accidents = ensure_accident_outcome_columns(node_accidents)

    if len(node_accidents) == 0:
        node_risk = pd.DataFrame(
            columns=[
                "nearest_node",
                "node_accidents",
                "node_severity_sum",
                "node_ksi_count",
                "node_fatal_count",
                "node_risk_norm",
            ]
        )
    else:
        node_agg_dict = {
            "node_accidents": ("nearest_node", "size"),
            "node_severity_sum": ("severity_weight", "sum"),
            "node_ksi_count": ("is_ksi", "sum"),
            "node_fatal_count": ("is_fatal", "sum"),
        }

        node_risk = (
            node_accidents.groupby("nearest_node")
            .agg(**node_agg_dict)
            .reset_index()
        )

        cap = node_risk["node_severity_sum"].quantile(0.95)
        if not np.isfinite(cap) or cap <= 0:
            cap = node_risk["node_severity_sum"].max()
        node_risk["node_risk_norm"] = (node_risk["node_severity_sum"] / cap).clip(0, 1) if cap > 0 else 0.0

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    node_risk.to_csv(output_file, index=False)
    print(f"saved node risk: {output_file} | {len(node_risk):,} risky nodes")
    return node_risk


def build_edge_risk(
    edge_accidents: pd.DataFrame,
    edge_features: pd.DataFrame | None = None,
    shrinkage: float = 5.0,
    output_file: str | Path = EDGE_RISK_FILE,
    write: bool = True,
) -> pd.DataFrame:
    """Build undirected segment historical GIS risk with empirical-Bayes shrinkage.

    `write=False` skips persisting the CSV; use it for the shrinkage-robustness
    loop (nb 02 [14]) so repeated calls do not overwrite the deployed m=5 file
    with a later variant.
    """
    edge_accidents = ensure_accident_outcome_columns(edge_accidents)
    edges = _load_edge_features(edge_features)

    if "pair_id" not in edges.columns:
        edges["pair_id"] = [pair_id(u, v) for u, v in zip(edges["u"], edges["v"])]

    # Use one record per undirected segment to avoid directed-edge bias.
    seg = (
        edges.sort_values("edge_length_m", ascending=False)
        .drop_duplicates("pair_id")
        .copy()
    )
    seg["len100"] = (
        pd.to_numeric(seg["edge_length_m"], errors="coerce").fillna(1.0) / 100.0
    ).clip(lower=0.01)

    required = {"pair_id", "severity_weight", "is_ksi", "is_fatal"}
    missing = required.difference(edge_accidents.columns)
    if missing and len(edge_accidents) > 0:
        raise ValueError(f"edge_accidents missing required columns after derivation: {sorted(missing)}")

    if len(edge_accidents) > 0:
        agg = (
            edge_accidents.groupby("pair_id")
            .agg(
                accident_count=("pair_id", "size"),
                severity_sum=("severity_weight", "sum"),
                serious_fatal_count=("is_ksi", "sum"),
                fatal_count=("is_fatal", "sum"),
            )
            .reset_index()
        )
    else:
        agg = pd.DataFrame(
            columns=[
                "pair_id",
                "accident_count",
                "severity_sum",
                "serious_fatal_count",
                "fatal_count",
            ]
        )

    risk = seg.merge(agg, on="pair_id", how="left")
    for col in ["accident_count", "severity_sum", "serious_fatal_count", "fatal_count"]:
        risk[col] = pd.to_numeric(risk[col], errors="coerce").fillna(0.0)

    # Highway-class prior on undirected segments.
    class_prior = (
        risk.groupby("highway_simple")["severity_sum"].sum()
        / risk.groupby("highway_simple")["len100"].sum().replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    default_prior = float(class_prior.mean()) if len(class_prior) else 0.0
    risk["class_prior"] = risk["highway_simple"].map(class_prior).fillna(default_prior)
    risk["risk_raw"] = risk["severity_sum"] / risk["len100"]
    risk["historical_risk"] = (
        risk["severity_sum"] + shrinkage * risk["class_prior"] * risk["len100"]
    ) / (risk["len100"] + shrinkage)

    # Normalise on a log scale with a high (p99.9) cap. A linear p95 cap
    # saturates catastrophically here: ~91% of segments have zero crashes, so
    # p95 lands just above zero and a single serious injury (severity_weight 23)
    # or fatality (222) blows past it, clipping ~17k segments to exactly 1.0.
    # The router then cannot tell a lone minor injury from a fatal cluster, and
    # the historical score collapses to a binary "had a crash" flag — which the
    # forward validation shows is *beaten* by the naive past-crash baseline.
    # log1p + p99.9 keeps the BASt severity weighting (1:23:222) and the
    # empirical-Bayes shrinkage as a graded signal. See docs finding #7.
    log_risk = np.log1p(risk["historical_risk"].clip(lower=0.0))
    cap = log_risk.quantile(0.999)
    if not np.isfinite(cap) or cap <= 0:
        cap = log_risk.max()
    risk["historical_risk_norm"] = (log_risk / cap).clip(0, 1) if cap > 0 else 0.0

    # Merge risk back to directed edges for routing.
    edge_risk = edges.merge(
        risk[
            [
                "pair_id",
                "accident_count",
                "severity_sum",
                "serious_fatal_count",
                "fatal_count",
                "len100",
                "class_prior",
                "risk_raw",
                "historical_risk",
                "historical_risk_norm",
            ]
        ],
        on="pair_id",
        how="left",
    )

    for col in ["accident_count", "severity_sum", "serious_fatal_count", "fatal_count", "historical_risk_norm"]:
        edge_risk[col] = pd.to_numeric(edge_risk[col], errors="coerce").fillna(0.0)

    if write:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        edge_risk.to_csv(output_file, index=False)
        print(f"saved edge risk: {output_file} | {len(edge_risk):,} directed edges")
    else:
        print(f"built edge risk (not written) | {len(edge_risk):,} directed edges")
    return edge_risk


def add_node_risk_to_edges(edge_risk: pd.DataFrame, node_risk: pd.DataFrame) -> pd.DataFrame:
    """Attach node risk to edge endpoints and build a combined spatial risk score."""
    out = edge_risk.copy()

    if len(node_risk) == 0:
        out["node_risk_raw"] = 0.0
        out["node_risk_norm"] = 0.0
    else:
        lookup = dict(zip(node_risk["nearest_node"].astype(str), node_risk["node_risk_norm"].astype(float)))
        out["u_node_risk"] = out["u"].astype(str).map(lookup).fillna(0.0)
        out["v_node_risk"] = out["v"].astype(str).map(lookup).fillna(0.0)
        out["node_risk_norm"] = out[["u_node_risk", "v_node_risk"]].max(axis=1)
        out["node_risk_raw"] = out["node_risk_norm"]

    # GIS baseline only. This column must not be used as a deployable ML feature.
    out["combined_spatial_risk"] = (
        0.75 * out["historical_risk_norm"] + 0.25 * out["node_risk_norm"]
    ).clip(0, 1)
    return out


def apply_historical_risk_cost(
    Gp,
    route_risk: pd.DataFrame,
    alpha: float = 2.0,
    node_penalty_m: float = 40.0,
):
    """Write historical risk costs to graph edges.

    risk_cost = length × (1 + alpha × edge risk) + node_penalty_m × node risk
    """
    lookup = route_risk.set_index("edge_uid").to_dict(orient="index")

    for u, v, k, data in Gp.edges(keys=True, data=True):
        uid = edge_uid(u, v, k)
        row = lookup.get(uid, {})
        length = float(data.get("length", data.get("edge_length_m", 1.0)))
        hist = float(row.get("historical_risk_norm", 0.0))
        node = float(row.get("node_risk_norm", 0.0))
        combined = float(row.get("combined_spatial_risk", 0.0))

        data["length_cost"] = length
        data["historical_risk"] = hist
        data["junction_risk"] = node
        data["combined_spatial_risk"] = combined
        data["historical_risk_cost"] = length * (1.0 + alpha * hist) + node_penalty_m * node

    return Gp


def build_spatial_risk_pipeline(
    Gp,
    accidents: pd.DataFrame,
    edge_features: pd.DataFrame | None = None,
    node_radius_m: float = NODE_RADIUS_M,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full improved GIS baseline pipeline."""
    accidents = ensure_accident_outcome_columns(accidents)

    snapped = snap_accidents_to_edges(Gp, accidents)
    node_accidents, edge_accidents = partition_node_edge_accidents(snapped, node_radius_m=node_radius_m)

    edge_risk = build_edge_risk(edge_accidents, edge_features=edge_features)
    node_risk = build_node_risk(node_accidents)
    route_risk = add_node_risk_to_edges(edge_risk, node_risk)

    ROUTE_RISK_FILE.parent.mkdir(parents=True, exist_ok=True)
    route_risk.to_csv(ROUTE_RISK_FILE, index=False)
    print(f"saved route risk table: {ROUTE_RISK_FILE}")
    return snapped, node_risk, route_risk


def _surface_topk_recall(
    score_by_pair, test_pairs, frac: float = 0.10, n_boot: int = 1000, seed: int = 0
) -> dict:
    """Top-`frac` segments by score (by COUNT, matching the historical metric):
    share of held-out crashes that land on them, plus lift over the random `frac`
    baseline and a 95% bootstrap CI. `score_by_pair` is indexed by pair_id and
    defines the universe.

    The top-`frac` flagged set is fixed (the model does not change); the only
    sampling uncertainty is which crashes occurred, so the CI comes from resampling
    the held-out crashes with replacement. A fixed seed keeps it reproducible and
    pairs the resamples across surfaces (same resampled crash sets), which is the
    right basis for comparing surfaces.
    """
    s = score_by_pair.dropna()
    empty = {"top_decile_recall": None, "lift_over_random": None, "ci_low": None, "ci_high": None}
    if s.empty:
        return empty
    n_top = max(1, int(round(len(s) * frac)))
    top = set(s.sort_values(ascending=False).head(n_top).index.astype(str))
    tp = test_pairs.astype(str)
    if len(tp) == 0:
        return empty

    hit = tp.isin(top).to_numpy().astype(float)
    recall = float(hit.mean())
    lift = float(recall / frac)

    rng = np.random.default_rng(seed)
    n = len(hit)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        boot[b] = hit[rng.integers(0, n, n)].mean()
    ci_low, ci_high = (float(x) for x in np.percentile(boot, [2.5, 97.5]))

    return {
        "top_decile_recall": recall,
        "lift_over_random": lift,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def _paired_diff_ci(hit_a, hit_b, n_boot: int = 1000, seed: int = 0):
    """95% bootstrap CI of mean(hit_a - hit_b), paired over the same resamples.

    `hit_a`/`hit_b` are 0/1 arrays aligned crash-for-crash (same held-out crashes,
    same order), so resampling the per-crash difference keeps the comparison
    paired: it cancels the shared sampling wobble and tests the leader against the
    runner-up directly. `significant` is True when the CI stays above 0, i.e. the
    leader beats the runner-up beyond noise even if their marginal CIs overlap.
    """
    a = np.asarray(hit_a, dtype=float)
    b = np.asarray(hit_b, dtype=float)
    if len(a) == 0 or len(a) != len(b):
        return None
    d = a - b
    n = len(d)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = d[rng.integers(0, n, n)].mean()
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    return {
        "diff": float(d.mean()),
        "diff_ci_low": lo,
        "diff_ci_high": hi,
        "significant": bool(lo > 0),
    }


def _occurrence_surface_by_pair(train_snap, edge_risk, edge_features, output_file):
    """Refit the leakage-safe road-only occurrence model on train-only crashes and
    return its per-segment risk (max over the pair's directed edges)."""
    from .model_training import (
        ROAD_CATEGORICAL_FEATURES,
        ROAD_NUMERIC_FEATURES,
        _make_pipeline,
        build_ml_dataset,
        predict_edge_occurrence_risk,
    )

    ml = build_ml_dataset(
        train_snap,
        edge_risk,
        output_file=Path(output_file).with_suffix(".ml_train.csv"),
    )
    numeric = [c for c in ROAD_NUMERIC_FEATURES if c in ml.columns]
    categorical = [c for c in ROAD_CATEGORICAL_FEATURES if c in ml.columns]
    cols = numeric + categorical
    pipe = _make_pipeline(numeric, categorical)
    pipe.fit(ml[cols], ml["accident_label"].astype(int))

    scored = predict_edge_occurrence_risk(edge_features, {"model": pipe, "features": cols})
    return scored.groupby(scored["pair_id"].astype(str))["ml_occurrence_risk"].max()


def _frequency_surface_by_pair(train_snap, edge_features, Gp):
    """Refit the negative-binomial SPF on train-only crash counts and return its
    per-segment expected crash count (identical for both directions of a pair)."""
    from .frequency_model import (
        add_junction_features,
        build_segment_table,
        fit_frequency_model,
        predict_edge_expected_crashes,
    )

    seg = build_segment_table(edges=edge_features, snapped=train_snap, verbose=False)
    seg = add_junction_features(seg, Gp, edge_lookup=edge_features)
    bundle, _ = fit_frequency_model(seg, save=False)

    edges_j = add_junction_features(edge_features, Gp)
    pred = predict_edge_expected_crashes(edges_j, bundle, strict=False)
    return pred.groupby(pred["pair_id"].astype(str))["expected_crashes"].first()


def temporal_validation(
    Gp,
    accidents: pd.DataFrame,
    edge_features: pd.DataFrame,
    train_end_year: int = 2023,
    test_start_year: int = 2024,
    output_file: str | Path = TEMPORAL_VALIDATION_FILE,
    compare_surfaces: bool = True,
) -> dict:
    """Forward validation of the risk surfaces.

    Build risk from earlier years and measure how many future crashes land in the
    top decile of the past-risk ranking. Random expectation is 10%.

    The historical metric and its length/memory baselines are unchanged. When
    `compare_surfaces` is True, a `surfaces` list additionally scores three surfaces
    on the SAME held-out crashes over a common segment universe with the same
    top-decile-by-count rule, so they are directly comparable:

        historical_gis  — severity-weighted empirical-Bayes spatial risk
        occurrence_ml   — leakage-safe road-only occurrence model (refit on train)
        frequency_nb    — negative-binomial SPF, expected crashes (refit on train)

    Each surface also gets an `nbc_recall`: the same top-decile metric measured only
    over held-out crashes on segments with NO crash in the training window
    ("never-before-crashed"). A historical crash map is blind on those segments, so
    this subset is the honest test for a road-feature model and is where the
    structural surfaces (occurrence, frequency) are expected to overtake historical.

    Each is rebuilt from crashes up to `train_end_year` only and scored on crashes
    from `test_start_year` on that it was not fit on. The occurrence and frequency
    surfaces are best-effort: if either fails it is omitted from the list and the
    historical validation still returns.
    """
    accidents = ensure_accident_outcome_columns(accidents)
    train = accidents[accidents["year"] <= train_end_year].copy()
    test = accidents[accidents["year"] >= test_start_year].copy()

    if len(train) == 0 or len(test) == 0:
        raise ValueError("Temporal validation needs non-empty train and test accident sets.")

    train_snap = snap_accidents_to_edges(
        Gp,
        train,
        output_file=Path(output_file).with_suffix(".train_snapped.csv"),
    )
    test_snap = snap_accidents_to_edges(
        Gp,
        test,
        output_file=Path(output_file).with_suffix(".test_snapped.csv"),
    )

    _, train_edge = partition_node_edge_accidents(train_snap)
    edge_risk = build_edge_risk(
        train_edge,
        edge_features=edge_features,
        output_file=Path(output_file).with_suffix(".edge_risk.csv"),
    )

    seg = edge_risk.drop_duplicates("pair_id").copy()
    seg["pair_id"] = seg["pair_id"].astype(str)
    seg["edge_length_m"] = pd.to_numeric(seg["edge_length_m"], errors="coerce").fillna(0.0)
    test_pairs = test_snap["pair_id"].astype(str)
    total_len = float(seg["edge_length_m"].sum())

    # True top decile by segment COUNT (not `>= quantile(0.90)`, which with many
    # tied low scores selects far more than 10% of segments).
    n_top = max(1, int(round(len(seg) * 0.10)))
    by_risk = seg.sort_values("historical_risk_norm", ascending=False)
    top_pairs = set(by_risk.head(n_top)["pair_id"])
    hit = float(test_pairs.isin(top_pairs).mean())
    top_len_share = (
        float(by_risk.head(n_top)["edge_length_m"].sum() / total_len) if total_len > 0 else None
    )

    # Baseline 1 — length. Crashes concentrate on long arterials, so "just rank
    # by segment length" is the baseline a reviewer will demand. Same decile budget.
    by_len = seg.sort_values("edge_length_m", ascending=False)
    length_pairs = set(by_len.head(n_top)["pair_id"])
    length_recall = float(test_pairs.isin(length_pairs).mean())

    # Baseline 2 — memory. "This segment had >=1 crash in the training window."
    # This is the strongest naive baseline and the honest bar to clear.
    memory_pairs = set(train_snap["pair_id"].astype(str))
    memory_recall = float(test_pairs.isin(memory_pairs).mean())

    result = {
        "train_years": f"<= {train_end_year}",
        "test_years": f">= {test_start_year}",
        "n_train_crashes": int(len(train_snap)),
        "n_test_crashes": int(len(test_snap)),
        "n_segments": int(len(seg)),
        "top_decile_recall": hit,
        "top_decile_network_length_share": top_len_share,
        "random_expectation": 0.10,
        "length_baseline_recall": length_recall,
        "memory_baseline_recall": memory_recall,
        "memory_baseline_segments": int(len(memory_pairs)),
        "lift_over_random": float(hit / 0.10) if np.isfinite(hit) else None,
        "lift_over_length": float(hit / length_recall) if length_recall > 0 else None,
        "lift_over_memory": float(hit / memory_recall) if memory_recall > 0 else None,
        "note": (
            "lift_over_random uses a 10% strawman. The defensible bars are "
            "lift_over_length and lift_over_memory; a value <= 1.0 there means the "
            "risk ranking does not beat that baseline. Memory is blind to crashes on "
            "never-before-crashed segments, which is where a structural model must win."
        ),
    }

    if compare_surfaces:
        # Same segment universe and the same top-decile-by-count rule as the
        # historical metric above, so all three surfaces are apples-to-apples.
        universe = pd.Index(seg["pair_id"].astype(str), name="pair_id")

        # "Never-before-crashed" test crashes: those on segments with no crash in
        # the training window. A historical crash map has ~zero signal there, so
        # this subset is where a model that predicts from road features must win.
        nbc_test_pairs = test_pairs[~test_pairs.isin(memory_pairs)]

        # Each surface's per-segment score (computed once, reused for both the
        # per-surface recall/CI and the paired-difference significance test).
        series_by_name = {
            "historical_gis": seg.set_index(seg["pair_id"].astype(str))["historical_risk_norm"],
        }
        try:
            series_by_name["occurrence_ml"] = _occurrence_surface_by_pair(
                train_snap, edge_risk, edge_features, output_file
            )
        except Exception as exc:  # best-effort; never break the pipeline
            print(f"occurrence surface skipped in temporal validation: {exc}")
        try:
            series_by_name["frequency_nb"] = _frequency_surface_by_pair(
                train_snap, edge_features, Gp
            )
        except Exception as exc:  # best-effort
            print(f"frequency surface skipped in temporal validation: {exc}")

        def _reindexed(name):
            return series_by_name[name].reindex(universe).astype(float).fillna(0.0)

        def _score(name):
            s = _reindexed(name)
            row = {"surface": name}
            row.update(_surface_topk_recall(s, test_pairs, frac=0.10))
            # Same top-10% flagged set, recall measured only over crashes on
            # never-before-crashed segments.
            nbc = _surface_topk_recall(s, nbc_test_pairs, frac=0.10)
            row["nbc_recall"] = nbc["top_decile_recall"]
            row["nbc_lift_over_random"] = nbc["lift_over_random"]
            row["nbc_ci_low"] = nbc["ci_low"]
            row["nbc_ci_high"] = nbc["ci_high"]
            return row

        surfaces = [_score(name) for name in series_by_name]

        # Paired-difference significance: is the top surface really ahead of the
        # runner-up, or only on a point estimate? Uses the same flagged sets and
        # paired crash resamples, so the difference has its own bootstrap CI.
        def _hit(name, tp):
            s = _reindexed(name)
            n_top = max(1, int(round(len(s) * 0.10)))
            top = set(s.sort_values(ascending=False).head(n_top).index.astype(str))
            return tp.astype(str).isin(top).to_numpy().astype(float)

        def _leader_test(tp, recall_key):
            ranked = sorted(
                (r for r in surfaces if r.get(recall_key) is not None),
                key=lambda r: r[recall_key],
                reverse=True,
            )
            if len(ranked) < 2 or len(tp) == 0:
                return None
            leader, runner = ranked[0]["surface"], ranked[1]["surface"]
            diff = _paired_diff_ci(_hit(leader, tp), _hit(runner, tp))
            if diff is None:
                return None
            return {"leader": leader, "runner_up": runner, **diff}

        result["overall_leader_test"] = _leader_test(test_pairs, "top_decile_recall")
        result["nbc_leader_test"] = _leader_test(nbc_test_pairs, "nbc_recall")

        result["surface_metric"] = (
            "share of held-out crashes captured in each surface's top-decile "
            "segments (top 10% by segment count over a common universe); random 10%"
        )
        result["nbc_metric"] = (
            "nbc_recall = the same top-decile metric restricted to held-out crashes "
            "on segments with no crash in the training window (never-before-crashed), "
            "where a historical crash map is blind"
        )
        result["n_nbc_test_crashes"] = int(len(nbc_test_pairs))
        result["ci_note"] = (
            "ci_low/ci_high (and nbc_ci_*) are 95% bootstrap intervals from 1000 "
            "resamples of the held-out crashes. Overlapping intervals between surfaces "
            "mean the difference is within noise — the surfaces are statistically tied."
        )
        result["significance_note"] = (
            "overall_leader_test / nbc_leader_test hold the paired bootstrap 95% CI of "
            "(top surface − runner-up). significant=true means diff_ci_low > 0, i.e. the "
            "leader beats the runner-up beyond noise even if the marginal CIs overlap."
        )
        result["surfaces"] = surfaces

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"temporal validation: top decile recall {hit:.1%} "
        f"(random {hit/0.10:.2f}×, length {hit/length_recall:.2f}×, "
        f"memory {hit/memory_recall:.2f}×)"
    )
    for s in result.get("surfaces", []):
        r = s.get("top_decile_recall")
        if r is not None:
            lo, hi = s.get("ci_low"), s.get("ci_high")
            ci_txt = f" [{lo:.1%}–{hi:.1%}]" if lo is not None and hi is not None else ""
            nbc = s.get("nbc_recall")
            nbc_txt = f" | never-before-crashed {nbc:.1%}" if nbc is not None else ""
            print(
                f"  surface {s['surface']:<16} {r:.1%}{ci_txt}  "
                f"({s.get('lift_over_random', 0):.2f}× random){nbc_txt}"
            )
    for key, label in [("overall_leader_test", "overall"),
                       ("nbc_leader_test", "never-before-crashed")]:
        t = result.get(key)
        if t:
            verdict = "SIGNIFICANT" if t["significant"] else "tied (not significant)"
            print(
                f"  {label}: {t['leader']} − {t['runner_up']} = {t['diff']:+.1%} "
                f"[{t['diff_ci_low']:+.1%}, {t['diff_ci_high']:+.1%}] → {verdict}"
            )
    return result