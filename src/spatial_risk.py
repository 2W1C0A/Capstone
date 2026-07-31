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
) -> pd.DataFrame:
    """Build undirected segment historical GIS risk with empirical-Bayes shrinkage."""
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

    cap = risk["historical_risk"].quantile(0.95)
    if not np.isfinite(cap) or cap <= 0:
        cap = risk["historical_risk"].max()
    risk["historical_risk_norm"] = (risk["historical_risk"] / cap).clip(0, 1) if cap > 0 else 0.0

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

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    edge_risk.to_csv(output_file, index=False)
    print(f"saved edge risk: {output_file} | {len(edge_risk):,} directed edges")
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


def temporal_validation(
    Gp,
    accidents: pd.DataFrame,
    edge_features: pd.DataFrame,
    train_end_year: int = 2023,
    test_start_year: int = 2024,
    output_file: str | Path = TEMPORAL_VALIDATION_FILE,
) -> dict:
    """Forward validation of historical risk ranking.

    Build risk from earlier years and measure how many future crashes land in the
    top decile of the past-risk ranking. Random expectation is 10%.
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
    threshold = seg["historical_risk_norm"].quantile(0.90)
    top_pairs = set(seg.loc[seg["historical_risk_norm"] >= threshold, "pair_id"].astype(str))

    hit = test_snap["pair_id"].astype(str).isin(top_pairs).mean()
    result = {
        "train_years": f"<= {train_end_year}",
        "test_years": f">= {test_start_year}",
        "n_train_crashes": int(len(train_snap)),
        "n_test_crashes": int(len(test_snap)),
        "top_decile_recall": float(hit),
        "random_expectation": 0.10,
        "lift_over_random": float(hit / 0.10) if np.isfinite(hit) else None,
    }

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"temporal validation: {hit:.1%} of future crashes in top decile; lift {hit/0.10:.2f}×")
    return result
