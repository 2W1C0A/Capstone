from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Point

from .config import (
    BERLIN_CRS,
    SNAPPED_ACCIDENTS_FILE,
    SPATIAL_RISK_FILE,
)
from .osm_network import make_edge_uid, make_pair_id, tag_to_string, simplify_highway


SEVERITY_KSI = {
    "minor_injury": 1.0,
    "serious_injury": 8.0,
    "fatal": 30.0,
    "unknown": 1.0,
}


def normalize_nearest_edge_results(edge_ids, n_expected: int) -> pd.DataFrame:
    """Normalize osmnx.nearest_edges output across OSMnx versions.

    OSMnx may return nearest edge IDs as either:
    1. a list/array of (u, v, key) tuples
    2. a tuple of three arrays: (u_array, v_array, key_array)

    This function converts both formats to a DataFrame with columns u, v, key.
    """
    # Format 1: tuple of three arrays: (u_array, v_array, key_array)
    if isinstance(edge_ids, tuple) and len(edge_ids) == 3:
        try:
            u_arr = list(edge_ids[0])
            v_arr = list(edge_ids[1])
            k_arr = list(edge_ids[2])
            if len(u_arr) == n_expected and len(v_arr) == n_expected and len(k_arr) == n_expected:
                return pd.DataFrame({"u": u_arr, "v": v_arr, "key": k_arr})
        except TypeError:
            # Could be a single edge tuple. Fall through to list-of-tuples handling.
            pass

    # Format 2: list/array of edge tuples.
    edge_list = list(edge_ids)
    if len(edge_list) != n_expected:
        raise ValueError(
            "Unexpected nearest_edges output length. "
            f"Expected {n_expected}, got {len(edge_list)}. "
            "This is usually caused by an OSMnx version return-format difference."
        )

    rows = []
    for e in edge_list:
        if isinstance(e, (list, tuple)) and len(e) >= 3:
            rows.append((e[0], e[1], e[2]))
        else:
            raise ValueError(
                "Unexpected edge id format from osmnx.nearest_edges. "
                f"Example value: {repr(e)}"
            )
    return pd.DataFrame(rows, columns=["u", "v", "key"])


def normalize_distance_results(dist, n_expected: int) -> list[float]:
    """Normalize distance output to a list of floats."""
    if hasattr(dist, "__len__") and not isinstance(dist, (str, bytes)):
        dist_list = list(dist)
    else:
        dist_list = [float(dist)]

    if len(dist_list) != n_expected:
        raise ValueError(
            f"Unexpected distance output length. Expected {n_expected}, got {len(dist_list)}."
        )
    return [float(x) for x in dist_list]


def accidents_to_gdf(
    accidents: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        accidents.copy(),
        geometry=gpd.points_from_xy(accidents[lon_col], accidents[lat_col]),
        crs=crs,
    )
    return gdf


def snap_accidents_to_edges(
    G_proj: nx.MultiDiGraph,
    accidents: pd.DataFrame,
    max_snap_dist_m: float = 25.0,
    output_file: str | Path = SNAPPED_ACCIDENTS_FILE,
) -> pd.DataFrame:
    """Attach each Unfallatlas accident to the nearest OSM graph edge.

    This follows Person B's stronger GIS design:
    - use metric CRS, not longitude/latitude degrees
    - keep only accidents snapped within a max distance threshold
    - report snap quality
    """
    if G_proj.graph.get("crs") in (None, "EPSG:4326", "epsg:4326"):
        raise ValueError(
            "G_proj must be projected to a metric CRS, e.g. EPSG:25833. "
            "Distance-based snapping in degrees is not meaningful."
        )

    target_crs = G_proj.graph.get("crs", BERLIN_CRS)
    acc_gdf = accidents_to_gdf(accidents).to_crs(target_crs)

    x = acc_gdf.geometry.x.to_numpy()
    y = acc_gdf.geometry.y.to_numpy()

    # osmnx returns nearest edge IDs and distances.
    # Different OSMnx versions return edge_ids in different formats:
    #   - list of (u, v, key) tuples
    #   - tuple of three arrays: (u_array, v_array, key_array)
    nearest = ox.distance.nearest_edges(G_proj, X=x, Y=y, return_dist=True)
    if isinstance(nearest, tuple) and len(nearest) == 2:
        edge_ids, dist = nearest
    else:
        raise RuntimeError("Unexpected return from ox.distance.nearest_edges")

    out = accidents.copy().reset_index(drop=True)
    edge_df = normalize_nearest_edge_results(edge_ids, n_expected=len(out))
    out[["u", "v", "key"]] = edge_df[["u", "v", "key"]].to_numpy()
    out["snap_dist_m"] = normalize_distance_results(dist, n_expected=len(out))
    out["edge_uid"] = [make_edge_uid(u, v, k) for u, v, k in zip(out["u"], out["v"], out["key"])]
    out["pair_id"] = [make_pair_id(u, v) for u, v in zip(out["u"], out["v"])]

    kept = out[out["snap_dist_m"] <= max_snap_dist_m].copy()
    median_offset = kept["snap_dist_m"].median() if len(kept) else np.nan
    print(
        f"Snapped {len(kept):,} / {len(out):,} accidents within {max_snap_dist_m:g} m "
        f"({len(kept)/len(out):.1%}); median offset {median_offset:.1f} m"
    )

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(output_file, index=False)
    print(f"Saved snapped accidents: {output_file}")
    return kept


def _edge_table_from_graph(G_proj: nx.MultiDiGraph) -> pd.DataFrame:
    edges = ox.graph_to_gdfs(G_proj, nodes=False, edges=True).reset_index()
    edges["edge_uid"] = [make_edge_uid(u, v, k) for u, v, k in zip(edges["u"], edges["v"], edges["key"])]
    edges["pair_id"] = [make_pair_id(u, v) for u, v in zip(edges["u"], edges["v"])]
    edges["highway_simple"] = edges["highway"].apply(simplify_highway) if "highway" in edges.columns else "other"

    if "length" not in edges.columns:
        edges["length"] = edges.geometry.length

    return edges[["edge_uid", "pair_id", "u", "v", "key", "length", "highway_simple"]].copy()


def build_node_risk(
    G_proj: nx.MultiDiGraph,
    snapped: pd.DataFrame,
    radius_m: float = 20.0,
    severity_col: str = "severity_label",
) -> pd.Series:
    """Estimate intersection/node risk from nearby accidents.

    Each accident is assigned to the nearest node. If the nearest node is farther
    than radius_m, it is ignored for node-risk scoring.
    """
    if len(snapped) == 0:
        return pd.Series(dtype=float)

    acc_gdf = accidents_to_gdf(snapped).to_crs(G_proj.graph.get("crs", BERLIN_CRS))
    x = acc_gdf.geometry.x.to_numpy()
    y = acc_gdf.geometry.y.to_numpy()

    nodes, dist = ox.distance.nearest_nodes(G_proj, X=x, Y=y, return_dist=True)

    tmp = snapped.copy().reset_index(drop=True)
    tmp["node"] = nodes
    tmp["node_dist_m"] = dist

    near = tmp[tmp["node_dist_m"] <= radius_m].copy()

    if severity_col in near.columns:
        near["node_weight"] = near[severity_col].map(SEVERITY_KSI).fillna(1.0)
    else:
        near["node_weight"] = 1.0

    load = near.groupby("node")["node_weight"].sum()
    print(f"{len(load):,} intersections carry accidents within {radius_m:g} m")
    return load


def build_edge_risk(
    G_proj: nx.MultiDiGraph,
    snapped: pd.DataFrame,
    severity_col: str = "severity_label",
    shrinkage: float = 5.0,
    node_risk: Optional[pd.Series] = None,
    output_file: str | Path = SPATIAL_RISK_FILE,
) -> pd.DataFrame:
    """Build Person-B-style severity-weighted historical risk per OSM edge.

    Main improvements over a simple accident-count baseline:
    - severity-weighted accident counts
    - normalisation by edge length per 100 m
    - empirical-Bayes shrinkage toward road-class prior
    - optional intersection/node risk
    - p95 cap and 0-1 normalisation
    """
    edges = _edge_table_from_graph(G_proj)

    # Aggregate accidents by undirected road segment.
    snap = snapped.copy()
    if severity_col in snap.columns:
        snap["w"] = snap[severity_col].map(SEVERITY_KSI).fillna(1.0)
    elif "severity_weight" in snap.columns:
        snap["w"] = snap["severity_weight"].fillna(1.0)
    else:
        snap["w"] = 1.0

    agg = (
        snap.groupby("pair_id")
        .agg(
            accident_count=("pair_id", "size"),
            severity_sum=("w", "sum"),
            serious_fatal_count=("serious_or_fatal", "sum") if "serious_or_fatal" in snap.columns else ("w", "size"),
        )
        .reset_index()
    )

    risk = edges.merge(agg, on="pair_id", how="left")
    for col in ["accident_count", "severity_sum", "serious_fatal_count"]:
        risk[col] = risk[col].fillna(0.0)

    # Exposure in 100m units. Avoid zero length.
    risk["len100"] = (risk["length"].clip(lower=1.0) / 100.0)

    # Compute road-class prior.
    class_sum = risk.groupby("highway_simple")["severity_sum"].sum()
    class_len = risk.groupby("highway_simple")["len100"].sum().replace(0, np.nan)
    class_prior = (class_sum / class_len).replace([np.inf, -np.inf], np.nan)

    global_prior = risk["severity_sum"].sum() / max(risk["len100"].sum(), 1.0)

    risk["class_prior"] = risk["highway_simple"].map(class_prior).fillna(global_prior)

    # Empirical-Bayes shrinkage:
    # score = (observed + m * prior * exposure) / (exposure + m)
    m = float(shrinkage)
    risk["risk_raw"] = (
        risk["severity_sum"] + m * risk["class_prior"] * risk["len100"]
    ) / (risk["len100"] + m)

    # Add node/intersection risk.
    risk["node_risk_raw"] = 0.0
    if node_risk is not None and len(node_risk) > 0:
        nr = node_risk.copy()
        cap_node = nr.quantile(0.95) if nr.quantile(0.95) > 0 else nr.max()
        if cap_node and cap_node > 0:
            nr_norm = (nr.clip(upper=cap_node) / cap_node).to_dict()
        else:
            nr_norm = nr.to_dict()

        risk["node_risk_raw"] = [
            max(float(nr_norm.get(u, 0.0)), float(nr_norm.get(v, 0.0)))
            for u, v in zip(risk["u"], risk["v"])
        ]

    # Cap extreme short-edge rates.
    cap = risk["risk_raw"].quantile(0.95)
    if cap <= 0 or np.isnan(cap):
        cap = risk["risk_raw"].max()

    if cap and cap > 0:
        risk["historical_risk_norm"] = risk["risk_raw"].clip(upper=cap) / cap
    else:
        risk["historical_risk_norm"] = 0.0

    risk["combined_spatial_risk"] = (
        0.75 * risk["historical_risk_norm"] + 0.25 * risk["node_risk_raw"]
    ).clip(0, 1)

    # For compatibility with route engine and model.
    risk["ml_accident_risk"] = risk["combined_spatial_risk"]
    risk["risk_score"] = risk["combined_spatial_risk"]

    n_segments = risk["pair_id"].nunique()
    pct_acc = 100 * (risk.groupby("pair_id")["accident_count"].max() > 0).mean()
    risk_cap = float(cap) if cap is not None else 0.0
    print(
        f"{n_segments:,} undirected segments | {pct_acc:.1f}% with >=1 accident | "
        f"risk cap (p95) = {risk_cap:.2f}"
    )

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    risk.to_csv(output_file, index=False)
    print(f"Saved spatial risk: {output_file}")
    return risk


def build_spatial_risk_pipeline(
    G_proj: nx.MultiDiGraph,
    accidents: pd.DataFrame,
    max_snap_dist_m: float = 25.0,
    node_radius_m: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    snapped = snap_accidents_to_edges(
        G_proj,
        accidents,
        max_snap_dist_m=max_snap_dist_m,
    )
    node_risk = build_node_risk(
        G_proj,
        snapped,
        radius_m=node_radius_m,
    )
    edge_risk = build_edge_risk(
        G_proj,
        snapped,
        node_risk=node_risk,
    )
    return snapped, edge_risk
