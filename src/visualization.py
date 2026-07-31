from __future__ import annotations

from pathlib import Path

import folium
import numpy as np
import osmnx as ox
import pandas as pd

from .config import RISK_STREETS_MAP_FILE
from .osm_network import edge_uid


def route_summary_table(result: dict) -> pd.DataFrame:
    """Robust route summary table."""
    rows = []

    if result.get("fastest_distance_summary"):
        rows.append({
            "route": "Fastest",
            "distance_km": result["fastest_distance_summary"]["distance_km"],
            "risk_model": "distance only",
            "risk_score": None,
            "risk_reduction_pct": None,
        })

    if result.get("historical_summary"):
        rows.append({
            "route": "Historical GIS-risk",
            "distance_km": result["historical_summary"]["distance_km"],
            "risk_model": "historical spatial risk",
            "risk_score": result["historical_summary"]["length_weighted_risk"],
            "risk_reduction_pct": result.get("historical_risk_reduction_pct"),
        })

    if result.get("ml_summary"):
        rows.append({
            "route": "ML road-risk",
            "distance_km": result["ml_summary"]["distance_km"],
            "risk_model": "leakage-safe road-only ML",
            "risk_score": result["ml_summary"]["length_weighted_risk"],
            "risk_reduction_pct": result.get("ml_risk_reduction_pct"),
        })

    return pd.DataFrame(rows)


def make_risk_street_map(
    Gp,
    edge_risk: pd.DataFrame,
    top_n: int = 1500,
    min_accidents: int = 1,
    output_file: str | Path = RISK_STREETS_MAP_FILE,
) -> folium.Map:
    """Draw top-risk street segments, not an empty placeholder map."""
    edges = ox.graph_to_gdfs(Gp, nodes=False).reset_index()
    edges["edge_uid"] = [edge_uid(u, v, k) for u, v, k in zip(edges["u"], edges["v"], edges["key"])]

    plot = edges.merge(edge_risk, on="edge_uid", how="left", suffixes=("", "_risk"))
    plot["accident_count"] = pd.to_numeric(plot.get("accident_count", 0), errors="coerce").fillna(0)
    plot["historical_risk_norm"] = pd.to_numeric(plot.get("historical_risk_norm", 0), errors="coerce").fillna(0)

    plot = plot[plot["accident_count"] >= min_accidents].copy()
    plot = plot.sort_values("historical_risk_norm", ascending=False).head(top_n)

    center = [52.52, 13.405]
    m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    if len(plot) == 0:
        folium.Marker(center, popup="No risk segments to display").add_to(m)
    else:
        plot_wgs = plot.to_crs("EPSG:4326")
        for _, row in plot_wgs.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            if geom.geom_type == "LineString":
                coords = [(lat, lon) for lon, lat in geom.coords]
            elif geom.geom_type == "MultiLineString":
                coords = []
                for part in geom.geoms:
                    coords.extend([(lat, lon) for lon, lat in part.coords])
            else:
                continue

            risk = float(row.get("historical_risk_norm", 0.0))
            weight = 2 + 5 * risk
            folium.PolyLine(
                coords,
                color="red",
                weight=weight,
                opacity=0.25 + 0.65 * risk,
                tooltip=(
                    f"{row.get('name', 'unnamed')} | "
                    f"accidents {int(row.get('accident_count', 0))} | "
                    f"risk {risk:.2f}"
                ),
            ).add_to(m)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    m.save(output_file)
    print(f"saved risk street map: {output_file} ({len(plot):,} segments)")
    return m


def top_risk_table(
    edge_risk: pd.DataFrame,
    min_length_km: float = 0.3,
    n: int = 20,
) -> pd.DataFrame:
    """Street-level table for presentation.

    Uses absolute weighted severity first, with a minimum street length to avoid
    per-km inflation on tiny streets.
    """
    df = edge_risk.copy()
    if "name" not in df.columns:
        df["name"] = "unknown"

    # Collapse directed edges.
    cols = {
        "edge_length_m": "sum",
        "accident_count": "sum",
        "severity_sum": "sum",
        "serious_fatal_count": "sum",
        "fatal_count": "sum",
    }
    cols = {k: v for k, v in cols.items() if k in df.columns}
    tbl = df.groupby("name").agg(cols).reset_index()
    tbl = tbl.rename(columns={"edge_length_m": "length_m", "severity_sum": "weighted"})
    tbl["length_km"] = tbl["length_m"] / 1000.0
    tbl = tbl[tbl["length_km"] >= min_length_km].copy()
    tbl["weighted_per_km"] = tbl["weighted"] / tbl["length_km"].replace(0, np.nan)
    tbl = tbl.sort_values("weighted", ascending=False).head(n)
    return tbl
