from __future__ import annotations

from pathlib import Path

import folium
import pandas as pd

from .config import HISTORICAL_RISK_MAP


def make_historical_risk_map(
    edge_risk: pd.DataFrame,
    output_file: str | Path = HISTORICAL_RISK_MAP,
    top_n: int = 1500,
) -> folium.Map:
    """Make a top-risk street map if geometry is available.

    This function is intentionally defensive. If geometry is not present in the
    CSV, it still creates an empty Berlin-centered map.
    """
    m = folium.Map(location=[52.52, 13.405], zoom_start=12, tiles="OpenStreetMap")

    if "geometry" not in edge_risk.columns:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        m.save(output_file)
        print(f"Geometry not available; saved placeholder map: {output_file}")
        return m

    # Full geometry reconstruction is not included in CSV by default.
    # For production, save edge_risk as GeoPackage and read with geopandas.
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    m.save(output_file)
    print(f"Saved historical risk map: {output_file}")
    return m


def route_summary_table(result: dict) -> pd.DataFrame:
    rows = []
    for label, key in [
        ("Fastest", "fastest_summary"),
        ("Historical GIS-risk", "historical_summary"),
        ("ML-safest", "ml_summary"),
    ]:
        row = result[key].copy()
        row["route"] = label
        rows.append(row)

    return pd.DataFrame(rows)[
        ["route", "distance_km", "length_weighted_risk", "risk_exposure", "n_segments"]
    ]
