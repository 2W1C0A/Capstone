"""Interactive map of the highest-risk street segments."""

from __future__ import annotations

import branca.colormap as cm
import folium
import osmnx as ox
import pandas as pd

BERLIN_CENTRE = (52.5200, 13.4050)


def risk_map(G_proj, edge_risk, top_n=1500, min_accidents=1,
             outfile="berlin_risk_streets.html"):
    """Draw the `top_n` riskiest segments, coloured by risk, with tooltips."""
    edges = ox.graph_to_gdfs(G_proj, nodes=False).reset_index()
    edges["pair"] = [frozenset(p) for p in zip(edges["u"], edges["v"])]
    edges = edges.drop_duplicates(subset="pair")

    cols = ["pair", "risk", "risk_norm", "n_accidents", "observed"]
    gdf = edges.merge(edge_risk[cols], on="pair", how="inner")

    gdf = gdf[gdf["n_accidents"] >= min_accidents].nlargest(top_n, "risk")
    if gdf.empty:
        raise ValueError("No segments met the filter; lower min_accidents.")

    for col in ("name", "highway"):
        if col in gdf.columns:
            gdf[col] = gdf[col].apply(
                lambda v: ", ".join(map(str, v)) if isinstance(v, list) else v
            )
    gdf["name"] = gdf.get("name", pd.Series(index=gdf.index)).fillna("(unnamed)")
    gdf["risk"] = gdf["risk"].round(2)
    gdf["length_m"] = gdf["length"].round(0)

    gdf = gdf.to_crs("EPSG:4326")

    lo, hi = gdf["risk"].min(), gdf["risk"].max()
    scale = cm.LinearColormap(
        ["#ffd24d", "#f58231", "#e6194b", "#7a0b21"], vmin=lo, vmax=hi
    )
    scale.caption = "Severity-weighted risk per 100 m (KSI weights)"

    m = folium.Map(location=BERLIN_CENTRE, zoom_start=12, tiles="cartodbpositron")

    folium.GeoJson(
        gdf[["geometry", "name", "highway", "risk", "n_accidents", "length_m"]],
        style_function=lambda f: {
            "color": scale(f["properties"]["risk"]),
            "weight": 4,
            "opacity": 0.85,
        },
        highlight_function=lambda _: {"weight": 7, "color": "#000000"},
        tooltip=folium.GeoJsonTooltip(
            fields=["name", "highway", "risk", "n_accidents", "length_m"],
            aliases=["Street", "Class", "Risk score", "Accidents", "Length (m)"],
            sticky=True,
        ),
    ).add_to(m)

    scale.add_to(m)

    if outfile:
        m.save(outfile)
        print(f"saved {outfile}  ({len(gdf):,} segments)")

    return m


def top_risk_table(edge_risk, G_proj, n=20):
    """The riskiest named streets, grouped by street rather than OSM fragment."""
    edges = ox.graph_to_gdfs(G_proj, nodes=False).reset_index()
    edges["pair"] = [frozenset(p) for p in zip(edges["u"], edges["v"])]
    edges = edges.drop_duplicates(subset="pair")
    edges["name"] = edges["name"].apply(
        lambda v: ", ".join(map(str, v)) if isinstance(v, list) else v
    )

    merged = edges.merge(
        edge_risk[["pair", "observed", "n_accidents", "length"]],
        on="pair", how="inner", suffixes=("", "_r"),
    )
    merged = merged[merged["name"].notna()]

    by_street = merged.groupby("name").agg(
        accidents=("n_accidents", "sum"),
        weighted=("observed", "sum"),
        length_km=("length_r", lambda s: s.sum() / 1000),
    )
    by_street["weighted_per_km"] = by_street["weighted"] / by_street["length_km"]

    return (
        by_street[by_street["length_km"] > 0.3]
        .nlargest(n, "weighted_per_km")
        .round(2)
    )
