from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import osmnx as ox
import pandas as pd

from .config import (
    BERLIN_CRS,
    BERLIN_PLACE,
    OSM_EDGE_FEATURES_FILE,
    OSM_GRAPH_FILE,
    OSM_PROJECTED_GRAPH_FILE,
)


HIGHWAY_ORDER = (
    "motorway",
    "trunk",
    "primary_link",
    "primary",
    "secondary_link",
    "secondary",
    "tertiary_link",
    "tertiary",
    "unclassified",
    "residential",
    "cycleway",
    "living_street",
    "pedestrian",
    "service",
    "path",
    "footway",
    "track",
    "bridleway",
)


def tag_to_string(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(map(str, value))
    if isinstance(value, tuple):
        return ",".join(map(str, value))
    if pd.isna(value):
        return "unknown"
    return str(value)


def simplify_highway(value: Any) -> str:
    """Classify OSM highway tags by priority order.

    This preserves _link classes where possible, because they often describe
    junction arms.
    """
    h = tag_to_string(value).lower()
    for cls in HIGHWAY_ORDER:
        if cls in h:
            return cls
    return "other"


def has_cycleway(row: pd.Series) -> int:
    """Return 1 if OSM tags indicate explicit cycle infrastructure."""
    fields = [
        "cycleway",
        "cycleway:left",
        "cycleway:right",
        "cycleway:both",
        "bicycle",
        "highway",
    ]
    text = " ".join(tag_to_string(row.get(c, "")) for c in fields).lower()
    positive_tokens = ["cycleway", "lane", "track", "opposite", "designated", "use_sidepath"]
    return int(any(tok in text for tok in positive_tokens))


def extract_maxspeed(value: Any) -> float:
    """First numeric maxspeed value in km/h.

    Handles examples such as "30", "30 mph", "30;50", "DE:urban", "walk".
    """
    if isinstance(value, list) and value:
        value = value[0]
    s = str(value).lower()
    m = re.search(r"\d+", s)
    if not m:
        return np.nan
    speed = float(m.group())
    return speed * 1.60934 if "mph" in s else speed


def edge_uid(u, v, k) -> str:
    return f"{u}|{v}|{k}"


def pair_id(u, v) -> str:
    a, b = sorted([str(u), str(v)])
    return f"{a}|{b}"


def load_or_download_graph(
    graph_file: str | Path = OSM_GRAPH_FILE,
    projected_graph_file: str | Path = OSM_PROJECTED_GRAPH_FILE,
):
    """Load or download Berlin bicycle graph and projected graph."""
    graph_file = Path(graph_file)
    projected_graph_file = Path(projected_graph_file)
    graph_file.parent.mkdir(parents=True, exist_ok=True)

    if graph_file.exists():
        G = ox.load_graphml(graph_file)
        print(f"loaded graph: {graph_file}")
    else:
        print(f"downloading OSM bicycle network for {BERLIN_PLACE} ...")
        ox.settings.use_cache = True
        G = ox.graph_from_place(BERLIN_PLACE, network_type="bike", simplify=True)
        ox.save_graphml(G, graph_file)
        print(f"saved graph: {graph_file}")

    if G.number_of_nodes() < 50_000:
        raise RuntimeError(
            f"Graph sanity check failed: only {G.number_of_nodes():,} nodes. "
            "This does not look like the Berlin bicycle graph."
        )

    if projected_graph_file.exists():
        Gp = ox.load_graphml(projected_graph_file)
        print(f"loaded projected graph: {projected_graph_file}")
    else:
        Gp = ox.project_graph(G, to_crs=BERLIN_CRS)
        ox.save_graphml(Gp, projected_graph_file)
        print(f"saved projected graph: {projected_graph_file}")

    return G, Gp


def build_edge_features(
    Gp,
    output_file: str | Path = OSM_EDGE_FEATURES_FILE,
) -> pd.DataFrame:
    """Extract per-edge OSM features for ML and routing."""
    edges = ox.graph_to_gdfs(Gp, nodes=False).reset_index()

    # Stable IDs.
    edges["edge_uid"] = [edge_uid(u, v, k) for u, v, k in zip(edges["u"], edges["v"], edges["key"])]
    edges["pair_id"] = [pair_id(u, v) for u, v in zip(edges["u"], edges["v"])]

    if "length" in edges.columns:
        edges["edge_length_m"] = pd.to_numeric(edges["length"], errors="coerce")
    else:
        edges["edge_length_m"] = edges.geometry.length

    edges["highway_raw"] = edges.get("highway", "unknown").apply(tag_to_string)
    edges["highway_simple"] = edges.get("highway", "unknown").apply(simplify_highway)
    edges["has_cycleway"] = edges.apply(has_cycleway, axis=1).astype(int)

    if "maxspeed" in edges.columns:
        edges["maxspeed_num"] = edges["maxspeed"].apply(extract_maxspeed)
    else:
        edges["maxspeed_num"] = np.nan

    edges["maxspeed_missing"] = edges["maxspeed_num"].isna().astype(int)

    # Fill missing speed by highway class median, then global median, then 30.
    edges["maxspeed_num"] = edges.groupby("highway_simple")["maxspeed_num"].transform(
        lambda s: s.fillna(s.median())
    )
    edges["maxspeed_num"] = edges["maxspeed_num"].fillna(edges["maxspeed_num"].median()).fillna(30.0)

    keep = [
        "edge_uid", "pair_id", "u", "v", "key",
        "edge_length_m", "highway_raw", "highway_simple",
        "has_cycleway", "maxspeed_num", "maxspeed_missing",
        "geometry",
    ]
    keep = [c for c in keep if c in edges.columns]
    out = edges[keep].copy()

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # Save non-geometry CSV for ML/app.
    out.drop(columns=["geometry"], errors="ignore").to_csv(output_file, index=False)
    print(f"saved edge features: {output_file} | {len(out):,} edges")
    print(f"maxspeed present on {(1 - edges['maxspeed_missing'].mean()):.1%} of edges before filling")
    return out
