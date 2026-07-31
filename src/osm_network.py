from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx

from .config import (
    BERLIN_PLACE,
    BERLIN_CRS,
    OSM_GRAPH_FILE,
    OSM_PROJECTED_GRAPH_FILE,
    OSM_EDGE_FEATURES_FILE,
)


def load_or_download_graph(
    graph_file: str | Path = OSM_GRAPH_FILE,
    projected_graph_file: str | Path = OSM_PROJECTED_GRAPH_FILE,
    place: str = BERLIN_PLACE,
    crs: str = BERLIN_CRS,
    force_download: bool = False,
) -> tuple[nx.MultiDiGraph, nx.MultiDiGraph]:
    """Load or download Berlin bicycle OSM graph.

    Returns:
        G: graph in WGS84 for geocoding/routing display
        G_proj: graph in metric CRS for distance-based spatial analysis
    """
    graph_file = Path(graph_file)
    projected_graph_file = Path(projected_graph_file)

    ox.settings.use_cache = True
    ox.settings.log_console = False

    if graph_file.exists() and projected_graph_file.exists() and not force_download:
        G = ox.load_graphml(graph_file)
        G_proj = ox.load_graphml(projected_graph_file)
        print(f"Loaded graph: {graph_file}")
        print(f"Loaded projected graph: {projected_graph_file}")
        return G, G_proj

    print(f"Downloading OSM bicycle network for {place}...")
    G = ox.graph_from_place(place, network_type="bike", simplify=True)

    print(f"Projecting graph to {crs}...")
    G_proj = ox.project_graph(G, to_crs=crs)

    graph_file.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, graph_file)
    ox.save_graphml(G_proj, projected_graph_file)

    print(f"Saved graph: {graph_file}")
    print(f"Saved projected graph: {projected_graph_file}")
    return G, G_proj


def tag_to_string(value) -> str:
    if isinstance(value, list):
        return ",".join(map(str, value))
    if value is None:
        return ""
    return str(value)


def simplify_highway(value) -> str:
    h = tag_to_string(value).lower()
    if "cycleway" in h:
        return "cycleway"
    if "residential" in h:
        return "residential"
    if "living_street" in h:
        return "living_street"
    if "tertiary" in h:
        return "tertiary"
    if "secondary" in h:
        return "secondary"
    if "primary" in h:
        return "primary"
    if "trunk" in h:
        return "trunk"
    if "service" in h:
        return "service"
    if "path" in h:
        return "path"
    if "footway" in h:
        return "footway"
    return "other"


def has_cycleway(row: pd.Series) -> int:
    highway = tag_to_string(row.get("highway", "")).lower()
    cycleway = tag_to_string(row.get("cycleway", "")).lower()

    if "cycleway" in highway:
        return 1
    if cycleway not in ["", "nan", "none"]:
        return 1
    return 0


def extract_maxspeed(value) -> float:
    if isinstance(value, list) and value:
        value = value[0]
    value = str(value)
    digits = "".join([c for c in value if c.isdigit()])
    if digits == "":
        return np.nan
    return float(digits)


def make_edge_uid(u, v, key) -> str:
    return f"{u}|{v}|{key}"


def make_pair_id(u, v) -> str:
    a, b = sorted([str(u), str(v)])
    return f"{a}|{b}"


def build_edge_features(
    G_proj: nx.MultiDiGraph,
    output_file: str | Path = OSM_EDGE_FEATURES_FILE,
) -> pd.DataFrame:
    """Create edge feature table from projected OSM graph."""
    edges = ox.graph_to_gdfs(G_proj, nodes=False, edges=True).reset_index()

    edges["edge_uid"] = [make_edge_uid(u, v, k) for u, v, k in zip(edges["u"], edges["v"], edges["key"])]
    edges["pair_id"] = [make_pair_id(u, v) for u, v in zip(edges["u"], edges["v"])]

    edges["highway_simple"] = edges["highway"].apply(simplify_highway) if "highway" in edges.columns else "other"
    edges["has_cycleway"] = edges.apply(has_cycleway, axis=1)

    if "maxspeed" in edges.columns:
        edges["maxspeed_num"] = edges["maxspeed"].apply(extract_maxspeed)
    else:
        edges["maxspeed_num"] = np.nan

    if edges["maxspeed_num"].notna().sum() > 0:
        edges["maxspeed_num"] = edges["maxspeed_num"].fillna(edges["maxspeed_num"].median())
    else:
        edges["maxspeed_num"] = 30.0

    if "length" not in edges.columns:
        edges["length"] = edges.geometry.length

    out = edges[
        [
            "edge_uid", "pair_id", "u", "v", "key",
            "length", "highway_simple", "has_cycleway", "maxspeed_num"
        ]
    ].copy()
    out = out.rename(columns={"length": "edge_length_m"})

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_file, index=False)
    print(f"Saved edge features: {output_file} ({out.shape[0]:,} edges)")
    return out
