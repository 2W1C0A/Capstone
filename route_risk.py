"""Risk-weighted bicycle routing for Berlin.

Maps Unfallatlas accident points onto an OSMnx graph, derives a per-edge risk
score, and routes on a cost that trades detour length against historical risk.

Pipeline
--------
    G = ox.graph_from_place("Berlin, Germany", network_type="bike")
    Gp = ox.project_graph(G, to_crs="EPSG:25833")      # UTM 33N
    snapped = snap_accidents(Gp, df)
    risk = build_edge_risk(Gp, snapped)
    apply_risk_weights(Gp, risk, alpha=2.0)
    route = safest_route(Gp, (52.52, 13.40), (52.49, 13.43))

Requires osmnx >= 2.0, geopandas, networkx, pandas, numpy.
"""

from __future__ import annotations

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

# Berlin sits in UTM zone 33N. Metric CRS is required for distance-based snapping.
BERLIN_CRS = "EPSG:25833"

# Severity weights. The 1/2/3 scheme mirrors the notebook's heatmap, but road
# safety practice weights fatalities far more steeply -- see SEVERITY_KSI.
SEVERITY_LINEAR = {"minor_injury": 1.0, "serious_injury": 2.0, "fatal": 3.0}
SEVERITY_KSI = {"minor_injury": 1.0, "serious_injury": 8.0, "fatal": 30.0}


def snap_accidents(
    G_proj: nx.MultiDiGraph,
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    max_snap_dist: float = 25.0,
) -> pd.DataFrame:
    """Attach each accident to its nearest graph edge.

    Unfallatlas coordinates are reduced in precision before publication, so a
    point can land tens of metres from the way it belongs to. Snaps further
    than `max_snap_dist` metres are dropped rather than assigned to whatever
    happens to be closest -- a parallel side street or a park path.

    Returns a copy of the kept rows with `u`, `v`, `key` and `snap_dist` added.
    """
    if G_proj.graph.get("crs") in (None, "EPSG:4326", "epsg:4326"):
        raise ValueError(
            "G_proj must be projected to a metric CRS "
            "(ox.project_graph(G, to_crs=BERLIN_CRS)); snap distances in "
            "degrees are meaningless."
        )

    pts = df[[lat_col, lon_col]].dropna()
    if pts.empty:
        raise ValueError("No rows with both latitude and longitude.")

    # Guard against the classic XGCSWGS84/YGCSWGS84 swap: X is longitude.
    if not (pts[lat_col].between(52.3, 52.7).mean() > 0.9):
        raise ValueError(
            f"{lat_col!r} does not look like Berlin latitude (~52.3-52.7). "
            "Check that XGCSWGS84 -> longitude and YGCSWGS84 -> latitude."
        )

    geom = gpd.GeoSeries(
        gpd.points_from_xy(pts[lon_col], pts[lat_col]), crs="EPSG:4326"
    ).to_crs(G_proj.graph["crs"])

    edges, dists = ox.nearest_edges(
        G_proj, X=geom.x.to_numpy(), Y=geom.y.to_numpy(), return_dist=True
    )

    out = df.loc[pts.index].copy()
    # nearest_edges returns a 1-D object array of (u, v, key) tuples.
    out[["u", "v", "key"]] = pd.DataFrame(
        list(edges), columns=["u", "v", "key"], index=out.index
    )
    out["snap_dist"] = np.asarray(dists)

    kept = out[out["snap_dist"] <= max_snap_dist]
    print(
        f"Snapped {len(kept):,} / {len(df):,} accidents within "
        f"{max_snap_dist:g} m ({len(kept) / len(df):.1%}); "
        f"median offset {kept['snap_dist'].median():.1f} m"
    )
    return kept


def build_edge_risk(
    G_proj: nx.MultiDiGraph,
    snapped: pd.DataFrame,
    severity_col: str = "severity_label",
    severity_weights: dict[str, float] | None = None,
    shrinkage: float = 5.0,
) -> pd.DataFrame:
    """Severity-weighted risk per 100 m of edge, shrunk toward a class prior.

    Eight years of accidents spread over a citywide network leaves the large
    majority of edges at zero and a handful of short edges with a freak count.
    Raw rates would rank those short edges as the most dangerous streets in
    Berlin. So each edge is pulled toward the mean rate for its `highway`
    class (empirical Bayes):

        score = (observed + m * prior * len100) / (len100 + m)

    `shrinkage` is m, in units of 100 m of road. Higher m trusts the class
    average more and the individual edge less.

    Risk is keyed on the undirected pair {u, v}: an accident on a two-way
    street is evidence about both directions of travel.
    """
    weights = severity_weights or SEVERITY_LINEAR

    edges = ox.graph_to_gdfs(G_proj, nodes=False)
    edges = edges.reset_index()[["u", "v", "key", "length", "highway"]]
    edges["highway"] = edges["highway"].apply(
        lambda h: h[0] if isinstance(h, list) else h
    )
    edges["pair"] = [frozenset(p) for p in zip(edges["u"], edges["v"])]

    # One row per undirected pair; length is shared, not summed.
    pairs = edges.groupby("pair", as_index=False).agg(
        length=("length", "max"), highway=("highway", "first")
    )
    pairs["len100"] = pairs["length"] / 100.0

    hit = snapped.copy()
    hit["pair"] = [frozenset(p) for p in zip(hit["u"], hit["v"])]
    if severity_col in hit.columns:
        hit["w"] = hit[severity_col].map(weights).fillna(1.0)
    else:
        print(f"No {severity_col!r} column; counting accidents unweighted.")
        hit["w"] = 1.0

    obs = hit.groupby("pair")["w"].agg(observed="sum", n_accidents="size")
    pairs = pairs.merge(obs, on="pair", how="left").fillna(
        {"observed": 0.0, "n_accidents": 0}
    )

    # Class prior: weighted accidents per 100 m for each highway type.
    prior = pairs.groupby("highway").apply(
        lambda g: g["observed"].sum() / max(g["len100"].sum(), 1e-9),
        include_groups=False,
    )
    pairs["prior"] = pairs["highway"].map(prior)
    global_prior = pairs["observed"].sum() / max(pairs["len100"].sum(), 1e-9)
    pairs["prior"] = pairs["prior"].fillna(global_prior)

    pairs["risk"] = (pairs["observed"] + shrinkage * pairs["prior"] * pairs["len100"]) / (
        pairs["len100"] + shrinkage
    )

    # Normalise to [0, 1], capped at the 95th percentile so a few extreme
    # junctions don't flatten the rest of the scale.
    cap = pairs["risk"].quantile(0.95)
    pairs["risk_norm"] = (pairs["risk"] / cap).clip(upper=1.0) if cap > 0 else 0.0

    print(
        f"{len(pairs):,} undirected segments | "
        f"{(pairs['n_accidents'] > 0).mean():.1%} with >=1 accident | "
        f"risk cap (p95) = {cap:.2f}"
    )
    return pairs


def build_node_risk(
    G_proj: nx.MultiDiGraph,
    snapped: pd.DataFrame,
    radius: float = 20.0,
    severity_col: str = "severity_label",
    severity_weights: dict[str, float] | None = None,
) -> pd.Series:
    """Weighted accident load within `radius` metres of each intersection.

    Turning and crossing conflicts dominate the accident-type breakdown, which
    means much of the risk lives at nodes, not along segments. Edge-only
    scoring smears an intersection's accidents across whichever approach arm
    they snapped to.
    """
    weights = severity_weights or SEVERITY_LINEAR

    pts = snapped[["latitude", "longitude"]].dropna()
    geom = gpd.GeoSeries(
        gpd.points_from_xy(pts["longitude"], pts["latitude"]), crs="EPSG:4326"
    ).to_crs(G_proj.graph["crs"])

    nodes, dists = ox.nearest_nodes(
        G_proj, X=geom.x.to_numpy(), Y=geom.y.to_numpy(), return_dist=True
    )

    near = pd.DataFrame({"node": nodes, "dist": dists}, index=pts.index)
    near = near[near["dist"] <= radius]
    if severity_col in snapped.columns:
        near["w"] = snapped.loc[near.index, severity_col].map(weights).fillna(1.0)
    else:
        near["w"] = 1.0

    load = near.groupby("node")["w"].sum()
    print(f"{len(load):,} intersections carry accidents within {radius:g} m")
    return load


def apply_risk_weights(
    G_proj: nx.MultiDiGraph,
    edge_risk: pd.DataFrame,
    alpha: float = 2.0,
    node_risk: pd.Series | None = None,
    node_penalty: float = 15.0,
) -> nx.MultiDiGraph:
    """Write a `risk_cost` attribute onto every edge, in place.

        risk_cost = length * (1 + alpha * risk_norm) + node_penalty * node_risk_norm

    alpha is the exchange rate between metres and risk. alpha=0 reproduces the
    shortest path; alpha=2 means a cyclist accepts up to ~3x the distance to
    reach the safest available street. Tune it by looking at the detour
    figures from `compare_routes`, not by taste.
    """
    risk_by_pair = dict(zip(edge_risk["pair"], edge_risk["risk_norm"]))

    nrisk = {}
    if node_risk is not None and len(node_risk):
        cap = node_risk.quantile(0.95)
        if cap > 0:
            nrisk = (node_risk / cap).clip(upper=1.0).to_dict()

    for u, v, k, data in G_proj.edges(keys=True, data=True):
        r = risk_by_pair.get(frozenset((u, v)), 0.0)
        length = data.get("length", 0.0)
        # Charged at the edge's destination node: entering an intersection.
        data["risk_cost"] = length * (1 + alpha * r) + node_penalty * nrisk.get(v, 0.0)
        data["risk_norm"] = r

    return G_proj


def safest_route(
    G_proj: nx.MultiDiGraph,
    origin: tuple[float, float],
    destination: tuple[float, float],
) -> list[int]:
    """Lowest-`risk_cost` path between two (lat, lon) pairs."""
    geom = gpd.GeoSeries(
        gpd.points_from_xy(
            [origin[1], destination[1]], [origin[0], destination[0]]
        ),
        crs="EPSG:4326",
    ).to_crs(G_proj.graph["crs"])

    orig, dest = ox.nearest_nodes(G_proj, X=geom.x.to_numpy(), Y=geom.y.to_numpy())
    return nx.shortest_path(G_proj, orig, dest, weight="risk_cost")


def compare_routes(
    G_proj: nx.MultiDiGraph,
    origin: tuple[float, float],
    destination: tuple[float, float],
) -> pd.DataFrame:
    """Shortest vs safest route: the detour/risk trade-off, for the deck.

    This table is the honest way to present the product. "3% longer, 22% less
    risk exposure" is a claim you can defend; "safest route" is not.
    """
    geom = gpd.GeoSeries(
        gpd.points_from_xy(
            [origin[1], destination[1]], [origin[0], destination[0]]
        ),
        crs="EPSG:4326",
    ).to_crs(G_proj.graph["crs"])
    orig, dest = ox.nearest_nodes(G_proj, X=geom.x.to_numpy(), Y=geom.y.to_numpy())

    rows = []
    for label, weight in (("shortest", "length"), ("safest", "risk_cost")):
        path = nx.shortest_path(G_proj, orig, dest, weight=weight)
        length = risk = 0.0
        for u, v in zip(path[:-1], path[1:]):
            d = min(G_proj[u][v].values(), key=lambda e: e.get("length", np.inf))
            length += d.get("length", 0.0)
            risk += d.get("length", 0.0) * d.get("risk_norm", 0.0)
        rows.append(
            {
                "route": label,
                "length_m": round(length),
                "risk_exposure": round(risk, 1),
                "n_segments": len(path) - 1,
            }
        )

    out = pd.DataFrame(rows)
    short, safe = out.iloc[0], out.iloc[1]
    out.attrs["detour_pct"] = round(
        (safe["length_m"] / short["length_m"] - 1) * 100, 1
    )
    out.attrs["risk_reduction_pct"] = round(
        (1 - safe["risk_exposure"] / max(short["risk_exposure"], 1e-9)) * 100, 1
    )
    return out
