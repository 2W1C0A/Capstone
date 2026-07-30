"""Synthetic end-to-end check of route_risk.py (no Overpass access needed)."""

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

import route_risk as rr

rng = np.random.default_rng(0)

# --- Build a fake 8x8 street grid over central Berlin -----------------------
N = 8
STEP = 0.0025  # ~180 m N-S
LAT0, LON0 = 52.500, 13.380

G = nx.MultiDiGraph(crs="EPSG:4326", simplified=True)
node_id = {}
for i in range(N):
    for j in range(N):
        nid = i * N + j
        node_id[(i, j)] = nid
        G.add_node(nid, x=LON0 + j * STEP * 1.6, y=LAT0 + i * STEP)

for i in range(N):
    for j in range(N):
        for di, dj in ((0, 1), (1, 0)):
            a, b = (i, j), (i + di, j + dj)
            if b not in node_id:
                continue
            u, v = node_id[a], node_id[b]
            dx = (G.nodes[v]["x"] - G.nodes[u]["x"]) * 68000
            dy = (G.nodes[v]["y"] - G.nodes[u]["y"]) * 111000
            length = float(np.hypot(dx, dy))
            hw = "primary" if i == 4 else "residential"
            G.add_edge(u, v, 0, length=length, highway=hw)
            G.add_edge(v, u, 0, length=length, highway=hw)

Gp = ox.project_graph(G, to_crs=rr.BERLIN_CRS)
print(f"graph: {Gp.number_of_nodes()} nodes, {Gp.number_of_edges()} edges, "
      f"crs={Gp.graph['crs']}\n")

# --- Fake accidents: heavily concentrated on the row i=4 "primary" corridor -
records = []
for _ in range(900):
    i = 4 if rng.random() < 0.7 else rng.integers(0, N)
    j = rng.integers(0, N - 1)
    t = rng.random()
    lat = LAT0 + i * STEP + rng.normal(0, 0.00004)
    lon = LON0 + (j + t) * STEP * 1.6 + rng.normal(0, 0.00004)
    sev = rng.choice(
        ["minor_injury", "serious_injury", "fatal"], p=[0.86, 0.13, 0.01]
    )
    records.append({"latitude": lat, "longitude": lon, "severity_label": sev})

# A handful of junk points far outside the network, to exercise the filter.
for _ in range(40):
    records.append({
        "latitude": LAT0 + rng.uniform(0, N * STEP),
        "longitude": LON0 + rng.uniform(0, N * STEP * 1.6),
        "severity_label": "minor_injury",
    })
df = pd.DataFrame(records)

# --- Run the pipeline -------------------------------------------------------
snapped = rr.snap_accidents(Gp, df, max_snap_dist=25.0)
edge_risk = rr.build_edge_risk(Gp, snapped, severity_weights=rr.SEVERITY_KSI)
node_risk = rr.build_node_risk(Gp, snapped, severity_weights=rr.SEVERITY_KSI)
rr.apply_risk_weights(Gp, edge_risk, alpha=2.0, node_risk=node_risk)

print("\nriskiest segments by highway class:")
print(edge_risk.groupby("highway")["risk"].agg(["mean", "max", "count"]).round(2))

# Route across the grid; the direct line runs along the dangerous corridor.
orig = (LAT0 + 4 * STEP, LON0)
dest = (LAT0 + 4 * STEP, LON0 + (N - 1) * STEP * 1.6)

table = rr.compare_routes(Gp, orig, dest)
print("\n", table.to_string(index=False), sep="")
print(f"detour: {table.attrs['detour_pct']}%  |  "
      f"risk reduction: {table.attrs['risk_reduction_pct']}%")

route = rr.safest_route(Gp, orig, dest)
print(f"\nsafest route: {len(route)} nodes")
assert len(route) >= 2
assert table.attrs["detour_pct"] >= 0, "safest route cannot be shorter"
assert table.attrs["risk_reduction_pct"] > 0, "risk routing did nothing"
print("\nOK")
