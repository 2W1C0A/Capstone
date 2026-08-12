from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import folium
import joblib
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from .config import (
    LOW_EXPOSURE_CLASSES,
    OCCURRENCE_MODEL_FILE,
    OSM_GRAPH_FILE,
    ROUTE_RISK_FILE,
    TEMPORAL_VALIDATION_FILE,
)
from .model_training import predict_edge_occurrence_risk
from .osm_network import edge_uid, load_graph_fast


Route = List[Any]


@dataclass
class RouteSummary:
    distance_km: float
    length_weighted_risk: float
    risk_exposure: float
    n_segments: int

    def as_dict(self):
        return {
            "distance_km": self.distance_km,
            "length_weighted_risk": self.length_weighted_risk,
            "risk_exposure": self.risk_exposure,
            "n_segments": self.n_segments,
        }


class RouteEngine:
    """Compare fastest, historical GIS-risk and ML road-risk routes.

    The third route is driven by the leakage-safe road-only occurrence model
    (src/model_training.py), which predicts relative crash probability per edge
    from road features only. Its 0-1 score feeds the shared cost function together
    with the measured low-exposure class constraint applied in `_cost_weight`.

    Routing costs are evaluated lazily by a NetworkX weight callable
    (`_cost_weight`) over the per-edge constants baked in once at init by
    `_precompute_edge_constants`, so a request no longer rewrites the whole graph.
    """

    def __init__(
        self,
        graph_file: str | Path = OSM_GRAPH_FILE,
        route_risk_file: str | Path = ROUTE_RISK_FILE,
        occurrence_model_file: str | Path = OCCURRENCE_MODEL_FILE,
    ):
        self.graph_file = Path(graph_file)
        self.route_risk_file = Path(route_risk_file)
        self.occurrence_model_file = Path(occurrence_model_file)

        if not self.graph_file.exists():
            raise FileNotFoundError(f"Missing OSM graph: {self.graph_file}")

        if not self.route_risk_file.exists():
            raise FileNotFoundError(f"Missing route risk table: {self.route_risk_file}")

        self.G = load_graph_fast(self.graph_file)
        self.route_risk = pd.read_csv(self.route_risk_file)
        self.route_risk["edge_uid"] = self.route_risk["edge_uid"].astype(str)

        self.historical_lookup = self._lookup("combined_spatial_risk")
        if not self.historical_lookup:
            self.historical_lookup = self._lookup("historical_risk_norm")

        # Third route = leakage-safe road-only occurrence model. Prefer a
        # per-edge column the pipeline may have persisted (instant); otherwise
        # load the saved occurrence model and score the network once at startup.
        if "ml_occurrence_risk" in self.route_risk.columns:
            self.ml_bundle = None
            self.ml_risk_table = self.route_risk
            self.ml_lookup = self._lookup_from_table(self.route_risk, "ml_occurrence_risk")
        else:
            self.ml_bundle = (
                joblib.load(self.occurrence_model_file)
                if self.occurrence_model_file.exists()
                else None
            )
            self.ml_risk_table = (
                self._build_ml_risk_table() if self.ml_bundle is not None else None
            )
            self.ml_lookup = (
                self._lookup_from_table(self.ml_risk_table, "ml_occurrence_risk")
                if self.ml_risk_table is not None
                else {}
            )

        # Bake per-edge constants onto the graph ONCE. Risk does not depend on
        # the request (only safety_preference and hour do), so there is no reason
        # to rewrite 441k edges on every route. See docs finding #10.
        self._precompute_edge_constants()

    def _precompute_edge_constants(self) -> None:
        """Attach immutable per-edge constants used by the routing weights.

        Runs a single pass at engine init. After this the graph is only ever
        read during routing, never written — which also removes the shared-state
        bug where two concurrent app sessions (the engine is cached with
        @st.cache_resource) overwrote each other's per-request risk attributes.
        """
        highway_lookup = dict(
            zip(
                self.route_risk["edge_uid"].astype(str),
                self.route_risk["highway_simple"].astype(str),
            )
        )
        for u, v, k, data in self.G.edges(keys=True, data=True):
            uid = edge_uid(u, v, k)
            data["length_m"] = float(data.get("length", 1.0))
            data["hist_risk"] = float(np.clip(self.historical_lookup.get(uid, 0.0), 0.0, 1.0))
            data["ml_risk"] = float(np.clip(self.ml_lookup.get(uid, 0.0), 0.0, 1.0))
            data["low_exposure"] = highway_lookup.get(uid, "") in LOW_EXPOSURE_CLASSES

        # Node lat/lon for the A* straight-line heuristic, and a per-engine
        # geocode cache (the engine is cached with @st.cache_resource, so this
        # persists across requests and skips repeat Nominatim calls).
        self._node_xy = {
            n: (float(d["y"]), float(d["x"]))
            for n, d in self.G.nodes(data=True)
            if "y" in d and "x" in d
        }
        self._geocode_cache: dict[str, tuple] = {}

    @staticmethod
    def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6_371_000.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlmb = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
        return 2 * r * math.asin(math.sqrt(a))

    def _straight_line_heuristic(self, u, target) -> float:
        """Admissible A* heuristic: straight-line metres from `u` to `target`.

        Every routing cost is `length * (1 + m*risk) + penalty >= length`, and
        the straight-line distance never exceeds the road distance, so this is a
        lower bound on the remaining cost for all three weightings — A* therefore
        returns the same optimal path as Dijkstra, but explores far fewer nodes.
        """
        a = self._node_xy.get(u)
        b = self._node_xy.get(target)
        if a is None or b is None:
            return 0.0
        return self._haversine_m(a[0], a[1], b[0], b[1])

    def _lookup_from_table(self, df: pd.DataFrame | None, col: str) -> dict[str, float]:
        if df is None or col not in df.columns:
            return {}
        tmp = df[["edge_uid", col]].copy()
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce").fillna(0.0).clip(0, 1)
        return dict(zip(tmp["edge_uid"].astype(str), tmp[col].astype(float)))

    def _lookup(self, col: str) -> dict[str, float]:
        return self._lookup_from_table(self.route_risk, col)

    def _build_ml_risk_table(self) -> pd.DataFrame | None:
        """Fallback: score the network with the occurrence model at startup.

        Only reached when the pipeline did not persist `ml_occurrence_risk` into
        the route-risk CSV. Degrades to no ML route on any failure.
        """
        try:
            return predict_edge_occurrence_risk(self.route_risk, self.ml_bundle)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully in the app
            print(f"ML route disabled: could not score edges ({exc}).")
            return None

    # ------------------------------------------------------------------
    # Robust address handling
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_address(address: str) -> str:
        address = (address or "").strip()
        address = " ".join(address.split())
        return address

    @staticmethod
    def _looks_like_coordinate_pair(text: str) -> bool:
        parts = [p.strip() for p in text.replace(";", ",").split(",")]
        if len(parts) != 2:
            return False
        try:
            float(parts[0])
            float(parts[1])
            return True
        except ValueError:
            return False

    @staticmethod
    def _parse_coordinate_pair(text: str) -> tuple[float, float]:
        """Parse 'lat, lon'."""
        parts = [p.strip() for p in text.replace(";", ",").split(",")]
        lat, lon = float(parts[0]), float(parts[1])
        return lat, lon

    @staticmethod
    def _candidate_queries(address: str) -> list[str]:
        """Generate Berlin-specific geocoding fallbacks.

        OSM/Nominatim often fails or returns the wrong city if the user only enters
        a short street or place name. We try increasingly explicit Berlin forms.
        """
        address = RouteEngine._normalise_address(address)
        if not address:
            return []

        lower = address.lower()
        queries = [address]

        if "berlin" not in lower:
            queries.append(f"{address}, Berlin")
            queries.append(f"{address}, Berlin, Germany")

        # German postal addresses often work better without extra punctuation.
        if "," in address:
            simplified = address.replace(",", " ")
            simplified = " ".join(simplified.split())
            if simplified not in queries:
                queries.append(simplified)
            if "berlin" not in simplified.lower():
                queries.append(f"{simplified}, Berlin, Germany")

        # Remove duplicates while preserving order.
        seen = set()
        out = []
        for q in queries:
            key = q.lower()
            if key not in seen:
                seen.add(key)
                out.append(q)
        return out

    @staticmethod
    def _geocode_with_fallbacks(address: str) -> tuple[float, float, str]:
        """Return lat, lon, query_used with useful error message."""
        address = RouteEngine._normalise_address(address)

        if not address:
            raise ValueError("Address is empty. Please enter a Berlin address or place name.")

        if RouteEngine._looks_like_coordinate_pair(address):
            lat, lon = RouteEngine._parse_coordinate_pair(address)
            return lat, lon, "coordinate pair"

        errors = []
        for query in RouteEngine._candidate_queries(address):
            try:
                lat, lon = ox.geocode(query)
                if lat is None or lon is None:
                    errors.append(f"{query}: returned empty coordinates")
                    continue
                # Loose Berlin bounding box guard.
                if not (52.30 <= float(lat) <= 52.75 and 13.00 <= float(lon) <= 13.90):
                    errors.append(f"{query}: found outside Berlin ({lat:.5f}, {lon:.5f})")
                    continue
                return float(lat), float(lon), query
            except Exception as exc:
                errors.append(f"{query}: {type(exc).__name__}: {exc}")

        tried = "\n".join(f"- {e}" for e in errors[-5:])
        raise ValueError(
            "Could not geocode this address inside Berlin.\n\n"
            f"Input: {address}\n\n"
            "Try a more complete format, for example:\n"
            "- Zillestraße 21, 10585 Berlin, Germany\n"
            "- Alexanderplatz, Berlin, Germany\n"
            "- 52.5219, 13.4132\n\n"
            f"Geocoding attempts:\n{tried}"
        )

    @staticmethod
    def validate_address_text(address: str) -> dict:
        """Validate an address without building a route.

        Used by the Streamlit text inputs. Pressing Enter in the address box runs
        this check and stores the result in session_state.
        """
        address = RouteEngine._normalise_address(address)
        try:
            lat, lon, query_used = RouteEngine._geocode_with_fallbacks(address)
            return {
                "ok": True,
                "input": address,
                "query_used": query_used,
                "lat": float(lat),
                "lon": float(lon),
                "message": "Address found inside Berlin.",
            }
        except Exception as exc:
            return {
                "ok": False,
                "input": address,
                "error": str(exc),
            }

    def nearest_node_from_address(self, address: str):
        # Cache (node, coords) per normalised address for the life of the engine.
        # Geocoding is a Nominatim round-trip and often the slowest part of a
        # request; re-running with a different safety slider should not repay it.
        key = self._normalise_address(address)
        cache = getattr(self, "_geocode_cache", None)
        if cache is not None and key in cache:
            return cache[key]

        lat, lon, query_used = self._geocode_with_fallbacks(address)
        try:
            node = ox.distance.nearest_nodes(self.G, X=lon, Y=lat)
        except Exception as exc:
            raise ValueError(
                f"Address was found as '{query_used}' at ({lat:.5f}, {lon:.5f}), "
                f"but snapping it to the OSM graph failed: {exc}"
            ) from exc
        result = (node, (lat, lon))
        if cache is not None:
            cache[key] = result
        return result

    # ------------------------------------------------------------------
    # Costs and routing
    # ------------------------------------------------------------------

    @staticmethod
    def _length_weight(u, v, keydict) -> float:
        """Distance-only weight. Parallel edges: take the shortest."""
        return min(d.get("length_m", d.get("length", 1.0)) for d in keydict.values())

    def _cost_weight(self, risk_attr: str, safety_preference: int, hour: int | None):
        """Return a NetworkX weight callable for one risk model.

        Dijkstra calls this only for the edges it actually relaxes — a few
        thousand on a city route — instead of the old approach that wrote a cost
        to all ~441k edges first. `safety_preference` and `hour` are captured in
        the closure; the per-edge constants (`length_m`, `<risk_attr>`,
        `low_exposure`) were baked in at init.
        """
        penalty_multiplier = float(np.clip(safety_preference, 1, 10)) / 2.0
        is_night = hour is not None and (int(hour) >= 22 or int(hour) <= 4)

        def weight(u, v, keydict) -> float:
            best = None
            for d in keydict.values():
                length = d.get("length_m", d.get("length", 1.0))
                risk = d.get(risk_attr, 0.0)
                hard_penalty = 0.0
                if d.get("low_exposure"):
                    hard_penalty += 60.0
                    if is_night:
                        hard_penalty += 180.0
                cost = length * (1.0 + penalty_multiplier * risk) + hard_penalty
                if best is None or cost < best:
                    best = cost
            return best

        return weight

    @staticmethod
    def _best_edge_data(G: nx.MultiDiGraph, u, v) -> dict | None:
        edges = G.get_edge_data(u, v)
        if edges is None:
            return None
        best_key = min(edges, key=lambda k: edges[k].get("length_m", edges[k].get("length", 1e12)))
        return edges[best_key]

    def summarize_route(self, route: Route, risk_attr: str | None = None) -> RouteSummary:
        """Length and length-weighted risk of a route.

        `risk_attr` selects which precomputed risk to weight by (`hist_risk`,
        `ml_risk`, or None for distance-only). No per-request graph state is
        read, so a route can be summarised under any risk model without
        re-attaching costs.
        """
        total_length = 0.0
        risk_exposure = 0.0
        n = 0

        for u, v in zip(route[:-1], route[1:]):
            data = self._best_edge_data(self.G, u, v)
            if data is None:
                continue
            length = float(data.get("length_m", data.get("length", 0.0)))
            risk = float(data.get(risk_attr, 0.0)) if risk_attr else 0.0
            total_length += length
            risk_exposure += length * risk
            n += 1

        lw_risk = risk_exposure / total_length if total_length > 0 else 0.0
        return RouteSummary(
            distance_km=total_length / 1000.0,
            length_weighted_risk=lw_risk,
            risk_exposure=risk_exposure,
            n_segments=n,
        )

    @staticmethod
    def _pct_reduction(base: float, new: float) -> float:
        if base <= 0:
            return 0.0
        return 100.0 * (base - new) / base

    def nearest_node_from_coords(self, lat: float, lon: float):
        """Snap already-resolved coordinates to the nearest graph node.

        Lets the app pass the lat/lon it already geocoded when validating the
        address, so a route does not re-hit Nominatim for the same text.
        """
        node = ox.distance.nearest_nodes(self.G, X=float(lon), Y=float(lat))
        return node, (float(lat), float(lon))

    def _resolve_node(self, address: str, coords: tuple | None):
        if coords is not None and coords[0] is not None and coords[1] is not None:
            return self.nearest_node_from_coords(coords[0], coords[1])
        return self.nearest_node_from_address(address)

    def compare_routes(
        self,
        start_address: str,
        destination_address: str,
        safety_preference: int = 7,
        hour: int = 8,
        start_coords: tuple | None = None,
        destination_coords: tuple | None = None,
    ) -> Dict[str, Any]:
        # Prefer coordinates the caller already geocoded (address validation);
        # only geocode here when they are not supplied.
        start_node, start_coords = self._resolve_node(start_address, start_coords)
        end_node, dest_coords = self._resolve_node(destination_address, destination_coords)

        try:
            fastest_route = nx.astar_path(
                self.G, start_node, end_node,
                heuristic=self._straight_line_heuristic,
                weight=self._length_weight,
            )
        except nx.NetworkXNoPath as exc:
            raise ValueError(
                "No bicycle-network path was found between the two addresses. "
                "Try addresses closer to central Berlin or check that both places are inside Berlin."
            ) from exc

        fastest_distance = self.summarize_route(fastest_route)

        hist_weight = self._cost_weight("hist_risk", safety_preference, hour)
        fastest_hist = self.summarize_route(fastest_route, "hist_risk")
        historical_route = nx.astar_path(
            self.G, start_node, end_node,
            heuristic=self._straight_line_heuristic, weight=hist_weight,
        )
        historical_summary = self.summarize_route(historical_route, "hist_risk")

        if self.ml_lookup:
            ml_weight = self._cost_weight("ml_risk", safety_preference, hour)
            fastest_ml = self.summarize_route(fastest_route, "ml_risk")
            ml_route = nx.astar_path(
                self.G, start_node, end_node,
                heuristic=self._straight_line_heuristic, weight=ml_weight,
            )
            ml_summary = self.summarize_route(ml_route, "ml_risk")
        else:
            fastest_ml = None
            ml_route = None
            ml_summary = None

        result = {
            "fastest_route": fastest_route,
            "historical_route": historical_route,
            "ml_route": ml_route,
            "fastest_distance_summary": fastest_distance.as_dict(),
            "fastest_historical_summary": fastest_hist.as_dict(),
            "historical_summary": historical_summary.as_dict(),
            "historical_risk_reduction_pct": self._pct_reduction(
                fastest_hist.length_weighted_risk, historical_summary.length_weighted_risk
            ),
            "start_coords": start_coords,
            "destination_coords": dest_coords,
        }

        if ml_summary is not None and fastest_ml is not None:
            result.update({
                "fastest_ml_summary": fastest_ml.as_dict(),
                "ml_summary": ml_summary.as_dict(),
                "ml_risk_reduction_pct": self._pct_reduction(
                    fastest_ml.length_weighted_risk, ml_summary.length_weighted_risk
                ),
            })

        result["recommendation_text"] = self.explain(result)
        return result

    def explain(self, result: dict) -> str:
        fastest = result["fastest_distance_summary"]
        hist_fast = result["fastest_historical_summary"]
        hist = result["historical_summary"]

        lines = []
        lines.append("Route comparison:")
        lines.append(f"- Fastest route: {fastest['distance_km']:.2f} km")
        lines.append(f"  Historical risk exposure score: {hist_fast['length_weighted_risk']:.4f}")
        lines.append("")
        lines.append(
            f"- Historical GIS-risk route: {hist['distance_km']:.2f} km, "
            f"historical risk {hist['length_weighted_risk']:.4f}"
        )
        lines.append(
            f"  Detour: {hist['distance_km'] - fastest['distance_km']:.2f} km; "
            f"historical risk reduction: {result['historical_risk_reduction_pct']:.1f}%"
        )

        if result.get("ml_summary") is not None:
            ml_fast = result["fastest_ml_summary"]
            ml = result["ml_summary"]
            lines.append("")
            lines.append(f"  ML road-risk evaluation of fastest route: {ml_fast['length_weighted_risk']:.4f}")
            lines.append(
                f"- ML road-risk route: {ml['distance_km']:.2f} km, "
                f"ML risk {ml['length_weighted_risk']:.4f}"
            )
            lines.append(
                f"  Detour: {ml['distance_km'] - fastest['distance_km']:.2f} km; "
                f"ML risk reduction: {result['ml_risk_reduction_pct']:.1f}%"
            )

        lines.append("")
        lines.append(
            "Interpretation: these are relative model scores, not personal crash probabilities. "
            "The historical route reduces historical spatial-risk exposure; the ML route uses a "
            "leakage-safe road-only occurrence model."
        )
        return "\n".join(lines)

    @staticmethod
    def _held_out_comparison_block(
        validation_file: str | Path = TEMPORAL_VALIDATION_FILE,
    ) -> list[str]:
        """One line per model: how well each predicts future crashes it never saw.

        Reads the JSON written by spatial_risk.temporal_validation. Best-effort:
        returns nothing if the file is absent or has no `surfaces`, so the
        recommendation is unchanged.
        """
        try:
            path = Path(validation_file)
            if not path.exists():
                return []
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = [
                s for s in (data.get("surfaces") or [])
                if s.get("top_decile_recall") is not None
            ]
            if not rows:
                return []

            label = {
                "historical_gis": "Historical GIS",
                "occurrence_ml": "ML occurrence",
                "frequency_nb": "Frequency SPF",
            }
            best = max(rows, key=lambda s: s["top_decile_recall"])
            train_years = str(data.get("train_years", "")).strip()
            test_years = str(data.get("test_years", "")).strip()

            lines = [
                "",
                f"Which model predicts best (tested on later crashes it never saw, "
                f"{train_years} → {test_years}; a random 10% of streets would catch 10%):",
            ]
            for s in rows:
                name = label.get(s["surface"], s["surface"])
                recall = 100.0 * float(s["top_decile_recall"])
                tag = "  ← best" if s is best else ""
                lines.append(
                    f"- {name}: {recall:.0f}% of future crashes fall in its top-10% streets{tag}"
                )
            return lines
        except Exception:
            return []

    def route_to_coordinates(self, route: Route | None) -> list[tuple[float, float]]:
        if route is None:
            return []
        coords = []
        for node in route:
            d = self.G.nodes[node]
            coords.append((float(d["y"]), float(d["x"])))
        return coords

    def make_map(self, result: dict) -> folium.Map:
        start = result["start_coords"]
        dest = result["destination_coords"]
        center = [(start[0] + dest[0]) / 2, (start[1] + dest[1]) / 2]

        m = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")
        folium.PolyLine(self.route_to_coordinates(result["fastest_route"]), color="red", weight=5, opacity=0.75, tooltip="Fastest").add_to(m)
        folium.PolyLine(self.route_to_coordinates(result["historical_route"]), color="#1f4e8c", weight=5, opacity=0.9, tooltip="Historical GIS-risk").add_to(m)

        if result.get("ml_route") is not None:
            folium.PolyLine(self.route_to_coordinates(result["ml_route"]), color="green", weight=5, opacity=0.85, tooltip="ML road-risk").add_to(m)

        folium.Marker(start, tooltip="Start", icon=folium.Icon(color="blue", icon="play")).add_to(m)
        folium.Marker(dest, tooltip="Destination", icon=folium.Icon(color="black", icon="flag")).add_to(m)

        legend = """
        <div style="position: fixed; bottom: 40px; left: 40px; z-index: 9999;
                    background: #ffffff; color: #1a1a1a; padding: 12px 14px;
                    border: 1px solid #cccccc; border-radius: 6px;
                    font-size: 14px; line-height: 1.7;">
            <div style="font-weight: 600; margin-bottom: 6px; color: #1a1a1a;">Route legend</div>
            <div style="color: #1a1a1a;"><span style="color:#d64545;">&#9632;</span>&nbsp; Fastest route</div>
            <div style="color: #1a1a1a;"><span style="color:#1F4E8C;">&#9632;</span>&nbsp; Historical GIS-risk route</div>
            <div style="color: #1a1a1a;"><span style="color:#12A55F;">&#9632;</span>&nbsp; ML road-risk route</div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend))
        return m

    def compare_and_map(self, *args, **kwargs):
        result = self.compare_routes(*args, **kwargs)
        return result, self.make_map(result)