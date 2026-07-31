from __future__ import annotations

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
)
from .model_training import predict_edge_occurrence_risk
from .osm_network import edge_uid


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
    """Compare fastest, historical GIS-risk and leakage-safe ML road-risk routes."""

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

        self.G = ox.load_graphml(self.graph_file)
        self.route_risk = pd.read_csv(self.route_risk_file)
        self.route_risk["edge_uid"] = self.route_risk["edge_uid"].astype(str)

        self.historical_lookup = self._lookup("combined_spatial_risk")
        if not self.historical_lookup:
            self.historical_lookup = self._lookup("historical_risk_norm")

        self.model_bundle = joblib.load(self.occurrence_model_file) if self.occurrence_model_file.exists() else None
        self.ml_risk_table = self._build_ml_risk_table() if self.model_bundle is not None else None
        self.ml_lookup = self._lookup_from_table(self.ml_risk_table, "ml_occurrence_risk") if self.ml_risk_table is not None else {}

    def _lookup_from_table(self, df: pd.DataFrame | None, col: str) -> dict[str, float]:
        if df is None or col not in df.columns:
            return {}
        tmp = df[["edge_uid", col]].copy()
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce").fillna(0.0).clip(0, 1)
        return dict(zip(tmp["edge_uid"].astype(str), tmp[col].astype(float)))

    def _lookup(self, col: str) -> dict[str, float]:
        return self._lookup_from_table(self.route_risk, col)

    def _build_ml_risk_table(self) -> pd.DataFrame:
        return predict_edge_occurrence_risk(self.route_risk, self.model_bundle)

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
        lat, lon, query_used = self._geocode_with_fallbacks(address)
        try:
            node = ox.distance.nearest_nodes(self.G, X=lon, Y=lat)
        except Exception as exc:
            raise ValueError(
                f"Address was found as '{query_used}' at ({lat:.5f}, {lon:.5f}), "
                f"but snapping it to the OSM graph failed: {exc}"
            ) from exc
        return node, (lat, lon)

    # ------------------------------------------------------------------
    # Costs and routing
    # ------------------------------------------------------------------

    def _attach_costs(self, risk_lookup: dict[str, float], safety_preference: int = 7, hour: int | None = None) -> None:
        penalty_multiplier = float(np.clip(safety_preference, 1, 10)) / 2.0
        is_night = hour is not None and (int(hour) >= 22 or int(hour) <= 4)

        highway_lookup = dict(zip(self.route_risk["edge_uid"].astype(str), self.route_risk["highway_simple"].astype(str)))

        for u, v, k, data in self.G.edges(keys=True, data=True):
            uid = edge_uid(u, v, k)
            length = float(data.get("length", 1.0))
            risk = float(np.clip(risk_lookup.get(uid, 0.0), 0.0, 1.0))
            highway = highway_lookup.get(uid, "")

            hard_penalty = 0.0
            if highway in LOW_EXPOSURE_CLASSES:
                hard_penalty += 60.0
            if is_night and highway in LOW_EXPOSURE_CLASSES:
                hard_penalty += 180.0

            data["length_cost"] = length
            data["active_risk"] = risk
            data["safe_cost"] = length * (1.0 + penalty_multiplier * risk) + hard_penalty

    def _attach_distance_costs(self) -> None:
        for _, _, _, data in self.G.edges(keys=True, data=True):
            data["length_cost"] = float(data.get("length", 1.0))

    @staticmethod
    def _best_edge_data(G: nx.MultiDiGraph, u, v, weight: str) -> dict | None:
        edges = G.get_edge_data(u, v)
        if edges is None:
            return None
        best_key = min(edges, key=lambda k: edges[k].get(weight, 1e12))
        return edges[best_key]

    def summarize_route(self, route: Route, weight: str = "length_cost") -> RouteSummary:
        total_length = 0.0
        risk_exposure = 0.0
        n = 0

        for u, v in zip(route[:-1], route[1:]):
            data = self._best_edge_data(self.G, u, v, weight)
            if data is None:
                continue
            length = float(data.get("length", 0.0))
            risk = float(data.get("active_risk", 0.0))
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

    def compare_routes(
        self,
        start_address: str,
        destination_address: str,
        safety_preference: int = 7,
        hour: int = 8,
    ) -> Dict[str, Any]:
        start_node, start_coords = self.nearest_node_from_address(start_address)
        end_node, dest_coords = self.nearest_node_from_address(destination_address)

        try:
            self._attach_distance_costs()
            fastest_route = nx.shortest_path(self.G, start_node, end_node, weight="length_cost")
        except nx.NetworkXNoPath as exc:
            raise ValueError(
                "No bicycle-network path was found between the two addresses. "
                "Try addresses closer to central Berlin or check that both places are inside Berlin."
            ) from exc

        fastest_distance = self.summarize_route(fastest_route, "length_cost")

        self._attach_costs(self.historical_lookup, safety_preference=safety_preference, hour=hour)
        fastest_hist = self.summarize_route(fastest_route, "length_cost")
        historical_route = nx.shortest_path(self.G, start_node, end_node, weight="safe_cost")
        historical_summary = self.summarize_route(historical_route, "safe_cost")

        if self.ml_lookup:
            self._attach_costs(self.ml_lookup, safety_preference=safety_preference, hour=hour)
            fastest_ml = self.summarize_route(fastest_route, "length_cost")
            ml_route = nx.shortest_path(self.G, start_node, end_node, weight="safe_cost")
            ml_summary = self.summarize_route(ml_route, "safe_cost")
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
        folium.PolyLine(self.route_to_coordinates(result["historical_route"]), color="orange", weight=5, opacity=0.75, tooltip="Historical GIS-risk").add_to(m)

        if result.get("ml_route") is not None:
            folium.PolyLine(self.route_to_coordinates(result["ml_route"]), color="green", weight=5, opacity=0.85, tooltip="ML road-risk").add_to(m)

        folium.Marker(start, tooltip="Start", icon=folium.Icon(color="blue", icon="play")).add_to(m)
        folium.Marker(dest, tooltip="Destination", icon=folium.Icon(color="black", icon="flag")).add_to(m)

        legend = """
        <div style="position: fixed; bottom: 40px; left: 40px; z-index: 9999;
                    background: white; padding: 12px; border: 2px solid grey;
                    border-radius: 6px; font-size: 14px;">
            <b>Route Legend</b><br>
            <span style="color:red;">■</span> Fastest route<br>
            <span style="color:orange;">■</span> Historical GIS-risk route<br>
            <span style="color:green;">■</span> ML road-risk route<br>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend))
        return m

    def compare_and_map(self, *args, **kwargs):
        result = self.compare_routes(*args, **kwargs)
        return result, self.make_map(result)
