from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import folium
import joblib
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from .config import (
    OSM_GRAPH_FILE,
    EDGE_RISK_FILE,
    MODEL_FILE,
    SPATIAL_RISK_FILE,
)
from .data_pipeline import get_season
from .model_training import MODEL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES


Route = List[Any]


@dataclass
class RouteSummary:
    """Summary statistics for one route under one active risk model."""
    distance_km: float
    mean_risk: float
    length_weighted_risk: float
    risk_exposure: float
    n_segments: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "distance_km": self.distance_km,
            "mean_risk": self.mean_risk,
            "length_weighted_risk": self.length_weighted_risk,
            "risk_exposure": self.risk_exposure,
            "n_segments": self.n_segments,
        }


def edge_uid(u, v, k) -> str:
    """Stable edge ID used to match OSM graph edges with risk-table rows."""
    return f"{u}|{v}|{k}"


class RouteEngine:
    """Fastest vs historical-risk vs ML-safest bicycle route engine.

    Important fix in this version
    -----------------------------
    The fastest route is calculated only by distance, but its risk must still be
    evaluated under the historical-risk model and under the ML-risk model.

    Otherwise the fastest route gets risk = 0.0000, because the distance-only
    mode has no active risk. That was the source of the previous wrong output.
    """

    def __init__(
        self,
        graph_file: str | Path = OSM_GRAPH_FILE,
        edge_risk_file: str | Path = EDGE_RISK_FILE,
        model_file: str | Path = MODEL_FILE,
        spatial_risk_file: str | Path = SPATIAL_RISK_FILE,
    ):
        self.graph_file = Path(graph_file)
        self.edge_risk_file = Path(edge_risk_file)
        self.model_file = Path(model_file)
        self.spatial_risk_file = Path(spatial_risk_file)

        if not self.graph_file.exists():
            raise FileNotFoundError(
                f"OSM graph not found: {self.graph_file}\n"
                "Run python run_pipeline.py --use-clean-data first."
            )

        self.G = ox.load_graphml(self.graph_file)

        self.edge_risk = self._load_edge_risk()
        self.model = joblib.load(self.model_file) if self.model_file.exists() else None

    # ------------------------------------------------------------------
    # Risk loading and prediction
    # ------------------------------------------------------------------

    def _load_edge_risk(self) -> pd.DataFrame:
        """Load ML edge-risk scores if available; otherwise use spatial-risk file."""
        if self.edge_risk_file.exists():
            return pd.read_csv(self.edge_risk_file)

        if self.spatial_risk_file.exists():
            return pd.read_csv(self.spatial_risk_file)

        raise FileNotFoundError(
            "No edge risk file found. Expected one of:\n"
            f"- {self.edge_risk_file}\n"
            f"- {self.spatial_risk_file}\n"
            "Run python run_pipeline.py --use-clean-data first."
        )

    @staticmethod
    def _lookup_from_df(df: pd.DataFrame, risk_col: str) -> Dict[str, float]:
        """Create edge_uid -> risk lookup dictionary."""
        if "edge_uid" not in df.columns:
            raise ValueError("Risk table must contain an 'edge_uid' column.")

        if risk_col not in df.columns:
            return {}

        tmp = df[["edge_uid", risk_col]].copy()
        tmp[risk_col] = pd.to_numeric(tmp[risk_col], errors="coerce").fillna(0.0)

        # Normalize if values are outside 0-1.
        mn = tmp[risk_col].min()
        mx = tmp[risk_col].max()
        if mx > 1.0 or mn < 0.0:
            if mx > mn:
                tmp[risk_col] = (tmp[risk_col] - mn) / (mx - mn)
            else:
                tmp[risk_col] = 0.0

        return {
            str(row["edge_uid"]): float(np.clip(row[risk_col], 0.0, 1.0))
            for _, row in tmp.iterrows()
        }

    def _predict_dynamic_ml_risk(
        self,
        month: int,
        hour: int,
        day_of_week: int,
    ) -> Dict[str, float]:
        """Predict edge risks dynamically for selected time if model exists.

        If the trained model is not available, fall back to a precomputed
        ml_accident_risk column in the edge-risk table.
        """
        if self.model is None:
            if "ml_accident_risk" in self.edge_risk.columns:
                return self._lookup_from_df(self.edge_risk, "ml_accident_risk")
            return self._historical_risk_lookup()

        df = self.edge_risk.copy()
        df["month"] = int(month)
        df["hour"] = int(hour)
        df["day_of_week"] = int(day_of_week)
        df["is_weekend"] = int(day_of_week in [1, 7])
        df["is_rush_hour"] = int((7 <= hour <= 9) or (16 <= hour <= 18))
        df["is_night"] = int((20 <= hour <= 23) or (0 <= hour <= 5))
        df["season"] = get_season(month)

        for col in NUMERIC_FEATURES:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        for col in CATEGORICAL_FEATURES:
            if col not in df.columns:
                df[col] = "unknown"
            df[col] = df[col].fillna("unknown").astype(str)

        df["dynamic_ml_risk"] = self.model.predict_proba(df[MODEL_FEATURES])[:, 1]
        return self._lookup_from_df(df, "dynamic_ml_risk")

    def _historical_risk_lookup(self) -> Dict[str, float]:
        """Return historical GIS risk lookup.

        Priority:
        1. combined_spatial_risk
        2. historical_risk
        3. historical_risk_norm
        4. risk_score
        5. ml_accident_risk
        """
        for col in [
            "combined_spatial_risk",
            "historical_risk",
            "historical_risk_norm",
            "risk_score",
            "ml_accident_risk",
        ]:
            if col in self.edge_risk.columns:
                return self._lookup_from_df(self.edge_risk, col)

        return {}

    def _attach_costs_from_lookup(
        self,
        risk_lookup: Dict[str, float],
        safety_preference: int,
    ) -> None:
        """Attach active risk and routing costs to the graph."""
        safety_preference = int(np.clip(safety_preference, 1, 10))
        penalty_multiplier = safety_preference / 2.0

        for u, v, k, data in self.G.edges(keys=True, data=True):
            uid = edge_uid(u, v, k)
            length = float(data.get("length", 1.0))
            risk = float(np.clip(risk_lookup.get(uid, 0.0), 0.0, 1.0))

            data["fast_cost"] = length
            data["active_risk"] = risk
            data["safe_cost"] = length * (1.0 + penalty_multiplier * risk)

    def _attach_distance_only_costs(self) -> None:
        """Attach distance-only costs without changing active_risk."""
        for _, _, _, data in self.G.edges(keys=True, data=True):
            length = float(data.get("length", 1.0))
            data["fast_cost"] = length

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def nearest_node_from_address(self, address: str) -> Tuple[Any, Tuple[float, float]]:
        """Geocode address and return nearest OSM graph node."""
        lat, lon = ox.geocode(address)
        node = ox.distance.nearest_nodes(self.G, X=lon, Y=lat)
        return node, (lat, lon)

    @staticmethod
    def _best_edge_data(G: nx.MultiDiGraph, u, v, weight: str) -> Optional[dict]:
        """For MultiDiGraph, choose the edge with the lowest selected weight."""
        edges = G.get_edge_data(u, v)
        if edges is None:
            return None
        best_key = min(edges, key=lambda k: edges[k].get(weight, 1e12))
        return edges[best_key]

    def summarize_route(self, route: Route, weight_choice: str = "fast_cost") -> RouteSummary:
        """Summarize route distance and active risk.

        Risk is always taken from data["active_risk"], so call
        _attach_costs_from_lookup(...) before summarizing.
        """
        total_length = 0.0
        risk_exposure = 0.0
        risks = []
        n_segments = 0

        for u, v in zip(route[:-1], route[1:]):
            data = self._best_edge_data(self.G, u, v, weight_choice)
            if data is None:
                continue

            length = float(data.get("length", 0.0))
            risk = float(data.get("active_risk", 0.0))

            total_length += length
            risk_exposure += length * risk
            risks.append(risk)
            n_segments += 1

        mean_risk = float(np.mean(risks)) if risks else 0.0
        length_weighted_risk = risk_exposure / total_length if total_length > 0 else 0.0

        return RouteSummary(
            distance_km=total_length / 1000.0,
            mean_risk=mean_risk,
            length_weighted_risk=length_weighted_risk,
            risk_exposure=risk_exposure,
            n_segments=n_segments,
        )

    @staticmethod
    def _safe_percent_reduction(base_risk: float, new_risk: float) -> float:
        """Risk reduction relative to base risk."""
        if base_risk <= 0:
            return 0.0
        return 100.0 * (base_risk - new_risk) / base_risk

    # ------------------------------------------------------------------
    # Main comparison
    # ------------------------------------------------------------------

    def compare_routes(
        self,
        start_address: str,
        destination_address: str,
        month: int = 7,
        hour: int = 8,
        day_of_week: int = 3,
        safety_preference: int = 7,
    ) -> Dict[str, Any]:
        """Compare fastest, historical-risk, and ML-safest routes.

        Correct comparison logic:
        1. Calculate fastest route using distance only.
        2. Evaluate that same fastest route under historical risk.
        3. Calculate historical-risk route and compare with fastest under historical risk.
        4. Evaluate the same fastest route under ML risk.
        5. Calculate ML-safest route and compare with fastest under ML risk.

        This avoids the old bug where fastest route risk was always 0.
        """
        start_node, start_coords = self.nearest_node_from_address(start_address)
        end_node, end_coords = self.nearest_node_from_address(destination_address)

        # --------------------------------------------------------------
        # 1. Fastest route by distance only
        # --------------------------------------------------------------
        self._attach_distance_only_costs()
        fastest_route = nx.shortest_path(
            self.G,
            start_node,
            end_node,
            weight="fast_cost",
        )

        # Distance-only summary is useful for reporting distance.
        # Risk in this summary is not used for risk comparison.
        fastest_distance_summary = self.summarize_route(
            fastest_route,
            weight_choice="fast_cost",
        )

        # --------------------------------------------------------------
        # 2. Historical GIS-risk comparison
        # --------------------------------------------------------------
        historical_lookup = self._historical_risk_lookup()
        self._attach_costs_from_lookup(
            risk_lookup=historical_lookup,
            safety_preference=safety_preference,
        )

        # Re-evaluate fastest route under historical risk.
        fastest_historical_summary = self.summarize_route(
            fastest_route,
            weight_choice="fast_cost",
        )

        historical_route = nx.shortest_path(
            self.G,
            start_node,
            end_node,
            weight="safe_cost",
        )
        historical_summary = self.summarize_route(
            historical_route,
            weight_choice="safe_cost",
        )

        # --------------------------------------------------------------
        # 3. ML-risk comparison
        # --------------------------------------------------------------
        ml_lookup = self._predict_dynamic_ml_risk(
            month=month,
            hour=hour,
            day_of_week=day_of_week,
        )
        self._attach_costs_from_lookup(
            risk_lookup=ml_lookup,
            safety_preference=safety_preference,
        )

        # Re-evaluate fastest route under ML risk.
        fastest_ml_summary = self.summarize_route(
            fastest_route,
            weight_choice="fast_cost",
        )

        ml_route = nx.shortest_path(
            self.G,
            start_node,
            end_node,
            weight="safe_cost",
        )
        ml_summary = self.summarize_route(
            ml_route,
            weight_choice="safe_cost",
        )

        # For backward compatibility with app.py and visualization.py:
        # fastest_summary now means "fastest route evaluated under ML risk".
        # More detailed summaries are also returned below.
        return {
            "fastest_route": fastest_route,
            "historical_route": historical_route,
            "ml_route": ml_route,

            "fastest_summary": fastest_ml_summary.as_dict(),
            "fastest_distance_summary": fastest_distance_summary.as_dict(),
            "fastest_historical_summary": fastest_historical_summary.as_dict(),
            "fastest_ml_summary": fastest_ml_summary.as_dict(),

            "historical_summary": historical_summary.as_dict(),
            "ml_summary": ml_summary.as_dict(),

            "historical_risk_reduction_pct": self._safe_percent_reduction(
                fastest_historical_summary.length_weighted_risk,
                historical_summary.length_weighted_risk,
            ),
            "ml_risk_reduction_pct": self._safe_percent_reduction(
                fastest_ml_summary.length_weighted_risk,
                ml_summary.length_weighted_risk,
            ),

            "start_coords": start_coords,
            "destination_coords": end_coords,
            "recommendation_text": self.explain(
                fastest_distance_summary=fastest_distance_summary,
                fastest_historical_summary=fastest_historical_summary,
                historical_summary=historical_summary,
                fastest_ml_summary=fastest_ml_summary,
                ml_summary=ml_summary,
            ),
        }

    @classmethod
    def explain(
        cls,
        fastest_distance_summary: RouteSummary,
        fastest_historical_summary: RouteSummary,
        historical_summary: RouteSummary,
        fastest_ml_summary: RouteSummary,
        ml_summary: RouteSummary,
    ) -> str:
        """Create clear recommendation text with correct risk baselines."""
        historical_reduction = cls._safe_percent_reduction(
            fastest_historical_summary.length_weighted_risk,
            historical_summary.length_weighted_risk,
        )
        ml_reduction = cls._safe_percent_reduction(
            fastest_ml_summary.length_weighted_risk,
            ml_summary.length_weighted_risk,
        )

        historical_detour = (
            historical_summary.distance_km - fastest_distance_summary.distance_km
        )
        ml_detour = (
            ml_summary.distance_km - fastest_distance_summary.distance_km
        )

        lines = []
        lines.append("Route comparison:")
        lines.append(
            f"- Fastest route: {fastest_distance_summary.distance_km:.2f} km"
        )
        lines.append(
            f"  Historical-risk evaluation: {fastest_historical_summary.length_weighted_risk:.4f}"
        )
        lines.append(
            f"  ML-risk evaluation: {fastest_ml_summary.length_weighted_risk:.4f}"
        )
        lines.append("")
        lines.append(
            f"- Historical-risk route: {historical_summary.distance_km:.2f} km, "
            f"historical risk {historical_summary.length_weighted_risk:.4f}"
        )
        lines.append(
            f"  Detour vs fastest: {historical_detour:.2f} km; "
            f"historical risk reduction: {historical_reduction:.1f}%"
        )
        lines.append("")
        lines.append(
            f"- ML-safest route: {ml_summary.distance_km:.2f} km, "
            f"ML risk {ml_summary.length_weighted_risk:.4f}"
        )
        lines.append(
            f"  Detour vs fastest: {ml_detour:.2f} km; "
            f"ML risk reduction: {ml_reduction:.1f}%"
        )
        lines.append("")

        if ml_reduction > 5 and ml_detour <= 1.5:
            lines.append(
                "Recommendation: choose the ML-safest route. "
                "It has lower predicted accident risk with a reasonable detour."
            )
        elif historical_reduction > 5 and historical_detour <= 1.5:
            lines.append(
                "Recommendation: the historical-risk route is a reasonable safer alternative. "
                "It reduces historical spatial risk with a moderate detour."
            )
        else:
            lines.append(
                "Recommendation: in this example, the fastest route is also reasonable "
                "because the safer alternatives do not provide a strong enough risk reduction "
                "for the additional distance."
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Map visualization
    # ------------------------------------------------------------------

    def route_to_coordinates(self, route: Route) -> List[Tuple[float, float]]:
        """Convert route node list to Folium coordinates."""
        coords = []
        for node in route:
            d = self.G.nodes[node]
            coords.append((float(d["y"]), float(d["x"])))
        return coords

    def make_map(
        self,
        result: Dict[str, Any],
        show_historical: bool = True,
    ) -> folium.Map:
        """Create Folium map with fastest, historical-risk, and ML-safest routes."""
        start = result["start_coords"]
        dest = result["destination_coords"]

        center_lat = (start[0] + dest[0]) / 2.0
        center_lon = (start[1] + dest[1]) / 2.0

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13,
            tiles="OpenStreetMap",
        )

        folium.PolyLine(
            self.route_to_coordinates(result["fastest_route"]),
            color="red",
            weight=5,
            opacity=0.75,
            tooltip="Fastest route",
        ).add_to(m)

        if show_historical:
            folium.PolyLine(
                self.route_to_coordinates(result["historical_route"]),
                color="orange",
                weight=5,
                opacity=0.75,
                tooltip="Historical GIS-risk route",
            ).add_to(m)

        folium.PolyLine(
            self.route_to_coordinates(result["ml_route"]),
            color="green",
            weight=5,
            opacity=0.85,
            tooltip="ML-safest route",
        ).add_to(m)

        folium.Marker(
            start,
            tooltip="Start",
            popup="Start",
            icon=folium.Icon(color="blue", icon="play"),
        ).add_to(m)

        folium.Marker(
            dest,
            tooltip="Destination",
            popup="Destination",
            icon=folium.Icon(color="black", icon="flag"),
        ).add_to(m)

        legend = """
        <div style="
            position: fixed;
            bottom: 40px;
            left: 40px;
            z-index: 9999;
            background: white;
            padding: 12px;
            border: 2px solid grey;
            border-radius: 6px;
            font-size: 14px;
        ">
            <b>Route Legend</b><br>
            <span style="color:red;">■</span> Fastest route<br>
            <span style="color:orange;">■</span> Historical-risk route<br>
            <span style="color:green;">■</span> ML-safest route
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend))
        return m

    def compare_and_map(
        self,
        start_address: str,
        destination_address: str,
        month: int = 7,
        hour: int = 8,
        day_of_week: int = 3,
        safety_preference: int = 7,
    ) -> Tuple[Dict[str, Any], folium.Map]:
        """Calculate route comparison and return result dictionary plus map."""
        result = self.compare_routes(
            start_address=start_address,
            destination_address=destination_address,
            month=month,
            hour=hour,
            day_of_week=day_of_week,
            safety_preference=safety_preference,
        )
        m = self.make_map(result)
        return result, m
