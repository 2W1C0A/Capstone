from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MAP_DIR = OUTPUT_DIR / "maps"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = PROJECT_ROOT / "reports"

# Standard processed files
CLEAN_ACCIDENT_FILE = PROCESSED_DIR / "berlin_bike_2018_2025.csv"
SNAPPED_ACCIDENTS_FILE = PROCESSED_DIR / "berlin_accidents_snapped_to_edges.csv"
OSM_EDGE_FEATURES_FILE = PROCESSED_DIR / "berlin_osm_edge_features.csv"
EDGE_RISK_FILE = PROCESSED_DIR / "berlin_edge_risk.csv"
NODE_RISK_FILE = PROCESSED_DIR / "berlin_node_risk.csv"
ROUTE_RISK_FILE = PROCESSED_DIR / "berlin_route_risk_edges.csv"
ML_DATASET_FILE = PROCESSED_DIR / "berlin_ml_risk_dataset.csv"

# OSM graph files
OSM_GRAPH_FILE = DATA_DIR / "berlin_bike.graphml"
OSM_PROJECTED_GRAPH_FILE = DATA_DIR / "berlin_bike_projected.graphml"

# Model and report files
OCCURRENCE_MODEL_FILE = MODEL_DIR / "occurrence_road_only_model.joblib"
OCCURRENCE_METRICS_FILE = MODEL_DIR / "occurrence_model_metrics.json"
OCCURRENCE_COMPARISON_FILE = MODEL_DIR / "occurrence_model_comparison.csv"
LEAKAGE_DIAGNOSTICS_FILE = MODEL_DIR / "leakage_diagnostics.json"

SEVERITY_MODEL_FILE = MODEL_DIR / "severity_logistic_model.joblib"
SEVERITY_METRICS_FILE = MODEL_DIR / "severity_model_metrics.json"
SEVERITY_BY_HOUR_FILE = PROCESSED_DIR / "severity_by_hour.csv"
SEVERITY_BY_HIGHWAY_FILE = PROCESSED_DIR / "severity_by_highway.csv"

TEMPORAL_VALIDATION_FILE = MODEL_DIR / "historical_risk_temporal_validation.json"

# Maps
RISK_STREETS_MAP_FILE = MAP_DIR / "berlin_risk_streets.html"
DEMO_ROUTE_MAP_FILE = MAP_DIR / "demo_route_comparison.html"


# ---------------------------------------------------------------------
# Berlin / OSM constants
# ---------------------------------------------------------------------

BERLIN_PLACE = "Berlin, Germany"
BERLIN_CRS = "EPSG:25833"  # UTM zone 33N; metric CRS suitable for Berlin

SNAP_MAX_DIST_M = 25.0
NODE_RADIUS_M = 20.0

# Rideable classes used for leakage-safe negative sampling.
# Excluding service/path/track avoids learning OSM network composition instead of risk.
RIDEABLE_CLASSES = {
    "residential",
    "cycleway",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "living_street",
}

LOW_EXPOSURE_CLASSES = {"service", "path", "track", "bridleway", "footway"}


# Accident-derived columns that must not be used by the deployable ML model.
LEAKY_FEATURES = {
    "accident_count",
    "severity_sum",
    "serious_fatal_count",
    "historical_risk_norm",
    "node_risk_raw",
    "combined_spatial_risk",
}


def ensure_dirs() -> None:
    """Create project output directories explicitly.

    This avoids making directories as an import side effect.
    """
    for path in [
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        MODEL_DIR,
        OUTPUT_DIR,
        MAP_DIR,
        FIGURE_DIR,
        REPORT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
