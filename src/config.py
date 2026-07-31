from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MAP_DIR = OUTPUT_DIR / "maps"
FIGURE_DIR = OUTPUT_DIR / "figures"

for path in [RAW_DIR, PROCESSED_DIR, MODEL_DIR, MAP_DIR, FIGURE_DIR]:
    path.mkdir(parents=True, exist_ok=True)

BERLIN_PLACE = "Berlin, Germany"
BERLIN_CRS = "EPSG:25833"

CLEAN_ACCIDENT_FILE = PROCESSED_DIR / "berlin_bike_2018_2025.csv"

OSM_GRAPH_FILE = PROCESSED_DIR / "berlin_bike_network.graphml"
OSM_PROJECTED_GRAPH_FILE = PROCESSED_DIR / "berlin_bike_network_projected.graphml"
OSM_EDGE_FEATURES_FILE = PROCESSED_DIR / "berlin_osm_edge_features.csv"

SNAPPED_ACCIDENTS_FILE = PROCESSED_DIR / "berlin_accidents_snapped_to_edges.csv"
SPATIAL_RISK_FILE = PROCESSED_DIR / "berlin_spatial_edge_risk.csv"
EDGE_RISK_FILE = PROCESSED_DIR / "berlin_edge_risk_scores.csv"
ML_DATASET_FILE = PROCESSED_DIR / "berlin_ml_risk_dataset.csv"

MODEL_FILE = MODEL_DIR / "risk_model.joblib"
METRICS_FILE = MODEL_DIR / "model_metrics.json"

DEMO_ROUTE_MAP = MAP_DIR / "demo_route_comparison.html"
HISTORICAL_RISK_MAP = MAP_DIR / "berlin_historical_risk_streets.html"
