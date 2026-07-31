from __future__ import annotations

import argparse

from src.config import (
    CLEAN_ACCIDENT_FILE,
    DEMO_ROUTE_MAP_FILE,
    ensure_dirs,
)
from src.data_pipeline import prepare_berlin_bicycle_accidents
from src.model_training import build_ml_dataset, train_occurrence_models
from src.osm_network import build_edge_features, load_or_download_graph
from src.route_engine import RouteEngine
from src.severity_model import build_severity_dataset, export_severity_tables, train_severity_models
from src.spatial_risk import build_spatial_risk_pipeline, temporal_validation
from src.visualization import make_risk_street_map


def main():
    parser = argparse.ArgumentParser(description="Run the 2W1C defensible pipeline.")
    parser.add_argument("--raw-file", default=None, help="Optional single raw Unfallatlas file.")
    parser.add_argument(
        "--use-existing-clean",
        action="store_true",
        help="Use data/processed/berlin_bike_2018_2025.csv if it exists.",
    )
    parser.add_argument(
        "--skip-osm-download",
        action="store_true",
        help="Require existing OSM graph files instead of downloading.",
    )
    parser.add_argument(
        "--skip-temporal-validation",
        action="store_true",
        help="Skip forward validation of historical GIS risk.",
    )
    parser.add_argument(
        "--demo-route",
        action="store_true",
        help="Generate a demo route map after training.",
    )
    args = parser.parse_args()

    ensure_dirs()

    use_existing = args.use_existing_clean or CLEAN_ACCIDENT_FILE.exists()
    print("Step 1/8 — Prepare Berlin bicycle accident data")
    accidents = prepare_berlin_bicycle_accidents(
        raw_file=args.raw_file,
        use_existing_clean=use_existing,
    )

    print("\nStep 2/8 — Load/download OSM bicycle graph")
    G, Gp = load_or_download_graph()

    print("\nStep 3/8 — Build OSM edge features")
    edge_features = build_edge_features(Gp)

    print("\nStep 4/8 — Build improved historical GIS risk baseline")
    snapped, node_risk, route_risk = build_spatial_risk_pipeline(
        Gp,
        accidents,
        edge_features=edge_features,
    )

    print("\nStep 5/8 — Forward temporal validation of historical risk")
    if not args.skip_temporal_validation:
        temporal_validation(Gp, accidents, edge_features)
    else:
        print("skipped")

    print("\nStep 6/8 — Build leakage-safe ML dataset and train occurrence models")
    ml_data = build_ml_dataset(snapped, route_risk, restrict_to_rideable=True)
    train_occurrence_models(ml_data)

    print("\nStep 7/8 — Train/evaluate severity evidence model")
    severity_data = build_severity_dataset(snapped, edge_features)
    train_severity_models(severity_data)
    export_severity_tables(severity_data)

    print("\nStep 8/8 — Generate historical risk street map")
    make_risk_street_map(Gp, route_risk)

    if args.demo_route:
        print("\nGenerating demo route map")
        engine = RouteEngine()
        result, m = engine.compare_and_map(
            start_address="Alexanderplatz, Berlin, Germany",
            destination_address="Brandenburg Gate, Berlin, Germany",
            safety_preference=7,
            hour=8,
        )
        DEMO_ROUTE_MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
        m.save(DEMO_ROUTE_MAP_FILE)
        print(result["recommendation_text"])
        print(f"saved {DEMO_ROUTE_MAP_FILE}")

    print("\nDone. Start the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
