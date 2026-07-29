from __future__ import annotations

import argparse

from src.config import (
    CLEAN_ACCIDENT_FILE,
    DEMO_ROUTE_MAP,
)
from src.data_pipeline import prepare_clean_accidents
from src.osm_network import load_or_download_graph, build_edge_features
from src.spatial_risk import build_spatial_risk_pipeline
from src.model_training import (
    build_ml_dataset,
    train_risk_model,
    make_edge_prediction_table,
)
from src.route_engine import RouteEngine


def parse_args():
    parser = argparse.ArgumentParser(description="Run 2W1C integrated end-to-end pipeline.")
    parser.add_argument("--accident-file", type=str, default=None, help="Path to raw Unfallatlas CSV file.")
    parser.add_argument("--use-clean-data", action="store_true", help="Use data/processed/berlin_bike_accidents_clean.csv.")
    parser.add_argument("--force-download-osm", action="store_true", help="Force download Berlin OSM graph.")
    parser.add_argument("--neg-ratio", type=int, default=3, help="Pseudo-negative samples per positive accident.")
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument("--hour", type=int, default=8)
    parser.add_argument("--day-of-week", type=int, default=3, help="1=Sunday, 2=Monday, ..., 7=Saturday in Unfallatlas convention.")
    parser.add_argument("--start", type=str, default="Alexanderplatz, Berlin, Germany")
    parser.add_argument("--destination", type=str, default="Brandenburg Gate, Berlin, Germany")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n[1/7] Loading and cleaning accident data")
    accidents = prepare_clean_accidents(
        accident_file=args.accident_file,
        use_clean_data=args.use_clean_data,
    )

    print("\n[2/7] Loading/downloading OSM bicycle network")
    G, G_proj = load_or_download_graph(force_download=args.force_download_osm)
    edge_features = build_edge_features(G_proj)

    print("\n[3/7] Building spatial GIS risk")
    snapped, edge_risk = build_spatial_risk_pipeline(G_proj, accidents)

    print("\n[4/7] Building ML dataset with positive and pseudo-negative samples")
    ml_data = build_ml_dataset(
        accidents=accidents,
        snapped=snapped,
        edge_risk=edge_risk,
        neg_ratio=args.neg_ratio,
    )

    print("\n[5/7] Training ML accident-risk model")
    model, metrics = train_risk_model(ml_data)

    print("\n[6/7] Predicting edge-level ML risk")
    make_edge_prediction_table(
        edge_risk=edge_risk,
        model=model,
        month=args.month,
        hour=args.hour,
        day_of_week=args.day_of_week,
    )

    print("\n[7/7] Creating demo route comparison")
    engine = RouteEngine()
    result, route_map = engine.compare_and_map(
        start_address=args.start,
        destination_address=args.destination,
        month=args.month,
        hour=args.hour,
        day_of_week=args.day_of_week,
        safety_preference=7,
    )
    route_map.save(DEMO_ROUTE_MAP)

    print(result["recommendation_text"])
    print(f"\nSaved demo map: {DEMO_ROUTE_MAP}")
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
