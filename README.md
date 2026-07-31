# 2W1C Defensible Project

**2W1C** is a bicycle safety-routing prototype for Berlin.  
This version integrates the improved GIS risk baseline notebook with a stricter, leakage-safe ML workflow and a separate severity-evidence module.

## Main change from the earlier integrated project

The earlier version mixed historical accident-derived columns into the ML feature set.  
This version separates the logic:

1. **Historical GIS-risk baseline**  
   Uses Unfallatlas bicycle crashes snapped to OpenStreetMap bicycle-network segments.  
   It computes severity-weighted historical spatial risk and junction risk.

2. **Leakage-safe ML occurrence model**  
   Uses only road attributes for the deployed model:
   - edge length
   - cycleway flag
   - speed limit
   - speed-limit missing flag
   - highway class

   It excludes accident-derived columns:
   - `accident_count`
   - `severity_sum`
   - `serious_fatal_count`
   - `historical_risk_norm`
   - `node_risk_raw`
   - `combined_spatial_risk`

3. **Severity evidence module**  
   Models KSI severity conditional on a crash occurring.  
   This supports interpretation and limitations, but it is not presented as a strong route-changing model.

4. **National robustness check**  
   Optional, read-only analysis for presentation evidence. It is not part of the default app runtime.

## Why this is more defensible

The project now reports diagnostic rows, including the leaky model clearly labelled as invalid.  
This makes the story honest:

> We found target leakage, diagnosed it, removed it, and rebuilt a leakage-safe model.

## Project structure

```text
2w1c_defensible_project/
├── app.py
├── run_pipeline.py
├── requirements.txt
├── pyproject.toml
├── src/
│   ├── config.py
│   ├── data_pipeline.py
│   ├── osm_network.py
│   ├── spatial_risk.py
│   ├── model_training.py
│   ├── severity_model.py
│   ├── national_robustness.py
│   ├── route_engine.py
│   └── visualization.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── outputs/
│   ├── maps/
│   └── figures/
├── reports/
├── notebooks/
└── docs/
```

## Data

Place raw Unfallatlas files in:

```text
data/raw/
```

or provide an already-cleaned file:

```text
data/processed/berlin_bike_2018_2025.csv
```

The pipeline automatically uses the processed file if it exists.

## Run

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python run_pipeline.py --demo-route
streamlit run app.py
```

## Important outputs

```text
data/processed/berlin_accidents_snapped_to_edges.csv
data/processed/berlin_osm_edge_features.csv
data/processed/berlin_route_risk_edges.csv
data/processed/berlin_ml_risk_dataset.csv

models/occurrence_model_comparison.csv
models/occurrence_model_metrics.json
models/leakage_diagnostics.json
models/severity_model_metrics.json
models/historical_risk_temporal_validation.json

outputs/maps/berlin_risk_streets.html
outputs/maps/demo_route_comparison.html
```

## App tabs

1. **Route comparison** — fastest vs historical GIS-risk vs ML road-risk route  
2. **Leakage-safe ML** — diagnostic model comparison, including leaky model labelled invalid  
3. **Historical GIS risk** — spatial risk table and forward validation  
4. **Severity evidence** — KSI severity analysis  
5. **Method & limitations** — exposure limitation, hour-slider limitation, and interpretation

## Honest claims to use in the presentation

Use:

> The historical GIS route reduces historical spatial-risk exposure according to our model.

Use:

> The ML road-risk route is trained without accident-derived leakage features.

Use:

> Severity findings are conditional on a crash occurring; they do not estimate crash probability.

Do **not** use:

> This route is objectively the safest route.

Do **not** use:

> The hour slider dynamically changes accident risk in the current model.

Do **not** use:

> Truck involvement is a route feature.

A cyclist cannot know in advance whether a future crash would involve a truck. Truck involvement is therefore a severity evidence finding, not a direct route-planning feature.

## Limitations

- Unfallatlas has crashes but not cycling exposure counts.
- Therefore route risk is a relative score, not a personal crash probability.
- Negative sampling is restricted to rideable classes, but still cannot fully correct exposure bias.
- Severity is weakly predictable from route-planning features.
- The hour slider is retained for transparency and future work, but does not currently drive visible route changes.
