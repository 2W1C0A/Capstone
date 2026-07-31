# How the uploaded notebooks were integrated

## 02_OSM_Berlin_Street_Risk_Analysis(1).ipynb

Integrated into:

- `src/spatial_risk.py`
- `src/visualization.py`
- `src/route_engine.py`

Key ideas:

- OSM bike graph
- EPSG:25833 projection
- snap accident points to OSM edges within 25 m
- severity-weighted historical risk
- empirical-Bayes shrinkage
- street-level risk map
- historical GIS-risk route

## 03_model_leakage_fix.ipynb

Integrated into:

- `src/model_training.py`
- `app.py`

Key ideas:

- remove accident-derived leakage features
- restrict negatives to rideable network classes
- report time_only, road_only, deployable_road_time and leaky diagnostic models
- deployed model = road_only
- hour slider does not currently drive route changes

## 04_severity_model.ipynb

Integrated into:

- `src/severity_model.py`
- `src/spatial_risk.py`
- `app.py`

Key ideas:

- severity model estimates KSI conditional on a crash
- route-planning features weakly predict severity
- severity evidence is useful for interpretation, not a strong route-changing model
- near_junction / node_degree are created upstream

## 05_national_robustness.ipynb

Integrated into:

- `src/national_robustness.py`
- `README.md`

Key idea:

- optional read-only evidence module for truck severity robustness
- not part of default runtime
