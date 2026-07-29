# 2W1C Integrated Project  
## ML-Based Bicycle Safety Routing for Berlin

**2W1C** is an end-to-end data science project for bicycle safety routing in Berlin.

The project combines:

1. **Unfallatlas bicycle accident data**
2. **OpenStreetMap / OSMnx bicycle road network**
3. **Person B's stronger GIS risk analysis**
   - metric CRS snapping
   - severity-weighted edge risk
   - empirical-Bayes shrinkage
   - intersection risk
   - shortest-vs-risk-aware routing
4. **Machine-learning accident-risk prediction**
   - positive samples from observed accidents
   - pseudo-negative samples from OSM road-segment/time combinations
5. **Streamlit app**
   - fastest route vs historical-risk route vs ML-safest route

---

## Why this integrated version?

The original end-to-end project had a complete app and ML pipeline, but the spatial-risk part was relatively simple.

Person B's notebook had a stronger OSM/GIS analysis:
- projected Berlin graph to `EPSG:25833`
- snapped Unfallatlas accidents to nearest OSM edges within a distance threshold
- built severity-weighted per-edge risk
- added intersection risk
- demonstrated shortest vs safest route comparison

This integrated project keeps the end-to-end project structure and replaces the weak GIS part with a stronger spatial-risk baseline.

---

## Final Workflow

```text
Unfallatlas bicycle accidents
+
OpenStreetMap bicycle network
↓
Clean accident data
↓
Project OSM graph to metric CRS
↓
Snap accident points to nearest OSM edges
↓
Build historical GIS risk baseline
↓
Create positive and pseudo-negative ML samples
↓
Train ML accident-risk model
↓
Predict risk for road segments
↓
Compare:
    1. fastest route
    2. historical-risk safest route
    3. ML-safest route
↓
Show result in Streamlit
```

---

## Project Structure

```text
capstone/
├── app.py
├── run_pipeline.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── config.py
│   ├── data_pipeline.py
│   ├── osm_network.py
│   ├── spatial_risk.py
│   ├── model_training.py
│   ├── route_engine.py
│   └── visualization.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
└── outputs/
    ├── maps/
    └── figures/
```

---

## Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Mac / Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data Input

### Option A: Use your already-cleaned EDA file

If you already have `df_berlin_rad_final` from your EDA notebook:

```python
df_berlin_rad_final.to_csv(
    "data/processed/berlin_bike_accidents_clean.csv",
    index=False
)
```

Then run:

```bash
python run_pipeline.py --use-clean-data
```

### Option B: Use raw Unfallatlas CSV

Put raw Unfallatlas CSV files into:

```text
data/raw/
```

Then run:

```bash
python run_pipeline.py --accident-file data/raw/your_unfallatlas_file.csv
```

---

## Run Full Pipeline

```bash
python run_pipeline.py --use-clean-data
```

This will:

1. Load clean Berlin bicycle accident data
2. Download or load Berlin OSM bicycle network
3. Project the graph to `EPSG:25833`
4. Snap accident points to road edges
5. Build historical GIS risk baseline
6. Train ML risk model
7. Predict edge-level ML risk
8. Generate demo route comparison map

---

## Run Streamlit App

```bash
streamlit run app.py
```

The app supports:

- Start address
- Destination address
- Month
- Hour
- Weekday
- Safety preference
- Route mode:
  - Fastest route
  - Historical-risk route
  - ML-safest route

---

## Positive and Negative Data

Unfallatlas only gives **positive samples**:

```text
A bicycle accident happened here at this time.
```

The **negative samples** are constructed from OpenStreetMap:

```text
A sampled bicycle road segment at a sampled time where no Unfallatlas accident was observed.
```

These are more accurately called **pseudo-negative samples**, because they mean "no observed accident", not "guaranteed safe".

---

## Main AI Task

```text
road segment + time + road features + historical GIS risk
→ predicted accident risk
```

The historical GIS risk baseline is not discarded. It is used in two ways:

1. As a strong baseline model
2. As an input feature for the ML model

---

## Model Evaluation

The pipeline uses a time-based split:

```text
Train: all years before the last available year
Test: last available year
```

Metrics:

- ROC-AUC
- PR-AUC
- Top-10% recall
- Top-20% recall

Top-k recall is important because for routing we care whether the model can identify the highest-risk road segments.

---

## Limitations

- The model predicts **relative risk**, not absolute individual accident probability.
- Bicycle traffic volume is not included.
- Pseudo-negative samples are approximations.
- Weather and real-time traffic are not included in the first version.
- This is a research/education prototype, not an official safety tool.

---

## Suggested Final Presentation Message

> Our baseline is not just a heatmap. We use a metric CRS to snap accidents to OSM road segments, build a severity-weighted GIS risk score, and include intersection risk. Then we use this historical spatial risk as a baseline and as an ML feature. The final model predicts segment-level bicycle accident risk and uses that risk to compare the fastest route with safer alternatives.
