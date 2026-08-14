# 2W1C — Two Wheels, One City

A bicycle safety-routing prototype for Berlin, built on 37,948 police-recorded
bicycle crashes (2018–2025) snapped to a 441k-edge OpenStreetMap bicycle network.

It compares three routes: the shortest, one that avoids segments with recorded
crash history, and one driven by a road-attribute occurrence model. It also marks
junctions with recorded serious injuries and deaths along whichever route you pick.

**What it is not.** There is no cyclist-volume denominator anywhere in this
project. Crash counts and model scores are per metre of road, not per
cyclist-kilometre, so a quiet street scores low partly because few people ride
it. This is decision support, not a safety guarantee, and no output here is a
personal crash probability.

---

## Run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_pipeline.py --use-existing-clean
streamlit run app.py
```

The pipeline step is required before the app: it fits the frequency model and
writes the per-edge risk columns the router reads. Without it the app falls back
to scoring the whole network on the first request, which takes minutes.

First graph load parses a 441k-edge GraphML (~60 s) and caches it as a pickle
beside the file; later launches take a few seconds. The pickle is disposable —
delete it to force a rebuild.

Raw Unfallatlas files go in `data/raw/`. If
`data/processed/berlin_bike_2018_2025.csv` already exists, `--use-existing-clean`
uses it; drop the flag to rebuild from raw.

---

## What the app shows

| Route | Cost function | What it claims |
|---|---|---|
| **Fastest** | distance only | nothing to defend |
| **Historical GIS-risk** | severity-weighted recorded crashes per segment, empirical-Bayes shrunk | a claim about the past, not about your trip |
| **ML road-risk** | expected crashes per metre from the negative-binomial SPF, plus a low-exposure class constraint | a structural estimate; see the limits below |
| **Crash markers** | none | recorded killed-or-seriously-injured crashes within 25 m, with year and whether a lorry was involved |

The "ML road-risk" label is the UI name. The model behind it is the
negative-binomial Safety Performance Function in `src/frequency_model.py`, not
the occurrence classifier that earlier versions deployed — see *What we
rejected*.

The crash markers are the only layer that needs no denominator and no model.
They are a record of what happened, which also makes them the one honest way to
compare routes: "this one passes six recorded serious injuries, that one passes
fourteen" is a single scale with no model assumption in it.

---

## What we found

**Lorries are 2.0% of crashes and 43.2% of cyclist deaths.** 767 of 37,948
crashes, 32 of 74 deaths. KSI rate 27.25% [24.22–30.51] against 10.6% for
car-only crashes. The mechanism is specific: turning conflicts are 52.2% of
lorry crashes against 37.9% for cars, and 23 of the 74 cyclists killed were hit
by a turning lorry. Replicated on 575,348 national crashes, where truck KSI is
28.78% [27.69–29.89].

**Road class matters, and junctions matter separately.** Crashes per metre
relative to `cycleway`, from the deployed NB SPF on 236,981 segments:

| Class | Rate ratio | 95% CI |
|---|---|---|
| `primary` | **3.96** | 2.40–6.52 |
| `secondary` | 2.28 | 1.39–3.75 |
| `tertiary` | 2.00 | 1.22–3.28 |
| `cycleway` | 1.00 | reference |
| `residential` | 0.64 | 0.39–1.05 |
| `living_street` | 0.33 | 0.20–0.56 |
| `pedestrian` | 0.26 | 0.14–0.47 |
| `service` | 0.05 | 0.03–0.09 |
| `path` | 0.009 | 0.005–0.015 |

Each junction endpoint multiplies crashes per metre by **2.77** [2.466–3.113],
holding road class constant — and adding it moves `primary` from 4.00 to 3.96, a
change of 1%. Junction structure and road class are separate mechanisms. A 50
km/h road carries 1.007 per km/h more than a 30 km/h road, so about 15% more.

**The risk surface predicts crashes it has not seen.** Fitted on 2018–2023,
tested on 2024–2025: the top decile of segments covers 21.6% of network length
and contains **55.0%** of the test-period crashes. Against a length-proportional
null model, that is **2.60×**.

**The street-level ranking splits two ways, and both are useful.** Weighted by
BASt injury costs (1 : 23 : 222 for slight, serious, fatal), the worst absolute
totals are the arterials — Landsberger Allee at 555 over 24.8 km, Kantstraße 377
over 5.4 km. Per kilometre the order changes completely: Delbrückstraße is 1.9 km
with three recorded crashes and ranks first at 228.6, because one of them was
fatal. Schönhauser Allee has 67 crashes and scores 45.8. The street-level version
of the same point: the place with the most crashes is not the most dangerous
place.

**Counts are not risk.** Cyclist traffic swings 35.7× across the day (03:00
against 18:00) while exposure-normalised expected harm varies only 1.27×. Night
looks safe in raw counts because almost nobody is riding.

**The bottom of the rate table is exposure, not safety.** Unlit edges are 7.8%
of network length and 0.5% of crashes — a fifteen-fold under-representation.
`path` at 0.009 is not fifty times safer than a cycleway; almost nobody rides
there. This is measured, and it is why the router applies a class constraint on
top of the model rather than trusting the model's own ordering at the low end.

---

## What we rejected

This section is the point of the project. Several things scored better and are
not deployed.

**Target leakage, ROC-AUC 0.976.** `accident_count` alone scores 0.976 on the
occurrence task — it is the label in disguise. Removing every accident-derived
column dropped the headline from 0.973 to a lift of 1.77. Two further sampling
artefacts came out of the same diagnosis: sampling negatives uniformly made 48%
of them `service` or `path` against 2% of positives, so "is it a service road"
became a near-perfect negative indicator; and drawing negative timestamps from
the positive pool made time features carry zero signal by construction.
`03_model_leakage_fix`.

**A length covariate, AIC 3,852 better.** Adding `log_len` as a free covariate
alongside the offset estimates a length exponent of 0.485 rather than the 1.0
the offset imposes, and reaches AIC 154,411 against the deployed 158,263. It is
rejected because below an exponent of 1 the cost is no longer additive along a
route: one 1000 m segment would cost 28.5 while ten 100 m segments covering the
same ground cost 93.3 — the same road priced 3.3× differently depending on how
finely OSM happened to split it. The 0.485 exponent is still a finding: crashes
scale roughly with the square root of length, concentrating at endpoints rather
than along the line, which is the same conclusion `junction_ends` reaches
independently. `09_frequency_model` §4.2.

**A junction-density term, AIC 1,279 better.** Junctions per 100 m is largely
`1/length` in disguise; it makes expected crashes non-monotonic in segment
length, so a 1 m junction stub scores worse than a 20 m one, and 42% of segments
are under 20 m.

**Severity as a route driver.** The severity model reaches lift 1.18 against a
base rate of 11.85%, and its Brier score is 0.1046 against the baseline's
0.1047 — identical to four decimals. Logistic regression beats both tree
ensembles, which is what you see when there is no structure to find rather than
when the classes are skewed. So severity is treated as approximately constant
and the cost function collapses to the frequency term. What determines how badly
a crash ends — impact angle, vehicle mass, whether the head hit the kerb — is
not knowable at route-planning time. `04_severity_model`.

**H2, surface hazards.** Rough surfaces carry a *lower* KSI rate than smooth
(10.61% [9.18–12.24] against 13.15% [12.79–13.51]), and solo crash shares are
identical across surface types. But that comparison cannot distinguish "surface
does not cause falls" from "falls caused by surface are not in this dataset" —
roughly 98% of crashes with no external vehicle never reach police records. H2
is untestable with this source, not refuted. `06_exposure_normalisation` §6.1.

**Class weighting and resampling.** The routing engine consumes probabilities
directly as a cost, so calibration matters more than class balance;
`class_weight` and SMOTE-style resampling both distort exactly the quantity the
cost function needs. The frequency model handles its 91.4% zeros through the
distribution instead — variance/mean is 3.48, so a negative binomial is required
rather than preferred.

---

## Data limitations

**The Unfallatlas is a case-only record.** No journey that ended safely appears
in it, so nothing here supports a statement about the probability of a crash.
Every claim is a count, or a proportion conditional on a crash having occurred.

**Under-reporting is large and not random.** A German matched-records study
(Juhra et al., UKM / Polizei Münster / UDV, 2009–2010) found 2,250 cyclist
casualties reaching clinics over twelve months against 723 in police records —
roughly 32%. Of 251 patients admitted as inpatients only 64 (25.5%) were
police-recorded, and among crashes with no external vehicle involved, 98% never
entered the statistics.

The selection is therefore on **motor-vehicle involvement**, not injury
severity: police attend when liability and insurance are at stake. Three
consequences carry through this project:

1. The truck comparison is between two well-captured groups, which is one reason
   it is the most robust result here.
2. Any crash-count surface maps where bicycle–motor-vehicle conflicts are
   *recorded*, not where cyclists are injured.
3. Single-bicycle crashes are largely absent by construction, so hypotheses
   about surface quality and lighting cannot be tested with this data.

**Exposure is unavailable.** Berlin's 35 permanent counting stations give a real
denominator at 22 usable locations against 238,951 network segments — enough to
build a citywide temporal index, not a spatial one.

**A structural quirk in the segment key.** `pair_id` keys on `sorted(u, v)`,
which merges the two directions of one street but also merges genuinely parallel
ways between the same two nodes. 1,970 pairs (0.82%) are affected and excluded
from the fit, at a cost of 182 crashes (0.5%). Excluding them shifts every
published rate ratio by about 20% because they are all measured against
`cycleway`, whose own estimate moves with them — but the absolute predicted
rates, which is what the router consumes, change by 1–3%.

---

## Honest claims to use

Use:

> The historical route reduces exposure to segments with recorded crash history,
> on our own measure.

> The road-class gradient survives controlling for motor traffic volume: with DTV
> in the model, `primary` moves from 5.02 to 4.59 on the 37% of segments where
> volume data exists — which carry 83.8% of crashes.

> Severity findings are conditional on a crash occurring. They measure how badly
> crashes end, not how likely one is.

> The top decile of our risk surface contains 55% of the crashes in the two years
> it never saw, against 21% for a length-proportional baseline.

Do **not** use:

> This is the safest route.

> The hour slider changes the route.

> Truck involvement is a route feature.

A cyclist cannot know in advance which vehicle a future crash would involve.
Truck involvement is severity evidence, not a routing input.

And do not compare the two risk reductions the app reports side by side. The
historical figure is measured on the historical scale and the frequency figure
on its own scale; the percentages are not commensurable. Compare the crash
markers instead — those are one scale.

---

## A note on the memory baseline

The risk surface does **not** beat a pure crash-memory baseline: 0.5495 against
0.5677 on top-decile recall. That is worth stating plainly, along with why it is
the wrong yardstick.

43.2% of 2024–2025 crashes are on segments with no prior recorded crash, so a
memory model scores every one of them as zero. Its mathematical ceiling is
0.5677 — and it measures 0.5677, sitting exactly at that ceiling with nowhere to
go. The structural model has no such ceiling, and a routing app has to send
people down roads that have not had their crash yet.

---

## Project structure

```text
Capstone/
├── app.py                          Streamlit interface
├── run_pipeline.py                 end-to-end: clean → snap → risk → fit → score
├── requirements.txt
├── src/
│   ├── config.py                   paths and constants
│   ├── data_pipeline.py            Unfallatlas harmonisation across 10 releases
│   ├── osm_network.py              graph download, per-edge features, pickle cache
│   ├── spatial_risk.py             snapping, severity-weighted risk, forward validation
│   ├── frequency_model.py          negative-binomial SPF — the deployed model
│   ├── model_training.py           occurrence classifier, retained as a diagnostic
│   ├── severity_model.py           KSI conditional on a crash
│   ├── national_robustness.py      optional, read-only national check
│   ├── route_engine.py             A* routing, lazy edge costs, map
│   └── visualization.py            risk street map, top-risk table
├── notebooks/                      see below
├── data/{raw,processed}/
├── models/
├── outputs/{maps,figures}/
├── reports/
└── docs/
    ├── NOTEBOOK_INTEGRATION.md     what has reached src/
    └── FINDINGS_TO_INTEGRATE.md    what has not, with rationale and owner
```

## Notebooks

| Question | Notebook |
|---|---|
| What is in eight years of Berlin bicycle crashes? | `01_eda_and_analysis` |
| Which streets carry recorded risk, and does it predict forward? | `02_OSM_Berlin_Street_Risk_Analysis` |
| Was the occurrence model measuring risk or sampling design? | `03_model_leakage_fix` |
| Can severity be predicted at route-planning time? | `04_severity_model` |
| Does the truck mechanism hold outside Berlin? | `05_national_robustness` |
| Do crash counts identify risky streets, or busy ones? | `06_exposure_normalisation` |
| What is in the counter data? | `07_EDA_Exposure_and_Counter_Data` |
| Does the risk score hold up against a real denominator? | `08_Exposure_and_Counter_Validation` |
| What is the defensible frequency model? | `09_frequency_model` |

## Key outputs

```text
data/processed/berlin_bike_2018_2025.csv          37,948 cleaned crashes
data/processed/berlin_accidents_snapped_to_edges.csv
data/processed/berlin_osm_edge_features.csv
data/processed/berlin_route_risk_edges.csv        per-edge risk the router reads

models/frequency_nb_model.joblib                  the deployed SPF coefficients
models/frequency_model_metrics.json               AIC ladder, dispersion, alpha
models/historical_risk_temporal_validation.json   forward validation and baselines
models/leakage_diagnostics.json                   single-feature AUC per leaky column
models/severity_model_metrics.json

outputs/maps/berlin_risk_streets.html
```


