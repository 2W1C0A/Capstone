"""Segment-level crash frequency model (negative binomial SPF).

This is the frequency term of the risk surface, replacing the occurrence
classifier in `model_training.py`. That classifier had to invent its own negative
examples, and roughly half of its apparent performance was network composition
rather than risk (see `03_model_leakage_fix`).

Here the unit is the undirected segment and the target is a count, so the zeros
are observed data rather than fabrications: eight years passed and no crash was
recorded. A log-length offset makes it a rate model — crashes per metre. In the
road safety literature this is a Safety Performance Function.

Deployed specification:

    crash_count ~ C(highway_simple, Treatment(reference="cycleway"))
                  + maxspeed_num + has_cycleway + junction_ends
                  offset = log(length_m)

Junction structure is included because it is where the project's headline
mechanism lives: 80.8% of crashes fall within 20 m of a node and 52.2% of truck
crashes are turning conflicts. A cost function without a junction term cannot
route around the thing the analysis is about.

Two specifications fit better and are deliberately NOT deployed, because both
buy their fit from OSM's segmentation rather than from risk:

  * adding `junction_density` (= ends * 100 / length) reaches AIC 157,944
    against 159,225, but the term is largely 1/length in disguise and makes
    expected crashes non-monotonic in length — a 1 m stub scores worse than a
    20 m one.
  * adding `log_len` as a free covariate reaches AIC 155,379 and estimates a
    length exponent of 0.486. That is a real finding — crashes scale
    sub-linearly with length, concentrating at junctions rather than along the
    line — but it cannot be a routing cost: one 1000 m segment would cost 28.7
    while ten 100 m segments covering the same ground cost 93.7. The router
    would price identical roads differently depending on how finely OSM split
    them.

Both live in notebook 09 as evidence. Only the pure-offset rate model is
deployed.

The DTV (motor traffic volume) extension in notebook 09 is also evidence only:
it covers 37% of segments, so it cannot score the whole network. Its role is to
show the road-class gradient survives controlling for traffic volume.

What this model does not have is a cyclist-volume denominator. It gives crashes
per metre, not per cyclist-kilometre, so classes almost nobody rides (`path`,
`service`, `track`) come out with the lowest rates. That is exposure, not
safety. The routing engine must apply a measured class constraint on top;
`06_exposure_normalisation` section 6.2 is the justification.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import (
    FREQUENCY_METRICS_FILE,
    FREQUENCY_MODEL_FILE,
    OSM_EDGE_FEATURES_FILE,
    SNAPPED_ACCIDENTS_FILE,
)

REFERENCE_CLASS = "cycleway"
WINDOW_YEARS = 8

# statsmodels' own negativebinomial estimates dispersion jointly and diverges on
# this data: alpha overflows and the Hessian cannot be inverted. Fitting NB as a
# GLM at fixed alpha and selecting alpha by likelihood over a grid is the
# standard alternative. Log-likelihoods are comparable at fixed alpha.
ALPHA_GRID = np.linspace(0.5, 12.0, 24)

# pair_id keys on sorted(u, v), which merges the two directions of one street
# but also merges genuinely parallel ways between the same two nodes. Where the
# directed edges disagree on length by more than this, "first" is picking one of
# several real roads and both the class label and the length offset become
# order-dependent.
LENGTH_DISAGREEMENT_M = 1.0
LENGTH_FLOOR_M = 1.0

# A class with a handful of segments and no crashes is perfectly separated, and
# its coefficient diverges to negative infinity. `other` has 4 segments and did
# exactly that. The fitted value then reaches the router as exp(-large) = 0,
# which prices those edges as free. Classes below this many segments are pooled
# into one `rare` category so every coefficient is estimated from real variation.
MIN_CLASS_SEGMENTS = 20
POOLED_CLASS_NAME = "rare"

FORMULA_BASE = "crash_count ~ C(highway_simple) + maxspeed_num + has_cycleway"
FORMULA_REFERENCED = (
    'crash_count ~ C(highway_simple, Treatment(reference="cycleway")) '
    "+ maxspeed_num + has_cycleway"
)
FORMULA_DEPLOYED = FORMULA_REFERENCED + " + junction_ends"

# Fits better, not deployed. Kept so the notebook can reproduce the comparison.
FORMULA_DENSITY = FORMULA_DEPLOYED + " + junction_density"
FORMULA_LOGLEN = FORMULA_DEPLOYED + " + log_len"

REQUIRED_PREDICT_COLUMNS = (
    "edge_uid",
    "edge_length_m",
    "highway_simple",
    "maxspeed_num",
    "has_cycleway",
    "junction_ends",
)


# ---------------------------------------------------------------------
# 1. Segment table
# ---------------------------------------------------------------------


def build_segment_table(
    edges: pd.DataFrame | None = None,
    snapped: pd.DataFrame | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Collapse directed OSM edges to undirected segments, attach crash counts.

    A two-way street is two directed edges in OSM but one physical segment.
    Counting it twice would double both the crash count and the length offset,
    which cancels in the ratio but inflates the sample and shrinks the standard
    errors — the estimates would look more precise than the data supports.

    Adds `length_ambiguous`: True where the directed edges of a pair disagree on
    length by more than a metre, which means several physical ways were merged
    and `first` picked one arbitrarily. `fit_frequency_model` excludes those.
    """
    if edges is None:
        edges = pd.read_csv(OSM_EDGE_FEATURES_FILE)
    if snapped is None:
        snapped = pd.read_csv(SNAPPED_ACCIDENTS_FILE)

    for frame, name, cols in [
        (edges, "edges", ["pair_id", "edge_length_m", "highway_simple"]),
        (snapped, "snapped", ["pair_id"]),
    ]:
        absent = [c for c in cols if c not in frame.columns]
        if absent:
            raise ValueError(f"{name} is missing required columns: {absent}")

    seg = (
        edges.groupby("pair_id")
        .agg(
            length_m=("edge_length_m", "first"),
            highway_simple=("highway_simple", "first"),
            has_cycleway=("has_cycleway", "max"),
            maxspeed_num=("maxspeed_num", "first"),
        )
        .reset_index()
    )

    spread = edges.groupby("pair_id")["edge_length_m"].agg(["min", "max"])
    ambiguous = set(
        spread.index[(spread["max"] - spread["min"]) > LENGTH_DISAGREEMENT_M]
    )
    seg["length_ambiguous"] = seg["pair_id"].isin(ambiguous)

    agg = {"crash_count": ("pair_id", "size")}
    for col, out in [
        ("is_ksi", "ksi_count"),
        ("is_fatal", "fatal_count"),
        ("is_truck", "truck_count"),
    ]:
        if col in snapped.columns:
            agg[out] = (col, "sum")
    if "near_junction" in snapped.columns:
        agg["near_junction"] = ("near_junction", "mean")

    crashes = snapped.groupby("pair_id").agg(**agg).reset_index()
    seg = seg.merge(crashes, on="pair_id", how="left")
    fill = [c for c in crashes.columns if c != "pair_id"]
    seg[fill] = seg[fill].fillna(0)

    # A zero length would break the log offset.
    seg = seg[seg["length_m"] > 0].copy()
    seg["log_len"] = np.log(seg["length_m"])

    if verbose:
        mean = seg["crash_count"].mean()
        var = seg["crash_count"].var()
        print(f"{len(seg):,} undirected segments")
        print(f"crashes assigned: {seg['crash_count'].sum():,.0f} of {len(snapped):,}")
        print(f"segments with >=1 crash: {(seg['crash_count'] > 0).mean():.1%}")
        print(f"\nmean {mean:.4f} | variance {var:.4f}")
        print(f"variance / mean = {var / mean:.2f}")
        n_amb = int(seg["length_ambiguous"].sum())
        print(
            f"\nlength-ambiguous pairs: {n_amb:,} ({n_amb / len(seg):.2%}), "
            f"carrying {seg.loc[seg['length_ambiguous'], 'crash_count'].sum():,.0f} "
            "crashes"
        )

    return seg


# ---------------------------------------------------------------------
# 2. Junction features
# ---------------------------------------------------------------------


def add_junction_features(
    df: pd.DataFrame,
    graph,
    edge_lookup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach junction structure derived from the graph, not from any OSM tag.

    Each endpoint of a segment has a degree. In a MultiDiGraph a degree of 2 is
    one edge in and one out — the road continuing, not a junction. Higher means
    a junction.

    Works on either a segment table (`pair_id`, needs `edge_lookup` to recover
    u/v) or a directed edge table that already carries `u` and `v`. Both
    directions of a pair share endpoints, so a directed edge receives its
    segment's junction values, which is the intended reading.

    `junction_density` is computed for the notebook's specification comparison.
    It is NOT in the deployed formula: it is largely 1/length in disguise and
    makes expected crashes non-monotonic in segment length.
    """
    length_col = "length_m" if "length_m" in df.columns else "edge_length_m"
    if length_col not in df.columns:
        raise ValueError("df needs either 'length_m' or 'edge_length_m'.")

    degree = dict(graph.degree())
    if not degree:
        raise ValueError("Graph has no nodes.")

    if {"u", "v"}.issubset(df.columns):
        ends = df[["u", "v"]].copy()
    elif "pair_id" in df.columns:
        if edge_lookup is None or not {"pair_id", "u", "v"}.issubset(
            edge_lookup.columns
        ):
            raise ValueError(
                "Segment tables need edge_lookup with pair_id, u and v to recover "
                "endpoints."
            )
        lookup = edge_lookup.drop_duplicates("pair_id").set_index("pair_id")
        ends = pd.DataFrame(
            {
                "u": df["pair_id"].map(lookup["u"]),
                "v": df["pair_id"].map(lookup["v"]),
            },
            index=df.index,
        )
    else:
        raise ValueError("df needs (u, v) or pair_id.")

    # graphml round-trips node ids as strings in some osmnx versions while the
    # CSV carries them as int64. A dtype mismatch makes every lookup miss, which
    # would silently set junction_ends to 0 everywhere and quietly remove the
    # strongest covariate in the model. Try both, then verify.
    str_degree = {str(k): v for k, v in degree.items()}

    def _map_degree(series: pd.Series) -> pd.Series:
        direct = series.map(degree)
        if direct.notna().mean() >= 0.99:
            return direct
        as_str = series.astype(str).map(str_degree)
        return as_str if as_str.notna().mean() > direct.notna().mean() else direct

    deg_u = _map_degree(ends["u"])
    deg_v = _map_degree(ends["v"])

    matched = float(min(deg_u.notna().mean(), deg_v.notna().mean()))
    if matched < 0.99:
        raise ValueError(
            f"Only {matched:.1%} of segment endpoints were found in the graph. "
            "The edge table and the graph are from different versions, so "
            "junction_ends would be silently zero. Regenerate both from one graph."
        )

    out = df.copy()
    deg_u = deg_u.fillna(0)
    deg_v = deg_v.fillna(0)
    out["junction_ends"] = (deg_u > 2).astype(int) + (deg_v > 2).astype(int)
    out["max_degree"] = np.maximum(deg_u, deg_v)

    # Diagnostic only — see the docstring.
    length = pd.to_numeric(out[length_col], errors="coerce").clip(lower=LENGTH_FLOOR_M)
    out["junction_density"] = out["junction_ends"] * 100.0 / length

    return out


# ---------------------------------------------------------------------
# 3. Fit
# ---------------------------------------------------------------------


def _fit_nb(formula: str, data: pd.DataFrame, alpha: float):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.glm(
            formula,
            data=data,
            family=sm.families.NegativeBinomial(alpha=alpha),
            offset=data["log_len"],
        ).fit()


def fit_frequency_model(
    seg: pd.DataFrame,
    alpha_grid: np.ndarray = ALPHA_GRID,
    exclude_ambiguous_pairs: bool = True,
    min_segments: int = MIN_CLASS_SEGMENTS,
    model_file: str | Path = FREQUENCY_MODEL_FILE,
    metrics_file: str | Path = FREQUENCY_METRICS_FILE,
    save: bool = True,
) -> tuple[dict, pd.DataFrame]:
    """Fit the deployed NB SPF and persist a coefficient bundle.

    Returns (bundle, rate_ratio_table).

    `exclude_ambiguous_pairs` drops the 0.8% of segments where several physical
    ways were merged under one pair_id, because both the class label and the
    length offset are order-dependent there. It costs 0.5% of crashes. Absolute
    predicted rates move by 1-3%; what shifts is how the fit divides itself
    between the intercept and the contrasts against the reference class, so the
    published rate ratios change by about 20% while `expected_crashes` does not.

    The fitted statsmodels object is deliberately not persisted. It holds a
    reference to every row, so the pickle would be large, and unpickling it
    would tie the router to a statsmodels version. The coefficients are all
    prediction needs, and they make the saved artefact readable.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    for col in ["junction_ends", "log_len"]:
        if col not in seg.columns:
            raise ValueError(
                f"seg is missing '{col}'. Call add_junction_features() first."
            )

    full = seg.copy()
    n_excluded = 0
    if exclude_ambiguous_pairs:
        if "length_ambiguous" not in full.columns:
            raise ValueError(
                "seg lacks 'length_ambiguous'; rebuild it with build_segment_table()."
            )
        excluded = full[full["length_ambiguous"]]
        n_excluded = len(excluded)
        crashes_lost = float(excluded["crash_count"].sum())
        data = full[~full["length_ambiguous"]].copy()
        print(
            f"excluded {n_excluded:,} length-ambiguous segments "
            f"({crashes_lost:,.0f} of {full['crash_count'].sum():,.0f} crashes, "
            f"{crashes_lost / full['crash_count'].sum():.1%})\n"
        )
    else:
        data = full

    data, pooled = _pool_rare_classes(data, min_segments)

    # Poisson is fitted only to measure overdispersion. Pearson chi2/df near 1.0
    # would mean its equal-variance assumption holds and NB is unnecessary.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pois = smf.glm(
            FORMULA_BASE,
            data=data,
            family=sm.families.Poisson(),
            offset=data["log_len"],
        ).fit()

    fits = [(a, _fit_nb(FORMULA_BASE, data, a)) for a in alpha_grid]
    best_alpha, nb_base = max(fits, key=lambda r: r[1].llf)

    nb_ref = _fit_nb(FORMULA_REFERENCED, data, best_alpha)
    nb = _fit_nb(FORMULA_DEPLOYED, data, best_alpha)

    print(f"n = {len(data):,}   alpha = {best_alpha:.2f}\n")
    print(f"Poisson              AIC {pois.aic:>12,.0f}")
    print(f"NB                   AIC {nb_base.aic:>12,.0f}")
    print(f"NB + junction_ends   AIC {nb.aic:>12,.0f}   ({nb.aic - nb_ref.aic:+,.0f})")
    print(
        f"\nPearson chi2/df: Poisson {pois.pearson_chi2 / pois.df_resid:.2f}"
        f"  ->  NB {nb_base.pearson_chi2 / nb_base.df_resid:.2f}"
    )
    print(
        f"maxspeed coefficient — Poisson {pois.params['maxspeed_num']:+.4f}, "
        f"NB {nb_base.params['maxspeed_num']:+.4f}"
    )

    table = _rate_ratio_table(nb, data)
    bundle = _build_bundle(
        nb, data, best_alpha, pois, nb_base, nb_ref, n_excluded,
        exclude_ambiguous_pairs, pooled,
    )

    if save:
        Path(model_file).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, model_file)
        Path(metrics_file).write_text(
            json.dumps(bundle["fit"], indent=2), encoding="utf-8"
        )
        print(f"\nsaved frequency model: {model_file}")
        print(f"saved frequency metrics: {metrics_file}")

    return bundle, table


def _pool_rare_classes(
    data: pd.DataFrame, min_segments: int
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Merge classes with too few segments into one category.

    Returns the reframed data and a mapping from original class to the label it
    was pooled into, so prediction can apply the same substitution.
    """
    counts = data["highway_simple"].value_counts()
    rare = [
        c
        for c, n in counts.items()
        if n < min_segments and c != REFERENCE_CLASS
    ]
    if not rare:
        return data, {}

    out = data.copy()
    out["highway_simple"] = out["highway_simple"].where(
        ~out["highway_simple"].isin(rare), POOLED_CLASS_NAME
    )
    print(
        f"pooled {len(rare)} class(es) with < {min_segments} segments into "
        f"'{POOLED_CLASS_NAME}': "
        + ", ".join(f"{c} (n={counts[c]})" for c in rare)
    )
    return out, {c: POOLED_CLASS_NAME for c in rare}


def _clean_name(name: str) -> str:
    return (
        name.replace('C(highway_simple, Treatment(reference="cycleway"))[T.', "")
        .replace("C(highway_simple)[T.", "")
        .rstrip("]")
    )


def _rate_ratio_table(nb, seg: pd.DataFrame) -> pd.DataFrame:
    ci = nb.conf_int()
    table = pd.DataFrame(
        {
            "rate_ratio": np.exp(nb.params),
            "ci_low": np.exp(ci[0]),
            "ci_high": np.exp(ci[1]),
            "p": nb.pvalues,
        }
    ).drop("Intercept")
    table.index = [_clean_name(i) for i in table.index]
    table["n_segments"] = table.index.map(seg["highway_simple"].value_counts())
    return table.sort_values("rate_ratio", ascending=False)


def _build_bundle(
    nb, seg, alpha, pois, nb_base, nb_ref, n_excluded, excluded_flag, pooled
) -> dict:
    params = nb.params
    class_coef = {
        _clean_name(k): float(v) for k, v in params.items() if "highway_simple" in k
    }
    # The reference class is absorbed into the intercept, so its own effect is 0.
    class_coef[REFERENCE_CLASS] = 0.0

    # Pooling should have removed every separable category. If a coefficient is
    # still non-finite or absurd, exp() of it reaches the router as either zero
    # or infinity, so fail here rather than ship a cost function with a free or
    # impassable class.
    bad = {
        k: v
        for k, v in class_coef.items()
        if not np.isfinite(v) or abs(v) > 25
    }
    if bad:
        raise ValueError(
            f"Degenerate class coefficients after pooling: {bad}. These classes "
            "are perfectly separated; raise min_segments so they are pooled."
        )

    return {
        "model_name": "negative_binomial_spf_rate",
        "intercept": float(params["Intercept"]),
        "class_coef": class_coef,
        "maxspeed_coef": float(params["maxspeed_num"]),
        "has_cycleway_coef": float(params["has_cycleway"]),
        "junction_ends_coef": float(params["junction_ends"]),
        "length_floor_m": LENGTH_FLOOR_M,
        "alpha": float(alpha),
        "reference_class": REFERENCE_CLASS,
        "formula": FORMULA_DEPLOYED,
        "window_years": WINDOW_YEARS,
        "offset": "log(length_m)",
        "target": "crash_count",
        "excluded_ambiguous_pairs": bool(excluded_flag),
        "pooled_classes": dict(pooled),
        "n_excluded_segments": int(n_excluded),
        "interpretation": (
            "expected_crashes = length_m * exp(intercept + class_coef "
            "+ maxspeed_coef*maxspeed_num + has_cycleway_coef*has_cycleway "
            "+ junction_ends_coef*junction_ends), over an 8-year window. Per "
            "metre, NOT per cyclist-kilometre: there is no volume denominator, "
            "so low-exposure classes read as safe. The router must apply a "
            "measured class constraint on top. Because the length exponent is "
            "fixed at 1 by the offset, expected_crashes is additive along a "
            "route and independent of how finely OSM split the way."
        ),
        "fit": {
            "n_segments": int(len(seg)),
            "n_excluded_segments": int(n_excluded),
            "zero_share": float((seg["crash_count"] == 0).mean()),
            "mean": float(seg["crash_count"].mean()),
            "variance": float(seg["crash_count"].var()),
            "variance_mean_ratio": float(
                seg["crash_count"].var() / seg["crash_count"].mean()
            ),
            "alpha": float(alpha),
            "poisson_aic": float(pois.aic),
            "nb_aic": float(nb_base.aic),
            "nb_referenced_aic": float(nb_ref.aic),
            "nb_deployed_aic": float(nb.aic),
            "aic_reduction_poisson_to_nb": float(pois.aic - nb_base.aic),
            "aic_reduction_junction_ends": float(nb_ref.aic - nb.aic),
            "poisson_pearson_chi2_df": float(pois.pearson_chi2 / pois.df_resid),
            "nb_pearson_chi2_df": float(nb_base.pearson_chi2 / nb_base.df_resid),
            "note": (
                "AIC values here are on the fitted subset. They are not "
                "comparable with fits on a different number of rows, so the "
                "excluded-pairs comparison is reported as a robustness check "
                "rather than as a model selection."
            ),
        },
    }


# ---------------------------------------------------------------------
# 4. Predict — this is what the router calls
# ---------------------------------------------------------------------


def predict_edge_expected_crashes(
    edge_features: pd.DataFrame,
    model_bundle: dict | str | Path = FREQUENCY_MODEL_FILE,
    strict: bool = True,
) -> pd.DataFrame:
    """Expected crashes per directed edge over the 8-year window.

    Returns `edge_features` with one column added:
        expected_crashes : float, strictly positive, = length_m * exp(Xb)

    The model was fitted on undirected segments, and both directed edges of a
    pair share length, class and endpoints, so each direction receives the whole
    segment's expected count. That is the intended reading: a cyclist traversing
    the segment is exposed to its hazard regardless of direction, and the
    recorded crashes cannot be attributed to one direction anyway.

    `junction_ends` must already be present — call `add_junction_features(edges,
    graph)` once in the pipeline and persist the result. It is required rather
    than defaulted because defaulting it to zero would silently drop the
    strongest covariate in the model.

    With `strict=True` an unrecognised `highway_simple` raises. It must: the
    silent alternative is to fall back to the reference class, and the reference
    is `cycleway`, so an unknown class would score as one of the safest options
    on the network.
    """
    if not isinstance(model_bundle, dict):
        model_bundle = joblib.load(model_bundle)

    absent = [c for c in REQUIRED_PREDICT_COLUMNS if c not in edge_features.columns]
    if absent:
        raise ValueError(
            f"edge_features is missing required columns: {absent}. "
            "junction_ends comes from add_junction_features()."
        )

    out = edge_features.copy()
    floor = float(model_bundle.get("length_floor_m", LENGTH_FLOOR_M))

    length = (
        pd.to_numeric(out["edge_length_m"], errors="coerce")
        .fillna(floor)
        .clip(lower=floor)
    )
    maxspeed = pd.to_numeric(out["maxspeed_num"], errors="coerce").fillna(30.0)
    cycleway = pd.to_numeric(out["has_cycleway"], errors="coerce").fillna(0)
    ends = pd.to_numeric(out["junction_ends"], errors="coerce").fillna(0)
    highway = out["highway_simple"].astype(str)

    # Classes pooled at fitting time have no coefficient of their own; map them
    # to the label they were pooled into before looking anything up.
    pooled = model_bundle.get("pooled_classes") or {}
    if pooled:
        highway = highway.replace(pooled)

    class_coef = model_bundle["class_coef"]
    unknown = sorted(set(highway.unique()) - set(class_coef))
    if unknown:
        message = (
            f"highway_simple values absent from the fitted model: {unknown}. "
            "The graph or the class simplification changed since fitting; refit "
            "before using these predictions."
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message, stacklevel=2)

    # Only reachable with strict=False. The median class effect keeps an unknown
    # class from being ranked as safe as a cycleway.
    fallback = float(np.median(list(class_coef.values())))

    linpred = (
        model_bundle["intercept"]
        + highway.map(class_coef).fillna(fallback).astype(float)
        + model_bundle["maxspeed_coef"] * maxspeed
        + model_bundle["has_cycleway_coef"] * cycleway
        + model_bundle["junction_ends_coef"] * ends
    )

    out["expected_crashes"] = length * np.exp(linpred)
    return out


def rate_ratio_summary(
    model_bundle: dict | str | Path = FREQUENCY_MODEL_FILE,
) -> pd.DataFrame:
    """Rate ratios relative to the reference class, from a saved bundle."""
    if not isinstance(model_bundle, dict):
        model_bundle = joblib.load(model_bundle)
    rows = {k: np.exp(v) for k, v in model_bundle["class_coef"].items()}
    for name in ["junction_ends", "maxspeed", "has_cycleway"]:
        key = f"{name}_coef"
        if key in model_bundle:
            rows[name] = np.exp(model_bundle[key])
    return pd.DataFrame({"rate_ratio": rows}).sort_values(
        "rate_ratio", ascending=False
    )


def compare_specifications(
    seg: pd.DataFrame,
    alpha: float,
    exclude_ambiguous_pairs: bool = True,
    min_segments: int = MIN_CLASS_SEGMENTS,
) -> pd.DataFrame:
    """The specification ladder from notebook 09, for the record.

    Two specifications fit better than the deployed one and neither can be a
    routing cost. This function exists so that claim stays reproducible.

    Fits on the same rows as the deployed model by default, because AIC is not
    comparable across different numbers of observations.
    """
    if exclude_ambiguous_pairs and "length_ambiguous" in seg.columns:
        seg = seg[~seg["length_ambiguous"]].copy()
    seg, _ = _pool_rare_classes(seg, min_segments)

    rows = []
    for label, formula in [
        ("NB (no junctions)", FORMULA_REFERENCED),
        ("+ junction_ends  [deployed]", FORMULA_DEPLOYED),
        ("+ junction_density", FORMULA_DENSITY),
        ("+ log_len covariate", FORMULA_LOGLEN),
    ]:
        if "junction_density" in formula and "junction_density" not in seg.columns:
            continue
        fit = _fit_nb(formula, seg, alpha)
        primary = next(
            (p for p in fit.params.index if p.endswith("[T.primary]")), None
        )
        rows.append(
            {
                "specification": label,
                "aic": round(float(fit.aic)),
                "primary_rate_ratio": (
                    round(float(np.exp(fit.params[primary])), 3) if primary else None
                ),
                "junction_ends_rate_ratio": (
                    round(float(np.exp(fit.params["junction_ends"])), 3)
                    if "junction_ends" in fit.params
                    else None
                ),
                "length_exponent": (
                    round(1 + float(fit.params["log_len"]), 3)
                    if "log_len" in fit.params
                    else 1.0
                ),
            }
        )
    return pd.DataFrame(rows)


def build_and_fit(graph, save: bool = True) -> tuple[dict, pd.DataFrame]:
    """Convenience entry point for run_pipeline.py.

    `graph` must be the projected graph the edge features were built from.
    """
    edges = pd.read_csv(OSM_EDGE_FEATURES_FILE)
    seg = build_segment_table(edges=edges)
    seg = add_junction_features(seg, graph, edge_lookup=edges)
    return fit_frequency_model(seg, save=save)
