from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from streamlit_folium import st_folium
    HAS_STREAMLIT_FOLIUM = True
except ImportError:
    import streamlit.components.v1 as components
    HAS_STREAMLIT_FOLIUM = False

from src.config import (
    CLEAN_ACCIDENT_FILE,
    EDGE_RISK_FILE,
    MODEL_FILE,
    METRICS_FILE,
    OSM_GRAPH_FILE,
    SPATIAL_RISK_FILE,
)
from src.route_engine import RouteEngine
from src.visualization import route_summary_table


st.set_page_config(
    page_title="2W1C Integrated Bicycle Safety Routing",
    page_icon="🚲",
    layout="wide",
)


def show_map(m, key="route_map"):
    if HAS_STREAMLIT_FOLIUM:
        st_folium(m, width=None, height=580, key=key, returned_objects=[])
    else:
        components.html(m._repr_html_(), height=580)


@st.cache_resource
def load_engine():
    return RouteEngine(
        graph_file=OSM_GRAPH_FILE,
        edge_risk_file=EDGE_RISK_FILE if EDGE_RISK_FILE.exists() else SPATIAL_RISK_FILE,
        model_file=MODEL_FILE,
        spatial_risk_file=SPATIAL_RISK_FILE,
    )


@st.cache_data
def load_accident_summary():
    if CLEAN_ACCIDENT_FILE.exists():
        df = pd.read_csv(CLEAN_ACCIDENT_FILE)
        return df
    return None


@st.cache_data
def load_metrics():
    if METRICS_FILE.exists():
        return json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    return None


for key, value in {
    "route_result": None,
    "route_map": None,
    "route_error": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


st.sidebar.title("🚲 2W1C")
st.sidebar.markdown("### Integrated ML + GIS bicycle safety routing")

start_address = st.sidebar.text_input(
    "Start address",
    value="Alexanderplatz, Berlin, Germany",
)

destination_address = st.sidebar.text_input(
    "Destination address",
    value="Brandenburg Gate, Berlin, Germany",
)

month = st.sidebar.slider("Month", min_value=1, max_value=12, value=7)
hour = st.sidebar.slider(
    "Travel hour",
    min_value=0,
    max_value=23,
    value=8,
    help=(
        "Currently this does not visibly change the route. "
        "The current model found that segment risk ranking is almost invariant "
        "across time windows. The slider is retained for a planned time-varying "
        "severity-weighting extension."
    ),
)
st.sidebar.caption(
    "Note: in the current version, route choice is mainly spatial. "
    "The hour slider is kept for transparency and future severity-based modelling."
)

weekday_map = {
    "Sunday": 1,
    "Monday": 2,
    "Tuesday": 3,
    "Wednesday": 4,
    "Thursday": 5,
    "Friday": 6,
    "Saturday": 7,
}
weekday_name = st.sidebar.selectbox(
    "Weekday",
    options=list(weekday_map.keys()),
    index=2,
)
day_of_week = weekday_map[weekday_name]

safety_preference = st.sidebar.slider(
    "Safety preference",
    min_value=1,
    max_value=10,
    value=7,
    help="Higher value means the route engine avoids predicted-risk segments more strongly.",
)

run_button = st.sidebar.button("Compare routes", type="primary")

st.title("🚲 2W1C: Integrated Bicycle Safety Routing for Berlin")

st.markdown(
    """
    This version integrates **Person B's GIS risk analysis** with an **end-to-end ML pipeline**.

    It compares three routes:

    1. **Fastest route** based on distance  
    2. **Historical GIS-risk route** based on severity-weighted spatial risk  
    3. **ML-safest route** based on predicted accident risk  
    """
)

if run_button:
    try:
        with st.spinner("Calculating fastest, historical-risk, and ML-safest routes..."):
            engine = load_engine()
            result, route_map = engine.compare_and_map(
                start_address=start_address,
                destination_address=destination_address,
                month=month,
                hour=hour,
                day_of_week=day_of_week,
                safety_preference=safety_preference,
            )

        st.session_state.route_result = result
        st.session_state.route_map = route_map
        st.session_state.route_error = None

    except Exception as exc:
        st.session_state.route_result = None
        st.session_state.route_map = None
        st.session_state.route_error = str(exc)


tab1, tab2, tab3 = st.tabs([
    "🗺️ Route comparison",
    "📊 Data",
    "🤖 Model",
])


with tab1:
    st.subheader("Route Comparison")

    if st.session_state.route_error:
        st.error(st.session_state.route_error)

    if st.session_state.route_result is None:
        st.info("Use the sidebar and click **Compare routes**.")
    else:
        result = st.session_state.route_result
        summary = route_summary_table(result)

        col1, col2, col3 = st.columns(3)

        fastest = result["fastest_summary"]
        historical = result["historical_summary"]
        ml = result["ml_summary"]

        col1.metric("Fastest distance", f"{fastest['distance_km']:.2f} km")
        col2.metric("Historical-risk distance", f"{historical['distance_km']:.2f} km")
        col3.metric("ML-safest distance", f"{ml['distance_km']:.2f} km")

        st.markdown("### Summary table")
        st.dataframe(summary, use_container_width=True)

        st.markdown("### Recommendation")
        st.text(result["recommendation_text"])

        st.markdown("### Map")
        show_map(st.session_state.route_map)


with tab2:
    st.subheader("Data Overview")
    df = load_accident_summary()

    if df is None:
        st.warning("Clean accident data not found. Run the pipeline first.")
    else:
        st.success(f"Clean accident data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{df.shape[0]:,}")
        col2.metric("Years", f"{int(df['year'].min())}–{int(df['year'].max())}" if "year" in df.columns else "NA")
        col3.metric("Serious/fatal rate", f"{100*df['serious_or_fatal'].mean():.1f}%" if "serious_or_fatal" in df.columns else "NA")

        st.dataframe(df.head(20), use_container_width=True)

        if "hour" in df.columns:
            st.markdown("### Accidents by hour")
            st.bar_chart(df["hour"].value_counts().sort_index())

        if "month" in df.columns:
            st.markdown("### Accidents by month")
            st.bar_chart(df["month"].value_counts().sort_index())


with tab3:
    st.subheader("Machine Learning Model")
    metrics = load_metrics()

    if metrics is None:
        st.warning("Model metrics not found. Run the pipeline first.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}" if metrics["roc_auc"] is not None else "NA")
        col2.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
        col3.metric("Top-10% recall", f"{metrics['top10_recall']:.3f}")
        col4.metric("Top-20% recall", f"{metrics['top20_recall']:.3f}")

        st.markdown("### Train/Test")
        st.write({
            "n_train": metrics.get("n_train"),
            "n_test": metrics.get("n_test"),
            "positive_rate_train": metrics.get("positive_rate_train"),
            "positive_rate_test": metrics.get("positive_rate_test"),
        })

    st.markdown(
        """
        **ML task:** road segment + time + road features + historical GIS risk → accident risk.

        The historical spatial-risk model is used as both:
        - a baseline
        - an input feature for the ML model
        """
    )

