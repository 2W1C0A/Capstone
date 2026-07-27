# ============================================================
# 2W1C Streamlit App Draft
# Machine-Learning-Based Bicycle Safety Routing for Berlin
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import folium
from pathlib import Path
from datetime import datetime

# Optional: better Folium display inside Streamlit
try:
    from streamlit_folium import st_folium
    STREAMLIT_FOLIUM_AVAILABLE = True
except ImportError:
    STREAMLIT_FOLIUM_AVAILABLE = False
    import streamlit.components.v1 as components


# ============================================================
# 1. Page configuration
# ============================================================

st.set_page_config(
    page_title="2W1C Berlin Bicycle Safety Routing",
    page_icon="🚲",
    layout="wide"
)


# ============================================================
# 2. Helper functions
# ============================================================

@st.cache_data
def load_accident_data():
    """
    Load cleaned Berlin bicycle accident data if available.
    If the file is not available, return None.
    """
    possible_paths = [
        Path("data/processed/berlin_bike_accidents_clean.csv"),
        Path("../data/processed/berlin_bike_accidents_clean.csv"),
        Path("berlin_bike_accidents_clean.csv")
    ]

    for path in possible_paths:
        if path.exists():
            return pd.read_csv(path)

    return None


def create_placeholder_map():
    """
    Create a simple Berlin map for the draft app.
    This will be replaced later by fastest vs safest route map.
    """
    berlin_center = [52.5200, 13.4050]

    m = folium.Map(
        location=berlin_center,
        zoom_start=12,
        tiles="OpenStreetMap"
    )

    # Example markers
    folium.Marker(
        location=[52.5219, 13.4132],
        popup="Example start: Alexanderplatz",
        tooltip="Start",
        icon=folium.Icon(color="blue", icon="play")
    ).add_to(m)

    folium.Marker(
        location=[52.5163, 13.3777],
        popup="Example destination: Brandenburg Gate",
        tooltip="Destination",
        icon=folium.Icon(color="black", icon="flag")
    ).add_to(m)

    # Example fastest route line
    folium.PolyLine(
        locations=[
            [52.5219, 13.4132],
            [52.5200, 13.4000],
            [52.5163, 13.3777]
        ],
        color="red",
        weight=5,
        opacity=0.8,
        tooltip="Fastest route example"
    ).add_to(m)

    # Example safer route line
    folium.PolyLine(
        locations=[
            [52.5219, 13.4132],
            [52.5250, 13.3950],
            [52.5220, 13.3850],
            [52.5163, 13.3777]
        ],
        color="green",
        weight=5,
        opacity=0.8,
        tooltip="ML-safest route example"
    ).add_to(m)

    return m


def mock_compare_routes(start_location, destination, hour, weekday, safety_preference):
    """
    Temporary mock function for draft app layout.

    Later, replace this with the real route engine:
        compare_routes_ml(start, end, hour, weekday, safety_preference)
    """

    fastest_distance = 3.2
    safest_distance = 3.7

    fastest_risk = 0.42
    safest_risk = 0.28

    extra_distance = safest_distance - fastest_distance
    risk_reduction = fastest_risk - safest_risk

    result = {
        "fastest_distance_km": fastest_distance,
        "safest_distance_km": safest_distance,
        "extra_distance_km": extra_distance,
        "fastest_risk": fastest_risk,
        "safest_risk": safest_risk,
        "risk_reduction": risk_reduction,
        "recommendation": (
            "Choose the ML-safest route. It is slightly longer, "
            "but the predicted accident risk is lower."
        )
    }

    return result


def display_folium_map(m):
    """
    Display Folium map in Streamlit.
    """
    if STREAMLIT_FOLIUM_AVAILABLE:
        st_folium(m, width=None, height=550)
    else:
        map_html = m._repr_html_()
        components.html(map_html, height=550)


# ============================================================
# 3. Sidebar
# ============================================================

st.sidebar.title("🚲 2W1C")
st.sidebar.markdown("### Bicycle Safety Routing for Berlin")

st.sidebar.markdown("---")

start_location = st.sidebar.text_input(
    "Start location",
    value="Alexanderplatz, Berlin"
)

destination = st.sidebar.text_input(
    "Destination",
    value="Brandenburg Gate, Berlin"
)

travel_date = st.sidebar.date_input(
    "Travel date",
    value=datetime.today()
)

travel_hour = st.sidebar.slider(
    "Travel hour",
    min_value=0,
    max_value=23,
    value=8
)

weekday_name = st.sidebar.selectbox(
    "Weekday",
    options=[
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ],
    index=2
)

safety_preference = st.sidebar.slider(
    "Safety preference",
    min_value=1,
    max_value=10,
    value=7,
    help="Higher value means the route engine will avoid risky road segments more strongly."
)

run_button = st.sidebar.button("Find safer route")

st.sidebar.markdown("---")

st.sidebar.info(
    "Current version: draft layout. "
    "The final version will use an ML model and OSM routing engine."
)


# ============================================================
# 4. Main title
# ============================================================

st.title("🚲 2W1C: ML-Based Bicycle Safety Routing for Berlin")

st.markdown(
    """
    This app compares the **fastest bicycle route** with an **ML-safest route**.

    The final system will use:

    - Unfallatlas bicycle accident data
    - OpenStreetMap road features
    - Time features such as hour, weekday, rush hour, and season
    - A machine-learning model to predict road-segment accident risk
    """
)


# ============================================================
# 5. Load data
# ============================================================

df_accidents = load_accident_data()

if df_accidents is not None:
    st.success(f"Accident dataset loaded successfully: {df_accidents.shape[0]:,} rows × {df_accidents.shape[1]} columns")
else:
    st.warning(
        "Clean accident dataset not found yet. "
        "The app is running in draft mode with example data."
    )


# ============================================================
# 6. Tabs
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Route Demo",
    "📊 Data Overview",
    "🤖 ML Model",
    "ℹ️ About Project"
])


# ============================================================
# Tab 1: Route Demo
# ============================================================

with tab1:
    st.subheader("Fastest Route vs ML-Safest Route")

    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.markdown("### User Input")

        st.write(f"**Start:** {start_location}")
        st.write(f"**Destination:** {destination}")
        st.write(f"**Date:** {travel_date}")
        st.write(f"**Hour:** {travel_hour}:00")
        st.write(f"**Weekday:** {weekday_name}")
        st.write(f"**Safety preference:** {safety_preference}/10")

    with col_result:
        st.markdown("### Route Summary")

        if run_button:
            result = mock_compare_routes(
                start_location=start_location,
                destination=destination,
                hour=travel_hour,
                weekday=weekday_name,
                safety_preference=safety_preference
            )

            st.metric(
                label="Fastest route distance",
                value=f"{result['fastest_distance_km']:.2f} km"
            )

            st.metric(
                label="ML-safest route distance",
                value=f"{result['safest_distance_km']:.2f} km",
                delta=f"+{result['extra_distance_km']:.2f} km"
            )

            st.metric(
                label="Predicted risk reduction",
                value=f"{result['risk_reduction']:.2f}",
                delta="Lower risk"
            )

            st.success(result["recommendation"])

        else:
            st.info("Click **Find safer route** in the sidebar to run the route comparison.")

    st.markdown("### Map")

    m = create_placeholder_map()
    display_folium_map(m)

    st.caption(
        "Red line = example fastest route. Green line = example ML-safest route. "
        "This draft map will be replaced by real OSM route results."
    )


# ============================================================
# Tab 2: Data Overview
# ============================================================

with tab2:
    st.subheader("Berlin Bicycle Accident Data")

    if df_accidents is not None:
        st.markdown("### Dataset Preview")
        st.dataframe(df_accidents.head(20), use_container_width=True)

        st.markdown("### Basic Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Rows", f"{df_accidents.shape[0]:,}")

        with col2:
            st.metric("Columns", f"{df_accidents.shape[1]:,}")

        with col3:
            if "year" in df_accidents.columns:
                st.metric(
                    "Years",
                    f"{df_accidents['year'].min()}–{df_accidents['year'].max()}"
                )
            else:
                st.metric("Years", "Not available")

        if "hour" in df_accidents.columns:
            st.markdown("### Accidents by Hour")
            hour_counts = df_accidents["hour"].value_counts().sort_index()
            st.bar_chart(hour_counts)

        if "month" in df_accidents.columns:
            st.markdown("### Accidents by Month")
            month_counts = df_accidents["month"].value_counts().sort_index()
            st.bar_chart(month_counts)

    else:
        st.info(
            """
            In the final version, this tab will show:

            - Number of Berlin bicycle accidents
            - Accidents by year
            - Accidents by hour
            - Accidents by weekday
            - Accident severity distribution
            - District-level patterns
            """
        )


# ============================================================
# Tab 3: ML Model
# ============================================================

with tab3:
    st.subheader("Machine Learning Accident-Risk Model")

    st.markdown(
        """
        The main AI task is:

        > Predict relative bicycle accident risk for each Berlin road segment at a given time.

        The model will use three groups of features:
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            ### Road features
            - Road type
            - Road length
            - Cycleway availability
            - Speed limit
            - Intersection density
            """
        )

    with col2:
        st.markdown(
            """
            ### Time features
            - Hour
            - Weekday
            - Weekend
            - Rush hour
            - Season
            """
        )

    with col3:
        st.markdown(
            """
            ### Historical features
            - Past accident count
            - Past serious accidents
            - Hotspot score
            - Severity-weighted risk
            """
        )

    st.markdown("---")

    st.markdown("### Baseline vs ML Model")

    model_table = pd.DataFrame({
        "Approach": [
            "Historical hotspot baseline",
            "Rule-based route score",
            "ML accident-risk model"
        ],
        "Description": [
            "Uses only past accident counts",
            "Combines accident counts and simple road rules",
            "Learns risk from road, time, and history features"
        ],
        "AI Level": [
            "Low",
            "Medium",
            "High"
        ]
    })

    st.dataframe(model_table, use_container_width=True)

    st.info(
        "In the final version, this tab will show model performance, "
        "ROC-AUC, PR-AUC, top-k recall, and feature importance."
    )


# ============================================================
# Tab 4: About Project
# ============================================================

with tab4:
    st.subheader("About 2W1C")

    st.markdown(
        """
        **2W1C** is a machine-learning-based bicycle safety routing system for Berlin.

        The project does not only show historical accident hotspots.  
        The core AI part is a machine-learning model that predicts road-segment accident risk.

        The predicted risk is then used by a route engine to compare:

        - the fastest route
        - the ML-safest route

        The final goal is to help cyclists understand the safety trade-off between speed and risk.
        """
    )

    st.markdown("### Project Workflow")

    st.code(
        """
Unfallatlas bicycle accident data
+
OpenStreetMap road network
+
Time features
↓
Map accidents to road segments
↓
Create positive and negative training samples
↓
Train ML accident-risk model
↓
Predict risk for each road segment
↓
Use predicted risk as routing cost
↓
Compare fastest route vs ML-safest route
↓
Show result in Streamlit
        """,
        language="text"
    )

    st.markdown("### Limitations")

    st.markdown(
        """
        - The model predicts **relative risk**, not absolute personal accident probability.
        - Bicycle traffic volume is not included yet.
        - Negative samples are estimated from road segments without observed accidents.
        - Weather and real-time traffic are not included in the first version.
        - The app is for educational and research purposes, not official route safety advice.
        """
    )


# ============================================================
# 7. Footer
# ============================================================

st.markdown("---")
st.caption(
    "2W1C Project | Berlin Bicycle Safety Routing | "
    "Unfallatlas + OpenStreetMap + Machine Learning"
)