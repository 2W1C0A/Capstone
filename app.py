from __future__ import annotations

import html
import json

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
    page_title="2W1C · Bicycle Safety Routing",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Visual layer
# =============================================================================

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Inter+Tight:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root{
  --asphalt:#15181C;
  --asphalt-2:#212832;
  --paper:#EDF1F4;
  --lane:#12A55F;
  --caution:#E8A317;
  --risk:#DC3B32;
  --ink:#12161A;
  --muted:#5C6874;
  --hair:rgba(18,22,26,.12);
}

/* ---- canvas: risk heatmap blooms under lane markings --------------------- */
.stApp{
  background-color:var(--paper);
  background-image:
    radial-gradient(circle at 10% 16%, rgba(18,165,95,.22), transparent 42%),
    radial-gradient(circle at 86% 10%, rgba(232,163,23,.20), transparent 40%),
    radial-gradient(circle at 70% 86%, rgba(220,59,50,.16), transparent 46%),
    repeating-linear-gradient(115deg, rgba(18,22,26,.05) 0 2px, transparent 2px 27px);
  background-attachment:fixed;
}
.stApp{
  font-family:'Inter Tight',system-ui,sans-serif;
  color:var(--ink);
}
.stApp p, .stApp li, .stApp label{
  font-family:'Inter Tight',system-ui,sans-serif;
}
h1,h2,h3,h4{
  font-family:'Bricolage Grotesque','Inter Tight',sans-serif !important;
  letter-spacing:-.02em; color:var(--ink) !important;
}
.block-container{padding-top:1.6rem; max-width:1440px;}

/* ---- hero --------------------------------------------------------------- */
.hero{
  position:relative; overflow:hidden;
  background-color:var(--asphalt);
  background-image:linear-gradient(100deg,var(--asphalt) 0%,var(--asphalt-2) 62%,#2B3440 100%);
  border-radius:22px; padding:30px 34px 26px;
  box-shadow:0 18px 44px rgba(12,18,26,.28);
  margin-bottom:24px;
}
.hero::after{
  content:""; position:absolute; left:0; right:0; bottom:16px; height:3px;
  background:repeating-linear-gradient(90deg,rgba(255,255,255,.40) 0 26px,transparent 26px 50px);
}
.hero .eyebrow{
  font-family:'JetBrains Mono',monospace; font-size:.7rem; letter-spacing:.22em;
  text-transform:uppercase; color:#2ED37F;
}
.hero .hero-title{
  font-family:'Bricolage Grotesque','Inter Tight',sans-serif;
  font-size:clamp(1.75rem,3.3vw,2.8rem); line-height:1.03; font-weight:800;
  letter-spacing:-.02em; color:#FFFFFF; margin:.4rem 0 .55rem;
}
.hero .hero-title .accent{color:#2ED37F;}
.hero .lede{color:#AEBBC8; max-width:64ch; margin:0 0 18px; font-size:.97rem;}
.hero .chips{display:flex; gap:8px; flex-wrap:wrap;}
.hero .chip{
  font-family:'JetBrains Mono',monospace; font-size:.75rem; letter-spacing:.03em;
  padding:6px 12px; border-radius:999px;
  border:1px solid rgba(255,255,255,.22);
  background:rgba(255,255,255,.08);
  color:#D3DFEA;                       /* ~9:1 against the asphalt band */
}
.hero .chip b{color:#FFFFFF; font-weight:700;}
.dot{display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:7px;}

/* ---- lane cards --------------------------------------------------------- */
.lane{
  position:relative; background:rgba(255,255,255,.84);
  backdrop-filter:blur(9px); border:1px solid rgba(255,255,255,.9);
  border-radius:18px; padding:18px 20px 16px 26px; height:100%;
  box-shadow:0 10px 26px rgba(16,32,64,.10); transition:transform .18s ease;
}
.lane:hover{transform:translateY(-3px);}
.lane::before{
  content:""; position:absolute; left:10px; top:18px; bottom:18px; width:4px;
  border-radius:4px; background:var(--c);
}
.lane .tag{
  font-family:'JetBrains Mono',monospace; font-size:.68rem; letter-spacing:.16em;
  text-transform:uppercase; color:var(--muted);
}
.lane .big{
  font-family:'Bricolage Grotesque',sans-serif; font-weight:800;
  font-size:2.1rem; line-height:1; margin:.28rem 0 .1rem; color:var(--ink);
}
.lane .big small{font-size:.9rem; font-weight:600; color:var(--muted); margin-left:4px;}
.lane .delta{font-family:'JetBrains Mono',monospace; font-size:.76rem; color:var(--muted);}
.bar{height:6px; border-radius:6px; background:rgba(18,22,26,.09); margin-top:14px; overflow:hidden;}
.bar span{display:block; height:100%; border-radius:6px;}
.bar-cap{
  display:flex; justify-content:space-between; margin-top:6px;
  font-family:'JetBrains Mono',monospace; font-size:.68rem; color:var(--muted);
}
.kv{
  display:flex; justify-content:space-between; gap:10px; padding:5px 0;
  border-top:1px solid rgba(18,22,26,.08);
  font-family:'JetBrains Mono',monospace; font-size:.72rem; color:var(--muted);
}
.kv b{color:var(--ink); font-weight:700;}
.kv-first{margin-top:12px;}

/* ---- panels, section labels, empty state -------------------------------- */
.sec{
  font-family:'JetBrains Mono',monospace; font-size:.7rem; letter-spacing:.18em;
  text-transform:uppercase; color:var(--muted); margin:2px 0 10px;
}
.panel{
  background:rgba(255,255,255,.80); backdrop-filter:blur(9px);
  border:1px solid rgba(255,255,255,.9); border-radius:18px; padding:20px 22px;
  box-shadow:0 10px 26px rgba(16,32,64,.08); color:var(--ink);
}
.rec{
  border-left:4px solid var(--lane); background:rgba(18,165,95,.07);
  border-radius:0 14px 14px 0; padding:16px 18px;
  font-family:'JetBrains Mono',monospace; font-size:.8rem; line-height:1.7;
  color:var(--ink); overflow-x:auto;
}
.empty{
  border:1.5px dashed rgba(18,22,26,.20); border-radius:20px;
  padding:44px 30px; text-align:center; background:rgba(255,255,255,.5);
}
.empty h3{margin:0 0 6px; font-size:1.3rem;}
.empty p{color:var(--muted); margin:0 auto; max-width:46ch;}
.legend{
  display:flex; gap:20px; flex-wrap:wrap; align-items:center;
  font-family:'JetBrains Mono',monospace; font-size:.73rem; color:var(--muted);
  margin:0 0 10px;
}

/* ---- sidebar ------------------------------------------------------------ */
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#14171B 0%,#1E252D 100%);
  border-right:1px solid rgba(255,255,255,.07);
}
section[data-testid="stSidebar"] *{color:#E6EDF3 !important;}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] [data-baseweb="select"] *{color:#12161A !important;}
section[data-testid="stSidebar"] input{
  background:#F6F8FA !important; border-radius:10px !important; border-color:transparent !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"]>div{
  background:#F6F8FA !important; border-radius:10px !important; border-color:transparent !important;
}
.sb-head{
  font-family:'JetBrains Mono',monospace; font-size:.68rem; letter-spacing:.2em;
  text-transform:uppercase; color:#2ED37F !important; margin:18px 0 2px;
}
.sb-brand{
  font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:1.45rem;
  color:#F4F8FB !important; line-height:1.1;
}
.sb-sub{font-size:.78rem; color:#8D9BA9 !important; margin-top:2px;}

/* ---- controls ----------------------------------------------------------- */
.stButton>button{
  font-family:'Inter Tight',sans-serif; font-weight:600; border-radius:12px;
  padding:.6rem 1rem; transition:transform .15s ease, box-shadow .15s ease;
}
.stButton>button[kind="primary"]{
  background:var(--lane); border:none; color:#fff;
  box-shadow:0 8px 20px rgba(18,165,95,.34);
}
.stButton>button[kind="primary"]:hover{transform:translateY(-2px);}
.stButton>button:focus-visible{outline:3px solid var(--caution); outline-offset:2px;}

div[data-testid="stMetric"]{
  background:rgba(255,255,255,.82); border:1px solid rgba(255,255,255,.9);
  border-radius:15px; padding:14px 16px; box-shadow:0 8px 22px rgba(16,32,64,.07);
}
div[data-testid="stMetricLabel"] *{
  font-family:'JetBrains Mono',monospace !important; font-size:.7rem !important;
  letter-spacing:.13em; text-transform:uppercase; color:var(--muted) !important;
}
div[data-testid="stMetricValue"]{
  font-family:'Bricolage Grotesque',sans-serif !important; font-weight:800 !important;
}
div[data-testid="stDataFrame"]{
  border-radius:14px; overflow:hidden; box-shadow:0 8px 22px rgba(16,32,64,.08);
}

/* Tabs. The active underline is painted with the theme's primaryColor, so the
   reliable fix lives in .streamlit/config.toml. These rules cover the label
   colours and act as a fallback across Streamlit's internal markup. */
.stTabs [data-baseweb="tab-list"]{gap:6px;}
.stTabs [data-baseweb="tab"]{
  border-radius:11px 11px 0 0; padding:9px 18px; background:transparent;
}
.stTabs [data-baseweb="tab"] p{
  font-family:'Inter Tight',sans-serif !important;
  font-size:.95rem !important; font-weight:600 !important;
  color:var(--muted) !important; margin:0 !important;
}
.stTabs [data-baseweb="tab"]:hover p{color:var(--ink) !important;}
.stTabs [aria-selected="true"]{background:rgba(255,255,255,.78);}
.stTabs [aria-selected="true"] p{color:var(--ink) !important;}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-testid="stTabsHighlight"]{background-color:var(--lane) !important;}
.stTabs [data-baseweb="tab-border"],
.stTabs [data-testid="stTabsBorder"]{background-color:var(--hair) !important;}

iframe{border-radius:16px; box-shadow:0 10px 26px rgba(16,32,64,.12);}
#MainMenu, footer{visibility:hidden;}

@media (max-width:820px){ .hero{padding:22px 20px;} .lane .big{font-size:1.7rem;} }
@media (prefers-reduced-motion:reduce){ *{transition:none !important; animation:none !important;} }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


ROUTE_STYLES = {
    "fastest": ("Fastest", "#2B3440"),
    "historical": ("Historical GIS risk", "#E8A317"),
    "ml": ("ML safest", "#12A55F"),
}

SAFETY_WORDS = {
    1: "Distance first — risk barely considered",
    3: "Slight preference for calmer streets",
    5: "Balanced — a small detour is acceptable",
    7: "Safety first — clear detours to avoid risk",
    9: "Avoid risk at almost any distance cost",
}


def safety_label(v: int) -> str:
    key = min(SAFETY_WORDS, key=lambda k: abs(k - v))
    return SAFETY_WORDS[key]


def num(d: dict, key: str, default: float = 0.0) -> float:
    """Read one RouteSummary field as a float."""
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        return default


def as_html_block(text: str) -> str:
    """Escape text and keep its line breaks and indentation.

    st.markdown still runs the markdown parser inside an HTML block, so a line
    starting with '- ' turns into a bullet and the monospace alignment breaks.
    Converting newlines to <br> first keeps every character where it was.
    """
    escaped = html.escape(str(text))
    lines = []
    for line in escaped.split("\n"):
        stripped = line.lstrip(" ")
        indent = "&nbsp;" * (len(line) - len(stripped))
        lines.append(indent + stripped)
    return "<br>".join(lines)


def show_map(m, key="route_map"):
    if HAS_STREAMLIT_FOLIUM:
        st_folium(m, width=None, height=560, key=key, returned_objects=[])
    else:
        components.html(m._repr_html_(), height=560)


# =============================================================================
# Data loading
# =============================================================================

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
        return pd.read_csv(CLEAN_ACCIDENT_FILE)
    return None


@st.cache_data
def load_metrics():
    if METRICS_FILE.exists():
        return json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    return None


for k, v in {
    "route_result": None,
    "route_map": None,
    "route_error": None,
    "run_example": False,
}.items():
    st.session_state.setdefault(k, v)


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown(
        '<div class="sb-brand">🚲 2W1C</div>'
        '<div class="sb-sub">Two wheels, one city — Berlin</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-head">Trip</div>', unsafe_allow_html=True)
    start_address = st.text_input(
        "Start address",
        value="Alexanderplatz, Berlin, Germany",
    )
    destination_address = st.text_input(
        "Destination address",
        value="Brandenburg Gate, Berlin, Germany",
    )

    st.markdown('<div class="sb-head">Priority</div>', unsafe_allow_html=True)
    safety_preference = st.slider(
        "Safety preference", 1, 10, 7,
        help="Higher values make the engine detour further around predicted-risk segments.",
    )
    st.caption(safety_label(safety_preference))

    st.markdown('<div class="sb-head">When you ride</div>', unsafe_allow_html=True)
    weekday_map = {"Monday": 2, "Tuesday": 3, "Wednesday": 4, "Thursday": 5,
                   "Friday": 6, "Saturday": 7, "Sunday": 1}
    weekday_name = st.selectbox("Weekday", list(weekday_map.keys()), index=1)
    day_of_week = weekday_map[weekday_name]

    c1, c2 = st.columns(2)
    month = c1.slider("Month", 1, 12, 7)
    hour = c2.slider("Hour", 0, 23, 8)

    with st.expander("Does the time change the route?"):
        st.write(
            "Only when the model's predictions differ enough across time windows. "
            "The engine ranks segment risk by percentile, so month and hour shift the "
            "ML route whenever those rankings move."
        )

    st.markdown("")
    run_button = st.button("Compare routes", type="primary", use_container_width=True)


# =============================================================================
# Hero
# =============================================================================

st.markdown(
    f"""
    <div class="hero">
      <div class="eyebrow">Berlin &middot; ML + GIS route engine</div>
      <div class="hero-title">🚲 <span class="accent">2W1C</span>: Integrated Bicycle Safety Routing</div>
      <p class="lede">Enter two addresses and 2W1C builds the shortest route, the route that
         avoids historical accident hotspots, and the route that avoids segments its model
         predicts to be risky — then shows what the safety costs in kilometres.</p>
      <div class="chips">
        <span class="chip"><span class="dot" style="background:{ROUTE_STYLES['fastest'][1]}"></span><b>Fastest</b> &middot; distance only</span>
        <span class="chip"><span class="dot" style="background:{ROUTE_STYLES['historical'][1]}"></span><b>Historical</b> &middot; severity-weighted GIS risk</span>
        <span class="chip"><span class="dot" style="background:{ROUTE_STYLES['ml'][1]}"></span><b>ML safest</b> &middot; predicted segment risk</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Run
# =============================================================================

if run_button or st.session_state.run_example:
    st.session_state.run_example = False
    try:
        with st.spinner("Building three routes across Berlin…"):
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


# =============================================================================
# Tabs
# =============================================================================

tab1, tab2, tab3 = st.tabs([
    "🗺️ Route comparison",
    "📊 Accident data",
    "🤖 Model",
])


def lane_card(
    kind: str,
    distance_km: float,
    caption: str,
    rows: list[tuple[str, str]],
    reduction_pct: float | None = None,
):
    """One route rendered as a lane: distance, detour caption, risk rows."""
    name, colour = ROUTE_STYLES[kind]

    bar = ""
    if reduction_pct is not None:
        # A positive reduction means less risk than the fastest route, so it is
        # shown with a minus. Printing '+53.7%' read as if risk had gone up.
        if reduction_pct >= 0:
            value, fill = f"−{reduction_pct:.1f}%", colour
        else:
            value, fill = f"+{abs(reduction_pct):.1f}%", "#DC3B32"
        width = max(3.0, min(100.0, abs(reduction_pct)))
        bar = (
            f'<div class="bar"><span style="width:{width:.0f}%;background:{fill}"></span></div>'
            f'<div class="bar-cap"><span>risk vs fastest</span><span>{value}</span></div>'
        )

    kv = "".join(
        f'<div class="kv{" kv-first" if i == 0 else ""}">'
        f'<span>{html.escape(k)}</span><b>{html.escape(v)}</b></div>'
        for i, (k, v) in enumerate(rows)
    )

    st.markdown(
        f'<div class="lane" style="--c:{colour}">'
        f'<div class="tag">{name}</div>'
        f'<div class="big">{distance_km:.2f}<small>km</small></div>'
        f'<div class="delta">{html.escape(caption)}</div>{bar}{kv}</div>',
        unsafe_allow_html=True,
    )


def detour_caption(km: float, ref_km: float) -> str:
    extra = km - ref_km
    if extra > 0.005:
        return f"+{extra:.2f} km vs fastest"
    if extra < -0.005:
        return f"{extra:.2f} km vs fastest"
    return "same length as fastest"


with tab1:
    if st.session_state.route_error:
        st.error(st.session_state.route_error)

    if st.session_state.route_result is None:
        st.markdown(
            '<div class="empty"><h3>No routes yet</h3>'
            '<p>Enter a start and a destination in the sidebar, '
            'then select <b>Compare routes</b>.</p></div>',
            unsafe_allow_html=True,
        )
        _, mid, _ = st.columns([1, 1, 1])
        if mid.button("Compare routes", use_container_width=True):
            st.session_state.run_example = True
            st.rerun()
    else:
        result = st.session_state.route_result

        # Each safer route is compared against the fastest route measured under the
        # *same* risk model. The two risk columns are different scales.
        fastest_distance = result.get("fastest_distance_summary", result["fastest_summary"])
        fastest_hist = result.get("fastest_historical_summary", result["fastest_summary"])
        fastest_ml = result.get("fastest_ml_summary", result["fastest_summary"])
        historical = result["historical_summary"]
        ml = result["ml_summary"]

        ref_km = num(fastest_distance, "distance_km")
        hist_reduction = float(result.get("historical_risk_reduction_pct") or 0.0)
        ml_reduction = float(result.get("ml_risk_reduction_pct") or 0.0)

        c1, c2, c3 = st.columns(3, gap="medium")

        with c1:
            lane_card(
                "fastest",
                ref_km,
                "distance only — the baseline",
                [
                    ("historical risk", f"{num(fastest_hist, 'length_weighted_risk'):.4f}"),
                    ("ML risk", f"{num(fastest_ml, 'length_weighted_risk'):.4f}"),
                    ("segments", f"{int(num(fastest_distance, 'n_segments')):d}"),
                ],
            )

        with c2:
            lane_card(
                "historical",
                num(historical, "distance_km"),
                detour_caption(num(historical, "distance_km"), ref_km),
                [
                    ("historical risk", f"{num(historical, 'length_weighted_risk'):.4f}"),
                    ("baseline", f"{num(fastest_hist, 'length_weighted_risk'):.4f}"),
                    ("segments", f"{int(num(historical, 'n_segments')):d}"),
                ],
                reduction_pct=hist_reduction,
            )

        with c3:
            lane_card(
                "ml",
                num(ml, "distance_km"),
                detour_caption(num(ml, "distance_km"), ref_km),
                [
                    ("ML risk", f"{num(ml, 'length_weighted_risk'):.4f}"),
                    ("baseline", f"{num(fastest_ml, 'length_weighted_risk'):.4f}"),
                    ("segments", f"{int(num(ml, 'n_segments')):d}"),
                ],
                reduction_pct=ml_reduction,
            )

        scale_note = (
            "Risk is the length-weighted mean of each segment's risk percentile (0–1)."
            if result.get("risk_normalization") == "rank"
            else "Risk is the length-weighted mean of each segment's risk score (0–1)."
        )
        if result.get("uses_ml_model") is False:
            scale_note += (
                "  No trained model was loaded, so the ML route falls back to a "
                "precomputed risk column."
            )
        st.caption(scale_note)

        st.markdown("")
        left, right = st.columns([1.4, 1], gap="large")

        with left:
            st.markdown('<div class="sec">Map</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="legend">'
                f'<span><span class="dot" style="background:{ROUTE_STYLES["fastest"][1]}"></span>Fastest</span>'
                f'<span><span class="dot" style="background:{ROUTE_STYLES["historical"][1]}"></span>Historical GIS risk</span>'
                f'<span><span class="dot" style="background:{ROUTE_STYLES["ml"][1]}"></span>ML safest</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            show_map(st.session_state.route_map)

        with right:
            st.markdown('<div class="sec">Recommendation</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="rec">{as_html_block(result["recommendation_text"])}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("")
            st.markdown('<div class="sec">Side by side</div>', unsafe_allow_html=True)
            st.dataframe(route_summary_table(result), use_container_width=True)


with tab2:
    df = load_accident_summary()

    if df is None:
        st.warning("Clean accident data not found. Run the pipeline, then reload this page.")
    else:
        st.markdown('<div class="sec">Accident records behind the risk surface</div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Records", f"{df.shape[0]:,}")
        c2.metric(
            "Years covered",
            f"{int(df['year'].min())}–{int(df['year'].max())}" if "year" in df.columns else "—",
        )
        c3.metric(
            "Serious or fatal",
            f"{100 * df['serious_or_fatal'].mean():.1f}%" if "serious_or_fatal" in df.columns else "—",
        )

        st.markdown("")
        h1, h2 = st.columns(2, gap="large")
        if "hour" in df.columns:
            with h1:
                st.markdown('<div class="sec">By hour of day</div>', unsafe_allow_html=True)
                st.bar_chart(df["hour"].value_counts().sort_index(), color="#DC3B32")
        if "month" in df.columns:
            with h2:
                st.markdown('<div class="sec">By month</div>', unsafe_allow_html=True)
                st.bar_chart(df["month"].value_counts().sort_index(), color="#E8A317")

        with st.expander(f"Preview the table · {df.shape[1]} columns"):
            st.dataframe(df.head(50), use_container_width=True)


with tab3:
    metrics = load_metrics()

    if metrics is None:
        st.warning("Model metrics not found. Run the pipeline, then reload this page.")
    else:
        st.markdown('<div class="sec">How well the model ranks risk</div>',
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}" if metrics.get("roc_auc") is not None else "—")
        c2.metric("PR-AUC", f"{metrics['pr_auc']:.3f}" if metrics.get("pr_auc") is not None else "—")
        c3.metric("Recall @ top 10%", f"{metrics['top10_recall']:.3f}" if metrics.get("top10_recall") is not None else "—")
        c4.metric("Recall @ top 20%", f"{metrics['top20_recall']:.3f}" if metrics.get("top20_recall") is not None else "—")

        st.markdown("")
        st.markdown('<div class="sec">Train and test split</div>', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(
                {
                    "Split": ["Train", "Test"],
                    "Rows": [metrics.get("n_train"), metrics.get("n_test")],
                    "Positive rate": [
                        metrics.get("positive_rate_train"),
                        metrics.get("positive_rate_test"),
                    ],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("")
    st.markdown(
        '<div class="panel"><b>The learning task</b><br>'
        'Road segment + time + road features + historical GIS risk → probability of an accident.'
        '<br><br>The historical spatial-risk model plays two roles: it is the baseline the ML '
        'model has to beat, and it is one of the features the ML model learns from.</div>',
        unsafe_allow_html=True,
    )