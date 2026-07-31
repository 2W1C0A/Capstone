from __future__ import annotations

import html
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
    OCCURRENCE_COMPARISON_FILE,
    OCCURRENCE_METRICS_FILE,
    ROUTE_RISK_FILE,
    SEVERITY_BY_HIGHWAY_FILE,
    SEVERITY_BY_HOUR_FILE,
    SEVERITY_METRICS_FILE,
    TEMPORAL_VALIDATION_FILE,
)
from src.route_engine import RouteEngine
from src.visualization import route_summary_table


st.set_page_config(
    page_title="2W1C · Defensible Bicycle Safety Routing",
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
  --blue:#3B82F6;
}

/* ---- canvas -------------------------------------------------------------- */
.stApp{
  background-color:var(--paper);
  background-image:
    radial-gradient(circle at 10% 16%, rgba(18,165,95,.22), transparent 42%),
    radial-gradient(circle at 86% 10%, rgba(232,163,23,.20), transparent 40%),
    radial-gradient(circle at 70% 86%, rgba(220,59,50,.16), transparent 46%),
    repeating-linear-gradient(115deg, rgba(18,22,26,.05) 0 2px, transparent 2px 27px);
  background-attachment:fixed;
  font-family:'Inter Tight',system-ui,sans-serif;
  color:var(--ink);
}
.stApp p, .stApp li, .stApp label{font-family:'Inter Tight',system-ui,sans-serif;}
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
.hero .lede{color:#AEBBC8; max-width:74ch; margin:0 0 18px; font-size:.98rem;}
.hero .chips{display:flex; gap:8px; flex-wrap:wrap;}
.hero .chip{
  font-family:'JetBrains Mono',monospace; font-size:.75rem; letter-spacing:.03em;
  padding:6px 12px; border-radius:999px;
  border:1px solid rgba(255,255,255,.22);
  background:rgba(255,255,255,.08);
  color:#D3DFEA;
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

/* ---- panels ------------------------------------------------------------- */
.sec{
  font-family:'JetBrains Mono',monospace; font-size:.7rem; letter-spacing:.18em;
  text-transform:uppercase; color:var(--muted); margin:2px 0 10px;
}
.panel{
  background:rgba(255,255,255,.80); backdrop-filter:blur(9px);
  border:1px solid rgba(255,255,255,.9); border-radius:18px; padding:20px 22px;
  box-shadow:0 10px 26px rgba(16,32,64,.08); color:var(--ink);
}
.panel b{font-weight:800;}
.rec{
  border-left:4px solid var(--lane); background:rgba(18,165,95,.07);
  border-radius:0 14px 14px 0; padding:16px 18px;
  font-family:'JetBrains Mono',monospace; font-size:.8rem; line-height:1.7;
  color:var(--ink); overflow-x:auto;
}
.warnbox{
  border-left:4px solid var(--caution); background:rgba(232,163,23,.11);
  border-radius:0 14px 14px 0; padding:15px 18px; color:var(--ink);
  font-size:.86rem; line-height:1.55;
}
.addr-ok,.addr-bad,.addr-wait{
  border-radius:12px; padding:9px 11px; margin:6px 0 10px;
  font-size:.78rem; line-height:1.45;
  border:1px solid rgba(255,255,255,.14);
}
.addr-ok{background:rgba(18,165,95,.16);}
.addr-bad{background:rgba(220,59,50,.18);}
.addr-wait{background:rgba(255,255,255,.08);}
.addr-ok b,.addr-bad b,.addr-wait b{color:#fff !important;}
.addr-small{font-family:'JetBrains Mono',monospace; font-size:.70rem; color:#AEBBC8 !important;}
.empty{
  border:1.5px dashed rgba(18,22,26,.20); border-radius:20px;
  padding:44px 30px; text-align:center; background:rgba(255,255,255,.5);
}
.empty h3{margin:0 0 6px; font-size:1.3rem;}
.empty p{color:var(--muted); margin:0 auto; max-width:50ch;}
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
    "ml": ("ML road risk", "#12A55F"),
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


def num(d: dict | None, key: str, default: float = 0.0) -> float:
    if not isinstance(d, dict):
        return default
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        return default


def pct(value: float | None) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "—"


def as_html_block(text: str) -> str:
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
    return RouteEngine()


@st.cache_data
def read_csv_if_exists(path: Path):
    if Path(path).exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def read_json_if_exists(path: Path):
    if Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return None


@st.cache_data(show_spinner=False)
def validate_address_cached(address: str) -> dict:
    """Validate an address with the same Berlin-aware geocoder used by routing."""
    return RouteEngine.validate_address_text(address)


def check_start_address() -> None:
    st.session_state.start_address_status = validate_address_cached(
        st.session_state.get("start_address_input", "")
    )


def check_destination_address() -> None:
    st.session_state.destination_address_status = validate_address_cached(
        st.session_state.get("destination_address_input", "")
    )


def render_address_status(status: dict | None) -> None:
    if not status:
        st.markdown(
            '<div class="addr-wait"><b>Not checked yet.</b><br>'
            '<span class="addr-small">Press Enter after typing the address.</span></div>',
            unsafe_allow_html=True,
        )
        return

    if status.get("ok"):
        st.markdown(
            '<div class="addr-ok"><b>Address found.</b><br>'
            f'<span class="addr-small">{html.escape(status.get("query_used", ""))}<br>'
            f'{float(status.get("lat", 0)):.5f}, {float(status.get("lon", 0)):.5f}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        msg = html.escape(str(status.get("error", "Address could not be found.")))
        st.markdown(
            '<div class="addr-bad"><b>Address not found.</b><br>'
            f'<span class="addr-small">{msg}</span></div>',
            unsafe_allow_html=True,
        )


def status_is_invalid(status: dict | None) -> bool:
    return isinstance(status, dict) and not bool(status.get("ok"))


def serious_rate(df: pd.DataFrame) -> str:
    for col in ["serious_or_fatal", "is_ksi"]:
        if col in df.columns:
            return f"{100 * pd.to_numeric(df[col], errors='coerce').fillna(0).mean():.1f}%"
    if "accident_severity" in df.columns:
        s = pd.to_numeric(df["accident_severity"], errors="coerce")
        return f"{100 * s.isin([1, 2]).mean():.1f}%"
    return "—"


for k, v in {
    "route_result": None,
    "route_map": None,
    "route_error": None,
    "run_example": False,
    "start_address_input": "Alexanderplatz, Berlin, Germany",
    "destination_address_input": "Brandenburg Gate, Berlin, Germany",
    "start_address_status": None,
    "destination_address_status": None,
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
        key="start_address_input",
        on_change=check_start_address,
        help="Press Enter after typing to validate the address.",
    )
    render_address_status(st.session_state.start_address_status)

    destination_address = st.text_input(
        "Destination address",
        key="destination_address_input",
        on_change=check_destination_address,
        help="Press Enter after typing to validate the address.",
    )
    render_address_status(st.session_state.destination_address_status)

    if st.button("Check addresses", use_container_width=True):
        check_start_address()
        check_destination_address()
        st.rerun()

    st.markdown('<div class="sb-head">Priority</div>', unsafe_allow_html=True)
    safety_preference = st.slider(
        "Safety preference",
        1,
        10,
        7,
        help="Higher values make the engine detour further around high-risk segments.",
    )
    st.caption(safety_label(safety_preference))

    st.markdown('<div class="sb-head">When you ride</div>', unsafe_allow_html=True)
    hour = st.slider(
        "Hour",
        0,
        23,
        8,
        help=(
            "In the current leakage-safe occurrence model, hour does not visibly "
            "change the route. The route ranking is mainly spatial."
        ),
    )

    with st.expander("Does the time change the route?"):
        st.write(
            "Not in the current deployed model. The earlier leakage analysis showed that "
            "time-only accident occurrence features carry almost no signal under the current "
            "negative-sampling design. Time is kept here for transparency and future "
            "severity modelling."
        )

    invalid_address = (
        status_is_invalid(st.session_state.start_address_status)
        or status_is_invalid(st.session_state.destination_address_status)
    )

    st.markdown("")
    run_button = st.button(
        "Compare routes",
        type="primary",
        use_container_width=True,
        disabled=invalid_address,
        help="Fix the address marked in red before comparing routes." if invalid_address else None,
    )


# =============================================================================
# Hero
# =============================================================================

st.markdown(
    f"""
    <div class="hero">
      <div class="eyebrow">Berlin &middot; defensible GIS + ML route engine</div>
      <div class="hero-title">🚲 <span class="accent">2W1C</span>: Bicycle Safety Routing</div>
      <p class="lede">Compare the shortest route, a severity-weighted historical GIS-risk
         route, and a leakage-safe ML road-risk route. The app separates spatial
         risk, ML diagnostics, and severity evidence instead of hiding model limitations.</p>
      <div class="chips">
        <span class="chip"><span class="dot" style="background:{ROUTE_STYLES['fastest'][1]}"></span><b>Fastest</b> &middot; distance only</span>
        <span class="chip"><span class="dot" style="background:{ROUTE_STYLES['historical'][1]}"></span><b>Historical</b> &middot; GIS risk baseline</span>
        <span class="chip"><span class="dot" style="background:{ROUTE_STYLES['ml'][1]}"></span><b>ML</b> &middot; leakage-safe road-only model</span>
        <span class="chip"><span class="dot" style="background:{ROUTE_STYLES['ml'][1]}"></span><b>Severity</b> &middot; evidence, not a route guarantee</span>
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

    if st.session_state.start_address_status is None:
        check_start_address()
    if st.session_state.destination_address_status is None:
        check_destination_address()

    try:
        with st.spinner("Building three routes across Berlin…"):
            engine = load_engine()
            result, route_map = engine.compare_and_map(
                start_address=st.session_state.start_address_input,
                destination_address=st.session_state.destination_address_input,
                safety_preference=safety_preference,
                hour=hour,
            )
        st.session_state.route_result = result
        st.session_state.route_map = route_map
        st.session_state.route_error = None
    except Exception as exc:
        st.session_state.route_result = None
        st.session_state.route_map = None
        st.session_state.route_error = str(exc)


# =============================================================================
# Route cards
# =============================================================================

def lane_card(
    kind: str,
    distance_km: float,
    caption: str,
    rows: list[tuple[str, str]],
    reduction_pct: float | None = None,
):
    name, colour = ROUTE_STYLES[kind]

    bar = ""
    if reduction_pct is not None:
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


# =============================================================================
# Tabs
# =============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Route comparison",
    "🤖 Leakage-safe ML",
    "🧭 Historical GIS risk",
    "⚠️ Severity evidence",
    "ℹ️ Method",
])


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
        if mid.button("Run example route", use_container_width=True):
            st.session_state.run_example = True
            st.rerun()
    else:
        result = st.session_state.route_result

        fastest_distance = result.get("fastest_distance_summary", result.get("fastest_summary", {}))
        fastest_hist = result.get("fastest_historical_summary", result.get("fastest_summary", {}))
        fastest_ml = result.get("fastest_ml_summary", result.get("fastest_summary", {}))
        historical = result.get("historical_summary", {})
        ml = result.get("ml_summary")

        ref_km = num(fastest_distance, "distance_km")
        hist_reduction = float(result.get("historical_risk_reduction_pct") or 0.0)
        ml_reduction = float(result.get("ml_risk_reduction_pct") or 0.0) if ml else None

        cols = st.columns(3, gap="medium")

        with cols[0]:
            lane_card(
                "fastest",
                ref_km,
                "distance only — the baseline",
                [
                    ("historical risk", f"{num(fastest_hist, 'length_weighted_risk'):.4f}"),
                    ("ML risk", f"{num(fastest_ml, 'length_weighted_risk'):.4f}" if ml else "—"),
                    ("segments", f"{int(num(fastest_distance, 'n_segments')):d}"),
                ],
            )

        with cols[1]:
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

        with cols[2]:
            if ml:
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
            else:
                st.markdown(
                    '<div class="lane" style="--c:#12A55F">'
                    '<div class="tag">ML road risk</div>'
                    '<div class="big">—<small>km</small></div>'
                    '<div class="delta">model file not loaded</div>'
                    '<div class="kv kv-first"><span>status</span><b>unavailable</b></div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

        st.caption(
            "Risk values are relative model scores. Historical GIS risk and ML road-risk "
            "are reported on their own scales and should not be interpreted as personal crash probabilities."
        )

        st.markdown("")
        left, right = st.columns([1.4, 1], gap="large")

        with left:
            st.markdown('<div class="sec">Map</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="legend">'
                f'<span><span class="dot" style="background:{ROUTE_STYLES["fastest"][1]}"></span>Fastest</span>'
                f'<span><span class="dot" style="background:{ROUTE_STYLES["historical"][1]}"></span>Historical GIS risk</span>'
                f'<span><span class="dot" style="background:{ROUTE_STYLES["ml"][1]}"></span>ML road risk</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            show_map(st.session_state.route_map)

        with right:
            st.markdown('<div class="sec">Recommendation</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="rec">{as_html_block(result.get("recommendation_text", ""))}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("")
            st.markdown('<div class="sec">Side by side</div>', unsafe_allow_html=True)
            st.dataframe(route_summary_table(result), use_container_width=True)


with tab2:
    comparison = read_csv_if_exists(OCCURRENCE_COMPARISON_FILE)
    metrics = read_json_if_exists(OCCURRENCE_METRICS_FILE)

    st.markdown('<div class="sec">Model comparison</div>', unsafe_allow_html=True)

    if comparison is None:
        st.warning("Model comparison not found. Run `python run_pipeline.py --demo-route`, then reload this page.")
    else:
        wanted = [
            "model", "status", "base", "pr_auc", "lift", "roc_auc",
            "top10_recall", "brier", "n_features",
        ]
        view = comparison[[c for c in wanted if c in comparison.columns]].copy()
        st.dataframe(view, use_container_width=True, hide_index=True)

        st.markdown(
            '<div class="warnbox"><b>Leakage guard:</b> the leaky diagnostic row is shown '
            'only to document the earlier target leakage. It is not a valid model for reporting performance.</div>',
            unsafe_allow_html=True,
        )

    if metrics:
        st.markdown("")
        st.markdown('<div class="sec">Deployed model</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Deployed", metrics.get("deployed_model", "—"))
        c2.metric("Leakage safe", "yes" if metrics.get("leakage_safe") else "—")
        c3.metric("Negative pool", "rideable classes")

        with st.expander("Features used by deployed model"):
            st.write(metrics.get("deployed_features", []))
            st.write("Excluded leakage features:")
            st.write(metrics.get("excluded_leaky_features", []))

    st.markdown("")
    st.markdown(
        '<div class="panel"><b>Learning task</b><br>'
        'Road segment features → relative accident-occurrence risk. The deployed model is '
        'road-only because the leakage analysis showed that time-only occurrence features '
        'currently have no useful signal under the negative-sampling design.</div>',
        unsafe_allow_html=True,
    )


with tab3:
    route_risk = read_csv_if_exists(ROUTE_RISK_FILE)
    temporal = read_json_if_exists(TEMPORAL_VALIDATION_FILE)

    st.markdown('<div class="sec">Spatial risk surface</div>', unsafe_allow_html=True)

    if route_risk is None:
        st.warning("Route-risk table not found. Run the pipeline first.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Edges", f"{len(route_risk):,}")
        if "accident_count" in route_risk.columns:
            c2.metric("Edges with crashes", f"{(route_risk['accident_count'] > 0).mean():.1%}")
        else:
            c2.metric("Edges with crashes", "—")
        if "combined_spatial_risk" in route_risk.columns:
            c3.metric("Mean spatial risk", f"{route_risk['combined_spatial_risk'].mean():.3f}")
        else:
            c3.metric("Mean spatial risk", "—")

        with st.expander("Preview route-risk edge table"):
            st.dataframe(route_risk.head(50), use_container_width=True)

    if temporal:
        st.markdown("")
        st.markdown('<div class="sec">Forward validation</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Top-decile recall", pct(100 * temporal.get("top_decile_recall", 0)))
        c2.metric("Random expectation", pct(100 * temporal.get("random_expectation", 0.10)))
        c3.metric("Lift over random", f"{temporal.get('lift_over_random', 0):.2f}×")
        with st.expander("Temporal validation JSON"):
            st.json(temporal)

    st.markdown("")
    st.markdown(
        '<div class="panel"><b>GIS baseline</b><br>'
        'Accidents are snapped to OSM bike-network segments, then converted into a '
        'severity-weighted historical risk surface with a junction component. This is a '
        'baseline risk surface, not an absolute personal danger probability.</div>',
        unsafe_allow_html=True,
    )


with tab4:
    sev_metrics = read_json_if_exists(SEVERITY_METRICS_FILE)
    by_hour = read_csv_if_exists(SEVERITY_BY_HOUR_FILE)
    by_highway = read_csv_if_exists(SEVERITY_BY_HIGHWAY_FILE)

    st.markdown('<div class="sec">Severity model</div>', unsafe_allow_html=True)

    if sev_metrics:
        c1, c2, c3 = st.columns(3)
        c1.metric("Target", sev_metrics.get("target", "is_ksi"))
        c2.metric("Train years", sev_metrics.get("train_years", "—"))
        c3.metric("Test years", sev_metrics.get("test_years", "—"))
        st.dataframe(pd.DataFrame(sev_metrics.get("results", [])), use_container_width=True, hide_index=True)
    else:
        st.warning("Severity metrics not found. Run the pipeline first.")

    st.markdown(
        '<div class="warnbox"><b>Interpretation:</b> severity is conditional on a crash occurring. '
        'It supports discussion of KSI severity but should not be presented as a strong route-changing model.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    h1, h2 = st.columns(2, gap="large")
    with h1:
        st.markdown('<div class="sec">KSI by hour</div>', unsafe_allow_html=True)
        if by_hour is not None:
            st.dataframe(by_hour, use_container_width=True, hide_index=True)
        else:
            st.info("No severity_by_hour.csv yet.")
    with h2:
        st.markdown('<div class="sec">KSI by highway class</div>', unsafe_allow_html=True)
        if by_highway is not None:
            st.dataframe(by_highway, use_container_width=True, hide_index=True)
        else:
            st.info("No severity_by_highway.csv yet.")


with tab5:
    df = read_csv_if_exists(CLEAN_ACCIDENT_FILE)

    st.markdown('<div class="sec">Accident data</div>', unsafe_allow_html=True)
    if df is None:
        st.warning("Clean accident data not found.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Records", f"{df.shape[0]:,}")
        c2.metric("Years", f"{int(df['year'].min())}–{int(df['year'].max())}" if "year" in df.columns else "—")
        c3.metric("KSI rate", serious_rate(df))

        h1, h2 = st.columns(2, gap="large")
        if "hour" in df.columns:
            with h1:
                st.markdown('<div class="sec">Accidents by hour</div>', unsafe_allow_html=True)
                st.bar_chart(df["hour"].value_counts().sort_index(), color="#DC3B32")
        if "month" in df.columns:
            with h2:
                st.markdown('<div class="sec">Accidents by month</div>', unsafe_allow_html=True)
                st.bar_chart(df["month"].value_counts().sort_index(), color="#E8A317")

        with st.expander(f"Preview clean table · {df.shape[1]} columns"):
            st.dataframe(df.head(50), use_container_width=True)

    st.markdown("")
    st.markdown(
        '<div class="panel"><b>Limitations</b><br>'
        'Unfallatlas has crashes but not bicycle exposure counts. Therefore the app reports '
        'relative model scores, not personal crash probabilities. The hour slider is retained '
        'for transparency, but the current deployed occurrence model is mainly spatial.</div>',
        unsafe_allow_html=True,
    )
