"""
Sales Forecasting & Demand Intelligence Dashboard
==================================================
A 4-page Streamlit app built on top of the Superstore sales analysis:

Page 1 - Sales Overview Dashboard
Page 2 - Forecast Explorer (XGBoost - best performing model)
Page 3 - Anomaly Report (Isolation Forest)
Page 4 - Product Demand Segments (K-Means clustering)

Run with:  streamlit run app.py

"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sales Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

RANDOM_STATE = 42

COLORS = {
    "bg": "#0A0B1E",
    "bg2": "#12142E",
    "panel": "rgba(23, 25, 54, 0.62)",
    "panel_border": "rgba(255, 255, 255, 0.09)",
    "violet": "#7B5CFA",
    "teal": "#22D3EE",
    "pink": "#FF3D71",
    "green": "#2FE6B8",
    "amber": "#FFB648",
    "blue": "#4C8DFF",
    "text": "#F5F6FD",
    "muted": "#9298C4",
}

PLOTLY_COLORWAY = [COLORS["pink"], COLORS["teal"], COLORS["violet"],
                   COLORS["amber"], COLORS["green"], COLORS["blue"]]


def inject_custom_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

    :root {{
        --bg: {COLORS['bg']};
        --bg2: {COLORS['bg2']};
        --panel: {COLORS['panel']};
        --panel-border: {COLORS['panel_border']};
        --violet: {COLORS['violet']};
        --teal: {COLORS['teal']};
        --pink: {COLORS['pink']};
        --green: {COLORS['green']};
        --amber: {COLORS['amber']};
        --blue: {COLORS['blue']};
        --text: {COLORS['text']};
        --muted: {COLORS['muted']};
    }}

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        font-size: 16px;
    }}

    /* ---------- Background ---------- */
    .stApp {{
        background: var(--bg);
        position: relative;
        z-index: 0;
        overflow-x: hidden;
    }}
    .stApp::before {{
        content: "";
        position: fixed;
        inset: -20%;
        z-index: -1;
        pointer-events: none;
        background:
            radial-gradient(36% 30% at 10% 15%, rgba(123,92,250,0.30), transparent 60%),
            radial-gradient(30% 26% at 88% 10%, rgba(34,211,238,0.22), transparent 60%),
            radial-gradient(38% 34% at 80% 85%, rgba(255,61,113,0.18), transparent 60%),
            radial-gradient(28% 28% at 15% 88%, rgba(47,230,184,0.16), transparent 60%);
        animation: auroraDrift 26s ease-in-out infinite alternate;
        filter: blur(10px);
    }}
    @keyframes auroraDrift {{
        0%   {{ transform: translate3d(0,0,0) scale(1); }}
        50%  {{ transform: translate3d(2%, -2%, 0) scale(1.05); }}
        100% {{ transform: translate3d(-3%, 3%, 0) scale(1.02); }}
    }}

    [data-testid="stAppViewContainer"] {{ position: relative; z-index: 1; background: transparent; }}
    [data-testid="stMain"] {{ position: relative; z-index: 1; background: transparent; }}
    [data-testid="stSidebar"] {{ position: relative; z-index: 2; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    .block-container {{ padding-top: 1.6rem; max-width: 1400px; }}

    /* ---------- Typography ---------- */
    h1, h2, h3 {{
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.01em;
    }}
    h1 {{ color: var(--text) !important; animation: fadeInUp 0.7s ease both; font-size: 2rem !important; }}
    h2 {{ font-size: 1.35rem !important; }}
    h3 {{ font-size: 1.15rem !important; color: var(--text) !important; }}
    p, span, label, .stMarkdown, .stCaption {{ color: var(--text); font-size: 1rem; }}

    /* ---------- Entrance animation ---------- */
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(14px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"],
    [data-testid="stVerticalBlockBorderWrapper"] {{
        animation: fadeInUp 0.55s cubic-bezier(.2,.7,.3,1) both;
    }}

    /* ---------- Glass cards ---------- */
    [data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {{
        background: var(--panel) !important;
        backdrop-filter: blur(18px) saturate(150%);
        -webkit-backdrop-filter: blur(18px) saturate(150%);
        border: 1px solid var(--panel-border) !important;
        border-radius: 20px !important;
        padding: 0.6rem 0.4rem;
        margin-bottom: 18px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.05);
        transition: transform 0.35s ease, box-shadow 0.35s ease, border-color 0.35s ease;
    }}
    p, span, label, .stMarkdown li {{ line-height: 1.55; }}
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]):hover {{
        transform: translateY(-3px);
        border-color: rgba(123,92,250,0.45) !important;
        box-shadow: 0 16px 40px rgba(0,0,0,0.45), 0 0 0 1px rgba(123,92,250,0.15), inset 0 1px 0 rgba(255,255,255,0.07);
    }}

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(18,20,46,0.95), rgba(10,11,30,0.97));
        border-right: 1px solid var(--panel-border);
        backdrop-filter: blur(20px);
    }}
    [data-testid="stSidebar"] h1 {{ font-size: 1.35rem !important; }}

    [data-testid="stSidebar"] [role="radiogroup"] label {{
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--panel-border);
        border-radius: 14px;
        padding: 11px 14px;
        margin-bottom: 8px;
        width: 100%;
        transition: all 0.25s ease;
        cursor: pointer;
        font-weight: 500;
    }}
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
        border-color: var(--teal);
        background: rgba(34,211,238,0.08);
        transform: translateX(3px);
    }}
    [data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] {{
        background: linear-gradient(90deg, rgba(123,92,250,0.28), rgba(34,211,238,0.16));
        border-color: var(--violet);
        box-shadow: 0 0 18px rgba(123,92,250,0.35);
    }}

    /* ---------- Buttons ---------- */
    .stButton > button, .stDownloadButton > button {{
        background: linear-gradient(90deg, var(--pink), var(--violet));
        color: #06071A;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 0.55rem 1.2rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
        box-shadow: 0 4px 18px rgba(255,61,113,0.35);
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        transform: translateY(-2px) scale(1.02);
        filter: brightness(1.08);
        box-shadow: 0 8px 26px rgba(34,211,238,0.45);
    }}

    /* ---------- Inputs ---------- */
    [data-baseweb="select"] > div, .stMultiSelect > div > div {{
        background: rgba(255,255,255,0.045) !important;
        border-radius: 12px !important;
        border: 1px solid var(--panel-border) !important;
        transition: border-color 0.25s ease;
    }}
    [data-baseweb="select"] > div:hover {{ border-color: var(--teal) !important; }}
    [data-baseweb="tag"] {{
        background: linear-gradient(90deg, var(--violet), var(--teal)) !important;
        border-radius: 8px !important;
    }}
    .stSlider [data-baseweb="slider"] div[role="slider"] {{
        box-shadow: 0 0 0 5px rgba(123,92,250,0.28);
        background: var(--teal) !important;
        transition: box-shadow 0.2s ease;
    }}
    .stSlider [data-baseweb="slider"] div[role="slider"]:hover {{
        box-shadow: 0 0 0 8px rgba(34,211,238,0.32);
    }}

    /* ---------- Toggle switches ---------- */
    [data-testid="stToggle"] div[role="switch"] {{
        background: rgba(255,255,255,0.16) !important;
        border-color: rgba(255,255,255,0.16) !important;
    }}
    [data-testid="stToggle"] div[role="switch"][aria-checked="true"] {{
        background: linear-gradient(90deg, var(--teal), var(--violet)) !important;
        border-color: var(--violet) !important;
    }}
    [data-testid="stToggle"] label p {{ font-weight: 500; }}

    /* ---------- Metrics ---------- */
    [data-testid="stMetric"] {{
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--panel-border);
        border-radius: 16px;
        padding: 14px 18px;
        transition: transform 0.25s ease, border-color 0.25s ease;
    }}
    [data-testid="stMetric"]:hover {{ transform: translateY(-2px); border-color: var(--teal); }}
    [data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', monospace !important;
        background: linear-gradient(90deg, var(--teal), var(--violet));
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent !important;
        font-weight: 700 !important;
        font-size: 1.9rem !important;
    }}
    [data-testid="stMetricLabel"] {{ color: var(--muted) !important; font-size: 0.9rem !important; }}

    /* ---------- Expander ---------- */
    [data-testid="stExpander"] {{
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--panel-border) !important;
        border-radius: 14px !important;
        overflow: hidden;
    }}

    /* ---------- Dataframes ---------- */
    [data-testid="stDataFrame"] {{ border-radius: 14px; overflow: hidden; border: 1px solid var(--panel-border); }}

    /* ---------- Alerts ---------- */
    [data-testid="stAlert"] {{
        border-radius: 14px; backdrop-filter: blur(10px);
        border: 1px solid var(--panel-border); animation: fadeInUp 0.5s ease both;
    }}
    .stCaption, [data-testid="stCaptionContainer"] {{ color: var(--muted) !important; font-size: 0.92rem !important; }}

    /* ---------- Scrollbar ---------- */
    ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
    ::-webkit-scrollbar-track {{ background: var(--bg); }}
    ::-webkit-scrollbar-thumb {{ background: linear-gradient(180deg, var(--pink), var(--violet)); border-radius: 10px; }}

    hr {{ border-color: var(--panel-border) !important; }}

    /* ---------- Top bar ---------- */
    .nova-topbar {{
        display: flex; align-items: center; justify-content: space-between;
        background: var(--panel); backdrop-filter: blur(18px) saturate(150%);
        border: 1px solid var(--panel-border); border-radius: 22px;
        padding: 16px 26px; margin-bottom: 22px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.05);
        animation: fadeInUp 0.5s ease both;
    }}
    .nova-topbar-left {{ display: flex; align-items: center; gap: 16px; }}
    .nova-badge {{
        width: 52px; height: 52px; border-radius: 16px; display: flex;
        align-items: center; justify-content: center; font-size: 1.6rem;
        background: linear-gradient(135deg, var(--pink), var(--violet));
        box-shadow: 0 6px 18px rgba(255,61,113,0.35);
        flex-shrink: 0;
    }}
    .nova-topbar-title {{
        font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.55rem;
        background: linear-gradient(90deg, var(--text) 0%, var(--teal) 60%, var(--violet) 100%);
        -webkit-background-clip: text; background-clip: text; color: transparent; line-height: 1.2;
    }}
    .nova-topbar-sub {{ color: var(--muted); font-size: 0.92rem; margin-top: 2px; }}
    .nova-topbar-icons {{ display: flex; align-items: center; gap: 12px; }}
    /* Static visual chrome (not clickable controls) — default cursor, no hover
       affordance, so they don't look like buttons that should do something. */
    .nova-icon-btn {{
        width: 42px; height: 42px; border-radius: 12px; display: flex;
        align-items: center; justify-content: center; font-size: 1.05rem;
        background: rgba(255,255,255,0.05); border: 1px solid var(--panel-border);
        position: relative; cursor: default; opacity: 0.85;
    }}
    .nova-icon-btn.dot::after {{
        content: ""; position: absolute; top: 8px; right: 9px; width: 8px; height: 8px;
        border-radius: 50%; background: var(--pink); box-shadow: 0 0 6px var(--pink);
    }}
    .nova-avatar {{
        width: 42px; height: 42px; border-radius: 50%; display: flex;
        align-items: center; justify-content: center; font-weight: 700; font-size: 0.85rem;
        background: linear-gradient(135deg, var(--teal), var(--violet)); color: #06071A;
        cursor: default;
    }}

    /* ---------- KPI tiles ---------- */
    .nova-kpi-icon {{
        width: 46px; height: 46px; border-radius: 13px; display: flex;
        align-items: center; justify-content: center; font-size: 1.3rem; flex-shrink: 0;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }}
    .nova-kpi-label {{ color: var(--muted); font-size: 0.86rem; font-weight: 500; margin-top: 10px; }}
    .nova-kpi-value {{
        font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 1.55rem;
        color: var(--text); margin-top: 2px;
    }}
    .nova-pill {{
        display: inline-block; padding: 3px 12px; border-radius: 999px;
        font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; font-weight: 700;
        color: #06071A;
    }}

    /* ---------- Gauge caption ---------- */
    .nova-gauge-label {{ text-align: center; color: var(--muted); font-size: 0.88rem; margin-top: -14px; }}
    .nova-gauge-title {{
        font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.05rem;
        color: var(--text); margin-bottom: 2px;
    }}
    .nova-gauge-desc {{ color: var(--muted); font-size: 0.85rem; line-height: 1.4; }}

    /* ---------- Progress bars ---------- */
    .nova-progress-row {{ margin-bottom: 16px; }}
    .nova-progress-top {{
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: 6px; font-size: 0.9rem;
    }}
    .nova-progress-track {{
        width: 100%; height: 10px; border-radius: 999px; background: rgba(255,255,255,0.07); overflow: hidden;
    }}
    .nova-progress-fill {{ height: 100%; border-radius: 999px; }}

    /* ---------- Legend dots ---------- */
    .nova-dot {{ width: 11px; height: 11px; border-radius: 50%; display: inline-block; margin-right: 8px; }}

    /* Respect reduced motion */
    @media (prefers-reduced-motion: reduce) {{
        .stApp::before, h1, [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"],
        [data-testid="stVerticalBlockBorderWrapper"], .nova-topbar {{ animation: none !important; }}
    }}
    </style>
    """, unsafe_allow_html=True)


def styled_fig(fig, height=440):
    existing_title = ""
    if fig.layout.title is not None and fig.layout.title.text:
        existing_title = fig.layout.title.text

    fig.update_layout(
        template="plotly_dark",
        colorway=PLOTLY_COLORWAY,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLORS["text"], size=13),
        title=dict(
            text=existing_title,
            font=dict(family="Space Grotesk, sans-serif", size=18, color=COLORS["text"]),
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.08)", borderwidth=1),
        margin=dict(t=60, l=10, r=10, b=10),
        height=height,
        transition=dict(duration=500, easing="cubic-in-out"),
        hoverlabel=dict(
            bgcolor="rgba(18,20,46,0.94)",
            bordercolor=COLORS["violet"],
            font=dict(family="JetBrains Mono, monospace", color=COLORS["text"]),
        ),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)")
    return fig


def style_gauge(fig, height=210):
    """Transparent, glass-friendly styling for go.Indicator gauge figures."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
        title=dict(text=""),
        margin=dict(t=10, b=0, l=20, r=20),
        height=height,
    )
    return fig


inject_custom_css()


def render_topbar(icon, title, subtitle):
    """Dashboard-style header bar: gradient icon badge + title on the left,
    decorative search / notification / avatar icons on the right."""
    st.markdown(
        f"""
        <div class="nova-topbar">
            <div class="nova-topbar-left">
                <div class="nova-badge">{icon}</div>
                <div>
                    <div class="nova-topbar-title">{title}</div>
                    <div class="nova-topbar-sub">{subtitle}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(icon, icon_bg, label, value, badge_text=None, badge_color=None):
    badge_html = ""
    if badge_text:
        badge_html = f'<span class="nova-pill" style="background:{badge_color};">{badge_text}</span>'
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:flex-start; padding: 4px 10px;">
            <div class="nova-kpi-icon" style="background:{icon_bg};">{icon}</div>
            {badge_html}
        </div>
        <div style="padding: 0 10px 6px 10px;">
            <div class="nova-kpi-label">{label}</div>
            <div class="nova-kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def gauge_indicator(value, max_value, color, suffix="%"):
    """Semicircular gauge dial (like the 51% / 28% dials in the reference
    design) built with a Plotly Indicator — purely a display element."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": ""},
        number={"suffix": suffix, "font": {"size": 30, "color": COLORS["text"], "family": "JetBrains Mono, monospace"}},
        gauge={
            "axis": {"range": [0, max_value], "visible": False},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(255,255,255,0.06)",
            "borderwidth": 0,
            "shape": "angular",
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    return style_gauge(fig)


def sparkline(series, color, height=60):
    """Tiny trend line with no axes — an accent under a KPI tile."""
    fig = go.Figure(go.Scatter(
        x=list(range(len(series))), y=series, mode="lines",
        line=dict(width=2.5, color=color), fill="tozeroy",
        fillcolor=color.replace(")", ", 0.15)").replace("rgb", "rgba") if color.startswith("rgb") else color + "26",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=0, b=0, l=0, r=0), height=height, showlegend=False,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def progress_row(label, percent, color, value_label=None):
    """Horizontal gradient progress bar with a label and percentage,
    matching the '55% / 76%' style rows in the reference design."""
    pct = max(0, min(100, percent))
    right_label = value_label if value_label is not None else f"{pct:.0f}%"
    st.markdown(
        f"""
        <div class="nova-progress-row">
            <div class="nova-progress-top">
                <span style="color:var(--text); font-weight:500;">{label}</span>
                <span style="color:{color}; font-family:'JetBrains Mono',monospace; font-weight:700;">{right_label}</span>
            </div>
            <div class="nova-progress-track">
                <div class="nova-progress-fill" style="width:{pct}%; background:linear-gradient(90deg,{color},rgba(255,255,255,0.4));"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# DATA LOADING & FEATURE ENGINEERING 
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and preparing data...")
def load_data(file):
    """Load the Superstore CSV and engineer date / calendar features."""
    df = pd.read_csv(file)

    # Parse dates (dayfirst, matching the source data: dd/mm/yyyy)
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    if "Ship Date" in df.columns:
        df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True, errors="coerce")

    df = df.dropna(subset=["Order Date"])

    # Fill missing postal codes if present
    if "Postal Code" in df.columns:
        df["Postal Code"] = df["Postal Code"].fillna(df["Postal Code"].mode()[0])

    # Calendar features
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = df["Order Date"].dt.quarter
    df["Week Number"] = df["Order Date"].dt.isocalendar().week
    df["Day of Week"] = df["Order Date"].dt.day_name()

    def get_season(m):
        if m in [12, 1, 2]:
            return "Winter"
        elif m in [3, 4, 5]:
            return "Spring"
        elif m in [6, 7, 8]:
            return "Summer"
        else:
            return "Autumn"

    df["Season"] = df["Month"].apply(get_season)

    if "Ship Date" in df.columns:
        df["Shipping Time"] = (df["Ship Date"] - df["Order Date"]).dt.days

    return df


def monthly_series(df, group_col=None, group_value=None):
    data = df.copy()
    if group_col is not None and group_value is not None:
        data = data[data[group_col] == group_value]
    monthly = (
        data.set_index("Order Date")
        .resample("ME")["Sales"]
        .sum()
        .rename("Sales")
        .to_frame()
    )
    return monthly


def add_lag_features(monthly):
    """Build the lag / rolling / calendar features used to train XGBoost."""
    ml = monthly.copy()
    ml["Lag1"] = ml["Sales"].shift(1)
    ml["Lag2"] = ml["Sales"].shift(2)
    ml["Lag3"] = ml["Sales"].shift(3)
    ml["RollingMean"] = ml["Sales"].rolling(3).mean()
    ml["Month"] = ml.index.month
    ml["Quarter"] = ml.index.quarter
    ml["Season"] = ((ml.index.month % 12 + 3) // 3)
    return ml.dropna()


@st.cache_resource(show_spinner="Training XGBoost model...")
def train_xgb(monthly_hash_key, ml_features):
    ml = ml_features.copy()
    if len(ml) < 6:
        return None, None, None, None

    train = ml.iloc[:-3]
    test = ml.iloc[-3:]

    X_train = train.drop("Sales", axis=1)
    y_train = train["Sales"]
    X_test = test.drop("Sales", axis=1)
    y_test = test["Sales"]

    model = XGBRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    # Re-fit on the FULL series so future forecasts use all available history
    X_full = ml.drop("Sales", axis=1)
    y_full = ml["Sales"]
    full_model = XGBRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
    )
    full_model.fit(X_full, y_full)

    return full_model, mae, rmse, (y_test, pred)


def forecast_future(model, monthly_with_lags, horizon):
    forecasts = []
    last = monthly_with_lags.iloc[-1:].copy()

    for _ in range(horizon):
        pred = model.predict(last.drop("Sales", axis=1))[0]
        forecasts.append(pred)

        new = last.copy()
        new["Sales"] = pred
        new["Lag3"] = new["Lag2"]
        new["Lag2"] = new["Lag1"]
        new["Lag1"] = pred
        new["RollingMean"] = (new["Lag1"] + new["Lag2"] + new["Lag3"]) / 3
        month = (int(last["Month"].iloc[0]) % 12) + 1
        quarter = (month - 1) // 3 + 1
        season = (month % 12 + 3) // 3
        new["Month"] = month
        new["Quarter"] = quarter
        new["Season"] = season
        last = new

    future_dates = pd.date_range(
        monthly_with_lags.index[-1] + pd.offsets.MonthEnd(), periods=horizon, freq="ME"
    )
    return pd.Series(forecasts, index=future_dates, name="Forecast")


@st.cache_data(show_spinner="Running anomaly detection...")
def detect_anomalies(df, contamination=0.05):
    """Weekly sales + rolling stats -> Isolation Forest anomaly flags."""
    weekly = (
        df.set_index("Order Date")
        .resample("W")["Sales"]
        .sum()
        .rename("Sales")
        .to_frame()
        .reset_index()
    )

    window = 8
    weekly["RollingMean"] = weekly["Sales"].rolling(window, min_periods=1).mean()
    weekly["RollingSTD"] = weekly["Sales"].rolling(window, min_periods=1).std().fillna(0)

    iso = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
    features = weekly[["Sales", "RollingMean", "RollingSTD"]]
    weekly["Anomaly"] = iso.fit_predict(features) == -1

    return weekly


@st.cache_data(show_spinner="Building product demand segments...")
def build_clusters(df, n_clusters=4):
    total_sales = df.groupby("Sub-Category")["Sales"].sum()

    yearly = df.groupby(["Sub-Category", "Year"])["Sales"].sum().reset_index()
    yearly["Growth"] = yearly.groupby("Sub-Category")["Sales"].pct_change()
    growth_rate = yearly.groupby("Sub-Category")["Growth"].mean()

    monthly_sc = df.groupby(["Sub-Category", "Month"])["Sales"].sum().reset_index()
    volatility = monthly_sc.groupby("Sub-Category")["Sales"].std()

    avg_order = df.groupby("Sub-Category")["Sales"].mean()

    features = pd.DataFrame(
        {
            "Total_Sales": total_sales,
            "Growth_Rate": growth_rate,
            "Volatility": volatility,
            "Avg_Order_Value": avg_order,
        }
    ).fillna(0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    features["Cluster"] = kmeans.fit_predict(scaled)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled)
    features["PC1"] = coords[:, 0]
    features["PC2"] = coords[:, 1]

    return features


def label_clusters(features):
    summary = features.groupby("Cluster")[
        ["Total_Sales", "Growth_Rate", "Volatility", "Avg_Order_Value"]
    ].mean()

    labels = {}
    remaining = list(summary.index)

    # 1) Highest growth rate, if meaningfully above the rest -> Growing Demand
    growth_leader = summary["Growth_Rate"].idxmax()
    if summary["Growth_Rate"].max() > summary["Growth_Rate"].median() * 1.5:
        labels[growth_leader] = "Growing Demand"
        remaining.remove(growth_leader)

    # 2) Highest total sales among what's left -> High Volume, Stable Demand
    if remaining:
        sales_leader = summary.loc[remaining, "Total_Sales"].idxmax()
        labels[sales_leader] = "High Volume, Stable Demand"
        remaining.remove(sales_leader)

    # 3) Highest volatility-to-sales ratio among what's left -> Low Volume, High Volatility
    if remaining:
        ratio = summary.loc[remaining, "Volatility"] / summary.loc[remaining, "Total_Sales"].replace(0, np.nan)
        volatile_leader = ratio.idxmax()
        labels[volatile_leader] = "Low Volume, High Volatility"
        remaining.remove(volatile_leader)

    # 4) Whatever is left -> Declining / Low Demand
    for c in remaining:
        labels[c] = "Declining / Low Demand"

    return labels, summary

st.sidebar.markdown(
    """
    <div style="padding:6px 2px 14px 2px;">
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="width:38px; height:38px; border-radius:11px; display:flex; align-items:center;
                        justify-content:center; font-size:1.2rem; background:linear-gradient(135deg,#FF3D71,#7B5CFA);
                        box-shadow:0 4px 14px rgba(255,61,113,0.35);">📊</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1.25rem; font-weight:700;
                        background:linear-gradient(90deg,#F5F6FD,#22D3EE 60%,#7B5CFA);
                        -webkit-background-clip:text; background-clip:text; color:transparent;">
                Sales Intelligence
            </div>
        </div>
        <div style="color:#9298C4; font-size:0.8rem; margin-top:6px;">
            Forecasting · Anomalies · Demand Segments
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.sidebar.file_uploader("Upload Superstore CSV (train.csv)", type=["csv"])

if uploaded_file is None:
    st.markdown(
        """
        <div style="text-align:center; padding: 10vh 1rem 1rem 1rem;">
            <div style="font-size:3.2rem;">📊</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("""
<div style="
background: linear-gradient(90deg,#4F46E5,#06B6D4);
padding:20px;
border-radius:15px;
text-align:center;
margin-bottom:20px;">

<h1 style="color:white;margin:0;">
📊 Sales Forecasting Dashboard
</h1>

<p style="color:white;font-size:18px;">
Forecasting · Anomalies · Demand Segments
</p>

</div>
""", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:#9298C4; font-size:1.05rem; margin-top:-0.6rem;'>"
        "Forecasts, anomalies, and demand segments rendered as a dashboard.\n\n"
        "BY MANSI"
        "</p>",
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        with st.container(border=True):
            st.info(
                "Upload your Superstore train.csv file using the sidebar to get started.\n\n"
                "Expected columns include: Order Date, Ship Date, Sales, Region, "
                "Category, Sub-Category, Segment, etc."
            )
    st.stop()

df = load_data(uploaded_file)

page = st.sidebar.radio(
    "Navigate",
    [
        "1️⃣ 🏠 Sales Overview Dashboard",
        "2️⃣ 🔮 Forecast Explorer",
        "3️⃣ 🚨 Anomaly Report",
        "4️⃣ 🧩 Product Demand Segments",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
                border-radius:14px; padding:12px 14px; font-family:'JetBrains Mono',monospace;">
        <div style="color:#9298C4; font-size:0.72rem; letter-spacing:0.04em; text-transform:uppercase;">Data range</div>
        <div style="color:#F5F6FD; font-size:0.85rem; margin-bottom:8px;">
            {df['Order Date'].min().date()} → {df['Order Date'].max().date()}
        </div>
        <div style="color:#9298C4; font-size:0.72rem; letter-spacing:0.04em; text-transform:uppercase;">Rows loaded</div>
        <div style="color:#22D3EE; font-size:1rem; font-weight:700;">{len(df):,}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# PAGE 1 — SALES OVERVIEW DASHBOARD
# ----------------------------------------------------------------------------
if page.startswith("1"):
    render_topbar("🏠", "Sales Overview Dashboard", "Live snapshot of revenue, orders, and regional performance")

    # --- KPI strip ---
    total_sales = df["Sales"].sum()
    total_orders = df["Order ID"].nunique() if "Order ID" in df.columns else len(df)
    avg_order = df["Sales"].mean()
    yrs = sorted(df["Year"].unique())
    if len(yrs) >= 2:
        this_yr_sales = df[df["Year"] == yrs[-1]]["Sales"].sum()
        prev_yr_sales = df[df["Year"] == yrs[-2]]["Sales"].sum()
        yoy = ((this_yr_sales - prev_yr_sales) / prev_yr_sales * 100) if prev_yr_sales else 0
        yoy_display = f"{yoy:+.1f}%"
    else:
        yoy = 0
        yoy_display = "N/A"

    yoy_badge_color = "linear-gradient(90deg,#2FE6B8,#22D3EE)" if yoy >= 0 else "linear-gradient(90deg,#FF3D71,#FFB648)"

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        with st.container(border=True):
            kpi_card("💰", "linear-gradient(135deg,#FF3D71,#7B5CFA)", "Total Sales",
                     f"${total_sales:,.0f}", "ALL TIME", "linear-gradient(90deg,#FF3D71,#7B5CFA)")
    with k2:
        with st.container(border=True):
            kpi_card("🧾", "linear-gradient(135deg,#22D3EE,#4C8DFF)", "Orders",
                     f"{total_orders:,}", "COUNT", "linear-gradient(90deg,#22D3EE,#4C8DFF)")
    with k3:
        with st.container(border=True):
            kpi_card("🛒", "linear-gradient(135deg,#2FE6B8,#22D3EE)", "Avg Order Value",
                     f"${avg_order:,.2f}", "AOV", "linear-gradient(90deg,#2FE6B8,#22D3EE)")
    with k4:
        with st.container(border=True):
            kpi_card("📈", "linear-gradient(135deg,#FFB648,#FF3D71)", "YoY Growth",
                     yoy_display, "VS LAST YR", yoy_badge_color)

    # --- Gauge dials: growth + top-region concentration ---
    by_region_all = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    top_region_share = (by_region_all.iloc[0] / by_region_all.sum() * 100) if len(by_region_all) else 0
    fulfilled_pct = min(100.0, (avg_order / (df["Sales"].max() or 1)) * 100) if len(df) else 0

    g1, g2 = st.columns(2)
    with g1:
        with st.container(border=True):
            gc1, gc2 = st.columns([1, 1.1])
            with gc1:
                st.plotly_chart(gauge_indicator(max(0, min(100, yoy if len(yrs) >= 2 else 0)), 100, COLORS["pink"]),
                                 use_container_width=True, config={"displayModeBar": False})
            with gc2:
                st.markdown(
                    f"""
                    <div style="padding-top:18px;">
                        <div class="nova-gauge-title">Year-over-Year Growth</div>
                        <div class="nova-gauge-desc">Sales growth in {yrs[-1] if yrs else '—'} compared to {yrs[-2] if len(yrs) >= 2 else '—'}, shown as a share of a 100% dial.</div>
                    </div>
                    """, unsafe_allow_html=True)
    with g2:
        with st.container(border=True):
            gc1, gc2 = st.columns([1, 1.1])
            with gc1:
                st.plotly_chart(gauge_indicator(top_region_share, 100, COLORS["violet"]),
                                 use_container_width=True, config={"displayModeBar": False})
            with gc2:
                st.markdown(
                    f"""
                    <div style="padding-top:18px;">
                        <div class="nova-gauge-title">Top Region Share</div>
                        <div class="nova-gauge-desc">{by_region_all.index[0] if len(by_region_all) else '—'} accounts for this share of total sales across all regions.</div>
                    </div>
                    """, unsafe_allow_html=True)

    # --- Display options ---
    with st.container(border=True):
        st.markdown("<div class='nova-gauge-title'>⚙️ Display Options</div>", unsafe_allow_html=True)
        opt1, opt2 = st.columns(2)
        with opt1:
            show_labels = st.toggle("Show value labels on bars", value=True)
        with opt2:
            show_raw = st.toggle("Show filtered raw data table", value=False)

    label_mode = ".2s" if show_labels else False

    # --- Total sales by year ---
    with st.container(border=True):
        st.subheader("Total Sales by Year")
        yearly_totals = df.groupby("Year")["Sales"].sum().reset_index()
        fig_year = px.bar(
            yearly_totals, x="Year", y="Sales", text_auto=label_mode,
            color="Sales", color_continuous_scale=["#22D3EE", "#7B5CFA", "#FF3D71"],
        )
        fig_year.update_layout(yaxis_title="Total Sales ($)", showlegend=False)
        fig_year.update_traces(marker_line_width=0)
        st.plotly_chart(styled_fig(fig_year), use_container_width=True)

    # --- Monthly sales trend ---
    with st.container(border=True):
        st.subheader("Monthly Sales Trend (All Years)")
        monthly_total = monthly_series(df)
        fig_month = px.area(
            monthly_total.reset_index(), x="Order Date", y="Sales", markers=True
        )
        fig_month.update_traces(
            line=dict(width=3, color=COLORS["teal"]),
            marker=dict(size=6, color=COLORS["violet"]),
            fillcolor="rgba(34,211,238,0.12)",
        )
        fig_month.update_layout(yaxis_title="Total Sales ($)")
        st.plotly_chart(styled_fig(fig_month), use_container_width=True)

    # --- Interactive filters: sales by region & category ---
    with st.container(border=True):
        st.subheader("Sales by Region & Category")

        col1, col2 = st.columns(2)
        with col1:
            regions = sorted(df["Region"].unique())
            selected_regions = st.multiselect("Filter Region(s)", regions, default=regions)
        with col2:
            categories = sorted(df["Category"].unique())
            selected_categories = st.multiselect("Filter Category(s)", categories, default=categories)

        filtered = df[df["Region"].isin(selected_regions) & df["Category"].isin(selected_categories)]

        colA, colB = st.columns(2)
        with colA:
            by_region = filtered.groupby("Region")["Sales"].sum().reset_index().sort_values("Sales", ascending=False)
            fig_region = px.bar(by_region, x="Region", y="Sales", color="Region", text_auto=label_mode,
                                 color_discrete_sequence=PLOTLY_COLORWAY)
            fig_region.update_layout(showlegend=False, yaxis_title="Total Sales ($)")
            fig_region.update_traces(marker_line_width=0)
            st.plotly_chart(styled_fig(fig_region, height=380), use_container_width=True)
        with colB:
            by_cat = filtered.groupby("Category")["Sales"].sum().reset_index().sort_values("Sales", ascending=False)
            fig_cat = px.bar(by_cat, x="Category", y="Sales", color="Category", text_auto=label_mode,
                              color_discrete_sequence=PLOTLY_COLORWAY[::-1])
            fig_cat.update_layout(showlegend=False, yaxis_title="Total Sales ($)")
            fig_cat.update_traces(marker_line_width=0)
            st.plotly_chart(styled_fig(fig_cat, height=380), use_container_width=True)

        # --- Region share progress bars ---
        st.markdown("<div class='nova-gauge-title' style='margin-top:10px;'>Region Share of Filtered Sales</div>", unsafe_allow_html=True)
        total_filtered = by_region["Sales"].sum()
        bar_colors = [COLORS["pink"], COLORS["teal"], COLORS["violet"], COLORS["amber"], COLORS["green"]]
        for i, row in by_region.iterrows():
            pct = (row["Sales"] / total_filtered * 100) if total_filtered else 0
            progress_row(row["Region"], pct, bar_colors[i % len(bar_colors)], value_label=f"{pct:.0f}%")

        if show_raw:
            st.dataframe(filtered, use_container_width=True)
        else:
            with st.expander("View filtered raw data"):
                st.dataframe(filtered, use_container_width=True)


# ----------------------------------------------------------------------------
# PAGE 2 — FORECAST EXPLORER
# ----------------------------------------------------------------------------
elif page.startswith("2"):
    render_topbar("🔮", "Forecast Explorer", "Powered by XGBoost — the best performing model from model comparison")

    with st.container(border=True):
        col1, col2, col3 = st.columns([1, 1, 1.2])

        with col1:
            dimension = st.selectbox("Select dimension", ["Category", "Region"])

        with col2:
            options = sorted(df[dimension].unique())
            selected_value = st.selectbox(f"Select {dimension}", options)

        with col3:
            horizon = st.slider("Forecast horizon (months ahead)", min_value=1, max_value=3, value=3)

    monthly = monthly_series(df, group_col=dimension, group_value=selected_value)
    ml_features = add_lag_features(monthly)

    if len(ml_features) < 6:
        st.warning("Not enough historical data for this segment to train a reliable forecast model.")
    else:
        cache_key = f"{dimension}-{selected_value}-{len(ml_features)}"
        model, mae, rmse, holdout = train_xgb(cache_key, ml_features)

        forecast_vals = forecast_future(model, ml_features, horizon=horizon)

        # --- Confidence gauge derived from holdout error (display only) ---
        avg_actual = monthly["Sales"].mean() if len(monthly) else 0
        mape = (mae / avg_actual * 100) if avg_actual else 0
        confidence = max(0, min(100, 100 - mape))

        gcol1, gcol2, gcol3 = st.columns([1, 1, 1.4])
        with gcol1:
            with st.container(border=True):
                st.plotly_chart(gauge_indicator(confidence, 100, COLORS["teal"]),
                                 use_container_width=True, config={"displayModeBar": False})
                st.markdown("<div class='nova-gauge-label'>Forecast Confidence</div>", unsafe_allow_html=True)
        with gcol2:
            with st.container(border=True):
                st.metric("MAE", f"{mae:,.2f}")
                st.metric("RMSE", f"{rmse:,.2f}")
        with gcol3:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div style="padding-top:6px;">
                        <div class="nova-gauge-title">How confidence is calculated</div>
                        <div class="nova-gauge-desc">
                        The model is evaluated on a 3-month holdout. Confidence = 100% − MAPE,
                        where MAPE is the Mean Absolute Error expressed as a percentage of average
                        monthly sales for <b>{selected_value}</b>. Higher is better.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # --- Plot actual vs forecast ---
        with st.container(border=True):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly.index, y=monthly["Sales"], mode="lines+markers", name="Actual Sales",
                line=dict(width=3, color=COLORS["teal"]),
                marker=dict(size=6, color=COLORS["teal"]),
            ))
            fig.add_trace(go.Scatter(
                x=forecast_vals.index, y=forecast_vals.values, mode="lines+markers",
                name="Forecast",
                line=dict(dash="dash", width=3, color=COLORS["pink"]),
                marker=dict(size=9, color=COLORS["pink"], symbol="diamond"),
            ))
            fig.update_layout(
                title=f"{dimension}: {selected_value} — {horizon}-Month Sales Forecast",
                xaxis_title="Month", yaxis_title="Sales ($)",
            )
            st.plotly_chart(styled_fig(fig), use_container_width=True)

        # --- Forecast table ---
        with st.container(border=True):
            st.subheader("Forecast Values")
            forecast_table = forecast_vals.reset_index()
            forecast_table.columns = ["Month", "Forecasted Sales"]
            forecast_table["Forecasted Sales"] = forecast_table["Forecasted Sales"].round(2)
            st.dataframe(forecast_table, use_container_width=True, hide_index=True)

        # --- Model performance metrics ---
        with st.container(border=True):
            st.subheader("Model Performance (3-month holdout evaluation)")
            m1, m2 = st.columns(2)
            with m1:
                st.metric("MAE", f"{mae:,.2f}")
            with m2:
                st.metric("RMSE", f"{rmse:,.2f}")
            st.caption(
                "MAE / RMSE are computed by holding out the last 3 actual months, "
                "training XGBoost on everything before that, and comparing predictions "
                "against the true values — independent of the forecast horizon chosen above."
            )


# ----------------------------------------------------------------------------
# PAGE 3 — ANOMALY REPORT
# ----------------------------------------------------------------------------
elif page.startswith("3"):
    render_topbar("🚨", "Anomaly Report", "Isolation Forest applied to weekly sales, rolling mean, and rolling volatility")

    with st.container(border=True):
        contamination = st.slider(
            "Sensitivity (expected % of anomalous weeks)", min_value=0.01, max_value=0.15, value=0.05, step=0.01
        )

    weekly = detect_anomalies(df, contamination=contamination)
    anomalies = weekly[weekly["Anomaly"]]
    anomaly_rate = (len(anomalies) / len(weekly) * 100) if len(weekly) else 0

    # --- KPI + gauge row ---
    k1, k2, k3 = st.columns(3)
    with k1:
        with st.container(border=True):
            kpi_card("📅", "linear-gradient(135deg,#4C8DFF,#7B5CFA)", "Weeks Analyzed",
                     f"{len(weekly):,}", "TOTAL", "linear-gradient(90deg,#4C8DFF,#7B5CFA)")
    with k2:
        with st.container(border=True):
            kpi_card("🚩", "linear-gradient(135deg,#FF3D71,#FFB648)", "Anomalies Found",
                     f"{len(anomalies):,}", "FLAGGED", "linear-gradient(90deg,#FF3D71,#FFB648)")
    with k3:
        with st.container(border=True):
            gc1, gc2 = st.columns([1, 1])
            with gc1:
                st.plotly_chart(gauge_indicator(anomaly_rate, 100, COLORS["pink"]),
                                 use_container_width=True, config={"displayModeBar": False})
            with gc2:
                st.markdown(
                    "<div style='padding-top:18px;'>"
                    "<div class='nova-gauge-title'>Anomaly Rate</div>"
                    "<div class='nova-gauge-desc'>Share of weeks flagged as anomalous by Isolation Forest at the current sensitivity.</div>"
                    "</div>", unsafe_allow_html=True)

    with st.container(border=True):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=weekly["Order Date"], y=weekly["Sales"], mode="lines", name="Weekly Sales",
            line=dict(width=2.5, color=COLORS["teal"]),
        ))
        fig.add_trace(go.Scatter(
            x=anomalies["Order Date"], y=anomalies["Sales"], mode="markers", name="Anomaly",
            marker=dict(color=COLORS["pink"], size=13, symbol="x",
                        line=dict(color="white", width=1)),
        ))
        fig.update_layout(
            title="Weekly Sales with Detected Anomalies",
            xaxis_title="Week", yaxis_title="Sales ($)",
        )
        st.plotly_chart(styled_fig(fig), use_container_width=True)

    with st.container(border=True):
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
                <span style="font-family:'Space Grotesk',sans-serif; font-size:1.25rem; font-weight:700; color:#F5F6FD;">
                    Detected Anomalies
                </span>
                <span style="background:linear-gradient(90deg,#FF3D71,#7B5CFA); color:#06071A; font-weight:700;
                             border-radius:999px; padding:2px 12px; font-family:'JetBrains Mono',monospace; font-size:0.85rem;">
                    {len(anomalies)} weeks
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        display_table = anomalies[["Order Date", "Sales"]].copy()
        display_table["Order Date"] = display_table["Order Date"].dt.date
        display_table["Sales"] = display_table["Sales"].round(2)
        display_table = display_table.sort_values("Order Date").reset_index(drop=True)
        st.dataframe(display_table, use_container_width=True, hide_index=True)

        csv = display_table.to_csv(index=False).encode("utf-8")
        st.download_button("Download anomaly list as CSV", csv, "anomalies.csv", "text/csv")


# ----------------------------------------------------------------------------
# PAGE 4 — PRODUCT DEMAND SEGMENTS
# ----------------------------------------------------------------------------
elif page.startswith("4"):
    render_topbar("🧩", "Product Demand Segments", "K-Means clustering on total sales, growth rate, volatility, and average order value")

    with st.container(border=True):
        n_clusters = st.slider("Number of clusters", min_value=2, max_value=6, value=4)

    if "Sub-Category" not in df.columns:
        st.error("This dataset has no 'Sub-Category' column, so demand segmentation can't be computed.")
    else:
        features = build_clusters(df, n_clusters=n_clusters)
        labels, summary = label_clusters(features)
        features["Demand Group"] = features["Cluster"].map(labels)

        # --- Cluster scatter plot ---
        with st.container(border=True):
            st.subheader("Demand Segmentation (PCA-reduced view)")
            st.caption("Point size = total sales. Hover any point for its sub-category and full stats — labels are shown on hover instead of on the chart to keep close points readable.")
            fig = px.scatter(
                features.reset_index(),
                x="PC1", y="PC2",
                color="Demand Group",
                hover_name="Sub-Category",
                hover_data={
                    "Total_Sales": ":,.0f",
                    "Growth_Rate": ":.2%",
                    "Volatility": ":,.0f",
                    "Avg_Order_Value": ":,.2f",
                },
                size="Total_Sales",
                size_max=42,
                color_discrete_sequence=PLOTLY_COLORWAY,
            )
            fig.update_traces(
                marker=dict(line=dict(width=1.5, color="rgba(255,255,255,0.55)"), opacity=0.9),
                selector=dict(mode="markers"),
            )
            fig.update_layout(
                xaxis_title="Principal Component 1",
                yaxis_title="Principal Component 2",
                legend_title_text="Demand Group",
            )
            fig.update_xaxes(automargin=True)
            fig.update_yaxes(automargin=True)
            st.plotly_chart(styled_fig(fig, height=560), use_container_width=True)

            # A readable, non-overlapping legend of which sub-categories sit in
            # each PCA cluster, since the chart itself no longer prints names.
            st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)
            legend_colors = {name: PLOTLY_COLORWAY[i % len(PLOTLY_COLORWAY)]
                              for i, name in enumerate(sorted(features["Demand Group"].unique()))}
            group_cols = st.columns(len(legend_colors))
            for col, (grp, color) in zip(group_cols, legend_colors.items()):
                members = features.reset_index().loc[features.reset_index()["Demand Group"] == grp, "Sub-Category"].tolist()
                with col:
                    st.markdown(
                        f"<span class='nova-dot' style='background:{color};'></span>"
                        f"<b style='color:var(--text);'>{grp}</b>",
                        unsafe_allow_html=True,
                    )
                    st.caption(", ".join(members))

        # --- Demand group share progress bars ---
        with st.container(border=True):
            st.markdown("<div class='nova-gauge-title'>Demand Group — Share of Total Sales</div>", unsafe_allow_html=True)
            group_sales = features.groupby("Demand Group")["Total_Sales"].sum().sort_values(ascending=False)
            total_group_sales = group_sales.sum()
            group_colors = {
                "Growing Demand": COLORS["green"],
                "High Volume, Stable Demand": COLORS["teal"],
                "Low Volume, High Volatility": COLORS["amber"],
                "Declining / Low Demand": COLORS["pink"],
            }
            for grp, val in group_sales.items():
                pct = (val / total_group_sales * 100) if total_group_sales else 0
                color = group_colors.get(grp, COLORS["violet"])
                progress_row(f"<span class='nova-dot' style='background:{color};'></span>{grp}", pct, color, value_label=f"{pct:.0f}%")

        # --- Cluster summary stats ---
        with st.container(border=True):
            st.subheader("Cluster Profile Summary")
            summary_display = summary.copy()
            summary_display["Demand Group"] = summary_display.index.map(labels)
            summary_display = summary_display[
                ["Demand Group", "Total_Sales", "Growth_Rate", "Volatility", "Avg_Order_Value"]
            ].round(2)
            st.dataframe(summary_display, use_container_width=True)

        # --- Sub-category -> cluster table ---
        with st.container(border=True):
            st.subheader("Sub-Categories by Demand Group")
            table = features.reset_index()[
                ["Sub-Category", "Demand Group", "Total_Sales", "Growth_Rate", "Volatility", "Avg_Order_Value"]
            ].sort_values(["Demand Group", "Total_Sales"], ascending=[True, False])
            table[["Total_Sales", "Growth_Rate", "Volatility", "Avg_Order_Value"]] = table[
                ["Total_Sales", "Growth_Rate", "Volatility", "Avg_Order_Value"]
            ].round(2)
            st.dataframe(table, use_container_width=True, hide_index=True)
