from pathlib import Path
from datetime import datetime, timezone
import warnings

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# ------------------------------ PAGE / THEME ------------------------------
st.set_page_config(
    page_title="India Landslide Watch",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: radial-gradient(circle at 80% 0%, rgba(38,95,94,.22), transparent 28%), #0b1220; }
.block-container { max-width: 1450px; padding-top: 1.35rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { background: #101a2b; border-right: 1px solid #26344b; }
.hero { padding: 1.6rem 1.8rem; border: 1px solid #26384b; border-radius: 22px;
        background: linear-gradient(120deg, rgba(24,54,68,.96), rgba(17,27,44,.98)); margin-bottom: 1rem; }
.hero h1 { font-family: 'Manrope', sans-serif; letter-spacing: -1.4px; font-size: clamp(2rem, 4vw, 3.2rem); margin: 0; }
.hero p { color: #b8c8d7; margin: .45rem 0 0 0; font-size: 1rem; }
.eyebrow { color: #78d6c6; text-transform: uppercase; letter-spacing: 2px; font-weight: 700; font-size: .72rem; }
.section-title { font-family: 'Manrope', sans-serif; font-size: 1.35rem; font-weight: 800; margin: 1.15rem 0 .25rem; }
.muted { color: #9caec2; font-size: .9rem; }
div[data-testid="stMetric"] { background: #121e30; border: 1px solid #26384b; padding: 1rem 1.1rem; border-radius: 16px; }
div[data-testid="stMetricLabel"] { color: #a8b8ca; }
div[data-testid="stMetricValue"] { font-family: 'Manrope', sans-serif; }
div.stButton > button { border-radius: 11px; min-height: 2.8rem; font-weight: 700; }
div[data-testid="stAlert"] { border-radius: 14px; }
hr { border-color: #26384b; }
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).parent
CSV_NAME = "INDIA_LANDSLIDE_FINAL_SUSCEPTIBILITY_TRAINING.csv"
CANDIDATES = [
    ROOT / CSV_NAME,
    ROOT / "data" / CSV_NAME,
    ROOT / "INDIA_MODELLING_DATASET_WITH_VERIFIED_PREDICTORS.csv",
]

st.markdown("""
<div class="hero">
  <div class="eyebrow">Geospatial intelligence · Research prototype</div>
  <h1>⛰️ India Landslide Watch</h1>
  <p>Location-based susceptibility screening with rainfall context — designed for transparent research, not as an official alert service.</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "RESEARCH PROTOTYPE — Do not use for evacuation or life-safety decisions. "
    "Sampled background points are not confirmed non-landslide locations. The model score is not a calibrated "
    "real-world event probability, and rainfall bands are provisional screening rules rather than validated thresholds."
)

# ------------------------------ DATA HELPERS ------------------------------
LABEL_NAMES = ["label", "target", "class", "landslide_label"]
DROP_COLS = {
    "source_record_id", "sample_type", "label_interpretation", "spatial_block_id",
    "predictor_source_elevation", "predictor_source_slope_aspect",
    "predictor_source_landcover", "predictor_source_ndvi",
    "event_date", "date", "longitude", "latitude"
}

def safe_label(series):
    """Return binary labels without truth-testing pd.NA."""
    if pd.api.types.is_numeric_dtype(series):
        y = pd.to_numeric(series, errors="coerce")
    else:
        s = series.astype("string").str.strip().str.lower()
        mapping = {
            "1": 1, "true": 1, "landslide": 1, "event": 1, "positive": 1,
            "0": 0, "false": 0, "background": 0, "non-landslide": 0, "negative": 0,
        }
        y = s.map(mapping)
        # Try numeric conversion only for values not handled by the map.
        y = y.where(y.notna(), pd.to_numeric(s, errors="coerce"))
    return y

def make_model(df):
    label_col = next((c for c in LABEL_NAMES if c in df.columns), None)
    if label_col is None:
        raise ValueError("No binary label column found. Expected 'label' in the training CSV.")

    y_all = safe_label(df[label_col])
    valid = y_all.notna() & y_all.isin([0, 1])
    clean = df.loc[valid].copy()
    y = y_all.loc[valid].astype(int)

    if y.nunique() != 2:
        raise ValueError(f"Training data must contain both classes 0 and 1. Found labels: {sorted(y.unique().tolist())}")

    # Keep numeric predictor columns and land-cover categorical. Exclude IDs, labels and coordinates.
    candidates = []
    for c in clean.columns:
        if c == label_col or c in DROP_COLS:
            continue
        if c == "landcover_class":
            if clean[c].notna().any():
                candidates.append(c)
        elif pd.to_numeric(clean[c], errors="coerce").notna().any():
            candidates.append(c)

    if not candidates:
        raise ValueError("No usable predictor columns were found.")

    X = clean[candidates].copy()
    cat_cols = [c for c in candidates if c == "landcover_class" or X[c].dtype == "object" or str(X[c].dtype).startswith("string")]
    num_cols = [c for c in candidates if c not in cat_cols]

    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].replace([np.inf, -np.inf], np.nan)
    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("Unknown").astype(str)

    transformers = []
    if num_cols:
        transformers.append(("numeric", SimpleImputer(strategy="median", keep_empty_features=True), num_cols))
    if cat_cols:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols))
    prep = ColumnTransformer(transformers, remainder="drop")
    model = Pipeline([
        ("prep", prep),
        ("classifier", RandomForestClassifier(
            n_estimators=350, min_samples_leaf=2, class_weight="balanced",
            random_state=20261014, n_jobs=-1
        )),
    ])
    model.fit(X, y)
    return model, candidates, num_cols, cat_cols, label_col, clean, y

@st.cache_data(show_spinner=False)
def load_csv_bytes(file_bytes):
    import io
    return pd.read_csv(io.BytesIO(file_bytes), low_memory=False)

@st.cache_data(show_spinner=False)
def read_local_csv(path):
    return pd.read_csv(path, low_memory=False)

@st.cache_data(ttl=900, show_spinner=False)
def fetch_forecast(lat, lon, days):
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_sum,precipitation_probability_max",
            "forecast_days": days,
            "timezone": "auto",
        },
        timeout=20,
    )
    response.raise_for_status()
    return response.json()

# ------------------------------ SIDEBAR ------------------------------
with st.sidebar:
    st.markdown("## Control centre")
    st.caption("Configure a site and load the training data.")
    uploaded = st.file_uploader("Upload training CSV", type=["csv"])
    st.markdown("**Expected repository file**")
    st.code(CSV_NAME, language=None)
    st.caption("Upload the file to the GitHub repo beside app.py, or use the uploader above.")
    st.divider()
    st.markdown("### Assessment location")
    lat = st.number_input("Latitude", min_value=6.0, max_value=37.5, value=11.40, step=0.01, format="%.4f")
    lon = st.number_input("Longitude", min_value=68.0, max_value=97.5, value=76.70, step=0.01, format="%.4f")
    horizon = st.selectbox("Rainfall forecast horizon", [1, 3, 7], index=1)
    st.divider()
    st.markdown("<div class='muted'>India-wide coordinates are accepted. Predictor values are not auto-extracted from a nationwide raster in this version.</div>", unsafe_allow_html=True)

# ------------------------------ LOAD / FIT ------------------------------
df = None
source = None
try:
    if uploaded is not None:
        df = load_csv_bytes(uploaded.getvalue())
        source = "Uploaded CSV"
    else:
        local = next((p for p in CANDIDATES if p.exists()), None)
        if local:
            df = read_local_csv(str(local))
            source = local.name
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

if df is None:
    st.info(
        f"**Training CSV required.** Add `{CSV_NAME}` to the GitHub repository root or upload it in the left sidebar. "
        "The app will not invent a dataset or display a fabricated prediction."
    )
    st.stop()

try:
    with st.spinner("Preparing data and fitting the dashboard model…"):
        model, features, num_cols, cat_cols, label_col, clean_df, y = make_model(df)
except Exception as exc:
    st.error(f"**Model setup failed:** {exc}")
    st.markdown(
        "Check that the CSV contains a binary `label` column with both `1` (recorded event) and `0` "
        "(sampled background), plus predictor columns such as elevation, slope, NDVI and land-cover class."
    )
    st.stop()

# ------------------------------ TOPLINE ------------------------------
st.markdown("<div class='section-title'>Overview</div>", unsafe_allow_html=True)
st.markdown(f"<div class='muted'>Source: {source} · Model fitted on app startup · Label field: `{label_col}`</div>", unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Usable training rows", f"{len(clean_df):,}")
k2.metric("Recorded-event labels", f"{int((y == 1).sum()):,}")
k3.metric("Sampled-background labels", f"{int((y == 0).sum()):,}")
k4.metric("Predictor fields", f"{len(features)}")

st.markdown("<div class='section-title'>1 · Site susceptibility screening</div>", unsafe_allow_html=True)
st.markdown("<div class='muted'>Edit the environmental predictors for the selected site. Initial values use dataset medians/modes and are not coordinate-specific measurements.</div>", unsafe_allow_html=True)

def default_value(c):
    if c == "landcover_class":
        modes = clean_df[c].dropna().astype(str).mode()
        return modes.iloc[0] if len(modes) else "Unknown"
    vals = pd.to_numeric(clean_df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.median()) if len(vals) else 0.0

with st.expander("Environmental predictor inputs", expanded=True):
    site = {}
    ui_cols = st.columns(3)
    for i, c in enumerate(features):
        with ui_cols[i % 3]:
            if c in cat_cols:
                options = sorted(clean_df[c].dropna().astype(str).unique().tolist())
                default = str(default_value(c))
                if options:
                    site[c] = st.selectbox(c, options, index=options.index(default) if default in options else 0, key=f"cat_{c}")
                else:
                    site[c] = st.text_input(c, value=default, key=f"cattext_{c}")
            else:
                site[c] = st.number_input(c, value=float(default_value(c)), format="%.4f", key=f"num_{c}")

if st.button("Run site screening", type="primary", use_container_width=True):
    st.session_state["site_inputs"] = site
    st.session_state["site_probability"] = None

site = st.session_state.get("site_inputs", site)
input_df = pd.DataFrame([{c: site[c] for c in features}], columns=features)
for c in num_cols:
    input_df[c] = pd.to_numeric(input_df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
for c in cat_cols:
    input_df[c] = input_df[c].astype("string").fillna("Unknown").astype(str)

try:
    probs = model.predict_proba(input_df)[0]
    class_index = list(model.classes_).index(1)
    probability = float(probs[class_index])
except Exception as exc:
    st.error(f"Prediction failed for the selected predictors: {exc}")
    st.stop()

if probability >= 0.70:
    level, emoji = "Higher relative susceptibility", "🔴"
    description = "This model score is in the higher screening band; it is not a warning that a landslide will occur."
elif probability >= 0.40:
    level, emoji = "Moderate relative susceptibility", "🟠"
    description = "This model score is in the middle screening band; confirm local conditions and official information."
else:
    level, emoji = "Lower relative susceptibility", "🟢"
    description = "This model score is in the lower screening band; it does not rule out a landslide."
a, b, c = st.columns([1, 1, 1.4])
a.metric("Event-class model score", f"{probability:.1%}")
b.metric("Screening band", level)
c.metric("Coordinate", f"{lat:.4f}, {lon:.4f}")
st.progress(min(max(probability, 0.0), 1.0), text="Relative model score — not a calibrated event probability")
st.info(f"{emoji} {description}")

# ------------------------------ RAINFALL ------------------------------
st.markdown("<div class='section-title'>2 · Forecast rainfall context</div>", unsafe_allow_html=True)
st.markdown("<div class='muted'>Forecast from Open-Meteo. Rainfall is displayed alongside susceptibility; it is not fed into this classifier.</div>", unsafe_allow_html=True)

try:
    forecast = fetch_forecast(float(lat), float(lon), int(horizon))
    daily = forecast.get("daily", {})
    dates = daily.get("time", [])
    rainfall = daily.get("precipitation_sum", [])
    rain_prob = daily.get("precipitation_probability_max", [None] * len(dates))
    rain_df = pd.DataFrame({
        "Date": dates,
        "Forecast rainfall (mm/day)": rainfall,
        "Maximum precipitation probability (%)": rain_prob,
    })
    rain_df["Forecast rainfall (mm/day)"] = pd.to_numeric(rain_df["Forecast rainfall (mm/day)"], errors="coerce")
    total = float(rain_df["Forecast rainfall (mm/day)"].fillna(0).sum())
    max_day = float(rain_df["Forecast rainfall (mm/day)"].fillna(0).max()) if len(rain_df) else 0.0
    r1, r2, r3 = st.columns(3)
    r1.metric(f"Total forecast · {horizon} day(s)", f"{total:.1f} mm")
    r2.metric("Maximum daily rainfall", f"{max_day:.1f} mm")
    r3.metric("Forecast timezone", str(forecast.get("timezone", "local")))
    fig = px.bar(
        rain_df, x="Date", y="Forecast rainfall (mm/day)",
        title="Daily forecast precipitation",
        template="plotly_dark",
        color_discrete_sequence=["#65d6c4"],
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(rain_df, use_container_width=True, hide_index=True)

    # Transparent provisional thresholds only; not scientifically validated warning thresholds.
    if total >= 100 or max_day >= 50:
        rain_band, rain_emoji = "Elevated rainfall screening", "🔴"
        advice = "Review official IMD forecasts and local disaster-management guidance promptly."
    elif total >= 50 or max_day >= 25:
        rain_band, rain_emoji = "Increased rainfall screening", "🟠"
        advice = "Monitor official forecasts and local slope/drainage conditions."
    else:
        rain_band, rain_emoji = "No high-rainfall flag from these provisional rules", "🟢"
        advice = "Continue following official forecasts. Low forecast rainfall does not mean zero landslide risk."
    st.markdown(f"### {rain_emoji} {rain_band}")
    st.write(advice)
    st.caption("The rainfall bands are illustrative rules only. They have not been calibrated against independent landslide events, false alarms, missed events or warning lead time.")
except Exception as exc:
    st.warning(f"Rainfall forecast is temporarily unavailable: {exc}")

# ------------------------------ MODEL EVIDENCE / DATA ------------------------------
st.markdown("<div class='section-title'>3 · Research context</div>", unsafe_allow_html=True)
with st.container(border=True):
    st.markdown("**Existing research result**")
    st.markdown(
        "The earlier research workflow reported Random Forest spatial cross-validation ROC-AUC of approximately **0.928** "
        "for recorded events versus sampled background points. This is a result from that evaluation, not a metric "
        "measured by this app's newly fitted model."
    )
    st.markdown(
        "**Interpretation:** background samples are not confirmed negatives. The model score is not a calibrated real-world "
        "probability. Operational use requires independent dated events, non-event periods, spatially independent validation, "
        "calibration, false-alarm/missed-event metrics and lead-time assessment."
    )

with st.expander("Preview training records"):
    st.dataframe(df.head(30), use_container_width=True)
with st.expander("Feature names used by the app"):
    st.code("\n".join(features), language=None)

st.divider()
st.markdown(
    f"<div class='muted'>India Landslide Watch · Research prototype · Rendered {datetime.now(timezone.utc).strftime('%d %b %Y %H:%M UTC')}</div>",
    unsafe_allow_html=True,
)
