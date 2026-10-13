import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib

st.set_page_config(page_title="India Landslide Watch", page_icon="⛰️", layout="wide")

ROOT = Path(__file__).parent
DATA_CANDIDATES = [
    ROOT / "INDIA_LANDSLIDE_FINAL_SUSCEPTIBILITY_TRAINING.csv",
    ROOT / "data" / "INDIA_LANDSLIDE_FINAL_SUSCEPTIBILITY_TRAINING.csv",
    ROOT / "INDIA_MODELLING_DATASET_WITH_VERIFIED_PREDICTORS.csv",
]
MODEL_PATH = ROOT / "model.joblib"
FEATURES_PATH = ROOT / "model_features.json"

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
.small-note {font-size: 0.88rem; color: #687385;}
</style>
""", unsafe_allow_html=True)

st.title("⛰️ India Landslide Watch")
st.caption("Research prototype · Susceptibility screening + rainfall-informed advisory")

st.warning(
    "Prototype only — not an official warning service. The susceptibility model is trained from historical "
    "event/background samples, and background points are not confirmed non-landslide locations. Rainfall "
    "advisory levels below are transparent screening rules, not validated warning thresholds. Do not use "
    "this dashboard for evacuation or life-safety decisions."
)

@st.cache_data(show_spinner=False)
def load_training_csv(path_str):
    return pd.read_csv(path_str, low_memory=False)

def find_data(uploaded):
    if uploaded is not None:
        return pd.read_csv(uploaded, low_memory=False), "Uploaded CSV"
    for p in DATA_CANDIDATES:
        if p.exists():
            return load_training_csv(str(p)), str(p.name)
    return None, None

def build_model(df):
    label_col = next((c for c in ["label", "target", "class", "landslide_label"] if c in df.columns), None)
    if label_col is None:
        raise ValueError("Could not find a label column. Expected a column named 'label'.")
    y_raw = df[label_col]
    if y_raw.dtype == object:
        y_text = y_raw.astype(str).str.strip().str.lower()
        mapping = {"1": 1, "true": 1, "landslide": 1, "event": 1, "positive": 1,
                   "0": 0, "false": 0, "background": 0, "non-landslide": 0, "negative": 0}
        y = y_text.map(mapping)
        if y.isna().any():
            y = pd.to_numeric(y_raw, errors="coerce")
    else:
        y = pd.to_numeric(y_raw, errors="coerce")
    valid = y.isin([0, 1])
    df = df.loc[valid].copy()
    y = y.loc[valid].astype(int)
    excluded = {
        label_col, "source_record_id", "sample_type", "label_interpretation",
        "spatial_block_id", "predictor_source_elevation", "predictor_source_slope_aspect",
        "predictor_source_landcover", "predictor_source_ndvi", "event_date", "date",
        "longitude", "latitude"
    }
    candidates = [c for c in df.columns if c not in excluded]
    # Use only useful predictors, excluding columns that are wholly missing or identifiers.
    candidates = [c for c in candidates if df[c].notna().sum() > 0 and
                  pd.to_numeric(df[c], errors="coerce").notna().sum() > 0]
    # Preserve landcover as categorical despite integer codes.
    if "landcover_class" in df.columns and "landcover_class" not in candidates:
        candidates.append("landcover_class")
    if not candidates:
        raise ValueError("No usable predictor columns found in the supplied training CSV.")
    X = df[candidates].copy()
    cat_cols = [c for c in candidates if c == "landcover_class" or X[c].dtype == "object"]
    num_cols = [c for c in candidates if c not in cat_cols]
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in cat_cols:
        X[c] = X[c].astype("string")
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols))
    prep = ColumnTransformer(transformers, remainder="drop")
    model = Pipeline([
        ("prep", prep),
        ("classifier", RandomForestClassifier(
            n_estimators=350, min_samples_leaf=2, class_weight="balanced",
            random_state=20261014, n_jobs=-1
        ))
    ])
    model.fit(X, y)
    return model, candidates, num_cols, cat_cols, label_col, int((y == 1).sum()), int((y == 0).sum())

@st.cache_resource(show_spinner="Loading the historical dataset and fitting the dashboard model…")
def get_model(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    return build_model(df)

with st.sidebar:
    st.header("Data & location")
    uploaded = st.file_uploader("Optional: upload your training CSV", type=["csv"])
    st.markdown("**Default expected file**")
    st.code("INDIA_LANDSLIDE_FINAL_SUSCEPTIBILITY_TRAINING.csv")
    st.caption("Place this CSV beside app.py in the GitHub repository, or upload it here for a local session.")
    lat = st.number_input("Latitude", min_value=6.0, max_value=37.5, value=11.40, step=0.05, format="%.4f")
    lon = st.number_input("Longitude", min_value=68.0, max_value=97.5, value=76.70, step=0.05, format="%.4f")
    forecast_days = st.selectbox("Forecast horizon", [1, 3, 7], index=1)

# Load data / model
training_df = None
model = None
features = []
num_cols = []
cat_cols = []
positive_n = negative_n = 0
data_source = None
try:
    if uploaded is not None:
        training_df = pd.read_csv(uploaded, low_memory=False)
        data_source = "Uploaded CSV"
        model, features, num_cols, cat_cols, label_col, positive_n, negative_n = build_model(training_df)
    else:
        path = next((p for p in DATA_CANDIDATES if p.exists()), None)
        if path is not None:
            data_source = path.name
            model, features, num_cols, cat_cols, label_col, positive_n, negative_n = get_model(str(path))
            training_df = load_training_csv(str(path))
except Exception as e:
    st.error(f"Could not load/train the model: {e}")

if model is None:
    st.info(
        "The dashboard needs your training CSV. Add "
        "`INDIA_LANDSLIDE_FINAL_SUSCEPTIBILITY_TRAINING.csv` to the repository root "
        "or upload it in the sidebar. This app intentionally does not invent model results."
    )
    st.stop()

# Fetch forecast from Open-Meteo (no API key generally required).
@st.cache_data(ttl=900, show_spinner=False)
def get_forecast(latitude, longitude, days):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude, "longitude": longitude,
        "daily": "precipitation_sum,precipitation_probability_max",
        "forecast_days": days, "timezone": "auto"
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

# Create editable point predictors using training medians/modes.
def default_value(col):
    if training_df is None or col not in training_df.columns:
        return 0.0
    s = training_df[col]
    if col == "landcover_class":
        mode = s.dropna().mode()
        return str(mode.iloc[0]) if len(mode) else "Unknown"
    vals = pd.to_numeric(s, errors="coerce").dropna()
    return float(vals.median()) if len(vals) else 0.0

st.subheader("1 · Susceptibility screening")
st.write(f"**Training source:** `{data_source}` · **Model:** Random Forest classifier · "
         f"**Training rows:** {len(training_df):,} · **Event-labelled rows:** {positive_n:,} · "
         f"**Background-labelled rows:** {negative_n:,}")
st.caption("The app fits a reproducible Random Forest to the supplied CSV when it starts. This is not automatically the same fitted model or cross-validation result reported in the research evaluation.")

with st.expander("Edit site predictors", expanded=True):
    st.caption("Enter values for the selected coordinate. Defaults are training-data medians/modes; they are not extracted automatically for this coordinate.")
    site = {}
    cols_ui = st.columns(3)
    for i, c in enumerate(features):
        default = default_value(c)
        with cols_ui[i % 3]:
            if c in cat_cols:
                choices = sorted(training_df[c].dropna().astype(str).unique().tolist()) if c in training_df else []
                if choices:
                    val = st.selectbox(c, choices, index=choices.index(default) if default in choices else 0)
                else:
                    val = st.text_input(c, value=str(default))
                site[c] = str(val)
            else:
                val = st.number_input(c, value=float(default), format="%.4f", key=f"pred_{c}")
                site[c] = float(val)

input_df = pd.DataFrame([site], columns=features)
try:
    probability = float(model.predict_proba(input_df)[0, list(model.classes_).index(1)])
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

if probability >= 0.70:
    sus_label, sus_color = "Higher relative susceptibility", "🔴"
elif probability >= 0.40:
    sus_label, sus_color = "Moderate relative susceptibility", "🟠"
else:
    sus_label, sus_color = "Lower relative susceptibility", "🟢"

m1, m2, m3 = st.columns(3)
m1.metric("Model score for event class", f"{probability:.1%}")
m2.metric("Screening category", sus_label)
m3.metric("Selected coordinate", f"{lat:.3f}, {lon:.3f}")
st.caption("Score is a model output for this case/background sampling design, not a calibrated probability that a landslide will occur.")

st.subheader("2 · Rainfall-informed advisory")
forecast = None
try:
    forecast = get_forecast(float(lat), float(lon), int(forecast_days))
    daily = forecast.get("daily", {})
    dates = daily.get("time", [])
    rain = daily.get("precipitation_sum", [])
    prob_rain = daily.get("precipitation_probability_max", [None] * len(dates))
    rain_df = pd.DataFrame({
        "Date": dates,
        "Forecast precipitation (mm/day)": rain,
        "Max precipitation probability (%)": prob_rain
    })
    rain_df["Forecast precipitation (mm/day)"] = pd.to_numeric(rain_df["Forecast precipitation (mm/day)"], errors="coerce")
    total_rain = float(rain_df["Forecast precipitation (mm/day)"].fillna(0).sum())
    max_daily = float(rain_df["Forecast precipitation (mm/day)"].fillna(0).max())
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Forecast total · {forecast_days} day(s)", f"{total_rain:.1f} mm")
    c2.metric("Highest forecast day", f"{max_daily:.1f} mm")
    c3.metric("Forecast updated", datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC"))
    st.plotly_chart(px.bar(rain_df, x="Date", y="Forecast precipitation (mm/day)", title="Open-Meteo forecast precipitation"), use_container_width=True)
    st.dataframe(rain_df, use_container_width=True, hide_index=True)

    # Explicitly provisional rainfall screening bands, not fitted/validated warning thresholds.
    if total_rain >= 100 or max_daily >= 50:
        rain_level = "Elevated rainfall screening"
        rain_color = "🔴"
        advice = "Check official IMD/local disaster-management bulletins and local slope/drainage conditions."
    elif total_rain >= 50 or max_daily >= 25:
        rain_level = "Increased rainfall screening"
        rain_color = "🟠"
        advice = "Monitor official forecasts and local conditions; this dashboard cannot confirm a landslide warning."
    else:
        rain_level = "No high-rainfall flag from these provisional rules"
        rain_color = "🟢"
        advice = "Continue to follow official forecasts; low forecast rainfall does not mean no landslide risk."
    st.markdown(f"### {rain_color} {rain_level}")
    st.write(advice)
    st.caption("Rainfall bands are provisional UI rules only. They have not been validated against event/non-event outcomes, probability of detection, false-alarm rate, or lead time.")
except Exception as e:
    st.warning(f"Forecast service could not be reached: {e}. Susceptibility screening remains available; rainfall advisory is unavailable.")

st.subheader("3 · Research model evidence")
st.write(
    "The research workflow reported spatial cross-validation ROC-AUC around 0.928 for Random Forest "
    "on recorded events versus sampled background points. That metric describes discrimination in this "
    "sampling setup; it is not proof of real-world warning accuracy."
)
st.write(
    "Before operational use, validate with independent dated events and non-event periods, event-aligned rainfall, "
    "distance-aware spatial folds, calibration, false-alarm/missed-event rates, and lead-time analysis."
)

if training_df is not None:
    with st.expander("Training data preview"):
        st.dataframe(training_df.head(20), use_container_width=True)
    if "landcover_class" in training_df.columns:
        counts = training_df["landcover_class"].astype(str).value_counts().head(12).rename_axis("Land-cover class").reset_index(name="Rows")
        st.plotly_chart(px.bar(counts, x="Land-cover class", y="Rows", title="Most frequent land-cover codes in training data"), use_container_width=True)

st.divider()
st.caption("Research prototype · Not an official warning system · Last app render: " + datetime.now().strftime("%Y-%m-%d %H:%M"))
