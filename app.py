"""Streamlit scenario simulator for the all-cities weekly mortality / morbidity model.

Run with:
    streamlit run app/streamlit_app.py

This is a what-if simulator, not a replay of the historical test set. The user
picks a target city + month and tweaks a curated set of human-readable levers
(recent caseload, climate, population/built environment, health access). The
model returns predicted `all_cause_cases` + `all_cause_deaths` counts plus a
SHAP waterfall explaining the prediction.

Calendar terms are driven entirely by the month picker; `year` is held fixed at
REFERENCE_YEAR so predictions read as "given these conditions -> expected counts"
rather than as a point on the learned year-trend. Every feature the user does not
override is filled from the per-city seasonal median computed in
03-feature-engineering.ipynb.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

REPO_ROOT = Path.cwd().resolve()
while not (REPO_ROOT / "data").exists() and REPO_ROOT.parent != REPO_ROOT:
    REPO_ROOT = REPO_ROOT.parent

DATA_DIR = REPO_ROOT / "data"
PROC_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "output" / "models"

# `year` is a strong learned driver but is not a controllable scenario lever, so it
# is hidden from the UI and held at this neutral reference (the headline pre-COVID
# evaluation year per CLAUDE.md).
REFERENCE_YEAR = 2019

# Curated, human-readable simulation levers: (feature_key, label, group).
# Every key is verified to exist in feature_schema.json; all are top model drivers
# (top_features_*.json) and/or intuitive scenario variables. Features not listed here
# are filled silently from the per-city seasonal median.
SIM_FEATURES = [
    # Recent caseload (autoregressive lag) — "given this recent load, predict next"
    ("all_cause_cases_lag4wk", "Recent weekly cases (~4 weeks prior)", "Recent caseload"),
    ("all_cause_deaths_lag4wk", "Recent weekly deaths (~4 weeks prior)", "Recent caseload"),
    # Climate & environment
    ("tave", "Average temperature (°C)", "Climate & environment"),
    ("pr_norm", "Rainfall (normalized)", "Climate & environment"),
    ("spi6", "Drought index (SPI-6)", "Climate & environment"),
    ("rh", "Relative humidity (%)", "Climate & environment"),
    ("ndvi", "Vegetation greenness (NDVI)", "Climate & environment"),
    ("pm25", "Air quality — PM2.5", "Climate & environment"),
    ("co", "Air quality — CO", "Climate & environment"),
    # Population & built environment
    ("pop_count_total", "Total population", "Population & built environment"),
    ("pop_density_max", "Peak population density", "Population & built environment"),
    ("google_bldgs_count", "Building count", "Population & built environment"),
    # Health access
    ("hospital_count", "Hospital count", "Health access"),
    ("pharmacy_count", "Pharmacy count", "Health access"),
    ("fire_station_count", "Fire station count", "Health access"),
]

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

st.set_page_config(page_title="CCHAIN morbidity / mortality simulator", layout="wide")


@st.cache_resource
def load_artifacts():
    cases_model = joblib.load(MODEL_DIR / "cases_hgbr.joblib")
    deaths_model = joblib.load(MODEL_DIR / "deaths_hgbr.joblib")
    feature_cols = json.loads((MODEL_DIR / "feature_columns.json").read_text())
    schema = json.loads((PROC_DIR / "feature_schema.json").read_text())
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    top_cases = json.loads((MODEL_DIR / "top_features_cases.json").read_text())
    top_deaths = json.loads((MODEL_DIR / "top_features_deaths.json").read_text())
    seasonal = pd.read_csv(PROC_DIR / "seasonal_median_features.csv")
    location = pd.read_csv(DATA_DIR / "location.csv")[["adm3_pcode", "adm3_en"]].drop_duplicates("adm3_pcode")
    return {
        "cases_model": cases_model,
        "deaths_model": deaths_model,
        "feature_cols": feature_cols,
        "schema": schema,
        "metrics": metrics,
        "top_cases": top_cases,
        "top_deaths": top_deaths,
        "seasonal": seasonal,
        "location": location,
    }


@st.cache_resource
def build_explainers(_cases_model, _deaths_model):
    return shap.TreeExplainer(_cases_model), shap.TreeExplainer(_deaths_model)


def seasonal_row(seasonal: pd.DataFrame, adm3_pcode: str, week_of_year: int) -> pd.Series:
    """Return one row of seasonal-median features for the chosen (city, week_of_year),
    falling back to (city, nearest_woy) -> (any-city, woy) if a specific cell is missing."""
    exact = seasonal[(seasonal["adm3_pcode"] == adm3_pcode) & (seasonal["week_of_year"] == week_of_year)]
    if len(exact):
        return exact.iloc[0]
    city = seasonal[seasonal["adm3_pcode"] == adm3_pcode]
    if len(city):
        # nearest week_of_year
        idx = (city["week_of_year"] - week_of_year).abs().idxmin()
        return city.loc[idx]
    return seasonal[seasonal["week_of_year"] == week_of_year].iloc[0] if len(seasonal) else pd.Series(dtype=float)


def month_to_week_of_year(month: int) -> int:
    """Representative ISO week for a month (mid-month), used to drive all calendar terms."""
    return int(pd.Timestamp(REFERENCE_YEAR, month, 15).isocalendar().week)


def assemble_input_row(artifacts: dict, adm3_pcode: str, month: int, overrides: dict) -> pd.DataFrame:
    """Build a single-row DataFrame in the order required by the model."""
    week_of_year = month_to_week_of_year(month)
    sched = artifacts["schema"]["features"]
    row = {}
    seas = seasonal_row(artifacts["seasonal"], adm3_pcode, week_of_year)
    for feat in artifacts["feature_cols"]:
        if feat in overrides and overrides[feat] is not None:
            row[feat] = overrides[feat]
            continue
        if feat.startswith("city_"):
            row[feat] = 1 if feat == f"city_{adm3_pcode}" else 0
            continue
        if feat == "year":
            row[feat] = REFERENCE_YEAR
            continue
        if feat == "month":
            row[feat] = month
            continue
        if feat == "week_of_year":
            row[feat] = week_of_year
            continue
        if feat == "week_sin":
            row[feat] = float(np.sin(2 * np.pi * week_of_year / 52.0))
            continue
        if feat == "week_cos":
            row[feat] = float(np.cos(2 * np.pi * week_of_year / 52.0))
            continue
        # numeric -> seasonal median for this (city, week) if present, else feature-wide median
        if feat in seas.index and pd.notna(seas[feat]):
            row[feat] = float(seas[feat])
        elif feat in sched:
            row[feat] = sched[feat]["p50"]
        else:
            row[feat] = 0.0
    return pd.DataFrame([row])[artifacts["feature_cols"]]


def slider_help(schema_feat: dict) -> str:
    return f"typical range p1={schema_feat['p1']:.2f} · median p50={schema_feat['p50']:.2f} · p99={schema_feat['p99']:.2f}"


def shap_waterfall_png(explainer, x_row: pd.DataFrame, title: str, top_n: int = 12) -> plt.Figure:
    sv = explainer.shap_values(x_row)
    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        expected = float(np.array(expected).flatten()[0])
    contributions = pd.Series(sv.flatten(), index=x_row.columns).sort_values(key=lambda s: s.abs(), ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(contributions.index[::-1], contributions.values[::-1], color=["#1a9850" if v > 0 else "#d73027" for v in contributions.values[::-1]])
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_title(f"{title}\nbase={expected:.2f}  pred={expected + sv.sum():.2f}", fontsize=10)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    return fig


def main():
    artifacts = load_artifacts()
    cases_expl, deaths_expl = build_explainers(artifacts["cases_model"], artifacts["deaths_model"])

    st.title("CCHAIN morbidity / mortality simulator")
    st.caption(
        "A what-if simulator for weekly all-cause case + death counts across the 12 CCHAIN "
        "cities. Pick a city + month, then move the levers in the sidebar to explore how "
        "recent caseload, climate, population, and health-access scenarios change the "
        "predicted counts."
    )

    # ----- sidebar inputs -----
    st.sidebar.header("Scenario")
    loc = artifacts["location"].sort_values("adm3_en")
    cities = artifacts["schema"]["adm3_pcode"]["choices"]
    loc = loc[loc["adm3_pcode"].isin(cities)]
    city_label = st.sidebar.selectbox(
        "City (ADM3)",
        options=loc["adm3_pcode"].tolist(),
        format_func=lambda code: f"{loc.set_index('adm3_pcode').loc[code, 'adm3_en']} ({code})",
    )
    month = MONTHS.index(st.sidebar.selectbox("Month", options=MONTHS, index=5)) + 1
    week_of_year = month_to_week_of_year(month)
    seas = seasonal_row(artifacts["seasonal"], city_label, week_of_year)

    st.sidebar.subheader("Levers")
    st.sidebar.caption(
        "Sliders start at this city/month's typical conditions. Move them to simulate a "
        "scenario; everything else stays at the seasonal median."
    )

    if "overrides" not in st.session_state:
        st.session_state["overrides"] = {}

    if st.sidebar.button("Reset to typical conditions for this city/month"):
        st.session_state["overrides"] = {}

    overrides: dict = {}
    last_group = None
    for feat, label, group in SIM_FEATURES:
        sf = artifacts["schema"]["features"].get(feat)
        if sf is None:
            continue
        if group != last_group:
            st.sidebar.markdown(f"**{group}**")
            last_group = group
        lo, hi = float(sf["p1"]), float(sf["p99"])
        if hi <= lo:
            hi = lo + max(1.0, abs(lo))
        # default to this city/month's seasonal median, else schema median, clamped to bounds
        if feat in seas.index and pd.notna(seas[feat]):
            default = float(seas[feat])
        else:
            default = float(sf["p50"])
        default = min(max(default, lo), hi)
        cur = float(st.session_state["overrides"].get(feat, default))
        cur = min(max(cur, lo), hi)
        overrides[feat] = st.sidebar.slider(
            label, min_value=lo, max_value=hi, value=cur, help=slider_help(sf)
        )
    st.session_state["overrides"] = overrides

    # ----- predictions -----
    X = assemble_input_row(artifacts, city_label, month, overrides)
    pred_cases = float(artifacts["cases_model"].predict(X)[0])
    pred_deaths = float(artifacts["deaths_model"].predict(X)[0])

    left, right = st.columns(2)
    left.metric("Predicted weekly cases (all-cause)",  f"{pred_cases:,.1f}")
    right.metric("Predicted weekly deaths (all-cause)", f"{pred_deaths:,.1f}")

    st.divider()

    expl_left, expl_right = st.columns(2)
    with expl_left:
        st.subheader("What drove the cases prediction")
        st.pyplot(shap_waterfall_png(cases_expl,  X, "Top contributions (cases)"))
    with expl_right:
        st.subheader("What drove the deaths prediction")
        st.pyplot(shap_waterfall_png(deaths_expl, X, "Top contributions (deaths)"))

    st.divider()

    with st.expander("Model performance on the 2019 test set (pre-COVID)"):
        overall = pd.DataFrame(artifacts["metrics"]["overall"]).set_index("label")
        st.dataframe(overall.style.format({"mae": "{:.2f}", "rmse": "{:.2f}", "r2": "{:.3f}", "poisson_dev": "{:.2f}"}))
        st.caption(
            "Headline metrics are on 2019, the last clean pre-COVID year. The 2020+ COVID era is held "
            "out as a separate stress test (metrics.json['stress_2020plus']); read predictions with that "
            "distribution shift in mind. See notebook 05 for per-city diagnostics."
        )

    with st.expander("Active feature row (full vector sent to the model)"):
        st.dataframe(X.T.rename(columns={0: "value"}))


if __name__ == "__main__":
    main()
