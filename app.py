"""Replay demo.  pip install streamlit matplotlib ;  streamlit run app.py"""
import glob

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import xarray as xr

from svara import config as C
from svara.align import valid_offset
from svara.fc_regimes import OUT_LABELS as FC_LABELS

st.set_page_config(page_title="SVARA-Rain", layout="wide")
st.title("SVARA-Rain: heavy-rain probability guidance (replay on archived forecasts)")
st.caption("AI-assisted post-processing guidance, not an official IMD forecast.")

PROD = C.DATA / "products"
years = sorted(p.split("prob_test_")[1].split(".")[0] for p in glob.glob(str(PROD / "prob_test_*.zarr")))


@st.cache_resource
def load_probs(year):
    return xr.open_zarr(PROD / f"prob_test_{year}.zarr").load()


@st.cache_resource
def load_obs():
    return xr.open_zarr(C.IMD_ZARR).obs


@st.cache_data
def load_regimes():
    return pd.read_csv(FC_LABELS, parse_dates=["init_time", "date"])


tab_map, tab_eval = st.tabs(["Map", "Verification"])

with tab_map:
    c1, c2, c3, c4 = st.columns(4)
    year = c1.selectbox("Test year", years)
    ds = load_probs(year)
    inits = pd.DatetimeIndex(ds.init_time.values)
    init = c2.selectbox("Forecast start date", inits, format_func=lambda d: d.strftime("%Y-%m-%d"))
    lead = c3.selectbox("Lead day", [1, 2, 3, 4, 5])
    var = c4.selectbox("Threshold", ["p64", "p115"], format_func=lambda v: ">= 64.5 mm (heavy)" if v == "p64" else ">= 115.6 mm (very heavy)")

    rg = load_regimes()
    row = rg[(rg.init_time == init) & (rg.lead == lead)]
    if len(row):
        r = row.iloc[0]
        st.markdown(f"**Forecast regime:** `{r.fc_regime}`  (CMZ anomaly z = {r.z_f:+.2f})")

    valid = init + valid_offset(lead)
    p = ds[var].sel(init_time=init, lead_day=lead)
    o = load_obs().sel(time=valid)

    left, right = st.columns(2)
    fig, ax = plt.subplots(figsize=(5, 5))
    m = ax.pcolormesh(p.lon, p.lat, p.values, vmin=0, vmax=1, cmap="YlOrRd")
    fig.colorbar(m, ax=ax, label="probability")
    ax.set_title(f"Probability, valid {valid:%Y-%m-%d}")
    left.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    m2 = ax2.pcolormesh(o.lon, o.lat, o.values, vmin=0, vmax=120, cmap="Blues")
    fig2.colorbar(m2, ax=ax2, label="mm/day")
    ax2.set_title("IMD observed rainfall (for checking)")
    right.pyplot(fig2)

with tab_eval:
    f = C.DATA / "reports" / "prob_pooled.csv"
    if f.exists():
        d = pd.read_csv(f)
        st.subheader("Brier skill score vs climatology (5 leave-one-year-out folds, 95% bootstrap CI)")
        st.dataframe(d[(d.regime == "all") & (d.model != "regime_minus_noregime")].round(3), use_container_width=True)
        st.subheader("Does regime information help? (regime minus no-regime)")
        st.dataframe(d[d.model == "regime_minus_noregime"].round(4), use_container_width=True)