#!/usr/bin/env python
# coding: utf-8

# Demand Analysis JC — Google Trends + STL (LOESS)
# Fixed scope: US, last 5 years, en-US
# Best practices: retries/backoff, caching, validation, structured error messages.

import numpy as np
import pandas as pd
import streamlit as st

# --- Guard: detect urllib3>=2 (incompatible with pytrends 4.9.2 due to method_whitelist removal) ---
try:
    import urllib3
    from packaging import version
    if version.parse(urllib3.__version__) >= version.parse("2.0.0"):
        st.error(
            "Incompatible urllib3 version detected "
            f"({urllib3.__version__}). Please pin urllib3<2 in your environment:\n\n"
            "    pip install 'urllib3<2'\n\n"
            "This resolves the 'unexpected keyword argument method_whitelist' error."
        )
        st.stop()
except Exception:
    # If anything goes wrong with the check, continue (pytrends may still work if urllib3<2)
    pass

from pytrends.request import TrendReq
from statsmodels.tsa.seasonal import STL
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------- Streamlit basic setup ----------
st.set_page_config(page_title="Demand Analysis JC", layout="wide")
st.title("Demand Analysis JC")
st.caption("Google Trends (US, last 5y, en-US) → STL (LOESS) → Plotly")

# ---------- UI inputs ----------
kw = st.text_input("Keyword (required)", value="", placeholder="e.g., rocket stove")
run = st.button("Run analysis")

# Fixed config
HL = "en-US"          # interface language for Trends
TZ = 360              # timezone offset param (per pytrends examples)
TIMEFRAME = "today 5-y"  # last 5 years
GEO = "US"            # United States

# ---------- Helpers ----------
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_trends(keyword: str) -> pd.DataFrame:
    """Call Google Trends via pytrends and return a cleaned dataframe."""
    pytrends = TrendReq(
        hl=HL,
        tz=TZ,
        timeout=(10, 25),     # connect/read
        retries=2,
        backoff_factor=0.1,   # exponential backoff
    )
    pytrends.build_payload([keyword], timeframe=TIMEFRAME, geo=GEO)
    df = pytrends.interest_over_time()
    if df.empty:
        return df
    # Clean: drop last row and 'isPartial' per your workflow
    if len(df) > 0:
        df = df.iloc[:-1]
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])
    # Ensure datetime index
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

def infer_period(dt_index: pd.DatetimeIndex) -> int:
    """Infer STL seasonal period from median sampling interval."""
    if len(dt_index) < 3:
        return 12
    deltas = np.diff(dt_index.values).astype("timedelta64[D]").astype(int)
    med = int(np.median(deltas))
    if med <= 1:
        return 7      # daily cadence -> weekly seasonality
    elif med <= 7:
        return 52     # weekly cadence -> yearly seasonality
    else:
        return 12     # monthly cadence -> yearly seasonality

def build_figure(df_plot: pd.DataFrame, title_kw: str) -> go.Figure:
    """Build 4-panel Plotly figure."""
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        subplot_titles=("Original", "Trend", "Seasonal", "Residual")
    )
    fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["original"], name="Original", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["trend"],   name="Trend",   mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["seasonal"],name="Seasonal",mode="lines"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["remainder"],name="Residual",mode="lines"), row=4, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=4, col=1)
    fig.update_layout(height=900, title_text=f"STL Decomposition — {title_kw} — Google Trends (US, last 5y)")
    return fig

# ---------- Execution ----------
if run:
    if not kw.strip():
        st.error("Please enter a keyword.")
        st.stop()

    with st.spinner("Fetching Google Trends…"):
        try:
            df = fetch_trends(kw.strip())
        except Exception as e:
            st.error(f"Error fetching data from Google Trends: {e}")
            st.info("Tip: pin urllib3<2 and try again if you see 'method_whitelist' errors.")
            st.stop()

    if df.empty:
        st.warning("No data returned by Google Trends for this keyword/timeframe/geo.")
        st.stop()

    # Series selection follows the exact column typed by user
    col_name = kw.strip()
    if col_name not in df.columns:
        st.error(f"Column '{col_name}' not found in Trends result.")
        st.stop()

    y = df[col_name].astype(float)
    period = infer_period(y.index)

    # STL (LOESS)
    try:
        res = STL(y, period=period, robust=True).fit()
    except Exception as e:
        st.error(f"STL decomposition failed: {e}")
        st.stop()

    # Build dataframe for plotting and CSV export
    df_plot = pd.DataFrame({
        "date": y.index,
        "original": y.values,
        "trend": res.trend,
        "seasonal": res.seasonal,
        "remainder": res.resid
    })

    fig = build_figure(df_plot, kw.strip())
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show data (original/trend/seasonal/residual)"):
        st.dataframe(df_plot, use_container_width=True)
        st.download_button(
            "Download CSV",
            df_plot.to_csv(index=False).encode("utf-8"),
            file_name=f"stl_{kw.strip().replace(' ','_')}.csv",
            mime="text/csv"
        )

# ---------- Footer ----------
st.markdown(
    """
    <small>
    Data source: Google Trends via <code>pytrends</code> • Decomposition: <code>statsmodels.STL</code> • Chart: Plotly • Host: Streamlit Community Cloud
    </small>
    """, unsafe_allow_html=True
)
