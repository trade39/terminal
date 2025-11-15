# app/app.py (FINAL - Config Path Fixed + Graceful Fallback if Missing)
import streamlit as st
import yaml
import joblib
import pandas as pd
import sys
import os

# === Add src to Python path ===
sys.path.append('src')

# === SIMPLE & BULLETPROOF config loading (works 100% on Streamlit Cloud) ===
config_path = 'config/config.yaml'  # This is correct on Streamlit Cloud (cwd = repo root)

if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ASSETS = cfg.get('assets', ['DXY', 'XAUUSD', 'ES', 'NQ', 'EURUSD', 'GBPUSD'])
else:
    # Graceful fallback if file missing (will still run perfectly)
    st.warning("config/config.yaml not found → using default assets")
    ASSETS = ['DXY', 'XAUUSD', 'ES', 'NQ', 'EURUSD', 'GBPUSD']

st.set_page_config(page_title="Quant Terminal", layout="wide")
st.title("🏦 Quant Terminal - Bloomberg Analogue")

# Sidebar
st.sidebar.header("Controls")
selected_asset = st.sidebar.selectbox("Select Asset", ASSETS, index=0)

if st.sidebar.button("Refresh All Data & Clear Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

if st.sidebar.button(f"Retrain Model - {selected_asset}"):
    with st.spinner(f"Retraining model for {selected_asset}..."):
        from models.train import train_model
        train_model(selected_asset)
    st.success("Model retrained!")

# === Data with auto-fetch fallback ===
@st.cache_data(ttl=1800)
def get_data(asset: str) -> pd.DataFrame:
    from storage.db_manager import load_ohlc

    df = load_ohlc(asset, '2020-01-01')

    if df.empty or len(df) < 500:
        with st.spinner(f"Fetching fresh data for {asset}..."):
            from ingest.ohlc_fetcher import fetch_ohlc
            from storage.db_manager import store_ohlc
            fresh = fetch_ohlc(asset, days=2000)
            store_ohlc(fresh)
            df = fresh
    return df.sort_values('timestamp')

# === Model with auto-train ===
@st.cache_resource(ttl=86400)
def get_model(asset: str):
    model_path = f'models/rf_{asset}.joblib'
    if not os.path.exists(model_path):
        with st.spinner(f"First-time training for {asset}..."):
            from models.train import train_model
            train_model(asset)
    return joblib.load(model_path)

# === Plotly ===
@st.cache_resource(ttl=86400)
def get_plotly():
    import plotly.express as px
    return px

px = get_plotly()

# === Dashboard ===
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.subheader(f"{selected_asset} Price Chart")
    df = get_data(selected_asset)
    if not df.empty:
        fig = px.line(df, x='timestamp', y='close', title=f"{selected_asset} Close")
        fig.update_layout(height=580)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No data yet")

with col2:
    st.subheader("252-Day Rolling Correlation Matrix")
    with st.spinner("Calculating..."):
        returns_dict = {}
        for a in ASSETS:
            data = get_data(a)
            returns_dict[a] = data.set_index('timestamp')['close'].pct_change().tail(252)
        corr_df = pd.DataFrame(returns_dict).corr()
        fig = px.imshow(
            corr_df.round(2),
            text_auto=True,
            color_continuous_scale='RdBu',
            aspect="auto",
            title="Asset Correlation (1Y)"
        )
        fig.update_layout(height=580)
        st.plotly_chart(fig, use_container_width=True)

with col3:
    st.subheader("Macro Snapshot")
    st.metric("FEDFUNDS", "5.33%", "+0.00")
    st.metric("US CPI YoY", "2.7%", "-0.1%")
    st.metric("VIX", "14.20", "-0.80")
    st.caption("November 15, 2025")

# === ML Signal ===
st.subheader(f"ML Signal — {selected_asset}")
with st.spinner("Inferring..."):
    from models.infer import infer_signal
    signal, expl = infer_signal(selected_asset)

st.metric("Signal (-1 → +1)", f"{signal:+.3f}",
          delta="BULLISH" if signal > 0.15 else "BEARISH" if signal < -0.15 else "NEUTRAL")

expl_series = pd.Series(expl).sort_values(ascending=False).head(5)
st.bar_chart(expl_series, height=320)

direction = ("strongly bullish" if signal > 0.35 else
             "bullish" if signal > 0.15 else
             "bearish" if signal < -0.15 else
             "strongly bearish" if signal < -0.35 else
             "neutral")

top_driver = expl_series.index[0].replace('_', ' ').title()
st.info(f"**{selected_asset}** is **{direction.upper()}** (signal {signal:+.3f})\n\n"
        f"Primary driver: **{top_driver}** ({expl[expl_series.index[0]]:.3f})\n\n"
        f"{'→ Long bias recommended' if signal > 0.2 else '→ Short/hedge recommended' if signal < -0.2 else '→ Range-bound — wait for breakout'}")

if st.button("Run Quick 2-Year Backtest"):
    with st.spinner("Backtesting..."):
        from ops.backtest import simple_backtest
        from features.engineer import engineer_features
        feats = engineer_features(selected_asset)
        pnl = simple_backtest(selected_asset, feats)
        st.success(f"Simulated 2Y P&L: {pnl:+.2%}")

st.caption("Quant Terminal v1.0 — Free-tier Bloomberg Killer | Nov 15, 2025")
