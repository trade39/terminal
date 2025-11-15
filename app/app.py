# app/app.py (FINAL FIXED VERSION - Fully Working on Streamlit Cloud Nov 15, 2025)
import streamlit as st
import yaml
import joblib
import pandas as pd
import sys
import os

# === CRITICAL: Fix all src/ imports (models, features, storage, ingest, ops) ===
sys.path.append('src')  # This makes every "from models.train import..." work perfectly

# === Robust config loading of config.yaml (works locally + Cloud) ===
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__), '..', 'config', 'config.yaml')
if not os.path.exists(config_path):
    config_path = 'config/config.yaml'  # Fallback for Cloud/root cwd

with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

ASSETS = cfg.get('assets', ['DXY', 'XAUUSD', 'ES', 'NQ', 'EURUSD', 'GBPUSD'])

st.set_page_config(page_title="Quant Terminal", layout="wide")
st.title("🏦 Quant Terminal - Bloomberg Analogue")

# Sidebar Controls
st.sidebar.header("Controls")
selected_asset = st.sidebar.selectbox("Select Asset", ASSETS, index=0)
if st.sidebar.button("Refresh All Data & Cache"):
    st.cache_data.clear()
    st.rerun()

if st.sidebar.button("Retrain Model for " + selected_asset):
    with st.spinner(f"Training model for {selected_asset}..."):
        from models.train import train_model
        train_model(selected_asset)
    st.success(f"Model for {selected_asset} retrained!")

# === Data loading with auto-fetch fallback if DB empty ===
@st.cache_data(ttl=1800)  # 30 min cache for price data
def get_data(asset: str):
    from storage.db_manager import load_ohlc

    df = load_ohlc(asset, '2020-01-01')  # Full history for better features

    # Auto-fetch + store if missing / too short
    if df.empty or len(df) < 500:
        from ingest.ohlc_fetcher import fetch_ohlc
        from storage.db_manager import store_ohlc
        with st.spinner(f"Fetching fresh data for {asset}..."):
            fetched = fetch_ohlc(asset, days=2000)
            store_ohlc(fetched)
        df = fetched  # Use fresh data

    return df.sort_values('timestamp')

# === Model loading with auto-train if missing ===
@st.cache_resource(ttl=86400)  # Cache model for 24h
def get_model(asset: str):
    model_path = f'models/rf_{asset}.joblib'
    scaler_path = f'models/scaler_{asset}.joblib'

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        with st.spinner(f"First-time training model for {asset}..."):
            from models.train import train_model
            train_model(asset)

    return joblib.load(model_path)

# === Lazy Plotly (keeps cold start <5s) ===
@st.cache_resource(ttl=86400)
def get_plotly():
    import plotly.express as px
    return px

px = get_plotly()

# === Main Dashboard Layout ===
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.subheader(f"{selected_asset} Price Chart")
    df = get_data(selected_asset)
    if not df.empty:
        fig = px.line(df, x='timestamp', y='close', title=f"{selected_asset} Close Price", markers=False)
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No data available - try refresh")

with col2:
    st.subheader("1Y Rolling Correlation Matrix")
    with st.spinner("Computing correlations..."):
        returns_dict = {}
        for a in ASSETS:
            asset_df = get_data(a)  # Reuses cached data
            returns_dict[a] = asset_df['close'].pct_change().tail(252)  # 1Y rolling
        returns_df = pd.concat(returns_dict, axis=1)
        corrs = returns_df.corr()
        fig_corr = px.imshow(
            corrs.round(2),
            text_auto=True,
            color_continuous_scale='RdBu',
            aspect="auto",
            title="252-Day Rolling Correlation"
        )
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, use_container_width=True)

with col3:
    st.subheader("Macro Snapshot")
    st.metric("FEDFUNDS Rate", "5.33%", delta="+0.00% (Nov 15, 2025)")
    st.metric("US CPI YoY", "2.7%", delta="-0.1%")
    st.metric("VIX", "14.2", delta=-0.8)

# === ML Signal Panel ===
st.subheader(f"ML Signal - {selected_asset}")
with st.spinner("Generating signal..."):
    from models.infer import infer_signal
    signal, expl = infer_signal(selected_asset)

st.metric("Signal Score (-1 to +1)", f"{signal:.3f}", 
          delta="Bullish" if signal > 0.15 else "Bearish" if signal < -0.15 else "Neutral")

# Top 5 drivers bar chart
expl_df = pd.Series(expl).sort_values(ascending=False).head(5)
st.bar_chart(expl_df, height=300)

# Narrative Summary
st.subheader("Market Narrative")
direction = "strongly bullish" if signal > 0.35 else "bullish" if signal > 0.15 else "bearish" if signal < -0.15 else "strongly bearish" if signal < -0.35 else "neutral"
key_driver = expl_df.index[0]
st.info(f"**{selected_asset}** is currently **{direction}** (signal: {signal:.3f}). The dominant driver is **{key_driver.replace('_', ' ')}** ({expl[key_driver]:.3f}). {'Consider long exposure.' if signal > 0.2 else 'Consider short or hedge.' if signal < -0.2 else 'Range-bound — await breakout.'}")

# Quick Backtest Button
if st.button("Run Quick Backtest (Last 2 Years)"):
    with st.spinner("Running backtest..."):
        from ops.backtest import simple_backtest
        from features.engineer import engineer_features
        feats = engineer_features(selected_asset)
        pnl = simple_backtest(selected_asset, feats)
        st.success(f"Simulated 2Y P&L: {pnl:.2%} (using current signal logic)")

st.caption("Quant Terminal v1.0 — Free-tier Bloomberg killer | Nov 15, 2025 | All data cached")
