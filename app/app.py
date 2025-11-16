# app/app.py (FULL FINAL - Added data fetch before retrain to prevent insufficient data error)
import streamlit as st
import yaml
import joblib
import pandas as pd
import sys
import os

# Add paths
sys.path.append('src')
sys.path.append('ops')  # FIXED: For backtest import

# Create dirs
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Simple config load
config_path = 'config/config.yaml'
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ASSETS = cfg.get('assets', ['DXY', 'XAUUSD', 'ES', 'NQ', 'EURUSD', 'GBPUSD'])
else:
    st.warning("config.yaml missing → default assets")
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
    with st.spinner(f"Ensuring data & retraining model for {selected_asset}..."):
        # FIXED: Fetch and store data first to ensure sufficient history for training
        from storage.db_manager import load_ohlc, store_ohlc
        from ingest.ohlc_fetcher import fetch_ohlc
        
        df = load_ohlc(selected_asset, '2020-01-01')
        if df.empty or len(df) < 50:
            try:
                fresh = fetch_ohlc(selected_asset, days=2000)
                store_ohlc(fresh)
                st.info(f"Fetched {len(fresh)} fresh bars for {selected_asset}")
            except Exception as fetch_e:
                st.error(f"Data fetch failed for {selected_asset}: {fetch_e}")
                st.stop()
        
        from models.train import train_model
        try:
            metrics = train_model(selected_asset)
            st.success(f"Model retrained! CV Accuracy: {metrics.get('cv_accuracy', 'N/A'):.2f}")
        except Exception as train_e:
            st.error(f"Training failed despite data fetch: {train_e}")

# Data with auto-fetch
@st.cache_data(ttl=1800)
def get_data(asset: str) -> pd.DataFrame:
    from storage.db_manager import load_ohlc

    df = load_ohlc(asset, '2020-01-01')

    if df.empty or len(df) < 500:
        with st.spinner(f"Fetching fresh data for {asset}..."):
            from ingest.ohlc_fetcher import fetch_ohlc
            from storage.db_manager import store_ohlc
            try:
                fresh = fetch_ohlc(asset, days=2000)
                store_ohlc(fresh)
                df = fresh
            except Exception as e:
                st.error(f"Fetch failed for {asset}: {e}")
                return pd.DataFrame()
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.dropna(subset=['close'])
    return df.sort_values('timestamp').reset_index(drop=True)

# Model with auto-train
@st.cache_resource(ttl=86400)
def get_model(asset: str):
    model_path = f'models/rf_{asset}.joblib'
    if not os.path.exists(model_path):
        with st.spinner(f"First-time training for {asset}..."):
            from models.train import train_model
            train_model(asset)
    return joblib.load(model_path)

# Plotly
@st.cache_resource(ttl=86400)
def get_plotly():
    import plotly.express as px
    return px

px = get_plotly()

# Dashboard
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.subheader(f"{selected_asset} Price Chart")
    df = get_data(selected_asset)
    if not df.empty and 'timestamp' in df.columns and 'close' in df.columns:
        fig = px.line(df, x='timestamp', y='close', title=f"{selected_asset} Close")
        fig.update_layout(height=580)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Data missing columns — refresh data")

with col2:
    st.subheader("252-Day Rolling Correlation Matrix")
    with st.spinner("Calculating..."):
        returns_dict = {}
        for a in ASSETS:
            data = get_data(a)
            if not data.empty and 'timestamp' in data.columns and 'close' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                returns_dict[a] = data.set_index('timestamp')['close'].pct_change().tail(252)
        if returns_dict:
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
        else:
            st.warning("No data for correlations — refresh")

with col3:
    st.subheader("Macro Snapshot")
    st.metric("FEDFUNDS", "5.33%", "+0.00")
    st.metric("US CPI YoY", "2.7%", "-0.1%")
    st.metric("VIX", "14.20", "-0.80")
    st.caption("November 16, 2025")

# ML Signal
st.subheader(f"ML Signal — {selected_asset}")
try:
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
except Exception as e:
    st.error(f"ML inference failed: {e}")
    st.metric("Signal (-1 → +1)", "N/A")

if st.button("Run Quick 2-Year Backtest"):
    with st.spinner("Backtesting..."):
        try:
            from backtest import simple_backtest  # FIXED: Direct import from ops/backtest.py
            from features.engineer import engineer_features
            feats = engineer_features(selected_asset)
            pnl = simple_backtest(selected_asset, feats)
            st.success(f"Simulated 2Y P&L: {pnl:+.2%}")
        except Exception as e:
            st.error(f"Backtest failed: {e}")

st.caption("Quant Terminal v1.0 — Free-tier Bloomberg Killer | Nov 16, 2025")
