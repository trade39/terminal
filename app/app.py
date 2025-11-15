# app/app.py
import streamlit as st
import plotly.express as px
import pandas as pd
from ingest.ohlc_fetcher import fetch_ohlc
from features.engineer import engineer_features
from models.train import train_model
from models.infer import infer_signal
from storage.db_manager import load_ohlc
import yaml
from utils.config import ASSETS

st.set_page_config(page_title="Quant Terminal", layout="wide")
st.title("🏦 Quant Terminal - Bloomberg Analogue")

# Config load
with open('config/config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
ASSETS = cfg['assets']  # ['DXY', 'XAUUSD', ...]

# Sidebar: Asset selector, refresh, retrain
st.sidebar.header("Controls")
selected_asset = st.sidebar.selectbox("Asset", ASSETS)
if st.sidebar.button("Refresh Data"):
    with st.spinner("Fetching..."):
        df = fetch_ohlc(selected_asset)
        st.cache_data.clear()  # Invalidate
st.sidebar.button("Retrain Model", on_click=lambda: train_model(selected_asset))

# Caching wrappers
@st.cache_data(ttl=3600)  # 1hr TTL for data
def get_data(asset: str):
    return load_ohlc(asset, '2023-01-01')

@st.cache_resource(ttl=7200)  # 2hr for model
def get_model(asset: str):
    from models.train import train_model
    train_model(asset)  # Ensure trained
    return joblib.load(f'models/rf_{asset}.joblib')

# Main dashboard: Multi-column
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Price Chart")
    df = get_data(selected_asset)
    fig = px.line(df, x='timestamp', y='close', title=f"{selected_asset} Price")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Correlation Matrix")
    corrs = pd.DataFrame({a: engineer_features(a)['returns'].tail(252) for a in ASSETS}).corr()
    fig_corr = px.imshow(corrs, title="Asset Correlations", color_continuous_scale='RdBu')
    st.plotly_chart(fig_corr, use_container_width=True)

with col3:
    st.subheader("Fundamental Snapshot")
    st.metric("FEDFUNDS Rate", 5.33, delta=0.25)  # Mock from FRED
    st.metric("CPI YoY", 3.2, delta=-0.1)

# ML Signal Panel
st.subheader("ML Signal")
signal, expl = infer_signal(selected_asset)
st.metric("Signal Score", f"{signal:.2f}", delta=signal > 0)
st.bar_chart(pd.Series(expl).sort_values(ascending=False).head(5))  # Top features

# Narrative Summary
st.subheader("Market Narrative")
if signal > 0.2:
    summary = f"{selected_asset} shows bullish momentum (score: {signal:.2f}), driven by {max(expl, key=expl.get)}."
else:
    summary = f"{selected_asset} neutral/bearish (score: {signal:.2f}); watch {max(expl, key=expl.get)}."
st.write(summary)

# Backtest Hook
if st.button("Run Quick Backtest"):
    feats = engineer_features(selected_asset)
    pnl = simple_backtest(selected_asset, feats)
    st.success(f"Simulated P&L: {pnl:.2%}")
