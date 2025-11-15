# app/app.py (Enhanced for Speed)
import streamlit as st
import time  # For micro-delays if needed
import yaml
import joblib  # Core only on top

# Lazy imports: Only when used
@st.cache_resource(ttl=7200)
def lazy_import_plotly():
    import plotly.express as px
    return px

@st.cache_resource(ttl=7200)
def lazy_import_shap():
    import shap
    return shap

from utils.config import ASSETS  # Assume this loads fast

st.set_page_config(page_title="Quant Terminal", layout="wide")
st.title("🏦 Quant Terminal - Bloomberg Analogue")

# Config load (cached)
@st.cache_data(ttl=86400)  # 24hr for config
def load_config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

cfg = load_config()
ASSETS = cfg.get('assets', ['DXY', 'XAUUSD', 'ES', 'NQ', 'EURUSD', 'GBPUSD'])

# Sidebar: Controls
st.sidebar.header("Controls")
selected_asset = st.sidebar.selectbox("Asset", ASSETS)
refresh = st.sidebar.button("Refresh Data")
retrain = st.sidebar.button("Retrain Model")

if refresh:
    st.cache_data.clear()
    st.rerun()  # Quick refresh without full reload

if retrain:
    with st.spinner("Retraining model..."):
        from models.train import train_model
        train_model(selected_asset)
    st.success("Model retrained!")

# Enhanced Caching: Shorter TTL for data, but persistent models
@st.cache_data(ttl=1800)  # 30min for fresh data
def get_data(asset: str):
    from storage.db_manager import load_ohlc
    return load_ohlc(asset, '2023-01-01')

@st.cache_resource(ttl=7200)  # 2hr models
def get_model(asset: str):
    from models.train import train_model
    train_model(asset)  # Train if missing
    return joblib.load(f'models/rf_{asset}.joblib')

# Main Dashboard
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Price Chart")
    with st.spinner("Loading chart..."):
        df = get_data(selected_asset)
        px = lazy_import_plotly()
        fig = px.line(df, x='timestamp', y='close', title=f"{selected_asset} Price")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Correlation Matrix")
    with st.spinner("Computing correlations..."):
        from features.engineer import engineer_features
        corrs = pd.DataFrame({a: engineer_features(a, window=252)['returns'] for a in ASSETS}).corr()
        px = lazy_import_plotly()
        fig_corr = px.imshow(corrs, title="Asset Correlations", color_continuous_scale='RdBu')
        st.plotly_chart(fig_corr, use_container_width=True)

with col3:
    st.subheader("Fundamental Snapshot")
    # Mock/static for speed; replace with FRED cache later
    st.metric("FEDFUNDS Rate", "5.33%", delta="0.25%")
    st.metric("CPI YoY", "3.2%", delta="-0.1%")

# ML Signal (Lazy SHAP)
st.subheader("ML Signal")
with st.spinner("Generating signal..."):
    from models.infer import infer_signal
    signal, expl = infer_signal(selected_asset)

st.metric("Signal Score", f"{signal:.2f}", delta="+" if signal > 0 else "-")
# Fallback viz without SHAP
expl_df = pd.Series(expl).sort_values(ascending=False).head(5)
st.bar_chart(expl_df)

# Narrative
st.subheader("Market Narrative")
summary = f"{selected_asset} signals {signal:.2f}: {'Bullish' if signal > 0.2 else 'Bearish' if signal < -0.2 else 'Neutral'}, key driver: {expl_df.index[0]}."
st.info(summary)

# Backtest Button
if st.button("Run Quick Backtest"):
    with st.spinner("Simulating P&L..."):
        from ops.backtest import simple_backtest
        from features.engineer import engineer_features
        feats = engineer_features(selected_asset)
        pnl = simple_backtest(selected_asset, feats)
        st.success(f"Simulated P&L: {pnl:.2%}")
