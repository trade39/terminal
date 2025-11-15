# src/models/infer.py
import pandas as pd
import joblib
import shap
import numpy as np
from typing import Dict, Tuple
from features.engineer import engineer_features

def infer_signal(symbol: str) -> Tuple[float, Dict]:
    """Infer score (-1 to 1); explain via importance/SHAP."""
    feats = engineer_features(symbol).iloc[-1:].drop(['returns'], axis=1)  # Latest
    model = joblib.load(f'models/rf_{symbol}.joblib')
    scaler = joblib.load(f'models/scaler_{symbol}.joblib')
    
    X_scaled = scaler.transform(feats.drop('target', axis=1) if 'target' in feats else feats)
    prob = model.predict_proba(X_scaled)[0][1]  # P(up)
    signal = (prob - 0.5) * 2  # -1 to 1
    
    # Explainability
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)[1]  # For class 1
        expl = dict(zip(feats.columns, shap_values[0]))
    except Exception:
        # Fallback: Feature importance
        expl = dict(zip(model.feature_names_in_, model.feature_importances_))
    
    print(f"Signal for {symbol}: {signal:.2f}")
    return signal, expl
