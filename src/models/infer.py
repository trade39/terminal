# src/models/infer.py
import pandas as pd
import joblib
from typing import Tuple, Dict

def infer_signal(symbol: str) -> Tuple[float, Dict[str, float]]:
    """Return signal (-1 to +1) + feature importance dict."""
    # Load latest features
    from features.engineer import engineer_features
    feats = engineer_features(symbol).iloc[-1:]  # Latest row
    
    model = joblib.load(f'models/rf_{symbol}.joblib')
    scaler = joblib.load(f'models/scaler_{symbol}.joblib')
    
    feature_cols = [c for c in feats.columns if c not in ['returns', 'target', 'symbol', 'timestamp']]
    X_scaled = scaler.transform(feats[feature_cols])
    
    prob_up = model.predict_proba(X_scaled)[0][1]
    signal = (prob_up - 0.5) * 2  # -1 to +1
    
    # Fast, reliable importance (no SHAP ever needed)
    importances = model.feature_importances_
    expl = dict(zip(feature_cols, importances))
    expl = dict(sorted(expl.items(), key=lambda x: x[1], reverse=True))
    
    return float(signal), expl
