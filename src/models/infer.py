# src/models/infer.py (UPDATED - No changes needed, but full for completeness)
import pandas as pd
import joblib
import numpy as np
from typing import Dict, Tuple
from features.engineer import engineer_features

def infer_signal(symbol: str) -> Tuple[float, Dict]:
    """Infer score (-1 to 1); explain via importance."""
    feats = engineer_features(symbol).iloc[-1:].drop(['returns'], axis=1)
    model = joblib.load(f'models/rf_{symbol}.joblib')
    scaler = joblib.load(f'models/scaler_{symbol}.joblib')
    
    feature_cols = [c for c in feats.columns if c not in ['target', 'symbol', 'timestamp']]
    X_scaled = scaler.transform(feats[feature_cols])
    prob = model.predict_proba(X_scaled)[0][1]
    signal = (prob - 0.5) * 2
    
    importances = model.feature_importances_
    expl = dict(zip(feature_cols, importances))
    expl = dict(sorted(expl.items(), key=lambda x: x[1], reverse=True))
    
    print(f"Signal for {symbol}: {signal:.2f}")
    return signal, expl
