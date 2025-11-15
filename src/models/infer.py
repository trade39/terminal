# src/models/infer.py (FULL FINAL - Auto-train if missing)
import os
import pandas as pd
import joblib
import numpy as np
from typing import Dict, Tuple
from features.engineer import engineer_features
from models.train import train_model

def infer_signal(symbol: str) -> Tuple[float, Dict]:
    model_path = f'models/rf_{symbol}.joblib'
    scaler_path = f'models/scaler_{symbol}.joblib'
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Model missing for {symbol} — training...")
        train_model(symbol)
    feats = engineer_features(symbol).iloc[-1:].drop(['returns'], axis=1)
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    feature_cols = [c for c in feats.columns if c not in ['target', 'symbol', 'timestamp']]
    X_scaled = scaler.transform(feats[feature_cols])
    prob = model.predict_proba(X_scaled)[0][1]
    signal = (prob - 0.5) * 2
    
    importances = model.feature_importances_
    expl = dict(zip(feature_cols, importances))
    expl = dict(sorted(expl.items(), key=lambda x: x[1], reverse=True))
    
    print(f"Signal for {symbol}: {signal:.2f}")
    return signal, expl
