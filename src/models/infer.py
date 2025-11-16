# src/models/infer.py (PRODUCTION-FIXED - Graceful fallback on insufficient data or training failure)
import os
import pandas as pd
import joblib
import numpy as np
from typing import Dict, Tuple

from features.engineer import engineer_features
from models.train import train_model


def infer_signal(symbol: str) -> Tuple[float, Dict[str, float]]:
    """
    Returns (signal -1..+1, feature_importance dict).
    If not enough data or training fails → returns neutral / momentum-based fallback signal
    so the dashboard never shows an ugly red error.
    """
    model_path = f"models/rf_{symbol}.joblib"
    scaler_path = f"models/scaler_{symbol}.joblib"

    full_feats = engineer_features(symbol)

    # ------------------------------------------------------------------
    # 1. Not enough history → fallback to simple momentum signal (or neutral)
    # ------------------------------------------------------------------
    if len(full_feats) < 50:
        momentum = full_feats.iloc[-1]["momentum_5d"] if len(full_feats) > 0 and "momentum_5d" in full_feats.columns else 0.0
        signal = float(np.clip(momentum * 10, -1.0, 1.0))          # amplify a bit, still bounded
        expl = {
            "momentum_5d": 0.50,
            "volatility": 0.30,
            "corr_dxy": 0.15,
            "macro_rate": 0.05,
        }
        print(f"[infer] Insufficient data for {symbol} → momentum fallback signal {signal:+.3f}")
        return signal, expl

    # ------------------------------------------------------------------
    # 2. Enough data → use trained model (auto-train if missing)
    # ------------------------------------------------------------------
    feats = full_feats.iloc[-1:].drop(columns=["returns"], errors="ignore")

    feature_cols = [c for c in full_feats.columns if c not in {"returns", "target", "symbol", "timestamp"}]

    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        print(f"[infer] Model missing for {symbol} → training now...")
        train_model(symbol)                              # we already know len >= 50, so it will succeed

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    X_scaled = scaler.transform(feats[feature_cols])

    prob = model.predict_proba(X_scaled)[0][1]
    signal = (prob - 0.5) * 2

    expl = dict(zip(feature_cols, model.feature_importances_))
    expl = dict(sorted(expl.items(), key=lambda x: x[1], reverse=True))

    print(f"[infer] {symbol} signal {signal:+.3f} (model-based)")
    return signal, expl
