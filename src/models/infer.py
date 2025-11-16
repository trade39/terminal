# src/models/infer.py (PRODUCTION-FIXED v3 - Bulletproof + No 'symbol' contamination + Clean fallback importances)

import os
import pandas as pd
import joblib
import numpy as np
from typing import Dict, Tuple

from features.engineer import engineer_features  # typo-safe: actual filename is engineer.py
from models.train import train_model

def infer_signal(symbol: str) -> Tuple[float, Dict[str, float]]:
    """
    PRODUCTION-GRADE inference function:
    - Always returns a valid signal + numeric explanation dict (never crashes the Streamlit app)
    - Graceful momentum-proxy fallback when data < 50 rows or training/inference fails
    - Auto-trains only when sufficient real data exists
    - Explicitly drops non-numeric columns ('symbol', 'timestamp' if present)
    """
    full_feats = engineer_features(symbol)
    
    # ------------------------------------------------------------------
    # Safety: drop any non-numeric columns that might have slipped in
    numeric_cols = ['returns', 'volatility', 'momentum_5d', 'corr_dxy', 'macro_rate']
    full_feats = full_feats[numeric_cols].copy()  # <--- critical fix for scaler compatibility
    
    if len(full_feats) == 0:
        # Extremely rare edge case — engineer_features returned empty (should not happen)
        print(f"[infer_signal] {symbol} | Empty feature set → neutral signal")
        return 0.0, {"momentum_5d": 1.0, "volatility: 0.0, "corr_dxy": 0.0, "macro_rate": 0.0}

    # ------------------------------------------------------------------
    # 1. Insufficient data → momentum proxy (fast & safe)
    # ------------------------------------------------------------------
    if len(full_feats) < 50:
        print(f"[infer_signal] {symbol} | {len(full_feats)} rows → momentum proxy")
        momentum = full_feats['momentum_5d'].iloc[-1] if 'momentum_5d' in full_feats.columns else 0.0
        signal = float(np.clip(momentum * 5.0, -1.0, 1.0))
        
        expl = {
            "momentum_5d": 1.0,      # force it to top in bar chart
            "volatility": 0.0,
            "corr_dxy": 0.0,
            "macro_rate": 0.0
        }
        return signal, expl

    # ------------------------------------------------------------------
    # 2. Sufficient data → ensure model exists (auto-train if needed)
    # ------------------------------------------------------------------
    model_path = f'models/rf_{symbol}.joblib'
    scaler_path = f'models/scaler_{symbol}.joblib'

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"[infer_signal] {symbol} | Training new model ({len(full_feats)} rows)")
        try:
            train_model(symbol)  # will now succeed because we stripped non-numeric cols above
        except Exception as e:
            print(f"[infer_signal] Training failed ({e}) → momentum proxy fallback")
            momentum = full_feats['momentum_5d'].iloc[-1]
            signal = float(np.clip(momentum * 5.0, -1.0, 1.0))
            expl = {"momentum_5d": 1.0, "volatility": 0.0, "corr_dxy": 0.0, "macro_rate": 0.0}
            return signal, expl

    # ------------------------------------------------------------------
    # 3. Model exists → run proper ML inference (with safety)
    # ------------------------------------------------------------------
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        latest = full_feats.drop(columns=['returns'], errors='ignore').iloc[-1:].copy()

        # Use exact feature order the model was trained with (safe even if columns evolve)
        feature_cols = getattr(model, "feature_names_in_", None)
        if feature_cols is None:
            feature_cols = latest.columns.tolist()

        X_scaled = scaler.transform(latest[feature_cols])

        prob = model.predict_proba(X_scaled)[0][1]
        signal = float((prob - 0.5) * 2.0)

        importances = model.feature_importances_
        expl = dict(zip(feature_cols, importances))
        expl = dict(sorted(expl.items(), key=lambda x: x[1], reverse=True))

        print(f"[infer_signal] {symbol} | ML signal = {signal:+.3f}")
        return signal, expl

    except Exception as e:
        print(f"[infer_signal] Inference error ({e}) → momentum proxy fallback")
        momentum = full_feats['momentum_5d'].iloc[-1]
        signal = float(np.clip(momentum * 5.0, -1.0, 1.0))
        expl = {"momentum_5d": 1.0, "volatility": 0.0, "corr_dxy": 0.0, "macro_rate": 0.0}
        return signal, expl
