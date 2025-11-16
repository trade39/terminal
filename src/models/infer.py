# src/models/infer.py (PRODUCTION-FIXED v2 - Never crashes, graceful fallback, no forced training on dummy data)

import os
import pandas as pd
import joblib
import numpy as np
from typing import Dict, Tuple

from features.engineer import engineer_features
from models.train import train_model

def infer_signal(symbol: str) -> Tuple[float, Dict]:
    """
    Returns a robust signal even when:
    - No historical data exists yet (fresh deploy / failed fetch)
    - Less than 50 rows (training would fail)
    - Model files missing or corrupted
    - Scaler/predict_proba fails
    → Falls back to simple momentum proxy so the terminal never shows "ML inference failed"
    """
    # Get the FULL feature history first — this tells us if we have real data
    full_feats = engineer_features(symbol)
    
    # ------------------------------------------------------------------
    # 1. Insufficient data → momentum proxy (very fast, no model needed)
    # ------------------------------------------------------------------
    if len(full_feats) < 50:
        print(f"[infer_signal] {symbol} | Insufficient data ({len(full_feats)} rows) → momentum proxy")
        
        momentum = full_feats['momentum_5d'].iloc[-1] if 'momentum_5d' in full_feats.columns and not full_feats['momentum_5d'].isna().all() else 0.0
        # Amplify momentum slightly so signal feels meaningful even on short history
        signal = float(np.clip(momentum * 5.0, -1.0, 1.0))
        
        expl = {
            "momentum_5d": 1.0,
            "volatility": 0.0,
            "corr_dxy": 0.0,
            "macro_rate": 0.0,
            "_fallback": "insufficient_data"
        }
        return signal, expl

    # ------------------------------------------------------------------
    # 2. We have enough data → ensure model exists
    # ------------------------------------------------------------------
    model_path = f'models/rf_{symbol}.joblib'
    scaler_path = f'models/scaler_{symbol}.joblib'

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"[infer_signal] {symbol} | Model missing → training on {len(full_feats)} rows")
        try:
            train_model(symbol)  # This will now succeed because we already checked len >= 50
        except Exception as e:
            print(f"[infer_signal] Training failed ({e}) → fallback to momentum")
            momentum = full_feats['momentum_5d'].iloc[-1]
            signal = float(np.clip(momentum * 5.0, -1.0, 1.0))
            expl = {"momentum_5d": 1.0, "_fallback": "training_failed"}
            return signal, expl

    # ------------------------------------------------------------------
    # 3. Model exists → run proper inference (with safety net)
    # ------------------------------------------------------------------
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        # Use only the latest row for live signal
        latest = full_feats.drop(columns=['returns', 'target'], errors='ignore').iloc[-1:].copy()

        # Ensure column order matches what the model was trained on
        feature_cols = [col for col in model.feature_names_in_ if col in latest.columns]  # RF has feature_names_in_
        X_scaled = scaler.transform(latest[feature_cols])

        prob = model.predict_proba(X_scaled)[0][1]
        signal = float((prob - 0.5) * 2.0)  # -1 to +1

        importances = model.feature_importances_
        expl = dict(zip(feature_cols, importances))
        expl = dict(sorted(expl.items(), key=lambda x: x[1], reverse=True))

        print(f"[infer_signal] {symbol} | ML signal = {signal:+.3f}")
        return signal, expl

    except Exception as e:
        print(f"[infer_signal] Inference exception ({e}) → momentum fallback")
        momentum = full_feats['momentum_5d'].iloc[-1]
        signal = float(np.clip(momentum * 5.0, -1.0, 1.0))
        expl = {"momentum_5d": 1.0, "_fallback": "inference_error"}
        return signal, expl
