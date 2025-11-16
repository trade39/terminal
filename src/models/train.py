# src/models/train.py (FULL FINAL - Lowered min data to 30, adaptive CV splits)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Dict
from features.engineer import engineer_features
from sqlalchemy import text
from storage.db_manager import engine
import os

os.makedirs('models', exist_ok=True)

def train_model(symbol: str, target_col: str = 'target') -> Dict:
    try:
        feats = engineer_features(symbol)
        if feats.empty or len(feats) < 30:  # FIXED: Lowered threshold to 30 for more flexibility
            raise ValueError("Insufficient data for training")
        feats['target'] = (feats['returns'].shift(-1) > 0).astype(int)
        feats = feats.dropna(subset=['target'])  # Only drop target NaNs
        if len(feats) < 30:  # FIXED: Consistent check
            raise ValueError("Insufficient data after target shift")
        
        X = feats.drop(['returns', 'target', 'symbol', 'timestamp'], axis=1, errors='ignore')  # FIXED: Drop non-features too
        y = feats['target']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        n_splits = max(2, min(5, len(X)//10))  # FIXED: At least 2 splits, adaptive
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            scores.append(accuracy_score(y_val, y_pred))
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_scaled, y)
        joblib.dump(model, f'models/rf_{symbol}.joblib')
        joblib.dump(scaler, f'models/scaler_{symbol}.joblib')
        
        metrics = {'cv_accuracy': np.mean(scores) if scores else 0.5, 'n_features': X.shape[1], 'n_samples': len(X)}
        
        try:
            with engine.connect() as conn:
                conn.execute(text("INSERT INTO model_metadata (model_name, version, metrics) VALUES (:name, :ver, :metrics)"),
                             {'name': f'RF_{symbol}', 'ver': '1.0', 'metrics': str(metrics)})
                conn.commit()
        except:
            pass
        
        print(f"Trained RF for {symbol}: CV Acc {metrics['cv_accuracy']:.2f} on {metrics['n_samples']} samples")
        return metrics
    except Exception as e:
        print(f"Training failed for {symbol}: {e}")
        raise
