# src/models/train.py (FULL FINAL - Added dir creation)
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
        if feats.empty:
            raise ValueError("No features for training")
        feats['target'] = (feats['returns'].shift(-1) > 0).astype(int)
        feats = feats.dropna()
        if len(feats) < 100:
            raise ValueError("Insufficient data for training")
        
        X = feats.drop(['returns', 'target'], axis=1)
        y = feats['target']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        tscv = TimeSeriesSplit(n_splits=5)
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
        
        metrics = {'cv_accuracy': np.mean(scores), 'n_features': X.shape[1]}
        
        try:
            with engine.connect() as conn:
                conn.execute(text("INSERT INTO model_metadata (model_name, version, metrics) VALUES (:name, :ver, :metrics)"),
                             {'name': f'RF_{symbol}', 'ver': '1.0', 'metrics': str(metrics)})
                conn.commit()
        except:
            pass
        
        print(f"Trained RF for {symbol}: CV Acc {metrics['cv_accuracy']:.2f}")
        return metrics
    except Exception as e:
        print(f"Training failed for {symbol}: {e}")
        raise
