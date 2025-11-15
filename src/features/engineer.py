# src/features/engineer.py
import pandas as pd
import numpy as np
from storage.db_manager import load_ohlc, load_fundamentals  # Assume similar load for funds
from typing import Dict, List
from utils.config import ASSETS

def engineer_features(symbol: str, window: int = 20) -> pd.DataFrame:
    """Pipeline: Load → Compute → Join → Store."""
    df = load_ohlc(symbol, '2020-01-01')
    if df.empty:
        raise ValueError(f"No data for {symbol}")
    
    # Returns, vol, momentum
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window).std() * np.sqrt(252)
    df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    
    # Rolling corr with DXY
    dxy = load_ohlc('DXY', '2020-01-01')['returns']
    df['corr_dxy'] = df['returns'].rolling(window).corr(dxy.reindex(df.index).fillna(0))
    
    # Macro join (e.g., FEDFUNDS)
    funds = load_fundamentals().query("metric == 'FEDFUNDS'")
    df = df.merge(funds[['date', 'value']].rename(columns={'value': 'macro_rate', 'date': 'timestamp'}), on='timestamp', how='left')
    df['macro_rate'] = df['macro_rate'].fillna(method='ffill')
    
    feats = df[['symbol', 'timestamp', 'returns', 'volatility', 'momentum_5d', 'corr_dxy', 'macro_rate']].dropna()
    feats.to_sql('features', engine, if_exists='append', index=False)  # From sqlalchemy
    print(f"Engineered {len(feats)} features for {symbol}")
    return feats.set_index('timestamp')
