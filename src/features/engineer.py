# src/features/engineer.py (UPDATED - Uses utils.config for ASSETS)
import pandas as pd
import numpy as np
from storage.db_manager import load_ohlc, load_fundamentals
from typing import Dict, List
from utils.config import ASSETS  # Now safe
from sqlalchemy import create_engine
from utils.config import DB_PATH
engine = create_engine(f'sqlite:///{DB_PATH}')

def engineer_features(symbol: str, window: int = 20) -> pd.DataFrame:
    """Pipeline: Load → Compute → Join → Store."""
    df = load_ohlc(symbol, '2020-01-01')
    if df.empty:
        raise ValueError(f"No data for {symbol}")
    
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window).std() * np.sqrt(252)
    df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    
    # Rolling corr with DXY
    dxy_df = load_ohlc('DXY', '2020-01-01')
    dxy_returns = dxy_df.set_index('timestamp')['close'].pct_change()
    df_returns = df.set_index('timestamp')['returns']
    df['corr_dxy'] = df_returns.rolling(window).corr(dxy_returns.reindex(df_returns.index).fillna(0))
    
    # Macro join mock (full FRED later)
    df['macro_rate'] = 5.33  # Static for now
    
    feats = df[['symbol', 'timestamp', 'returns', 'volatility', 'momentum_5d', 'corr_dxy', 'macro_rate']].dropna()
    feats.to_sql('features', engine, if_exists='append', index=False)
    print(f"Engineered {len(feats)} features for {symbol}")
    return feats.set_index('timestamp')
