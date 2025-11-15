# src/features/engineer.py (FULL FINAL - Fill NaNs to avoid empty feats)
import pandas as pd
import numpy as np
from storage.db_manager import load_ohlc
from sqlalchemy import create_engine
from utils.config import DB_PATH
engine = create_engine(f'sqlite:///{DB_PATH}')

def engineer_features(symbol: str, window: int = 20) -> pd.DataFrame:
    df = load_ohlc(symbol, '2020-01-01')
    if df.empty:
        # Return dummy feats for safe ML (use mean values or 0)
        dummy = pd.DataFrame({
            'returns': [0.0],
            'volatility': [0.15],
            'momentum_5d': [0.0],
            'corr_dxy': [0.0],
            'macro_rate': [5.33]
        }, index=[pd.Timestamp.now()])
        print(f"No data for {symbol} — using dummy features")
        return dummy
    
    df = df.copy()
    df['returns'] = df['close'].pct_change().fillna(0)
    df['volatility'] = df['returns'].rolling(window).std() * np.sqrt(252).fillna(0.15)  # Default vol 15%
    df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_5d'] = df['momentum_5d'].fillna(0)
    
    dxy_df = load_ohlc('DXY', '2020-01-01')
    if not dxy_df.empty:
        dxy_returns = dxy_df.set_index('timestamp')['close'].pct_change().fillna(0)
        df_returns = df.set_index('timestamp')['returns']
        df['corr_dxy'] = df_returns.rolling(window).corr(dxy_returns.reindex(df_returns.index).fillna(0)).fillna(0)
    else:
        df['corr_dxy'] = 0.0
    
    df['macro_rate'] = 5.33
    
    feats = df[['symbol', 'timestamp', 'returns', 'volatility', 'momentum_5d', 'corr_dxy', 'macro_rate']]
    feats = feats.fillna(0)  # FIXED: Fill NaNs instead of dropna to avoid empty
    feats.to_sql('features', engine, if_exists='append', index=False)
    print(f"Engineered {len(feats)} features for {symbol}")
    return feats.set_index('timestamp').tail(1000)  # Limit for training speed
