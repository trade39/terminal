# src/features/engineer.py (FULL FINAL - No raise, return empty if no data)
import pandas as pd
import numpy as np
from storage.db_manager import load_ohlc
from sqlalchemy import create_engine
from utils.config import DB_PATH
engine = create_engine(f'sqlite:///{DB_PATH}')

def engineer_features(symbol: str, window: int = 20) -> pd.DataFrame:
    df = load_ohlc(symbol, '2020-01-01')
    if df.empty:
        # Return empty with columns for safe ML
        return pd.DataFrame(columns=['returns', 'volatility', 'momentum_5d', 'corr_dxy', 'macro_rate'])
    
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window).std() * np.sqrt(252)
    df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    
    dxy_df = load_ohlc('DXY', '2020-01-01')
    if not dxy_df.empty:
        dxy_returns = dxy_df.set_index('timestamp')['close'].pct_change()
        df_returns = df.set_index('timestamp')['returns']
        df['corr_dxy'] = df_returns.rolling(window).corr(dxy_returns.reindex(df_returns.index).fillna(0))
    else:
        df['corr_dxy'] = 0.0
    
    df['macro_rate'] = 5.33
    
    feats = df[['symbol', 'timestamp', 'returns', 'volatility', 'momentum_5d', 'corr_dxy', 'macro_rate']].dropna()
    if not feats.empty:
        feats.to_sql('features', engine, if_exists='append', index=False)
        print(f"Engineered {len(feats)} features for {symbol}")
    else:
        feats = pd.DataFrame(columns=['returns', 'volatility', 'momentum_5d', 'corr_dxy', 'macro_rate'])
    return feats.set_index('timestamp')
