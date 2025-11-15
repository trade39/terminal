# ops/backtest.py (FULL FINAL - Renamed to backtest.py if needed, but with path fix it's ok)
import pandas as pd
from models.infer import infer_signal
from features.engineer import engineer_features

def simple_backtest(symbol: str, feats: pd.DataFrame) -> float:
    if len(feats) < 50:
        return 0.0
    # Simplified: Use momentum as proxy for signal if infer fails
    signals = (feats['momentum_5d'] > 0).astype(int) * 2 - 1  # -1 to 1
    signals = signals.iloc[:-1]  # Align with returns
    returns = feats['returns'].iloc[1:]
    pnl = (signals * returns).cumsum().iloc[-1]
    print(f"Backtest P&L for {symbol}: {pnl:.2%}")
    return pnl
