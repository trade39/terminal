# ops/backtest.py (FULL FINAL - Simple P&L)
import pandas as pd
from models.infer import infer_signal
from features.engineer import engineer_features

def simple_backtest(symbol: str, feats: pd.DataFrame) -> float:
    if len(feats) < 100:
        return 0.0  # Too little data
    signals = [1 if infer_signal(symbol)[0] > 0 else -1 for _ in range(len(feats) - 1)]
    returns = feats['returns'].iloc[1:]
    pnl = (pd.Series(signals) * returns).cumsum().iloc[-1]
    print(f"Backtest P&L for {symbol}: {pnl:.2%}")
    return pnl
