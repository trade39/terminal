# ops/backtest.py (FULL - Hook for P&L)
from models.infer import infer_signal
import pandas as pd

def simple_backtest(symbol: str, feats: pd.DataFrame) -> float:
    """Map signals to cumulative P&L."""
    signals = []
    for i in range(len(feats) - 1):
        signal, _ = infer_signal(symbol)  # Simplified re-infer
        signals.append(1 if signal > 0 else -1)
    returns = feats['returns'].iloc[1:]
    pnl = (pd.Series(signals) * returns).cumsum().iloc[-1]
    print(f"Backtest P&L for {symbol}: {pnl:.2%}")
    return pnl
