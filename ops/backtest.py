# ops/backtest.py (hook)
from models.infer import infer_signal
import pandas as pd

def simple_backtest(symbol: str, feats: pd.DataFrame) -> float:
    """Map signals to cumulative P&L."""
    signals = []
    for i in range(len(feats) - 1):
        _, expl = infer_signal(symbol)  # Re-infer on window (simplified)
        signals.append(1 if expl.get('momentum_5d', 0) > 0.01 else -1)  # Threshold
    returns = feats['returns'].iloc[1:]
    pnl = (pd.Series(signals) * returns).cumsum().iloc[-1]
    print(f"Backtest P&L for {symbol}: {pnl:.2%}")
    return pnl
