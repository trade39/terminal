# src/ingest/ohlc_fetcher.py (UPDATED - Uses utils.config)
import pandas as pd
import yfinance as yf
from alpha_vantage.foreignexchange import ForeignExchange
from polygon import RESTClient
from typing import Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from utils.config import API_KEY_AV, API_KEY_POLYGON, SYMBOL_MAP  # Now safe

# Hard-code SYMBOL_MAP here too for resilience
SYMBOL_MAP = {
    'DXY': 'DX-Y.NYB', 'XAUUSD': 'GC=F', 'ES': 'ES=F', 
    'NQ': 'NQ=F', 'EURUSD': 'EURUSD=X', 'GBPUSD': 'GBPUSD=X'
}

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_av(symbol: str, api_key: str) -> Optional[pd.DataFrame]:
    if not api_key:
        return None  # Skip if no key
    try:
        fx = ForeignExchange(key=api_key)
        data, _ = fx.get_currency_exchange_daily_from_symbol(f"{symbol}=X" if symbol in ['EURUSD', 'GBPUSD'] else symbol)
        if not data:
            return None
        df = pd.DataFrame(data).T.astype(float).rename(columns={
            '1. open': 'open', '2. high': 'high', '3. low': 'low',
            '4. close': 'close', '5. volume': 'volume'
        })
        df.index = pd.to_datetime(df.index)
        df['symbol'] = symbol
        df['source'] = 'AV'
        return df[['symbol', 'open', 'high', 'low', 'close', 'volume', 'timestamp']].reset_index().rename(columns={'index': 'timestamp'})
    except Exception as e:
        logger.warning(f"AV fetch failed for {symbol}: {e}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_polygon(symbol: str, api_key: str) -> Optional[pd.DataFrame]:
    if not api_key:
        return None
    try:
        client = RESTClient(api_key)
        ticker = SYMBOL_MAP.get(symbol, symbol)
        aggs = list(client.get_aggs(ticker, 1, "day", "2020-01-01", "2025-11-15", limit=50000))
        df = pd.DataFrame([{
            'timestamp': pd.Timestamp(a.timestamp), 'open': a.open, 'high': a.high,
            'low': a.low, 'close': a.close, 'volume': a.volume, 'symbol': symbol, 'source': 'Polygon'
        } for a in aggs])
        return df.set_index('timestamp').reset_index()
    except Exception as e:
        logger.warning(f"Polygon fetch failed for {symbol}: {e}")
        return None

def fetch_yahoo(symbol: str) -> pd.DataFrame:
    try:
        ticker = SYMBOL_MAP.get(symbol, symbol)
        df = yf.download(ticker, start='2020-01-01', end='2025-11-15', progress=False)
        if df.empty:
            raise ValueError("No data")
        df['symbol'] = symbol
        df['source'] = 'Yahoo'
        df.reset_index(inplace=True)
        df.rename(columns={'Date': 'timestamp', 'Open': 'open', 'High': 'high', 'Low': 'low',
                           'Close': 'close', 'Volume': 'volume'}, inplace=True)
        return df[['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        logger.error(f"Yahoo fallback failed for {symbol}: {e}")
        raise

def fetch_ohlc(symbol: str, days: int = 1000) -> pd.DataFrame:
    """Multi-source fetch with fallbacks."""
    df = fetch_av(symbol, API_KEY_AV)
    if df is None or df.empty:
        df = fetch_polygon(symbol, API_KEY_POLYGON)
    if df is None or df.empty:
        df = fetch_yahoo(symbol)
    if df.empty:
        raise ValueError(f"No data for {symbol}")
    df = df.tail(days).sort_values('timestamp')
    logger.info(f"Fetched {len(df)} bars for {symbol}")
    return df
