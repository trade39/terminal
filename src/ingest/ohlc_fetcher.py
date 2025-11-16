# src/ingest/ohlc_fetcher.py (FULL FINAL - Support full_history via days=None)
import os
import pandas as pd
import yfinance as yf
from alpha_vantage.foreignexchange import ForeignExchange
try:
    from polygon import RESTClient
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False
    print("Polygon not available — using Yahoo fallback")

from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from dotenv import load_dotenv
import time

load_dotenv()
API_KEY_AV = os.getenv('ALPHA_VANTAGE_KEY', '')
API_KEY_POLYGON = os.getenv('POLYGON_KEY', '')

SYMBOL_MAP = {
    'DXY': 'DX-Y.NYB',
    'XAUUSD': 'GC=F',
    'ES': 'ES=F',
    'NQ': 'NQ=F',
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X'
}

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=5, max=30))
def fetch_yahoo(symbol: str) -> pd.DataFrame:
    try:
        ticker = SYMBOL_MAP.get(symbol, symbol)
        # Removed hard-coded end date — always fetch everything available
        df = yf.download(ticker, start='2000-01-01', progress=False)
        if df.empty:
            raise ValueError("Yahoo no data")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df['symbol'] = symbol
        df['source'] = 'Yahoo'
        df.reset_index(inplace=True)
        df.rename(columns={'Date': 'timestamp', 'Open': 'open', 'High': 'high', 'Low': 'low',
                           'Close': 'close', 'Volume': 'volume'}, inplace=True)
        return df[['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        logger.error(f"Yahoo failed for {symbol}: {e}")
        raise

def fetch_ohlc(symbol: str, days: Optional[int] = 2000) -> pd.DataFrame:
    df = fetch_av(symbol, API_KEY_AV)
    if df is None or df.empty:
        df = fetch_polygon(symbol, API_KEY_POLYGON)
    if df is None or df.empty:
        df = fetch_yahoo(symbol)
    if df.empty:
        raise ValueError(f"No data for {symbol}")
    # Removed tail — no tail, get full history for better training / optional recent for display
    if days is not None:
        df = df.tail(days)
    df = df.sort_values('timestamp').reset_index(drop=True)
    logger.info(f"Fetched {len(df)} bars for {symbol}")
    return df
