# src/ingest/ohlc_fetcher.py (FULL FIXED - Polygon timestamp + dynamic dates + production hardened)
import os
import pandas as pd
import yfinance as yf
from alpha_vantage.foreignexchange import ForeignExchange
from datetime import datetime
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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_av(symbol: str, api_key: str) -> Optional[pd.DataFrame]:
    if not api_key:
        logger.info("No AV key — skipping to next source")
        return None
    try:
        fx = ForeignExchange(key=api_key)
        data, _ = fx.get_currency_exchange_daily(symbol if symbol == 'DXY' else f"{symbol}=X")
        if not data:
            return None
        df = pd.DataFrame(data).T.astype(float).rename(columns={
            '1. open': 'open', '2. high': 'high', '3. low': 'low',
            '4. close': 'close', '5. volume': 'volume'
        })
        df.index = pd.to_datetime(df.index)
        df['symbol'] = symbol
        df['source'] = 'AV'
        df = df.reset_index().rename(columns={'index': 'timestamp'})
        return df[['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        logger.warning(f"AV failed for {symbol}: {e}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_polygon(symbol: str, api_key: str) -> Optional[pd.DataFrame]:
    if not POLYGON_AVAILABLE or not api_key:
        logger.info("No Polygon key or client — skipping to Yahoo")
        return None
    try:
        client = RESTClient(api_key)
        ticker = SYMBOL_MAP.get(symbol, symbol)
        to_date = datetime.utcnow().strftime('%Y-%m-%d')
        aggs = list(client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_="2020-01-01",
            to=to_date,
            adjusted=True,
            limit=50000
        ))
        if not aggs:
            return None
        df = pd.DataFrame([{
            'timestamp': pd.to_datetime(a.timestamp, unit='ms'),  # CRITICAL FIX
            'open': a.open,
            'high': a.high,
            'low': a.low,
            'close': a.close,
            'volume': a.volume,
            'symbol': symbol,
            'source': 'Polygon'
        } for a in aggs])
        return df
    except Exception as e:
        logger.warning(f"Polygon failed for {symbol}: {e}")
        return None

@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=5, max=30))
def fetch_yahoo(symbol: str) -> pd.DataFrame:
    try:
        ticker = SYMBOL_MAP.get(symbol, symbol)
        df = yf.download(ticker, start='2020-01-01', progress=False, interval='1d')  # end=None → latest
        if df.empty:
            raise ValueError("Yahoo returned no data")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df['symbol'] = symbol
        df['source'] = 'Yahoo'
        df.rename(columns={
            'Date': 'timestamp', 'Open': 'open', 'High': 'high',
            'Low': 'low', 'Close': 'close', 'Volume': 'volume'
        }, inplace=True)
        df = df[['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
        return df
    except Exception as e:
        logger.error(f"Yahoo failed for {symbol}: {e}")
        raise

def fetch_ohlc(symbol: str, days: int = 1000) -> pd.DataFrame:
    df = fetch_av(symbol, API_KEY_AV)
    if df is None or df.empty:
        df = fetch_polygon(symbol, API_KEY_POLYGON)
    if df is None or df.empty:
        df = fetch_yahoo(symbol)
    if df.empty:
        raise ValueError(f"No data retrieved for {symbol} from any source")
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.tail(days)  # keep only recent N days for efficiency
    logger.info(f"Fetched {len(df)} bars for {symbol} (source: {df['source'].iloc[0] if len(df)>0 else 'N/A'})")
    return df
