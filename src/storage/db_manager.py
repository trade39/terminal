# src/storage/db_manager.py (FIXED - Direct .env load, no utils.config dependency)
import sqlite3
import pandas as pd
from typing import List
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

# Direct load of .env (no utils.config needed)
load_dotenv()
DB_PATH = os.getenv('DB_PATH', 'data/quant_terminal.db')

engine = create_engine(f'sqlite:///{DB_PATH}')

def store_ohlc(df: pd.DataFrame) -> None:
    """Upsert raw OHLC to DB with error handling."""
    if df.empty:
        return
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    try:
        with engine.connect() as conn:
            upsert_sql = text("""
                INSERT OR REPLACE INTO raw_ohlc (symbol, timestamp, open, high, low, close, volume, source)
                VALUES (:symbol, :timestamp, :open, :high, :low, :close, :volume, :source)
            """)
            for _, row in df.iterrows():
                conn.execute(upsert_sql, row.to_dict())
            conn.commit()
        print(f"Stored {len(df)} OHLC rows.")
    except Exception as e:
        print(f"DB store error: {e}")

def store_fundamentals(df: pd.DataFrame) -> None:
    """Upsert fundamentals."""
    if df.empty:
        return
    df['date'] = pd.to_datetime(df['date']).dt.date
    try:
        with engine.connect() as conn:
            upsert_sql = text("""
                INSERT OR REPLACE INTO fundamentals (date, metric, value)
                VALUES (:date, :metric, :value)
            """)
            for _, row in df.iterrows():
                conn.execute(upsert_sql, row.to_dict())
            conn.commit()
        print(f"Stored {len(df)} fundamental rows.")
    except Exception as e:
        print(f"DB store error: {e}")

def load_ohlc(symbol: str, start_date: str) -> pd.DataFrame:
    """Load OHLC with error handling."""
    try:
        query = text("SELECT * FROM raw_ohlc WHERE symbol = :symbol AND timestamp >= :start_date ORDER BY timestamp")
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={'symbol': symbol, 'start_date': start_date})
        return df
    except Exception as e:
        print(f"DB load error: {e}")
        return pd.DataFrame()  # Empty fallback
