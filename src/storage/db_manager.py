# src/storage/db_manager.py (snippet)
import sqlite3
import pandas as pd
from typing import List
from sqlalchemy import create_engine, text
from utils.config import DB_PATH

engine = create_engine(f'sqlite:///{DB_PATH}')

def store_ohlc(df: pd.DataFrame) -> None:
    """Upsert raw OHLC to DB."""
    if df.empty:
        return
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    with engine.connect() as conn:
        # Upsert via INSERT OR REPLACE
        upsert_sql = text("""
            INSERT OR REPLACE INTO raw_ohlc (symbol, timestamp, open, high, low, close, volume, source)
            VALUES (:symbol, :timestamp, :open, :high, :low, :close, :volume, :source)
        """)
        for _, row in df.iterrows():
            conn.execute(upsert_sql, row.to_dict())
        conn.commit()
    print(f"Stored {len(df)} OHLC rows.")

def store_fundamentals(df: pd.DataFrame) -> None:
    """Similar for fundamentals."""
    if df.empty:
        return
    df['date'] = pd.to_datetime(df['date']).dt.date
    with engine.connect() as conn:
        upsert_sql = text("""
            INSERT OR REPLACE INTO fundamentals (date, metric, value)
            VALUES (:date, :metric, :value)
        """)
        for _, row in df.iterrows():
            conn.execute(upsert_sql, row.to_dict())
        conn.commit()
    print(f"Stored {len(df)} fundamental rows.")

# Load func for completeness
def load_ohlc(symbol: str, start_date: str) -> pd.DataFrame:
    query = text("SELECT * FROM raw_ohlc WHERE symbol = :symbol AND timestamp >= :start_date ORDER BY timestamp")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={'symbol': symbol, 'start_date': start_date})
    return df
