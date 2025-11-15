# src/storage/db_manager.py (FULL FIXED - Auto-creates dir, DB, and schema)
import os
import sqlite3
import pandas as pd
from typing import List
from sqlalchemy import create_engine, text
from utils.config import DB_PATH  # Safe now

# Ensure data dir exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Create engine with check_same_thread=False for SQLite in multi-thread (Streamlit)
engine = create_engine(f'sqlite:///{DB_PATH}', connect_args={"check_same_thread": False})

def init_schema_if_needed():
    """Lazy init schema if tables missing."""
    inspector = sqlalchemy.inspect(engine)
    if not inspector.has_table('raw_ohlc'):
        schema_sql = """
        -- Raw OHLC table
        CREATE TABLE raw_ohlc (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol VARCHAR(10) NOT NULL,
            timestamp DATETIME NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            source VARCHAR(20),
            UNIQUE(symbol, timestamp)
        );

        -- Fundamentals table
        CREATE TABLE fundamentals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL,
            metric VARCHAR(50) NOT NULL,
            value REAL,
            UNIQUE(metric, date)
        );

        -- Features table
        CREATE TABLE features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol VARCHAR(10) NOT NULL,
            date DATE NOT NULL,
            returns REAL,
            volatility REAL,
            momentum_5d REAL,
            corr_dxy REAL,
            macro_rate REAL,
            UNIQUE(symbol, date)
        );

        -- Model metadata
        CREATE TABLE model_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name VARCHAR(50) NOT NULL,
            version VARCHAR(20),
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            params TEXT,
            metrics TEXT
        );
        """
        with engine.connect() as conn:
            conn.execute(text(schema_sql))
            conn.commit()
        print("Schema initialized.")

# Call init on module load
try:
    init_schema_if_needed()
except Exception as e:
    print(f"Schema init error: {e}")

def store_ohlc(df: pd.DataFrame) -> None:
    """Upsert raw OHLC to DB with error handling."""
    if df.empty:
        return
    df = df.copy()
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
    df = df.copy()
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
        return df if not df.empty else pd.DataFrame()
    except Exception as e:
        print(f"DB load error: {e}")
        return pd.DataFrame()  # Empty fallback
