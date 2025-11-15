# src/storage/db_manager.py (FULL FINAL - Convert datetime on load)
import os
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from utils.config import DB_PATH

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

engine = create_engine(f'sqlite:///{DB_PATH}', connect_args={"check_same_thread": False})

def init_schema_if_needed():
    inspector = inspect(engine)
    if not inspector.has_table('raw_ohlc'):
        schema_sql = """
        CREATE TABLE IF NOT EXISTS raw_ohlc (
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

        CREATE TABLE IF NOT EXISTS fundamentals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL,
            metric VARCHAR(50) NOT NULL,
            value REAL,
            UNIQUE(metric, date)
        );

        CREATE TABLE IF NOT EXISTS features (
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

        CREATE TABLE IF NOT EXISTS model_metadata (
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
        print("Schema created.")

try:
    init_schema_if_needed()
except Exception as e:
    print(f"Schema error: {e}")

def store_ohlc(df: pd.DataFrame) -> None:
    if df.empty:
        return
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=['close'])
    try:
        with engine.connect() as conn:
            upsert_sql = text("""
                INSERT OR REPLACE INTO raw_ohlc (symbol, timestamp, open, high, low, close, volume, source)
                VALUES (:symbol, :timestamp, :open, :high, :low, :close, :volume, :source)
            """)
            for _, row in df.iterrows():
                conn.execute(upsert_sql, row.to_dict())
            conn.commit()
        print(f"Stored {len(df)} rows.")
    except Exception as e:
        print(f"Store error: {e}")

def load_ohlc(symbol: str, start_date: str) -> pd.DataFrame:
    try:
        query = text("SELECT * FROM raw_ohlc WHERE symbol = :symbol AND timestamp >= :start_date ORDER BY timestamp")
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={'symbol': symbol, 'start_date': start_date})
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df = df.dropna(subset=['close'])
        return df
    except Exception as e:
        print(f"Load error: {e}")
        return pd.DataFrame()
