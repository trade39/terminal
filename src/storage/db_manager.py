# src/storage/db_manager.py (FULL FINAL - Removed to_numeric)
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
    # Ensure volume is numeric (some sources return float)
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
    if 'source' not in df.columns:
        df['source'] = 'Unknown'
    
    records = df[['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'source']].to_dict(orient='records')
    
    upsert_sql = text("""
        INSERT OR REPLACE INTO raw_ohlc 
        (symbol, timestamp, open, high, low, close, volume, source)
        VALUES (:symbol, :timestamp, :open, :high, :low, :close, :volume, :source)
    """)
    
    try:
        with engine.connect() as conn:
            conn.execute(upsert_sql, records)  # ← executemany, single transaction
            conn.commit()
        print(f"Bulk stored/updated {len(df)} rows for {df['symbol'].iloc[0]}")
    except Exception as e:
        print(f"Bulk store error: {e}")
        raise
