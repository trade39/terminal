# src/storage/db_manager.py (FULL FIXED - executemany bulk upsert, production-safe)
import os
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from utils.config import DB_PATH

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

engine = create_engine(f'sqlite:///{DB_PATH}', connect_args={"check_same_thread": False})

# schema init unchanged...

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

# src/storage/db_manager.py (FULL FIXED - executemany bulk upsert, production-safe)
import os
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from utils.config import DB_PATH

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

engine = create_engine(f'sqlite:///{DB_PATH}', connect_args={"check_same_thread": False})

# schema init unchanged...

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
            conn.executemany(upsert_sql, records)  # ← CRITICAL: executemany, single transaction
            conn.commit()
        print(f"Bulk stored/updated {len(df)} rows for {df['symbol'].iloc[0]}")
    except Exception as e:
        print(f"Bulk store error: {e}")
        raise
