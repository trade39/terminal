# src/utils/config.py (FULL FINAL - All exports)
import os
from dotenv import load_dotenv
import yaml

load_dotenv()

DB_PATH = os.getenv('DB_PATH', 'data/quant_terminal.db')

ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY', '')
POLYGON_KEY = os.getenv('POLYGON_KEY', '')
FRED_API_KEY = os.getenv('FRED_API_KEY', '')

ASSETS = ['DXY', 'XAUUSD', 'ES', 'NQ', 'EURUSD', 'GBPUSD']

SYMBOL_MAP = {
    'DXY': 'DX-Y.NYB',
    'XAUUSD': 'GC=F',
    'ES': 'ES=F',
    'NQ': 'NQ=F',
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X'
}

config_path = 'config/config.yaml'
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        ASSETS = cfg.get('assets', ASSETS)

WINDOW_DAYS = 1000
ML_N_ESTIMATORS = 100
