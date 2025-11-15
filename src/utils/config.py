# src/utils/config.py (NEW FILE - Minimal Config Loader)
import os
from dotenv import load_dotenv
import yaml

load_dotenv()

# Core paths and params
DB_PATH = os.getenv('DB_PATH', 'data/quant_terminal.db')

# API Keys (fallback to empty for Yahoo-only mode)
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY', '')
POLYGON_KEY = os.getenv('POLYGON_KEY', '')
FRED_API_KEY = os.getenv('FRED_API_KEY', '')

# Assets (hard-coded fallback if config.yaml missing)
ASSETS = ['DXY', 'XAUUSD', 'ES', 'NQ', 'EURUSD', 'GBPUSD']

# Load from config.yaml if exists
config_path = 'config/config.yaml'
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        ASSETS = cfg.get('assets', ASSETS)

# Other params
WINDOW_DAYS = 1000
ML_N_ESTIMATORS = 100
