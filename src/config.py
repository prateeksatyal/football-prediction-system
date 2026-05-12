"""
Configuration and constants for the football prediction system.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DATA_DIR / "football.db"

# Create directories
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# API Configuration
FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "")  # Get free key from football-data.org
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")  # Get free key from api-football.com

FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"
API_FOOTBALL_BASE_URL = "https://v3.football.dataitech.com"

# Rate limiting
RATE_LIMIT_DELAY = 1.0  # seconds between requests
MAX_RETRIES = 3
RETRY_DELAY = 2.0

# Leagues to track (use football-data.org league codes)
TARGET_LEAGUES = {
    "PL": "Premier League",  # England
    "PD": "La Liga",  # Spain
    "SA": "Serie A",  # Italy
    "BL1": "Bundesliga",  # Germany
    "FL1": "Ligue 1",  # France
}

# Seasons to fetch (format: YYYY)
TARGET_SEASONS = ["2021", "2022", "2023", "2024", "2025", "2026"]

# Feature Engineering
ROLLING_WINDOWS = [3, 5, 10, 15, 20]  # Match windows for statistics
MIN_MATCHES_REQUIRED = 5  # Minimum historical matches required

# Model Training
TRAIN_TEST_SPLIT_DATE = "2024-01-01"  # Use data before this for training
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
RANDOM_STATE = 42

# Prediction targets
CLASSES = ["H", "D", "A"]  # Home, Draw, Away
CLASS_TO_IDX = {"H": 0, "D": 1, "A": 2}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

# Model hyperparameters (baseline - to be tuned)
XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "tree_method": "hist",
}

LIGHTGBM_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
}

# Logging
LOG_LEVEL = "INFO"