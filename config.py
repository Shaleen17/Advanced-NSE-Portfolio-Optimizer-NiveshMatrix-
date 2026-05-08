"""Central configuration for the Advanced NSE Portfolio Optimizer project."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

PROJECT_NAME = "Advanced NSE Portfolio Optimizer"
BRAND_NAME = "NiveshMatrix"

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DATA_DIR = DATA_DIR / "outputs"
CACHE_DATA_DIR = DATA_DIR / "cache"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MPL_CONFIG_DIR = BASE_DIR / ".matplotlib"
LOGO_PATH = ASSETS_DIR / "Brand_Logo.png"
CLEANED_PRICE_FILE = PROCESSED_DATA_DIR / "nse_adjusted_close_cleaned.csv"
BENCHMARK_RETURNS_FILE = OUTPUT_DATA_DIR / "nifty50_benchmark_returns.csv"
LIVE_QUOTES_CACHE_FILE = CACHE_DATA_DIR / "live_quotes_latest.csv"

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

TRADING_DAYS = 252
RISK_FREE_RATE = 0.06
DEFAULT_START_DATE = date(2021, 1, 1)
DEFAULT_END_DATE = date.today()
BENCHMARK_TICKER = "^NSEI"
DEFAULT_TRANSACTION_COST = 0.001

CHART_COLORS = {
    "background": "#000000",
    "text": "#FFFFFF",
    "profit": "#00FF00",
    "loss": "#FF0000",
    "panel": "#101010",
    "panel_2": "#171717",
    "border": "#2A2A2A",
    "muted": "#B8B8B8",
}

NSE_TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "ITC.NS",
    "HINDUNILVR.NS",
    "BHARTIARTL.NS",
    "ASIANPAINT.NS",
    "MARUTI.NS",
    "TITAN.NS",
    "SUNPHARMA.NS",
    "CIPLA.NS",
    "DRREDDY.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "HCLTECH.NS",
    "WIPRO.NS",
    "TECHM.NS",
    "ULTRACEMCO.NS",
    "NESTLEIND.NS",
    "POWERGRID.NS",
    "NTPC.NS",
    "ONGC.NS",
    "COALINDIA.NS",
    "TATASTEEL.NS",
    "JSWSTEEL.NS",
    "HINDALCO.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "GRASIM.NS",
    "M&M.NS",
    "EICHERMOT.NS",
    "HEROMOTOCO.NS",
    "BAJAJ-AUTO.NS",
    "BRITANNIA.NS",
    "DIVISLAB.NS",
    "APOLLOHOSP.NS",
    "BPCL.NS",
    "IOC.NS",
    "TATAMOTORS.NS",
    "TATACONSUM.NS",
    "UPL.NS",
    "SHREECEM.NS",
    "INDUSINDBK.NS",
    "SBILIFE.NS",
]


def ensure_project_folders() -> None:
    """Create the folders used by the application if they are missing."""
    for folder in [
        ASSETS_DIR,
        DATA_DIR,
        CACHE_DATA_DIR,
        PROCESSED_DATA_DIR,
        OUTPUT_DATA_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        MPL_CONFIG_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)
