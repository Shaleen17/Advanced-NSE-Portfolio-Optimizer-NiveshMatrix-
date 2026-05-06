"""
Advanced Portfolio Optimization Dashboard for Indian NSE stocks.

Run locally:

    streamlit run dashboard/app.py

This dashboard is educational. It is not financial advice.
"""

from __future__ import annotations

import logging
import os
import warnings
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

warnings.filterwarnings("ignore", message=".*urllib3.*", category=Warning)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as error:
    RandomForestRegressor = None
    LinearRegression = None
    TimeSeriesSplit = None
    Pipeline = None
    StandardScaler = None
    SKLEARN_IMPORT_ERROR = error
else:
    SKLEARN_IMPORT_ERROR = None

# Keep yfinance cache files inside the project folder to avoid permission issues.
YFINANCE_CACHE_DIR = Path(".yfinance_cache")
YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
try:
    yf.cache.set_cache_location(str(YFINANCE_CACHE_DIR.resolve()))
except Exception:
    pass


# -----------------------------
# 1. Dashboard configuration
# -----------------------------

st.set_page_config(
    page_title="NSE Portfolio Optimizer",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

PURE_BLACK = "#000000"
PURE_WHITE = "#FFFFFF"
PURE_GREEN = "#00FF00"
PURE_RED = "#FF0000"
PANEL_BORDER = "#242424"

TRADING_DAYS = 252
DEFAULT_RISK_FREE_RATE = 0.06
MAX_RAW_MISSING_PERCENT = 0.40
BENCHMARK_TICKER = "^NSEI"
VAR_CONFIDENCE_LEVEL = 0.95
DEFAULT_TRAINING_WINDOW = 252
DEFAULT_REBALANCE_FREQUENCY = 21
DEFAULT_TRANSACTION_COST = 0.001
PREDICTION_HORIZON_DAYS = 21
RSI_PERIOD = 14
ML_TIME_SPLITS = 5
ML_MIN_DATASET_ROWS = 250
RANDOM_STATE = 42

ML_FEATURE_COLUMNS = [
    "Return 5D",
    "Return 10D",
    "Return 20D",
    "MA 50 Ratio",
    "MA 100 Ratio",
    "Volatility 20D",
    "RSI 14D",
]

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


def inject_css() -> None:
    """Force a pure black fintech-style Streamlit UI."""
    st.markdown(
        f"""
        <style>
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stSidebar"],
        [data-testid="stToolbar"],
        [data-testid="stBottomBlockContainer"] {{
            background: {PURE_BLACK} !important;
            color: {PURE_WHITE} !important;
        }}

        html, body, p, span, label, div, h1, h2, h3, h4, h5, h6,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"] {{
            color: {PURE_WHITE} !important;
        }}

        .metric-card {{
            border: 1px solid {PANEL_BORDER};
            border-radius: 8px;
            padding: 16px;
            background: #050505;
            min-height: 116px;
        }}

        .metric-label {{
            color: #B8B8B8 !important;
            font-size: 0.86rem;
            margin-bottom: 8px;
        }}

        .metric-value {{
            color: {PURE_WHITE} !important;
            font-size: 1.8rem;
            font-weight: 700;
            line-height: 1.1;
        }}

        .metric-help {{
            color: #A0A0A0 !important;
            font-size: 0.78rem;
            margin-top: 8px;
        }}

        .green-text {{
            color: {PURE_GREEN} !important;
            font-weight: 700;
        }}

        .red-text {{
            color: {PURE_RED} !important;
            font-weight: 700;
        }}

        .info-panel {{
            border: 1px solid {PANEL_BORDER};
            border-radius: 8px;
            padding: 18px;
            background: #050505;
            color: {PURE_WHITE} !important;
        }}

        .stButton > button,
        .stDownloadButton > button {{
            background: {PURE_GREEN} !important;
            color: {PURE_BLACK} !important;
            border: 1px solid {PURE_GREEN} !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
        }}

        .stButton > button:hover,
        .stDownloadButton > button:hover {{
            background: {PURE_BLACK} !important;
            color: {PURE_GREEN} !important;
            border: 1px solid {PURE_GREEN} !important;
        }}

        div[data-testid="stDataFrame"] {{
            border: 1px solid {PANEL_BORDER};
            border-radius: 8px;
        }}

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        textarea,
        input {{
            background: #070707 !important;
            color: {PURE_WHITE} !important;
            border-color: {PANEL_BORDER} !important;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 6px;
        }}

        .stTabs [data-baseweb="tab"] {{
            background: #070707;
            border: 1px solid {PANEL_BORDER};
            border-radius: 8px;
            color: {PURE_WHITE};
            padding: 10px 14px;
        }}

        .stTabs [aria-selected="true"] {{
            border-color: {PURE_GREEN};
            color: {PURE_GREEN};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# 2. Data and finance functions
# -----------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def download_price_data(
    tickers: tuple[str, ...],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Download Adjusted Close data from Yahoo Finance."""
    # yfinance treats end date as exclusive, so one day is added.
    yf_end_date = end_date + timedelta(days=1)

    data = yf.download(
        tickers=list(tickers),
        start=start_date.isoformat(),
        end=yf_end_date.isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    if data.empty:
        return pd.DataFrame()

    # Multi-ticker downloads usually produce a MultiIndex column structure.
    if isinstance(data.columns, pd.MultiIndex):
        first_level = data.columns.get_level_values(0)
        if "Adj Close" in first_level:
            prices = data["Adj Close"].copy()
        elif "Close" in first_level:
            prices = data["Close"].copy()
        else:
            return pd.DataFrame()
    else:
        # Single-ticker downloads usually produce normal OHLCV columns.
        if "Adj Close" in data.columns:
            prices = data[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
        elif "Close" in data.columns:
            prices = data[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            return pd.DataFrame()

    prices = prices.sort_index()
    prices = prices.dropna(axis=1, how="all")

    return prices


@st.cache_data(show_spinner=False, ttl=3600)
def download_benchmark_returns(start_date: date, end_date: date) -> tuple[pd.Series | None, str]:
    """Download NIFTY 50 daily returns for benchmark risk metrics."""
    try:
        data = yf.download(
            BENCHMARK_TICKER,
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception as error:
        return None, f"NIFTY 50 benchmark download failed: {error}"

    if data.empty:
        return None, "NIFTY 50 benchmark download returned empty data."

    if isinstance(data.columns, pd.MultiIndex):
        if ("Adj Close", BENCHMARK_TICKER) in data.columns:
            benchmark_prices = data[("Adj Close", BENCHMARK_TICKER)]
        elif ("Close", BENCHMARK_TICKER) in data.columns:
            benchmark_prices = data[("Close", BENCHMARK_TICKER)]
        else:
            return None, "NIFTY 50 benchmark data did not contain Adj Close or Close prices."
    else:
        if "Adj Close" in data.columns:
            benchmark_prices = data["Adj Close"]
        elif "Close" in data.columns:
            benchmark_prices = data["Close"]
        else:
            return None, "NIFTY 50 benchmark data did not contain Adj Close or Close prices."

    benchmark_returns = benchmark_prices.sort_index().pct_change().dropna()
    benchmark_returns.name = "NIFTY 50"

    if benchmark_returns.empty:
        return None, "NIFTY 50 benchmark returns are empty after calculating daily returns."

    return benchmark_returns, ""


def clean_prices(prices: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Clean price data and return warnings for removed stocks."""
    warnings = []

    if prices.empty:
        return prices, ["No usable price data was returned by Yahoo Finance."]

    missing_before = prices.isna().mean()
    keep_columns = missing_before[missing_before <= MAX_RAW_MISSING_PERCENT].index.tolist()
    removed_columns = missing_before[missing_before > MAX_RAW_MISSING_PERCENT].index.tolist()

    if removed_columns:
        warnings.append(
            "Removed tickers with too much missing data: " + ", ".join(removed_columns)
        )

    cleaned = prices[keep_columns].copy()
    cleaned = cleaned.ffill().bfill()
    cleaned = cleaned.dropna(axis=1, how="any")
    cleaned = cleaned.dropna(axis=0, how="any")

    if cleaned.empty:
        warnings.append("All selected tickers became empty after missing-value cleaning.")

    return cleaned, warnings


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily percentage returns."""
    returns = prices.pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(axis=0, how="any")
    return returns


def calculate_annual_inputs(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Calculate annual expected returns and annual covariance matrix."""
    annual_returns = returns.mean() * TRADING_DAYS
    annual_covariance = returns.cov() * TRADING_DAYS
    return annual_returns, annual_covariance


def portfolio_return(weights: np.ndarray, annual_returns: pd.Series) -> float:
    """Calculate annual portfolio return."""
    return float(weights @ annual_returns.to_numpy())


def portfolio_risk(weights: np.ndarray, annual_covariance: pd.DataFrame) -> float:
    """Calculate annual portfolio volatility."""
    covariance_values = annual_covariance.to_numpy()
    variance = float(weights.T @ covariance_values @ weights)
    return float(np.sqrt(max(variance, 0)))


def sharpe_ratio(
    weights: np.ndarray,
    annual_returns: pd.Series,
    annual_covariance: pd.DataFrame,
    risk_free_rate: float,
) -> float:
    """Calculate portfolio Sharpe ratio."""
    risk = portfolio_risk(weights, annual_covariance)
    if risk == 0:
        return -np.inf
    return (portfolio_return(weights, annual_returns) - risk_free_rate) / risk


def negative_sharpe_ratio(
    weights: np.ndarray,
    annual_returns: pd.Series,
    annual_covariance: pd.DataFrame,
    risk_free_rate: float,
) -> float:
    """SciPy minimizes, so maximize Sharpe by minimizing negative Sharpe."""
    return -sharpe_ratio(weights, annual_returns, annual_covariance, risk_free_rate)


def optimize_max_sharpe(
    annual_returns: pd.Series,
    annual_covariance: pd.DataFrame,
    risk_free_rate: float,
) -> tuple[np.ndarray | None, str]:
    """Find the maximum Sharpe portfolio with long-only constraints."""
    n_assets = len(annual_returns)
    initial_weights = np.repeat(1 / n_assets, n_assets)
    bounds = tuple((0, 1) for _ in range(n_assets))
    constraints = ({"type": "eq", "fun": lambda weights: np.sum(weights) - 1},)

    try:
        result = minimize(
            negative_sharpe_ratio,
            initial_weights,
            args=(annual_returns, annual_covariance, risk_free_rate),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
        )
    except Exception as error:
        return None, f"Maximum Sharpe optimization failed: {error}"

    if not result.success:
        return None, f"Maximum Sharpe optimization failed: {result.message}"

    return result.x, ""


def optimize_min_volatility(annual_covariance: pd.DataFrame) -> tuple[np.ndarray | None, str]:
    """Find the minimum volatility portfolio with long-only constraints."""
    n_assets = len(annual_covariance)
    initial_weights = np.repeat(1 / n_assets, n_assets)
    bounds = tuple((0, 1) for _ in range(n_assets))
    constraints = ({"type": "eq", "fun": lambda weights: np.sum(weights) - 1},)

    try:
        result = minimize(
            portfolio_risk,
            initial_weights,
            args=(annual_covariance,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
        )
    except Exception as error:
        return None, f"Minimum volatility optimization failed: {error}"

    if not result.success:
        return None, f"Minimum volatility optimization failed: {result.message}"

    return result.x, ""


def random_weights(n_assets: int, seed: int = 42) -> np.ndarray:
    """Create one random fully invested long-only portfolio."""
    rng = np.random.default_rng(seed)
    weights = rng.random(n_assets)
    return weights / weights.sum()


def run_monte_carlo(
    annual_returns: pd.Series,
    annual_covariance: pd.DataFrame,
    risk_free_rate: float,
    n_portfolios: int,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate random portfolios and calculate their metrics."""
    rng = np.random.default_rng(seed)
    n_assets = len(annual_returns)
    weights = rng.random((n_portfolios, n_assets))
    weights = weights / weights.sum(axis=1, keepdims=True)

    return_values = weights @ annual_returns.to_numpy()
    covariance_values = annual_covariance.to_numpy()
    variances = np.einsum("ij,jk,ik->i", weights, covariance_values, weights)
    risks = np.sqrt(np.maximum(variances, 0))
    sharpes = (return_values - risk_free_rate) / risks

    results = pd.DataFrame(
        {
            "Annual Return": return_values,
            "Annual Risk": risks,
            "Sharpe Ratio": sharpes,
        }
    )

    return results, weights


def calculate_strategy_row(
    strategy: str,
    weights: np.ndarray,
    annual_returns: pd.Series,
    annual_covariance: pd.DataFrame,
    risk_free_rate: float,
) -> dict[str, float | str]:
    """Calculate metrics for one portfolio strategy."""
    ret = portfolio_return(weights, annual_returns)
    risk = portfolio_risk(weights, annual_covariance)
    sharpe = (ret - risk_free_rate) / risk if risk != 0 else -np.inf
    return {"Strategy": strategy, "Annual Return": ret, "Annual Risk": risk, "Sharpe Ratio": sharpe}


def calculate_portfolio_daily_returns(stock_returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Calculate one strategy's daily portfolio returns."""
    returns = stock_returns @ weights
    returns.name = "Portfolio Return"
    return returns.dropna()


def cumulative_return_series(portfolio_returns: pd.Series) -> pd.Series:
    """Calculate cumulative return over time."""
    return (1 + portfolio_returns).cumprod() - 1


def drawdown_series(portfolio_returns: pd.Series) -> pd.Series:
    """Calculate drawdown from previous wealth peaks."""
    wealth_index = (1 + portfolio_returns).cumprod()
    running_peak = wealth_index.cummax()
    return (wealth_index / running_peak) - 1


def annualized_return_from_daily(portfolio_returns: pd.Series) -> float:
    """Calculate compounded annual return from daily returns."""
    if portfolio_returns.empty:
        return np.nan
    total_return = (1 + portfolio_returns).prod() - 1
    years = len(portfolio_returns) / TRADING_DAYS
    if years <= 0:
        return np.nan
    return (1 + total_return) ** (1 / years) - 1


def calculate_sortino_ratio(portfolio_returns: pd.Series, risk_free_rate: float) -> float:
    """Calculate Sortino ratio using downside volatility."""
    annual_return = annualized_return_from_daily(portfolio_returns)
    daily_target = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    downside_returns = portfolio_returns[portfolio_returns < daily_target] - daily_target
    downside_deviation = downside_returns.std() * np.sqrt(TRADING_DAYS)
    if downside_deviation == 0 or np.isnan(downside_deviation):
        return np.nan
    return (annual_return - risk_free_rate) / downside_deviation


def calculate_var(portfolio_returns: pd.Series, confidence_level: float = VAR_CONFIDENCE_LEVEL) -> float:
    """Calculate historical VaR as a positive loss number."""
    tail_probability = 1 - confidence_level
    return -np.percentile(portfolio_returns, tail_probability * 100)


def calculate_cvar(portfolio_returns: pd.Series, confidence_level: float = VAR_CONFIDENCE_LEVEL) -> float:
    """Calculate historical CVaR as average loss beyond VaR."""
    tail_probability = 1 - confidence_level
    threshold = np.percentile(portfolio_returns, tail_probability * 100)
    tail_returns = portfolio_returns[portfolio_returns <= threshold]
    if tail_returns.empty:
        return np.nan
    return -tail_returns.mean()


def calculate_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series | None) -> float:
    """Calculate portfolio beta against NIFTY 50 when benchmark is available."""
    if benchmark_returns is None:
        return np.nan
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if aligned.empty or aligned.iloc[:, 1].var() == 0:
        return np.nan
    return aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / aligned.iloc[:, 1].var()


def calculate_tracking_error(portfolio_returns: pd.Series, benchmark_returns: pd.Series | None) -> float:
    """Calculate annualized tracking error against NIFTY 50."""
    if benchmark_returns is None:
        return np.nan
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if aligned.empty:
        return np.nan
    active_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return active_returns.std() * np.sqrt(TRADING_DAYS)


def calculate_information_ratio(portfolio_returns: pd.Series, benchmark_returns: pd.Series | None) -> float:
    """Calculate information ratio against NIFTY 50."""
    if benchmark_returns is None:
        return np.nan
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if aligned.empty:
        return np.nan
    active_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    annual_active_return = active_returns.mean() * TRADING_DAYS
    annual_tracking_error = active_returns.std() * np.sqrt(TRADING_DAYS)
    if annual_tracking_error == 0:
        return np.nan
    return annual_active_return / annual_tracking_error


def calculate_advanced_risk_metrics(
    strategy: str,
    portfolio_returns: pd.Series,
    risk_free_rate: float,
    benchmark_returns: pd.Series | None,
) -> dict[str, float | str]:
    """Calculate advanced risk metrics for one strategy."""
    cumulative_return = cumulative_return_series(portfolio_returns).iloc[-1]
    annual_return = annualized_return_from_daily(portfolio_returns)
    annual_volatility = portfolio_returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else np.nan
    max_drawdown = drawdown_series(portfolio_returns).min()
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.nan

    return {
        "Strategy": strategy,
        "Cumulative Return": cumulative_return,
        "Annual Return": annual_return,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": calculate_sortino_ratio(portfolio_returns, risk_free_rate),
        "Maximum Drawdown": max_drawdown,
        "Calmar Ratio": calmar,
        "VaR 95%": calculate_var(portfolio_returns),
        "CVaR 95%": calculate_cvar(portfolio_returns),
        "Beta vs NIFTY 50": calculate_beta(portfolio_returns, benchmark_returns),
        "Tracking Error": calculate_tracking_error(portfolio_returns, benchmark_returns),
        "Information Ratio": calculate_information_ratio(portfolio_returns, benchmark_returns),
    }


def optimize_training_window_max_sharpe(
    training_returns: pd.DataFrame,
    risk_free_rate: float,
) -> tuple[np.ndarray, bool, str]:
    """Optimize max-Sharpe weights using only one historical training window."""
    n_assets = training_returns.shape[1]
    train_annual_returns = training_returns.mean() * TRADING_DAYS
    train_annual_covariance = training_returns.cov() * TRADING_DAYS

    initial_weights = np.repeat(1 / n_assets, n_assets)
    bounds = tuple((0, 1) for _ in range(n_assets))
    constraints = ({"type": "eq", "fun": lambda weights: np.sum(weights) - 1},)

    try:
        result = minimize(
            negative_sharpe_ratio,
            initial_weights,
            args=(train_annual_returns, train_annual_covariance, risk_free_rate),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
        )
    except Exception as error:
        return initial_weights, False, str(error)

    if not result.success:
        return initial_weights, False, str(result.message)

    return result.x, True, ""


def run_walk_forward_backtest(
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series | None,
    training_window: int,
    rebalance_frequency: int,
    transaction_cost_rate: float,
    risk_free_rate: float,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, list[str], str]:
    """Run walk-forward rebalancing backtest without look-ahead bias."""
    if len(stock_returns) <= training_window + rebalance_frequency:
        return None, None, None, None, [], (
            "Not enough historical return rows for the selected training window and "
            "rebalance frequency. Use a longer date range or smaller training window."
        )

    n_assets = stock_returns.shape[1]
    tickers = list(stock_returns.columns)
    equal_weights = np.repeat(1 / n_assets, n_assets)

    optimized_returns = pd.Series(0.0, index=stock_returns.index, name="Optimized Portfolio")
    equal_returns = pd.Series(0.0, index=stock_returns.index, name="Equal Weight Portfolio")
    previous_optimized_weights = np.zeros(n_assets)
    previous_equal_weights = np.zeros(n_assets)

    turnover_records = []
    weight_records = []
    warnings = []

    rebalance_positions = list(range(training_window, len(stock_returns), rebalance_frequency))

    for rebalance_number, start_position in enumerate(rebalance_positions, start=1):
        end_position = min(start_position + rebalance_frequency, len(stock_returns))

        # Training data ends before the holding period begins, preventing look-ahead bias.
        training_returns = stock_returns.iloc[start_position - training_window:start_position]
        holding_returns = stock_returns.iloc[start_position:end_position]

        optimized_weights, success, message = optimize_training_window_max_sharpe(
            training_returns,
            risk_free_rate,
        )
        if not success:
            warnings.append(f"Rebalance {rebalance_number} used equal weights because optimization failed: {message}")
            optimized_weights = equal_weights.copy()

        optimized_turnover = float(np.sum(np.abs(optimized_weights - previous_optimized_weights)))
        equal_turnover = float(np.sum(np.abs(equal_weights - previous_equal_weights)))

        optimized_period_returns = holding_returns @ optimized_weights
        equal_period_returns = holding_returns @ equal_weights

        # Deduct transaction cost on the first day of each holding period.
        if not optimized_period_returns.empty:
            optimized_period_returns.iloc[0] -= transaction_cost_rate * optimized_turnover
            equal_period_returns.iloc[0] -= transaction_cost_rate * equal_turnover

        optimized_returns.loc[holding_returns.index] = optimized_period_returns
        equal_returns.loc[holding_returns.index] = equal_period_returns

        rebalance_date = holding_returns.index[0]
        turnover_records.append(
            {
                "Date": rebalance_date,
                "Optimized Turnover": optimized_turnover,
                "Equal Weight Turnover": equal_turnover,
                "Optimization Success": success,
            }
        )

        weight_row = {"Date": rebalance_date}
        for ticker, weight in zip(tickers, optimized_weights):
            weight_row[ticker] = weight
        weight_records.append(weight_row)

        previous_optimized_weights = optimized_weights.copy()
        previous_equal_weights = equal_weights.copy()

    backtest_returns = pd.DataFrame(
        {
            "Optimized Portfolio": optimized_returns,
            "Equal Weight Portfolio": equal_returns,
        }
    )

    if benchmark_returns is not None:
        aligned_benchmark = benchmark_returns.reindex(backtest_returns.index).dropna()
        backtest_returns = backtest_returns.loc[aligned_benchmark.index]
        backtest_returns["NIFTY 50 Benchmark"] = aligned_benchmark

    backtest_returns = backtest_returns.dropna()
    weights_table = pd.DataFrame(weight_records).set_index("Date")
    turnover_table = pd.DataFrame(turnover_records).set_index("Date")
    metrics_table = calculate_backtest_metrics(backtest_returns, turnover_table, risk_free_rate)

    return backtest_returns, weights_table, turnover_table, metrics_table, warnings, ""


def calculate_backtest_metrics(
    backtest_returns: pd.DataFrame,
    turnover_table: pd.DataFrame,
    risk_free_rate: float,
) -> pd.DataFrame:
    """Calculate performance metrics for walk-forward backtest results."""
    rows = []
    number_of_rebalances = len(turnover_table)

    for column in backtest_returns.columns:
        total_return = (1 + backtest_returns[column]).prod() - 1
        annual_return = annualized_return_from_daily(backtest_returns[column])
        annual_volatility = backtest_returns[column].std() * np.sqrt(TRADING_DAYS)
        sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else np.nan
        max_drawdown = drawdown_series(backtest_returns[column]).min()

        if column == "Optimized Portfolio":
            average_turnover = turnover_table["Optimized Turnover"].mean()
            rebalances = number_of_rebalances
        elif column == "Equal Weight Portfolio":
            average_turnover = turnover_table["Equal Weight Turnover"].mean()
            rebalances = number_of_rebalances
        else:
            average_turnover = np.nan
            rebalances = 0

        rows.append(
            {
                "Strategy": column,
                "Total Return": total_return,
                "Annual Return": annual_return,
                "Annual Volatility": annual_volatility,
                "Sharpe Ratio": sharpe,
                "Maximum Drawdown": max_drawdown,
                "Average Turnover": average_turnover,
                "Number of Rebalances": rebalances,
            }
        )

    return pd.DataFrame(rows)


def black_litterman_market_weights(
    tickers: list[str],
    custom_weights: pd.Series | None = None,
) -> tuple[pd.Series, str]:
    """Use normalized custom market weights or equal weights as fallback."""
    if custom_weights is not None:
        aligned = custom_weights.reindex(tickers).fillna(0).astype(float)
        if aligned.sum() > 0:
            return aligned / aligned.sum(), "Custom market weights used and normalized to 100%."

    equal = pd.Series(np.repeat(1 / len(tickers), len(tickers)), index=tickers)
    return equal, "Equal weights used as market-weight fallback."


def black_litterman_risk_aversion(
    daily_returns: pd.DataFrame,
    market_weights: pd.Series,
    risk_free_rate: float,
) -> float:
    """Estimate risk aversion from market-weighted portfolio returns."""
    market_daily_returns = daily_returns[market_weights.index] @ market_weights
    market_return = market_daily_returns.mean() * TRADING_DAYS
    market_variance = market_daily_returns.var() * TRADING_DAYS
    if market_variance <= 0:
        return 2.5
    value = (market_return - risk_free_rate) / market_variance
    if not np.isfinite(value) or value <= 0:
        return 2.5
    return float(value)


def black_litterman_implied_returns(
    annual_covariance: pd.DataFrame,
    market_weights: pd.Series,
    risk_aversion: float,
    risk_free_rate: float,
) -> pd.Series:
    """Calculate market-implied equilibrium returns."""
    implied_excess_returns = risk_aversion * (annual_covariance @ market_weights)
    implied_returns = risk_free_rate + implied_excess_returns
    implied_returns.name = "Implied Equilibrium Return"
    return implied_returns


def black_litterman_views_matrices(
    tickers: list[str],
    views: list[dict],
    annual_covariance: pd.DataFrame,
    tau: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, list[str]]:
    """Build P, Q, and Omega matrices from investor views."""
    ticker_to_index = {ticker: index for index, ticker in enumerate(tickers)}
    p_rows = []
    q_values = []
    omega_values = []
    descriptions = []

    for view in views:
        p_row = np.zeros(len(tickers))

        if view["type"] == "relative":
            if view["long"] not in ticker_to_index or view["short"] not in ticker_to_index:
                continue
            p_row[ticker_to_index[view["long"]]] = 1
            p_row[ticker_to_index[view["short"]]] = -1
        elif view["type"] == "absolute":
            if view["ticker"] not in ticker_to_index:
                continue
            p_row[ticker_to_index[view["ticker"]]] = 1
        else:
            continue

        confidence = min(max(float(view["confidence"]), 0.01), 0.99)
        view_variance = float(p_row @ (tau * annual_covariance.to_numpy()) @ p_row.T)
        omega_value = max(view_variance * ((1 - confidence) / confidence), 1e-8)

        p_rows.append(p_row)
        q_values.append(float(view["value"]))
        omega_values.append(omega_value)
        descriptions.append(view["description"])

    if not p_rows:
        return None, None, None, []

    return np.vstack(p_rows), np.array(q_values), np.diag(omega_values), descriptions


def black_litterman_posterior_returns(
    implied_returns: pd.Series,
    annual_covariance: pd.DataFrame,
    p_matrix: np.ndarray,
    q_vector: np.ndarray,
    omega_matrix: np.ndarray,
    tau: float,
) -> pd.Series:
    """Calculate Black-Litterman posterior expected returns."""
    sigma = annual_covariance.to_numpy()
    pi = implied_returns.to_numpy()
    tau_sigma = tau * sigma
    middle = np.linalg.inv(p_matrix @ tau_sigma @ p_matrix.T + omega_matrix)
    posterior = pi + tau_sigma @ p_matrix.T @ middle @ (q_vector - p_matrix @ pi)
    return pd.Series(posterior, index=implied_returns.index, name="Black-Litterman Return")


def black_litterman_allocation_comparison(
    tickers: list[str],
    historical_weights: np.ndarray,
    black_litterman_weights: np.ndarray,
) -> pd.DataFrame:
    """Create allocation comparison table."""
    table = pd.DataFrame(
        {
            "Ticker": tickers,
            "Historical Mean Weight": historical_weights,
            "Black-Litterman Weight": black_litterman_weights,
        }
    )
    table["Historical Mean Allocation %"] = table["Historical Mean Weight"] * 100
    table["Black-Litterman Allocation %"] = table["Black-Litterman Weight"] * 100
    return table.sort_values("Black-Litterman Weight", ascending=False)


def factor_zscore(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Normalize factor values and assign neutral score 0 to missing values."""
    clean = series.replace([np.inf, -np.inf], np.nan).astype(float)
    if not higher_is_better:
        clean = -clean
    mean = clean.mean(skipna=True)
    std = clean.std(skipna=True)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return ((clean - mean) / std).fillna(0)


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_factor_fundamentals(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Fetch PE, ROE, and market cap from yfinance where available."""
    rows = []

    for ticker in tickers:
        pe_ratio = np.nan
        return_on_equity = np.nan
        market_cap = np.nan

        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            pe_ratio = info.get("trailingPE", np.nan)
            return_on_equity = info.get("returnOnEquity", np.nan)
            market_cap = info.get("marketCap", np.nan)
        except Exception:
            pass

        rows.append(
            {
                "Ticker": ticker,
                "PE Ratio": pe_ratio,
                "Return on Equity": return_on_equity,
                "Market Cap": market_cap,
            }
        )

    return pd.DataFrame(rows).set_index("Ticker")


def build_factor_score_table(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    fundamentals: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build practical factor proxy scores from price and fundamental data."""
    tickers = list(returns.columns)
    factor_table = pd.DataFrame(index=tickers)

    if len(prices) > 252:
        factor_table["Momentum"] = (prices.iloc[-1] / prices.iloc[-252] - 1).reindex(tickers)
    elif len(prices) > 126:
        factor_table["Momentum"] = (prices.iloc[-1] / prices.iloc[-126] - 1).reindex(tickers)
    else:
        factor_table["Momentum"] = (prices.iloc[-1] / prices.iloc[0] - 1).reindex(tickers)

    factor_table["Annual Volatility"] = (returns.std() * np.sqrt(TRADING_DAYS)).reindex(tickers)

    if fundamentals is None:
        fundamentals = pd.DataFrame(index=tickers, columns=["PE Ratio", "Return on Equity", "Market Cap"], dtype=float)
    else:
        fundamentals = fundamentals.reindex(tickers)

    factor_table = factor_table.join(fundamentals)

    factor_table["Momentum Score"] = factor_zscore(factor_table["Momentum"], higher_is_better=True)
    factor_table["Low Volatility Score"] = factor_zscore(factor_table["Annual Volatility"], higher_is_better=False)
    factor_table["Value Score"] = factor_zscore(factor_table["PE Ratio"], higher_is_better=False)
    factor_table["Quality Score"] = factor_zscore(factor_table["Return on Equity"], higher_is_better=True)
    factor_table["Log Market Cap"] = np.log(factor_table["Market Cap"].replace(0, np.nan))
    factor_table["Size Score"] = factor_zscore(factor_table["Log Market Cap"], higher_is_better=False)

    score_columns = [
        "Momentum Score",
        "Low Volatility Score",
        "Value Score",
        "Quality Score",
        "Size Score",
    ]
    factor_table["Combined Factor Score"] = factor_table[score_columns].mean(axis=1)
    factor_table["Rank"] = factor_table["Combined Factor Score"].rank(ascending=False, method="min").astype(int)
    factor_table["Missing Fundamental Count"] = factor_table[["PE Ratio", "Return on Equity", "Market Cap"]].isna().sum(axis=1)

    return factor_table.sort_values("Combined Factor Score", ascending=False)


def factor_portfolio_weights(factor_scores: pd.DataFrame, portfolio_size: int) -> pd.Series:
    """Create equal-weight portfolio from top-ranked factor stocks."""
    selected = factor_scores.head(min(portfolio_size, len(factor_scores))).index
    weights = pd.Series(0.0, index=factor_scores.index)
    weights.loc[selected] = 1 / len(selected)
    return weights


def factor_strategy_metrics(
    strategy: str,
    weights: pd.Series,
    returns: pd.DataFrame,
    risk_free_rate: float,
) -> dict[str, float | str]:
    """Calculate comparison metrics for a factor strategy."""
    aligned_weights = weights.reindex(returns.columns).fillna(0)
    strategy_returns = returns @ aligned_weights
    total_return = (1 + strategy_returns).prod() - 1
    annual_return = strategy_returns.mean() * TRADING_DAYS
    annual_volatility = strategy_returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else np.nan
    return {
        "Strategy": strategy,
        "Total Return": total_return,
        "Annual Return": annual_return,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": sharpe,
    }


def calculate_rsi_indicator(price_series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Calculate RSI momentum indicator from adjusted close prices."""
    delta = price_series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.rolling(period).mean()
    average_loss = losses.rolling(period).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi.where(~((average_loss == 0) & (average_gain > 0)), 100)


def build_ml_stock_feature_frame(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Create ML features and next-21-day target for one stock."""
    close = prices[ticker].astype(float)
    daily_returns = close.pct_change()

    feature_frame = pd.DataFrame(index=prices.index)
    feature_frame["Ticker"] = ticker
    feature_frame["Return 5D"] = close.pct_change(5)
    feature_frame["Return 10D"] = close.pct_change(10)
    feature_frame["Return 20D"] = close.pct_change(20)
    feature_frame["MA 50 Ratio"] = close / close.rolling(50).mean() - 1
    feature_frame["MA 100 Ratio"] = close / close.rolling(100).mean() - 1
    feature_frame["Volatility 20D"] = daily_returns.rolling(20).std()
    feature_frame["RSI 14D"] = calculate_rsi_indicator(close)
    feature_frame["Target 21D Return"] = close.shift(-PREDICTION_HORIZON_DAYS) / close - 1

    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).dropna()
    feature_frame = feature_frame.reset_index().rename(columns={"index": "Date"})
    return feature_frame


def build_ml_feature_dataset(prices: pd.DataFrame) -> pd.DataFrame:
    """Build a long-format feature table across all selected stocks."""
    frames = [
        build_ml_stock_feature_frame(prices, ticker)
        for ticker in prices.columns
        if prices[ticker].notna().sum() > 120
    ]

    if not frames:
        return pd.DataFrame()

    dataset = pd.concat(frames, ignore_index=True)
    dataset["Date"] = pd.to_datetime(dataset["Date"])
    return dataset.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def build_latest_ml_feature_table(prices: pd.DataFrame) -> pd.DataFrame:
    """Create the newest usable feature row for each stock."""
    rows = []

    for ticker in prices.columns:
        close = prices[ticker].astype(float)
        daily_returns = close.pct_change()

        feature_frame = pd.DataFrame(index=prices.index)
        feature_frame["Ticker"] = ticker
        feature_frame["Return 5D"] = close.pct_change(5)
        feature_frame["Return 10D"] = close.pct_change(10)
        feature_frame["Return 20D"] = close.pct_change(20)
        feature_frame["MA 50 Ratio"] = close / close.rolling(50).mean() - 1
        feature_frame["MA 100 Ratio"] = close / close.rolling(100).mean() - 1
        feature_frame["Volatility 20D"] = daily_returns.rolling(20).std()
        feature_frame["RSI 14D"] = calculate_rsi_indicator(close)
        feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).dropna()

        if feature_frame.empty:
            continue

        latest_row = feature_frame.iloc[-1].copy()
        latest_row["Date"] = feature_frame.index[-1]
        rows.append(latest_row)

    if not rows:
        return pd.DataFrame()

    latest_features = pd.DataFrame(rows)
    return latest_features[["Date", "Ticker", *ML_FEATURE_COLUMNS]].reset_index(drop=True)


def make_ml_model_definitions() -> dict[str, object]:
    """Create simple ML model definitions."""
    if SKLEARN_IMPORT_ERROR is not None:
        raise ImportError(
            "scikit-learn is required for this section. Install it with: pip install -r requirements.txt"
        ) from SKLEARN_IMPORT_ERROR

    return {
        "Linear Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=120,
            max_depth=5,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }


def calculate_directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate how often predicted and actual return signs match."""
    same_direction = ((actual >= 0) & (predicted >= 0)) | ((actual < 0) & (predicted < 0))
    return float(np.mean(same_direction))


def evaluate_ml_models_time_series(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Evaluate ML models with expanding time-series splits over dates."""
    unique_dates = pd.Series(dataset["Date"].drop_duplicates()).sort_values().reset_index(drop=True)
    usable_splits = min(ML_TIME_SPLITS, len(unique_dates) - 1)

    if usable_splits < 2:
        raise ValueError("Not enough unique dates for time-series validation.")

    splitter = TimeSeriesSplit(n_splits=usable_splits)
    models = make_ml_model_definitions()
    performance_rows = []
    prediction_frames = []

    for fold_number, (train_date_index, test_date_index) in enumerate(splitter.split(unique_dates), start=1):
        train_dates = unique_dates.iloc[train_date_index]
        test_dates = unique_dates.iloc[test_date_index]

        train_data = dataset[dataset["Date"].isin(train_dates)]
        test_data = dataset[dataset["Date"].isin(test_dates)]

        x_train = train_data[ML_FEATURE_COLUMNS]
        y_train = train_data["Target 21D Return"]
        x_test = test_data[ML_FEATURE_COLUMNS]
        y_test = test_data["Target 21D Return"]

        for model_name, model in models.items():
            model.fit(x_train, y_train)
            predicted = model.predict(x_test)

            performance_rows.append(
                {
                    "Model": model_name,
                    "Fold": fold_number,
                    "Train Start": train_dates.iloc[0],
                    "Train End": train_dates.iloc[-1],
                    "Test Start": test_dates.iloc[0],
                    "Test End": test_dates.iloc[-1],
                    "Train Rows": len(train_data),
                    "Test Rows": len(test_data),
                    "MAE": mean_absolute_error(y_test, predicted),
                    "RMSE": float(np.sqrt(mean_squared_error(y_test, predicted))),
                    "Directional Accuracy": calculate_directional_accuracy(y_test.to_numpy(), predicted),
                }
            )

            prediction_frames.append(
                pd.DataFrame(
                    {
                        "Date": test_data["Date"].to_numpy(),
                        "Ticker": test_data["Ticker"].to_numpy(),
                        "Model": model_name,
                        "Actual 21D Return": y_test.to_numpy(),
                        "Predicted 21D Return": predicted,
                        "Fold": fold_number,
                    }
                )
            )

    performance = pd.DataFrame(performance_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    average_performance = (
        performance.groupby("Model", as_index=False)[["MAE", "RMSE", "Directional Accuracy"]]
        .mean()
        .sort_values(["MAE", "RMSE"], ascending=True)
    )
    best_model_name = str(average_performance.iloc[0]["Model"])

    return performance, predictions, best_model_name


def train_ml_model_and_predict_latest(
    dataset: pd.DataFrame,
    prices: pd.DataFrame,
    best_model_name: str,
) -> pd.DataFrame:
    """Train the best validation model on all known rows and predict latest returns."""
    models = make_ml_model_definitions()
    model = models[best_model_name]
    model.fit(dataset[ML_FEATURE_COLUMNS], dataset["Target 21D Return"])

    latest_features = build_latest_ml_feature_table(prices)
    if latest_features.empty:
        return pd.DataFrame()

    predicted_21d = model.predict(latest_features[ML_FEATURE_COLUMNS])
    latest_predictions = latest_features[["Date", "Ticker"]].copy()
    latest_predictions["Predicted 21D Return"] = predicted_21d
    latest_predictions["Annualized Predicted Return"] = predicted_21d * (TRADING_DAYS / PREDICTION_HORIZON_DAYS)

    return latest_predictions.sort_values("Predicted 21D Return", ascending=False).reset_index(drop=True)


def build_ml_portfolio_tables(
    stock_returns: pd.DataFrame,
    latest_predictions: pd.DataFrame,
    risk_free_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build historical, ML-predicted, and equal weight portfolio tables."""
    historical_annual_returns = stock_returns.mean() * TRADING_DAYS
    annual_covariance = stock_returns.cov() * TRADING_DAYS
    ml_annual_returns = latest_predictions.set_index("Ticker")["Annualized Predicted Return"]

    tickers = [ticker for ticker in stock_returns.columns if ticker in ml_annual_returns.index]
    if len(tickers) < 2:
        raise ValueError("ML prediction produced fewer than two valid stocks.")

    historical_annual_returns = historical_annual_returns.reindex(tickers)
    ml_annual_returns = ml_annual_returns.reindex(tickers)
    annual_covariance = annual_covariance.loc[tickers, tickers]

    historical_weights, historical_error = optimize_max_sharpe(
        historical_annual_returns,
        annual_covariance,
        risk_free_rate,
    )
    if historical_weights is None:
        raise RuntimeError(historical_error)

    ml_weights, ml_error = optimize_max_sharpe(
        ml_annual_returns,
        annual_covariance,
        risk_free_rate,
    )
    if ml_weights is None:
        raise RuntimeError(ml_error)

    equal_weights = np.repeat(1 / len(tickers), len(tickers))
    strategies = {
        "Historical Mean Optimization": historical_weights,
        "ML Prediction Optimization": ml_weights,
        "Equal Weight Portfolio": equal_weights,
    }

    comparison_rows = []
    for strategy, weights in strategies.items():
        annual_risk = portfolio_risk(weights, annual_covariance)
        historical_return = portfolio_return(weights, historical_annual_returns)
        ml_return = portfolio_return(weights, ml_annual_returns)

        comparison_rows.append(
            {
                "Strategy": strategy,
                "Historical Expected Annual Return": historical_return,
                "ML Expected Annual Return": ml_return,
                "Annual Risk": annual_risk,
                "Sharpe Using Historical Return": (
                    (historical_return - risk_free_rate) / annual_risk if annual_risk != 0 else np.nan
                ),
                "Sharpe Using ML Return": (
                    (ml_return - risk_free_rate) / annual_risk if annual_risk != 0 else np.nan
                ),
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    weights_table = pd.DataFrame(
        {
            "Ticker": tickers,
            "Historical Mean Optimization": historical_weights,
            "ML Prediction Optimization": ml_weights,
            "Equal Weight Portfolio": equal_weights,
            "Historical Annual Return": historical_annual_returns.values,
            "ML Annualized Predicted Return": ml_annual_returns.values,
        }
    ).sort_values("ML Prediction Optimization", ascending=False)

    return comparison, weights_table


@st.cache_data(show_spinner=False, ttl=3600)
def run_ml_return_prediction(
    prices: pd.DataFrame,
    stock_returns: pd.DataFrame,
    risk_free_rate: float,
) -> dict[str, object]:
    """Run ML feature engineering, validation, prediction, and optimization."""
    dataset = build_ml_feature_dataset(prices)
    if len(dataset) < ML_MIN_DATASET_ROWS:
        raise ValueError("Not enough ML rows. Use a longer date range or more selected stocks.")

    performance, predictions, best_model_name = evaluate_ml_models_time_series(dataset)
    latest_predictions = train_ml_model_and_predict_latest(dataset, prices, best_model_name)

    if latest_predictions.empty:
        raise ValueError("No latest feature rows were available for ML prediction.")

    portfolio_comparison, weights_table = build_ml_portfolio_tables(
        stock_returns,
        latest_predictions,
        risk_free_rate,
    )

    performance_summary = (
        performance.groupby("Model", as_index=False)[["MAE", "RMSE", "Directional Accuracy"]]
        .mean()
        .sort_values(["MAE", "RMSE"], ascending=True)
    )

    return {
        "dataset": dataset,
        "performance": performance,
        "performance_summary": performance_summary,
        "predictions": predictions,
        "best_model_name": best_model_name,
        "latest_predictions": latest_predictions,
        "portfolio_comparison": portfolio_comparison,
        "weights_table": weights_table,
    }


def build_ml_report_section(best_model_name: str, performance_summary: pd.DataFrame) -> str:
    """Create a beginner-friendly report section for the dashboard."""
    best_row = performance_summary[performance_summary["Model"] == best_model_name].iloc[0]
    return f"""
    <div class="info-panel">
    <h3>Machine Learning Based Return Prediction</h3>
    This section uses adjusted close price data to predict each selected stock's next
    21 trading day return, then compares historical mean-return optimization,
    ML-predicted-return optimization, and equal weighting. This is educational
    only and does not guarantee profit.<br><br>
    <b>Feature guide:</b><br>
    5-day, 10-day, and 20-day returns measure recent momentum over short windows.
    The 50-day and 100-day moving average ratios show whether price is above or
    below its trend. 20-day volatility measures how unstable recent daily returns
    have been. RSI compares recent gains with recent losses and describes recent
    momentum strength.<br><br>
    <b>Why time-series split matters:</b><br>
    Random train-test split is wrong for market time series because it can train
    on future dates and test on earlier dates, creating look-ahead bias. The
    validation here trains on earlier dates and tests on later dates.<br><br>
    <b>Validation summary:</b><br>
    Best model by MAE: <span class="green-text">{best_model_name}</span><br>
    MAE: {best_row["MAE"]:.4f} | RMSE: {best_row["RMSE"]:.4f} |
    Directional accuracy: {best_row["Directional Accuracy"] * 100:.2f}%<br><br>
    <b>Risk notes:</b><br>
    Financial ML can overfit because historical relationships may be temporary.
    Predictions are noisy because prices react to new information, macro events,
    earnings, sentiment, liquidity, and random market behavior. Treat the ML
    portfolio as an experiment, not a promise.
    </div>
    """


def allocation_table(tickers: list[str], weights: np.ndarray) -> pd.DataFrame:
    """Create a readable portfolio allocation table."""
    table = pd.DataFrame({"Ticker": tickers, "Weight": weights})
    table["Display Weight"] = table["Weight"].where(table["Weight"] >= 0.001, 0)
    table["Allocation %"] = table["Display Weight"] * 100
    return table.sort_values("Display Weight", ascending=False)


# -----------------------------
# 3. Chart functions
# -----------------------------

def style_axis(ax: plt.Axes) -> None:
    """Apply pure black/pure white chart theme."""
    ax.set_facecolor(PURE_BLACK)
    ax.figure.set_facecolor(PURE_BLACK)
    ax.title.set_color(PURE_WHITE)
    ax.xaxis.label.set_color(PURE_WHITE)
    ax.yaxis.label.set_color(PURE_WHITE)
    ax.tick_params(axis="x", colors=PURE_WHITE)
    ax.tick_params(axis="y", colors=PURE_WHITE)
    ax.grid(color=PURE_WHITE, alpha=0.12)
    for spine in ax.spines.values():
        spine.set_color(PURE_WHITE)


def price_trend_chart(prices: pd.DataFrame, tickers: list[str]) -> plt.Figure:
    """Plot selected price trends."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for ticker in tickers:
        if ticker in prices.columns:
            ax.plot(prices.index, prices[ticker], linewidth=1.8, label=ticker)
    ax.set_title("Adjusted Close Price Trends")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted Close Price")
    style_axis(ax)
    legend = ax.legend(facecolor=PURE_BLACK, edgecolor=PURE_WHITE, fontsize=8)
    for text in legend.get_texts():
        text.set_color(PURE_WHITE)
    fig.tight_layout()
    return fig


def returns_bar_chart(summary: pd.DataFrame) -> plt.Figure:
    """Plot annual expected returns for all selected stocks."""
    plot_data = summary.sort_values("Annual Expected Return", ascending=True)
    colors = [PURE_GREEN if value >= 0 else PURE_RED for value in plot_data["Annual Expected Return"]]
    fig, ax = plt.subplots(figsize=(11, max(6, len(plot_data) * 0.22)))
    ax.barh(plot_data["Ticker"], plot_data["Annual Expected Return"] * 100, color=colors)
    ax.set_title("Annual Expected Return by Stock")
    ax.set_xlabel("Annual Expected Return (%)")
    ax.set_ylabel("Ticker")
    style_axis(ax)
    fig.tight_layout()
    return fig


def correlation_heatmap(correlation: pd.DataFrame) -> plt.Figure:
    """Plot a dark correlation heatmap."""
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor(PURE_BLACK)
    sns.heatmap(
        correlation,
        ax=ax,
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.2,
        linecolor="#222222",
        cbar_kws={"label": "Correlation"},
    )
    ax.set_title("Correlation Matrix of Daily Returns")
    style_axis(ax)
    colorbar = ax.collections[0].colorbar
    colorbar.ax.yaxis.label.set_color(PURE_WHITE)
    colorbar.ax.tick_params(colors=PURE_WHITE)
    fig.tight_layout()
    return fig


def allocation_chart(title: str, table: pd.DataFrame) -> plt.Figure:
    """Plot portfolio allocation."""
    plot_data = table[table["Display Weight"] > 0].sort_values("Allocation %", ascending=True)
    fig, ax = plt.subplots(figsize=(11, max(5, len(plot_data) * 0.35)))
    ax.barh(plot_data["Ticker"], plot_data["Allocation %"], color=PURE_GREEN, edgecolor=PURE_WHITE)
    ax.set_title(title)
    ax.set_xlabel("Allocation (%)")
    ax.set_ylabel("Ticker")
    style_axis(ax)
    fig.tight_layout()
    return fig


def monte_carlo_chart(
    results: pd.DataFrame,
    max_sharpe_row: pd.Series,
    min_vol_row: pd.Series,
) -> plt.Figure:
    """Plot Monte Carlo portfolios colored by Sharpe ratio."""
    fig, ax = plt.subplots(figsize=(12, 7))
    scatter = ax.scatter(
        results["Annual Risk"] * 100,
        results["Annual Return"] * 100,
        c=results["Sharpe Ratio"],
        cmap="viridis",
        s=18,
        alpha=0.65,
        edgecolors="none",
    )
    ax.scatter(
        max_sharpe_row["Annual Risk"] * 100,
        max_sharpe_row["Annual Return"] * 100,
        marker="*",
        s=460,
        color=PURE_GREEN,
        edgecolors=PURE_WHITE,
        linewidths=1.2,
        label="Max Sharpe",
    )
    ax.scatter(
        min_vol_row["Annual Risk"] * 100,
        min_vol_row["Annual Return"] * 100,
        marker="D",
        s=180,
        color=PURE_WHITE,
        edgecolors=PURE_GREEN,
        linewidths=1.2,
        label="Min Volatility",
    )
    ax.set_title("Efficient Frontier Style Monte Carlo Simulation")
    ax.set_xlabel("Annual Risk / Volatility (%)")
    ax.set_ylabel("Annual Expected Return (%)")
    style_axis(ax)
    legend = ax.legend(facecolor=PURE_BLACK, edgecolor=PURE_WHITE)
    for text in legend.get_texts():
        text.set_color(PURE_WHITE)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Sharpe Ratio", color=PURE_WHITE)
    colorbar.ax.tick_params(colors=PURE_WHITE)
    colorbar.outline.set_edgecolor(PURE_WHITE)
    fig.tight_layout()
    return fig


def comparison_chart(comparison: pd.DataFrame, metric: str, title: str, percent: bool) -> plt.Figure:
    """Plot strategy comparison bars."""
    fig, ax = plt.subplots(figsize=(10, 5))
    values = comparison[metric] * 100 if percent else comparison[metric]
    colors = [PURE_GREEN if value >= 0 else PURE_RED for value in comparison[metric]]
    bars = ax.bar(comparison["Strategy"], values, color=colors, edgecolor=PURE_WHITE)
    ax.set_title(title)
    ax.set_xlabel("Strategy")
    ax.set_ylabel(f"{metric} (%)" if percent else metric)
    style_axis(ax)
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    for bar, value in zip(bars, values):
        label = f"{value:.2f}%" if percent else f"{value:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom", color=PURE_WHITE)
    fig.tight_layout()
    return fig


def cumulative_return_chart(portfolio_returns: pd.DataFrame, benchmark_returns: pd.Series | None) -> plt.Figure:
    """Plot cumulative return comparison for all strategies."""
    cumulative_returns = portfolio_returns.apply(cumulative_return_series)

    fig, ax = plt.subplots(figsize=(12, 6))
    for column in cumulative_returns.columns:
        ax.plot(cumulative_returns.index, cumulative_returns[column] * 100, linewidth=2, label=column)

    if benchmark_returns is not None:
        aligned_benchmark = benchmark_returns.reindex(cumulative_returns.index).dropna()
        benchmark_cumulative = cumulative_return_series(aligned_benchmark)
        ax.plot(
            benchmark_cumulative.index,
            benchmark_cumulative * 100,
            color=PURE_WHITE,
            linestyle="--",
            linewidth=2,
            label="NIFTY 50",
        )

    ax.set_title("Cumulative Return Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    style_axis(ax)
    legend = ax.legend(facecolor=PURE_BLACK, edgecolor=PURE_WHITE, fontsize=8)
    for text in legend.get_texts():
        text.set_color(PURE_WHITE)
    fig.tight_layout()
    return fig


def drawdown_chart(portfolio_returns: pd.DataFrame) -> plt.Figure:
    """Plot drawdown comparison for all strategies."""
    drawdowns = portfolio_returns.apply(drawdown_series)

    fig, ax = plt.subplots(figsize=(12, 6))
    for column in drawdowns.columns:
        ax.plot(drawdowns.index, drawdowns[column] * 100, linewidth=2, label=column)

    ax.axhline(0, color=PURE_WHITE, linewidth=0.8)
    ax.set_title("Drawdown Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    style_axis(ax)
    legend = ax.legend(facecolor=PURE_BLACK, edgecolor=PURE_WHITE, fontsize=8)
    for text in legend.get_texts():
        text.set_color(PURE_WHITE)
    fig.tight_layout()
    return fig


def backtest_equity_chart(backtest_returns: pd.DataFrame) -> plt.Figure:
    """Plot walk-forward equity curve."""
    equity_curve = (1 + backtest_returns).cumprod()

    fig, ax = plt.subplots(figsize=(12, 6))
    for column in equity_curve.columns:
        ax.plot(equity_curve.index, equity_curve[column], linewidth=2, label=column)

    ax.set_title("Walk-Forward Backtest Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of 1 Rupee")
    style_axis(ax)
    legend = ax.legend(facecolor=PURE_BLACK, edgecolor=PURE_WHITE, fontsize=8)
    for text in legend.get_texts():
        text.set_color(PURE_WHITE)
    fig.tight_layout()
    return fig


def backtest_drawdown_chart(backtest_returns: pd.DataFrame) -> plt.Figure:
    """Plot walk-forward drawdown curve."""
    drawdowns = backtest_returns.apply(drawdown_series)

    fig, ax = plt.subplots(figsize=(12, 6))
    for column in drawdowns.columns:
        ax.plot(drawdowns.index, drawdowns[column] * 100, linewidth=2, label=column)

    ax.axhline(0, color=PURE_WHITE, linewidth=0.8)
    ax.set_title("Walk-Forward Backtest Drawdown Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    style_axis(ax)
    legend = ax.legend(facecolor=PURE_BLACK, edgecolor=PURE_WHITE, fontsize=8)
    for text in legend.get_texts():
        text.set_color(PURE_WHITE)
    fig.tight_layout()
    return fig


def backtest_weights_chart(weights_table: pd.DataFrame, top_n: int = 12) -> plt.Figure:
    """Plot optimized weights over time for the largest average allocations."""
    top_columns = weights_table.mean().sort_values(ascending=False).head(top_n).index
    plot_data = weights_table[top_columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(plot_data.index, plot_data.T.values, labels=plot_data.columns)
    ax.set_title(f"Optimized Portfolio Weights Over Time - Top {top_n}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    style_axis(ax)
    legend = ax.legend(facecolor=PURE_BLACK, edgecolor=PURE_WHITE, fontsize=7, loc="upper left")
    for text in legend.get_texts():
        text.set_color(PURE_WHITE)
    fig.tight_layout()
    return fig


def backtest_turnover_chart(turnover_table: pd.DataFrame) -> plt.Figure:
    """Plot turnover at each rebalance date."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(turnover_table.index, turnover_table["Optimized Turnover"] * 100, color=PURE_GREEN)
    ax.set_title("Optimized Portfolio Turnover at Rebalance Dates")
    ax.set_xlabel("Rebalance Date")
    ax.set_ylabel("Turnover (%)")
    style_axis(ax)
    fig.tight_layout()
    return fig


def black_litterman_allocation_chart(allocation: pd.DataFrame, top_n: int = 20) -> plt.Figure:
    """Plot historical mean vs Black-Litterman allocation."""
    plot_data = allocation.head(top_n).sort_values("Black-Litterman Allocation %", ascending=True)
    y_positions = np.arange(len(plot_data))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(
        y_positions - width / 2,
        plot_data["Historical Mean Allocation %"],
        height=width,
        color=PURE_WHITE,
        label="Historical Mean",
    )
    ax.barh(
        y_positions + width / 2,
        plot_data["Black-Litterman Allocation %"],
        height=width,
        color=PURE_GREEN,
        label="Black-Litterman",
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_data["Ticker"])
    ax.set_title("Historical Mean vs Black-Litterman Allocation")
    ax.set_xlabel("Allocation (%)")
    ax.set_ylabel("Ticker")
    style_axis(ax)
    legend = ax.legend(facecolor=PURE_BLACK, edgecolor=PURE_WHITE)
    for text in legend.get_texts():
        text.set_color(PURE_WHITE)
    fig.tight_layout()
    return fig


def factor_score_chart(factor_scores: pd.DataFrame, top_n: int = 20) -> plt.Figure:
    """Plot factor score ranking."""
    plot_data = factor_scores.head(top_n).sort_values("Combined Factor Score", ascending=True)
    colors = [PURE_GREEN if value >= 0 else PURE_RED for value in plot_data["Combined Factor Score"]]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(plot_data.index, plot_data["Combined Factor Score"], color=colors)
    ax.set_title("Factor Score Ranking")
    ax.set_xlabel("Combined Factor Score")
    ax.set_ylabel("Ticker")
    style_axis(ax)
    fig.tight_layout()
    return fig


def factor_performance_chart(portfolio_returns: pd.DataFrame) -> plt.Figure:
    """Plot cumulative performance for factor strategy comparison."""
    cumulative = (1 + portfolio_returns).cumprod() - 1

    fig, ax = plt.subplots(figsize=(12, 6))
    for column in cumulative.columns:
        ax.plot(cumulative.index, cumulative[column] * 100, linewidth=2, label=column)

    ax.set_title("Factor Portfolio Performance Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    style_axis(ax)
    legend = ax.legend(facecolor=PURE_BLACK, edgecolor=PURE_WHITE, fontsize=8)
    for text in legend.get_texts():
        text.set_color(PURE_WHITE)
    fig.tight_layout()
    return fig


def ml_predicted_vs_actual_chart(predictions: pd.DataFrame, best_model_name: str) -> plt.Figure:
    """Plot out-of-fold predicted returns against actual returns."""
    plot_data = predictions[predictions["Model"] == best_model_name].copy()
    colors = [PURE_GREEN if value >= 0 else PURE_RED for value in plot_data["Actual 21D Return"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        plot_data["Actual 21D Return"] * 100,
        plot_data["Predicted 21D Return"] * 100,
        c=colors,
        alpha=0.65,
        edgecolors=PURE_WHITE,
        linewidths=0.25,
    )

    min_value = min(plot_data["Actual 21D Return"].min(), plot_data["Predicted 21D Return"].min()) * 100
    max_value = max(plot_data["Actual 21D Return"].max(), plot_data["Predicted 21D Return"].max()) * 100
    ax.plot([min_value, max_value], [min_value, max_value], color=PURE_WHITE, linestyle="--", linewidth=1)

    ax.set_title(f"Predicted vs Actual 21-Day Returns - {best_model_name}")
    ax.set_xlabel("Actual 21-Day Return (%)")
    ax.set_ylabel("Predicted 21-Day Return (%)")
    style_axis(ax)
    fig.tight_layout()
    return fig


def ml_allocation_comparison_chart(weights_table: pd.DataFrame, top_n: int = 20) -> plt.Figure:
    """Plot historical, ML, and equal weight allocations."""
    weight_columns = [
        "Historical Mean Optimization",
        "ML Prediction Optimization",
        "Equal Weight Portfolio",
    ]
    plot_data = weights_table.copy()
    plot_data["Max Weight"] = plot_data[weight_columns].max(axis=1)
    plot_data = plot_data.sort_values("Max Weight", ascending=False).head(top_n)
    plot_data = plot_data.sort_values("Max Weight", ascending=True)

    y_positions = np.arange(len(plot_data))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, max(7, len(plot_data) * 0.35)))
    ax.barh(
        y_positions - width,
        plot_data["Historical Mean Optimization"] * 100,
        height=width,
        color=PURE_WHITE,
        label="Historical Mean",
    )
    ax.barh(
        y_positions,
        plot_data["ML Prediction Optimization"] * 100,
        height=width,
        color=PURE_GREEN,
        label="ML Prediction",
    )
    ax.barh(
        y_positions + width,
        plot_data["Equal Weight Portfolio"] * 100,
        height=width,
        color="#808080",
        label="Equal Weight",
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_data["Ticker"])
    ax.set_title("Portfolio Allocation Comparison")
    ax.set_xlabel("Allocation (%)")
    ax.set_ylabel("Ticker")
    style_axis(ax)
    legend = ax.legend(facecolor=PURE_BLACK, edgecolor=PURE_WHITE)
    for text in legend.get_texts():
        text.set_color(PURE_WHITE)
    fig.tight_layout()
    return fig


# -----------------------------
# 4. UI helpers
# -----------------------------

def fmt_pct(value: float) -> str:
    """Format decimal number as percentage."""
    return f"{value * 100:.2f}%"


def metric_card(label: str, value: str, help_text: str = "", positive: bool | None = None) -> None:
    """Render a small dashboard card."""
    value_class = ""
    if positive is True:
        value_class = " green-text"
    elif positive is False:
        value_class = " red-text"

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value{value_class}">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def dataframe_download(dataframe: pd.DataFrame, label: str, file_name: str) -> None:
    """Create a CSV download button for a DataFrame."""
    csv_buffer = StringIO()
    dataframe.to_csv(csv_buffer, index=True)
    st.download_button(
        label=label,
        data=csv_buffer.getvalue(),
        file_name=file_name,
        mime="text/csv",
    )


def style_return_risk_table(dataframe: pd.DataFrame):
    """Style numeric finance tables with green/red returns."""
    def color_return(value):
        if isinstance(value, (int, float, np.floating)):
            return f"color: {PURE_GREEN}; font-weight: 700;" if value >= 0 else f"color: {PURE_RED}; font-weight: 700;"
        return f"color: {PURE_WHITE};"

    return dataframe.style.map(color_return, subset=[column for column in dataframe.columns if "Return" in column or "Sharpe" in column])


# -----------------------------
# 5. Sidebar controls
# -----------------------------

inject_css()

st.sidebar.title("Controls")

today = date.today()
default_start = today - timedelta(days=365 * 5)

start_date = st.sidebar.date_input("Start date", value=default_start)
end_date = st.sidebar.date_input("End date", value=today)

selected_tickers = st.sidebar.multiselect(
    "Select NSE stocks",
    options=NSE_TICKERS,
    default=NSE_TICKERS[:12],
)

risk_free_rate = st.sidebar.slider(
    "Risk-free rate",
    min_value=0.00,
    max_value=0.12,
    value=DEFAULT_RISK_FREE_RATE,
    step=0.005,
    format="%.3f",
)

n_random_portfolios = st.sidebar.slider(
    "Random portfolios",
    min_value=1000,
    max_value=50000,
    value=10000,
    step=1000,
)

training_window = st.sidebar.slider(
    "Backtest training window",
    min_value=126,
    max_value=504,
    value=DEFAULT_TRAINING_WINDOW,
    step=21,
)

rebalance_frequency = st.sidebar.slider(
    "Rebalance frequency",
    min_value=10,
    max_value=63,
    value=DEFAULT_REBALANCE_FREQUENCY,
    step=1,
)

transaction_cost = st.sidebar.slider(
    "Transaction cost per turnover",
    min_value=0.000,
    max_value=0.005,
    value=DEFAULT_TRANSACTION_COST,
    step=0.0005,
    format="%.4f",
)

run_analysis = st.sidebar.button("Run Full Analysis", use_container_width=True)


# -----------------------------
# 6. App title and analysis run
# -----------------------------

st.title("Advanced Portfolio Optimization Dashboard")
st.caption("Indian NSE stocks | Modern Portfolio Theory | Monte Carlo | SciPy optimization")

if run_analysis:
    if not selected_tickers:
        st.error("Please select at least one stock before running the dashboard.")
        st.stop()

    if start_date >= end_date:
        st.error("Start date must be earlier than end date.")
        st.stop()

    st.session_state.pop("ml_analysis", None)

    with st.spinner("Downloading NSE stock data and running portfolio analysis..."):
        try:
            raw_prices = download_price_data(tuple(selected_tickers), start_date, end_date)
        except Exception as error:
            st.error(f"Yahoo Finance download failed: {error}")
            st.stop()

        prices, cleaning_warnings = clean_prices(raw_prices)

        if prices.empty:
            st.error("No usable stock data remained after cleaning. Try different tickers or dates.")
            st.stop()

        returns = calculate_returns(prices)

        if returns.empty:
            st.error("Returns data is empty. Try a longer date range.")
            st.stop()

        if len(returns.columns) < 2:
            st.error("Please use at least two valid stocks for portfolio optimization.")
            st.stop()

        annual_returns, annual_covariance = calculate_annual_inputs(returns)
        tickers = list(annual_returns.index)
        n_assets = len(tickers)

        one_random_weights = random_weights(n_assets)

        max_sharpe_weights, max_sharpe_error = optimize_max_sharpe(
            annual_returns,
            annual_covariance,
            risk_free_rate,
        )
        if max_sharpe_weights is None:
            st.error(max_sharpe_error)
            st.stop()

        min_vol_weights, min_vol_error = optimize_min_volatility(annual_covariance)
        if min_vol_weights is None:
            st.error(min_vol_error)
            st.stop()

        monte_carlo_results, monte_carlo_weights = run_monte_carlo(
            annual_returns,
            annual_covariance,
            risk_free_rate,
            n_random_portfolios,
        )

        mc_max_idx = int(monte_carlo_results["Sharpe Ratio"].idxmax())
        mc_min_idx = int(monte_carlo_results["Annual Risk"].idxmin())

        equal_weights = np.repeat(1 / n_assets, n_assets)

        strategy_comparison = pd.DataFrame(
            [
                calculate_strategy_row("Equal Weight", equal_weights, annual_returns, annual_covariance, risk_free_rate),
                calculate_strategy_row("Random", one_random_weights, annual_returns, annual_covariance, risk_free_rate),
                calculate_strategy_row("Max Sharpe", max_sharpe_weights, annual_returns, annual_covariance, risk_free_rate),
                calculate_strategy_row("Min Volatility", min_vol_weights, annual_returns, annual_covariance, risk_free_rate),
            ]
        )

        portfolio_returns = pd.DataFrame(
            {
                "Equal Weight": calculate_portfolio_daily_returns(returns, equal_weights),
                "Random": calculate_portfolio_daily_returns(returns, one_random_weights),
                "Max Sharpe": calculate_portfolio_daily_returns(returns, max_sharpe_weights),
                "Min Volatility": calculate_portfolio_daily_returns(returns, min_vol_weights),
            }
        ).dropna()

        benchmark_returns, benchmark_warning = download_benchmark_returns(start_date, end_date)

        advanced_risk_metrics = pd.DataFrame(
            [
                calculate_advanced_risk_metrics(
                    strategy,
                    portfolio_returns[strategy],
                    risk_free_rate,
                    benchmark_returns,
                )
                for strategy in portfolio_returns.columns
            ]
        )

        (
            backtest_returns,
            backtest_weights,
            backtest_turnover,
            backtest_metrics,
            backtest_warnings,
            backtest_error,
        ) = run_walk_forward_backtest(
            returns,
            benchmark_returns,
            training_window,
            rebalance_frequency,
            transaction_cost,
            risk_free_rate,
        )

        summary_table = pd.DataFrame(
            {
                "Ticker": annual_returns.index,
                "Annual Expected Return": annual_returns.values,
                "Annual Volatility": np.sqrt(np.diag(annual_covariance.to_numpy())),
            }
        )
        median_return = summary_table["Annual Expected Return"].median()
        median_risk = summary_table["Annual Volatility"].median()
        summary_table["Risk-Return Category"] = np.select(
            [
                (summary_table["Annual Expected Return"] >= median_return) & (summary_table["Annual Volatility"] < median_risk),
                (summary_table["Annual Expected Return"] >= median_return) & (summary_table["Annual Volatility"] >= median_risk),
                (summary_table["Annual Expected Return"] < median_return) & (summary_table["Annual Volatility"] < median_risk),
            ],
            [
                "High Return / Lower Risk",
                "High Return / Higher Risk",
                "Lower Return / Lower Risk",
            ],
            default="Lower Return / Higher Risk",
        )

        st.session_state.analysis = {
            "prices": prices,
            "returns": returns,
            "annual_returns": annual_returns,
            "annual_covariance": annual_covariance,
            "correlation": returns.corr(),
            "summary_table": summary_table.sort_values("Annual Expected Return", ascending=False),
            "random_weights": one_random_weights,
            "max_sharpe_weights": max_sharpe_weights,
            "min_vol_weights": min_vol_weights,
            "monte_carlo_results": monte_carlo_results,
            "monte_carlo_weights": monte_carlo_weights,
            "mc_max_idx": mc_max_idx,
            "mc_min_idx": mc_min_idx,
            "strategy_comparison": strategy_comparison,
            "portfolio_returns": portfolio_returns,
            "benchmark_returns": benchmark_returns,
            "benchmark_warning": benchmark_warning,
            "advanced_risk_metrics": advanced_risk_metrics,
            "backtest_returns": backtest_returns,
            "backtest_weights": backtest_weights,
            "backtest_turnover": backtest_turnover,
            "backtest_metrics": backtest_metrics,
            "backtest_warnings": backtest_warnings,
            "backtest_error": backtest_error,
            "training_window": training_window,
            "rebalance_frequency": rebalance_frequency,
            "transaction_cost": transaction_cost,
            "risk_free_rate": risk_free_rate,
            "cleaning_warnings": cleaning_warnings,
        }

if "analysis" not in st.session_state:
    st.markdown(
        """
        <div class="info-panel">
            Select stocks and dates in the sidebar, then click <b>Run Full Analysis</b>.
            The dashboard will download adjusted close prices, clean missing data,
            calculate returns and risk, run Monte Carlo simulation, optimize portfolios,
            and generate downloadable CSV results.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

analysis = st.session_state.analysis
required_analysis_keys = [
    "portfolio_returns",
    "advanced_risk_metrics",
    "backtest_returns",
    "backtest_weights",
    "backtest_turnover",
    "backtest_metrics",
]
missing_analysis_keys = [key for key in required_analysis_keys if key not in analysis]
if missing_analysis_keys:
    st.info("The dashboard code was updated. Please click Run Full Analysis again to generate the new analytics.")
    st.stop()

prices = analysis["prices"]
returns = analysis["returns"]
annual_returns = analysis["annual_returns"]
annual_covariance = analysis["annual_covariance"]
correlation = analysis["correlation"]
summary_table = analysis["summary_table"]
random_weights_arr = analysis["random_weights"]
max_sharpe_weights_arr = analysis["max_sharpe_weights"]
min_vol_weights_arr = analysis["min_vol_weights"]
monte_carlo_results = analysis["monte_carlo_results"]
mc_max_idx = analysis["mc_max_idx"]
mc_min_idx = analysis["mc_min_idx"]
strategy_comparison = analysis["strategy_comparison"]
portfolio_returns = analysis["portfolio_returns"]
benchmark_returns = analysis["benchmark_returns"]
benchmark_warning = analysis["benchmark_warning"]
advanced_risk_metrics = analysis["advanced_risk_metrics"]
backtest_returns = analysis["backtest_returns"]
backtest_weights = analysis["backtest_weights"]
backtest_turnover = analysis["backtest_turnover"]
backtest_metrics = analysis["backtest_metrics"]
backtest_warnings = analysis["backtest_warnings"]
backtest_error = analysis["backtest_error"]
active_tickers = list(annual_returns.index)
active_risk_free_rate = analysis["risk_free_rate"]


# -----------------------------
# 7. Dashboard tabs
# -----------------------------

tabs = st.tabs(
    [
        "Overview",
        "Data",
        "Prices",
        "Returns",
        "Risk Table",
        "Matrices",
        "Random",
        "Monte Carlo",
        "Efficient Frontier",
        "Max Sharpe",
        "Min Volatility",
        "Comparison",
        "Advanced Risk",
        "Backtest",
        "Black-Litterman",
        "Factors",
        "Machine Learning",
        "Conclusion",
        "Future Scope",
        "Disclaimer",
    ]
)

with tabs[0]:
    st.subheader("Project Overview")
    best_strategy = strategy_comparison.loc[strategy_comparison["Sharpe Ratio"].idxmax()]
    lowest_risk = strategy_comparison.loc[strategy_comparison["Annual Risk"].idxmin()]
    best_return = strategy_comparison.loc[strategy_comparison["Annual Return"].idxmax()]

    cols = st.columns(4)
    with cols[0]:
        metric_card("Active Stocks", str(len(active_tickers)), "After data cleaning")
    with cols[1]:
        metric_card("Best Strategy", str(best_strategy["Strategy"]), "Highest Sharpe ratio")
    with cols[2]:
        metric_card("Best Return", fmt_pct(float(best_return["Annual Return"])), str(best_return["Strategy"]), True)
    with cols[3]:
        metric_card("Lowest Risk", fmt_pct(float(lowest_risk["Annual Risk"])), str(lowest_risk["Strategy"]))

    st.markdown(
        """
        <div class="info-panel">
        This dashboard applies Modern Portfolio Theory to Indian NSE stocks.
        It uses adjusted close prices, daily returns, annual expected returns,
        covariance, correlation, Monte Carlo simulation, and SciPy optimization.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tabs[1]:
    st.subheader("Stock Data Download")
    if analysis["cleaning_warnings"]:
        for warning in analysis["cleaning_warnings"]:
            st.warning(warning)

    cols = st.columns(3)
    with cols[0]:
        metric_card("Price Rows", f"{len(prices):,}", "Trading dates")
    with cols[1]:
        metric_card("Return Rows", f"{len(returns):,}", "Daily return observations")
    with cols[2]:
        metric_card("Date Range", f"{prices.index.min().date()} to {prices.index.max().date()}")

    st.dataframe(prices.head(), use_container_width=True)
    dataframe_download(prices, "Download cleaned stock data CSV", "cleaned_stock_data.csv")

with tabs[2]:
    st.subheader("Price Trend Charts")
    chart_tickers = active_tickers[: min(10, len(active_tickers))]
    st.pyplot(price_trend_chart(prices, chart_tickers), use_container_width=True)
    st.caption("Only the first 10 active tickers are plotted to keep the chart readable.")

with tabs[3]:
    st.subheader("Daily Returns Analysis")
    st.dataframe(returns.head(), use_container_width=True)
    st.pyplot(returns_bar_chart(summary_table), use_container_width=True)
    dataframe_download(returns, "Download returns data CSV", "daily_returns.csv")

with tabs[4]:
    st.subheader("Risk and Return Table")
    display_summary = summary_table.copy()
    display_summary["Annual Expected Return"] = display_summary["Annual Expected Return"].map(fmt_pct)
    display_summary["Annual Volatility"] = display_summary["Annual Volatility"].map(fmt_pct)
    st.dataframe(display_summary, use_container_width=True)

with tabs[5]:
    st.subheader("Covariance and Correlation Matrix")
    st.write("Covariance shows how stocks move together in return units. Correlation standardizes this relationship between -1 and +1.")
    st.pyplot(correlation_heatmap(correlation), use_container_width=True)
    matrix_choice = st.radio("Matrix table", ["Correlation", "Covariance"], horizontal=True)
    st.dataframe(correlation if matrix_choice == "Correlation" else annual_covariance, use_container_width=True)

with tabs[6]:
    st.subheader("Random Portfolio Analysis")
    random_table = allocation_table(active_tickers, random_weights_arr)
    random_row = calculate_strategy_row("Random", random_weights_arr, annual_returns, annual_covariance, active_risk_free_rate)
    cols = st.columns(3)
    with cols[0]:
        metric_card("Return", fmt_pct(float(random_row["Annual Return"])), positive=float(random_row["Annual Return"]) >= 0)
    with cols[1]:
        metric_card("Risk", fmt_pct(float(random_row["Annual Risk"])))
    with cols[2]:
        metric_card("Sharpe", f"{float(random_row['Sharpe Ratio']):.4f}", positive=float(random_row["Sharpe Ratio"]) >= 0)
    st.dataframe(random_table, use_container_width=True)
    st.pyplot(allocation_chart("Random Portfolio Allocation", random_table), use_container_width=True)

with tabs[7]:
    st.subheader("Monte Carlo Simulation")
    max_mc = monte_carlo_results.loc[mc_max_idx]
    min_mc = monte_carlo_results.loc[mc_min_idx]
    cols = st.columns(3)
    with cols[0]:
        metric_card("Random Portfolios", f"{len(monte_carlo_results):,}")
    with cols[1]:
        metric_card("Best MC Sharpe", f"{max_mc['Sharpe Ratio']:.4f}", "Approximate, random search", True)
    with cols[2]:
        metric_card("Lowest MC Risk", fmt_pct(float(min_mc["Annual Risk"])))
    st.dataframe(monte_carlo_results.sort_values("Sharpe Ratio", ascending=False).head(10), use_container_width=True)

with tabs[8]:
    st.subheader("Efficient Frontier")
    max_mc = monte_carlo_results.loc[mc_max_idx]
    min_mc = monte_carlo_results.loc[mc_min_idx]
    st.pyplot(monte_carlo_chart(monte_carlo_results, max_mc, min_mc), use_container_width=True)
    st.markdown(
        """
        The Efficient Frontier shows portfolios that offer better return for a given risk level.
        In simple words, portfolios higher and further left are more attractive because they
        offer more return with less volatility.
        """
    )

with tabs[9]:
    st.subheader("Maximum Sharpe Portfolio")
    max_table = allocation_table(active_tickers, max_sharpe_weights_arr)
    max_row = calculate_strategy_row("Max Sharpe", max_sharpe_weights_arr, annual_returns, annual_covariance, active_risk_free_rate)
    cols = st.columns(3)
    with cols[0]:
        metric_card("Optimized Return", fmt_pct(float(max_row["Annual Return"])), positive=float(max_row["Annual Return"]) >= 0)
    with cols[1]:
        metric_card("Optimized Risk", fmt_pct(float(max_row["Annual Risk"])))
    with cols[2]:
        metric_card("Optimized Sharpe", f"{float(max_row['Sharpe Ratio']):.4f}", positive=float(max_row["Sharpe Ratio"]) >= 0)
    st.dataframe(max_table, use_container_width=True)
    st.pyplot(allocation_chart("Maximum Sharpe Allocation", max_table), use_container_width=True)
    dataframe_download(max_table.set_index("Ticker"), "Download optimized allocation CSV", "max_sharpe_allocation.csv")

with tabs[10]:
    st.subheader("Minimum Volatility Portfolio")
    min_table = allocation_table(active_tickers, min_vol_weights_arr)
    min_row = calculate_strategy_row("Min Volatility", min_vol_weights_arr, annual_returns, annual_covariance, active_risk_free_rate)
    cols = st.columns(3)
    with cols[0]:
        metric_card("Portfolio Return", fmt_pct(float(min_row["Annual Return"])), positive=float(min_row["Annual Return"]) >= 0)
    with cols[1]:
        metric_card("Minimum Risk", fmt_pct(float(min_row["Annual Risk"])))
    with cols[2]:
        metric_card("Sharpe Ratio", f"{float(min_row['Sharpe Ratio']):.4f}", positive=float(min_row["Sharpe Ratio"]) >= 0)
    st.dataframe(min_table, use_container_width=True)
    st.pyplot(allocation_chart("Minimum Volatility Allocation", min_table), use_container_width=True)

with tabs[11]:
    st.subheader("Portfolio Strategy Comparison")
    formatted_comparison = strategy_comparison.copy()
    formatted_comparison["Annual Return"] = formatted_comparison["Annual Return"].map(fmt_pct)
    formatted_comparison["Annual Risk"] = formatted_comparison["Annual Risk"].map(fmt_pct)
    formatted_comparison["Sharpe Ratio"] = formatted_comparison["Sharpe Ratio"].map(lambda value: f"{value:.4f}")
    st.dataframe(formatted_comparison, use_container_width=True)
    chart_cols = st.columns(3)
    with chart_cols[0]:
        st.pyplot(comparison_chart(strategy_comparison, "Annual Return", "Return Comparison", True), use_container_width=True)
    with chart_cols[1]:
        st.pyplot(comparison_chart(strategy_comparison, "Annual Risk", "Risk Comparison", True), use_container_width=True)
    with chart_cols[2]:
        st.pyplot(comparison_chart(strategy_comparison, "Sharpe Ratio", "Sharpe Comparison", False), use_container_width=True)
    dataframe_download(strategy_comparison.set_index("Strategy"), "Download comparison result CSV", "portfolio_strategy_comparison.csv")

with tabs[12]:
    st.subheader("Advanced Risk Analytics")
    if benchmark_warning:
        st.warning(benchmark_warning + " Benchmark-specific metrics are shown as N/A.")
    else:
        st.success("NIFTY 50 benchmark data loaded successfully for beta, tracking error, and information ratio.")

    metric_cols = st.columns(4)
    best_sharpe = advanced_risk_metrics.loc[advanced_risk_metrics["Sharpe Ratio"].idxmax()]
    lowest_var = advanced_risk_metrics.loc[advanced_risk_metrics["VaR 95%"].idxmin()]
    lowest_drawdown = advanced_risk_metrics.loc[advanced_risk_metrics["Maximum Drawdown"].idxmax()]
    best_sortino = advanced_risk_metrics.loc[advanced_risk_metrics["Sortino Ratio"].idxmax()]
    with metric_cols[0]:
        metric_card("Best Sharpe", str(best_sharpe["Strategy"]), f"{best_sharpe['Sharpe Ratio']:.4f}", True)
    with metric_cols[1]:
        metric_card("Best Sortino", str(best_sortino["Strategy"]), f"{best_sortino['Sortino Ratio']:.4f}", True)
    with metric_cols[2]:
        metric_card("Lowest VaR 95%", str(lowest_var["Strategy"]), fmt_pct(float(lowest_var["VaR 95%"])))
    with metric_cols[3]:
        metric_card("Smallest Drawdown", str(lowest_drawdown["Strategy"]), fmt_pct(float(lowest_drawdown["Maximum Drawdown"])), False)

    advanced_display = advanced_risk_metrics.copy()
    percent_columns = [
        "Cumulative Return",
        "Annual Return",
        "Annual Volatility",
        "Maximum Drawdown",
        "VaR 95%",
        "CVaR 95%",
        "Tracking Error",
    ]
    number_columns = [
        "Sharpe Ratio",
        "Sortino Ratio",
        "Calmar Ratio",
        "Beta vs NIFTY 50",
        "Information Ratio",
    ]
    for column in percent_columns:
        advanced_display[column] = advanced_display[column].map(lambda value: "N/A" if pd.isna(value) else fmt_pct(float(value)))
    for column in number_columns:
        advanced_display[column] = advanced_display[column].map(lambda value: "N/A" if pd.isna(value) else f"{float(value):.4f}")

    st.dataframe(advanced_display, use_container_width=True)
    dataframe_download(advanced_risk_metrics.set_index("Strategy"), "Download advanced risk metrics CSV", "advanced_risk_metrics.csv")

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.pyplot(cumulative_return_chart(portfolio_returns, benchmark_returns), use_container_width=True)
    with chart_cols[1]:
        st.pyplot(drawdown_chart(portfolio_returns), use_container_width=True)

    higher_is_better = [
        "Cumulative Return",
        "Annual Return",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Calmar Ratio",
        "Information Ratio",
    ]
    lower_is_better = [
        "Annual Volatility",
        "VaR 95%",
        "CVaR 95%",
        "Tracking Error",
    ]
    best_rows = []
    for metric in higher_is_better:
        valid = advanced_risk_metrics.dropna(subset=[metric])
        if not valid.empty:
            best_rows.append({"Metric": metric, "Best Strategy": valid.loc[valid[metric].idxmax(), "Strategy"]})
    valid_drawdown = advanced_risk_metrics.dropna(subset=["Maximum Drawdown"])
    if not valid_drawdown.empty:
        best_rows.append({"Metric": "Maximum Drawdown", "Best Strategy": valid_drawdown.loc[valid_drawdown["Maximum Drawdown"].idxmax(), "Strategy"]})
    for metric in lower_is_better:
        valid = advanced_risk_metrics.dropna(subset=[metric])
        if not valid.empty:
            best_rows.append({"Metric": metric, "Best Strategy": valid.loc[valid[metric].idxmin(), "Strategy"]})
    st.dataframe(pd.DataFrame(best_rows), use_container_width=True)

    st.markdown(
        """
        <div class="info-panel">
        <b>Formula guide:</b><br>
        Sortino Ratio = (Annual Return - Risk-Free Rate) / Downside Volatility.<br>
        Maximum Drawdown = largest percentage fall from a previous portfolio peak.<br>
        VaR 95% = estimated daily loss level not exceeded on 95% of historical days.<br>
        CVaR 95% = average loss on the worst 5% historical days.<br>
        Beta = sensitivity of portfolio returns to NIFTY 50 returns.<br>
        Tracking Error = annualized volatility of portfolio return minus benchmark return.<br>
        Information Ratio = annualized active return divided by tracking error.<br><br>
        These metrics are based on historical data only and do not guarantee future results.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tabs[13]:
    st.subheader("Walk-Forward Backtesting")
    if backtest_error:
        st.warning(backtest_error)
    else:
        if backtest_warnings:
            st.warning(f"{len(backtest_warnings)} rebalance warnings occurred. The app used equal weights as fallback when needed.")

        st.markdown(
            f"""
            <div class="info-panel">
            This backtest uses a rolling training window of
            <b>{analysis["training_window"]}</b> trading days and rebalances every
            <b>{analysis["rebalance_frequency"]}</b> trading days. Transaction cost is
            <b>{analysis["transaction_cost"] * 100:.2f}% per turnover</b>.
            The optimizer uses only past data before each holding period.
            </div>
            """,
            unsafe_allow_html=True,
        )

        backtest_display = backtest_metrics.copy()
        percent_columns = [
            "Total Return",
            "Annual Return",
            "Annual Volatility",
            "Maximum Drawdown",
            "Average Turnover",
        ]
        for column in percent_columns:
            backtest_display[column] = backtest_display[column].map(lambda value: "N/A" if pd.isna(value) else fmt_pct(float(value)))
        backtest_display["Sharpe Ratio"] = backtest_display["Sharpe Ratio"].map(lambda value: "N/A" if pd.isna(value) else f"{float(value):.4f}")

        best_backtest = backtest_metrics.loc[backtest_metrics["Sharpe Ratio"].idxmax()]
        lowest_backtest_risk = backtest_metrics.loc[backtest_metrics["Annual Volatility"].idxmin()]
        cols = st.columns(4)
        with cols[0]:
            metric_card("Best Backtest Sharpe", str(best_backtest["Strategy"]), f"{best_backtest['Sharpe Ratio']:.4f}", True)
        with cols[1]:
            metric_card("Lowest Volatility", str(lowest_backtest_risk["Strategy"]), fmt_pct(float(lowest_backtest_risk["Annual Volatility"])))
        with cols[2]:
            metric_card("Rebalances", str(int(backtest_metrics["Number of Rebalances"].max())))
        with cols[3]:
            metric_card("Avg Optimized Turnover", fmt_pct(float(backtest_turnover["Optimized Turnover"].mean())))

        st.dataframe(backtest_display, use_container_width=True)
        dataframe_download(backtest_metrics.set_index("Strategy"), "Download backtest metrics CSV", "walk_forward_backtest_metrics.csv")
        dataframe_download(backtest_returns, "Download backtest daily returns CSV", "walk_forward_backtest_returns.csv")
        dataframe_download(backtest_weights, "Download rebalance weights CSV", "walk_forward_backtest_weights.csv")

        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.pyplot(backtest_equity_chart(backtest_returns), use_container_width=True)
        with chart_cols[1]:
            st.pyplot(backtest_drawdown_chart(backtest_returns), use_container_width=True)
        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.pyplot(backtest_weights_chart(backtest_weights), use_container_width=True)
        with chart_cols[1]:
            st.pyplot(backtest_turnover_chart(backtest_turnover), use_container_width=True)

        st.markdown(
            """
            <div class="info-panel">
            <b>Look-ahead bias</b> means accidentally using future information when making
            a historical decision. This backtest avoids that by optimizing only on past
            returns before each rebalance. Backtesting is more realistic than showing one
            optimized allocation because it tests repeated decisions through time.
            Transaction costs matter because frequent rebalancing can reduce returns.
            Backtest results are historical simulations and do not guarantee future profit.
            </div>
            """,
            unsafe_allow_html=True,
        )

with tabs[14]:
    st.subheader("Black-Litterman Model")
    st.markdown(
        """
        <div class="info-panel">
        Black-Litterman combines market equilibrium returns with investor views.
        This can reduce the extreme weights that sometimes come from historical
        mean-variance optimization. Views are subjective, so wrong views can make
        results worse.
        </div>
        """,
        unsafe_allow_html=True,
    )

    bl_tau = st.slider("Tau: uncertainty in market equilibrium returns", 0.01, 0.20, 0.05, 0.01)
    market_weight_mode = st.radio("Market weight source", ["Equal weights", "Custom weights"], horizontal=True)

    custom_market_weights = None
    if market_weight_mode == "Custom weights":
        default_weight_table = pd.DataFrame(
            {
                "Ticker": active_tickers,
                "Custom Weight": np.repeat(1 / len(active_tickers), len(active_tickers)),
            }
        )
        edited_weight_table = st.data_editor(
            default_weight_table,
            use_container_width=True,
            hide_index=True,
            key="black_litterman_custom_weights",
        )
        custom_market_weights = edited_weight_table.set_index("Ticker")["Custom Weight"]

    st.write("Investor views")
    view_cols = st.columns(3)
    with view_cols[0]:
        reliance_vs_tcs = st.number_input("RELIANCE outperforms TCS by", value=3.0, step=0.5, format="%.2f") / 100
        reliance_conf = st.slider("Confidence: RELIANCE vs TCS", 0.01, 0.99, 0.60, 0.01)
    with view_cols[1]:
        infy_absolute = st.number_input("INFY expected annual return", value=12.0, step=0.5, format="%.2f") / 100
        infy_conf = st.slider("Confidence: INFY absolute view", 0.01, 0.99, 0.55, 0.01)
    with view_cols[2]:
        hdfc_vs_icici = st.number_input("HDFCBANK outperforms ICICIBANK by", value=2.0, step=0.5, format="%.2f") / 100
        hdfc_conf = st.slider("Confidence: HDFCBANK vs ICICIBANK", 0.01, 0.99, 0.60, 0.01)

    bl_views = [
        {
            "type": "relative",
            "long": "RELIANCE.NS",
            "short": "TCS.NS",
            "value": reliance_vs_tcs,
            "confidence": reliance_conf,
            "description": f"RELIANCE.NS expected to outperform TCS.NS by {reliance_vs_tcs * 100:.2f}%",
        },
        {
            "type": "absolute",
            "ticker": "INFY.NS",
            "value": infy_absolute,
            "confidence": infy_conf,
            "description": f"INFY.NS expected annual return is {infy_absolute * 100:.2f}%",
        },
        {
            "type": "relative",
            "long": "HDFCBANK.NS",
            "short": "ICICIBANK.NS",
            "value": hdfc_vs_icici,
            "confidence": hdfc_conf,
            "description": f"HDFCBANK.NS expected to outperform ICICIBANK.NS by {hdfc_vs_icici * 100:.2f}%",
        },
    ]

    bl_market_weights, bl_weight_message = black_litterman_market_weights(active_tickers, custom_market_weights)
    bl_risk_aversion = black_litterman_risk_aversion(returns, bl_market_weights, active_risk_free_rate)
    bl_implied_returns = black_litterman_implied_returns(
        annual_covariance,
        bl_market_weights,
        bl_risk_aversion,
        active_risk_free_rate,
    )
    p_matrix, q_vector, omega_matrix, valid_view_descriptions = black_litterman_views_matrices(
        active_tickers,
        bl_views,
        annual_covariance,
        bl_tau,
    )

    if p_matrix is None:
        st.warning("None of the Black-Litterman views matched the selected stocks. Select RELIANCE, TCS, INFY, HDFCBANK, and ICICIBANK to use the default views.")
    else:
        bl_posterior_returns = black_litterman_posterior_returns(
            bl_implied_returns,
            annual_covariance,
            p_matrix,
            q_vector,
            omega_matrix,
            bl_tau,
        )
        bl_weights, bl_error = optimize_max_sharpe(
            bl_posterior_returns,
            annual_covariance,
            active_risk_free_rate,
        )

        if bl_weights is None:
            st.error(bl_error)
        else:
            historical_result = calculate_strategy_row(
                "Historical Mean Optimization",
                max_sharpe_weights_arr,
                annual_returns,
                annual_covariance,
                active_risk_free_rate,
            )
            bl_result = calculate_strategy_row(
                "Black-Litterman Optimization",
                bl_weights,
                bl_posterior_returns,
                annual_covariance,
                active_risk_free_rate,
            )
            bl_result_table = pd.DataFrame([historical_result, bl_result])
            bl_allocation = black_litterman_allocation_comparison(active_tickers, max_sharpe_weights_arr, bl_weights)

            bl_cols = st.columns(3)
            with bl_cols[0]:
                metric_card("Risk Aversion", f"{bl_risk_aversion:.4f}", bl_weight_message)
            with bl_cols[1]:
                metric_card("BL Return", fmt_pct(float(bl_result["Annual Return"])), positive=float(bl_result["Annual Return"]) >= 0)
            with bl_cols[2]:
                metric_card("BL Sharpe", f"{float(bl_result['Sharpe Ratio']):.4f}", positive=float(bl_result["Sharpe Ratio"]) >= 0)

            st.write("Valid views used:")
            for description in valid_view_descriptions:
                st.markdown(f"- {description}")

            display_bl_results = bl_result_table.copy()
            display_bl_results["Annual Return"] = display_bl_results["Annual Return"].map(fmt_pct)
            display_bl_results["Annual Risk"] = display_bl_results["Annual Risk"].map(fmt_pct)
            display_bl_results["Sharpe Ratio"] = display_bl_results["Sharpe Ratio"].map(lambda value: f"{value:.4f}")
            st.dataframe(display_bl_results, use_container_width=True)

            display_bl_allocation = bl_allocation.copy()
            for column in ["Historical Mean Allocation %", "Black-Litterman Allocation %"]:
                display_bl_allocation[column] = display_bl_allocation[column].map(lambda value: f"{value:.2f}%")
            st.dataframe(display_bl_allocation, use_container_width=True)
            st.pyplot(black_litterman_allocation_chart(bl_allocation), use_container_width=True)
            dataframe_download(bl_allocation.set_index("Ticker"), "Download Black-Litterman allocation CSV", "black_litterman_allocation.csv")

            expected_return_table = pd.DataFrame(
                {
                    "Ticker": active_tickers,
                    "Historical Mean Return": annual_returns.values,
                    "Implied Equilibrium Return": bl_implied_returns.values,
                    "Black-Litterman Return": bl_posterior_returns.values,
                    "Market Weight": bl_market_weights.values,
                }
            )
            dataframe_download(expected_return_table.set_index("Ticker"), "Download Black-Litterman expected returns CSV", "black_litterman_expected_returns.csv")

    st.markdown(
        """
        <div class="info-panel">
        <b>Simple explanation:</b> historical mean returns can be noisy, and small
        changes in expected returns can create extreme optimized weights.
        Black-Litterman starts with market-implied equilibrium returns, then gently
        adjusts them using investor views and confidence levels. This often creates
        more stable portfolios, but views are subjective and can hurt results if
        the assumptions are wrong.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tabs[15]:
    st.subheader("Factor Investing Analysis")
    st.markdown(
        """
        <div class="info-panel">
        Factor investing studies stock characteristics that may explain returns.
        This dashboard uses practical proxy factors: momentum, low volatility,
        value, quality, and size. Indian fundamental data from free sources can
        be incomplete, so missing PE, ROE, or market-cap values are treated as
        neutral scores.
        </div>
        """,
        unsafe_allow_html=True,
    )

    fetch_fundamentals_for_factors = st.checkbox(
        "Fetch PE, ROE, and market cap from yfinance",
        value=False,
        help="This can be slower and yfinance fundamental data may be incomplete.",
    )
    factor_portfolio_size = st.slider(
        "Number of top-ranked stocks in factor portfolio",
        min_value=3,
        max_value=max(3, min(25, len(active_tickers))),
        value=min(10, len(active_tickers)),
        step=1,
    )

    if fetch_fundamentals_for_factors:
        with st.spinner("Fetching yfinance fundamental proxies..."):
            factor_fundamentals = fetch_factor_fundamentals(tuple(active_tickers))
    else:
        factor_fundamentals = None

    factor_scores = build_factor_score_table(prices, returns, factor_fundamentals)
    factor_weights = factor_portfolio_weights(factor_scores, factor_portfolio_size)

    equal_weight_series = pd.Series(np.repeat(1 / len(active_tickers), len(active_tickers)), index=active_tickers)
    max_sharpe_series = pd.Series(max_sharpe_weights_arr, index=active_tickers)
    min_vol_series = pd.Series(min_vol_weights_arr, index=active_tickers)

    factor_strategies = {
        "Factor Portfolio": factor_weights,
        "Equal Weight Portfolio": equal_weight_series,
        "Maximum Sharpe Portfolio": max_sharpe_series,
        "Minimum Volatility Portfolio": min_vol_series,
    }

    factor_comparison = pd.DataFrame(
        [
            factor_strategy_metrics(name, weights, returns, active_risk_free_rate)
            for name, weights in factor_strategies.items()
        ]
    )
    factor_daily_returns = pd.DataFrame(
        {
            name: returns @ weights.reindex(active_tickers).fillna(0)
            for name, weights in factor_strategies.items()
        }
    )

    best_factor_strategy = factor_comparison.loc[factor_comparison["Sharpe Ratio"].idxmax()]
    top_factor_stock = factor_scores.iloc[0]

    factor_cols = st.columns(4)
    with factor_cols[0]:
        metric_card("Top Factor Stock", str(top_factor_stock.name), f"Score {top_factor_stock['Combined Factor Score']:.3f}", True)
    with factor_cols[1]:
        metric_card("Factor Portfolio Size", str(int((factor_weights > 0).sum())))
    with factor_cols[2]:
        metric_card("Best Strategy Sharpe", str(best_factor_strategy["Strategy"]), f"{best_factor_strategy['Sharpe Ratio']:.4f}", True)
    with factor_cols[3]:
        missing_count = int(factor_scores["Missing Fundamental Count"].sum())
        metric_card("Missing Fundamental Fields", str(missing_count), "Neutral scores used")

    factor_display = factor_scores.copy()
    percent_columns = ["Momentum", "Annual Volatility", "Return on Equity"]
    for column in percent_columns:
        factor_display[column] = factor_display[column].map(lambda value: "N/A" if pd.isna(value) else fmt_pct(float(value)))
    factor_display["Market Cap"] = factor_display["Market Cap"].map(lambda value: "N/A" if pd.isna(value) else f"{float(value):,.0f}")
    for column in ["PE Ratio", "Combined Factor Score"]:
        factor_display[column] = factor_display[column].map(lambda value: "N/A" if pd.isna(value) else f"{float(value):.4f}")

    st.dataframe(factor_display, use_container_width=True)
    st.pyplot(factor_score_chart(factor_scores), use_container_width=True)

    factor_comp_display = factor_comparison.copy()
    for column in ["Total Return", "Annual Return", "Annual Volatility"]:
        factor_comp_display[column] = factor_comp_display[column].map(lambda value: "N/A" if pd.isna(value) else fmt_pct(float(value)))
    factor_comp_display["Sharpe Ratio"] = factor_comp_display["Sharpe Ratio"].map(lambda value: "N/A" if pd.isna(value) else f"{float(value):.4f}")
    st.dataframe(factor_comp_display, use_container_width=True)
    st.pyplot(factor_performance_chart(factor_daily_returns), use_container_width=True)

    factor_weight_table = pd.DataFrame({"Ticker": factor_weights.index, "Weight": factor_weights.values})
    dataframe_download(factor_scores, "Download factor score table CSV", "factor_score_table.csv")
    dataframe_download(factor_weight_table.set_index("Ticker"), "Download factor portfolio weights CSV", "factor_portfolio_weights.csv")
    dataframe_download(factor_comparison.set_index("Strategy"), "Download factor comparison CSV", "factor_portfolio_comparison.csv")

    st.markdown(
        """
        <div class="info-panel">
        <b>Factor guide:</b><br>
        Market factor means broad market exposure. Size compares smaller and larger companies.
        Value looks for cheaper stocks, proxied here by lower PE. Momentum favors stocks
        with stronger recent returns. Quality favors stronger ROE. Low volatility favors
        stocks with smoother historical returns.<br><br>
        Factor investing is useful because it explains portfolios through clear drivers,
        but it is not guaranteed to work. Factors can underperform for long periods, and
        proxy data from yfinance may be incomplete or inconsistent.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tabs[16]:
    st.subheader("Machine Learning Return Prediction")
    st.markdown(
        """
        <div class="info-panel">
        This section uses basic machine learning to predict each selected stock's
        next 21 trading day return. The predictions are then used as expected
        returns for a portfolio optimizer and compared with historical mean-return
        optimization and equal weighting. This is for education only.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if SKLEARN_IMPORT_ERROR is not None:
        st.error(
            "scikit-learn is not installed in this Python environment. "
            "Install project dependencies with: pip install -r requirements.txt"
        )
    else:
        run_ml_analysis = st.button(
            "Run ML Return Prediction",
            key="run_ml_return_prediction",
            use_container_width=True,
        )

        if run_ml_analysis:
            with st.spinner("Training ML models with time-series validation..."):
                try:
                    st.session_state.ml_analysis = run_ml_return_prediction(
                        prices,
                        returns,
                        active_risk_free_rate,
                    )
                except Exception as error:
                    st.error(f"ML analysis failed: {error}")
                    st.stop()

        ml_analysis = st.session_state.get("ml_analysis")

        if ml_analysis is None:
            st.info("Click Run ML Return Prediction to train models and build the ML portfolio comparison.")
        else:
            best_model_name = ml_analysis["best_model_name"]
            performance_summary = ml_analysis["performance_summary"]
            performance = ml_analysis["performance"]
            predictions = ml_analysis["predictions"]
            latest_predictions = ml_analysis["latest_predictions"]
            portfolio_comparison_ml = ml_analysis["portfolio_comparison"]
            weights_table_ml = ml_analysis["weights_table"]

            best_performance = performance_summary[performance_summary["Model"] == best_model_name].iloc[0]
            ml_cols = st.columns(4)
            with ml_cols[0]:
                metric_card("Best ML Model", str(best_model_name), "Lowest average MAE")
            with ml_cols[1]:
                metric_card("MAE", f"{best_performance['MAE']:.4f}", "Lower is better")
            with ml_cols[2]:
                metric_card("RMSE", f"{best_performance['RMSE']:.4f}", "Penalizes large errors")
            with ml_cols[3]:
                metric_card(
                    "Directional Accuracy",
                    fmt_pct(float(best_performance["Directional Accuracy"])),
                    "Correct positive/negative sign",
                    positive=float(best_performance["Directional Accuracy"]) >= 0.50,
                )

            st.markdown(build_ml_report_section(best_model_name, performance_summary), unsafe_allow_html=True)

            st.subheader("Model Performance")
            performance_display = performance_summary.copy()
            performance_display["MAE"] = performance_display["MAE"].map(lambda value: f"{value:.4f}")
            performance_display["RMSE"] = performance_display["RMSE"].map(lambda value: f"{value:.4f}")
            performance_display["Directional Accuracy"] = performance_display["Directional Accuracy"].map(fmt_pct)
            st.dataframe(performance_display, use_container_width=True)

            with st.expander("View fold-by-fold validation results"):
                fold_display = performance.copy()
                for column in ["MAE", "RMSE"]:
                    fold_display[column] = fold_display[column].map(lambda value: f"{value:.4f}")
                fold_display["Directional Accuracy"] = fold_display["Directional Accuracy"].map(fmt_pct)
                st.dataframe(fold_display, use_container_width=True)

            st.subheader("Predicted vs Actual Returns")
            st.pyplot(ml_predicted_vs_actual_chart(predictions, best_model_name), use_container_width=True)

            st.subheader("Latest ML Expected Returns")
            latest_display = latest_predictions.copy()
            st.dataframe(
                style_return_risk_table(latest_display).format(
                    {
                        "Predicted 21D Return": "{:.2%}",
                        "Annualized Predicted Return": "{:.2%}",
                    }
                ),
                use_container_width=True,
            )

            st.subheader("Portfolio Comparison")
            st.dataframe(
                style_return_risk_table(portfolio_comparison_ml).format(
                    {
                        "Historical Expected Annual Return": "{:.2%}",
                        "ML Expected Annual Return": "{:.2%}",
                        "Annual Risk": "{:.2%}",
                        "Sharpe Using Historical Return": "{:.4f}",
                        "Sharpe Using ML Return": "{:.4f}",
                    }
                ),
                use_container_width=True,
            )
            st.pyplot(ml_allocation_comparison_chart(weights_table_ml), use_container_width=True)

            weights_display = weights_table_ml.copy()
            st.dataframe(
                style_return_risk_table(weights_display).format(
                    {
                        "Historical Mean Optimization": "{:.2%}",
                        "ML Prediction Optimization": "{:.2%}",
                        "Equal Weight Portfolio": "{:.2%}",
                        "Historical Annual Return": "{:.2%}",
                        "ML Annualized Predicted Return": "{:.2%}",
                    }
                ),
                use_container_width=True,
            )

            dataframe_download(ml_analysis["dataset"], "Download ML feature dataset CSV", "ml_feature_dataset.csv")
            dataframe_download(performance, "Download ML model performance CSV", "ml_model_performance.csv")
            dataframe_download(predictions, "Download ML predictions CSV", "ml_out_of_fold_predictions.csv")
            dataframe_download(latest_predictions, "Download latest predicted returns CSV", "ml_latest_predicted_returns.csv")
            dataframe_download(portfolio_comparison_ml.set_index("Strategy"), "Download ML portfolio comparison CSV", "ml_portfolio_comparison.csv")
            dataframe_download(weights_table_ml.set_index("Ticker"), "Download ML portfolio weights CSV", "ml_portfolio_weights.csv")

with tabs[17]:
    st.subheader("Final Conclusion")
    best_strategy = strategy_comparison.loc[strategy_comparison["Sharpe Ratio"].idxmax()]
    st.markdown(
        f"""
        <div class="info-panel">
        Based on the selected data, the strongest risk-adjusted strategy is
        <span class="green-text">{best_strategy["Strategy"]}</span>, with a Sharpe ratio of
        <span class="green-text">{best_strategy["Sharpe Ratio"]:.4f}</span>.
        A high return alone is not enough; the return must be judged relative to risk.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tabs[18]:
    st.subheader("Future Scope")
    st.markdown(
        """
        - Add NIFTY 50 benchmark comparison.
        - Add transaction costs and taxes.
        - Add rolling Sharpe ratio and drawdown analysis.
        - Add sector constraints.
        - Add VaR and CVaR risk measures.
        - Add backtesting for optimized portfolios.
        """
    )

with tabs[19]:
    st.subheader("Educational Disclaimer")
    st.markdown(
        f"""
        <div class="info-panel">
        <span class="red-text">This project is for educational purposes only.</span>
        It uses historical market data and mathematical assumptions. Past performance
        does not guarantee future results. This dashboard should not be treated as
        investment, trading, tax, or financial advice.
        </div>
        """,
        unsafe_allow_html=True,
    )
