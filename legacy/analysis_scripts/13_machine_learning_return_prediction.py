"""
Machine learning based return prediction for the NSE portfolio project.

This script uses simple supervised machine learning models to predict each
stock's next 21 trading day return, then compares portfolio optimization based
on historical mean returns with optimization based on ML-predicted returns.

Run:

    python scripts/13_machine_learning_return_prediction.py

Educational warning:
Stock return prediction is difficult, uncertain, and noisy. This script is for
learning only. It does not guarantee profit and is not financial advice.
"""

from __future__ import annotations

import os
import logging
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

MPL_CONFIG_DIR = Path(".matplotlib")
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR.resolve()))

warnings.filterwarnings("ignore", message=".*urllib3.*", category=Warning)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as error:  # pragma: no cover - gives a clean runtime message
    raise ImportError(
        "scikit-learn is required for the machine learning module. "
        "Install dependencies with: pip install -r requirements.txt"
    ) from error

YFINANCE_CACHE_DIR = Path(".yfinance_cache")
YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
try:
    yf.cache.set_cache_location(str(YFINANCE_CACHE_DIR.resolve()))
except Exception:
    pass


# -----------------------------
# 1. Settings
# -----------------------------

SELECTED_NSE_TICKERS = [
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
]

PERIOD = "5y"
TRADING_DAYS = 252
RISK_FREE_RATE = 0.06
PREDICTION_HORIZON_DAYS = 21
RSI_PERIOD = 14
N_TIME_SPLITS = 5
RANDOM_STATE = 42
MAX_MISSING_PERCENT = 0.40
MIN_DATASET_ROWS = 250

FEATURE_COLUMNS = [
    "Return 5D",
    "Return 10D",
    "Return 20D",
    "MA 50 Ratio",
    "MA 100 Ratio",
    "Volatility 20D",
    "RSI 14D",
]

OUTPUT_DATA_DIR = Path("data/outputs")
FIGURE_OUTPUT_DIR = Path("reports/figures")
REPORT_OUTPUT_DIR = Path("reports")

FEATURE_DATASET_FILE = OUTPUT_DATA_DIR / "ml_feature_dataset.csv"
MODEL_PERFORMANCE_FILE = OUTPUT_DATA_DIR / "ml_model_performance.csv"
MODEL_PREDICTIONS_FILE = OUTPUT_DATA_DIR / "ml_out_of_fold_predictions.csv"
LATEST_PREDICTIONS_FILE = OUTPUT_DATA_DIR / "ml_latest_predicted_returns.csv"
PORTFOLIO_COMPARISON_FILE = OUTPUT_DATA_DIR / "ml_portfolio_comparison.csv"
PORTFOLIO_WEIGHTS_FILE = OUTPUT_DATA_DIR / "ml_portfolio_weights.csv"

PREDICTED_VS_ACTUAL_CHART_FILE = FIGURE_OUTPUT_DIR / "ml_predicted_vs_actual_returns.png"
ALLOCATION_COMPARISON_CHART_FILE = FIGURE_OUTPUT_DIR / "ml_portfolio_allocation_comparison.png"
REPORT_FILE = REPORT_OUTPUT_DIR / "machine_learning_based_return_prediction.md"

PURE_BLACK = "#000000"
PURE_WHITE = "#FFFFFF"
PURE_GREEN = "#00FF00"
PURE_RED = "#FF0000"
NEUTRAL_GRAY = "#808080"


# -----------------------------
# 2. Data functions
# -----------------------------

def create_output_folders() -> None:
    """Create output folders for CSV files, charts, and the report."""
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_adjusted_close(tickers: list[str], period: str = PERIOD) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance."""
    data = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            prices = data["Adj Close"].copy()
        elif "Close" in data.columns.get_level_values(0):
            prices = data["Close"].copy()
        else:
            return pd.DataFrame()
    else:
        if "Adj Close" in data.columns:
            prices = data[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
        elif "Close" in data.columns:
            prices = data[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            return pd.DataFrame()

    return prices.sort_index()


def clean_prices(prices: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Clean missing values and remove tickers with too much missing data."""
    warnings = []

    if prices.empty:
        return prices, ["No price data was downloaded."]

    missing_percent = prices.isna().mean()
    keep_columns = missing_percent[missing_percent <= MAX_MISSING_PERCENT].index
    removed_columns = missing_percent[missing_percent > MAX_MISSING_PERCENT].index.tolist()

    if removed_columns:
        warnings.append("Removed tickers with too much missing data: " + ", ".join(removed_columns))

    cleaned = prices[keep_columns].ffill().bfill()
    cleaned = cleaned.dropna(axis=1, how="any").dropna(axis=0, how="any")

    return cleaned, warnings


def load_fallback_prices(tickers: list[str]) -> pd.DataFrame:
    """Use the saved project price file if the live yfinance download is unavailable."""
    fallback_file = Path("data/processed/nse_adjusted_close_cleaned.csv")
    if not fallback_file.exists():
        return pd.DataFrame()

    prices = pd.read_csv(fallback_file, index_col=0, parse_dates=True)
    available_tickers = [ticker for ticker in tickers if ticker in prices.columns]

    if not available_tickers:
        return pd.DataFrame()

    return prices[available_tickers].sort_index()


def calculate_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily stock returns."""
    returns = prices.pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan)
    return returns.dropna()


# -----------------------------
# 3. Feature engineering
# -----------------------------

def calculate_rsi(price_series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """
    Calculate RSI, a simple momentum indicator.

    RSI compares average recent gains with average recent losses. Values above
    70 are often read as strong upward momentum, and values below 30 as weak
    momentum, although those rules are not reliable trading signals by themselves.
    """
    delta = price_series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.rolling(period).mean()
    average_loss = losses.rolling(period).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))

    # If average loss is zero and gains exist, RSI is treated as 100.
    rsi = rsi.where(~((average_loss == 0) & (average_gain > 0)), 100)
    return rsi


def build_stock_feature_frame(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Create all ML features and the next-21-day target for one stock."""
    close = prices[ticker].astype(float)
    daily_returns = close.pct_change()

    feature_frame = pd.DataFrame(index=prices.index)
    feature_frame["Ticker"] = ticker

    # Recent returns summarize short-term momentum.
    feature_frame["Return 5D"] = close.pct_change(5)
    feature_frame["Return 10D"] = close.pct_change(10)
    feature_frame["Return 20D"] = close.pct_change(20)

    # Moving-average ratios show whether price is above or below its trend.
    feature_frame["MA 50 Ratio"] = close / close.rolling(50).mean() - 1
    feature_frame["MA 100 Ratio"] = close / close.rolling(100).mean() - 1

    # Rolling volatility measures how much the stock has been moving recently.
    feature_frame["Volatility 20D"] = daily_returns.rolling(20).std()

    # RSI is another way to describe recent momentum.
    feature_frame["RSI 14D"] = calculate_rsi(close)

    # The target is the forward return over the next 21 trading days.
    feature_frame["Target 21D Return"] = close.shift(-PREDICTION_HORIZON_DAYS) / close - 1

    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).dropna()
    feature_frame = feature_frame.reset_index().rename(columns={"index": "Date"})

    return feature_frame


def build_ml_dataset(prices: pd.DataFrame) -> pd.DataFrame:
    """Create a long-format ML dataset across all stocks."""
    frames = [
        build_stock_feature_frame(prices, ticker)
        for ticker in prices.columns
        if prices[ticker].notna().sum() > 120
    ]

    if not frames:
        return pd.DataFrame()

    dataset = pd.concat(frames, ignore_index=True)
    dataset["Date"] = pd.to_datetime(dataset["Date"])
    dataset = dataset.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    return dataset


def build_latest_feature_table(prices: pd.DataFrame) -> pd.DataFrame:
    """Create the most recent valid feature row for each stock, without needing a known target."""
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
        feature_frame["RSI 14D"] = calculate_rsi(close)

        feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).dropna()
        if feature_frame.empty:
            continue

        latest_row = feature_frame.iloc[-1].copy()
        latest_row["Date"] = feature_frame.index[-1]
        rows.append(latest_row)

    if not rows:
        return pd.DataFrame()

    latest_features = pd.DataFrame(rows)
    return latest_features[["Date", "Ticker", *FEATURE_COLUMNS]].reset_index(drop=True)


# -----------------------------
# 4. Model training and evaluation
# -----------------------------

def make_model_definitions() -> dict[str, object]:
    """Create simple ML model definitions."""
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


def directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Measure how often predicted direction matches actual direction."""
    same_direction = ((actual >= 0) & (predicted >= 0)) | ((actual < 0) & (predicted < 0))
    return float(np.mean(same_direction))


def evaluate_models_with_time_series_split(
    dataset: pd.DataFrame,
    n_splits: int = N_TIME_SPLITS,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Evaluate models using expanding time-series splits over dates."""
    unique_dates = pd.Series(dataset["Date"].drop_duplicates()).sort_values().reset_index(drop=True)
    usable_splits = min(n_splits, len(unique_dates) - 1)

    if usable_splits < 2:
        raise ValueError("Not enough unique dates for time-series validation.")

    splitter = TimeSeriesSplit(n_splits=usable_splits)
    models = make_model_definitions()
    performance_rows = []
    prediction_frames = []

    for fold_number, (train_date_index, test_date_index) in enumerate(splitter.split(unique_dates), start=1):
        train_dates = unique_dates.iloc[train_date_index]
        test_dates = unique_dates.iloc[test_date_index]

        train_data = dataset[dataset["Date"].isin(train_dates)]
        test_data = dataset[dataset["Date"].isin(test_dates)]

        x_train = train_data[FEATURE_COLUMNS]
        y_train = train_data["Target 21D Return"]
        x_test = test_data[FEATURE_COLUMNS]
        y_test = test_data["Target 21D Return"]

        for model_name, model in models.items():
            model.fit(x_train, y_train)
            predicted = model.predict(x_test)

            mae = mean_absolute_error(y_test, predicted)
            rmse = float(np.sqrt(mean_squared_error(y_test, predicted)))
            direction = directional_accuracy(y_test.to_numpy(), predicted)

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
                    "MAE": mae,
                    "RMSE": rmse,
                    "Directional Accuracy": direction,
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


def train_best_model_and_predict_latest(
    dataset: pd.DataFrame,
    prices: pd.DataFrame,
    best_model_name: str,
) -> pd.DataFrame:
    """Train the selected model on all labeled rows and predict latest stock returns."""
    models = make_model_definitions()
    model = models[best_model_name]

    model.fit(dataset[FEATURE_COLUMNS], dataset["Target 21D Return"])

    latest_features = build_latest_feature_table(prices)
    if latest_features.empty:
        return pd.DataFrame()

    predicted_21d = model.predict(latest_features[FEATURE_COLUMNS])
    predictions = latest_features[["Date", "Ticker"]].copy()
    predictions["Predicted 21D Return"] = predicted_21d
    predictions["Annualized Predicted Return"] = predicted_21d * (TRADING_DAYS / PREDICTION_HORIZON_DAYS)

    return predictions.sort_values("Predicted 21D Return", ascending=False).reset_index(drop=True)


# -----------------------------
# 5. Portfolio optimization
# -----------------------------

def portfolio_return(weights: np.ndarray, annual_returns: pd.Series) -> float:
    """Calculate portfolio expected annual return."""
    return float(weights @ annual_returns.to_numpy())


def portfolio_risk(weights: np.ndarray, annual_covariance: pd.DataFrame) -> float:
    """Calculate portfolio annual volatility."""
    covariance_values = annual_covariance.to_numpy()
    variance = float(weights.T @ covariance_values @ weights)
    return float(np.sqrt(max(variance, 0)))


def sharpe_ratio(
    weights: np.ndarray,
    annual_returns: pd.Series,
    annual_covariance: pd.DataFrame,
    risk_free_rate: float,
) -> float:
    """Calculate Sharpe ratio."""
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
    """Return negative Sharpe because scipy.optimize.minimize minimizes."""
    return -sharpe_ratio(weights, annual_returns, annual_covariance, risk_free_rate)


def optimize_max_sharpe(
    annual_returns: pd.Series,
    annual_covariance: pd.DataFrame,
    risk_free_rate: float,
) -> np.ndarray:
    """Find the long-only maximum Sharpe ratio portfolio."""
    n_assets = len(annual_returns)
    initial_weights = np.repeat(1 / n_assets, n_assets)
    bounds = tuple((0, 1) for _ in range(n_assets))
    constraints = ({"type": "eq", "fun": lambda weights: np.sum(weights) - 1},)

    result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(annual_returns, annual_covariance, risk_free_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
    )

    if not result.success:
        raise RuntimeError(f"Maximum Sharpe optimization failed: {result.message}")

    return result.x


def build_ml_portfolios(
    returns: pd.DataFrame,
    latest_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build historical, ML-predicted, and equal-weight portfolios."""
    historical_annual_returns = returns.mean() * TRADING_DAYS
    annual_covariance = returns.cov() * TRADING_DAYS
    ml_annual_returns = latest_predictions.set_index("Ticker")["Annualized Predicted Return"]

    tickers = [ticker for ticker in returns.columns if ticker in ml_annual_returns.index]
    historical_annual_returns = historical_annual_returns.reindex(tickers)
    ml_annual_returns = ml_annual_returns.reindex(tickers)
    annual_covariance = annual_covariance.loc[tickers, tickers]

    historical_weights = optimize_max_sharpe(
        historical_annual_returns,
        annual_covariance,
        RISK_FREE_RATE,
    )
    ml_weights = optimize_max_sharpe(
        ml_annual_returns,
        annual_covariance,
        RISK_FREE_RATE,
    )
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
                    (historical_return - RISK_FREE_RATE) / annual_risk if annual_risk != 0 else np.nan
                ),
                "Sharpe Using ML Return": (
                    (ml_return - RISK_FREE_RATE) / annual_risk if annual_risk != 0 else np.nan
                ),
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    weight_table = pd.DataFrame(
        {
            "Ticker": tickers,
            "Historical Mean Optimization": historical_weights,
            "ML Prediction Optimization": ml_weights,
            "Equal Weight Portfolio": equal_weights,
            "Historical Annual Return": historical_annual_returns.values,
            "ML Annualized Predicted Return": ml_annual_returns.values,
        }
    ).sort_values("ML Prediction Optimization", ascending=False)

    return comparison, weight_table


# -----------------------------
# 6. Charts
# -----------------------------

def apply_black_chart_theme(ax: plt.Axes) -> None:
    """Apply pure black background and pure white text."""
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


def save_or_show_chart(fig: plt.Figure) -> None:
    """Show chart only if backend supports it."""
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)


def plot_predicted_vs_actual(predictions: pd.DataFrame, best_model_name: str) -> None:
    """Plot out-of-fold predicted returns against actual returns."""
    plot_data = predictions[predictions["Model"] == best_model_name].copy()
    colors = [
        PURE_GREEN if actual >= 0 else PURE_RED
        for actual in plot_data["Actual 21D Return"]
    ]

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
    apply_black_chart_theme(ax)
    fig.tight_layout()
    plt.savefig(PREDICTED_VS_ACTUAL_CHART_FILE, dpi=300, facecolor=PURE_BLACK)
    save_or_show_chart(fig)


def plot_allocation_comparison(weight_table: pd.DataFrame, top_n: int = 20) -> None:
    """Plot allocation comparison across historical, ML, and equal weight portfolios."""
    weight_columns = [
        "Historical Mean Optimization",
        "ML Prediction Optimization",
        "Equal Weight Portfolio",
    ]

    plot_data = weight_table.copy()
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
        color=NEUTRAL_GRAY,
        label="Equal Weight",
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_data["Ticker"])
    ax.set_title("Portfolio Allocation Comparison")
    ax.set_xlabel("Allocation (%)")
    ax.set_ylabel("Ticker")
    apply_black_chart_theme(ax)

    legend = ax.legend(facecolor=PURE_BLACK, edgecolor=PURE_WHITE)
    for text in legend.get_texts():
        text.set_color(PURE_WHITE)

    fig.tight_layout()
    plt.savefig(ALLOCATION_COMPARISON_CHART_FILE, dpi=300, facecolor=PURE_BLACK)
    save_or_show_chart(fig)


# -----------------------------
# 7. Report text
# -----------------------------

def build_report_section(best_model_name: str, average_performance: pd.DataFrame) -> str:
    """Create a report-ready explanation section."""
    best_row = average_performance[average_performance["Model"] == best_model_name].iloc[0]

    return f"""# Machine Learning Based Return Prediction

This section adds a basic machine learning layer to the portfolio optimization project. The goal is educational: predict each selected NSE stock's next 21 trading day return and compare an ML-predicted-return portfolio with a historical mean-return optimized portfolio and an equal weight portfolio.

The input data is adjusted close price data downloaded from Yahoo Finance. Daily returns are calculated as the percentage change from one trading day to the next. The prediction target is the return from today's close to the close 21 trading days later.

Feature guide:
- 5-day return: the stock's return over roughly one trading week. It captures very short-term momentum.
- 10-day return: the stock's return over roughly two trading weeks. It smooths short-term movement slightly more than the 5-day return.
- 20-day return: the stock's return over roughly one trading month. It captures monthly momentum.
- 50-day moving average ratio: today's price divided by the 50-day average price, minus 1. A positive value means price is above its medium-term trend.
- 100-day moving average ratio: today's price divided by the 100-day average price, minus 1. This gives a slower trend signal.
- 20-day volatility: the standard deviation of daily returns over the last 20 trading days. Higher volatility means recent price movement has been less stable.
- RSI 14D: a momentum indicator comparing recent gains with recent losses. It can describe whether recent price action has been strong or weak, but it is not reliable as a standalone trading signal.

Random train-test splitting is wrong for time series because it allows the model to train on future market regimes and test on earlier dates. That creates look-ahead bias. This project uses time-series splits, where each test period happens after its training period, which better matches how a real forecasting workflow would operate.

Two simple models are trained: Linear Regression and Random Forest Regressor. Linear Regression checks whether the features have a simple linear relationship with future returns. Random Forest can capture nonlinear relationships, but it can also overfit if it learns patterns that only existed in the past.

Average validation results from time-series splits:
- Best model by MAE: {best_model_name}
- MAE: {best_row["MAE"]:.4f}
- RMSE: {best_row["RMSE"]:.4f}
- Directional accuracy: {best_row["Directional Accuracy"] * 100:.2f}%

Overfitting risk is high in financial machine learning. A model may look good on historical data because it memorized noise, one market cycle, or a temporary relationship. Simpler models, time-series validation, and conservative interpretation help reduce this risk, but they do not remove it.

ML predictions in finance are noisy because stock prices react to new information, macro events, earnings, liquidity, sentiment, regulations, and random market behavior. Historical price features contain only a small part of that information. For that reason, the ML portfolio should be treated as an experiment, not as a guarantee of profit.
"""


# -----------------------------
# 8. Main workflow
# -----------------------------

def main() -> None:
    """Run the full ML return prediction workflow."""
    create_output_folders()

    raw_prices = download_adjusted_close(SELECTED_NSE_TICKERS)
    prices, warnings = clean_prices(raw_prices)

    for warning in warnings:
        print(f"Warning: {warning}")

    if prices.empty:
        print("Live yfinance download was unavailable. Trying saved project price data.")
        prices = load_fallback_prices(SELECTED_NSE_TICKERS)

    if prices.empty:
        raise ValueError("No usable adjusted close price data was available.")

    returns = calculate_daily_returns(prices)
    dataset = build_ml_dataset(prices)

    if len(dataset) < MIN_DATASET_ROWS:
        raise ValueError("Not enough rows for machine learning. Use a longer price history.")

    performance, out_of_fold_predictions, best_model_name = evaluate_models_with_time_series_split(dataset)
    latest_predictions = train_best_model_and_predict_latest(dataset, prices, best_model_name)

    if latest_predictions.empty:
        raise ValueError("No latest feature rows were available for prediction.")

    portfolio_comparison, portfolio_weights = build_ml_portfolios(returns, latest_predictions)

    average_performance = (
        performance.groupby("Model", as_index=False)[["MAE", "RMSE", "Directional Accuracy"]]
        .mean()
        .sort_values(["MAE", "RMSE"], ascending=True)
    )

    dataset.to_csv(FEATURE_DATASET_FILE, index=False)
    performance.to_csv(MODEL_PERFORMANCE_FILE, index=False)
    out_of_fold_predictions.to_csv(MODEL_PREDICTIONS_FILE, index=False)
    latest_predictions.to_csv(LATEST_PREDICTIONS_FILE, index=False)
    portfolio_comparison.to_csv(PORTFOLIO_COMPARISON_FILE, index=False)
    portfolio_weights.to_csv(PORTFOLIO_WEIGHTS_FILE, index=False)

    plot_predicted_vs_actual(out_of_fold_predictions, best_model_name)
    plot_allocation_comparison(portfolio_weights)

    report_text = build_report_section(best_model_name, average_performance)
    REPORT_FILE.write_text(report_text, encoding="utf-8")

    display_performance = average_performance.copy()
    display_performance["MAE"] = display_performance["MAE"].map(lambda value: f"{value:.4f}")
    display_performance["RMSE"] = display_performance["RMSE"].map(lambda value: f"{value:.4f}")
    display_performance["Directional Accuracy"] = display_performance["Directional Accuracy"].map(lambda value: f"{value * 100:.2f}%")

    display_portfolio = portfolio_comparison.copy()
    for column in [
        "Historical Expected Annual Return",
        "ML Expected Annual Return",
        "Annual Risk",
    ]:
        display_portfolio[column] = display_portfolio[column].map(lambda value: f"{value * 100:.2f}%")
    for column in ["Sharpe Using Historical Return", "Sharpe Using ML Return"]:
        display_portfolio[column] = display_portfolio[column].map(lambda value: f"{value:.4f}")

    print("Machine learning return prediction completed.")
    print("\nAverage model performance:")
    print(display_performance.to_string(index=False))
    print("\nLatest predicted 21-day returns:")
    print(latest_predictions[["Ticker", "Predicted 21D Return", "Annualized Predicted Return"]].head(10).to_string(index=False))
    print("\nPortfolio comparison:")
    print(display_portfolio.to_string(index=False))
    print(f"\nFeature dataset saved to: {FEATURE_DATASET_FILE}")
    print(f"Model performance saved to: {MODEL_PERFORMANCE_FILE}")
    print(f"Out-of-fold predictions saved to: {MODEL_PREDICTIONS_FILE}")
    print(f"Latest predictions saved to: {LATEST_PREDICTIONS_FILE}")
    print(f"Portfolio comparison saved to: {PORTFOLIO_COMPARISON_FILE}")
    print(f"Portfolio weights saved to: {PORTFOLIO_WEIGHTS_FILE}")
    print(f"Predicted vs actual chart saved to: {PREDICTED_VS_ACTUAL_CHART_FILE}")
    print(f"Allocation chart saved to: {ALLOCATION_COMPARISON_CHART_FILE}")
    print(f"Report section saved to: {REPORT_FILE}")
    print("\n" + report_text)


if __name__ == "__main__":
    main()
