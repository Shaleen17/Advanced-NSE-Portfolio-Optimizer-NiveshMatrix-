"""
Walk-forward portfolio backtest for the NSE portfolio optimization project.

This module tests a more realistic historical workflow:

1. Download at least 5 years of adjusted close prices.
2. Use a rolling training window of past returns only.
3. Rebalance approximately monthly.
4. Optimize weights at each rebalance date using only past data.
5. Apply those weights to the next holding period.
6. Deduct transaction costs based on portfolio turnover.
7. Compare against an equal weight portfolio and NIFTY 50 benchmark.

This avoids look-ahead bias by never using future data during optimization.

Run:

    python scripts/10_walk_forward_backtest.py

Educational warning:
Backtests are based on historical data and do not guarantee future profits.
This project is for educational purposes only and is not financial advice.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

# Keep Matplotlib and yfinance cache files inside the project folder.
MPL_CONFIG_DIR = Path(".matplotlib")
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR.resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

YFINANCE_CACHE_DIR = Path(".yfinance_cache")
YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
try:
    yf.cache.set_cache_location(str(YFINANCE_CACHE_DIR.resolve()))
except Exception:
    pass


# -----------------------------
# 1. Settings
# -----------------------------

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

BENCHMARK_TICKER = "^NSEI"
PERIOD = "5y"
TRADING_DAYS = 252
TRAINING_WINDOW = 252
REBALANCE_FREQUENCY = 21
RISK_FREE_RATE = 0.06
TRANSACTION_COST_RATE = 0.001
MAX_MISSING_PERCENT = 0.40

OUTPUT_DATA_DIR = Path("data/outputs")
FIGURE_OUTPUT_DIR = Path("reports/figures")

BACKTEST_RETURNS_FILE = OUTPUT_DATA_DIR / "walk_forward_backtest_returns.csv"
BACKTEST_METRICS_FILE = OUTPUT_DATA_DIR / "walk_forward_backtest_metrics.csv"
BACKTEST_WEIGHTS_FILE = OUTPUT_DATA_DIR / "walk_forward_backtest_weights.csv"
BACKTEST_TURNOVER_FILE = OUTPUT_DATA_DIR / "walk_forward_backtest_turnover.csv"

EQUITY_CURVE_FILE = FIGURE_OUTPUT_DIR / "walk_forward_equity_curve.png"
DRAWDOWN_CURVE_FILE = FIGURE_OUTPUT_DIR / "walk_forward_drawdown_curve.png"
WEIGHTS_CHART_FILE = FIGURE_OUTPUT_DIR / "walk_forward_weights_over_time.png"
TURNOVER_CHART_FILE = FIGURE_OUTPUT_DIR / "walk_forward_turnover.png"

PURE_BLACK = "#000000"
PURE_WHITE = "#FFFFFF"
PURE_GREEN = "#00FF00"
PURE_RED = "#FF0000"


# -----------------------------
# 2. Data functions
# -----------------------------

def create_output_folders() -> None:
    """Create output folders if they do not already exist."""
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


def clean_price_data(prices: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Handle missing values and remove unusable stocks."""
    warnings = []
    if prices.empty:
        return prices, ["No price data was downloaded."]

    missing_percent = prices.isna().mean()
    keep_columns = missing_percent[missing_percent <= MAX_MISSING_PERCENT].index.tolist()
    removed_columns = missing_percent[missing_percent > MAX_MISSING_PERCENT].index.tolist()

    if removed_columns:
        warnings.append("Removed tickers with too much missing data: " + ", ".join(removed_columns))

    cleaned = prices[keep_columns].ffill().bfill()
    cleaned = cleaned.dropna(axis=1, how="any")
    cleaned = cleaned.dropna(axis=0, how="any")

    if cleaned.empty:
        warnings.append("No usable stock data remained after cleaning.")

    return cleaned, warnings


def calculate_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily returns from adjusted close prices."""
    returns = prices.pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan)
    return returns.dropna()


def download_benchmark_returns(period: str = PERIOD) -> tuple[pd.Series | None, str]:
    """Download NIFTY 50 benchmark daily returns."""
    benchmark_prices = download_adjusted_close([BENCHMARK_TICKER], period=period)

    if benchmark_prices.empty:
        return None, "NIFTY 50 benchmark data was not available."

    if isinstance(benchmark_prices, pd.DataFrame):
        benchmark_series = benchmark_prices.iloc[:, 0]
    else:
        benchmark_series = benchmark_prices

    benchmark_returns = benchmark_series.pct_change().dropna()
    benchmark_returns.name = "NIFTY 50"

    if benchmark_returns.empty:
        return None, "NIFTY 50 benchmark returns were empty."

    return benchmark_returns, ""


# -----------------------------
# 3. Optimization functions
# -----------------------------

def portfolio_return(weights: np.ndarray, annual_returns: pd.Series) -> float:
    """Calculate expected annual portfolio return."""
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
    """SciPy minimizes functions, so maximize Sharpe by minimizing negative Sharpe."""
    return -sharpe_ratio(weights, annual_returns, annual_covariance, risk_free_rate)


def optimize_max_sharpe(training_returns: pd.DataFrame) -> tuple[np.ndarray, bool, str]:
    """Optimize max Sharpe weights using only training data."""
    n_assets = training_returns.shape[1]
    annual_returns = training_returns.mean() * TRADING_DAYS
    annual_covariance = training_returns.cov() * TRADING_DAYS

    initial_weights = np.repeat(1 / n_assets, n_assets)
    bounds = tuple((0, 1) for _ in range(n_assets))
    constraints = ({"type": "eq", "fun": lambda weights: np.sum(weights) - 1},)

    try:
        result = minimize(
            negative_sharpe_ratio,
            initial_weights,
            args=(annual_returns, annual_covariance, RISK_FREE_RATE),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
        )
    except Exception as error:
        return initial_weights, False, str(error)

    if not result.success:
        return initial_weights, False, result.message

    return result.x, True, ""


# -----------------------------
# 4. Backtest functions
# -----------------------------

def calculate_turnover(new_weights: np.ndarray, previous_weights: np.ndarray) -> float:
    """Calculate portfolio turnover as total absolute change in weights."""
    return float(np.sum(np.abs(new_weights - previous_weights)))


def run_walk_forward_backtest(
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series | None,
    training_window: int = TRAINING_WINDOW,
    rebalance_frequency: int = REBALANCE_FREQUENCY,
    transaction_cost_rate: float = TRANSACTION_COST_RATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Run walk-forward max-Sharpe backtest with monthly rebalancing."""
    if len(stock_returns) <= training_window + rebalance_frequency:
        raise ValueError("Not enough return rows for the selected training window and rebalance frequency.")

    n_assets = stock_returns.shape[1]
    tickers = list(stock_returns.columns)

    optimized_returns = pd.Series(0.0, index=stock_returns.index, name="Optimized Portfolio")
    equal_weight_returns = pd.Series(0.0, index=stock_returns.index, name="Equal Weight Portfolio")
    turnover_records = []
    weight_records = []
    warnings = []

    previous_optimized_weights = np.zeros(n_assets)
    previous_equal_weights = np.zeros(n_assets)
    equal_weights = np.repeat(1 / n_assets, n_assets)

    rebalance_positions = list(range(training_window, len(stock_returns), rebalance_frequency))

    for rebalance_number, start_position in enumerate(rebalance_positions, start=1):
        end_position = min(start_position + rebalance_frequency, len(stock_returns))

        # Training data is strictly before the holding period. This avoids look-ahead bias.
        training_returns = stock_returns.iloc[start_position - training_window:start_position]
        holding_returns = stock_returns.iloc[start_position:end_position]

        optimized_weights, success, message = optimize_max_sharpe(training_returns)
        if not success:
            warnings.append(
                f"Rebalance {rebalance_number} used equal weights because optimization failed: {message}"
            )
            optimized_weights = equal_weights.copy()

        optimized_turnover = calculate_turnover(optimized_weights, previous_optimized_weights)
        equal_turnover = calculate_turnover(equal_weights, previous_equal_weights)

        optimized_period_returns = holding_returns @ optimized_weights
        equal_period_returns = holding_returns @ equal_weights

        # Transaction cost is deducted on the first day of each holding period.
        if not optimized_period_returns.empty:
            optimized_period_returns.iloc[0] -= transaction_cost_rate * optimized_turnover
            equal_period_returns.iloc[0] -= transaction_cost_rate * equal_turnover

        optimized_returns.loc[holding_returns.index] = optimized_period_returns
        equal_weight_returns.loc[holding_returns.index] = equal_period_returns

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
            "Equal Weight Portfolio": equal_weight_returns,
        }
    )

    if benchmark_returns is not None:
        aligned_benchmark = benchmark_returns.reindex(backtest_returns.index).dropna()
        backtest_returns = backtest_returns.loc[aligned_benchmark.index]
        backtest_returns["NIFTY 50 Benchmark"] = aligned_benchmark

    backtest_returns = backtest_returns.dropna()
    turnover_table = pd.DataFrame(turnover_records).set_index("Date")
    weights_table = pd.DataFrame(weight_records).set_index("Date")

    return backtest_returns, weights_table, turnover_table, warnings


# -----------------------------
# 5. Performance metrics
# -----------------------------

def cumulative_return(returns: pd.Series) -> float:
    """Calculate total cumulative return."""
    return float((1 + returns).prod() - 1)


def annualized_return(returns: pd.Series) -> float:
    """Calculate compounded annual return."""
    total_return = cumulative_return(returns)
    years = len(returns) / TRADING_DAYS
    if years <= 0:
        return np.nan
    return float((1 + total_return) ** (1 / years) - 1)


def annualized_volatility(returns: pd.Series) -> float:
    """Calculate annualized volatility."""
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Calculate drawdown over time."""
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    return (wealth / peak) - 1


def maximum_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown."""
    return float(drawdown_series(returns).min())


def sharpe_from_returns(returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> float:
    """Calculate Sharpe ratio from daily returns."""
    annual_return = annualized_return(returns)
    annual_volatility = annualized_volatility(returns)
    if annual_volatility == 0:
        return np.nan
    return float((annual_return - risk_free_rate) / annual_volatility)


def calculate_backtest_metrics(
    backtest_returns: pd.DataFrame,
    turnover_table: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate final performance metrics for all strategies."""
    rows = []
    number_of_rebalances = len(turnover_table)

    for column in backtest_returns.columns:
        if column == "Optimized Portfolio":
            average_turnover = turnover_table["Optimized Turnover"].mean()
        elif column == "Equal Weight Portfolio":
            average_turnover = turnover_table["Equal Weight Turnover"].mean()
        else:
            average_turnover = np.nan

        rows.append(
            {
                "Strategy": column,
                "Total Return": cumulative_return(backtest_returns[column]),
                "Annual Return": annualized_return(backtest_returns[column]),
                "Annual Volatility": annualized_volatility(backtest_returns[column]),
                "Sharpe Ratio": sharpe_from_returns(backtest_returns[column]),
                "Maximum Drawdown": maximum_drawdown(backtest_returns[column]),
                "Average Turnover": average_turnover,
                "Number of Rebalances": number_of_rebalances if column != "NIFTY 50 Benchmark" else 0,
            }
        )

    return pd.DataFrame(rows)


def format_metrics_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Create a readable percentage-formatted metrics table."""
    formatted = metrics.copy()
    percent_columns = [
        "Total Return",
        "Annual Return",
        "Annual Volatility",
        "Maximum Drawdown",
        "Average Turnover",
    ]

    for column in percent_columns:
        formatted[column] = formatted[column].map(lambda value: "N/A" if pd.isna(value) else f"{value * 100:.2f}%")

    formatted["Sharpe Ratio"] = formatted["Sharpe Ratio"].map(lambda value: "N/A" if pd.isna(value) else f"{value:.4f}")

    return formatted


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
    """Show chart only if the backend supports interactive display."""
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)


def plot_equity_curve(backtest_returns: pd.DataFrame) -> None:
    """Plot equity curve for all strategies."""
    equity_curve = (1 + backtest_returns).cumprod()

    fig, ax = plt.subplots(figsize=(13, 7))
    for column in equity_curve.columns:
        ax.plot(equity_curve.index, equity_curve[column], linewidth=2, label=column)

    ax.set_title("Walk-Forward Backtest Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of 1 Rupee")
    apply_black_chart_theme(ax)
    legend = ax.legend(facecolor=PURE_BLACK, edgecolor=PURE_WHITE)
    for text in legend.get_texts():
        text.set_color(PURE_WHITE)

    plt.tight_layout()
    plt.savefig(EQUITY_CURVE_FILE, dpi=300, facecolor=PURE_BLACK)
    save_or_show_chart(fig)


def plot_drawdown_curve(backtest_returns: pd.DataFrame) -> None:
    """Plot drawdown curves for all strategies."""
    drawdowns = backtest_returns.apply(drawdown_series)

    fig, ax = plt.subplots(figsize=(13, 7))
    for column in drawdowns.columns:
        ax.plot(drawdowns.index, drawdowns[column] * 100, linewidth=2, label=column)

    ax.axhline(0, color=PURE_WHITE, linewidth=0.8)
    ax.set_title("Walk-Forward Backtest Drawdown Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    apply_black_chart_theme(ax)
    legend = ax.legend(facecolor=PURE_BLACK, edgecolor=PURE_WHITE)
    for text in legend.get_texts():
        text.set_color(PURE_WHITE)

    plt.tight_layout()
    plt.savefig(DRAWDOWN_CURVE_FILE, dpi=300, facecolor=PURE_BLACK)
    save_or_show_chart(fig)


def plot_weights_over_time(weights_table: pd.DataFrame, top_n: int = 12) -> None:
    """Plot top portfolio weights through time."""
    top_columns = weights_table.mean().sort_values(ascending=False).head(top_n).index
    plot_data = weights_table[top_columns]

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.stackplot(plot_data.index, plot_data.T.values, labels=plot_data.columns)
    ax.set_title(f"Optimized Portfolio Weights Over Time - Top {top_n}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    apply_black_chart_theme(ax)
    legend = ax.legend(facecolor=PURE_BLACK, edgecolor=PURE_WHITE, fontsize=8, loc="upper left")
    for text in legend.get_texts():
        text.set_color(PURE_WHITE)

    plt.tight_layout()
    plt.savefig(WEIGHTS_CHART_FILE, dpi=300, facecolor=PURE_BLACK)
    save_or_show_chart(fig)


def plot_turnover(turnover_table: pd.DataFrame) -> None:
    """Plot turnover at each rebalance date."""
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(turnover_table.index, turnover_table["Optimized Turnover"] * 100, color=PURE_GREEN)
    ax.set_title("Optimized Portfolio Turnover at Rebalance Dates")
    ax.set_xlabel("Rebalance Date")
    ax.set_ylabel("Turnover (%)")
    apply_black_chart_theme(ax)

    plt.tight_layout()
    plt.savefig(TURNOVER_CHART_FILE, dpi=300, facecolor=PURE_BLACK)
    save_or_show_chart(fig)


# -----------------------------
# 7. Main workflow
# -----------------------------

def main() -> None:
    """Run the walk-forward backtest."""
    create_output_folders()

    print("Downloading NSE stock prices...")
    raw_prices = download_adjusted_close(NSE_TICKERS, period=PERIOD)
    prices, warnings = clean_price_data(raw_prices)

    for warning in warnings:
        print(f"Warning: {warning}")

    if prices.empty:
        raise ValueError("No usable stock data available for backtest.")

    stock_returns = calculate_daily_returns(prices)

    print("Downloading NIFTY 50 benchmark...")
    benchmark_returns, benchmark_warning = download_benchmark_returns(PERIOD)
    if benchmark_warning:
        print(f"Benchmark warning: {benchmark_warning}")

    backtest_returns, weights_table, turnover_table, backtest_warnings = run_walk_forward_backtest(
        stock_returns,
        benchmark_returns,
        TRAINING_WINDOW,
        REBALANCE_FREQUENCY,
        TRANSACTION_COST_RATE,
    )

    for warning in backtest_warnings[:5]:
        print(f"Backtest warning: {warning}")
    if len(backtest_warnings) > 5:
        print(f"Backtest warning: {len(backtest_warnings) - 5} additional rebalance warnings omitted.")

    metrics = calculate_backtest_metrics(backtest_returns, turnover_table)
    formatted_metrics = format_metrics_table(metrics)

    backtest_returns.to_csv(BACKTEST_RETURNS_FILE)
    metrics.to_csv(BACKTEST_METRICS_FILE, index=False)
    weights_table.to_csv(BACKTEST_WEIGHTS_FILE)
    turnover_table.to_csv(BACKTEST_TURNOVER_FILE)

    print("\nWalk-forward backtest metrics:")
    print(formatted_metrics.to_string(index=False))

    print(f"\nBacktest returns saved to: {BACKTEST_RETURNS_FILE}")
    print(f"Backtest metrics saved to: {BACKTEST_METRICS_FILE}")
    print(f"Backtest weights saved to: {BACKTEST_WEIGHTS_FILE}")
    print(f"Backtest turnover saved to: {BACKTEST_TURNOVER_FILE}")

    plot_equity_curve(backtest_returns)
    plot_drawdown_curve(backtest_returns)
    plot_weights_over_time(weights_table)
    plot_turnover(turnover_table)

    print(f"\nEquity curve saved to: {EQUITY_CURVE_FILE}")
    print(f"Drawdown curve saved to: {DRAWDOWN_CURVE_FILE}")
    print(f"Weights chart saved to: {WEIGHTS_CHART_FILE}")
    print(f"Turnover chart saved to: {TURNOVER_CHART_FILE}")

    print(
        "\nBacktest results are historical simulations only. They do not guarantee "
        "future profit or future risk behavior."
    )


if __name__ == "__main__":
    main()
