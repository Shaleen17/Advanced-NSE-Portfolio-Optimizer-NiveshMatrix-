"""
Advanced Risk Analytics for the NSE portfolio optimization project.

This script calculates professional portfolio risk metrics for:

1. Equal Weight Portfolio
2. One Random Portfolio
3. Maximum Sharpe Ratio Portfolio
4. Minimum Volatility Portfolio

It also tries to download NIFTY 50 benchmark data from Yahoo Finance using
the symbol ^NSEI. If benchmark data is available, beta, tracking error, and
information ratio are calculated. If benchmark data is not available, those
benchmark metrics are reported as missing instead of making false claims.

Run after previous project steps:

    python scripts/02_intermediate_portfolio_analysis.py
    python scripts/03_random_portfolio_calculation.py
    python scripts/06_scipy_max_sharpe_optimization.py
    python scripts/07_scipy_min_volatility_optimization.py
    python scripts/09_advanced_risk_analytics.py

Educational warning:
These risk metrics are based on historical data and do not guarantee future
performance. This project is for educational purposes only.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

# Matplotlib may need a writable cache folder on some Windows systems.
MPL_CONFIG_DIR = Path(".matplotlib")
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR.resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# Keep yfinance cache files inside the project folder to avoid permission issues.
YFINANCE_CACHE_DIR = Path(".yfinance_cache")
YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
try:
    yf.cache.set_cache_location(str(YFINANCE_CACHE_DIR.resolve()))
except Exception:
    pass


# -----------------------------
# 1. Project settings
# -----------------------------

TRADING_DAYS = 252
RISK_FREE_RATE = 0.06
VAR_CONFIDENCE_LEVEL = 0.95
BENCHMARK_TICKER = "^NSEI"

DAILY_RETURNS_FILE = Path("data/outputs/daily_returns.csv")
RANDOM_WEIGHTS_FILE = Path("data/outputs/random_portfolio_weights.csv")
MAX_SHARPE_ALLOCATION_FILE = Path("data/outputs/scipy_max_sharpe_allocation.csv")
MIN_VOL_ALLOCATION_FILE = Path("data/outputs/scipy_min_volatility_allocation.csv")

OUTPUT_DATA_DIR = Path("data/outputs")
FIGURE_OUTPUT_DIR = Path("reports/figures")

RISK_METRICS_FILE = OUTPUT_DATA_DIR / "advanced_risk_metrics.csv"
RISK_METRICS_FORMATTED_FILE = OUTPUT_DATA_DIR / "advanced_risk_metrics_formatted.csv"
PORTFOLIO_RETURNS_FILE = OUTPUT_DATA_DIR / "strategy_portfolio_daily_returns.csv"
BENCHMARK_RETURNS_FILE = OUTPUT_DATA_DIR / "nifty50_benchmark_returns.csv"

CUMULATIVE_RETURN_CHART_FILE = FIGURE_OUTPUT_DIR / "advanced_cumulative_return_comparison.png"
DRAWDOWN_CHART_FILE = FIGURE_OUTPUT_DIR / "advanced_drawdown_comparison.png"

PURE_BLACK = "#000000"
PURE_WHITE = "#FFFFFF"
PURE_GREEN = "#00FF00"
PURE_RED = "#FF0000"


# -----------------------------
# 2. Loading data
# -----------------------------

def create_output_folders() -> None:
    """Create output folders if they do not already exist."""
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def require_file(file_path: Path, setup_hint: str) -> None:
    """Raise a useful error if a required file is missing."""
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}\n{setup_hint}")


def load_daily_returns() -> pd.DataFrame:
    """Load stock daily returns from step 2."""
    require_file(DAILY_RETURNS_FILE, "Please run 02_intermediate_portfolio_analysis.py first.")
    returns = pd.read_csv(DAILY_RETURNS_FILE, index_col=0, parse_dates=True)
    return returns.dropna()


def align_and_normalize_weights(weights: pd.Series, tickers: list[str]) -> pd.Series:
    """Align weights to tickers and normalize them so total weight is 1."""
    aligned = weights.reindex(tickers).fillna(0).astype(float)
    total = aligned.sum()
    if total <= 0:
        raise ValueError("Portfolio weights sum to zero, so risk metrics cannot be calculated.")
    return aligned / total


def load_random_weights(tickers: list[str]) -> pd.Series:
    """Load one random portfolio weight vector from step 3."""
    require_file(RANDOM_WEIGHTS_FILE, "Please run 03_random_portfolio_calculation.py first.")
    table = pd.read_csv(RANDOM_WEIGHTS_FILE)
    weights = table.set_index("Stock Ticker")["Weight"]
    return align_and_normalize_weights(weights, tickers)


def load_optimized_weights(file_path: Path, tickers: list[str], setup_hint: str) -> pd.Series:
    """Load SciPy optimized weights."""
    require_file(file_path, setup_hint)
    table = pd.read_csv(file_path)
    weights = table.set_index("Stock Ticker")["Optimized Weight"]
    return align_and_normalize_weights(weights, tickers)


def build_strategy_weights(stock_returns: pd.DataFrame) -> dict[str, pd.Series]:
    """Create or load all strategy weights."""
    tickers = list(stock_returns.columns)
    n_assets = len(tickers)

    equal_weights = pd.Series(np.repeat(1 / n_assets, n_assets), index=tickers)
    random_weights = load_random_weights(tickers)
    max_sharpe_weights = load_optimized_weights(
        MAX_SHARPE_ALLOCATION_FILE,
        tickers,
        "Please run 06_scipy_max_sharpe_optimization.py first.",
    )
    min_volatility_weights = load_optimized_weights(
        MIN_VOL_ALLOCATION_FILE,
        tickers,
        "Please run 07_scipy_min_volatility_optimization.py first.",
    )

    return {
        "Equal Weight Portfolio": equal_weights,
        "One Random Portfolio": random_weights,
        "Maximum Sharpe Ratio Portfolio": max_sharpe_weights,
        "Minimum Volatility Portfolio": min_volatility_weights,
    }


def download_benchmark_returns(start_date: pd.Timestamp, end_date: pd.Timestamp) -> tuple[pd.Series | None, str]:
    """Download NIFTY 50 benchmark returns from Yahoo Finance if possible."""
    try:
        data = yf.download(
            BENCHMARK_TICKER,
            start=start_date.date().isoformat(),
            end=(end_date + pd.Timedelta(days=1)).date().isoformat(),
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception as error:
        return None, f"Benchmark download failed: {error}"

    if data.empty:
        return None, "Benchmark download returned empty data."

    if isinstance(data.columns, pd.MultiIndex):
        if ("Adj Close", BENCHMARK_TICKER) in data.columns:
            benchmark_prices = data[("Adj Close", BENCHMARK_TICKER)]
        elif ("Close", BENCHMARK_TICKER) in data.columns:
            benchmark_prices = data[("Close", BENCHMARK_TICKER)]
        else:
            return None, "Benchmark data did not contain Adj Close or Close prices."
    else:
        if "Adj Close" in data.columns:
            benchmark_prices = data["Adj Close"]
        elif "Close" in data.columns:
            benchmark_prices = data["Close"]
        else:
            return None, "Benchmark data did not contain Adj Close or Close prices."

    benchmark_returns = benchmark_prices.sort_index().pct_change().dropna()
    benchmark_returns.name = "NIFTY 50"

    if benchmark_returns.empty:
        return None, "Benchmark returns are empty after pct_change()."

    return benchmark_returns, ""


# -----------------------------
# 3. Risk metrics
# -----------------------------

def portfolio_daily_returns(stock_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Calculate portfolio daily returns from stock returns and weights."""
    returns = stock_returns[weights.index] @ weights
    returns.name = "Portfolio Return"
    return returns.dropna()


def cumulative_return_series(returns: pd.Series) -> pd.Series:
    """Calculate cumulative return over time."""
    return (1 + returns).cumprod() - 1


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Calculate drawdown series from daily returns."""
    wealth_index = (1 + returns).cumprod()
    running_peak = wealth_index.cummax()
    return (wealth_index / running_peak) - 1


def annualized_return(returns: pd.Series) -> float:
    """Calculate compounded annual return from daily returns."""
    if returns.empty:
        return np.nan
    total_return = (1 + returns).prod() - 1
    years = len(returns) / TRADING_DAYS
    if years <= 0:
        return np.nan
    return (1 + total_return) ** (1 / years) - 1


def sortino_ratio(returns: pd.Series, risk_free_rate: float) -> float:
    """Calculate Sortino ratio using downside volatility."""
    annual_return = annualized_return(returns)
    daily_target = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    downside_returns = returns[returns < daily_target] - daily_target
    downside_deviation = downside_returns.std() * np.sqrt(TRADING_DAYS)

    if downside_deviation == 0 or np.isnan(downside_deviation):
        return np.nan

    return (annual_return - risk_free_rate) / downside_deviation


def value_at_risk(returns: pd.Series, confidence_level: float) -> float:
    """Calculate historical Value at Risk as a positive loss number."""
    tail_probability = 1 - confidence_level
    return -np.percentile(returns, tail_probability * 100)


def conditional_value_at_risk(returns: pd.Series, confidence_level: float) -> float:
    """Calculate historical Conditional VaR as average loss beyond VaR."""
    tail_probability = 1 - confidence_level
    threshold = np.percentile(returns, tail_probability * 100)
    tail_losses = returns[returns <= threshold]
    if tail_losses.empty:
        return np.nan
    return -tail_losses.mean()


def beta_against_benchmark(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate beta against benchmark returns."""
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if aligned.empty or aligned.iloc[:, 1].var() == 0:
        return np.nan
    return aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / aligned.iloc[:, 1].var()


def tracking_error(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate annualized tracking error versus benchmark."""
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if aligned.empty:
        return np.nan
    active_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return active_returns.std() * np.sqrt(TRADING_DAYS)


def information_ratio(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate annualized information ratio versus benchmark."""
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if aligned.empty:
        return np.nan
    active_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    annual_active_return = active_returns.mean() * TRADING_DAYS
    annual_tracking_error = active_returns.std() * np.sqrt(TRADING_DAYS)
    if annual_tracking_error == 0:
        return np.nan
    return annual_active_return / annual_tracking_error


def calculate_risk_metrics(
    strategy_name: str,
    returns: pd.Series,
    risk_free_rate: float,
    benchmark_returns: pd.Series | None,
) -> dict[str, float | str]:
    """Calculate all advanced risk metrics for one portfolio."""
    cumulative_return = cumulative_return_series(returns).iloc[-1]
    annual_return = annualized_return(returns)
    annual_volatility = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else np.nan
    sortino = sortino_ratio(returns, risk_free_rate)
    max_drawdown = drawdown_series(returns).min()
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.nan
    var_95 = value_at_risk(returns, VAR_CONFIDENCE_LEVEL)
    cvar_95 = conditional_value_at_risk(returns, VAR_CONFIDENCE_LEVEL)

    if benchmark_returns is not None:
        beta = beta_against_benchmark(returns, benchmark_returns)
        te = tracking_error(returns, benchmark_returns)
        ir = information_ratio(returns, benchmark_returns)
    else:
        beta = np.nan
        te = np.nan
        ir = np.nan

    return {
        "Strategy": strategy_name,
        "Cumulative Return": cumulative_return,
        "Annual Return": annual_return,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Maximum Drawdown": max_drawdown,
        "Calmar Ratio": calmar,
        "VaR 95%": var_95,
        "CVaR 95%": cvar_95,
        "Beta vs NIFTY 50": beta,
        "Tracking Error": te,
        "Information Ratio": ir,
    }


def build_risk_metrics_table(
    stock_returns: pd.DataFrame,
    weights_by_strategy: dict[str, pd.Series],
    benchmark_returns: pd.Series | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate portfolio returns and risk metrics for all strategies."""
    portfolio_returns = pd.DataFrame(
        {
            strategy: portfolio_daily_returns(stock_returns, weights)
            for strategy, weights in weights_by_strategy.items()
        }
    ).dropna()

    rows = [
        calculate_risk_metrics(strategy, portfolio_returns[strategy], RISK_FREE_RATE, benchmark_returns)
        for strategy in portfolio_returns.columns
    ]

    metrics = pd.DataFrame(rows)
    return metrics, portfolio_returns


# -----------------------------
# 4. Formatting and best metrics
# -----------------------------

def format_percent(value: float) -> str:
    """Format decimal value as percentage."""
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def format_number(value: float) -> str:
    """Format number with four decimals."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.4f}"


def format_metrics_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Create a report-friendly formatted metrics table."""
    formatted = metrics.copy()
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
        formatted[column] = formatted[column].map(format_percent)
    for column in number_columns:
        formatted[column] = formatted[column].map(format_number)

    return formatted


def print_best_by_metric(metrics: pd.DataFrame) -> None:
    """Print which strategy is best according to each metric."""
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
        "Maximum Drawdown",
        "VaR 95%",
        "CVaR 95%",
        "Tracking Error",
    ]

    print("\nBest strategy by metric")
    print("-----------------------")

    for metric in higher_is_better:
        valid = metrics.dropna(subset=[metric])
        if not valid.empty:
            row = valid.loc[valid[metric].idxmax()]
            print(f"{metric}: {row['Strategy']}")

    for metric in lower_is_better:
        valid = metrics.dropna(subset=[metric])
        if not valid.empty:
            if metric == "Maximum Drawdown":
                row = valid.loc[valid[metric].idxmax()]
            else:
                row = valid.loc[valid[metric].idxmin()]
            print(f"{metric}: {row['Strategy']}")


# -----------------------------
# 5. Charts
# -----------------------------

def apply_black_chart_theme(ax: plt.Axes) -> None:
    """Apply pure black background and pure white text to a chart."""
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
    """Show chart only if backend supports interactive display."""
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)


def plot_cumulative_returns(portfolio_returns: pd.DataFrame, benchmark_returns: pd.Series | None) -> None:
    """Plot cumulative return comparison."""
    cumulative = portfolio_returns.apply(cumulative_return_series)

    fig, ax = plt.subplots(figsize=(13, 7))
    for column in cumulative.columns:
        ax.plot(cumulative.index, cumulative[column] * 100, linewidth=2, label=column)

    if benchmark_returns is not None:
        aligned_benchmark = benchmark_returns.reindex(cumulative.index).dropna()
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
    apply_black_chart_theme(ax)
    legend = ax.legend(facecolor=PURE_BLACK, edgecolor=PURE_WHITE, fontsize=8)
    for text in legend.get_texts():
        text.set_color(PURE_WHITE)

    plt.tight_layout()
    plt.savefig(CUMULATIVE_RETURN_CHART_FILE, dpi=300, facecolor=PURE_BLACK)
    save_or_show_chart(fig)

    print(f"\nCumulative return chart saved to: {CUMULATIVE_RETURN_CHART_FILE}")


def plot_drawdowns(portfolio_returns: pd.DataFrame) -> None:
    """Plot drawdown comparison."""
    drawdowns = portfolio_returns.apply(drawdown_series)

    fig, ax = plt.subplots(figsize=(13, 7))
    for column in drawdowns.columns:
        ax.plot(drawdowns.index, drawdowns[column] * 100, linewidth=2, label=column)

    ax.axhline(0, color=PURE_WHITE, linewidth=0.8)
    ax.set_title("Drawdown Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    apply_black_chart_theme(ax)
    legend = ax.legend(facecolor=PURE_BLACK, edgecolor=PURE_WHITE, fontsize=8)
    for text in legend.get_texts():
        text.set_color(PURE_WHITE)

    plt.tight_layout()
    plt.savefig(DRAWDOWN_CHART_FILE, dpi=300, facecolor=PURE_BLACK)
    save_or_show_chart(fig)

    print(f"Drawdown chart saved to: {DRAWDOWN_CHART_FILE}")


# -----------------------------
# 6. Main workflow
# -----------------------------

def main() -> None:
    """Run advanced risk analytics."""
    create_output_folders()

    stock_returns = load_daily_returns()
    weights_by_strategy = build_strategy_weights(stock_returns)

    benchmark_returns, benchmark_warning = download_benchmark_returns(
        stock_returns.index.min(),
        stock_returns.index.max(),
    )

    if benchmark_warning:
        print(f"\nBenchmark warning: {benchmark_warning}")
        print("Benchmark-specific metrics will be reported as N/A.")
    else:
        benchmark_returns.to_csv(BENCHMARK_RETURNS_FILE)
        print(f"\nBenchmark returns saved to: {BENCHMARK_RETURNS_FILE}")

    metrics, portfolio_returns = build_risk_metrics_table(
        stock_returns,
        weights_by_strategy,
        benchmark_returns,
    )

    metrics.to_csv(RISK_METRICS_FILE, index=False)
    portfolio_returns.to_csv(PORTFOLIO_RETURNS_FILE)

    formatted_metrics = format_metrics_table(metrics)
    formatted_metrics.to_csv(RISK_METRICS_FORMATTED_FILE, index=False)

    print("\nAdvanced risk metrics table:")
    print(formatted_metrics.to_string(index=False))

    print_best_by_metric(metrics)

    print(f"\nAdvanced risk metrics saved to: {RISK_METRICS_FILE}")
    print(f"Formatted risk metrics saved to: {RISK_METRICS_FORMATTED_FILE}")
    print(f"Strategy daily returns saved to: {PORTFOLIO_RETURNS_FILE}")

    plot_cumulative_returns(portfolio_returns, benchmark_returns)
    plot_drawdowns(portfolio_returns)

    print(
        "\nThese metrics are based on historical returns only. They do not guarantee "
        "future portfolio performance."
    )


if __name__ == "__main__":
    main()
