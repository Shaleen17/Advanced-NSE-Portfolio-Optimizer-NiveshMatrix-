"""
Factor Investing Analysis for the NSE portfolio optimization project.

This module builds practical factor proxies from available data:

1. Momentum proxy: 12-month return, with 6-month fallback when needed
2. Low volatility proxy: annualized volatility
3. Value proxy: PE ratio from yfinance when available
4. Quality proxy: return on equity from yfinance when available
5. Size proxy: market capitalization from yfinance when available

Because free Indian fundamental data can be incomplete or inconsistent, missing
fundamental values are handled carefully and given neutral normalized scores.

Run:

    python scripts/12_factor_investing_analysis.py

Educational warning:
Factor investing is based on historical and fundamental proxies. It is not
guaranteed to work in the future and is not financial advice.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

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
]

PERIOD = "5y"
TRADING_DAYS = 252
RISK_FREE_RATE = 0.06
FACTOR_PORTFOLIO_SIZE = 10
MAX_MISSING_PERCENT = 0.40

OUTPUT_DATA_DIR = Path("data/outputs")
FIGURE_OUTPUT_DIR = Path("reports/figures")

FACTOR_SCORE_FILE = OUTPUT_DATA_DIR / "factor_score_table.csv"
FACTOR_COMPARISON_FILE = OUTPUT_DATA_DIR / "factor_portfolio_comparison.csv"
FACTOR_WEIGHTS_FILE = OUTPUT_DATA_DIR / "factor_portfolio_weights.csv"
FACTOR_RANKING_CHART_FILE = FIGURE_OUTPUT_DIR / "factor_score_ranking.png"
FACTOR_PERFORMANCE_CHART_FILE = FIGURE_OUTPUT_DIR / "factor_portfolio_performance_comparison.png"

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


def clean_prices(prices: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Clean missing values and remove unusable tickers."""
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


def calculate_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily stock returns."""
    returns = prices.pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan)
    return returns.dropna()


# -----------------------------
# 3. Factor proxy functions
# -----------------------------

def zscore(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Normalize a factor into z-scores and give missing values neutral score 0."""
    clean = series.replace([np.inf, -np.inf], np.nan).astype(float)

    if not higher_is_better:
        clean = -clean

    mean = clean.mean(skipna=True)
    std = clean.std(skipna=True)

    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)

    return ((clean - mean) / std).fillna(0)


def calculate_momentum(prices: pd.DataFrame) -> pd.Series:
    """Calculate 12-month momentum, with 6-month fallback if needed."""
    if len(prices) > 252:
        momentum = prices.iloc[-1] / prices.iloc[-252] - 1
    elif len(prices) > 126:
        momentum = prices.iloc[-1] / prices.iloc[-126] - 1
    else:
        momentum = prices.iloc[-1] / prices.iloc[0] - 1

    momentum.name = "Momentum"
    return momentum


def calculate_annual_volatility(daily_returns: pd.DataFrame) -> pd.Series:
    """Calculate annualized volatility."""
    volatility = daily_returns.std() * np.sqrt(TRADING_DAYS)
    volatility.name = "Annual Volatility"
    return volatility


def fetch_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """Fetch PE ratio, ROE, and market cap from yfinance where possible."""
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


def build_factor_score_table(prices: pd.DataFrame, fetch_fundamental_data: bool = True) -> pd.DataFrame:
    """Create normalized factor scores and combined ranking table."""
    daily_returns = calculate_daily_returns(prices)
    tickers = list(daily_returns.columns)

    factor_table = pd.DataFrame(index=tickers)
    factor_table["Momentum"] = calculate_momentum(prices).reindex(tickers)
    factor_table["Annual Volatility"] = calculate_annual_volatility(daily_returns).reindex(tickers)

    if fetch_fundamental_data:
        fundamentals = fetch_fundamentals(tickers)
    else:
        fundamentals = pd.DataFrame(index=tickers, columns=["PE Ratio", "Return on Equity", "Market Cap"], dtype=float)

    factor_table = factor_table.join(fundamentals)

    # Higher momentum is better.
    factor_table["Momentum Score"] = zscore(factor_table["Momentum"], higher_is_better=True)

    # Lower volatility is better for the low-volatility factor.
    factor_table["Low Volatility Score"] = zscore(factor_table["Annual Volatility"], higher_is_better=False)

    # Lower PE is treated as better value.
    factor_table["Value Score"] = zscore(factor_table["PE Ratio"], higher_is_better=False)

    # Higher ROE is treated as better quality.
    factor_table["Quality Score"] = zscore(factor_table["Return on Equity"], higher_is_better=True)

    # For a traditional size factor, smaller market capitalization gets a higher score.
    factor_table["Log Market Cap"] = np.log(factor_table["Market Cap"].replace(0, np.nan))
    factor_table["Size Score"] = zscore(factor_table["Log Market Cap"], higher_is_better=False)

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


def create_factor_portfolio_weights(factor_scores: pd.DataFrame, portfolio_size: int) -> pd.Series:
    """Create equal-weight factor portfolio using top-ranked stocks."""
    selected = factor_scores.head(min(portfolio_size, len(factor_scores))).index
    weights = pd.Series(0.0, index=factor_scores.index)
    weights.loc[selected] = 1 / len(selected)
    return weights


# -----------------------------
# 4. Portfolio comparison
# -----------------------------

def portfolio_return(weights: np.ndarray, annual_returns: pd.Series) -> float:
    """Calculate expected portfolio return."""
    return float(weights @ annual_returns.to_numpy())


def portfolio_risk(weights: np.ndarray, annual_covariance: pd.DataFrame) -> float:
    """Calculate portfolio volatility."""
    covariance_values = annual_covariance.to_numpy()
    variance = float(weights.T @ covariance_values @ weights)
    return float(np.sqrt(max(variance, 0)))


def negative_sharpe_ratio(weights: np.ndarray, annual_returns: pd.Series, annual_covariance: pd.DataFrame) -> float:
    """Minimize negative Sharpe ratio."""
    risk = portfolio_risk(weights, annual_covariance)
    if risk == 0:
        return 1e6
    return -((portfolio_return(weights, annual_returns) - RISK_FREE_RATE) / risk)


def optimize_max_sharpe(annual_returns: pd.Series, annual_covariance: pd.DataFrame) -> np.ndarray:
    """Optimize maximum Sharpe portfolio."""
    n_assets = len(annual_returns)
    initial_weights = np.repeat(1 / n_assets, n_assets)
    bounds = tuple((0, 1) for _ in range(n_assets))
    constraints = ({"type": "eq", "fun": lambda weights: np.sum(weights) - 1},)

    result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(annual_returns, annual_covariance),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        return initial_weights

    return result.x


def optimize_min_volatility(annual_covariance: pd.DataFrame) -> np.ndarray:
    """Optimize minimum volatility portfolio."""
    n_assets = len(annual_covariance)
    initial_weights = np.repeat(1 / n_assets, n_assets)
    bounds = tuple((0, 1) for _ in range(n_assets))
    constraints = ({"type": "eq", "fun": lambda weights: np.sum(weights) - 1},)

    result = minimize(
        portfolio_risk,
        initial_weights,
        args=(annual_covariance,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        return initial_weights

    return result.x


def calculate_strategy_metrics(
    strategy: str,
    weights: pd.Series,
    daily_returns: pd.DataFrame,
    annual_returns: pd.Series,
    annual_covariance: pd.DataFrame,
) -> dict[str, float | str]:
    """Calculate performance-style metrics for one strategy."""
    aligned_weights = weights.reindex(daily_returns.columns).fillna(0)
    portfolio_daily_returns = daily_returns @ aligned_weights
    total_return = (1 + portfolio_daily_returns).prod() - 1
    annual_return = portfolio_daily_returns.mean() * TRADING_DAYS
    annual_volatility = portfolio_daily_returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (annual_return - RISK_FREE_RATE) / annual_volatility if annual_volatility != 0 else np.nan

    return {
        "Strategy": strategy,
        "Total Return": total_return,
        "Annual Return": annual_return,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": sharpe,
    }


def build_strategy_comparison(prices: pd.DataFrame, factor_weights: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare factor portfolio with common optimization portfolios."""
    daily_returns = calculate_daily_returns(prices)
    annual_returns = daily_returns.mean() * TRADING_DAYS
    annual_covariance = daily_returns.cov() * TRADING_DAYS
    tickers = list(daily_returns.columns)

    equal_weights = pd.Series(np.repeat(1 / len(tickers), len(tickers)), index=tickers)
    max_sharpe_weights = pd.Series(optimize_max_sharpe(annual_returns, annual_covariance), index=tickers)
    min_volatility_weights = pd.Series(optimize_min_volatility(annual_covariance), index=tickers)

    strategies = {
        "Factor Portfolio": factor_weights.reindex(tickers).fillna(0),
        "Equal Weight Portfolio": equal_weights,
        "Maximum Sharpe Portfolio": max_sharpe_weights,
        "Minimum Volatility Portfolio": min_volatility_weights,
    }

    comparison = pd.DataFrame(
        [
            calculate_strategy_metrics(name, weights, daily_returns, annual_returns, annual_covariance)
            for name, weights in strategies.items()
        ]
    )

    portfolio_returns = pd.DataFrame(
        {
            name: daily_returns @ weights.reindex(tickers).fillna(0)
            for name, weights in strategies.items()
        }
    )

    return comparison, portfolio_returns


# -----------------------------
# 5. Charts
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


def plot_factor_ranking(factor_scores: pd.DataFrame, top_n: int = 20) -> None:
    """Plot factor score ranking."""
    plot_data = factor_scores.head(top_n).sort_values("Combined Factor Score", ascending=True)
    colors = [PURE_GREEN if value >= 0 else PURE_RED for value in plot_data["Combined Factor Score"]]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(plot_data.index, plot_data["Combined Factor Score"], color=colors)
    ax.set_title("Factor Score Ranking")
    ax.set_xlabel("Combined Factor Score")
    ax.set_ylabel("Ticker")
    apply_black_chart_theme(ax)

    plt.tight_layout()
    plt.savefig(FACTOR_RANKING_CHART_FILE, dpi=300, facecolor=PURE_BLACK)
    save_or_show_chart(fig)


def plot_performance_comparison(portfolio_returns: pd.DataFrame) -> None:
    """Plot cumulative performance comparison."""
    cumulative = (1 + portfolio_returns).cumprod() - 1

    fig, ax = plt.subplots(figsize=(13, 7))
    for column in cumulative.columns:
        ax.plot(cumulative.index, cumulative[column] * 100, linewidth=2, label=column)

    ax.set_title("Factor Portfolio Performance Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    apply_black_chart_theme(ax)
    legend = ax.legend(facecolor=PURE_BLACK, edgecolor=PURE_WHITE)
    for text in legend.get_texts():
        text.set_color(PURE_WHITE)

    plt.tight_layout()
    plt.savefig(FACTOR_PERFORMANCE_CHART_FILE, dpi=300, facecolor=PURE_BLACK)
    save_or_show_chart(fig)


# -----------------------------
# 6. Main workflow
# -----------------------------

def main() -> None:
    """Run factor investing analysis."""
    create_output_folders()

    raw_prices = download_adjusted_close(NSE_TICKERS)
    prices, warnings = clean_prices(raw_prices)

    for warning in warnings:
        print(f"Warning: {warning}")

    if prices.empty:
        fallback_file = Path("data/processed/nse_adjusted_close_cleaned.csv")
        if fallback_file.exists():
            print("Using saved cleaned price data as fallback.")
            prices = pd.read_csv(fallback_file, index_col=0, parse_dates=True)
            prices = prices[[ticker for ticker in NSE_TICKERS if ticker in prices.columns]]
        else:
            raise ValueError("No usable price data available for factor analysis.")

    factor_scores = build_factor_score_table(prices, fetch_fundamental_data=True)
    factor_weights = create_factor_portfolio_weights(factor_scores, FACTOR_PORTFOLIO_SIZE)
    comparison, portfolio_returns = build_strategy_comparison(prices, factor_weights)

    factor_scores.to_csv(FACTOR_SCORE_FILE)
    pd.DataFrame({"Ticker": factor_weights.index, "Weight": factor_weights.values}).to_csv(FACTOR_WEIGHTS_FILE, index=False)
    comparison.to_csv(FACTOR_COMPARISON_FILE, index=False)

    plot_factor_ranking(factor_scores)
    plot_performance_comparison(portfolio_returns)

    display_comparison = comparison.copy()
    for column in ["Total Return", "Annual Return", "Annual Volatility"]:
        display_comparison[column] = display_comparison[column].map(lambda value: f"{value * 100:.2f}%")
    display_comparison["Sharpe Ratio"] = display_comparison["Sharpe Ratio"].map(lambda value: f"{value:.4f}")

    print("Factor investing analysis completed.")
    print("\nTop factor-ranked stocks:")
    print(factor_scores.head(10)[["Momentum", "Annual Volatility", "PE Ratio", "Return on Equity", "Market Cap", "Combined Factor Score", "Rank"]].to_string())

    print("\nStrategy comparison:")
    print(display_comparison.to_string(index=False))

    print(f"\nFactor score table saved to: {FACTOR_SCORE_FILE}")
    print(f"Factor portfolio weights saved to: {FACTOR_WEIGHTS_FILE}")
    print(f"Factor comparison saved to: {FACTOR_COMPARISON_FILE}")
    print(f"Factor ranking chart saved to: {FACTOR_RANKING_CHART_FILE}")
    print(f"Factor performance chart saved to: {FACTOR_PERFORMANCE_CHART_FILE}")
    print(
        "\nyfinance fundamental data may be incomplete or inconsistent. Missing PE, "
        "ROE, or market-cap values are given neutral normalized scores."
    )


if __name__ == "__main__":
    main()
