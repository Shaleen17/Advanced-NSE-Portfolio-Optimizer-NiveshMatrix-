"""
Black-Litterman model for the NSE portfolio optimization project.

The Black-Litterman model combines:

1. Market equilibrium returns
2. Investor views
3. Confidence in those views
4. Historical covariance

This can be more stable than using historical average returns alone.

Run:

    python scripts/11_black_litterman_model.py

Educational warning:
Investor views are subjective. If the views are wrong, Black-Litterman
results can become worse. This project is educational and not financial advice.
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

DAILY_RETURNS_FILE = Path("data/outputs/daily_returns.csv")
OUTPUT_DATA_DIR = Path("data/outputs")
FIGURE_OUTPUT_DIR = Path("reports/figures")

BLACK_LITTERMAN_RETURNS_FILE = OUTPUT_DATA_DIR / "black_litterman_expected_returns.csv"
BLACK_LITTERMAN_ALLOCATION_FILE = OUTPUT_DATA_DIR / "black_litterman_allocation_comparison.csv"
BLACK_LITTERMAN_RESULT_FILE = OUTPUT_DATA_DIR / "black_litterman_result_comparison.csv"
BLACK_LITTERMAN_CHART_FILE = FIGURE_OUTPUT_DIR / "black_litterman_allocation_comparison.png"

TRADING_DAYS = 252
RISK_FREE_RATE = 0.06
TAU = 0.05
DEFAULT_RISK_AVERSION = 2.5
MIN_VISIBLE_WEIGHT = 0.001
USE_YFINANCE_MARKET_CAPS = False

# Optional custom market weights. Leave empty to use equal weights.
# Example:
# CUSTOM_MARKET_WEIGHTS = {"RELIANCE.NS": 0.12, "TCS.NS": 0.10, "INFY.NS": 0.08}
CUSTOM_MARKET_WEIGHTS = {}

PURE_BLACK = "#000000"
PURE_WHITE = "#FFFFFF"
PURE_GREEN = "#00FF00"
PURE_RED = "#FF0000"

# Easy-to-edit investor views.
# type = "relative": first ticker expected to outperform second ticker by value.
# type = "absolute": ticker expected to have a specific annual return.
INVESTOR_VIEWS = [
    {
        "type": "relative",
        "long": "RELIANCE.NS",
        "short": "TCS.NS",
        "value": 0.03,
        "confidence": 0.60,
        "description": "RELIANCE.NS expected to outperform TCS.NS by 3%",
    },
    {
        "type": "absolute",
        "ticker": "INFY.NS",
        "value": 0.12,
        "confidence": 0.55,
        "description": "INFY.NS expected annual return is 12%",
    },
    {
        "type": "relative",
        "long": "HDFCBANK.NS",
        "short": "ICICIBANK.NS",
        "value": 0.02,
        "confidence": 0.60,
        "description": "HDFCBANK.NS expected to outperform ICICIBANK.NS by 2%",
    },
]


# -----------------------------
# 2. Data loading
# -----------------------------

def create_output_folders() -> None:
    """Create output folders if needed."""
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_daily_returns() -> pd.DataFrame:
    """Load daily stock returns from the earlier project step."""
    if not DAILY_RETURNS_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {DAILY_RETURNS_FILE}\n"
            "Please run 02_intermediate_portfolio_analysis.py first."
        )

    returns = pd.read_csv(DAILY_RETURNS_FILE, index_col=0, parse_dates=True)
    return returns.dropna()


def calculate_annual_inputs(daily_returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Calculate annual historical returns and annual covariance."""
    historical_returns = daily_returns.mean() * TRADING_DAYS
    annual_covariance = daily_returns.cov() * TRADING_DAYS
    return historical_returns, annual_covariance


# -----------------------------
# 3. Market weights and priors
# -----------------------------

def fetch_market_cap(ticker: str) -> float | None:
    """Try to fetch one stock's market capitalization from Yahoo Finance."""
    try:
        ticker_obj = yf.Ticker(ticker)
        fast_info = ticker_obj.fast_info

        market_cap = None
        if hasattr(fast_info, "get"):
            market_cap = fast_info.get("marketCap") or fast_info.get("market_cap")
        if market_cap is None:
            market_cap = getattr(fast_info, "market_cap", None)

        if market_cap is not None and market_cap > 0:
            return float(market_cap)
    except Exception:
        return None

    return None


def get_market_weights(
    tickers: list[str],
    custom_market_weights: dict[str, float] | None = None,
    use_yfinance_market_caps: bool = False,
) -> tuple[pd.Series, str]:
    """
    Use market-cap weights if enough market caps are available.

    If market-cap data cannot be fetched reliably, equal weights are used as a
    clear and safe fallback.
    """
    if custom_market_weights:
        custom_weights = pd.Series(custom_market_weights, dtype=float).reindex(tickers).fillna(0)
        if custom_weights.sum() > 0:
            weights = custom_weights / custom_weights.sum()
            return weights, "Custom market weights used and normalized to 100%."

    if not use_yfinance_market_caps:
        weights = pd.Series(np.repeat(1 / len(tickers), len(tickers)), index=tickers)
        return weights, "Equal weights used because yfinance market-cap lookup is disabled."

    market_caps = {}

    for ticker in tickers:
        market_cap = fetch_market_cap(ticker)
        if market_cap is not None:
            market_caps[ticker] = market_cap

    coverage = len(market_caps) / len(tickers)

    if coverage < 0.70:
        weights = pd.Series(np.repeat(1 / len(tickers), len(tickers)), index=tickers)
        return weights, "Equal weights used because market-cap coverage was below 70%."

    caps = pd.Series(market_caps)
    median_cap = caps.median()
    caps = caps.reindex(tickers).fillna(median_cap)
    weights = caps / caps.sum()

    return weights, f"Market-cap weights used with {coverage:.0%} direct market-cap coverage."


def calculate_risk_aversion(
    daily_returns: pd.DataFrame,
    market_weights: pd.Series,
    risk_free_rate: float,
) -> float:
    """Estimate risk aversion from the market-weighted portfolio."""
    market_daily_returns = daily_returns[market_weights.index] @ market_weights
    market_annual_return = market_daily_returns.mean() * TRADING_DAYS
    market_annual_variance = market_daily_returns.var() * TRADING_DAYS

    if market_annual_variance <= 0:
        return DEFAULT_RISK_AVERSION

    risk_aversion = (market_annual_return - risk_free_rate) / market_annual_variance

    if not np.isfinite(risk_aversion) or risk_aversion <= 0:
        return DEFAULT_RISK_AVERSION

    return float(risk_aversion)


def calculate_implied_equilibrium_returns(
    annual_covariance: pd.DataFrame,
    market_weights: pd.Series,
    risk_aversion: float,
    risk_free_rate: float,
) -> pd.Series:
    """Calculate market-implied equilibrium returns."""
    implied_excess_returns = risk_aversion * (annual_covariance @ market_weights)
    implied_total_returns = risk_free_rate + implied_excess_returns
    implied_total_returns.name = "Implied Equilibrium Return"
    return implied_total_returns


# -----------------------------
# 4. Views and posterior returns
# -----------------------------

def create_views_matrices(
    tickers: list[str],
    views: list[dict],
    annual_covariance: pd.DataFrame,
    tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Create P matrix, Q vector, and Omega matrix from investor views."""
    valid_p_rows = []
    valid_q_values = []
    valid_omega_values = []
    valid_descriptions = []

    ticker_to_index = {ticker: index for index, ticker in enumerate(tickers)}

    for view in views:
        p_row = np.zeros(len(tickers))

        if view["type"] == "relative":
            long_ticker = view["long"]
            short_ticker = view["short"]

            if long_ticker not in ticker_to_index or short_ticker not in ticker_to_index:
                continue

            p_row[ticker_to_index[long_ticker]] = 1
            p_row[ticker_to_index[short_ticker]] = -1

        elif view["type"] == "absolute":
            ticker = view["ticker"]

            if ticker not in ticker_to_index:
                continue

            p_row[ticker_to_index[ticker]] = 1

        else:
            continue

        confidence = float(view.get("confidence", 0.50))
        confidence = min(max(confidence, 0.01), 0.99)

        view_variance = float(p_row @ (tau * annual_covariance.to_numpy()) @ p_row.T)
        omega_value = view_variance * ((1 - confidence) / confidence)

        # Keep Omega positive even for very small numerical values.
        omega_value = max(omega_value, 1e-8)

        valid_p_rows.append(p_row)
        valid_q_values.append(float(view["value"]))
        valid_omega_values.append(omega_value)
        valid_descriptions.append(view["description"])

    if not valid_p_rows:
        raise ValueError("No valid investor views matched the available ticker universe.")

    p_matrix = np.vstack(valid_p_rows)
    q_vector = np.array(valid_q_values)
    omega_matrix = np.diag(valid_omega_values)

    return p_matrix, q_vector, omega_matrix, valid_descriptions


def calculate_black_litterman_returns(
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


# -----------------------------
# 5. Optimization
# -----------------------------

def portfolio_return(weights: np.ndarray, expected_returns: pd.Series) -> float:
    """Calculate expected portfolio return."""
    return float(weights @ expected_returns.to_numpy())


def portfolio_risk(weights: np.ndarray, annual_covariance: pd.DataFrame) -> float:
    """Calculate portfolio volatility."""
    covariance_values = annual_covariance.to_numpy()
    variance = float(weights.T @ covariance_values @ weights)
    return float(np.sqrt(max(variance, 0)))


def negative_sharpe_ratio(
    weights: np.ndarray,
    expected_returns: pd.Series,
    annual_covariance: pd.DataFrame,
    risk_free_rate: float,
) -> float:
    """Minimize negative Sharpe ratio because scipy minimizes functions."""
    risk = portfolio_risk(weights, annual_covariance)
    if risk == 0:
        return 1e6
    return -((portfolio_return(weights, expected_returns) - risk_free_rate) / risk)


def optimize_max_sharpe(
    expected_returns: pd.Series,
    annual_covariance: pd.DataFrame,
    risk_free_rate: float,
) -> np.ndarray:
    """Optimize long-only maximum Sharpe portfolio."""
    n_assets = len(expected_returns)
    initial_weights = np.repeat(1 / n_assets, n_assets)
    bounds = tuple((0, 1) for _ in range(n_assets))
    constraints = ({"type": "eq", "fun": lambda weights: np.sum(weights) - 1},)

    result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(expected_returns, annual_covariance, risk_free_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    return result.x


def calculate_portfolio_metrics(
    name: str,
    weights: np.ndarray,
    expected_returns: pd.Series,
    annual_covariance: pd.DataFrame,
) -> dict[str, float | str]:
    """Calculate return, risk, and Sharpe ratio for a portfolio."""
    expected_return = portfolio_return(weights, expected_returns)
    risk = portfolio_risk(weights, annual_covariance)
    sharpe = (expected_return - RISK_FREE_RATE) / risk if risk != 0 else np.nan
    return {
        "Strategy": name,
        "Expected Return": expected_return,
        "Risk": risk,
        "Sharpe Ratio": sharpe,
    }


def create_allocation_comparison(
    tickers: list[str],
    historical_weights: np.ndarray,
    black_litterman_weights: np.ndarray,
) -> pd.DataFrame:
    """Create clean allocation comparison table."""
    table = pd.DataFrame(
        {
            "Ticker": tickers,
            "Historical Mean Weight": historical_weights,
            "Black-Litterman Weight": black_litterman_weights,
        }
    )

    table["Historical Mean Display %"] = np.where(
        table["Historical Mean Weight"] >= MIN_VISIBLE_WEIGHT,
        table["Historical Mean Weight"] * 100,
        0,
    )
    table["Black-Litterman Display %"] = np.where(
        table["Black-Litterman Weight"] >= MIN_VISIBLE_WEIGHT,
        table["Black-Litterman Weight"] * 100,
        0,
    )

    return table.sort_values("Black-Litterman Weight", ascending=False)


# -----------------------------
# 6. Charts
# -----------------------------

def apply_black_chart_theme(ax: plt.Axes) -> None:
    """Apply pure black background and pure white chart text."""
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
    """Show chart only when backend supports it."""
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)


def plot_allocation_comparison(allocation_table: pd.DataFrame, top_n: int = 20) -> None:
    """Plot historical mean vs Black-Litterman allocation comparison."""
    plot_data = allocation_table.head(top_n).sort_values("Black-Litterman Display %", ascending=True)
    y_positions = np.arange(len(plot_data))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 9))
    ax.barh(
        y_positions - width / 2,
        plot_data["Historical Mean Display %"],
        height=width,
        color=PURE_WHITE,
        label="Historical Mean Optimization",
    )
    ax.barh(
        y_positions + width / 2,
        plot_data["Black-Litterman Display %"],
        height=width,
        color=PURE_GREEN,
        label="Black-Litterman Optimization",
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_data["Ticker"])
    ax.set_title("Historical Mean vs Black-Litterman Allocation")
    ax.set_xlabel("Allocation (%)")
    ax.set_ylabel("Ticker")
    apply_black_chart_theme(ax)

    legend = ax.legend(facecolor=PURE_BLACK, edgecolor=PURE_WHITE)
    for text in legend.get_texts():
        text.set_color(PURE_WHITE)

    plt.tight_layout()
    plt.savefig(BLACK_LITTERMAN_CHART_FILE, dpi=300, facecolor=PURE_BLACK)
    save_or_show_chart(fig)


# -----------------------------
# 7. Main workflow
# -----------------------------

def main() -> None:
    """Run Black-Litterman analysis."""
    create_output_folders()

    daily_returns = load_daily_returns()
    historical_returns, annual_covariance = calculate_annual_inputs(daily_returns)
    tickers = list(historical_returns.index)

    market_weights, market_weight_message = get_market_weights(
        tickers,
        custom_market_weights=CUSTOM_MARKET_WEIGHTS,
        use_yfinance_market_caps=USE_YFINANCE_MARKET_CAPS,
    )
    risk_aversion = calculate_risk_aversion(daily_returns, market_weights, RISK_FREE_RATE)
    implied_returns = calculate_implied_equilibrium_returns(
        annual_covariance,
        market_weights,
        risk_aversion,
        RISK_FREE_RATE,
    )

    p_matrix, q_vector, omega_matrix, valid_views = create_views_matrices(
        tickers,
        INVESTOR_VIEWS,
        annual_covariance,
        TAU,
    )

    black_litterman_returns = calculate_black_litterman_returns(
        implied_returns,
        annual_covariance,
        p_matrix,
        q_vector,
        omega_matrix,
        TAU,
    )

    historical_weights = optimize_max_sharpe(historical_returns, annual_covariance, RISK_FREE_RATE)
    black_litterman_weights = optimize_max_sharpe(black_litterman_returns, annual_covariance, RISK_FREE_RATE)

    allocation = create_allocation_comparison(tickers, historical_weights, black_litterman_weights)
    allocation.to_csv(BLACK_LITTERMAN_ALLOCATION_FILE, index=False)

    expected_return_table = pd.DataFrame(
        {
            "Ticker": tickers,
            "Historical Mean Return": historical_returns.values,
            "Implied Equilibrium Return": implied_returns.values,
            "Black-Litterman Return": black_litterman_returns.values,
            "Market Weight": market_weights.values,
        }
    )
    expected_return_table.to_csv(BLACK_LITTERMAN_RETURNS_FILE, index=False)

    results = pd.DataFrame(
        [
            calculate_portfolio_metrics(
                "Historical Mean Optimization",
                historical_weights,
                historical_returns,
                annual_covariance,
            ),
            calculate_portfolio_metrics(
                "Black-Litterman Optimization",
                black_litterman_weights,
                black_litterman_returns,
                annual_covariance,
            ),
        ]
    )
    results.to_csv(BLACK_LITTERMAN_RESULT_FILE, index=False)

    plot_allocation_comparison(allocation)

    print("Black-Litterman model completed.")
    print(market_weight_message)
    print(f"Risk aversion parameter: {risk_aversion:.4f}")
    print("\nValid investor views used:")
    for view in valid_views:
        print(f"- {view}")

    print("\nResult comparison:")
    display_results = results.copy()
    for column in ["Expected Return", "Risk"]:
        display_results[column] = display_results[column].map(lambda value: f"{value * 100:.2f}%")
    display_results["Sharpe Ratio"] = display_results["Sharpe Ratio"].map(lambda value: f"{value:.4f}")
    print(display_results.to_string(index=False))

    print(f"\nExpected return table saved to: {BLACK_LITTERMAN_RETURNS_FILE}")
    print(f"Allocation comparison saved to: {BLACK_LITTERMAN_ALLOCATION_FILE}")
    print(f"Result comparison saved to: {BLACK_LITTERMAN_RESULT_FILE}")
    print(f"Allocation chart saved to: {BLACK_LITTERMAN_CHART_FILE}")
    print(
        "\nInvestor views are subjective. If the assumptions are wrong, "
        "Black-Litterman results can become worse."
    )


if __name__ == "__main__":
    main()
