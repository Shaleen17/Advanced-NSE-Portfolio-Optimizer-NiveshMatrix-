"""Core portfolio return, risk, and comparison calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import RISK_FREE_RATE, TRADING_DAYS


def calculate_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily percentage returns from price data."""
    if prices.empty:
        raise ValueError("Price data is empty.")
    returns = prices.pct_change().dropna(how="all")
    return returns.dropna(axis=1, how="all")


def calculate_annual_returns(daily_returns: pd.DataFrame) -> pd.Series:
    """Calculate annual expected return from mean daily return."""
    if daily_returns.empty:
        raise ValueError("Daily returns are empty.")
    return daily_returns.mean() * TRADING_DAYS


def calculate_annual_volatility(daily_returns: pd.DataFrame) -> pd.Series:
    """Calculate annual volatility from daily return standard deviation."""
    if daily_returns.empty:
        raise ValueError("Daily returns are empty.")
    return daily_returns.std() * np.sqrt(TRADING_DAYS)


def calculate_annual_covariance(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate annualized covariance matrix."""
    if daily_returns.empty:
        raise ValueError("Daily returns are empty.")
    return daily_returns.cov() * TRADING_DAYS


def calculate_correlation_matrix(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate return correlation matrix."""
    if daily_returns.empty:
        raise ValueError("Daily returns are empty.")
    return daily_returns.corr()


def equal_weight_vector(asset_count: int) -> np.ndarray:
    """Create equal portfolio weights."""
    if asset_count <= 0:
        raise ValueError("Asset count must be positive.")
    return np.repeat(1.0 / asset_count, asset_count)


def random_weight_vector(asset_count: int, seed: int = 42) -> np.ndarray:
    """Create a random long-only weight vector that sums to one."""
    if asset_count <= 0:
        raise ValueError("Asset count must be positive.")
    rng = np.random.default_rng(seed)
    weights = rng.random(asset_count)
    return weights / weights.sum()


def portfolio_return(weights: np.ndarray, annual_returns: pd.Series) -> float:
    """Calculate expected annual portfolio return."""
    return float(np.dot(weights, annual_returns.values))


def portfolio_risk(weights: np.ndarray, annual_covariance: pd.DataFrame) -> float:
    """Calculate annualized portfolio risk."""
    variance = float(np.dot(weights.T, np.dot(annual_covariance.values, weights)))
    return float(np.sqrt(max(variance, 0.0)))


def sharpe_ratio(
    expected_return: float,
    expected_risk: float,
    risk_free_rate: float = RISK_FREE_RATE,
) -> float:
    """Calculate Sharpe Ratio with defensive zero-risk handling."""
    if expected_risk <= 0:
        return 0.0
    return float((expected_return - risk_free_rate) / expected_risk)


def calculate_portfolio_performance(
    weights: np.ndarray,
    annual_returns: pd.Series,
    annual_covariance: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict[str, float]:
    """Return expected return, risk, and Sharpe Ratio for one portfolio."""
    expected_return = portfolio_return(weights, annual_returns)
    expected_risk = portfolio_risk(weights, annual_covariance)
    return {
        "Expected Annual Return": expected_return,
        "Annual Risk": expected_risk,
        "Sharpe Ratio": sharpe_ratio(expected_return, expected_risk, risk_free_rate),
    }


def build_allocation_table(tickers: list[str], weights: np.ndarray) -> pd.DataFrame:
    """Build a readable allocation table."""
    return (
        pd.DataFrame({"Ticker": tickers, "Weight": weights})
        .sort_values("Weight", ascending=False)
        .reset_index(drop=True)
    )


def summarize_assets(prices: pd.DataFrame) -> pd.DataFrame:
    """Create a stock-level return and risk summary."""
    daily_returns = calculate_daily_returns(prices)
    annual_returns = calculate_annual_returns(daily_returns)
    annual_volatility = calculate_annual_volatility(daily_returns)
    total_return = prices.iloc[-1] / prices.iloc[0] - 1

    return (
        pd.DataFrame(
            {
                "Ticker": prices.columns,
                "Total Return": total_return.values,
                "Expected Annual Return": annual_returns.values,
                "Annual Volatility": annual_volatility.values,
            }
        )
        .sort_values("Expected Annual Return", ascending=False)
        .reset_index(drop=True)
    )


def build_strategy_comparison(
    strategy_weights: dict[str, np.ndarray],
    annual_returns: pd.Series,
    annual_covariance: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
) -> pd.DataFrame:
    """Compare multiple strategy weight vectors using common assumptions."""
    rows = []
    for strategy_name, weights in strategy_weights.items():
        rows.append(
            {
                "Strategy": strategy_name,
                **calculate_portfolio_performance(
                    weights, annual_returns, annual_covariance, risk_free_rate
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("Sharpe Ratio", ascending=False).reset_index(drop=True)
