"""A compact Black-Litterman implementation for educational portfolio analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import RISK_FREE_RATE, TRADING_DAYS
from src.metrics import calculate_annual_covariance, calculate_annual_returns, calculate_daily_returns, equal_weight_vector
from src.optimizer import optimize_max_sharpe


def calculate_market_implied_returns(
    annual_covariance: pd.DataFrame,
    market_weights: np.ndarray,
    risk_aversion: float,
) -> pd.Series:
    """Calculate equilibrium returns implied by market weights."""
    implied = risk_aversion * annual_covariance.values.dot(market_weights)
    return pd.Series(implied, index=annual_covariance.index)


def estimate_risk_aversion(daily_returns: pd.DataFrame, risk_free_rate: float = RISK_FREE_RATE) -> float:
    """Estimate a simple market risk aversion coefficient from equal-weight returns."""
    market_proxy = daily_returns.dot(equal_weight_vector(daily_returns.shape[1]))
    annual_return = market_proxy.mean() * TRADING_DAYS
    annual_variance = market_proxy.var() * TRADING_DAYS
    if annual_variance <= 0:
        return 2.5
    return float(max((annual_return - risk_free_rate) / annual_variance, 0.1))


def build_momentum_views(prices: pd.DataFrame, lookback_days: int = 126) -> pd.Series:
    """Build simple return views from recent momentum."""
    if len(prices) <= lookback_days:
        lookback_days = max(21, len(prices) // 3)
    recent_return = prices.iloc[-1] / prices.iloc[-lookback_days] - 1
    annualized_view = (1 + recent_return) ** (TRADING_DAYS / lookback_days) - 1
    return annualized_view.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def black_litterman_posterior_returns(
    prices: pd.DataFrame,
    tau: float = 0.05,
    view_uncertainty: float = 0.35,
) -> tuple[pd.Series, pd.DataFrame]:
    """Blend equilibrium returns with momentum views using Black-Litterman logic."""
    daily_returns = calculate_daily_returns(prices)
    historical_returns = calculate_annual_returns(daily_returns)
    annual_covariance = calculate_annual_covariance(daily_returns)
    market_weights = equal_weight_vector(len(historical_returns))
    risk_aversion = estimate_risk_aversion(daily_returns)
    equilibrium_returns = calculate_market_implied_returns(
        annual_covariance, market_weights, risk_aversion
    )
    views = build_momentum_views(prices).reindex(historical_returns.index).fillna(0.0)

    cov = annual_covariance.values
    tau_cov = tau * cov
    p_matrix = np.eye(len(historical_returns))
    q_vector = views.values
    omega = np.diag(np.maximum(np.diag(p_matrix @ tau_cov @ p_matrix.T), 1e-8))
    omega = omega / max(view_uncertainty, 1e-6)

    tau_cov_inverse = np.linalg.pinv(tau_cov)
    omega_inverse = np.linalg.pinv(omega)
    posterior_covariance = np.linalg.pinv(
        tau_cov_inverse + p_matrix.T @ omega_inverse @ p_matrix
    )
    posterior_mean = posterior_covariance @ (
        tau_cov_inverse @ equilibrium_returns.values
        + p_matrix.T @ omega_inverse @ q_vector
    )
    posterior_returns = pd.Series(posterior_mean, index=historical_returns.index)

    details = pd.DataFrame(
        {
            "Ticker": historical_returns.index,
            "Historical Annual Return": historical_returns.values,
            "Equilibrium Return": equilibrium_returns.values,
            "Momentum View Return": views.values,
            "Black-Litterman Return": posterior_returns.values,
        }
    ).sort_values("Black-Litterman Return", ascending=False)
    return posterior_returns, details.reset_index(drop=True)


def black_litterman_allocation(prices: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Optimize a portfolio using Black-Litterman posterior returns."""
    posterior_returns, details = black_litterman_posterior_returns(prices)
    annual_covariance = calculate_annual_covariance(calculate_daily_returns(prices))
    weights = optimize_max_sharpe(posterior_returns, annual_covariance)
    allocation = pd.DataFrame(
        {"Ticker": posterior_returns.index, "Black-Litterman Weight": weights}
    ).sort_values("Black-Litterman Weight", ascending=False)
    return weights, details.merge(allocation, on="Ticker", how="left")
