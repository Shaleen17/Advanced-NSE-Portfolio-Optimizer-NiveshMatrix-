"""Portfolio optimization and Monte Carlo simulation functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import RISK_FREE_RATE
from src.metrics import portfolio_return, portfolio_risk, sharpe_ratio


class OptimizationError(RuntimeError):
    """Raised when portfolio optimization fails."""


def validate_optimizer_inputs(
    annual_returns: pd.Series,
    annual_covariance: pd.DataFrame,
) -> None:
    """Validate return vector and covariance matrix before optimization."""
    if annual_returns.empty or annual_covariance.empty:
        raise OptimizationError("Optimizer inputs are empty.")
    if annual_covariance.shape[0] != len(annual_returns):
        raise OptimizationError("Covariance matrix size does not match return vector.")
    if annual_returns.isna().any() or annual_covariance.isna().any().any():
        raise OptimizationError("Optimizer inputs contain missing values.")


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Normalize a positive vector so weights sum to one."""
    total = float(np.sum(weights))
    if total <= 0:
        raise OptimizationError("Weight total must be positive.")
    return weights / total


def optimize_max_sharpe(
    annual_returns: pd.Series,
    annual_covariance: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
) -> np.ndarray:
    """Find the long-only portfolio with maximum Sharpe Ratio."""
    validate_optimizer_inputs(annual_returns, annual_covariance)
    asset_count = len(annual_returns)
    initial_weights = np.repeat(1.0 / asset_count, asset_count)
    bounds = tuple((0.0, 1.0) for _ in range(asset_count))
    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}

    def objective(weights: np.ndarray) -> float:
        expected_return = portfolio_return(weights, annual_returns)
        expected_risk = portfolio_risk(weights, annual_covariance)
        return -sharpe_ratio(expected_return, expected_risk, risk_free_rate)

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        raise OptimizationError(f"Maximum Sharpe optimization failed: {result.message}")
    return normalize_weights(np.clip(result.x, 0.0, 1.0))


def optimize_min_volatility(annual_covariance: pd.DataFrame) -> np.ndarray:
    """Find the long-only portfolio with minimum volatility."""
    if annual_covariance.empty or annual_covariance.isna().any().any():
        raise OptimizationError("Covariance matrix is empty or invalid.")

    asset_count = annual_covariance.shape[0]
    initial_weights = np.repeat(1.0 / asset_count, asset_count)
    bounds = tuple((0.0, 1.0) for _ in range(asset_count))
    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}

    result = minimize(
        lambda weights: portfolio_risk(weights, annual_covariance),
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        raise OptimizationError(f"Minimum volatility optimization failed: {result.message}")
    return normalize_weights(np.clip(result.x, 0.0, 1.0))


def run_monte_carlo_simulation(
    annual_returns: pd.Series,
    annual_covariance: pd.DataFrame,
    portfolio_count: int = 2000,
    risk_free_rate: float = RISK_FREE_RATE,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate random portfolios and calculate risk-return statistics."""
    validate_optimizer_inputs(annual_returns, annual_covariance)
    if portfolio_count < 100:
        raise OptimizationError("Monte Carlo simulation needs at least 100 portfolios.")

    rng = np.random.default_rng(seed)
    tickers = annual_returns.index.tolist()
    rows: list[dict[str, float]] = []

    for _ in range(portfolio_count):
        weights = normalize_weights(rng.random(len(tickers)))
        expected_return = portfolio_return(weights, annual_returns)
        expected_risk = portfolio_risk(weights, annual_covariance)
        row = {
            "Expected Annual Return": expected_return,
            "Annual Risk": expected_risk,
            "Sharpe Ratio": sharpe_ratio(expected_return, expected_risk, risk_free_rate),
        }
        row.update({ticker: weight for ticker, weight in zip(tickers, weights)})
        rows.append(row)

    return pd.DataFrame(rows)


def best_random_portfolios(results: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return the best Sharpe and lowest volatility rows from simulation results."""
    if results.empty:
        raise OptimizationError("Monte Carlo results are empty.")
    max_sharpe = results.loc[results["Sharpe Ratio"].idxmax()]
    min_volatility = results.loc[results["Annual Risk"].idxmin()]
    return max_sharpe, min_volatility
