"""Monthly rebalancing backtest utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DEFAULT_TRANSACTION_COST
from src.metrics import (
    calculate_annual_covariance,
    calculate_annual_returns,
    equal_weight_vector,
)
from src.optimizer import OptimizationError, optimize_max_sharpe
from src.risk import build_risk_table


def monthly_rebalance_dates(daily_returns: pd.DataFrame) -> pd.DatetimeIndex:
    """Return the last available trading day of each month."""
    if daily_returns.empty:
        return pd.DatetimeIndex([])
    return daily_returns.groupby(daily_returns.index.to_period("M")).tail(1).index


def optimize_training_window(training_returns: pd.DataFrame) -> np.ndarray:
    """Optimize max Sharpe weights for a training window with safe fallback."""
    try:
        annual_returns = calculate_annual_returns(training_returns)
        annual_covariance = calculate_annual_covariance(training_returns)
        return optimize_max_sharpe(annual_returns, annual_covariance)
    except (OptimizationError, ValueError):
        return equal_weight_vector(training_returns.shape[1])


def run_monthly_rebalance_backtest(
    daily_returns: pd.DataFrame,
    training_window: int = 252,
    transaction_cost: float = DEFAULT_TRANSACTION_COST,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Backtest monthly max-Sharpe rebalancing with transaction cost.

    Transaction cost is applied on rebalance days as:
    turnover * transaction_cost.
    """
    if daily_returns.empty:
        raise ValueError("Daily returns are empty.")
    if len(daily_returns) <= training_window + 21:
        raise ValueError("Not enough observations for monthly backtesting.")

    rebalance_dates = monthly_rebalance_dates(daily_returns)
    portfolio_returns = pd.Series(index=daily_returns.index, dtype=float)
    previous_weights = equal_weight_vector(daily_returns.shape[1])
    weight_rows: list[dict[str, float]] = []
    turnover_rows: list[dict[str, float]] = []

    valid_dates = [date for date in rebalance_dates if daily_returns.index.get_loc(date) >= training_window]
    for index, rebalance_date in enumerate(valid_dates[:-1]):
        start_location = daily_returns.index.get_loc(rebalance_date)
        next_date = valid_dates[index + 1]
        end_location = daily_returns.index.get_loc(next_date)

        training_returns = daily_returns.iloc[start_location - training_window : start_location]
        new_weights = optimize_training_window(training_returns)
        turnover = float(np.abs(new_weights - previous_weights).sum() / 2)

        holding_period = daily_returns.iloc[start_location + 1 : end_location + 1]
        period_returns = holding_period.dot(new_weights)
        if not period_returns.empty:
            period_returns.iloc[0] -= turnover * transaction_cost
            portfolio_returns.loc[period_returns.index] = period_returns

        row = {"Date": rebalance_date}
        row.update({ticker: weight for ticker, weight in zip(daily_returns.columns, new_weights)})
        weight_rows.append(row)
        turnover_rows.append({"Date": rebalance_date, "Turnover": turnover})
        previous_weights = new_weights

    clean_portfolio_returns = portfolio_returns.dropna()
    weights_table = pd.DataFrame(weight_rows)
    turnover_table = pd.DataFrame(turnover_rows)
    return clean_portfolio_returns, weights_table, turnover_table


def compare_backtest_to_equal_weight(
    daily_returns: pd.DataFrame,
    backtest_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare rebalanced strategy against equal weight using risk metrics."""
    equal_returns = daily_returns.loc[backtest_returns.index].dot(
        equal_weight_vector(daily_returns.shape[1])
    )
    comparison_returns = pd.DataFrame(
        {
            "Monthly Rebalanced Max Sharpe": backtest_returns,
            "Equal Weight": equal_returns,
        }
    ).dropna()
    return comparison_returns, build_risk_table(comparison_returns, benchmark_returns)
