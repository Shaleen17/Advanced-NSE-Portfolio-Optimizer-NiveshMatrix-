"""Advanced risk analytics for portfolio strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import RISK_FREE_RATE, TRADING_DAYS
from src.metrics import sharpe_ratio


def align_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None,
) -> tuple[pd.Series, pd.Series | None]:
    """Align portfolio and benchmark returns on common dates."""
    clean_portfolio = portfolio_returns.dropna()
    if benchmark_returns is None or benchmark_returns.empty:
        return clean_portfolio, None
    clean_benchmark = benchmark_returns.dropna()
    common_index = clean_portfolio.index.intersection(clean_benchmark.index)
    if common_index.empty:
        return clean_portfolio, None
    return clean_portfolio.loc[common_index], clean_benchmark.loc[common_index]


def calculate_portfolio_daily_returns(
    daily_returns: pd.DataFrame,
    weights: np.ndarray,
) -> pd.Series:
    """Calculate daily portfolio returns from stock returns and weights."""
    if daily_returns.empty:
        raise ValueError("Daily returns are empty.")
    return daily_returns.dot(weights)


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Convert daily returns into cumulative returns."""
    return (1 + returns.dropna()).cumprod() - 1


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Calculate drawdown from cumulative wealth."""
    wealth = (1 + returns.dropna()).cumprod()
    previous_peak = wealth.cummax()
    return wealth / previous_peak - 1


def maximum_drawdown(returns: pd.Series) -> float:
    """Return the largest historical drawdown."""
    if returns.dropna().empty:
        return 0.0
    return float(drawdown_series(returns).min())


def annualized_return(returns: pd.Series) -> float:
    """Calculate annualized return from daily returns."""
    clean_returns = returns.dropna()
    if clean_returns.empty:
        return 0.0
    years = len(clean_returns) / TRADING_DAYS
    if years <= 0:
        return 0.0
    growth = float((1 + clean_returns).prod())
    return growth ** (1 / years) - 1


def annualized_volatility(returns: pd.Series) -> float:
    """Calculate annualized volatility from daily returns."""
    clean_returns = returns.dropna()
    if clean_returns.empty:
        return 0.0
    return float(clean_returns.std() * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> float:
    """Calculate Sortino Ratio using downside deviation."""
    clean_returns = returns.dropna()
    downside_returns = clean_returns[clean_returns < 0]
    downside_risk = float(downside_returns.std() * np.sqrt(TRADING_DAYS))
    if downside_risk <= 0 or np.isnan(downside_risk):
        return 0.0
    return float((annualized_return(clean_returns) - risk_free_rate) / downside_risk)


def calmar_ratio(returns: pd.Series) -> float:
    """Calculate Calmar Ratio as annualized return divided by absolute drawdown."""
    max_dd = abs(maximum_drawdown(returns))
    if max_dd <= 0:
        return 0.0
    return float(annualized_return(returns) / max_dd)


def value_at_risk(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculate historical Value at Risk."""
    clean_returns = returns.dropna()
    if clean_returns.empty:
        return 0.0
    return float(np.quantile(clean_returns, 1 - confidence_level))


def conditional_value_at_risk(
    returns: pd.Series,
    confidence_level: float = 0.95,
) -> float:
    """Calculate historical Conditional Value at Risk."""
    clean_returns = returns.dropna()
    if clean_returns.empty:
        return 0.0
    var_threshold = value_at_risk(clean_returns, confidence_level)
    tail_returns = clean_returns[clean_returns <= var_threshold]
    if tail_returns.empty:
        return var_threshold
    return float(tail_returns.mean())


def beta_vs_benchmark(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None,
) -> float:
    """Calculate beta against benchmark returns."""
    aligned_portfolio, aligned_benchmark = align_returns(portfolio_returns, benchmark_returns)
    if aligned_benchmark is None or len(aligned_portfolio) < 2:
        return np.nan
    benchmark_variance = float(aligned_benchmark.var())
    if benchmark_variance <= 0:
        return np.nan
    covariance = float(np.cov(aligned_portfolio, aligned_benchmark)[0, 1])
    return covariance / benchmark_variance


def tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None,
) -> float:
    """Calculate annualized tracking error against a benchmark."""
    aligned_portfolio, aligned_benchmark = align_returns(portfolio_returns, benchmark_returns)
    if aligned_benchmark is None:
        return np.nan
    active_returns = aligned_portfolio - aligned_benchmark
    return float(active_returns.std() * np.sqrt(TRADING_DAYS))


def information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None,
) -> float:
    """Calculate information ratio against a benchmark."""
    aligned_portfolio, aligned_benchmark = align_returns(portfolio_returns, benchmark_returns)
    if aligned_benchmark is None:
        return np.nan
    active_return = annualized_return(aligned_portfolio) - annualized_return(aligned_benchmark)
    active_risk = tracking_error(aligned_portfolio, aligned_benchmark)
    if active_risk <= 0 or np.isnan(active_risk):
        return np.nan
    return float(active_return / active_risk)


def build_strategy_return_frame(
    daily_returns: pd.DataFrame,
    strategy_weights: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Build daily return series for each strategy."""
    return pd.DataFrame(
        {
            strategy_name: calculate_portfolio_daily_returns(daily_returns, weights)
            for strategy_name, weights in strategy_weights.items()
        },
        index=daily_returns.index,
    )


def build_risk_table(
    strategy_returns: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = RISK_FREE_RATE,
) -> pd.DataFrame:
    """Create advanced risk metrics for each strategy."""
    rows = []
    for strategy_name in strategy_returns.columns:
        returns = strategy_returns[strategy_name].dropna()
        ann_return = annualized_return(returns)
        ann_risk = annualized_volatility(returns)
        rows.append(
            {
                "Strategy": strategy_name,
                "Annualized Return": ann_return,
                "Annualized Volatility": ann_risk,
                "Sharpe Ratio": sharpe_ratio(ann_return, ann_risk, risk_free_rate),
                "Sortino Ratio": sortino_ratio(returns, risk_free_rate),
                "Calmar Ratio": calmar_ratio(returns),
                "Maximum Drawdown": maximum_drawdown(returns),
                "Daily VaR 95%": value_at_risk(returns, 0.95),
                "Daily CVaR 95%": conditional_value_at_risk(returns, 0.95),
                "Beta vs NIFTY 50": beta_vs_benchmark(returns, benchmark_returns),
                "Tracking Error": tracking_error(returns, benchmark_returns),
                "Information Ratio": information_ratio(returns, benchmark_returns),
            }
        )
    return pd.DataFrame(rows).sort_values("Sharpe Ratio", ascending=False).reset_index(drop=True)
