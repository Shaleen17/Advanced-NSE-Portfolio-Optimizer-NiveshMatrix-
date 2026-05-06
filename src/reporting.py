"""Reporting helpers, formula explanations, and CSV utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import OUTPUT_DATA_DIR, ensure_project_folders


def save_dataframe(dataframe: pd.DataFrame, file_name: str, folder: Path = OUTPUT_DATA_DIR) -> Path:
    """Save a DataFrame as CSV and return the path."""
    ensure_project_folders()
    output_path = folder / file_name
    dataframe.to_csv(output_path, index=False)
    return output_path


def dataframe_to_csv_bytes(dataframe: pd.DataFrame, include_index: bool = False) -> bytes:
    """Convert DataFrame to CSV bytes for Streamlit downloads."""
    return dataframe.to_csv(index=include_index).encode("utf-8")


def project_file_purpose() -> pd.DataFrame:
    """Explain what each important project file does."""
    rows = [
        ("app.py", "Streamlit dashboard entrypoint and UI composition."),
        ("config.py", "Project settings, NSE tickers, dates, risk-free rate, colors, and paths."),
        ("src/data_loader.py", "Downloads, validates, cleans, caches, and loads stock and benchmark data."),
        ("src/metrics.py", "Calculates returns, volatility, covariance, correlation, Sharpe Ratio, and allocations."),
        ("src/optimizer.py", "Runs SciPy optimization and Monte Carlo portfolio simulation."),
        ("src/risk.py", "Calculates Sortino, drawdown, Calmar, VaR, CVaR, beta, tracking error, and information ratio."),
        ("src/backtest.py", "Runs monthly rebalancing backtest with transaction cost assumption."),
        ("src/black_litterman.py", "Implements an educational Black-Litterman return blend and allocation."),
        ("src/factor_investing.py", "Builds momentum, low-volatility, trend, and factor-weighted portfolio tables."),
        ("src/ml_models.py", "Builds features and trains a Random Forest model for experimental return prediction."),
        ("src/visualization.py", "Creates pure black themed charts for Streamlit."),
        ("src/reporting.py", "Prepares file explanations and downloadable CSV outputs."),
        ("project_report.md", "Final report content for college submission."),
        ("README.md", "GitHub project documentation and run/deploy instructions."),
    ]
    return pd.DataFrame(rows, columns=["File", "Purpose"])


def formula_reference() -> pd.DataFrame:
    """Return a beginner-friendly table of formulas used in the project."""
    rows = [
        ("Daily Return", "r_t = (P_t / P_(t-1)) - 1", "One-day percentage change in Adjusted Close price."),
        ("Annual Return", "Mean daily return * 252", "Expected yearly return estimate from historical daily returns."),
        ("Annual Volatility", "Std(daily returns) * sqrt(252)", "Annualized risk based on daily return variation."),
        ("Portfolio Return", "Sum(w_i * R_i)", "Weighted average expected return of portfolio holdings."),
        ("Portfolio Risk", "sqrt(W^T * Cov * W)", "Portfolio volatility using covariance between stocks."),
        ("Sharpe Ratio", "(R_p - R_f) / sigma_p", "Risk-adjusted return using total volatility."),
        ("Sortino Ratio", "(R_p - R_f) / downside deviation", "Risk-adjusted return using only downside volatility."),
        ("Maximum Drawdown", "(Value / previous peak) - 1", "Worst fall from a prior portfolio peak."),
        ("VaR 95%", "5th percentile of returns", "Historical loss threshold at 95 percent confidence."),
        ("CVaR 95%", "Average returns worse than VaR", "Average tail loss beyond the VaR threshold."),
    ]
    return pd.DataFrame(rows, columns=["Concept", "Formula", "Explanation"])


def methodology_steps() -> list[str]:
    """Return the project methodology flow."""
    return [
        "Data collection from Yahoo Finance using yfinance.",
        "Data cleaning using Adjusted Close prices and missing value handling.",
        "Daily return calculation from price changes.",
        "Risk calculation using volatility, covariance, and correlation.",
        "Random portfolio generation and baseline comparison.",
        "Monte Carlo simulation and Efficient Frontier visualization.",
        "SciPy optimization for Maximum Sharpe and Minimum Volatility portfolios.",
        "Advanced risk analytics, backtesting, Black-Litterman, factor investing, and ML experiments.",
        "Streamlit dashboard visualization and CSV download outputs.",
    ]


def final_conclusion() -> str:
    """Return final project conclusion text."""
    return (
        "The Advanced NSE Portfolio Optimizer demonstrates how Modern Portfolio "
        "Theory and Python can be used to analyze Indian NSE stocks, compare "
        "risk-return trade-offs, and build optimized portfolios. The project "
        "combines data collection, mathematical finance, optimization, risk "
        "analytics, backtesting, and dashboard design. It is suitable for "
        "college submission, GitHub presentation, and educational demo use. "
        "The system does not guarantee profits and should not be treated as "
        "financial advice."
    )


def future_scope_items() -> list[str]:
    """Return future improvements for the project."""
    return [
        "Real-time broker API integration.",
        "More accurate Indian fundamental data.",
        "Sector and concentration constraints.",
        "Tax-aware optimization.",
        "ESG, spiritual, and ethical investing filters.",
        "Live portfolio tracking.",
        "User login and saved portfolios.",
        "More advanced ML and time-series forecasting models.",
        "PDF report export from the dashboard.",
    ]
