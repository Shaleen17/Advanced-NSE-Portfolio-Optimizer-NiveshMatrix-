# Advanced NSE Portfolio Optimizer - Project Report

## Title

**Advanced NSE Portfolio Optimizer**  
Python and Streamlit based portfolio optimization dashboard for Indian NSE stocks.

## Abstract

This project builds a complete portfolio optimization web app using Python and Streamlit. It uses Indian NSE stock data collected through `yfinance`, works with Adjusted Close prices, calculates returns and risk, simulates random portfolios, optimizes portfolio weights, evaluates expert risk metrics, runs monthly rebalancing backtests, and presents results in a modern fintech dashboard.

The project includes Modern Portfolio Theory, Efficient Frontier visualization, Sharpe Ratio optimization, Minimum Volatility optimization, SciPy constrained optimization, risk analytics, Black-Litterman concepts, factor investing, and machine learning return prediction experiments.

This project is educational. It does not guarantee profit and should not be treated as financial advice.

## Problem Statement

Investors often select stocks based only on returns, tips, or short-term performance. This ignores volatility, correlation, diversification, drawdown risk, transaction costs, and changing market behavior. The problem is to build a system that helps analyze Indian NSE stocks using return, risk, diversification, and optimization methods in a transparent way.

## Objectives

- Collect NSE stock data using `yfinance`.
- Use Adjusted Close prices.
- Calculate daily returns, annual returns, and annual volatility.
- Build covariance and correlation matrices.
- Generate random portfolios.
- Run Monte Carlo simulation.
- Plot Efficient Frontier style charts.
- Find Maximum Sharpe and Minimum Volatility portfolios.
- Use SciPy optimization with long-only constraints.
- Calculate advanced risk metrics.
- Run monthly rebalancing backtest with transaction cost assumption.
- Add Black-Litterman, factor investing, and ML return prediction.
- Build a clean Streamlit dashboard.

## Technology Used

- Python
- Streamlit
- yfinance
- pandas
- NumPy
- SciPy
- Matplotlib
- Seaborn
- scikit-learn

## Dataset

The dataset consists of historical Indian NSE stock prices from Yahoo Finance. The project uses NSE tickers with the `.NS` suffix and uses Adjusted Close prices for return calculation. Adjusted Close is preferred because it reflects corporate actions such as dividends and stock splits better than raw close prices.

## Methodology

1. Data collection from Yahoo Finance using `yfinance`.
2. Data cleaning and missing value handling.
3. Daily return calculation from Adjusted Close prices.
4. Annual return and volatility calculation.
5. Covariance and correlation matrix creation.
6. Random portfolio generation.
7. Monte Carlo simulation.
8. SciPy optimization.
9. Advanced risk analytics.
10. Monthly rebalancing backtest.
11. Black-Litterman, factor investing, and ML experiments.
12. Streamlit dashboard visualization.

## Mathematical Formulas

### Daily Return

```text
r_t = (P_t / P_(t-1)) - 1
```

### Annual Return

```text
Annual Return = Mean Daily Return * 252
```

### Annual Volatility

```text
Annual Volatility = Std(Daily Returns) * sqrt(252)
```

### Portfolio Return

```text
R_p = Sum(w_i * R_i)
```

### Portfolio Risk

```text
Portfolio Risk = sqrt(W^T * Cov * W)
```

### Sharpe Ratio

```text
Sharpe Ratio = (R_p - R_f) / sigma_p
```

### Sortino Ratio

```text
Sortino Ratio = (R_p - R_f) / Downside Deviation
```

### Maximum Drawdown

```text
Drawdown = (Portfolio Value / Previous Peak) - 1
```

### VaR

```text
VaR 95% = 5th percentile of daily returns
```

### CVaR

```text
CVaR 95% = Average returns worse than VaR 95%
```

## System Architecture

The project uses a modular architecture:

- `app.py` handles Streamlit UI.
- `config.py` stores settings, ticker list, dates, and colors.
- `src/data_loader.py` handles data.
- `src/metrics.py` handles core calculations.
- `src/optimizer.py` handles optimization.
- `src/risk.py` handles advanced risk metrics.
- `src/backtest.py` handles monthly rebalancing.
- `src/black_litterman.py` handles Black-Litterman calculations.
- `src/factor_investing.py` handles factor scores.
- `src/ml_models.py` handles ML prediction experiments.
- `src/visualization.py` handles charts.

## Results and Analysis

The dashboard provides strategy comparison between Equal Weight, Random Portfolio, Maximum Sharpe, and Minimum Volatility portfolios. It shows return, risk, Sharpe Ratio, allocation weights, Monte Carlo simulation, Efficient Frontier style charts, cumulative returns, drawdowns, and risk analytics.

The results depend on selected stocks, date range, risk-free rate, transaction cost assumption, and market data availability. The system does not claim that optimized portfolios will perform better in the future.

## Dashboard Sections

- Basic
- Intermediate
- Advanced
- Expert Risk
- Backtesting
- Black-Litterman
- Factor Investing
- Machine Learning
- Report
- Downloads

## Limitations

- Historical data does not guarantee future performance.
- yfinance data may have missing values or delays.
- Transaction costs, taxes, and slippage may differ in real life.
- ML predictions can be unreliable.
- Black-Litterman depends on subjective views.
- Factor investing depends on factor definitions and data quality.
- The project does not execute trades.

## Future Scope

- Broker API integration.
- Better Indian fundamental data.
- Sector constraints.
- Tax-aware optimization.
- ESG, spiritual, and ethical investing filters.
- Live portfolio tracking.
- User login and saved portfolios.
- More advanced ML models.
- PDF export.

## Final Conclusion

The Advanced NSE Portfolio Optimizer successfully demonstrates a complete Python-based workflow for portfolio analysis and optimization. It combines finance theory, data science, optimization, machine learning experiments, and Streamlit dashboard development. It is suitable for college submission, GitHub presentation, and demo purposes.

## Educational Disclaimer

This project is for educational and academic use only. It is not financial advice, investment advice, or a trading recommendation. It does not guarantee profit or future returns.
