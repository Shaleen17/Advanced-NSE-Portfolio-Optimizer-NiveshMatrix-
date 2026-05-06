# Advanced NSE Portfolio Optimizer

**Advanced NSE Portfolio Optimizer** is a Python and Streamlit web app for Indian NSE stock portfolio optimization. It uses historical Adjusted Close data, Modern Portfolio Theory, Monte Carlo simulation, Efficient Frontier visualization, SciPy optimization, backtesting, advanced risk metrics, Black-Litterman concepts, factor investing, and machine learning experiments.

> This project is for education and college demonstration only. It is not financial advice and does not guarantee returns.

## UI Theme

- Full website background: pure black
- Normal text: pure white
- Positive/profit values: pure green
- Negative/loss values: pure red
- Fintech-style cards, tables, charts, and sidebar controls

## Features

### Basic

- Simple explanation of portfolio optimization
- Explanation of why stock data is required
- Explanation of Yahoo Finance and `yfinance`
- NSE stock data loading
- Adjusted Close price usage
- First rows of price data
- Missing value handling
- Stock price trend charts

### Intermediate

- Daily returns
- Annual expected return
- Annual volatility
- Covariance matrix
- Correlation matrix
- Correlation heatmap
- Diversification explanation
- Random portfolio generation
- Portfolio return, risk, and Sharpe Ratio

### Advanced

- Thousands of random portfolios
- Return, risk, and Sharpe Ratio for each simulation
- Efficient Frontier style visualization
- Maximum Sharpe portfolio
- Minimum Volatility portfolio
- SciPy constrained optimization
- Long-only constraints:
  - Total weights = 1
  - No short selling
  - Each stock weight between 0 and 1
- Optimized allocation tables and charts
- Comparison of Equal Weight, Random Portfolio, Maximum Sharpe, and Minimum Volatility portfolios

### Expert Extensions

- Sortino Ratio
- Maximum Drawdown
- Calmar Ratio
- Historical VaR
- Historical CVaR
- Beta vs NIFTY 50
- Tracking Error
- Information Ratio
- Monthly rebalancing backtest
- Transaction cost assumption
- Black-Litterman model
- Factor investing analysis
- Machine learning return prediction
- Downloadable CSV outputs
- College project report content inside the app

## Tech Stack

| Library | Use |
| --- | --- |
| Python | Main programming language |
| Streamlit | Web dashboard |
| yfinance | Yahoo Finance data access |
| pandas | Data cleaning and time-series analysis |
| NumPy | Numerical computation |
| SciPy | Portfolio optimization |
| Matplotlib | Charts |
| Seaborn | Heatmaps |
| scikit-learn | Machine learning experiments |

## NSE Stock Universe

The project uses these NSE tickers:

```text
RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS,
SBIN.NS, AXISBANK.NS, KOTAKBANK.NS, LT.NS, ITC.NS,
HINDUNILVR.NS, BHARTIARTL.NS, ASIANPAINT.NS, MARUTI.NS,
TITAN.NS, SUNPHARMA.NS, CIPLA.NS, DRREDDY.NS,
BAJFINANCE.NS, BAJAJFINSV.NS, HCLTECH.NS, WIPRO.NS,
TECHM.NS, ULTRACEMCO.NS, NESTLEIND.NS, POWERGRID.NS,
NTPC.NS, ONGC.NS, COALINDIA.NS, TATASTEEL.NS,
JSWSTEEL.NS, HINDALCO.NS, ADANIENT.NS, ADANIPORTS.NS,
GRASIM.NS, M&M.NS, EICHERMOT.NS, HEROMOTOCO.NS,
BAJAJ-AUTO.NS, BRITANNIA.NS, DIVISLAB.NS, APOLLOHOSP.NS,
BPCL.NS, IOC.NS, TATAMOTORS.NS, TATACONSUM.NS,
UPL.NS, SHREECEM.NS, INDUSINDBK.NS, SBILIFE.NS
```

## Complete Folder Structure

```text
Advanced_NSE_Portfolio_Optimizer/
|-- app.py                         # Streamlit dashboard
|-- config.py                      # Tickers, colors, risk-free rate, paths
|-- requirements.txt               # Python dependencies
|-- README.md                      # GitHub documentation
|-- project_report.md              # College project report
|-- run_app.bat                    # Windows launcher
|-- .gitignore
|-- .streamlit/
|   `-- config.toml                # Streamlit black theme
|-- assets/
|   |-- Brand_Logo.png
|   `-- screenshots/
|-- data/
|   |-- processed/                 # Cleaned stock data
|   `-- outputs/                   # CSV outputs
|-- reports/
|   `-- figures/                   # Generated charts
|-- src/
|   |-- data_loader.py             # Data download, cache, cleaning
|   |-- metrics.py                 # Returns, risk, Sharpe, allocations
|   |-- optimizer.py               # SciPy optimization and Monte Carlo
|   |-- risk.py                    # Advanced risk analytics
|   |-- backtest.py                # Monthly rebalancing backtest
|   |-- black_litterman.py         # Educational Black-Litterman model
|   |-- factor_investing.py        # Factor score analysis
|   |-- ml_models.py               # ML return prediction helpers
|   |-- visualization.py           # Dark theme charts
|   `-- reporting.py               # Formula and report helpers
|-- docs/                          # Social, deploy, resume notes
`-- legacy/                        # Earlier scripts kept for reference
```

## Installation

```bash
git clone https://github.com/<your-username>/Advanced_NSE_Portfolio_Optimizer.git
cd Advanced_NSE_Portfolio_Optimizer
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## How To Run Locally

```bash
streamlit run app.py
```

If that does not work:

```bash
python -m streamlit run app.py
```

On Windows, you can also double-click:

```text
run_app.bat
```

## Methodology

1. Collect NSE stock data from Yahoo Finance using `yfinance`.
2. Use Adjusted Close prices for return calculation.
3. Clean missing values and align all selected stocks.
4. Calculate daily returns.
5. Calculate annual expected return and annual volatility.
6. Build covariance and correlation matrices.
7. Create a random portfolio.
8. Generate thousands of random portfolios using Monte Carlo simulation.
9. Plot the Efficient Frontier style risk-return map.
10. Optimize portfolios using SciPy.
11. Calculate expert risk metrics.
12. Run monthly rebalancing backtest.
13. Apply Black-Litterman, factor investing, and ML experiments.
14. Visualize results in Streamlit.

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
Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Volatility
```

### Sortino Ratio

```text
Sortino Ratio = (Portfolio Return - Risk-Free Rate) / Downside Deviation
```

### Maximum Drawdown

```text
Drawdown = (Portfolio Value / Previous Peak) - 1
```

### VaR

```text
VaR 95% = 5th percentile of historical daily returns
```

### CVaR

```text
CVaR 95% = Average of returns worse than or equal to VaR 95%
```

## Dashboard Sections

- **Basic:** Explanations, data loading, Adjusted Close prices, missing values, price trends
- **Intermediate:** Returns, volatility, covariance, correlation, random portfolio
- **Advanced:** Monte Carlo simulation, Efficient Frontier, SciPy optimization, allocations
- **Authentication:** MongoDB-backed login and signup before dashboard access
- **Expert Risk:** Sortino, drawdown, Calmar, VaR, CVaR, beta, tracking error, information ratio
- **Backtesting:** Monthly rebalancing with transaction cost assumption
- **Black-Litterman:** Equilibrium return and momentum-view blended allocation
- **Factor Investing:** Momentum, low-volatility, trend, and factor-weighted portfolio
- **Machine Learning:** Random Forest return prediction experiment
- **Report:** Methodology, formulas, file map, conclusion, future scope, disclaimer
- **Downloads:** CSV exports

## Example Outputs

- Cleaned Adjusted Close price table
- Daily return table
- Annual return and volatility table
- Covariance matrix
- Correlation matrix
- Correlation heatmap
- Random portfolio metrics
- Monte Carlo simulation table
- Efficient Frontier chart
- Optimized allocation chart
- Strategy comparison table
- Risk metrics table
- Backtest performance table
- Black-Litterman allocation
- Factor portfolio weights
- ML prediction table

## Authentication Setup

The app uses MongoDB for real login and signup. Keep credentials out of GitHub.

1. Copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml`.
2. Add your MongoDB Atlas connection string in `.streamlit/secrets.toml`.
3. Keep `.streamlit/secrets.toml` private. It is ignored by Git.
4. For Streamlit Cloud, add the same `[mongo]` values in the app secrets settings.

## Deploy On Streamlit Cloud

1. Push the project to GitHub.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io).
3. Sign in with GitHub.
4. Click **Create app**.
5. Select the repository and branch.
6. Set the app file path to:

```text
app.py
```

7. Confirm that `requirements.txt` is in the repository root.
8. Add the MongoDB `[mongo]` values in **Secrets**.
9. Start the deployment.

## Final Conclusion

The Advanced NSE Portfolio Optimizer demonstrates how Modern Portfolio Theory and Python can be used to analyze Indian NSE stocks, compare portfolio risk-return trade-offs, and visualize optimized allocations. It is suitable for college submission, GitHub presentation, and demo use. The project uses historical data honestly and does not claim to guarantee future performance.

## Future Scope

- Real-time broker API integration
- More accurate Indian fundamental data
- Sector constraints
- Tax-aware optimization
- ESG, spiritual, and ethical investing filters
- Live portfolio tracking
- Saved portfolio history and user roles
- Advanced ML models and stress testing
- PDF report generation

## Educational Disclaimer

This project is for educational and academic use only. It is not financial advice, not investment advice, and not a trading recommendation. It does not guarantee profit or future returns. Historical performance may not repeat in the future.

## License

Add a `LICENSE` file before public release. MIT License is a common option for educational open-source projects.
