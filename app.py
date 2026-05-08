"""Advanced NSE Portfolio Optimizer Streamlit application."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    BRAND_NAME,
    CHART_COLORS,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_TRANSACTION_COST,
    LOGO_PATH,
    NSE_TICKERS,
    PROJECT_NAME,
    RISK_FREE_RATE,
    ensure_project_folders,
)
from src.backtest import compare_backtest_to_equal_weight, run_monthly_rebalance_backtest
from src.black_litterman import black_litterman_allocation
from src.data_loader import DataLoadError, get_benchmark_returns, get_price_data
from src.factor_investing import build_factor_portfolio_weights, build_factor_score_table
from src.metrics import (
    build_allocation_table,
    build_strategy_comparison,
    calculate_annual_covariance,
    calculate_annual_returns,
    calculate_annual_volatility,
    calculate_correlation_matrix,
    calculate_daily_returns,
    calculate_portfolio_performance,
    equal_weight_vector,
    random_weight_vector,
    summarize_assets,
)
from src.ml_models import build_feature_dataset, predict_latest_returns, train_return_model
from src.optimizer import (
    OptimizationError,
    optimize_max_sharpe,
    optimize_min_volatility,
    run_monte_carlo_simulation,
)
from src.providers import fetch_live_quotes, get_configured_provider_names
from src.reporting import (
    dataframe_to_csv_bytes,
    final_conclusion,
    formula_reference,
    future_scope_items,
    methodology_steps,
    project_file_purpose,
)
from src.risk import build_risk_table, build_strategy_return_frame
from src.visualization import (
    allocation_chart,
    correlation_heatmap,
    cumulative_return_chart,
    drawdown_chart,
    efficient_frontier_chart,
    factor_score_chart,
    ml_prediction_chart,
    price_trend_chart,
    returns_distribution_chart,
    turnover_chart,
)


st.set_page_config(
    page_title=f"{BRAND_NAME} | {PROJECT_NAME}",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    """Inject the pure black fintech dashboard theme."""
    st.markdown(
        f"""
        <style>
        #MainMenu, footer, .stDeployButton {{
            display: none !important;
            visibility: hidden !important;
        }}
        header[data-testid="stHeader"] {{
            background: rgba(0, 0, 0, 0.72) !important;
            backdrop-filter: blur(14px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        }}
        [data-testid="collapsedControl"] {{
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            position: fixed !important;
            top: 12px !important;
            left: 12px !important;
            z-index: 999999 !important;
        }}
        [data-testid="collapsedControl"] button,
        [data-testid="stSidebarCollapseButton"] button,
        button[aria-label="Open sidebar"],
        button[aria-label="Close sidebar"] {{
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 42px !important;
            height: 42px !important;
            min-width: 42px !important;
            min-height: 42px !important;
            color: #000000 !important;
            background: #FFB51C !important;
            border: 1px solid rgba(255, 211, 106, 0.95) !important;
            border-radius: 8px !important;
            box-shadow: 0 12px 28px rgba(255, 181, 28, 0.22) !important;
        }}
        .stApp {{
            background:
                radial-gradient(circle at 80% 0%, rgba(255, 181, 28, 0.08), transparent 22%),
                radial-gradient(circle at 0% 24%, rgba(255, 0, 51, 0.10), transparent 24%),
                {CHART_COLORS["background"]};
            color: {CHART_COLORS["text"]};
        }}
        [data-testid="stSidebar"] {{
            background:
                radial-gradient(circle at 25% 0%, rgba(255, 0, 45, 0.22), transparent 30%),
                linear-gradient(180deg, #000000 0%, #090000 45%, #160006 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.10);
            box-shadow: 18px 0 40px rgba(255, 0, 45, 0.10);
        }}
        [data-testid="stSidebar"] > div:first-child {{
            padding-top: 18px;
        }}
        [data-testid="stSidebar"] img {{
            max-width: 160px;
            margin: 10px auto 18px auto;
            display: block;
        }}
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
            color: #FFFFFF !important;
            letter-spacing: 0;
        }}
        [data-testid="stSidebar"] label {{
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }}
        [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {{
            background: #FF0033 !important;
            color: #FFFFFF !important;
            border-radius: 7px !important;
        }}
        [data-testid="stSidebar"] .stCheckbox [data-testid="stMarkdownContainer"] p {{
            color: #FFFFFF !important;
        }}
        [data-testid="stSidebar"] div[data-baseweb="select"],
        [data-testid="stSidebar"] input {{
            background-color: #141414 !important;
            color: #FFFFFF !important;
            border-color: rgba(255, 255, 255, 0.14) !important;
        }}
        [data-testid="stSidebar"] .stSlider div[role="slider"] {{
            background-color: #FF0033 !important;
            border-color: #FFFFFF !important;
        }}
        [data-testid="stSidebar"] .stSlider div[data-testid="stTickBar"] {{
            background: rgba(255, 255, 255, 0.18) !important;
        }}
        [data-testid="stSidebar"] button {{
            border-radius: 8px !important;
            border: 1px solid rgba(255, 0, 45, 0.65) !important;
            color: #FFFFFF !important;
        }}
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {{
            margin-top: 18px;
            padding-top: 14px;
            border-top: 1px solid rgba(255, 255, 255, 0.10);
        }}
        h1, h2, h3, h4, h5, h6, p, span, div, label {{
            color: {CHART_COLORS["text"]};
        }}
        .block-container {{
            padding-top: 4.25rem;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 1480px;
        }}
        .hero {{
            background:
                linear-gradient(135deg, rgba(255, 181, 28, 0.10) 0%, rgba(255, 0, 51, 0.04) 42%, #070707 100%);
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.34);
        }}
        .hero-title {{
            font-size: 36px;
            font-weight: 800;
            line-height: 1.15;
        }}
        .hero-subtitle {{
            color: {CHART_COLORS["muted"]};
            font-size: 16px;
            line-height: 1.55;
            margin-top: 8px;
            max-width: 960px;
        }}
        .metric-card {{
            background: linear-gradient(180deg, #141414 0%, #0B0B0B 100%);
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 8px;
            padding: 18px;
            min-height: 108px;
            box-shadow: 0 14px 34px rgba(0, 0, 0, 0.24);
        }}
        .metric-label {{
            color: {CHART_COLORS["muted"]};
            font-size: 13px;
            line-height: 1.35;
        }}
        .metric-value {{
            font-size: 25px;
            font-weight: 800;
            margin-top: 8px;
        }}
        .positive {{
            color: {CHART_COLORS["profit"]};
        }}
        .negative {{
            color: {CHART_COLORS["loss"]};
        }}
        .info-card {{
            background: linear-gradient(180deg, #171717 0%, #101010 100%);
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.20);
        }}
        .small-muted {{
            color: {CHART_COLORS["muted"]};
            font-size: 13px;
            line-height: 1.45;
            overflow-wrap: anywhere;
        }}
        div[data-testid="stTabs"] [role="tablist"] {{
            gap: 8px;
            overflow-x: auto;
            padding: 0 0 10px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.10);
            scrollbar-width: thin;
        }}
        div[data-testid="stTabs"] [role="tab"] {{
            min-height: 42px;
            background: #111111;
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 8px;
            padding: 0 14px;
            color: #EDEDED !important;
            flex: 0 0 auto;
        }}
        div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
            background: #FFB51C;
            color: #000000 !important;
            border-color: #FFB51C;
        }}
        div[data-testid="stTabs"] [role="tabpanel"] {{
            padding-top: 18px;
        }}
        .stButton button,
        .stDownloadButton button {{
            border-radius: 8px !important;
            min-height: 42px;
            font-weight: 750 !important;
        }}
        .stSelectbox div[data-baseweb="select"],
        .stMultiSelect div[data-baseweb="select"],
        .stDateInput input,
        .stNumberInput input,
        .stTextInput input {{
            border-radius: 8px !important;
        }}
        div[data-testid="stDataFrame"] {{
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 8px;
            overflow: hidden;
        }}
        @media (max-width: 760px) {{
            .block-container {{
                padding-top: 4.75rem;
                padding-left: 0.9rem;
                padding-right: 0.9rem;
            }}
            [data-testid="collapsedControl"] {{
                top: 10px !important;
                left: 10px !important;
            }}
            [data-testid="collapsedControl"] button,
            [data-testid="stSidebarCollapseButton"] button,
            button[aria-label="Open sidebar"],
            button[aria-label="Close sidebar"] {{
                width: 44px !important;
                height: 44px !important;
                min-width: 44px !important;
                min-height: 44px !important;
            }}
            .hero {{
                padding: 18px;
            }}
            .hero-title {{
                font-size: 28px;
            }}
            .hero-subtitle {{
                font-size: 14px;
            }}
            .metric-card {{
                min-height: 92px;
                padding: 14px;
            }}
            .metric-value {{
                font-size: 22px;
            }}
            div[data-testid="stTabs"] [role="tab"] {{
                min-height: 40px;
                padding: 0 12px;
            }}
        }}
        div[data-testid="stDataFrame"] {{
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 8px;
            overflow: hidden;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, signed_value: float | None = None) -> None:
    """Render a custom metric card with green/red signed value styling."""
    value_class = ""
    if signed_value is not None:
        if signed_value > 0:
            value_class = "positive"
        elif signed_value < 0:
            value_class = "negative"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {value_class}">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(title: str, body: str) -> None:
    """Render an explanatory card."""
    st.markdown(
        f"""
        <div class="info-card">
            <strong>{title}</strong>
            <div class="small-muted">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_percent(value: float) -> str:
    """Format a float as a percentage."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.2%}"


def format_number(value: float) -> str:
    """Format a float with four decimals."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.4f}"


def style_numeric_table(
    dataframe: pd.DataFrame,
    percent_columns: list[str] | None = None,
    number_columns: list[str] | None = None,
):
    """Style numeric tables with green positives and red negatives."""
    percent_columns = percent_columns or []
    number_columns = number_columns or []
    formatters = {column: "{:.2%}" for column in percent_columns if column in dataframe.columns}
    formatters.update({column: "{:.4f}" for column in number_columns if column in dataframe.columns})

    def color_signed(value):
        if isinstance(value, (int, float, np.number)) and not pd.isna(value):
            if value > 0:
                return f"color: {CHART_COLORS['profit']}"
            if value < 0:
                return f"color: {CHART_COLORS['loss']}"
        return f"color: {CHART_COLORS['text']}"

    return dataframe.style.format(formatters).map(color_signed)


@st.cache_data(show_spinner=False, ttl=60)
def load_live_quotes_cached(tickers: tuple[str, ...], refresh_nonce: int) -> pd.DataFrame:
    """Cache multi-provider quote calls to protect free API limits."""
    del refresh_nonce
    return fetch_live_quotes(tickers)


@st.cache_data(show_spinner=False)
def load_prices_cached(
    tickers: tuple[str, ...],
    start_date: date,
    end_date: date,
    use_cache_first: bool,
) -> pd.DataFrame:
    """Cache price loading for faster Streamlit reruns."""
    return get_price_data(list(tickers), start_date, end_date, use_cache_first)


@st.cache_data(show_spinner=False)
def load_benchmark_cached(start_date: date, end_date: date, use_cache_first: bool) -> pd.Series | None:
    """Cache benchmark loading for faster Streamlit reruns."""
    return get_benchmark_returns(start_date, end_date, use_cache_first)


def sidebar_realtime_controls(selected_count: int) -> tuple[bool, int, int]:
    """Collect live data controls without slowing the historical workflow."""
    st.sidebar.markdown("### Real-Time Data")
    live_quotes_enabled = st.sidebar.checkbox("Load live quote snapshot", value=False)
    max_quote_count = min(max(selected_count, 1), 20)
    live_quote_count = st.sidebar.slider(
        "Live quote symbols",
        min_value=1,
        max_value=max_quote_count,
        value=max(1, min(selected_count, 8)),
        step=1,
    )
    st.session_state.setdefault("quote_refresh_nonce", 0)
    if st.sidebar.button("Refresh live quotes now", width="stretch"):
        st.session_state.quote_refresh_nonce += 1
    return live_quotes_enabled, live_quote_count, int(st.session_state.quote_refresh_nonce)


def sidebar_inputs() -> tuple[list[str], date, date, bool, int, int, float, float, int]:
    """Collect sidebar settings."""
    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), width="stretch")
    st.sidebar.markdown(f"## {BRAND_NAME}")
    st.sidebar.markdown(
        f"""
        <div class="info-card">
            <strong>Public Dashboard</strong>
            <div class="small-muted">No account required. All portfolio tools are available immediately.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### Portfolio Controls")

    selected_tickers = st.sidebar.multiselect(
        "NSE stock universe",
        options=NSE_TICKERS,
        default=NSE_TICKERS[:12],
    )
    start_date = st.sidebar.date_input("Start date", value=DEFAULT_START_DATE)
    end_date = st.sidebar.date_input("End date", value=DEFAULT_END_DATE)
    use_cache_first = st.sidebar.checkbox("Use cached data first", value=True)
    portfolio_count = st.sidebar.slider("Monte Carlo portfolios", 500, 10000, 2500, 500)
    random_seed = st.sidebar.number_input("Random seed", min_value=1, value=42, step=1)
    risk_free_rate = st.sidebar.slider("Risk-free rate", 0.0, 0.15, RISK_FREE_RATE, 0.005)
    transaction_cost = st.sidebar.slider(
        "Transaction cost assumption",
        0.0,
        0.01,
        DEFAULT_TRANSACTION_COST,
        0.0005,
        format="%.4f",
    )
    factor_size = st.sidebar.slider("Factor portfolio size", 2, 20, 10, 1)
    return (
        selected_tickers,
        start_date,
        end_date,
        use_cache_first,
        portfolio_count,
        int(random_seed),
        risk_free_rate,
        transaction_cost,
        factor_size,
    )


def render_hero() -> None:
    """Render the main dashboard heading."""
    st.markdown(
        f"""
        <div class="dashboard-marker"></div>
        <div class="hero">
            <div class="hero-title">{PROJECT_NAME}</div>
            <div class="hero-subtitle">
                Python and Streamlit dashboard for Indian NSE portfolio optimization using
                Modern Portfolio Theory, Efficient Frontier analysis, SciPy optimization,
                backtesting, risk analytics, Black-Litterman concepts, factor investing,
                and machine learning experiments.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Run the Streamlit dashboard."""
    ensure_project_folders()
    inject_css()

    (
        selected_tickers,
        start_date,
        end_date,
        use_cache_first,
        portfolio_count,
        random_seed,
        risk_free_rate,
        transaction_cost,
        factor_size,
    ) = sidebar_inputs()
    live_quotes_enabled, live_quote_count, quote_refresh_nonce = sidebar_realtime_controls(
        len(selected_tickers)
    )

    render_hero()

    try:
        prices = load_prices_cached(tuple(selected_tickers), start_date, end_date, use_cache_first)
        daily_returns = calculate_daily_returns(prices)
        annual_returns = calculate_annual_returns(daily_returns)
        annual_volatility = calculate_annual_volatility(daily_returns)
        annual_covariance = calculate_annual_covariance(daily_returns)
        correlation_matrix = calculate_correlation_matrix(daily_returns)
        benchmark_returns = load_benchmark_cached(start_date, end_date, use_cache_first)

        equal_weights = equal_weight_vector(len(prices.columns))
        random_weights = random_weight_vector(len(prices.columns), random_seed)
        max_sharpe_weights = optimize_max_sharpe(annual_returns, annual_covariance, risk_free_rate)
        min_volatility_weights = optimize_min_volatility(annual_covariance)
        random_results = run_monte_carlo_simulation(
            annual_returns,
            annual_covariance,
            portfolio_count=portfolio_count,
            risk_free_rate=risk_free_rate,
            seed=random_seed,
        )
    except (DataLoadError, OptimizationError, ValueError) as error:
        st.error(str(error))
        st.stop()

    strategy_weights = {
        "Equal Weight": equal_weights,
        "Random Portfolio": random_weights,
        "Maximum Sharpe": max_sharpe_weights,
        "Minimum Volatility": min_volatility_weights,
    }
    comparison = build_strategy_comparison(
        strategy_weights, annual_returns, annual_covariance, risk_free_rate
    )
    strategy_returns = build_strategy_return_frame(daily_returns, strategy_weights)
    risk_table = build_risk_table(strategy_returns, benchmark_returns, risk_free_rate)
    best_strategy = comparison.iloc[0]

    top_cards = st.columns(4)
    with top_cards[0]:
        metric_card("Selected Stocks", str(len(prices.columns)))
    with top_cards[1]:
        metric_card("Trading Days", str(len(prices)))
    with top_cards[2]:
        metric_card("Best Strategy", best_strategy["Strategy"], best_strategy["Sharpe Ratio"])
    with top_cards[3]:
        metric_card("Best Sharpe", format_number(best_strategy["Sharpe Ratio"]), best_strategy["Sharpe Ratio"])

    tabs = st.tabs(
        [
            "Basic",
            "Intermediate",
            "Advanced",
            "Expert Risk",
            "Backtesting",
            "Black-Litterman",
            "Factor Investing",
            "Machine Learning",
            "Report",
            "Live Market",
            "Downloads",
        ]
    )

    with tabs[0]:
        st.subheader("Basic: Understanding Portfolio Optimization")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            info_card(
                "What is portfolio optimization?",
                "It is the process of choosing stock weights to balance expected return and risk.",
            )
        with col_b:
            info_card(
                "Why is stock data required?",
                "Historical prices are needed to estimate returns, volatility, covariance, and drawdowns.",
            )
        with col_c:
            info_card(
                "What is yfinance?",
                "yfinance is a Python library that retrieves market data from Yahoo Finance for research use.",
            )

        st.markdown("#### Adjusted Close Price Data")
        st.dataframe(prices.head(), width="stretch")

        missing_table = pd.DataFrame(
            {
                "Ticker": prices.columns,
                "Missing Values After Cleaning": prices.isna().sum().values,
            }
        )
        st.markdown("#### Missing Value Handling")
        st.dataframe(missing_table, width="stretch", hide_index=True)
        st.pyplot(price_trend_chart(prices), width="stretch")

    with tabs[1]:
        st.subheader("Intermediate: Returns, Risk, Covariance, and Diversification")
        info_card(
            "Diversification",
            "Diversification means combining stocks that do not move exactly together. "
            "Correlation and covariance help measure this relationship.",
        )

        st.markdown("#### Daily Returns")
        st.dataframe(daily_returns.head(), width="stretch")

        asset_summary = summarize_assets(prices)
        st.markdown("#### Annual Expected Return and Annual Volatility")
        st.dataframe(
            style_numeric_table(
                asset_summary,
                percent_columns=["Total Return", "Expected Annual Return", "Annual Volatility"],
            ),
            width="stretch",
        )

        st.markdown("#### Covariance Matrix")
        st.dataframe(annual_covariance, width="stretch")
        st.markdown("#### Correlation Matrix")
        st.dataframe(correlation_matrix, width="stretch")
        st.pyplot(correlation_heatmap(correlation_matrix), width="stretch")

        random_performance = calculate_portfolio_performance(
            random_weights, annual_returns, annual_covariance, risk_free_rate
        )
        cards = st.columns(3)
        with cards[0]:
            metric_card("Random Portfolio Return", format_percent(random_performance["Expected Annual Return"]), random_performance["Expected Annual Return"])
        with cards[1]:
            metric_card("Random Portfolio Risk", format_percent(random_performance["Annual Risk"]))
        with cards[2]:
            metric_card("Random Portfolio Sharpe", format_number(random_performance["Sharpe Ratio"]), random_performance["Sharpe Ratio"])
        st.pyplot(returns_distribution_chart(daily_returns), width="stretch")

    with tabs[2]:
        st.subheader("Advanced: Monte Carlo, Efficient Frontier, and SciPy Optimization")
        info_card(
            "Optimization constraints",
            "The optimized portfolios are long-only. Total weights must equal 1, no short selling is allowed, and each stock weight is between 0 and 1.",
        )
        st.pyplot(efficient_frontier_chart(random_results, comparison), width="stretch")

        st.markdown("#### Strategy Comparison")
        st.dataframe(
            style_numeric_table(
                comparison,
                percent_columns=["Expected Annual Return", "Annual Risk"],
                number_columns=["Sharpe Ratio"],
            ),
            width="stretch",
        )

        allocations = {
            "Equal Weight": build_allocation_table(prices.columns.tolist(), equal_weights),
            "Random Portfolio": build_allocation_table(prices.columns.tolist(), random_weights),
            "Maximum Sharpe": build_allocation_table(prices.columns.tolist(), max_sharpe_weights),
            "Minimum Volatility": build_allocation_table(prices.columns.tolist(), min_volatility_weights),
        }
        selected_allocation = st.selectbox("Allocation to visualize", list(allocations.keys()))
        selected_table = allocations[selected_allocation]
        st.dataframe(
            style_numeric_table(selected_table, percent_columns=["Weight"]),
            width="stretch",
        )
        st.pyplot(
            allocation_chart(selected_table, "Weight", f"{selected_allocation} Allocation"),
            width="stretch",
        )

    with tabs[3]:
        st.subheader("Expert: Advanced Risk Metrics")
        st.dataframe(
            style_numeric_table(
                risk_table,
                percent_columns=[
                    "Annualized Return",
                    "Annualized Volatility",
                    "Maximum Drawdown",
                    "Daily VaR 95%",
                    "Daily CVaR 95%",
                    "Tracking Error",
                ],
                number_columns=[
                    "Sharpe Ratio",
                    "Sortino Ratio",
                    "Calmar Ratio",
                    "Beta vs NIFTY 50",
                    "Information Ratio",
                ],
            ),
            width="stretch",
        )
        st.pyplot(cumulative_return_chart(strategy_returns, "Strategy Cumulative Returns"), width="stretch")
        st.pyplot(drawdown_chart(strategy_returns, "Strategy Drawdowns"), width="stretch")

    with tabs[4]:
        st.subheader("Expert: Monthly Rebalancing Backtest")
        try:
            backtest_returns, backtest_weights, turnover_table = run_monthly_rebalance_backtest(
                daily_returns,
                transaction_cost=transaction_cost,
            )
            backtest_frame, backtest_metrics = compare_backtest_to_equal_weight(
                daily_returns,
                backtest_returns,
                benchmark_returns,
            )
            st.dataframe(
                style_numeric_table(
                    backtest_metrics,
                    percent_columns=[
                        "Annualized Return",
                        "Annualized Volatility",
                        "Maximum Drawdown",
                        "Daily VaR 95%",
                        "Daily CVaR 95%",
                        "Tracking Error",
                    ],
                    number_columns=["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"],
                ),
                width="stretch",
            )
            st.pyplot(cumulative_return_chart(backtest_frame, "Backtest Cumulative Returns"), width="stretch")
            st.pyplot(turnover_chart(turnover_table), width="stretch")
            with st.expander("View monthly backtest weights"):
                st.dataframe(backtest_weights, width="stretch")
        except ValueError as error:
            st.warning(str(error))

    with tabs[5]:
        st.subheader("Expert: Black-Litterman Model")
        try:
            bl_weights, bl_table = black_litterman_allocation(prices)
            bl_allocation = pd.DataFrame(
                {"Ticker": prices.columns, "Black-Litterman Weight": bl_weights}
            ).sort_values("Black-Litterman Weight", ascending=False)
            st.dataframe(
                style_numeric_table(
                    bl_table,
                    percent_columns=[
                        "Historical Annual Return",
                        "Equilibrium Return",
                        "Momentum View Return",
                        "Black-Litterman Return",
                        "Black-Litterman Weight",
                    ],
                ),
                width="stretch",
            )
            st.pyplot(
                allocation_chart(bl_allocation, "Black-Litterman Weight", "Black-Litterman Allocation"),
                width="stretch",
            )
        except Exception as error:
            st.warning(f"Black-Litterman model could not be calculated: {error}")

    with tabs[6]:
        st.subheader("Expert: Factor Investing Analysis")
        factor_scores = build_factor_score_table(prices)
        factor_weights = build_factor_portfolio_weights(factor_scores, factor_size)
        st.dataframe(
            style_numeric_table(
                factor_scores,
                percent_columns=["Momentum 6M", "Momentum 12M", "Annual Volatility", "Trend Ratio"],
                number_columns=["Momentum Score", "Low Volatility Score", "Trend Score", "Overall Factor Score"],
            ),
            width="stretch",
        )
        st.pyplot(factor_score_chart(factor_scores), width="stretch")
        st.markdown("#### Factor Portfolio Weights")
        st.dataframe(
            style_numeric_table(
                factor_weights,
                percent_columns=["Factor Weight"],
                number_columns=["Overall Factor Score"],
            ),
            width="stretch",
        )

    with tabs[7]:
        st.subheader("Expert: Machine Learning Return Prediction")
        info_card(
            "Important ML note",
            "Machine learning predictions are experimental. Stock returns are noisy, and this model should not be treated as a trading signal or profit guarantee.",
        )
        if st.button("Run ML return prediction"):
            try:
                dataset = build_feature_dataset(prices)
                model, ml_metrics = train_return_model(dataset)
                predictions = predict_latest_returns(prices, model)
                st.markdown("#### Validation Metrics")
                st.dataframe(
                    style_numeric_table(
                        ml_metrics,
                        percent_columns=["Directional Accuracy"],
                        number_columns=["MAE", "RMSE"],
                    ),
                    width="stretch",
                )
                st.markdown("#### Latest Predictions")
                st.dataframe(
                    style_numeric_table(
                        predictions,
                        percent_columns=["Predicted 21D Return", "Annualized Predicted Return"],
                    ),
                    width="stretch",
                )
                st.pyplot(ml_prediction_chart(predictions), width="stretch")
            except Exception as error:
                st.warning(f"ML prediction could not be completed: {error}")
        else:
            st.caption("Click the button to train the model. This may take a little time.")

    with tabs[8]:
        st.subheader("College Project Report Content")
        st.markdown("#### Methodology")
        for step in methodology_steps():
            st.markdown(f"- {step}")
        st.markdown("#### Formula Reference")
        st.dataframe(formula_reference(), width="stretch", hide_index=True)
        st.markdown("#### System File Map")
        st.dataframe(project_file_purpose(), width="stretch", hide_index=True)
        st.markdown("#### Final Conclusion")
        st.write(final_conclusion())
        st.markdown("#### Future Scope")
        for item in future_scope_items():
            st.markdown(f"- {item}")
        st.markdown("#### Educational Disclaimer")
        st.warning(
            "This project is for educational and academic use only. It is not financial advice, "
            "not investment advice, and does not guarantee returns."
        )

    with tabs[9]:
        st.subheader("Live / Near-Real-Time Market Snapshot")
        info_card(
            "Provider fallback",
            "Quotes use your configured free API keys with caching and fallback. "
            "Free APIs may be delayed, limited, or missing some NSE symbols.",
        )
        configured_providers = get_configured_provider_names()
        if configured_providers:
            st.caption("Configured providers: " + ", ".join(configured_providers))
        else:
            st.warning("No external API keys are configured. Add keys in Streamlit secrets.")

        quote_tickers = tuple(selected_tickers[:live_quote_count])
        if live_quotes_enabled and configured_providers:
            with st.spinner("Fetching cached live quote snapshot..."):
                live_quotes = load_live_quotes_cached(quote_tickers, quote_refresh_nonce)
            if live_quotes.empty:
                st.warning("No quote data was returned by the configured providers.")
            else:
                live_quotes = live_quotes.copy()
                last_prices = prices.iloc[-1].reindex(live_quotes["Ticker"]).reset_index(drop=True)
                live_quotes["Latest Historical Close"] = last_prices
                live_quotes["Gap vs Historical Close %"] = (
                    live_quotes["Price"] / live_quotes["Latest Historical Close"] - 1
                )
                live_quotes["Quote Available"] = live_quotes["Price"].notna()

                available_count = int(live_quotes["Quote Available"].sum())
                provider_count = live_quotes.loc[live_quotes["Quote Available"], "Provider"].nunique()
                provider_text = ", ".join(
                    live_quotes.loc[live_quotes["Quote Available"], "Provider"].dropna().unique().tolist()
                )

                live_cards = st.columns(4)
                with live_cards[0]:
                    metric_card("Quote Symbols", f"{available_count}/{len(live_quotes)}")
                with live_cards[1]:
                    metric_card("Providers Used", str(provider_count))
                with live_cards[2]:
                    best_change = live_quotes["Change %"].dropna().max()
                    metric_card(
                        "Best Live Move",
                        format_percent(best_change) if pd.notna(best_change) else "N/A",
                        best_change if pd.notna(best_change) else None,
                    )
                with live_cards[3]:
                    worst_change = live_quotes["Change %"].dropna().min()
                    metric_card(
                        "Worst Live Move",
                        format_percent(worst_change) if pd.notna(worst_change) else "N/A",
                        worst_change if pd.notna(worst_change) else None,
                    )

                st.caption(
                    "Source used: "
                    + (provider_text or "None")
                    + ". Cached for 60 seconds to protect free API limits."
                )
                st.dataframe(
                    style_numeric_table(
                        live_quotes,
                        percent_columns=["Change %", "Gap vs Historical Close %"],
                        number_columns=[
                            "Price",
                            "Previous Close",
                            "Change",
                            "Open",
                            "High",
                            "Low",
                            "Latest Historical Close",
                        ],
                    ),
                    width="stretch",
                    hide_index=True,
                )

                available_quotes = live_quotes[live_quotes["Quote Available"]].copy()
                if not available_quotes.empty:
                    movers = available_quotes.sort_values("Change %", ascending=False)
                    col_gain, col_loss = st.columns(2)
                    with col_gain:
                        st.markdown("#### Top Positive Moves")
                        st.dataframe(
                            style_numeric_table(
                                movers.head(5)[["Ticker", "Provider", "Price", "Change %"]],
                                percent_columns=["Change %"],
                                number_columns=["Price"],
                            ),
                            width="stretch",
                            hide_index=True,
                        )
                    with col_loss:
                        st.markdown("#### Top Negative Moves")
                        st.dataframe(
                            style_numeric_table(
                                movers.tail(5).sort_values("Change %")[
                                    ["Ticker", "Provider", "Price", "Change %"]
                                ],
                                percent_columns=["Change %"],
                                number_columns=["Price"],
                            ),
                            width="stretch",
                            hide_index=True,
                        )
        else:
            st.info("Enable live quote snapshot in the sidebar to call configured free APIs.")

        st.warning(
            "Educational disclaimer: this is near-real-time or delayed third-party data, "
            "not official exchange-grade NSE live data and not financial advice."
        )

    with tabs[10]:
        st.subheader("Downloadable CSV Outputs")
        st.download_button(
            "Download strategy comparison",
            data=dataframe_to_csv_bytes(comparison),
            file_name="strategy_comparison.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download risk metrics",
            data=dataframe_to_csv_bytes(risk_table),
            file_name="risk_metrics.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download asset summary",
            data=dataframe_to_csv_bytes(summarize_assets(prices)),
            file_name="asset_summary.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download Monte Carlo portfolios",
            data=dataframe_to_csv_bytes(random_results),
            file_name="monte_carlo_portfolios.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
