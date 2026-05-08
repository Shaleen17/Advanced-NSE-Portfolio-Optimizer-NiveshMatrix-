"""Advanced NSE Portfolio Optimizer Streamlit application."""

from __future__ import annotations

from datetime import date, datetime, time
import os
from pathlib import Path
import time as time_module
from typing import Callable
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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
from src.live_signals import (
    calculate_intraday_metrics,
    generate_alerts,
    generate_live_signal,
    generate_market_breadth,
)
from src.live_market import (
    LiveMarketSnapshot,
    fetch_live_market_snapshot,
    filter_live_market_frame as lm_filter_live_market_frame,
    prepare_live_market_frame as lm_prepare_live_market_frame,
)
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
from src.providers import get_configured_provider_names
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
    apply_chart_theme,
    allocation_chart,
    correlation_heatmap,
    cumulative_return_chart,
    drawdown_chart,
    efficient_frontier_chart,
    factor_score_chart,
    live_breadth_chart,
    live_change_distribution_chart,
    live_top_movers_chart,
    live_volume_leaders_chart,
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
        .terminal-band {{
            background:
                linear-gradient(90deg, rgba(255, 181, 28, 0.12), rgba(255, 0, 51, 0.06), rgba(0, 0, 0, 0));
            border-top: 1px solid rgba(255, 181, 28, 0.28);
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            padding: 14px 0;
            margin: 18px 0 12px 0;
        }}
        .terminal-title {{
            font-size: 17px;
            font-weight: 850;
            letter-spacing: 0;
        }}
        .terminal-subtitle {{
            color: {CHART_COLORS["muted"]};
            font-size: 12px;
            margin-top: 3px;
            overflow-wrap: anywhere;
        }}
        .terminal-status {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            border: 1px solid rgba(255, 181, 28, 0.32);
            border-radius: 999px;
            padding: 6px 10px;
            background: rgba(255, 181, 28, 0.08);
            color: #FFB51C;
            font-size: 12px;
            font-weight: 800;
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


def terminal_band(title: str, subtitle: str = "") -> None:
    """Render a full-width finance terminal section label."""
    st.markdown(
        f"""
        <div class="terminal-band">
            <div class="terminal-title">{title}</div>
            <div class="terminal-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_float(value, digits: int = 2, suffix: str = "") -> str:
    """Format numeric values without crashing on missing provider data."""
    try:
        if value is None:
            return "N/A"
        if isinstance(value, str) and not value.strip():
            return "N/A"
        if pd.isna(value):
            return "N/A"
        return f"{float(value):,.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


def safe_percent(value, digits: int = 2) -> str:
    """Format percentage values without crashing on missing provider data."""
    try:
        if value is None:
            return "N/A"
        if isinstance(value, str) and not value.strip():
            return "N/A"
        if pd.isna(value):
            return "N/A"
        return f"{float(value):,.{digits}%}"
    except (TypeError, ValueError):
        return "N/A"


def safe_integer(value) -> str:
    """Format whole-number market values without crashing on missing data."""
    return safe_float(value, digits=0)


def format_percent(value: float) -> str:
    """Format a float as a percentage."""
    return safe_percent(value)


def format_number(value: float) -> str:
    """Format a float with four decimals."""
    return safe_float(value, digits=4)


def format_ratio(value: float) -> str:
    """Format market breadth ratios, including infinite advancer-only breadth."""
    try:
        if value is None or pd.isna(value):
            return "N/A"
        if np.isinf(value):
            return "∞"
        return f"{float(value):,.2f}x"
    except (TypeError, ValueError):
        return "N/A"


def format_duration(seconds: float | None) -> str:
    """Format diagnostic durations."""
    try:
        if seconds is None or pd.isna(seconds):
            return "N/A"
        seconds = float(seconds)
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = seconds / 60
        if minutes < 60:
            return f"{minutes:.1f}m"
        return f"{minutes / 60:.1f}h"
    except (TypeError, ValueError):
        return "N/A"


def style_numeric_table(
    dataframe: pd.DataFrame,
    percent_columns: list[str] | None = None,
    number_columns: list[str] | None = None,
):
    """Style numeric tables with green positives and red negatives."""
    percent_columns = percent_columns or []
    number_columns = number_columns or []
    formatters = {
        column: safe_percent
        for column in percent_columns
        if column in dataframe.columns
    }
    formatters.update(
        {
            column: lambda value: safe_float(value, digits=4)
            for column in number_columns
            if column in dataframe.columns
        }
    )

    def color_signed(value):
        if isinstance(value, (int, float, np.number)) and not pd.isna(value):
            if value > 0:
                return f"color: {CHART_COLORS['profit']}"
            if value < 0:
                return f"color: {CHART_COLORS['loss']}"
        return f"color: {CHART_COLORS['text']}"

    return dataframe.style.format(formatters).map(color_signed)


@st.cache_data(show_spinner=False, ttl=60)
def load_live_quotes_cached(
    tickers: tuple[str, ...],
    refresh_nonce: int,
    historical_last_items: tuple[tuple[str, float], ...],
    historical_previous_items: tuple[tuple[str, float], ...],
    provider_preference: str,
    use_fallback_cache: bool,
) -> LiveMarketSnapshot:
    """Cache multi-provider quote calls to protect free API limits."""
    del refresh_nonce
    return fetch_live_market_snapshot(
        tickers,
        historical_last_prices=dict(historical_last_items),
        historical_previous_closes=dict(historical_previous_items),
        provider_preference=provider_preference,
        use_fallback_cache=use_fallback_cache,
    )


def historical_quote_items(
    prices: pd.DataFrame,
    offset: int = 1,
) -> tuple[tuple[str, float], ...]:
    """Build cache-safe historical fallback quote values from loaded prices."""
    if prices.empty or len(prices) < offset:
        return tuple()
    row = prices.iloc[-offset]
    values = pd.to_numeric(row, errors="coerce").dropna()
    return tuple((str(ticker), float(value)) for ticker, value in values.items())


def format_market_timestamp(value: str | None) -> str:
    """Format provider timestamps in India market time."""
    if not value:
        return "N/A"
    try:
        timestamp = pd.to_datetime(value, utc=True)
        if pd.isna(timestamp):
            return "N/A"
        return timestamp.tz_convert("Asia/Kolkata").strftime("%d %b %Y, %H:%M:%S IST")
    except (TypeError, ValueError):
        return str(value)


def market_status() -> tuple[str, float | None]:
    """Infer regular NSE session status from India local time."""
    try:
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        if now.weekday() >= 5:
            return "Closed", -1
        if time(9, 15) <= now.time() <= time(15, 30):
            return "Open", 1
        return "Closed", -1
    except Exception:
        return "Unknown", None


def prepare_live_market_frame(live_quotes: pd.DataFrame) -> pd.DataFrame:
    """Add terminal metrics, signals, and data-quality labels to live quotes."""
    frame = live_quotes.copy()
    numeric_columns = [
        "Last Price",
        "Previous Close",
        "Change",
        "Change %",
        "Open",
        "High",
        "Low",
        "Volume",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    range_denominator = frame["Previous Close"].where(
        frame["Previous Close"].notna() & (frame["Previous Close"] != 0)
    )
    open_denominator = frame["Open"].where(frame["Open"].notna() & (frame["Open"] != 0))
    last_price_denominator = frame["Last Price"].where(
        frame["Last Price"].notna() & (frame["Last Price"] != 0)
    )
    range_denominator = range_denominator.fillna(open_denominator).fillna(last_price_denominator)
    frame["Day Range %"] = np.where(
        frame["High"].notna()
        & frame["Low"].notna()
        & range_denominator.notna()
        & (range_denominator != 0),
        (frame["High"] - frame["Low"]) / range_denominator,
        np.nan,
    )

    def quote_signal(value) -> str:
        if pd.isna(value):
            return "— N/A"
        if value > 0:
            return "▲ Gainer"
        if value < 0:
            return "▼ Loser"
        return "— Flat"

    def data_quality(row: pd.Series) -> str:
        if pd.isna(row.get("Last Price")) or row.get("Status") == "failed":
            return "Unavailable"
        status = str(row.get("Status", "")).lower()
        source = str(row.get("Source", "")).lower()
        if status == "cached" or "cache" in source:
            return "Cached"
        if status == "fallback" or "fallback" in source:
            return "Fallback"
        return "Live"

    frame["Signal"] = frame["Change %"].apply(quote_signal)
    frame["Data Quality"] = frame.apply(data_quality, axis=1)
    frame["Quote Available"] = frame["Last Price"].notna()
    return frame


def quote_card_value(
    frame: pd.DataFrame,
    column: str,
    label_formatter: Callable[[float], str],
    ascending: bool = False,
    require_sign: str | None = None,
) -> tuple[str, float | None]:
    """Return a ticker/value pair for a market overview card."""
    valid = frame.dropna(subset=[column]).copy()
    if require_sign == "positive":
        valid = valid[valid[column] > 0]
    elif require_sign == "negative":
        valid = valid[valid[column] < 0]
    if valid.empty:
        return "N/A", None
    row = valid.sort_values(column, ascending=ascending).iloc[0]
    value = float(row[column])
    return f"{row['Ticker']} {label_formatter(value)}", value


def filter_live_market_frame(
    frame: pd.DataFrame,
    search_query: str,
    move_filter: str,
    minimum_volume: int,
    sort_option: str,
    holdings_only: bool,
    selected_tickers: list[str],
) -> pd.DataFrame:
    """Apply tab-level search, filters, and sort controls to quote rows."""
    filtered = frame.copy()
    if holdings_only:
        filtered = filtered[filtered["Ticker"].isin(selected_tickers)]
    search_query = search_query.strip().upper()
    if search_query:
        filtered = filtered[
            filtered["Ticker"].astype(str).str.upper().str.contains(search_query, regex=False, na=False)
        ]
    if move_filter == "Gainers":
        filtered = filtered[filtered["Change %"] > 0]
    elif move_filter == "Losers":
        filtered = filtered[filtered["Change %"] < 0]
    if minimum_volume > 0:
        filtered = filtered[filtered["Volume"].fillna(0) >= minimum_volume]

    sort_map = {
        "Change %": ("Change %", False),
        "Volume": ("Volume", False),
        "Volatility": ("Day Range %", False),
        "Ticker": ("Ticker", True),
    }
    sort_column, ascending = sort_map.get(sort_option, ("Change %", False))
    return filtered.sort_values(sort_column, ascending=ascending, na_position="last")


def style_live_quote_table(dataframe: pd.DataFrame):
    """Style the live market terminal table."""
    formatters = {
        "Last Price": lambda value: safe_float(value, digits=2),
        "Change": lambda value: safe_float(value, digits=2),
        "Change %": safe_percent,
        "Open": lambda value: safe_float(value, digits=2),
        "High": lambda value: safe_float(value, digits=2),
        "Low": lambda value: safe_float(value, digits=2),
        "Previous Close": lambda value: safe_float(value, digits=2),
        "Day Range %": safe_percent,
        "Volume": safe_integer,
    }

    def signed_color(value):
        if not isinstance(value, (int, float, np.number)) or pd.isna(value):
            return "color: #FFB51C"
        if value > 0:
            return f"color: {CHART_COLORS['profit']}"
        if value < 0:
            return f"color: {CHART_COLORS['loss']}"
        return "color: #FFB51C"

    def signal_color(value):
        text = str(value)
        if text.startswith("▲"):
            return f"color: {CHART_COLORS['profit']}; font-weight: 800"
        if text.startswith("▼"):
            return f"color: {CHART_COLORS['loss']}; font-weight: 800"
        return "color: #FFB51C; font-weight: 800"

    def quality_color(value):
        text = str(value).lower()
        if text == "live":
            return f"color: {CHART_COLORS['profit']}; font-weight: 800"
        if text in {"cached", "fallback", "stale cache"}:
            return "color: #FFB51C; font-weight: 800"
        return f"color: {CHART_COLORS['loss']}; font-weight: 800"

    styled = dataframe.style.format(
        {column: formatter for column, formatter in formatters.items() if column in dataframe.columns}
    )
    signed_columns = [column for column in ["Change", "Change %"] if column in dataframe.columns]
    if signed_columns:
        styled = styled.map(signed_color, subset=signed_columns)
    if "Signal" in dataframe.columns:
        styled = styled.map(signal_color, subset=["Signal"])
    if "Data Quality" in dataframe.columns:
        styled = styled.map(quality_color, subset=["Data Quality"])
    return styled


def style_signal_table(dataframe: pd.DataFrame):
    """Style rule-based signal output with readable finance colors."""
    formatters = {
        "Last Price": lambda value: safe_float(value, digits=2),
        "Change %": safe_percent,
        "Intraday return %": safe_percent,
        "Gap from previous close %": safe_percent,
        "Distance from day high %": safe_percent,
        "Distance from day low %": safe_percent,
        "Volume rank": safe_integer,
        "Volatility proxy": safe_percent,
    }

    def signal_color(value):
        text = str(value)
        if "Buy" in text:
            return f"color: {CHART_COLORS['profit']}; font-weight: 850"
        if "Sell" in text or text == "Weakness":
            return f"color: {CHART_COLORS['loss']}; font-weight: 850"
        if text == "Watchlist":
            return "color: #FFB51C; font-weight: 850"
        if text == "Insufficient Data":
            return f"color: {CHART_COLORS['loss']}; font-weight: 850"
        return f"color: {CHART_COLORS['muted']}; font-weight: 750"

    styled = dataframe.style.format(
        {column: formatter for column, formatter in formatters.items() if column in dataframe.columns}
    )
    if "Live Signal" in dataframe.columns:
        styled = styled.map(signal_color, subset=["Live Signal"])
    return styled


def build_live_strategy_weights(
    tickers: list[str],
    strategy_weights: dict[str, np.ndarray],
    black_litterman_weights: np.ndarray | None = None,
    factor_weights: pd.DataFrame | None = None,
) -> dict[str, pd.Series]:
    """Convert all available strategy allocations into ticker-indexed Series."""
    weights_by_strategy = {
        name: pd.Series(weights, index=tickers, dtype="float64")
        for name, weights in strategy_weights.items()
    }
    if black_litterman_weights is not None:
        weights_by_strategy["Black-Litterman"] = pd.Series(
            black_litterman_weights,
            index=tickers,
            dtype="float64",
        )
    if factor_weights is not None and not factor_weights.empty:
        factor_series = (
            factor_weights.set_index("Ticker")["Factor Weight"]
            .reindex(tickers)
            .fillna(0.0)
            .astype("float64")
        )
        weights_by_strategy["Factor Portfolio"] = factor_series
    return weights_by_strategy


def calculate_live_portfolio_returns(
    live_df: pd.DataFrame,
    weights_by_strategy: dict[str, pd.Series],
) -> pd.DataFrame:
    """Estimate intraday strategy returns from live Change % values."""
    change = pd.to_numeric(live_df.set_index("Ticker")["Change %"], errors="coerce")
    rows = []
    for strategy, weights in weights_by_strategy.items():
        aligned_weights = weights.reindex(change.index).fillna(0.0)
        available = change.notna()
        contribution = aligned_weights[available] * change[available]
        rows.append(
            {
                "Strategy": strategy,
                "Estimated Today": contribution.sum(),
                "Covered Weight": aligned_weights[available].sum(),
                "Missing Symbols": int((aligned_weights[~available] > 0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("Estimated Today", ascending=False)


def build_holding_contribution_table(
    live_df: pd.DataFrame,
    strategy_weights: pd.Series,
) -> pd.DataFrame:
    """Build holding-level live return contribution for one strategy."""
    frame = live_df.copy()
    frame["Strategy Weight"] = (
        strategy_weights.reindex(frame["Ticker"]).fillna(0.0).reset_index(drop=True)
    )
    frame["Contribution to Portfolio Return"] = np.where(
        frame["Change %"].notna(),
        frame["Strategy Weight"] * frame["Change %"],
        np.nan,
    )
    return frame[
        [
            "Ticker",
            "Strategy Weight",
            "Change %",
            "Contribution to Portfolio Return",
            "Live Signal",
            "Last Price",
            "Data Quality",
        ]
    ].sort_values("Contribution to Portfolio Return", ascending=False, na_position="last")


def build_live_risk_overlay(live_df: pd.DataFrame, daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Compare live moves with each stock's historical daily volatility."""
    historical_volatility = daily_returns.std().replace(0, np.nan)
    frame = live_df[["Ticker", "Change %", "Live Signal", "Data Quality"]].copy()
    frame["Historical Daily Volatility"] = frame["Ticker"].map(historical_volatility)
    frame["Live move vs historical daily volatility"] = np.where(
        frame["Historical Daily Volatility"].notna() & frame["Change %"].notna(),
        frame["Change %"].abs() / frame["Historical Daily Volatility"],
        np.nan,
    )
    frame["Z-score of today's move"] = np.where(
        frame["Historical Daily Volatility"].notna() & frame["Change %"].notna(),
        frame["Change %"] / frame["Historical Daily Volatility"],
        np.nan,
    )
    frame["Risk Flag"] = np.where(
        frame["Z-score of today's move"].abs() >= 2,
        "Unusual Move",
        "Normal",
    )
    frame.loc[frame["Z-score of today's move"].isna(), "Risk Flag"] = "Insufficient Data"
    return frame.sort_values(
        "Z-score of today's move",
        key=lambda series: series.abs(),
        ascending=False,
        na_position="last",
    )


def calculate_portfolio_shocks(
    live_returns: pd.DataFrame,
    weights_by_strategy: dict[str, pd.Series],
    selected_stock: str,
    stock_shock: float,
    market_shock: float,
) -> pd.DataFrame:
    """Estimate selected-stock and market-wide shock impact by strategy."""
    base_returns = live_returns.set_index("Strategy")["Estimated Today"]
    rows = []
    for strategy, weights in weights_by_strategy.items():
        stock_weight = float(weights.get(selected_stock, 0.0))
        total_weight = float(weights.sum())
        stock_impact = stock_weight * stock_shock
        market_impact = total_weight * market_shock
        base_return = float(base_returns.get(strategy, 0.0))
        rows.append(
            {
                "Strategy": strategy,
                "Base Estimated Return": base_return,
                "Selected Stock Impact": stock_impact,
                "Market Shock Impact": market_impact,
                "Shocked Estimated Return": base_return + stock_impact + market_impact,
            }
        )
    return pd.DataFrame(rows).sort_values("Shocked Estimated Return", ascending=False)


def contribution_bar_chart(contribution_table: pd.DataFrame, strategy_name: str) -> plt.Figure:
    """Plot live contribution by ticker for the selected strategy."""
    plot_data = contribution_table.dropna(subset=["Contribution to Portfolio Return"]).copy()
    plot_data = plot_data.sort_values("Contribution to Portfolio Return").tail(20)
    colors = [
        CHART_COLORS["profit"] if value >= 0 else CHART_COLORS["loss"]
        for value in plot_data["Contribution to Portfolio Return"]
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.barh(plot_data["Ticker"], plot_data["Contribution to Portfolio Return"], color=colors)
    apply_chart_theme(ax, f"{strategy_name} Live Contribution by Holding")
    ax.set_xlabel("Contribution to Portfolio Return")
    fig.tight_layout()
    return fig


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
    live_quotes_enabled, live_quote_count, _quote_refresh_nonce = sidebar_realtime_controls(
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

    black_litterman_weights = None
    black_litterman_table = None
    black_litterman_error = ""
    try:
        black_litterman_weights, black_litterman_table = black_litterman_allocation(prices)
    except Exception as error:
        black_litterman_error = str(error)

    factor_scores = pd.DataFrame()
    factor_weights = pd.DataFrame()
    factor_error = ""
    try:
        factor_scores = build_factor_score_table(prices)
        factor_weights = build_factor_portfolio_weights(factor_scores, factor_size)
    except Exception as error:
        factor_error = str(error)

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
        if black_litterman_weights is not None and black_litterman_table is not None:
            bl_allocation = pd.DataFrame(
                {"Ticker": prices.columns, "Black-Litterman Weight": black_litterman_weights}
            ).sort_values("Black-Litterman Weight", ascending=False)
            st.dataframe(
                style_numeric_table(
                    black_litterman_table,
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
        else:
            st.warning(f"Black-Litterman model could not be calculated: {black_litterman_error}")

    with tabs[6]:
        st.subheader("Expert: Factor Investing Analysis")
        if factor_error:
            st.warning(f"Factor portfolio could not be calculated: {factor_error}")
        else:
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
        terminal_band(
            "Market Terminal",
            "Yahoo Finance, configured API providers, local cache, and historical fallback are used in order.",
        )
        provider_order = get_configured_provider_names()
        terminal_controls = st.columns([2.6, 1, 1])
        with terminal_controls[0]:
            st.markdown(
                f"<span class='terminal-status'>Provider order: {', '.join(provider_order)}</span>",
                unsafe_allow_html=True,
            )
        with terminal_controls[1]:
            auto_refresh_enabled = st.checkbox(
                "Auto refresh",
                value=False,
                key="live_market_auto_refresh",
            )
        with terminal_controls[2]:
            refresh_interval = st.selectbox(
                "Refresh interval",
                [30, 60, 120, 300],
                index=1,
                format_func=lambda value: f"{value}s",
                key="live_market_refresh_interval",
            )

        settings_cols = st.columns([1.1, 1.1, 1.1, 1.1, 1.1, 1])
        with settings_cols[0]:
            include_all_universe = st.checkbox(
                "Include all NSE universe tickers",
                value=False,
                key="live_market_all_universe",
            )
        live_universe = NSE_TICKERS if include_all_universe else selected_tickers
        with settings_cols[1]:
            max_symbols_to_fetch = st.slider(
                "Max symbols to fetch",
                min_value=1,
                max_value=max(1, len(live_universe)),
                value=min(max(live_quote_count, 8), max(1, len(live_universe))),
                step=1,
                key=f"live_market_max_symbols_{'all' if include_all_universe else 'selected'}",
            )
        with settings_cols[2]:
            provider_preference = st.selectbox(
                "Provider preference",
                ["Auto"] + provider_order,
                key="live_market_provider_preference",
            )
        with settings_cols[3]:
            use_fallback_cache = st.checkbox(
                "Use fallback cache",
                value=True,
                key="live_market_use_fallback_cache",
            )
        with settings_cols[4]:
            show_diagnostics = st.checkbox(
                "Show diagnostics",
                value=True,
                key="live_market_show_diagnostics",
            )
        with settings_cols[5]:
            refresh_clicked = st.button(
                "Refresh Now",
                key="live_market_refresh_now",
                width="stretch",
            )
        if refresh_clicked:
            st.session_state.quote_refresh_nonce += 1
            live_quotes_enabled = True
        if auto_refresh_enabled:
            live_quotes_enabled = True

        if auto_refresh_enabled:
            auto_bucket = int(time_module.time() // int(refresh_interval))
            components.html(
                f"""
                <script>
                setTimeout(function() {{
                    window.parent.location.reload();
                }}, {int(refresh_interval) * 1000});
                </script>
                """,
                height=0,
            )
        else:
            auto_bucket = 0

        quote_tickers = tuple(live_universe[:max_symbols_to_fetch])
        if live_quotes_enabled:
            historical_last_items = historical_quote_items(prices, offset=1)
            historical_previous_items = historical_quote_items(prices, offset=2)
            with st.spinner("Fetching cached live quote snapshot..."):
                quote_result = load_live_quotes_cached(
                    quote_tickers,
                    int(st.session_state.quote_refresh_nonce) + auto_bucket,
                    historical_last_items,
                    historical_previous_items,
                    provider_preference,
                    use_fallback_cache,
                )
            live_quotes = lm_prepare_live_market_frame(quote_result.dataframe)
            live_quotes = calculate_intraday_metrics(live_quotes)
            live_quotes["Live Signal"] = live_quotes.apply(generate_live_signal, axis=1)
            if live_quotes.empty:
                st.warning("No quote data could be prepared for the selected symbols.")
            else:
                last_prices = prices.iloc[-1].reindex(live_quotes["Ticker"]).reset_index(drop=True)
                live_quotes["Latest Historical Close"] = last_prices
                live_quotes["Latest Historical Close"] = pd.to_numeric(
                    live_quotes["Latest Historical Close"],
                    errors="coerce",
                )
                live_quotes["Gap vs Historical Close %"] = np.where(
                    live_quotes["Last Price"].notna()
                    & live_quotes["Latest Historical Close"].notna()
                    & (live_quotes["Latest Historical Close"] != 0),
                    live_quotes["Last Price"] / live_quotes["Latest Historical Close"] - 1,
                    np.nan,
                )

                available_count = int(live_quotes["Quote Available"].sum())
                failed_count = len(quote_result.failed_symbols)
                provider_text = quote_result.source or "Unavailable"
                status_text = quote_result.status.title()
                last_updated_text = format_market_timestamp(quote_result.timestamp)
                market_status_text, market_status_sign = market_status()
                diagnostics = quote_result.diagnostics

                if refresh_clicked and quote_result.status != "failed":
                    st.toast("Live market snapshot refreshed.")

                terminal_band("Market Status Header", "Session, provider, cache, and symbol coverage.")
                status_cards_top = st.columns(3)
                with status_cards_top[0]:
                    metric_card("Market Status", market_status_text, market_status_sign)
                with status_cards_top[1]:
                    metric_card("Last Updated", last_updated_text)
                with status_cards_top[2]:
                    metric_card("Data Provider", provider_text)

                status_cards_bottom = st.columns(3)
                with status_cards_bottom[0]:
                    metric_card("Successful Symbols", f"{available_count}/{len(live_quotes)}")
                with status_cards_bottom[1]:
                    metric_card("Failed Symbols", str(failed_count), -1 if failed_count else None)
                with status_cards_bottom[2]:
                    metric_card("Refresh / Cache", f"60s / {status_text}")

                if quote_result.status == "partial":
                    failed_symbols = ", ".join(quote_result.failed_symbols) or "some symbols"
                    st.warning(
                        "Some live quotes could not be resolved. "
                        f"Missing symbols: {failed_symbols}."
                    )
                elif quote_result.status == "stale_cache":
                    st.warning(
                        "Live providers did not return a fresh snapshot. Showing stale cached data from "
                        f"{format_market_timestamp(diagnostics.cache_timestamp)}."
                    )
                elif quote_result.status in {"cached", "fallback"}:
                    st.warning(
                        "Live APIs were incomplete or unavailable, so the table includes cached "
                        "or historical fallback prices."
                    )
                elif quote_result.status == "failed":
                    st.error(
                        "Live APIs and fallback sources could not return usable prices. "
                        "The table remains available with N/A values instead of crashing."
                    )
                if quote_result.errors:
                    with st.expander("Provider diagnostics"):
                        for message in quote_result.errors:
                            st.write(message)

                if show_diagnostics:
                    terminal_band("Data Quality Diagnostics", "Provider, cache, coverage, and fallback details.")
                    diagnostic_rows = pd.DataFrame(
                        [
                            {"Metric": "Provider used", "Value": diagnostics.provider_used},
                            {"Metric": "API response time", "Value": format_duration(diagnostics.api_response_time)},
                            {"Metric": "Cache age", "Value": format_duration(diagnostics.cache_age_seconds)},
                            {"Metric": "Symbols requested", "Value": diagnostics.symbols_requested},
                            {"Metric": "Symbols returned", "Value": diagnostics.symbols_returned},
                            {"Metric": "Missing price count", "Value": diagnostics.missing_price_count},
                            {
                                "Metric": "Failed symbols",
                                "Value": ", ".join(diagnostics.failed_symbols) if diagnostics.failed_symbols else "None",
                            },
                            {
                                "Metric": "Last successful refresh",
                                "Value": format_market_timestamp(diagnostics.last_successful_refresh),
                            },
                            {"Metric": "Current fallback mode", "Value": diagnostics.current_fallback_mode},
                        ]
                    )
                    diagnostic_rows["Value"] = diagnostic_rows["Value"].astype(str)
                    st.dataframe(diagnostic_rows, width="stretch", hide_index=True)

                terminal_band("Live Quote Overview", "Leaders, laggards, liquidity, volatility, and breadth.")
                best_gainer, best_gainer_value = quote_card_value(
                    live_quotes,
                    "Change %",
                    safe_percent,
                    require_sign="positive",
                )
                worst_loser, worst_loser_value = quote_card_value(
                    live_quotes,
                    "Change %",
                    safe_percent,
                    ascending=True,
                    require_sign="negative",
                )
                highest_volume, _ = quote_card_value(live_quotes, "Volume", safe_integer)
                most_volatile, _ = quote_card_value(live_quotes, "Day Range %", safe_percent)
                average_change = live_quotes["Change %"].dropna().mean()
                advancing_count = int((live_quotes["Change %"] > 0).sum())
                declining_count = int((live_quotes["Change %"] < 0).sum())

                overview_top = st.columns(3)
                with overview_top[0]:
                    metric_card("Best Gainer", best_gainer, best_gainer_value)
                with overview_top[1]:
                    metric_card("Worst Loser", worst_loser, worst_loser_value)
                with overview_top[2]:
                    metric_card("Highest Volume", highest_volume)

                overview_bottom = st.columns(3)
                with overview_bottom[0]:
                    metric_card("Most Volatile Intraday", most_volatile)
                with overview_bottom[1]:
                    metric_card(
                        "Average Change %",
                        format_percent(average_change) if pd.notna(average_change) else "N/A",
                        average_change if pd.notna(average_change) else None,
                    )
                with overview_bottom[2]:
                    breadth_sign = None
                    if advancing_count > declining_count:
                        breadth_sign = 1
                    elif declining_count > advancing_count:
                        breadth_sign = -1
                    metric_card("Advancing / Declining", f"{advancing_count} / {declining_count}", breadth_sign)

                terminal_band("Live Charts", "Market movers, liquidity, distribution, and breadth.")
                chart_row_one = st.columns(2)
                with chart_row_one[0]:
                    st.pyplot(live_top_movers_chart(live_quotes, gainers=True), width="stretch")
                with chart_row_one[1]:
                    st.pyplot(live_top_movers_chart(live_quotes, gainers=False), width="stretch")
                chart_row_two = st.columns(2)
                with chart_row_two[0]:
                    st.pyplot(live_volume_leaders_chart(live_quotes), width="stretch")
                with chart_row_two[1]:
                    st.pyplot(live_change_distribution_chart(live_quotes), width="stretch")
                st.pyplot(
                    live_breadth_chart(advancing_count, declining_count, int((live_quotes["Change %"] == 0).sum())),
                    width="stretch",
                )

                terminal_band(
                    "Market Intelligence",
                    "Explainable rule-based signals, breadth diagnostics, and live alerts.",
                )
                market_breadth = generate_market_breadth(live_quotes)
                breadth_top = st.columns(3)
                with breadth_top[0]:
                    metric_card("Advancers", str(market_breadth["Advancers"]), 1 if market_breadth["Advancers"] else None)
                with breadth_top[1]:
                    metric_card("Decliners", str(market_breadth["Decliners"]), -1 if market_breadth["Decliners"] else None)
                with breadth_top[2]:
                    metric_card("Unchanged", str(market_breadth["Unchanged"]))

                breadth_bottom = st.columns(3)
                with breadth_bottom[0]:
                    ratio_value = market_breadth["Advance-decline ratio"]
                    ratio_sign = None
                    if pd.notna(ratio_value) and not np.isinf(ratio_value):
                        ratio_sign = 1 if ratio_value >= 1 else -1
                    elif np.isinf(ratio_value):
                        ratio_sign = 1
                    metric_card("Advance / Decline Ratio", format_ratio(ratio_value), ratio_sign)
                with breadth_bottom[1]:
                    average_return = market_breadth["Average return"]
                    metric_card(
                        "Average Return",
                        format_percent(average_return) if pd.notna(average_return) else "N/A",
                        average_return if pd.notna(average_return) else None,
                    )
                with breadth_bottom[2]:
                    positive_breadth = market_breadth["Positive breadth %"]
                    metric_card(
                        "Positive Breadth",
                        format_percent(positive_breadth) if pd.notna(positive_breadth) else "N/A",
                        positive_breadth - 0.5 if pd.notna(positive_breadth) else None,
                    )

                signal_filter = st.selectbox(
                    "Signal filter",
                    ["All Signals", "Buy Signals", "Sell Signals", "Watchlist", "No Signal", "Insufficient Data"],
                    key="live_market_signal_filter",
                )
                signal_groups = {
                    "Buy Signals": ["Strong Buy Momentum", "Buy Momentum"],
                    "Sell Signals": ["Weakness", "Strong Sell Pressure"],
                    "Watchlist": ["Watchlist"],
                    "No Signal": ["No Signal"],
                    "Insufficient Data": ["Insufficient Data"],
                }
                signal_rows = live_quotes.copy()
                if signal_filter != "All Signals":
                    signal_rows = signal_rows[
                        signal_rows["Live Signal"].isin(signal_groups.get(signal_filter, []))
                    ]
                signal_columns = [
                    "Ticker",
                    "Live Signal",
                    "Last Price",
                    "Change %",
                    "Intraday return %",
                    "Gap from previous close %",
                    "Distance from day high %",
                    "Distance from day low %",
                    "Volume rank",
                    "Volatility proxy",
                    "Data Quality",
                ]
                st.markdown("#### Signal Table")
                if signal_rows.empty:
                    st.warning("No rows match the current signal filter.")
                else:
                    st.dataframe(
                        style_signal_table(signal_rows[signal_columns].reset_index(drop=True)),
                        width="stretch",
                        hide_index=True,
                    )

                st.markdown("#### Alert Feed")
                alerts = generate_alerts(live_quotes)
                if not alerts:
                    st.info("No rule-based market alerts triggered for this snapshot.")
                for alert in alerts:
                    message = alert.get("message", "")
                    if alert.get("level") == "success":
                        st.success(message)
                    elif alert.get("level") == "warning":
                        st.warning(message)
                    else:
                        st.info(message)

                terminal_band(
                    "Live Portfolio Impact",
                    "Estimated intraday movement using live Change %. Missing symbols are excluded.",
                )
                st.info(
                    "Live portfolio impact is an estimate only. Historical optimization inputs are unchanged."
                )
                live_weights_by_strategy = build_live_strategy_weights(
                    prices.columns.tolist(),
                    strategy_weights,
                    black_litterman_weights,
                    None if factor_error else factor_weights,
                )
                live_portfolio_returns = calculate_live_portfolio_returns(
                    live_quotes,
                    live_weights_by_strategy,
                )
                missing_live_symbols = [
                    ticker
                    for ticker in selected_tickers
                    if ticker
                    not in live_quotes.loc[live_quotes["Change %"].notna(), "Ticker"].astype(str).tolist()
                ]
                if missing_live_symbols:
                    st.warning(
                        "Excluded from live impact because live Change % is missing: "
                        + ", ".join(missing_live_symbols[:12])
                        + ("..." if len(missing_live_symbols) > 12 else "")
                    )

                portfolio_return_map = live_portfolio_returns.set_index("Strategy")[
                    "Estimated Today"
                ].to_dict()
                best_live_strategy = live_portfolio_returns.iloc[0]
                worst_live_strategy = live_portfolio_returns.iloc[-1]

                impact_cards_top = st.columns(3)
                with impact_cards_top[0]:
                    equal_today = portfolio_return_map.get("Equal Weight", np.nan)
                    metric_card(
                        "Equal Weight Today",
                        format_percent(equal_today) if pd.notna(equal_today) else "N/A",
                        equal_today if pd.notna(equal_today) else None,
                    )
                with impact_cards_top[1]:
                    max_sharpe_today = portfolio_return_map.get("Maximum Sharpe", np.nan)
                    metric_card(
                        "Max Sharpe Today",
                        format_percent(max_sharpe_today) if pd.notna(max_sharpe_today) else "N/A",
                        max_sharpe_today if pd.notna(max_sharpe_today) else None,
                    )
                with impact_cards_top[2]:
                    min_vol_today = portfolio_return_map.get("Minimum Volatility", np.nan)
                    metric_card(
                        "Min Volatility Today",
                        format_percent(min_vol_today) if pd.notna(min_vol_today) else "N/A",
                        min_vol_today if pd.notna(min_vol_today) else None,
                    )

                impact_cards_bottom = st.columns(2)
                with impact_cards_bottom[0]:
                    metric_card(
                        "Best Strategy Today",
                        f"{best_live_strategy['Strategy']} {format_percent(best_live_strategy['Estimated Today'])}",
                        best_live_strategy["Estimated Today"],
                    )
                with impact_cards_bottom[1]:
                    metric_card(
                        "Worst Strategy Today",
                        f"{worst_live_strategy['Strategy']} {format_percent(worst_live_strategy['Estimated Today'])}",
                        worst_live_strategy["Estimated Today"],
                    )

                st.markdown("#### Holding-Level Contribution")
                contribution_strategy = st.selectbox(
                    "Contribution strategy",
                    list(live_weights_by_strategy.keys()),
                    index=0,
                    key="live_portfolio_contribution_strategy",
                )
                contribution_table = build_holding_contribution_table(
                    live_quotes,
                    live_weights_by_strategy[contribution_strategy],
                )
                st.dataframe(
                    style_numeric_table(
                        contribution_table,
                        percent_columns=[
                            "Strategy Weight",
                            "Change %",
                            "Contribution to Portfolio Return",
                        ],
                        number_columns=["Last Price"],
                    ),
                    width="stretch",
                    hide_index=True,
                )
                if contribution_table["Contribution to Portfolio Return"].notna().any():
                    st.pyplot(
                        contribution_bar_chart(contribution_table, contribution_strategy),
                        width="stretch",
                    )

                st.markdown("#### Portfolio Shock View")
                shock_cols = st.columns(3)
                with shock_cols[0]:
                    shock_ticker = st.selectbox(
                        "Selected stock shock",
                        selected_tickers,
                        key="live_portfolio_shock_ticker",
                    )
                with shock_cols[1]:
                    stock_shock_label = st.radio(
                        "Simulate selected stock move",
                        ["None", "+1%", "-1%"],
                        horizontal=True,
                        key="live_portfolio_stock_shock",
                    )
                with shock_cols[2]:
                    market_shock_label = st.selectbox(
                        "Market-wide shock",
                        ["None", "-1%", "-2%", "-5%"],
                        key="live_portfolio_market_shock",
                    )
                stock_shock_value = {"+1%": 0.01, "-1%": -0.01}.get(stock_shock_label, 0.0)
                market_shock_value = {"-1%": -0.01, "-2%": -0.02, "-5%": -0.05}.get(
                    market_shock_label,
                    0.0,
                )
                shock_table = calculate_portfolio_shocks(
                    live_portfolio_returns,
                    live_weights_by_strategy,
                    shock_ticker,
                    stock_shock_value,
                    market_shock_value,
                )
                st.dataframe(
                    style_numeric_table(
                        shock_table,
                        percent_columns=[
                            "Base Estimated Return",
                            "Selected Stock Impact",
                            "Market Shock Impact",
                            "Shocked Estimated Return",
                        ],
                    ),
                    width="stretch",
                    hide_index=True,
                )

                st.markdown("#### Risk Overlay")
                risk_overlay = build_live_risk_overlay(live_quotes, daily_returns)
                unusual_moves = risk_overlay[risk_overlay["Risk Flag"] == "Unusual Move"]
                if unusual_moves.empty:
                    st.success("No live move is currently above the 2.0 daily-volatility z-score threshold.")
                else:
                    for _, row in unusual_moves.head(6).iterrows():
                        z_score = row["Z-score of today's move"]
                        direction = "above" if z_score > 0 else "below"
                        st.warning(
                            f"{row['Ticker']} is moving {abs(z_score):.1f} standard deviations "
                            f"{direction} normal daily volatility."
                        )
                st.dataframe(
                    style_numeric_table(
                        risk_overlay,
                        percent_columns=["Change %", "Historical Daily Volatility"],
                        number_columns=[
                            "Live move vs historical daily volatility",
                            "Z-score of today's move",
                        ],
                    ),
                    width="stretch",
                    hide_index=True,
                )

                terminal_band("Search And Filters", "Filter the active snapshot without calling the APIs again.")
                filter_cols = st.columns([1.3, 1.1, 1.1, 1.1, 1.1])
                with filter_cols[0]:
                    search_query = st.text_input("Search ticker", key="live_market_search")
                with filter_cols[1]:
                    move_filter = st.radio(
                        "Move filter",
                        ["All", "Gainers", "Losers"],
                        horizontal=True,
                        key="live_market_move_filter",
                    )
                with filter_cols[2]:
                    minimum_volume = st.number_input(
                        "Minimum volume",
                        min_value=0,
                        value=0,
                        step=10000,
                        key="live_market_minimum_volume",
                    )
                with filter_cols[3]:
                    sort_option = st.selectbox(
                        "Sort by",
                        ["Change %", "Volume", "Volatility", "Ticker"],
                        key="live_market_sort",
                    )
                with filter_cols[4]:
                    holdings_only = st.checkbox(
                        "Show only portfolio holdings",
                        value=True,
                        key="live_market_holdings_only",
                    )

                filtered_quotes = lm_filter_live_market_frame(
                    live_quotes,
                    search_query,
                    move_filter,
                    int(minimum_volume),
                    sort_option,
                    holdings_only,
                    selected_tickers,
                )
                table_columns = [
                    "Ticker",
                    "Last Price",
                    "Change",
                    "Change %",
                    "Open",
                    "High",
                    "Low",
                    "Previous Close",
                    "Day Range %",
                    "Volume",
                    "Signal",
                    "Data Quality",
                ]
                quote_table = filtered_quotes[table_columns].reset_index(drop=True)

                terminal_band("Advanced Live Quote Table", f"{len(quote_table)} visible symbols.")
                if quote_table.empty:
                    st.warning("No symbols match the current Live Market filters.")
                else:
                    st.dataframe(
                        style_live_quote_table(quote_table),
                        width="stretch",
                        hide_index=True,
                    )

                download_cols = st.columns([1, 3])
                with download_cols[0]:
                    st.download_button(
                        "Download Snapshot CSV",
                        data=dataframe_to_csv_bytes(quote_table),
                        file_name=f"live_market_snapshot_{date.today().isoformat()}.csv",
                        mime="text/csv",
                        width="stretch",
                    )
        else:
            st.info("Enable live quote snapshot in the sidebar or press Refresh Now to load market data.")

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
