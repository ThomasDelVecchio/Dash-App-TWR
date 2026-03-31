"""
Portfolio Optimization Page
============================
Interactive Efficient-Frontier optimizer with constraint sliders,
Monte Carlo fan chart, underwater plot, and rolling Sharpe.

All heavy math lives in components/optimization_engine.py.
"""

import dash
from dash import dcc, html, callback, Input, Output, State, ALL, ctx
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import Dict

from components.page_header import page_header
from components.optimization_engine import (
    fetch_optimization_prices,
    compute_efficient_frontier,
    compute_target_volatility_portfolio,
    run_optimization_monte_carlo,
    backtest_optimized_weights,
)
from data_loader import load_holdings
from dash_wrappers import get_data
from portfolio_engine import compute_drawdown_series
from financial_math import is_market_holiday
from components.monte_carlo import run_monte_carlo_simulation
from config import RISK_FREE_RATE, GLOBAL_PALETTE, CLR_ACCENT, CLR_POSITIVE, CLR_NEGATIVE, CLR_NEUTRAL


# ============================================================
# DEFAULTS
# ============================================================

def _get_default_tickers() -> list:
    """Pull active (shares > 0) non-CASH tickers from current holdings."""
    try:
        h = load_holdings()
        active = h[(h["shares"] > 0) & (h["ticker"].str.upper() != "CASH")]
        tickers = sorted(active["ticker"].str.upper().unique().tolist())
        if len(tickers) >= 2:
            return tickers
    except Exception:
        pass
    return ["VOO", "QQQ", "VXUS", "BND", "GLD", "VNQ"]


CHART_TEMPLATE = "plotly_dark"
CHART_BG = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(255,255,255,0.06)"

# Optimization-page-only unification toggles (easy rollback)
OPT_USE_APP_STANDARD_METRICS = True
OPT_USE_APP_STANDARD_MC = True

# ============================================================
# HELPERS
# ============================================================

def _load_target_weight_map() -> dict:
    """Load target weights from holdings as percent values in [0, 100]."""
    try:
        holdings = load_holdings()
        if holdings.empty or "ticker" not in holdings.columns:
            return {}

        holdings = holdings.copy()
        holdings["ticker"] = holdings["ticker"].astype(str).str.upper()
        cols_lower = [str(c).strip().lower() for c in holdings.columns]

        target_col = None
        if "target_pct" in cols_lower:
            target_col = holdings.columns[cols_lower.index("target_pct")]
        else:
            for key in cols_lower:
                if "target" in key and "pct" in key:
                    target_col = holdings.columns[cols_lower.index(key)]
                    break
            if target_col is None:
                for key in cols_lower:
                    if "target" in key:
                        target_col = holdings.columns[cols_lower.index(key)]
                        break

        if target_col is None:
            return {}

        targets = {}
        for _, row in holdings.iterrows():
            ticker = str(row.get("ticker", "")).strip().upper()
            if not ticker or ticker == "CASH":
                continue
            raw_val = row.get(target_col)
            if pd.isna(raw_val):
                continue
            try:
                t = float(raw_val)
            except Exception:
                continue
            if t < 0:
                continue
            if 0 < t <= 1:
                t *= 100.0
            t = min(max(t, 0.0), 100.0)
            targets[ticker] = t

        return targets
    except Exception:
        return {}


_TARGET_WEIGHT_MAP = _load_target_weight_map()

def _empty_fig(msg: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        annotations=[dict(text=msg, xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False,
                          font=dict(size=16, color="#94a3b8"))],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _get_holdings_asset_class_map() -> dict:
    try:
        holdings = load_holdings()
        if holdings.empty:
            return {}
        h = holdings.copy()
        h["ticker"] = h["ticker"].astype(str).str.upper()
        if "asset_class" not in h.columns:
            return {}
        return h.set_index("ticker")["asset_class"].to_dict()
    except Exception:
        return {}


def _build_asset_class_weights(ticker_weights: dict, holdings_map: dict) -> dict:
    ac_weights = {}
    for ticker, weight in (ticker_weights or {}).items():
        if not weight or weight <= 0:
            continue
        ac = holdings_map.get(str(ticker).upper(), "US Large Cap")
        ac_weights[ac] = ac_weights.get(ac, 0.0) + float(weight)

    total = sum(ac_weights.values())
    if total <= 0:
        return {}
    return {ac: w / total for ac, w in ac_weights.items()}


def _calculate_efficiency_metrics_app_standard(twr_series: pd.Series, start_date=None, end_date=None) -> dict:
    """Mirror dash_wrappers.calculate_efficiency_metrics behavior for consistency."""
    default_res = {
        "sharpe": np.nan,
        "sortino": np.nan,
        "vol": np.nan,
        "ret": np.nan,
        "rf": RISK_FREE_RATE,
    }

    if twr_series is None or twr_series.empty or len(twr_series) < 2:
        return default_res

    daily_rets = twr_series.pct_change().dropna()
    if daily_rets.empty:
        return default_res

    if isinstance(daily_rets.index, pd.DatetimeIndex):
        trading_mask = ~daily_rets.index.map(is_market_holiday)
        daily_rets = daily_rets[trading_mask]
        if daily_rets.empty:
            return default_res

    should_annualize = True
    if start_date is not None and end_date is not None:
        days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
        years = days / 365.25
        should_annualize = years > 1.0

    std_dev_daily = daily_rets.std()
    mean_daily_ret = daily_rets.mean()
    rf = RISK_FREE_RATE
    rf_daily = (1 + rf) ** (1 / 252) - 1

    if should_annualize:
        vol = std_dev_daily * np.sqrt(252)
        ret = mean_daily_ret * 252
        rf_used = rf
    else:
        vol = std_dev_daily
        ret = mean_daily_ret
        rf_used = rf_daily

    sharpe = (ret - rf_used) / vol if vol and vol > 0 else np.nan
    return {"sharpe": sharpe, "vol": vol, "ret": ret, "rf": rf_used}


def _compute_underwater_app_standard(growth: pd.Series) -> pd.Series:
    dd_series, _, _ = compute_drawdown_series(growth)
    return dd_series


def _compute_rolling_sharpe_app_standard(
    prices: pd.DataFrame,
    weights: Dict[str, float],
    window_years: int = 3,
) -> pd.Series:
    if prices is None or prices.empty:
        return pd.Series(dtype=float)

    daily_rets = prices.pct_change().dropna()
    tickers = [t for t in weights if t in daily_rets.columns and weights[t] > 0]
    if not tickers:
        return pd.Series(dtype=float)

    w = np.array([weights[t] for t in tickers], dtype=float)
    if w.sum() <= 0:
        return pd.Series(dtype=float)
    w = w / w.sum()

    port_ret = (daily_rets[tickers] * w).sum(axis=1)
    if isinstance(port_ret.index, pd.DatetimeIndex):
        trading_mask = ~port_ret.index.map(is_market_holiday)
        port_ret = port_ret[trading_mask]

    window = window_years * 252
    rolling_mean_ann = port_ret.rolling(window).mean() * 252
    rolling_std_ann = port_ret.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = (rolling_mean_ann - RISK_FREE_RATE) / rolling_std_ann
    return rolling_sharpe.dropna()


def _chart_layout(title: str = "", xlab: str = "", ylab: str = "") -> dict:
    return dict(
        template=CHART_TEMPLATE,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        title=dict(text=title, font=dict(size=14)),
        margin=dict(l=60, r=30, t=40, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11)),
        xaxis=dict(title=xlab, gridcolor=GRID_COLOR,
                   zeroline=False, showgrid=True),
        yaxis=dict(title=ylab, gridcolor=GRID_COLOR,
                   zeroline=False, showgrid=True),
        hoverlabel=dict(
            bgcolor="rgba(15, 23, 42, 0.96)",
            bordercolor="rgba(148, 163, 184, 0.45)",
            font=dict(color="#E2E8F0", size=12),
        ),
        hovermode="x unified",
    )


def _get_current_portfolio_value(default_value: float = 100_000.0) -> float:
    try:
        data = get_data()
        if isinstance(data, dict):
            pv = data.get("pv")
            if pv is not None and len(pv) > 0:
                value = float(pv.iloc[-1])
                if np.isfinite(value) and value > 0:
                    return value
    except Exception:
        pass
    return float(default_value)


# ============================================================
# LAYOUT
# ============================================================

def _ticker_cap_row(ticker: str, index: int) -> dbc.Row:
    """Build a single per-asset min/max input row."""
    target = _TARGET_WEIGHT_MAP.get(ticker)
    if target is None:
        min_default = 0.0
        max_default = 100.0
    else:
        min_default = max(0.0, target - 5.0)
        max_default = min(100.0, target + 5.0)

    ticker_cell = [html.Span(ticker, className="fw-bold")]
    if target is not None:
        ticker_cell.append(
            html.Div(f"Target {target:.1f}%", className="small text-muted")
        )

    return dbc.Row([
        dbc.Col(ticker_cell, width=3),
        dbc.Col(
            dbc.Input(
                id={"type": "opt-ticker-floor", "index": ticker},
                type="number", min=0, max=100, step=1, value=min_default,
                size="sm", className="text-end",
                persistence=True, persistence_type="session",
            ), width=4
        ),
        dbc.Col(
            dbc.Input(
                id={"type": "opt-ticker-cap", "index": ticker},
                type="number", min=0, max=100, step=1, value=max_default,
                size="sm", className="text-end",
                persistence=True, persistence_type="session",
            ), width=5
        ),
    ], className="mb-1 align-items-center")


layout = html.Div([
    page_header(
        title="Portfolio Optimization",
        icon="bi-bullseye",
        subtitle="Efficient Frontier analysis with interactive constraints, Monte Carlo projections & drawdown analytics"
    ),

    # ── Control Panel ──────────────────────────────────────────────
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                # Tickers
                dbc.Col([
                    dbc.Label("ETF Tickers (comma separated)"),
                    dbc.Input(
                        id="opt-tickers",
                        type="text",
                        value=", ".join(_get_default_tickers()),
                        debounce=True,
                        className="mb-2",
                        persistence=True,
                        persistence_type="session",
                    ),
                ], md=8),
                # Lookback
                dbc.Col([
                    dbc.Label("Historical Lookback"),
                    dbc.Select(
                        id="opt-lookback",
                        options=[
                            {"label": "3 Years", "value": "3"},
                            {"label": "5 Years", "value": "5"},
                            {"label": "10 Years", "value": "10"},
                            {"label": "15 Years", "value": "15"},
                        ],
                        value="10",
                        className="mb-2",
                        persistence=True,
                        persistence_type="session",
                    ),
                ], md=4),
            ]),

            dbc.Row([
                dbc.Col([
                    dbc.Label("History Mode"),
                    dbc.RadioItems(
                        id="opt-history-mode",
                        options=[
                            {"label": "Strict overlap", "value": "strict"},
                            {"label": "Proxy-spliced", "value": "proxy"},
                        ],
                        value="strict",
                        inline=True,
                        className="mb-1",
                        persistence=True,
                        persistence_type="session",
                    ),
                    html.Div(
                        "Strict overlap uses only shared live history; Proxy-spliced backfills pre-inception history with replacement proxies.",
                        className="small text-muted",
                    ),
                ], md=12),
            ]),

            html.Hr(className="my-2"),

            dbc.Row([
                # Global Min Weight
                dbc.Col([
                    dbc.Label(id="opt-min-weight-label", children="Min Weight per Asset: 0%"),
                    dcc.Slider(
                        id="opt-min-weight",
                        min=0, max=30, step=1, value=0,
                        marks={0: "0%", 5: "5%", 10: "10%", 20: "20%", 30: "30%"},
                        tooltip={"placement": "bottom", "always_visible": False},
                        persistence=True,
                        persistence_type="session",
                    ),
                ], md=4),
                # Global Max Weight
                dbc.Col([
                    dbc.Label(id="opt-max-weight-label", children="Max Weight per Asset: 100%"),
                    dcc.Slider(
                        id="opt-max-weight",
                        min=10, max=100, step=5, value=100,
                        marks={10: "10%", 25: "25%", 50: "50%", 75: "75%", 100: "100%"},
                        tooltip={"placement": "bottom", "always_visible": False},
                        persistence=True,
                        persistence_type="session",
                    ),
                ], md=4),
                # Target Volatility
                dbc.Col([
                    dbc.Label(id="opt-target-vol-label", children="Target Volatility: 15%"),
                    dcc.Slider(
                        id="opt-target-vol",
                        min=3, max=35, step=1, value=15,
                        marks={5: "5%", 10: "10%", 15: "15%", 20: "20%", 25: "25%", 35: "35%"},
                        tooltip={"placement": "bottom", "always_visible": False},
                        persistence=True,
                        persistence_type="session",
                    ),
                ], md=4),
            ]),

            html.Hr(className="my-2"),

            # Per-asset caps (collapsible)
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        [html.I(className="bi bi-sliders2 me-2"), "Per-Asset Weight Caps"],
                        id="opt-btn-caps-toggle",
                        color="secondary",
                        size="sm",
                    ),
                ], width="auto"),
            ]),

            dbc.Collapse([
                dbc.Row([
                    dbc.Col(html.Span("Ticker", className="small text-muted fw-semibold"), width=3),
                    dbc.Col(html.Span("Min %", className="small text-muted fw-semibold text-end d-block"), width=4),
                    dbc.Col(html.Span("Max %", className="small text-muted fw-semibold text-end d-block"), width=5),
                ], className="mb-1"),
                html.Div(id="opt-ticker-caps-container", className="mt-2"),
                html.P(
                    "Set individual minimum/maximum weights (%) for each ticker. "
                    "These act as per-ticker bounds during optimization.",
                    className="text-muted small mt-1"
                ),
            ], id="opt-caps-collapse", is_open=False, className="mt-2"),

            html.Hr(className="my-2"),

            # Monte Carlo settings
            dbc.Row([
                dbc.Col([
                    dbc.Label("MC Horizon (years)"),
                    dbc.Input(
                        id="opt-mc-horizon", type="number",
                        min=1, max=30, step=1, value=10,
                        size="sm",
                        persistence=True,
                        persistence_type="session",
                    ),
                ], md=2),
                dbc.Col([
                    dbc.Label("MC Simulations"),
                    dbc.Select(
                        id="opt-mc-sims",
                        options=[
                            {"label": "500", "value": "500"},
                            {"label": "1,000", "value": "1000"},
                            {"label": "5,000", "value": "5000"},
                        ],
                        value="1000",
                        className="form-select-sm",
                        persistence=True,
                        persistence_type="session",
                    ),
                ], md=2),
                dbc.Col([
                    dbc.Label("MC Engine"),
                    dbc.Select(
                        id="opt-mc-mode",
                        options=[
                            {"label": "App Standard (Bootstrap)", "value": "app-standard"},
                            {"label": "Optimization GBM", "value": "optimization-gbm"},
                        ],
                        value="app-standard",
                        className="form-select-sm",
                        persistence=True,
                        persistence_type="session",
                    ),
                    html.Div(
                        "App Standard matches Trade Lab bootstrap logic. Optimization GBM is an alternative parametric model and may differ.",
                        className="small text-muted mt-1",
                    ),
                ], md=3),
                dbc.Col([
                    dbc.Label("Monthly Contribution ($)"),
                    dbc.Input(
                        id="opt-mc-contribution", type="number",
                        min=0, step=100, value=0,
                        size="sm",
                        persistence=True,
                        persistence_type="session",
                    ),
                    html.Div(
                        f"MC start value: ${_get_current_portfolio_value():,.0f} (auto)",
                        className="small text-muted mt-1",
                    ),
                ], md=2),
                dbc.Col([
                    dbc.Label(" "),
                    html.Div([
                        dbc.Button(
                            [html.I(className="bi bi-lightning-charge me-1"), "Optimize"],
                            id="opt-btn-run",
                            color="primary",
                            className="w-100",
                        ),
                    ]),
                ], md=2),
            ]),
        ])
    ], className="mb-4"),

    # ── Hidden stores ──────────────────────────────────────────────
    dcc.Store(id="opt-frontier-store"),
    dcc.Store(id="opt-error-store"),

    # ── Status / Error banner ──────────────────────────────────────
    html.Div(id="opt-error-banner"),

    # ── Row 1: Frontier + Weights ──────────────────────────────────
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Efficient Frontier", className="card-title section-header"),
                dcc.Loading(dcc.Graph(id="opt-frontier-chart", figure=_empty_fig("Click ⚡ Optimize to compute the Efficient Frontier"), config={"displayModeBar": False})),
                html.P(
                    "* Dots = individual assets. Star = Max Sharpe. Diamond = Min Volatility. "
                    "Triangle = Target Volatility portfolio.",
                    className="footnote small text-muted px-3"
                ),
            ])
        ]), md=8, className="mb-4"),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Optimal Weights", className="card-title section-header"),
                html.Div(id="opt-sample-info", className="small text-muted mb-2"),
                dcc.Loading(dag.AgGrid(
                    id="opt-weights-grid",
                    columnDefs=[
                        {"headerName": "Ticker", "field": "Ticker", "flex": 1,
                         "pinned": "left", "cellClass": "lock-pinned"},
                        {"headerName": "Max Sharpe", "field": "Max Sharpe",
                         "type": "numericColumn", "flex": 1,
                         "valueFormatter": {"function": "d3.format('.1%')(params.value)"}},
                        {"headerName": "Min Vol", "field": "Min Vol",
                         "type": "numericColumn", "flex": 1,
                         "valueFormatter": {"function": "d3.format('.1%')(params.value)"}},
                        {"headerName": "Target Vol", "field": "Target Vol",
                         "type": "numericColumn", "flex": 1,
                         "valueFormatter": {"function": "d3.format('.1%')(params.value)"}},
                    ],
                    rowData=[],
                    defaultColDef={"resizable": True, "sortable": True},
                    dashGridOptions={"domLayout": "autoHeight"},
                    className="ag-theme-alpine-dark audit-target",
                    style={"width": "100%"},
                )),
                # KPI row beneath weights
                html.Div(id="opt-kpi-row", className="mt-3"),
                html.Hr(className="my-2"),
                dbc.Button(
                    [html.I(className="bi bi-journal-text me-1"), "Proxy Log"],
                    id="opt-btn-proxy-log-toggle",
                    color="secondary",
                    size="sm",
                    className="mb-2",
                ),
                dbc.Collapse([
                    dag.AgGrid(
                        id="opt-proxy-log-grid",
                        columnDefs=[
                            {"headerName": "Ticker", "field": "Ticker", "flex": 1, "minWidth": 100},
                            {"headerName": "Status", "field": "Status", "flex": 1, "minWidth": 120},
                            {"headerName": "Proxy Used", "field": "Proxy Used", "flex": 1.4, "minWidth": 160},
                            {"headerName": "Original Start", "field": "Original Start", "flex": 1, "minWidth": 130},
                            {"headerName": "Proxy Start", "field": "Proxy Start", "flex": 1, "minWidth": 120},
                        ],
                        rowData=[],
                        defaultColDef={"resizable": True, "sortable": True, "filter": True},
                        dashGridOptions={"domLayout": "autoHeight"},
                        className="ag-theme-alpine-dark",
                        style={"width": "100%"},
                    )
                ], id="opt-proxy-log-collapse", is_open=False),
            ])
        ]), md=4, className="mb-4"),
    ]),

    # ── Row 2: Monte Carlo fan chart ───────────────────────────────
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Monte Carlo Wealth Projection (Max Sharpe)", className="card-title section-header"),
                dcc.Loading(dcc.Graph(id="opt-mc-chart", figure=_empty_fig("Run optimization to project Monte Carlo paths"), config={"displayModeBar": False})),
                html.P(
                    "* 1,000 GBM simulated paths based on optimised portfolio's μ and σ. "
                    "Bands show 10th / 25th / 50th / 75th / 90th percentiles.",
                    className="footnote small text-muted px-3"
                ),
            ])
        ]), width=12, className="mb-4"),
    ]),

    # ── Row 3: Underwater + Rolling Sharpe ─────────────────────────
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Underwater Plot (Max Sharpe Weights)", className="card-title section-header"),
                dcc.Loading(dcc.Graph(id="opt-drawdown-chart", figure=_empty_fig("Awaiting optimization"), config={"displayModeBar": False})),
                html.P(
                    "* Historical peak-to-trough drawdowns of the Max Sharpe portfolio.",
                    className="footnote small text-muted px-3"
                ),
            ])
        ]), md=6, className="mb-4"),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Rolling 3-Year Sharpe Ratio", className="card-title section-header"),
                dcc.Loading(dcc.Graph(id="opt-rolling-sharpe-chart", figure=_empty_fig("Awaiting optimization"), config={"displayModeBar": False})),
                html.P(
                    "* 3-year rolling annualised Sharpe across market regimes.",
                    className="footnote small text-muted px-3"
                ),
            ])
        ]), md=6, className="mb-4"),
    ]),
], className="optimization-page")


# ============================================================
# CALLBACKS
# ============================================================

# ── Toggle per-asset caps panel ────────────────────────────────────
@callback(
    Output("opt-caps-collapse", "is_open"),
    Input("opt-btn-caps-toggle", "n_clicks"),
    State("opt-caps-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_caps(n, is_open):
    return not is_open


@callback(
    Output("opt-proxy-log-collapse", "is_open"),
    Input("opt-btn-proxy-log-toggle", "n_clicks"),
    State("opt-proxy-log-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_proxy_log(n, is_open):
    return not is_open


# ── Dynamic slider labels ─────────────────────────────────────────
@callback(
    Output("opt-min-weight-label", "children"),
    Input("opt-min-weight", "value"),
)
def update_min_label(v):
    return f"Min Weight per Asset: {v}%"


@callback(
    Output("opt-max-weight-label", "children"),
    Input("opt-max-weight", "value"),
)
def update_max_label(v):
    return f"Max Weight per Asset: {v}%"


@callback(
    Output("opt-target-vol-label", "children"),
    Input("opt-target-vol", "value"),
)
def update_vol_label(v):
    return f"Target Volatility: {v}%"


# ── Build per-asset cap inputs when tickers change ─────────────────
@callback(
    Output("opt-ticker-caps-container", "children"),
    Input("opt-tickers", "value"),
)
def build_ticker_cap_inputs(raw_tickers: str):
    if not raw_tickers:
        return []
    tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
    return [_ticker_cap_row(t, i) for i, t in enumerate(tickers)]


# ============================================================
# MAIN OPTIMIZATION CALLBACK
# ============================================================

@callback(
    [
        Output("opt-frontier-chart", "figure"),
        Output("opt-weights-grid", "rowData"),
        Output("opt-proxy-log-grid", "rowData"),
        Output("opt-sample-info", "children"),
        Output("opt-kpi-row", "children"),
        Output("opt-mc-chart", "figure"),
        Output("opt-drawdown-chart", "figure"),
        Output("opt-rolling-sharpe-chart", "figure"),
        Output("opt-error-banner", "children"),
    ],
    Input("opt-btn-run", "n_clicks"),
    [
        State("opt-tickers", "value"),
        State("opt-lookback", "value"),
        State("opt-history-mode", "value"),
        State("opt-min-weight", "value"),
        State("opt-max-weight", "value"),
        State("opt-target-vol", "value"),
        State("opt-mc-horizon", "value"),
        State("opt-mc-sims", "value"),
        State("opt-mc-mode", "value"),
        State("opt-mc-contribution", "value"),
        State({"type": "opt-ticker-floor", "index": ALL}, "value"),
        State({"type": "opt-ticker-floor", "index": ALL}, "id"),
        State({"type": "opt-ticker-cap", "index": ALL}, "value"),
        State({"type": "opt-ticker-cap", "index": ALL}, "id"),
    ],
    prevent_initial_call=True,
)
def run_optimization(
    n_clicks,
    raw_tickers, lookback, history_mode,
    min_wt, max_wt, target_vol,
    mc_horizon, mc_sims, mc_mode, mc_contribution,
    floor_values, floor_ids,
    cap_values, cap_ids,
):
    empty = _empty_fig("Click Optimize to begin")
    blank = [empty, [], [], "", [], empty, empty, empty, ""]
    if not n_clicks:
        return blank

    # ── Parse inputs ──────────────────────────────────────────────
    try:
        tickers = [t.strip().upper() for t in (raw_tickers or "").split(",") if t.strip()]
        # Remove CASH — can't optimise a cash position
        tickers = [t for t in tickers if t != "CASH"]
        if len(tickers) < 2:
            raise ValueError("Provide at least 2 non-CASH tickers.")

        years = int(lookback or 10)
        min_w = (min_wt or 0) / 100
        max_w = (max_wt or 100) / 100
        t_vol = (target_vol or 15) / 100
        horizon = int(mc_horizon or 10)
        sims = int(mc_sims or 1000)
        contrib = float(mc_contribution or 0)

        # Per-asset floor/cap bounds from pattern-matching inputs
        ticker_floors = {}
        ticker_caps = {}
        if floor_ids and floor_values:
            for fid, fval in zip(floor_ids, floor_values):
                tk = fid["index"]
                if fval is not None and float(fval) > 0:
                    ticker_floors[tk] = float(fval) / 100

        if cap_ids and cap_values:
            for cid, cval in zip(cap_ids, cap_values):
                tk = cid["index"]
                if cval is not None and float(cval) < 100:
                    ticker_caps[tk] = float(cval) / 100

        for tk in tickers:
            lo = ticker_floors.get(tk, min_w)
            hi = ticker_caps.get(tk, max_w)
            if lo > hi:
                raise ValueError(
                    f"Infeasible bounds for {tk}: min {lo*100:.1f}% exceeds max {hi*100:.1f}%"
                )
    except Exception as e:
        err = dbc.Alert(str(e), color="warning", dismissable=True)
        return [empty, [], [], "", [], empty, empty, empty, err]

    # ── Fetch prices ──────────────────────────────────────────────
    try:
        use_proxy_splice = str(history_mode or "strict").lower() == "proxy"
        prices = fetch_optimization_prices(
            tickers,
            years_back=years,
            use_proxy_splice=use_proxy_splice,
        )
        price_meta = dict(prices.attrs) if hasattr(prices, "attrs") else {}
        # Keep only tickers that survived the fetch
        available = [t for t in tickers if t in prices.columns]
        if len(available) < 2:
            missing = [t for t in tickers if t not in prices.columns]
            raise ValueError(
                f"Only {len(available)} ticker(s) returned price data (need ≥ 2). "
                f"Missing: {missing}"
            )
        prices = prices[available]
        prices.attrs = price_meta
    except Exception as e:
        print(f"[OPT] Price fetch error: {e}")
        err = dbc.Alert(f"Price fetch error: {e}", color="danger", dismissable=True)
        return [empty, [], [], "", [], empty, empty, empty, err]

    # ── Compute Frontier ──────────────────────────────────────────
    try:
        result = compute_efficient_frontier(
            prices,
            weight_bounds=(min_w, max_w),
            ticker_floors=ticker_floors,
            ticker_caps=ticker_caps,
            risk_free_rate=RISK_FREE_RATE,
        )
    except Exception as e:
        print(f"[OPT] Optimization error: {e}")
        import traceback; traceback.print_exc()
        err = dbc.Alert(f"Optimization error: {e}", color="danger", dismissable=True)
        return [empty, [], [], "", [], empty, empty, empty, err]

    sample_start = pd.Timestamp(prices.index.min()).strftime("%Y-%m-%d")
    sample_end = pd.Timestamp(prices.index.max()).strftime("%Y-%m-%d")
    n_obs = int(len(prices))
    date_span_days = max((pd.Timestamp(prices.index.max()) - pd.Timestamp(prices.index.min())).days, 0)
    sample_years = date_span_days / 365.25 if date_span_days else 0.0
    sample_text = (
        f"Effective sample window: {sample_start} -> {sample_end} "
        f"({n_obs} trading days, ~{sample_years:.1f} years)"
    )

    sample_info_children = [html.Div(sample_text)]
    proxy_log = prices.attrs.get("proxy_log", []) if hasattr(prices, "attrs") else []
    proxy_log_rows = proxy_log if isinstance(proxy_log, list) else []
    spliced_rows = [r for r in proxy_log if r.get("Status") in {"Spliced", "Partial History"}]
    if spliced_rows:
        tickers_spliced = ", ".join(sorted({r.get("Ticker") for r in spliced_rows if r.get("Ticker")}))
        sample_info_children.append(
            html.Div(
                [
                    html.I(className="bi bi-exclamation-triangle me-1"),
                    "Proxy-spliced history applied for: ",
                    html.Span(tickers_spliced, className="fw-semibold"),
                    ". Results use synthetic pre-inception returns from replacement proxies.",
                ],
                className="text-warning mt-1",
            )
        )
    elif str(history_mode or "strict").lower() == "strict":
        sample_info_children.append(
            html.Div(
                [
                    html.I(className="bi bi-shield-check me-1"),
                    "Strict overlap mode: no synthetic proxy history used.",
                ],
                className="text-success mt-1",
            )
        )
    sample_info = html.Div(sample_info_children)

    max_sharpe = result["max_sharpe"]
    min_vol = result["min_vol"]
    mu = result["mu"]
    cov = result["cov"]

    # ── Target-Vol portfolio ──────────────────────────────────────
    target_vol_port = compute_target_volatility_portfolio(
        mu, cov, t_vol,
        weight_bounds=(min_w, max_w),
        ticker_floors=ticker_floors,
        ticker_caps=ticker_caps,
    )

    # App-standard realized metrics (for consistency with other pages)
    realized_metrics = {}
    if OPT_USE_APP_STANDARD_METRICS:
        for name, port in [("max_sharpe", max_sharpe), ("min_vol", min_vol), ("target_vol", target_vol_port)]:
            if not port:
                continue
            growth_series = backtest_optimized_weights(prices, port.get("weights", {}))
            eff = _calculate_efficiency_metrics_app_standard(
                growth_series,
                start_date=prices.index.min(),
                end_date=prices.index.max(),
            )
            realized_metrics[name] = eff

    # ==========================================================
    # 1. EFFICIENT FRONTIER CHART
    # ==========================================================
    fig_frontier = go.Figure()

    # Frontier curve
    if result["frontier_vols"]:
        fig_frontier.add_trace(go.Scatter(
            x=[v * 100 for v in result["frontier_vols"]],
            y=[r * 100 for r in result["frontier_rets"]],
            mode="lines",
            name="Efficient Frontier",
            line=dict(color=CLR_ACCENT, width=3),
            hovertemplate="Vol: %{x:.1f}%<br>Ret: %{y:.1f}%<extra></extra>",
        ))

    # Individual assets
    for a in result["individual"]:
        fig_frontier.add_trace(go.Scatter(
            x=[a["vol"] * 100], y=[a["ret"] * 100],
            mode="markers+text",
            name=a["ticker"],
            marker=dict(size=10, color=CLR_NEUTRAL, symbol="circle"),
            text=[a["ticker"]], textposition="top center",
            textfont=dict(size=10),
            hovertemplate="Vol: %{x:.1f}%<br>Ret: %{y:.1f}%<extra></extra>",
        ))

    # Max Sharpe star
    if max_sharpe:
        sharpe_label = max_sharpe['sharpe']
        if OPT_USE_APP_STANDARD_METRICS and "max_sharpe" in realized_metrics and pd.notna(realized_metrics["max_sharpe"]["sharpe"]):
            sharpe_label = float(realized_metrics["max_sharpe"]["sharpe"])
        fig_frontier.add_trace(go.Scatter(
            x=[max_sharpe["vol"] * 100], y=[max_sharpe["ret"] * 100],
            mode="markers",
            name=f"Max Sharpe ({sharpe_label:.2f})",
            marker=dict(size=18, color=CLR_POSITIVE, symbol="star",
                        line=dict(width=1, color="white")),
            hovertemplate=(
                "Vol: %{x:.1f}%<br>Ret: %{y:.1f}%"
                f"<br>Sharpe: {sharpe_label:.2f}<extra></extra>"
            ),
        ))

    # Min Volatility diamond
    if min_vol:
        sharpe_label = min_vol['sharpe']
        if OPT_USE_APP_STANDARD_METRICS and "min_vol" in realized_metrics and pd.notna(realized_metrics["min_vol"]["sharpe"]):
            sharpe_label = float(realized_metrics["min_vol"]["sharpe"])
        fig_frontier.add_trace(go.Scatter(
            x=[min_vol["vol"] * 100], y=[min_vol["ret"] * 100],
            mode="markers",
            name=f"Min Volatility",
            marker=dict(size=14, color="#8B5CF6", symbol="diamond",
                        line=dict(width=1, color="white")),
            hovertemplate=(
                "Vol: %{x:.1f}%<br>Ret: %{y:.1f}%"
                f"<br>Sharpe: {sharpe_label:.2f}<extra></extra>"
            ),
        ))

    # Target-Vol triangle
    if target_vol_port:
        sharpe_label = target_vol_port['sharpe']
        if OPT_USE_APP_STANDARD_METRICS and "target_vol" in realized_metrics and pd.notna(realized_metrics["target_vol"]["sharpe"]):
            sharpe_label = float(realized_metrics["target_vol"]["sharpe"])
        fig_frontier.add_trace(go.Scatter(
            x=[target_vol_port["vol"] * 100], y=[target_vol_port["ret"] * 100],
            mode="markers",
            name=f"Target Vol ({t_vol*100:.0f}%)",
            marker=dict(size=14, color="#F59E0B", symbol="triangle-up",
                        line=dict(width=1, color="white")),
            hovertemplate=(
                "Vol: %{x:.1f}%<br>Ret: %{y:.1f}%"
                f"<br>Sharpe: {sharpe_label:.2f}<extra></extra>"
            ),
        ))

    fig_frontier.update_layout(
        **_chart_layout(xlab="Annualised Volatility (%)", ylab="Annualised Return (%)"),
    )
    fig_frontier.update_layout(hovermode="closest")
    fig_frontier.update_xaxes(hoverformat=".1f")
    fig_frontier.update_yaxes(hoverformat=".1f")

    # ==========================================================
    # 2. WEIGHTS TABLE
    # ==========================================================
    all_tickers = sorted(set(
        list((max_sharpe or {}).get("weights", {}).keys()) +
        list((min_vol or {}).get("weights", {}).keys()) +
        list((target_vol_port or {}).get("weights", {}).keys())
    ))
    rows = []
    for t in all_tickers:
        rows.append({
            "Ticker": t,
            "Max Sharpe": (max_sharpe["weights"].get(t, 0) if max_sharpe else 0),
            "Min Vol": (min_vol["weights"].get(t, 0) if min_vol else 0),
            "Target Vol": (target_vol_port["weights"].get(t, 0) if target_vol_port else 0),
        })

    # ==========================================================
    # 3. KPI CARDS
    # ==========================================================
    kpi_items = []
    for label, port, color in [
        ("Max Sharpe", max_sharpe, CLR_POSITIVE),
        ("Min Vol", min_vol, "#8B5CF6"),
        ("Target Vol", target_vol_port, "#F59E0B"),
    ]:
        if port:
            key = "max_sharpe" if label == "Max Sharpe" else ("min_vol" if label == "Min Vol" else "target_vol")
            kpi_ret = port['ret']
            kpi_vol = port['vol']
            kpi_sharpe = port['sharpe']
            if OPT_USE_APP_STANDARD_METRICS and key in realized_metrics:
                eff = realized_metrics[key]
                if pd.notna(eff.get("ret", np.nan)):
                    kpi_ret = float(eff["ret"])
                if pd.notna(eff.get("vol", np.nan)):
                    kpi_vol = float(eff["vol"])
                if pd.notna(eff.get("sharpe", np.nan)):
                    kpi_sharpe = float(eff["sharpe"])
            kpi_items.append(
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.P(label, className="small text-muted mb-1"),
                        html.H5(f"{kpi_ret*100:.1f}% ret", style={"color": color}),
                        html.P(f"{kpi_vol*100:.1f}% vol  |  Sharpe {kpi_sharpe:.2f}",
                               className="small mb-0"),
                    ], className="p-2 text-center"),
                ], className="border-0"), md=4)
            )
    kpis = dbc.Row(kpi_items) if kpi_items else html.Div()

    # ==========================================================
    # 4. MONTE CARLO FAN CHART
    # ==========================================================
    if max_sharpe:
        capital = _get_current_portfolio_value()
        mc_engine = str(mc_mode or "app-standard")
        use_app_mc = OPT_USE_APP_STANDARD_MC and mc_engine == "app-standard"

        if use_app_mc:
            holdings_map = _get_holdings_asset_class_map()
            ac_weights = _build_asset_class_weights(max_sharpe["weights"], holdings_map)
            mc = run_monte_carlo_simulation(
                current_value=capital,
                weights=ac_weights,
                horizon_years=horizon,
                n_simulations=sims,
                monthly_contribution=contrib,
                prices_df=prices,
                random_seed=42,
                ticker_weights=max_sharpe["weights"],
                holdings_map=holdings_map,
            )
        else:
            mc = run_optimization_monte_carlo(
                mu=mu, cov=cov,
                weights=max_sharpe["weights"],
                initial_value=capital,
                horizon_years=horizon,
                n_simulations=sims,
                monthly_contribution=contrib,
            )

        fig_mc = _build_fan_chart(mc, capital)
    else:
        fig_mc = _empty_fig("Max Sharpe portfolio could not be computed")

    # ==========================================================
    # 5. UNDERWATER PLOT
    # ==========================================================
    if max_sharpe:
        growth = backtest_optimized_weights(prices, max_sharpe["weights"])
        dd = _compute_underwater_app_standard(growth)
        fig_dd = _build_underwater(dd)
    else:
        fig_dd = _empty_fig("No drawdown data")

    # ==========================================================
    # 6. ROLLING SHARPE
    # ==========================================================
    if max_sharpe:
        rolling = _compute_rolling_sharpe_app_standard(prices, max_sharpe["weights"], window_years=3)
        fig_rs = _build_rolling_sharpe(rolling)
    else:
        fig_rs = _empty_fig("No rolling Sharpe data")

    return [fig_frontier, rows, proxy_log_rows, sample_info, kpis, fig_mc, fig_dd, fig_rs, ""]


# ============================================================
# FIGURE BUILDERS
# ============================================================

def _build_fan_chart(mc: dict, initial: float) -> go.Figure:
    """Build the Monte Carlo percentile fan chart."""
    if not mc:
        return _empty_fig("Monte Carlo simulation unavailable")

    years = mc["years"]
    pcts = mc["percentiles"]
    fig = go.Figure()

    # 10-90 band
    fig.add_trace(go.Scatter(
        x=years + years[::-1],
        y=pcts["90"] + pcts["10"][::-1],
        fill="toself",
        fillcolor="rgba(0,212,255,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="10th – 90th pctl",
        hoverinfo="skip",
    ))

    # 25-75 band
    if "75" in pcts and "25" in pcts:
        fig.add_trace(go.Scatter(
            x=years + years[::-1],
            y=pcts["75"] + pcts["25"][::-1],
            fill="toself",
            fillcolor="rgba(0,212,255,0.22)",
            line=dict(color="rgba(0,0,0,0)"),
            name="25th – 75th pctl",
            hoverinfo="skip",
        ))

    # Median line
    fig.add_trace(go.Scatter(
        x=years, y=pcts["50"],
        mode="lines",
        name="Median (50th)",
        line=dict(color=CLR_ACCENT, width=2.5),
        hovertemplate="Year %{x:.0f}: $%{y:,.0f}<extra></extra>",
    ))

    # Starting value reference
    fig.add_hline(
        y=initial, line_dash="dot",
        line_color=CLR_NEUTRAL, opacity=0.5,
        annotation_text=f"Start ${initial:,.0f}",
        annotation_position="bottom right",
    )

    metrics = mc.get("metrics", {})
    title_extra = ""
    if metrics:
        median_final = metrics.get("median_final")
        if median_final is None and mc.get("final_distribution"):
            try:
                median_final = float(np.median(mc.get("final_distribution", [])))
            except Exception:
                median_final = None

        title_extra = (
            f"  (μ={metrics.get('mu', 0.0)*100:.1f}%, σ={metrics.get('sigma', 0.0)*100:.1f}%"
        )
        if median_final is not None:
            title_extra += f", Median Final=${median_final:,.0f}"
        title_extra += ")"

    fig.update_layout(**_chart_layout(xlab="Year", ylab="Portfolio Value ($)"))
    fig.update_layout(title=dict(text=f"Monte Carlo Projection{title_extra}", font=dict(size=13)))
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(hoverformat=".1f")
    fig.update_yaxes(tickprefix="$", tickformat=",")
    return fig


def _build_underwater(dd: pd.Series) -> go.Figure:
    """Underwater (drawdown) area chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        fill="tozeroy",
        mode="lines",
        line=dict(color=CLR_NEGATIVE, width=1),
        fillcolor="rgba(239,68,68,0.25)",
        name="Drawdown",
        hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(**_chart_layout(xlab="", ylab="Drawdown (%)"))
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(ticksuffix="%")
    return fig


def _build_rolling_sharpe(rs: pd.Series) -> go.Figure:
    """Rolling Sharpe line chart with zero reference."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rs.index, y=rs.values,
        mode="lines",
        line=dict(color=CLR_ACCENT, width=1.5),
        name="Rolling Sharpe",
        hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=CLR_NEUTRAL, opacity=0.4)
    fig.update_layout(**_chart_layout(xlab="", ylab="Sharpe Ratio"))
    fig.update_layout(hovermode="closest")
    return fig
