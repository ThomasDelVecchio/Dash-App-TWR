"""
Custom Report Builder Page
==========================
Institutional-quality report generator with Print Preview (Light Mode) toggle.
Allows configuring sections and generating print-ready PDF reports.

Key Features:
- "Print Preview" toggle switches charts to plotly_white template
- A4-proportion report container for consistent PDF output
- Ghost footer with disclaimers (hidden on screen, shown in print)
- Page break classes for multi-page PDF generation
"""

import dash
from dash import dcc, html, callback, Input, Output, State, ALL, clientside_callback
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import dash_wrappers as dw
from dash_wrappers import send_to_drive
from components.page_header import page_header
from report_formatting import fmt_pct_clean, fmt_dollar_clean, generate_word_report
from components.ai_brief import generate_ai_summary_period
from datetime import datetime
import pandas as pd
from tax_engine import build_tax_lots, normalize_ticker
from data_loader import load_transactions_raw, fetch_price_history
from financial_math import get_portfolio_horizon_start, modified_dietz_for_ticker_window, annualize_return, compute_horizon_twr
from portfolio_engine import calculate_horizon_pl
from config import TAX_RATE_ST, TAX_RATE_LT, GLOBAL_PALETTE
import numpy as np
from io import BytesIO

# ============================================================
# CONSTANTS
# ============================================================
REPORT_SECTIONS_OPTIONS = [
    {"label": "01. Morning AI Summary", "value": "morning_brief"},
    {"label": "02. Executive Summary (KPIs)", "value": "summary"},
    {"label": "03. Performance Chart", "value": "performance_chart"},
    {"label": "04. Attribution Analysis (Waterfall)", "value": "attribution"},
    {"label": "05. Asset Class Perf. Table", "value": "ac_perf_table"},
    {"label": "06. Asset Class P/L Table", "value": "ac_pl_table"},
    {"label": "07. Performance Deep Dive", "value": "perf_deep_dive"},
    {"label": "09. Horizon Analysis Table", "value": "horizon_table"},
    {"label": "10. Asset Allocation", "value": "allocation"},
    {"label": "11. Sector Breakdown", "value": "sector"},
    {"label": "12. Top Holdings", "value": "holdings"},
    {"label": "13. Flows Report (Enhanced)", "value": "flows"},
    {"label": "14. Risk Analysis Charts", "value": "risk_charts"},
    {"label": "15. Tax Lot Explorer", "value": "tax_lots"},
]

# User-Defined Default Selection
DEFAULT_SELECTION = [
    "morning_brief",
    "summary",
    "performance_chart",
    "attribution",
    "ac_perf_table",
    "ac_pl_table",
    "perf_deep_dive",
    "horizon_table",
    "allocation"
]

DEFAULT_ORDER = [opt["value"] for opt in REPORT_SECTIONS_OPTIONS]


# ============================================================
# HORIZON FILTERING HELPER
# ============================================================
def get_valid_horizons_for_period(start_date, end_date, full_horizons=None, selected_period=None):
    """
    Determines which horizon columns should be displayed based on the reporting period.
    
    Logic:
    - Calculates duration in days from start_date to end_date
    - Only includes horizons where the period_days >= threshold
    - SI (Since Inception) is ALWAYS included regardless of period
    - The selected_period horizon is ALWAYS included (user explicitly requested it)
    
    Args:
        start_date: Period start date (pd.Timestamp or datetime)
        end_date: Period end date (pd.Timestamp or datetime)
        full_horizons: List of all possible horizons (default uses standard set)
        selected_period: The user's selected reporting period (e.g., "YTD", "3M", "1M")
                        This horizon will always be included regardless of thresholds.
    
    Returns:
        List of valid horizon strings to display
    """
    if full_horizons is None:
        full_horizons = ["1D", "1W", "MTD", "1M", "3M", "6M", "YTD", "1Y", "SI"]
    
    # SI always displays
    if start_date is None or end_date is None:
        return full_horizons
    
    # Calculate period duration
    period_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
    
    # Threshold mapping: minimum days needed for each horizon to be meaningful
    thresholds = {
        "1D": 1,
        "1W": 7,
        "MTD": 1,    # MTD can show for any period (resets monthly)
        "1M": 28,
        "3M": 84,
        "6M": 168,
        "YTD": 180,  # Roughly 6 months into the year
        "1Y": 350,
    }
    
    # Map selected_period to horizon code (handle dropdown values like "12M" -> "1Y")
    period_to_horizon = {
        "SI": "SI",
        "YTD": "YTD",
        "12M": "1Y",
        "3M": "3M",
        "1M": "1M",
    }
    forced_horizon = period_to_horizon.get(selected_period) if selected_period else None
    
    valid = []
    for h in full_horizons:
        if h == "SI":
            # SI always included
            valid.append(h)
        elif forced_horizon and h == forced_horizon:
            # User's selected period horizon is always included
            valid.append(h)
        elif h in thresholds:
            if period_days >= thresholds[h]:
                valid.append(h)
        else:
            # Unknown horizon - include by default
            valid.append(h)
    
    return valid


# ============================================================
# LOCAL REPORTING LOGIC (ISOLATED)
# ============================================================

# --- Local Helpers for Tax Logic ---
# (Removed: Now using tax_engine.build_tax_lots with as_of_date support)

# --- Local Helpers for Flows Logic ---
def _get_flows_summary_period(data, start_date=None, end_date=None):
    """
    Local implementation of flows summary that respects specific period.
    Replaces get_flows_summary_ytd logic.
    """
    # NOTE: 'data' dict usually contains 'pv', 'cf_ext', 'tx_raw', 'dividends'
    
    pv = data.get("pv")
    cf_ext = data.get("cf_ext")
    tx_raw = data.get("tx_raw")
    dividends = data.get("dividends")
    
    if pv is None or pv.empty: 
        return pd.DataFrame(), f"N/A thru N/A"

    # Determine Bounds
    # If no start_date provided, assume Inception ("SI")
    # If no end_date provided, assume As Of "Now"
    
    max_date = pv.index.max()
    eff_end = pd.Timestamp(end_date) if end_date else max_date
    eff_start = pd.Timestamp(start_date) if start_date else pv.index.min()
    
    period_label = f"{eff_start.strftime('%Y-%m-%d')} thru {eff_end.strftime('%Y-%m-%d')}"
    if not start_date: 
        period_label = f"Inception thru {eff_end.strftime('%Y-%m-%d')}"

    # Filter Data
    def filter_df(df, date_col="date"):
        if df is None or df.empty: return df
        mask = (df[date_col] >= eff_start) & (df[date_col] <= eff_end)
        return df[mask]

    flows_ext = filter_df(cf_ext)
    tx_period = filter_df(tx_raw)
    div_period = filter_df(dividends)
    
    # 1. External Flows
    deposits = 0.0
    withdrawals = 0.0
    net_ext = 0.0
    most_recent_ext = None
    
    if flows_ext is not None and not flows_ext.empty:
        deposits = flows_ext.loc[flows_ext["amount"] > 0, "amount"].sum()
        withdrawals = flows_ext.loc[flows_ext["amount"] < 0, "amount"].sum()
        net_ext = flows_ext["amount"].sum()
        most_recent_ext = flows_ext["date"].max()
        
    # 2. Internal Activity
    buys = 0.0
    sells = 0.0
    most_recent_tx = None
    
    if tx_period is not None and not tx_period.empty:
        buys = tx_period.loc[tx_period["amount"] < 0, "amount"].sum()
        sells = tx_period.loc[tx_period["amount"] > 0, "amount"].sum()
        most_recent_tx = tx_period["date"].max()
        
    # 3. Income
    income = 0.0
    most_recent_div = None
    if div_period is not None and not div_period.empty:
        income = div_period["amount"].sum()
        most_recent_div = div_period["date"].max()
        
    net_internal = buys + sells + income
    
    # Recent Date
    dates = [d for d in [most_recent_ext, most_recent_tx, most_recent_div] if d is not pd.NaT and d is not None]
    most_recent_any = max(dates).strftime("%Y-%m-%d") if dates else "N/A"

    rows = [
        {"Metric": "Net External Flows", "Value": fmt_dollar_clean(net_ext)},
        {"Metric": "• Deposits", "Value": fmt_dollar_clean(deposits)},
        {"Metric": "• Withdrawals", "Value": fmt_dollar_clean(withdrawals)},
        {"Metric": "Net Internal Activity", "Value": fmt_dollar_clean(net_internal)},
        {"Metric": "• Buys (Cash Out)", "Value": fmt_dollar_clean(buys)},
        {"Metric": "• Sells (Cash In)", "Value": fmt_dollar_clean(sells)},
        {"Metric": "• Income (Divs)", "Value": fmt_dollar_clean(income)},
        {"Metric": "Most Recent Flow", "Value": most_recent_any},
    ]
    return pd.DataFrame(rows), period_label

# ============================================================
# HELPER: Apply Light Theme to Figure
# ============================================================
def apply_print_theme(fig):
    """
    Deprecated. Returns figure as-is.
    """
    return fig


# ============================================================
# LAYOUT
# ============================================================
layout = html.Div(className="custom-report-page", children=[
    # Page Header
    page_header(
        title="Custom Report Builder",
        icon="bi-file-earmark-text",
        subtitle="Configure and generate institutional-quality PDF reports"
    ),
    
    # Report Configuration Accordion
    dbc.Accordion([
        dbc.AccordionItem([
            dbc.Row([
                # Column 1: General Settings
                dbc.Col([
                    html.H5("General Settings", className="mb-3 text-body"),
                    dbc.Label("Report Title", className="fw-bold"),
                    dbc.Input(
                        id="report-title-input",
                        type="text",
                        value="Portfolio Performance Report",
                        placeholder="Enter report title...",
                        className="mb-3",
                        persistence=True,
                        persistence_type="session"
                    ),
                    dbc.Label("Reporting Period", className="fw-bold"),
                    dbc.Select(
                        id="report-period-select",
                        options=[
                            {"label": "Since Inception", "value": "SI"},
                            {"label": "Year-to-Date", "value": "YTD"},
                            {"label": "Last 12 Months", "value": "12M"},
                            {"label": "Last Quarter", "value": "3M"},
                            {"label": "Last Month", "value": "1M"},
                        ],
                        value="SI",
                        className="mb-3 text-dark",
                        persistence=True,
                        persistence_type="session"
                    ),
                    dbc.Switch(
                        id="report-mobile-mode",
                        label="One Chart Per Page (Mobile)",
                        value=False,
                        className="mb-2",
                        persistence=True,
                        persistence_type="session"
                    ),
                    dbc.Switch(
                        id="report-include-exited",
                        label="Include Exited Holdings",
                        value=False,
                        className="mb-2",
                        persistence=True,
                        persistence_type="session"
                    ),
                ], width=12, md=4),
                
                # Column 2: Section Selection
                dbc.Col([
                    html.Div([
                        html.H5("Report Sections", className="d-inline-block me-3 text-body"),
                        dbc.Button(
                            "Select All",
                            id="toggle-all-sections",
                            color="link",
                            size="sm",
                            className="p-0 text-decoration-none ai-text-accent",
                        )
                    ], className="mb-2"),
                    
                    # Stores for Order and Selection
                    dcc.Store(id="report-order-store", storage_type="session", data=DEFAULT_ORDER),
                    dcc.Store(id="report-selection-store", storage_type="session", data=DEFAULT_SELECTION),

                    # Draggable Container
                    html.Div(
                        id="report-sections-container",
                        className="list-group report-sections-list",
                        style={"maxHeight": "400px", "overflowY": "auto", "overflowX": "hidden"}
                    ),
                    html.Div(
                        [html.I(className="bi bi-info-circle me-1"), "Drag handles (☰) to reorder sections"], 
                        className="text-muted small mt-2 text-end"
                    )
                ], width=12, md=4),
                
                # Column 3: Actions
                dbc.Col([
                    html.H5("Actions", className="mb-3 text-body"),
                    dbc.Button(
                        [html.I(className="bi bi-arrow-clockwise me-2"), "Refresh Preview"],
                        id="btn-refresh-report",
                        color="primary",
                        className="w-100 mb-2"
                    ),
                    dbc.Alert(
                        id="download-status-alert",
                        is_open=False,
                        dismissable=True,
                        className="mb-2 small"
                    ),
                    dbc.Button(
                        [html.I(className="bi bi-file-earmark-word me-2"), "Download Word Report"],
                        id="btn-download-word",
                        color="success",
                        className="w-100 mb-2",
                        n_clicks=0
                    ),
                    dbc.Button(
                        [html.I(className="bi bi-google me-2"), "Export to Drive"],
                        id="btn-export-drive",
                        color="warning",
                        className="w-100",
                        n_clicks=0
                    ),
                    html.Div(id="drive-export-feedback", className="mt-2 text-center small fw-bold"),
                    dcc.Download(id="download-word-report")
                ], width=12, md=4),
            ])
        ], title="Report Configuration")
    ], className="mb-4 d-print-none", start_collapsed=False),

    dbc.Row([
        # Right Panel: Report Preview
        dbc.Col([
            # Report Container (A4 proportions)
            html.Div([
                # Report Header
                html.Div([
                    html.Div([
                        html.H1(id="report-header-title", className="report-title mb-1"),
                        html.P(id="report-header-subtitle", className="report-subtitle text-muted mb-0")
                    ], className="report-header-content"),
                    html.Div([
                        html.Img(
                            src="/assets/logo.png",
                            className="report-logo",
                            style={"height": "50px", "display": "none"}  # Hidden by default, enable if logo exists
                        )
                    ], className="report-logo-container")
                ], className="report-header d-flex justify-content-between align-items-center mb-4 pb-3 border-bottom"),
                
                # Report Date
                html.P(id="report-date-line", className="text-end text-muted small mb-4"),
                
                # Dynamic Report Content Container
                html.Div(id="report-content-container"),
                
                # Ghost Footer (Hidden on screen, visible in print)
                html.Div([
                    html.Hr(className="mt-4"),
                    html.P([
                        html.Strong("Important Disclosures: "),
                        "Past performance is not indicative of future results. Investment returns and principal value will fluctuate. ",
                        "This report is for informational purposes only and does not constitute investment advice. ",
                        "Data is sourced from third-party providers and may be delayed or inaccurate. ",
                        "Please consult a qualified financial advisor before making investment decisions."
                    ], className="small text-muted mb-2"),
                    html.P([
                        "Generated by DELVEX Portfolio Analytics | ",
                        html.Span(id="report-footer-timestamp")
                    ], className="small text-muted text-center")
                ], className="print-footer", id="report-ghost-footer")
                
            ], id="report-container", className="report-container")
        ], width=12, className="report-preview-col")
    ]),
    
    # Hidden Print Trigger Script
    dcc.Store(id="print-trigger-store")
])


# ============================================================
# CALLBACKS
# ============================================================

# 1. Initialize SortableJS
clientside_callback(
    """
    function(trigger) {
        return window.dash_clientside.report_sorting.enable_sortable(trigger);
    }
    """,
    Output("report-sections-container", "data-sortable-active"), # Dummy output
    Input("report-sections-container", "children")
)

# 2. Update Selection Store (User clicks Checkbox)
@callback(
    Output("report-selection-store", "data", allow_duplicate=True),
    Input({"type": "section-selector", "index": ALL}, "value"),
    State({"type": "section-selector", "index": ALL}, "id"),
    prevent_initial_call=True
)
def update_selection(values, ids):
    # Filter for True values and extract index from id
    selected = [id_dict['index'] for val, id_dict in zip(values, ids) if val]
    return selected

# 3. Render Draggable List
@callback(
    Output("report-sections-container", "children"),
    [Input("report-order-store", "data"),
     Input("report-selection-store", "data")]
)
def render_draggable_list(order_list, selected_list):
    order_list = order_list or DEFAULT_ORDER
    selected_list = selected_list or []
    
    # Map values to labels
    label_map = {opt["value"]: opt["label"] for opt in REPORT_SECTIONS_OPTIONS}
    
    # Ensure all options are present logic
    current_values = set(order_list)
    options_dict = {opt["value"]: opt for opt in REPORT_SECTIONS_OPTIONS}
    
    # Filter order list to remove obsolete keys
    valid_order = [k for k in order_list if k in options_dict]
    
    # Find missing keys
    missing = [k for k in options_dict if k not in current_values]
    
    final_order = valid_order + missing
    
    children = []
    for val in final_order:
        is_selected = val in selected_list
        label = label_map.get(val, val)
        
        item = html.Div([
            # Drag Handle
            html.Span(
                html.I(className="bi bi-grip-vertical"),
                className="drag-handle me-3 p-2",
                style={"cursor": "grab", "color": "#000000", "fontSize": "1.2rem"}
            ),
            # Label
            html.Div(label, className="flex-grow-1 user-select-none"),
            # Checkbox
            dbc.Checkbox(
                id={"type": "section-selector", "index": val},
                value=is_selected,
                className="form-check-input",
                style={"cursor": "pointer"}
            )
        ], className="list-group-item d-flex align-items-center bg-dark text-white border-secondary mb-1", key=val, **{"data-value": val})
        children.append(item)
        
    return children

# 4. Toggle All
@callback(
    Output("report-selection-store", "data", allow_duplicate=True),
    Input("toggle-all-sections", "n_clicks"),
    State("report-selection-store", "data"),
    prevent_initial_call=True
)
def toggle_all_report_sections(n_clicks, current_selection):
    if n_clicks is None:
        return dash.no_update
    
    all_values = [opt["value"] for opt in REPORT_SECTIONS_OPTIONS]
    current_selection = current_selection or []
    
    # If not all are selected, select all. Otherwise, deselect all.
    if len(current_selection) < len(all_values):
        return all_values
    return []

# 5. Update Toggle Label
@callback(
    Output("toggle-all-sections", "children"),
    Input("report-selection-store", "data")
)
def update_toggle_button_label(current_selection):
    current_selection = current_selection or []
    if len(current_selection) < len(REPORT_SECTIONS_OPTIONS):
        return "Select All"
    return "Deselect All"

# 6. Generate Report
@callback(
    [Output("report-header-title", "children"),
     Output("report-header-subtitle", "children"),
     Output("report-date-line", "children"),
     Output("report-footer-timestamp", "children"),
     Output("report-content-container", "children"),
     Output("report-container", "className")],
    [Input("btn-refresh-report", "n_clicks"),
     Input("data-signal", "data"),
     Input("report-order-store", "data"),
     Input("report-selection-store", "data"),
     Input("report-title-input", "value"),
     Input("report-period-select", "value"),
     Input("report-include-exited", "value")]
)
def update_report(n_clicks, signal, order_list, selected_list, title, period, include_exited):
    print_preview = False # Defaults to dark mode for screen preview

    """
    Main callback to generate report content based on configuration.
    """
    data = dw.get_data()
    if not data:
        return "Loading...", "", "", "", html.Div("Loading data..."), "report-container"
    
    # Theme setup
    theme = "light" if print_preview else "dark"
    container_class = "report-container print-preview-mode" if print_preview else "report-container"
    
    # Header content
    report_title = title or "Portfolio Performance Report"
    
    # Date line
    now = datetime.now()
    date_line = f"Report Date: {now.strftime('%B %d, %Y')}"
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    # =========================================================
    # 1. Determine Start Date and Handle Short History
    pv = data.get("pv")
    start_date = None
    is_short_history = False
    
    # Dynamic 1D Label
    end_date_report = pv.index.max() if pv is not None and not pv.empty else datetime.now()
    label_1d = dw.get_display_label_for_1d(end_date_report)

    if pv is not None and not pv.empty:
        end_date = pv.index.max()
        inception = data.get("inception_date", pv.index.min())

        horizon_map = {
            "YTD": "YTD",
            "12M": "1Y", # Standardize to financial_math labelling
            "3M": "3M",
            "1M": "1M"
        }
        
        target_h = horizon_map.get(period)
        if target_h:
            # use centralized logic to find EXACT matching trading day
            calc_start = get_portfolio_horizon_start(pv, inception, target_h)
            if calc_start is not None:
                start_date = calc_start
            elif period != "SI":
                # If target horizon requested but calc_start is None -> Short History
                # Fallback to SI behavior so we don't return partial weird data
                is_short_history = True
                period = "SI"
        # If SI or calc_start is None (for SI), start_date remains None (Inception)

    period_labels = {
        "SI": "Since Inception",
        "YTD": "Year-to-Date",
        "12M": "Trailing 12 Months",
        "3M": "Last Quarter",
        "1M": "Last Month"
    }
    
    base_subtitle = period_labels.get(period, 'Since Inception')
    if is_short_history:
        base_subtitle += " (Short History)"
    subtitle = f"{base_subtitle} Analysis"
    
    # Resolve Section Order
    order_list = order_list or DEFAULT_ORDER
    selected_list = selected_list or []
    
    current_values = set(order_list)
    options_dict = {opt["value"]: opt for opt in REPORT_SECTIONS_OPTIONS}
    missing = [k for k in options_dict if k not in current_values]
    final_order = [k for k in order_list if k in options_dict] + missing
    
    report_sections = []
    
    # Loop through sections in order
    for section_key in final_order:
        if section_key not in selected_list:
            continue
            
        # 0. Morning AI Summary
        if section_key == "morning_brief":
            # Generate summary text using unified period-aware logic
            summary_text = generate_ai_summary_period(data, start_date=start_date, end_date=end_date)
            
            brief_section = html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H4("Period Performance Summary", className="section-title mb-3"),
                        dcc.Markdown(summary_text, className="text-muted" if print_preview else "text-white")
                    ], width=12)
                ])
            ], className="report-section no-break")
            report_sections.append(brief_section)
        
        # 1. Executive Summary
        elif section_key == "summary":
            metrics = dw.get_snapshot_metrics(data)
            
            # Default Labels
            twr_label = "Total Return (TWR)"
            pl_label = "Total P/L"
            
            # Override if specific period selected and not SI
            if period != "SI" and start_date:
                try:
                    # Get centralized parameters
                    pv = data.get("pv", pd.Series())
                    cf = data.get("cf_ext", pd.DataFrame())
                    inception_date = data.get("inception_date")
                    effective_as_of = data.get("effective_as_of")
                    
                    # 1. Calculate Period Return (TWR) using CENTRALIZED function
                    # Uses compute_horizon_twr which applies proper snap-back logic
                    period_return = compute_horizon_twr(
                        pv, cf, inception_date, target_h, effective_as_of=effective_as_of
                    )
                    
                    if not pd.isna(period_return):
                        metrics['twr_si'] = period_return
                        twr_label = f"Period Return ({period})"
                    
                    # 2. Calculate Stats (Sharpe, Sortino, Max DD) on sliced curve
                    # Use effective_as_of for consistent end date
                    twr_curve = dw._get_daily_twr_curve(data)
                    calc_end = effective_as_of if effective_as_of else (pv.index.max() if not pv.empty else end_date)
                    sub_curve = twr_curve[(twr_curve.index >= start_date) & (twr_curve.index <= calc_end)]

                    if not sub_curve.empty:
                        # Efficiency (GIPS-compliant: pass dates for duration-aware annualization)
                        eff = dw.calculate_efficiency_metrics(sub_curve, start_date=start_date, end_date=calc_end)
                        metrics['sharpe'] = eff['sharpe']
                        metrics['sortino'] = eff['sortino']
                        
                        # Drawdown
                        _, dd_val, _ = dw.compute_drawdown_series(sub_curve)
                        metrics['max_dd'] = dd_val / 100.0
                        
                    # 3. Calculate Period P/L using CENTRALIZED function
                    # Uses calculate_horizon_pl which applies proper snap-back logic
                    period_pl = calculate_horizon_pl(
                        pv, inception_date, cf, target_h, effective_as_of=effective_as_of
                    )
                    
                    if period_pl is not None:
                        metrics['pl_si'] = period_pl
                        pl_label = f"Period P/L ({period})"
                
                except Exception as e:
                    # Fallback to defaults or print error
                    print(f"Error calculating Custom Report period metrics: {e}")

            # Calculate MTD Return if missing
            mtd_ret = 0.0
            twr_df = data.get("twr_df", pd.DataFrame())
            if not twr_df.empty and "Horizon" in twr_df.columns:
                 row = twr_df[twr_df["Horizon"] == "MTD"]
                 if not row.empty:
                     mtd_ret = row["Return"].iloc[0]
                     
            summary_section = html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H4("Executive Summary", className="section-title mb-3"),
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Div("Current Value", className="kpi-label-print text-white"),
                            html.Div(fmt_dollar_clean(metrics.get('current_mv', 0) if metrics else 0), className="kpi-value-print text-white")
                        ], className="kpi-box-print")
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.Div(twr_label, className="kpi-label-print text-white"),
                            html.Div(fmt_pct_clean(metrics.get('twr_si', 0) if metrics else 0), className="kpi-value-print",
                                     style={"color": "#28a745" if (metrics.get('twr_si', 0) or 0) >= 0 else "#dc3545"})
                        ], className="kpi-box-print")
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.Div(pl_label, className="kpi-label-print text-white"),
                            html.Div(fmt_dollar_clean(metrics.get('pl_si', 0) if metrics else 0), className="kpi-value-print",
                                     style={"color": "#28a745" if (metrics.get('pl_si', 0) or 0) >= 0 else "#dc3545"})
                        ], className="kpi-box-print")
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.Div("MTD Return", className="kpi-label-print text-white"),
                            html.Div(fmt_pct_clean(mtd_ret), className="kpi-value-print",
                                     style={"color": "#28a745" if (mtd_ret or 0) >= 0 else "#dc3545"})
                        ], className="kpi-box-print")
                    ], width=3),
                ], className="mb-4"),
                
                # Risk Metrics Row
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Div("Max Drawdown", className="kpi-label-print text-white"),
                            html.Div(fmt_pct_clean(metrics.get('max_dd', 0) if metrics else 0), className="kpi-value-print",
                                     style={"color": "#dc3545"})
                        ], className="kpi-box-print")
                    ], width=4),
                    dbc.Col([
                        html.Div([
                            html.Div("Sharpe Ratio", className="kpi-label-print text-white"),
                            html.Div(f"{metrics.get('sharpe', 'N/A'):.2f}" if isinstance(metrics.get('sharpe'), (int, float)) else str(metrics.get('sharpe', 'N/A')), className="kpi-value-print text-white")
                        ], className="kpi-box-print")
                    ], width=4),
                    dbc.Col([
                        html.Div([
                            html.Div("Sortino Ratio", className="kpi-label-print text-white"),
                            html.Div(f"{metrics.get('sortino', 'N/A'):.2f}" if isinstance(metrics.get('sortino'), (int, float)) else str(metrics.get('sortino', 'N/A')), className="kpi-value-print text-white")
                        ], className="kpi-box-print")
                    ], width=4),
                ], className="mb-4"),
            ], className="report-section no-break")
            report_sections.append(summary_section)
        
        # 2. Performance Chart
        elif section_key == "performance_chart":
            fig = dw.get_pv_mountain_chart(data, theme)
            if print_preview:
                fig = apply_print_theme(fig)
            
            # Zoom if period selected
            if start_date:
                fig.update_xaxes(range=[start_date, end_date])

            fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=30), autosize=True)
            
            perf_section = html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H4("Portfolio Performance", className="section-title mb-3"),
                        dcc.Graph(figure=fig, config={'displayModeBar': False}, style={'height': '500px', 'width': '100%'}, responsive=True),
                    ], width=12)
                ])
            ], className="report-section page-break-before")
            report_sections.append(perf_section)
        
        # 3. Horizon Analysis Table
        elif section_key == "horizon_table":
            horizon_df = dw.get_horizon_analysis(data)
            
            # Remove Sharpe/Sortino/Max DD columns if present
            for col in ['Sharpe', 'Sortino', 'Max DD']:
                if col in horizon_df.columns:
                    horizon_df = horizon_df.drop(columns=[col])
            
            # Filter rows based on valid horizons for the reporting period
            valid_horizons = get_valid_horizons_for_period(start_date, end_date, selected_period=period)
            # Match horizon labels (handle "Since Inception" -> "SI", "(Ann.)" suffix)
            def matches_valid_horizon(h_label):
                if h_label is None:
                    return False
                h_str = str(h_label)
                # Check for "Since Inception" variants
                if "Since Inception" in h_str:
                    return "SI" in valid_horizons
                # Check for standard horizons (strip "(Ann.)" suffix if present)
                for vh in valid_horizons:
                    if h_str.startswith(vh):
                        return True
                return False
            
            horizon_df = horizon_df[horizon_df['Horizon'].apply(matches_valid_horizon)]
            
            # Format
            if 'Return' in horizon_df.columns:
                horizon_df['Return'] = horizon_df['Return'].apply(fmt_pct_clean)
            if 'P/L' in horizon_df.columns:
                horizon_df['P/L'] = horizon_df['P/L'].apply(fmt_dollar_clean)
            
            # Filter meta columns
            display_cols = [c for c in horizon_df.columns if not c.startswith('meta_')]
            
            grid_class = "ag-theme-alpine" if print_preview else "ag-theme-alpine-dark"
            
            horizon_section = html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H4("Horizon Analysis", className="section-title mb-3"),
                        dag.AgGrid(
                            rowData=horizon_df[display_cols].to_dict('records'),
                            columnDefs=[{"field": col, "headerName": col, "flex": 1} for col in display_cols],
                            defaultColDef={"sortable": True, "resizable": True},
                            className=grid_class,
                            dashGridOptions={"domLayout": "autoHeight"},
                            style={"width": "100%"}
                        )
                    ], width=12)
                ])
            ], className="report-section no-break")
            report_sections.append(horizon_section)
        
        # 4. Asset Allocation
        elif section_key == "allocation":
            pie_fig, bar_fig = dw.get_asset_allocation_charts(data, theme)
            hist_fig = dw.get_allocation_history_chart(data, theme)
            
            if print_preview:
                pie_fig = apply_print_theme(pie_fig)
                bar_fig = apply_print_theme(bar_fig)
                hist_fig = apply_print_theme(hist_fig)
            
            if start_date:
                hist_fig.update_xaxes(range=[start_date, end_date])

            pie_fig.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=20), autosize=True)
            bar_fig.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=20), autosize=True)
            hist_fig.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=40), autosize=True)
            
            alloc_section = html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H4("Asset Allocation", className="section-title mb-3"),
                    ], width=12),
                    dbc.Col([
                        dcc.Graph(figure=pie_fig, config={'displayModeBar': False}, style={'height': '450px', 'width': '100%'}, responsive=True)
                    ], width=12, className="mb-4"),
                    dbc.Col([
                        dcc.Graph(figure=bar_fig, config={'displayModeBar': False}, style={'height': '450px', 'width': '100%'}, responsive=True)
                    ], width=12, className="mb-4"),
                    dbc.Col([
                        html.H5("Allocation History", className="mt-2 mb-2"),
                        dcc.Graph(figure=hist_fig, config={'displayModeBar': False}, style={'height': '500px', 'width': '100%'}, responsive=True),
                        html.Small("Historical evolution of asset class weights over time.", className="text-muted d-block mt-2", style={"fontSize": "0.85rem"})
                    ], width=12),
                ])
            ], className="report-section page-break-before")
            report_sections.append(alloc_section)
        
        # 5. Sector Breakdown
        elif section_key == "sector":
            sector_fig = dw.get_sector_allocation_chart(data, theme)
            if print_preview:
                sector_fig = apply_print_theme(sector_fig)
            sector_fig.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=40), autosize=True)
            
            sector_section = html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H4("Sector Breakdown", className="section-title mb-3"),
                        dcc.Graph(figure=sector_fig, config={'displayModeBar': False}, style={'height': '500px', 'width': '100%'}, responsive=True)
                    ], width=12)
                ])
            ], className="report-section no-break")
            report_sections.append(sector_section)
        
        # 6. Top Holdings
        elif section_key == "holdings":
            # Dynamic switch for exited
            sec_source = data['sec_table'] if include_exited else data['sec_table_current']

            sec_table = data.get('sec_table_current', pd.DataFrame())
            
            if not sec_table.empty:
                # Get top 10 by market_value
                holdings_df = sec_table[sec_table['ticker'] != 'CASH'].nlargest(10, 'market_value').copy()
                
                # Calculate weight
                total_value = sec_table['market_value'].sum()
                holdings_df['Weight'] = (holdings_df['market_value'] / total_value * 100).round(2).astype(str) + '%'
                holdings_df['Value'] = holdings_df['market_value'].apply(fmt_dollar_clean)
                holdings_df['Shares'] = holdings_df['shares'].round(2)
                
                display_df = holdings_df[['ticker', 'asset_class', 'Shares', 'Value', 'Weight']].copy()
                display_df.columns = ['Ticker', 'Asset Class', 'Shares', 'Value', 'Weight']
                
                grid_class = "ag-theme-alpine" if print_preview else "ag-theme-alpine-dark"
                
                holdings_section = html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.H4("Top 10 Holdings", className="section-title mb-3"),
                            dag.AgGrid(
                                rowData=display_df.to_dict('records'),
                                columnDefs=[{"field": col, "headerName": col, "flex": 1} for col in display_df.columns],
                                defaultColDef={"sortable": True, "resizable": True},
                                className=grid_class,
                                dashGridOptions={"domLayout": "autoHeight"},
                                style={"width": "100%"}
                            )
                        ], width=12)
                    ])
                ], className="report-section no-break")
                report_sections.append(holdings_section)
        
        # 7. Risk Metrics
        elif section_key == "risk":
            risk_df = dw.get_risk_diversification(data)
            grid_class = "ag-theme-alpine" if print_preview else "ag-theme-alpine-dark"
            
            risk_section = html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H4("Risk & Diversification", className="section-title mb-3"),
                        dag.AgGrid(
                            rowData=risk_df.to_dict('records'),
                            columnDefs=[
                                {"field": "Metric", "headerName": "Metric", "flex": 2},
                                {"field": "Value", "headerName": "Value", "flex": 1}
                            ],
                            defaultColDef={"sortable": True, "resizable": True},
                            className=grid_class,
                            dashGridOptions={"domLayout": "autoHeight", "headerHeight": 0},
                            style={"width": "100%"}
                        )
                    ], width=12)
                ])
            ], className="report-section no-break")
            report_sections.append(risk_section)
        


        # 16. Contribution Schedule
        elif section_key == "contrib_schedule":
            # Generate the schedule
            sched_df, sched_footer, is_empty = dw.get_monthly_contribution_schedule(data)
            
            if not is_empty:
                grid_class = "ag-theme-alpine" if print_preview else "ag-theme-alpine-dark"
                
                contrib_section = html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.H4("Monthly Contribution Schedule (Illustrative)", className="section-title mb-3"),
                            dag.AgGrid(
                                rowData=sched_df.to_dict('records'),
                                columnDefs=[
                                    {"field": "Ticker", "headerName": "Ticker", "flex": 1},
                                    {"field": "Gap to Target", "headerName": "Gap to Target", "flex": 1},
                                    {"field": "Monthly Contrib", "headerName": "Monthly Allocation", "flex": 1},
                                    {"field": "Share of Monthly", "headerName": "% of Monthly", "flex": 1}
                                ],
                                defaultColDef={"sortable": True, "resizable": True},
                                className=grid_class,
                                dashGridOptions={"domLayout": "autoHeight"},
                                style={"width": "100%"}
                            ),
                            html.P(sched_footer, className="text-muted mt-2 small")
                        ], width=12)
                    ])
                ], className="report-section no-break")
                report_sections.append(contrib_section)
        
        # 8. Flows Summary
        elif section_key == "flows":
            # Use local period-aware flows logic
            flows_df, flow_subtitle = _get_flows_summary_period(data, start_date=start_date, end_date=end_date)
            grid_class = "ag-theme-alpine" if print_preview else "ag-theme-alpine-dark"
            
            # Pass dates to chart for period-aware filtering
            flows_fig = dw.get_flows_chart(data, theme, start_date=start_date, end_date=end_date)
            if print_preview:
                flows_fig = apply_print_theme(flows_fig)
            flows_fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=30), autosize=True)
            
            flows_section = html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H4(f"Cash Flow Summary ({flow_subtitle})", className="section-title mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                dag.AgGrid(
                                    rowData=flows_df.to_dict('records'),
                                    columnDefs=[
                                        {"field": "Metric", "headerName": "Metric", "flex": 2},
                                        {"field": "Value", "headerName": "Value", "flex": 1}
                                    ],
                                    defaultColDef={"sortable": True, "resizable": True},
                                    className=grid_class,
                                    dashGridOptions={"domLayout": "normal", "headerHeight": 0},
                                    style={"height": "350px", "width": "100%"}
                                )
                            ], width=12, md=6),
                            dbc.Col([
                                dcc.Graph(figure=flows_fig, config={'displayModeBar': False}, style={'height': '350px', 'width': '100%'}, responsive=True)
                            ], width=12, md=6),
                        ])
                    ], width=12)
                ])
            ], className="report-section no-break")
            report_sections.append(flows_section)

        # 9. Tax Lot Explorer (Open Lots)
        elif section_key == "tax_lots":
            # Calculate tax lots using optimized Engine Logic (Time Machine supported)
            # If no specific period, use As Of Now (default behavior)
            as_of_dt = end_date if end_date else datetime.now()
            
            # Capture realized events even if not displayed in this specific table (for consistency)
            open_lots, realized_events = build_tax_lots(strategy="FIFO", as_of_date=as_of_dt)
            grid_class = "ag-theme-alpine" if print_preview else "ag-theme-alpine-dark"
            
            title_suffix = f" (As of {as_of_dt.strftime('%Y-%m-%d')})" if end_date else ""
            
            if not open_lots.empty:
                explorer_df = open_lots.copy()
                explorer_df["Date Acquired"] = explorer_df["Date Acquired"].dt.strftime("%Y-%m-%d")
                
                cols_currency = ["Cost Basis", "Current Price", "Market Value", "Unrealized P/L", "Est Tax Liability"]
                cols_hide = ["Is Near Cliff", "Days to LT", "Cost Per Share"]
                col_defs = []
                for c in explorer_df.columns:
                    hide = c in cols_hide
                    # Ensure currency and long header columns have extra width to prevent truncation
                    min_w = 120 if print_preview else 150
                    if c in cols_currency or len(c) > 12:
                        min_w = 140 if print_preview else 180
                    
                    cd = {"field": c, "headerName": c, "hide": hide, "minWidth": min_w, "flex": 1}
                    
                    if c in cols_currency:
                        cd["valueFormatter"] = {"function": "d3.format('$,.2f')(params.value)"}
                    elif c == "Shares":
                        cd["valueFormatter"] = {"function": "d3.format(',.2f')(params.value)"}
                    
                    col_defs.append(cd)
                    
                tax_lots_section = html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.H4(f"Tax Lot Explorer{title_suffix}", className="section-title mb-3"),
                            dag.AgGrid(
                                rowData=explorer_df.to_dict("records"),
                                columnDefs=col_defs,
                                defaultColDef={"sortable": True, "resizable": True},
                                className=grid_class,
                                dashGridOptions={"domLayout": "autoHeight"},
                                style={"width": "100%"}
                            ),
                            html.Div([
                                html.Small("⚠️ Cliff Watch: Lots maturing to Long-Term (>365 days) within 30 days. Recommend HOLD to lower tax rate.", className="text-warning d-block mt-2"),
                                html.Small("ℹ️ Harvesting Rules: Wash Sale window is ±30 days (61 days total). Losses in this window may be disallowed.", className="text-muted d-block")
                            ])
                        ], width=12)
                    ])
                ], className="report-section page-break-before")
                report_sections.append(tax_lots_section)
            else:
                report_sections.append(html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.H4("Tax Lot Explorer (Open Lots)", className="section-title mb-3"),
                            html.Div("No open lots found.", className="text-muted p-3")
                        ], width=12)
                    ])
                ], className="report-section no-break"))

        # 17. Tax Liability Sunburst (New)
        elif section_key == "tax_sunburst":
            # Generate Data
            as_of_dt = end_date if end_date else datetime.now()
            open_lots, realized_events = build_tax_lots(strategy="FIFO", as_of_date=as_of_dt)
            
            tax_fig = dw.get_tax_liability_sunburst(open_lots, realized_events, theme)
            
            if print_preview:
                tax_fig = apply_print_theme(tax_fig)
                
            tax_fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=30), autosize=True)
            
            sunburst_section = html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H4("Tax Liability Breakdown", className="section-title mb-3"),
                        dcc.Graph(figure=tax_fig, config={'displayModeBar': False}, style={'height': '500px', 'width': '100%'}, responsive=True),
                        html.Small("Breakdown of Estimated Tax Liability by Status (Realized vs Unrealized) and Term (Short-Term vs Long-Term).", className="text-muted d-block mt-2", style={"fontSize": "0.85rem"})
                    ], width=12)
                ])
            ], className="report-section no-break")
            report_sections.append(sunburst_section)

        # 11. Risk Analysis Charts
        elif section_key == "risk_charts":
            risk_fig = dw.get_risk_return_chart(data, theme)
            corr_fig = dw.get_correlation_heatmap(data, theme)
            dd_fig = dw.get_drawdown_chart(data, theme)
            
            if print_preview:
                risk_fig = apply_print_theme(risk_fig)
                corr_fig = apply_print_theme(corr_fig)
                dd_fig = apply_print_theme(dd_fig)
            
            if start_date:
                dd_fig.update_xaxes(range=[start_date, end_date])
                
            risk_fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=30), autosize=True)
            corr_fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=30), autosize=True)
            dd_fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=30), autosize=True)
                
            risk_section = html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H4("Risk Analysis", className="section-title mb-3"),
                    ], width=12),
                    dbc.Col([
                        dcc.Graph(figure=risk_fig, config={'displayModeBar': False}, style={'height': '500px', 'width': '100%'}, responsive=True),
                        html.Small("Risk/Return Profile: Calculated using 10-year realized volatility and geometric returns. Markers represent asset classes.", className="text-muted d-block mt-2 mb-4", style={"fontSize": "0.85rem"})
                    ], width=12),
                    dbc.Col([
                        dcc.Graph(figure=corr_fig, config={'displayModeBar': False}, style={'height': '500px', 'width': '100%'}, responsive=True),
                        html.Small("90-Day Rolling Correlation matrix of top holdings. Red indicates high correlation, Blue indicates diversification benefit.", className="text-muted d-block mt-2 mb-4", style={"fontSize": "0.85rem"})
                    ], width=12),
                    dbc.Col([
                        dcc.Graph(figure=dd_fig, config={'displayModeBar': False}, style={'height': '500px', 'width': '100%'}, responsive=True),
                        html.Small("Underwater Plot: Peak-to-trough decline over time. Shaded area represents deviation from all-time highs.", className="text-muted d-block mt-2", style={"fontSize": "0.85rem"})
                    ], width=12)
                ])
            ], className="report-section page-break-before")
            report_sections.append(risk_section)

        # 12. Performance Deep Dive
        elif section_key == "perf_deep_dive":
            bm_map = {
                "S&P 500": "SPY",
                "Total US Market": "VTI",
                "Aggressive Alloc": "AOA"
            }
            # Pass start_date if available, otherwise it defaults to None (inception)
            cum_fig = dw.get_cumulative_return_chart(data, start_date, bm_map, theme)
            growth_fig = dw.get_growth_of_capital_chart(data, "Total", theme)
            
            if print_preview:
                cum_fig = apply_print_theme(cum_fig)
                growth_fig = apply_print_theme(growth_fig)
            
            if start_date:
                growth_fig.update_xaxes(range=[start_date, end_date])
                
            cum_fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=30), autosize=True)
            growth_fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=30), autosize=True)
                
            perf_deep_section = html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H4("Performance Deep Dive", className="section-title mb-3"),
                        html.H5("Cumulative Return vs Benchmark", className="mb-2"),
                        dcc.Graph(figure=cum_fig, config={'displayModeBar': False}, style={'height': '500px', 'width': '100%'}, responsive=True),
                        html.H5("Growth of Invested Capital", className="mt-4 mb-2"),
                        dcc.Graph(figure=growth_fig, config={'displayModeBar': False}, style={'height': '500px', 'width': '100%'}, responsive=True)
                    ], width=12)
                ])
            ], className="report-section page-break-before")
            report_sections.append(perf_deep_section)

        # 13. Attribution Analysis
        elif section_key == "attribution":
            # Pass user-selected dates to ensure proper data aggregation (Daily vs Weekly vs Monthly)
            attr_fig = dw.get_smart_attribution_chart(data, start_date, end_date, theme)
            if print_preview:
                attr_fig = apply_print_theme(attr_fig)

            attr_fig.update_layout(height=600, margin=dict(l=0, r=0, t=30, b=30), autosize=True)
                
            attr_section = html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H4("Attribution Analysis", className="section-title mb-3"),
                        dcc.Graph(figure=attr_fig, config={'displayModeBar': False}, style={'height': '600px', 'width': '100%'}, responsive=True),
                        html.Small([
                            html.I(className="fa-solid fa-triangle-exclamation me-1"),
                            "Note: Attribution effects are approximated using an arithmetic difference method. Minor discrepancies labeled as 'Recon/Residual' are expected when compared to the geometrically-linked Time-Weighted Return (TWR) shown in the Performance section. ",
                            html.A("Learn more", href="/help#attribution-methodology", target="_blank", className="text-muted text-decoration-underline")
                        ], className="text-muted d-block mt-2", style={"fontSize": "0.85rem"})
                    ], width=12)
                ])
            ], className="report-section page-break-before")
            report_sections.append(attr_section)

        # 14. Asset Class Performance Table
        elif section_key == "ac_perf_table":
            class_df = data['class_df']
            # Conditional Selection based on include_exited
            sec_table = data['sec_table'] if include_exited else data['sec_table_current']
            
            visibility_text = "Tables display ALL positions (active and exited) with valid history in the period." if include_exited else "Tables display currently active positions only."

            # Filter horizons based on reporting period duration
            horizons = get_valid_horizons_for_period(start_date, end_date, selected_period=period)
            
            rows = []
            # Sort classes
            ac_rank_map = {ac: i for i, ac in enumerate(class_df['asset_class'].unique())}
            
            for _, crow in class_df.iterrows():
                ac = crow['asset_class']
                rank = ac_rank_map.get(ac, 999)
                
                # Class Row
                r_vals = {
                    "Asset Class / Ticker": ac, 
                    "Type": "Class", 
                    "_sort_rank": rank
                }
                for h in horizons:
                    val = crow.get(h)
                    r_vals[h] = fmt_pct_clean(val) if pd.notna(val) else "N/A"
                rows.append(r_vals)
                
                # Ticker Rows
                tickers = sec_table[sec_table['asset_class'] == ac]
                for _, trow in tickers.iterrows():
                    t = trow['ticker']
                    tr_vals = {
                        "Asset Class / Ticker": f"  {t}", 
                        "Type": "Ticker", 
                        "_sort_rank": rank
                    }
                    for h in horizons:
                        val = trow.get(h)
                        tr_vals[h] = fmt_pct_clean(val) if pd.notna(val) else "N/A"
                    rows.append(tr_vals)
            
            grid_class = "ag-theme-alpine" if print_preview else "ag-theme-alpine-dark"
            cols = ["Asset Class / Ticker"] + horizons
            
            # Map 1D column header
            final_col_defs = []
            for c in cols:
                h_name = c
                if c == "1D": h_name = label_1d
                
                final_col_defs.append({
                    "field": c, 
                    "headerName": h_name, 
                    "flex": 1, 
                    "pinned": "left" if c == "Asset Class / Ticker" else None, 
                    "minWidth": (120 if print_preview else 150) if c == "Asset Class / Ticker" else (60 if print_preview else 80)
                })
            
            perf_table_section = html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H4("Asset Class Performance", className="section-title mb-3"),
                        dag.AgGrid(
                            rowData=rows,
                            columnDefs=final_col_defs,
                            defaultColDef={"sortable": False, "resizable": True}, # Disable sort to keep hierarchy
                            className=grid_class,
                            dashGridOptions={
                                "domLayout": "autoHeight",
                                "getRowStyle": {
                                    "function": "params.data.Type === 'Class' ? {'fontWeight': 'bold', 'backgroundColor': 'rgba(0,0,0,0.05)'} : {}"
                                }
                            },
                            style={"width": "100%"}
                        ),
                        html.Small([
                            html.I(className="fa-solid fa-info-circle me-1"),
                            visibility_text
                        ], className="text-muted d-block mt-2", style={"fontSize": "0.85rem", "fontStyle": "italic"})
                    ], width=12)
                ])
            ], className="report-section page-break-before")
            report_sections.append(perf_table_section)

        # 15. Asset Class P/L Table
        elif section_key == "ac_pl_table":
            class_df = data['class_df']
            # Dynamic switch for PL table
            sec_table = data['sec_table'] if include_exited else data['sec_table_current']
            
            # Footer text if mode is active
            pl_vis_text = None
            if include_exited:
                 pl_vis_text = html.P("Includes realized P/L from closed positions.", className="text-muted small mt-2 fst-italic")
            
            # Filter horizons based on reporting period duration
            horizons = get_valid_horizons_for_period(start_date, end_date, selected_period=period)
            
            # Pre-fetch ticker P/L for all horizons (use cache if available)
            ticker_pl_cache = data.get("ticker_pl_cache")
            if not ticker_pl_cache:
                ticker_pl_cache = {}
                for h in horizons:
                    ticker_pl_cache[h] = dw.get_ticker_pl_df(data, h)
                
            rows = []
            ac_rank_map = {ac: i for i, ac in enumerate(class_df['asset_class'].unique())}
            
            for _, crow in class_df.iterrows():
                ac = crow['asset_class']
                rank = ac_rank_map.get(ac, 999)
                
                # Class Row
                r_vals = {
                    "Asset Class / Ticker": ac, 
                    "Type": "Class", 
                    "_sort_rank": rank
                }
                asset_class_pl_cache = data.get("asset_class_pl_cache", {})
                for h in horizons:
                    res = asset_class_pl_cache.get(h, {}).get(ac)
                    if res is None:
                        res = dw.get_asset_class_pl(data, ac, h, return_components=False)
                    r_vals[h] = fmt_dollar_clean(res) if res is not None else "N/A"
                rows.append(r_vals)
                
                # Ticker Rows
                tickers = sec_table[sec_table['asset_class'] == ac]
                for _, trow in tickers.iterrows():
                    t = trow['ticker']
                    tr_vals = {
                        "Asset Class / Ticker": f"  {t}", 
                        "Type": "Ticker", 
                        "_sort_rank": rank
                    }
                    for h in horizons:
                        pl_df = ticker_pl_cache.get(h, pd.DataFrame())
                        if not pl_df.empty and t in pl_df.index:
                            val = pl_df.loc[t, "pl"]
                            tr_vals[h] = fmt_dollar_clean(val)
                        else:
                            tr_vals[h] = "N/A"
                    rows.append(tr_vals)
                    
            grid_class = "ag-theme-alpine" if print_preview else "ag-theme-alpine-dark"
            cols = ["Asset Class / Ticker"] + horizons
            
            # Map 1D column header
            final_col_defs_pl = []
            for c in cols:
                h_name = c
                if c == "1D": h_name = label_1d
                
                final_col_defs_pl.append({
                    "field": c, 
                    "headerName": h_name, 
                    "flex": 1, 
                    "pinned": "left" if c == "Asset Class / Ticker" else None, 
                    "minWidth": (120 if print_preview else 150) if c == "Asset Class / Ticker" else (60 if print_preview else 80)
                })
            
            pl_table_section = html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H4("Asset Class P/L (Economic)", className="section-title mb-3"),
                        dag.AgGrid(
                            rowData=rows,
                            columnDefs=final_col_defs_pl,
                            defaultColDef={"sortable": False, "resizable": True},
                            className=grid_class,
                            dashGridOptions={
                                "domLayout": "autoHeight",
                                "getRowStyle": {
                                    "function": "params.data.Type === 'Class' ? {'fontWeight': 'bold', 'backgroundColor': 'rgba(0,0,0,0.05)'} : {}"
                                }
                            },
                            style={"width": "100%"}
                        ),
                        pl_vis_text
                    ], width=12)
                ])
            ], className="report-section page-break-before")
            report_sections.append(pl_table_section)
    
    return report_title, subtitle, date_line, timestamp, report_sections, container_class


# ============================================================
# WORD REPORT GENERATION CALLBACK
# ============================================================

def _clip_data(data, start_date=None, end_date=None):
    """
    Creates a shallow copy of the data dictionary with time-series filtered
    to the [start_date, end_date] window. Matches 'as_of' logic for snapshot data where possible.
    """
    if not start_date:
        return data 
        
    clipped = data.copy()
    eff_start = pd.Timestamp(start_date)
    eff_end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()
    
    # Filter Function
    def _filter_ts(df, date_col=None):
        if df is None or df.empty: return df
        if date_col:
            mask = (df[date_col] >= eff_start) & (df[date_col] <= eff_end)
            return df[mask]
        else: # index based
            mask = (df.index >= eff_start) & (df.index <= eff_end)
            return df[mask]

    if "pv" in clipped: clipped["pv"] = _filter_ts(clipped["pv"])
    if "cf_ext" in clipped: clipped["cf_ext"] = _filter_ts(clipped["cf_ext"], "date")
    if "cf_all" in clipped: clipped["cf_all"] = _filter_ts(clipped["cf_all"], "date")
    if "tx_raw" in clipped: clipped["tx_raw"] = _filter_ts(clipped["tx_raw"], "date")
    if "dividends" in clipped: clipped["dividends"] = _filter_ts(clipped["dividends"], "date")
    if "prices" in clipped: clipped["prices"] = _filter_ts(clipped["prices"])
    
    # TWR Curve: We want the curve segment, but careful about the anchor point.
    # If we cut strictly >= start, the first point is T+1 return relative to start.
    # This is generally what charts expect (they index from start).
    if "twr_curve" in clipped: clipped["twr_curve"] = _filter_ts(clipped["twr_curve"])
    
    return clipped

@callback(
    [Output("download-word-report", "data"),
     Output("download-status-alert", "children"),
     Output("download-status-alert", "color"),
     Output("download-status-alert", "is_open")],
    Input("btn-download-word", "n_clicks"),
    State("report-order-store", "data"), # Changed
    State("report-selection-store", "data"), # Changed
    State("report-title-input", "value"),
    State("report-period-select", "value"),
    State("report-mobile-mode", "value"),
    State("date-range-store", "data"),  # Sidebar Analysis End Date
    prevent_initial_call=True
)
def download_word_report(n_clicks, order_list, selected_list, title, period, mobile_mode, date_range_store):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update, False
        
    data = dw.get_data()
    
    # Resolve Sections Logic
    order_list = order_list or DEFAULT_ORDER
    selected_list = selected_list or []
    current_values = set(order_list)
    options_dict = {opt["value"]: opt for opt in REPORT_SECTIONS_OPTIONS}
    missing = [k for k in options_dict if k not in current_values]
    final_order = [k for k in order_list if k in options_dict] + missing
    
    # Generate ordered sections list for Word Report
    sections = [k for k in final_order if k in selected_list]
    
    title = title or "Portfolio Value Report"
    

    
    period_labels = {
        "SI": "Since Inception",
        "YTD": "Year-to-Date",
        "12M": "Trailing 12 Months",
        "3M": "Last Quarter",
        "1M": "Last Month"
    }
    subtitle = f"{period_labels.get(period, 'Since Inception')} Analysis"
    
    # Generate Doc
    try:
        if not data:
            return dash.no_update, "No data available. Please load the portfolio first.", "warning", True
        
        # Determine Report Horizon - Use sidebar Analysis End Date
        end_date = datetime.now()  # Default fallback
        if date_range_store and date_range_store.get("end"):
            try:
                end_date = datetime.strptime(date_range_store["end"], "%Y-%m-%d")
            except (ValueError, TypeError):
                pass  # Keep default if parsing fails
        
        start_date = None

        # CLIP DATA FOR CHARTS BUT PRESERVE UNCLIPPED FOR P/L CALCULATIONS
        data_unclipped = data  # Keep original data for P/L calculations
        data_clipped = data    # Will be clipped for visual elements
        
        if period and period != "SI":
            # MAP UI LABEL TO MATH LABEL (Crucial: 12M -> 1Y)
            math_period = "1Y" if period == "12M" else period
            
            pv = data.get("pv")
            if pv is not None and not pv.empty:
                inception_date = pv.index.min()
                start_date = get_portfolio_horizon_start(pv, inception_date, math_period)
                if start_date:
                    data_clipped = _clip_data(data, start_date=start_date, end_date=end_date)
            
        p_label = period_labels.get(period, period)
        # Pass both datasets: unclipped for P/L, clipped for charts
        doc = generate_word_report(data_clipped, sections, title, subtitle, p_label, 
                                   mobile_mode=mobile_mode, start_date=start_date, end_date=end_date,
                                   data_unclipped=data_unclipped)
        
        # Save to buffer
        buffer = BytesIO()
        doc.save(buffer)
        buffer.seek(0)

        filename_prefix = "Investment_Report_Mobile_" if mobile_mode else "Investment_Report_"
        return dcc.send_bytes(buffer.getvalue(), filename=f"{filename_prefix}{datetime.now().strftime('%Y%m%d')}.docx"), "Report generated successfully!", "success", True
    except Exception as e:
        # In case of error (e.g. kaleido issue), maybe return a text file with error?
        # Or just let Dash handle it (it will show error in debug mode)
        # For now, simplistic error handling
        print(f"Error generating report: {e}")
        return dash.no_update, f"Error: {str(e)}", "danger", True

# ============================================================
# CALLBACK: GOOGLE DRIVE EXPORT
# ============================================================
@callback(
    Output("drive-export-feedback", "children"),
    Input("btn-export-drive", "n_clicks"),
    State("report-order-store", "data"),
    State("report-selection-store", "data"),
    State("report-title-input", "value"),
    State("report-period-select", "value"),
    State("report-mobile-mode", "value"),
    State("date-range-store", "data"),  # Sidebar Analysis End Date
    prevent_initial_call=True
)
def export_word_to_drive(n_clicks, order_list, selected_list, title, period, mobile_mode, date_range_store):
    """
    Exports the Word Report (.docx) to Google Drive.
    """
    if not n_clicks:
        return ""
    
    msg = "⏳ Generating & Exporting..."
    try:
        data = dw.get_data()
        if not data:
            return "❌ No data to export."
            
        # 1. Resolve Sections Logic (Mirrors download_word_report)
        order_list = order_list or DEFAULT_ORDER
        selected_list = selected_list or []
        current_values = set(order_list)
        options_dict = {opt["value"]: opt for opt in REPORT_SECTIONS_OPTIONS}
        missing = [k for k in options_dict if k not in current_values]
        final_order = [k for k in order_list if k in options_dict] + missing
        
        # Generate ordered sections list
        sections = [k for k in final_order if k in selected_list]
        
        title = title or "Portfolio Value Report"
        
        period_labels = {
            "SI": "Since Inception",
            "YTD": "Year-to-Date",
            "12M": "Trailing 12 Months",
            "3M": "Last Quarter",
            "1M": "Last Month"
        }
        subtitle = f"{period_labels.get(period, 'Since Inception')} Analysis"

        # 2. Determine Report Horizon & Clip Data - Use sidebar Analysis End Date
        end_date = datetime.now()  # Default fallback
        if date_range_store and date_range_store.get("end"):
            try:
                end_date = datetime.strptime(date_range_store["end"], "%Y-%m-%d")
            except (ValueError, TypeError):
                pass  # Keep default if parsing fails
        
        start_date = None

        # CLIP DATA FOR CHARTS BUT PRESERVE UNCLIPPED FOR P/L CALCULATIONS
        data_unclipped = data  # Keep original data for P/L calculations
        data_clipped = data    # Will be clipped for visual elements
        
        if period and period != "SI":
            math_period = "1Y" if period == "12M" else period
            pv = data.get("pv")
            if pv is not None and not pv.empty:
                inception_date = pv.index.min()
                start_date = get_portfolio_horizon_start(pv, inception_date, math_period)
                if start_date:
                    data_clipped = _clip_data(data, start_date=start_date, end_date=end_date)
            
        p_label = period_labels.get(period, period)
        
        # 3. Generate Word Document - pass both datasets
        doc = generate_word_report(data_clipped, sections, title, subtitle, p_label, 
                                   mobile_mode=mobile_mode, start_date=start_date, end_date=end_date,
                                   data_unclipped=data_unclipped)
        
        # 4. Save to buffer
        buffer = BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        
        # 5. Define Filename
        filename_prefix = "Investment_Report_Mobile_" if mobile_mode else "Investment_Report_"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{filename_prefix}{timestamp}.docx"
        
        # 6. Upload to Drive
        # MIME type for docx: application/vnd.openxmlformats-officedocument.wordprocessingml.document
        result_msg = send_to_drive(
            buffer.getvalue(), 
            filename, 
            mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )
        return result_msg
        
    except Exception as e:
        print(f"Error exporting to drive: {e}") 
        return f"❌ Error: {str(e)}"
