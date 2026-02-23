import dash
from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import dash_wrappers as dw
from report_formatting import fmt_pct_clean, fmt_dollar_clean
import pandas as pd
from components.ai_brief import generate_ai_summary
from components.data_source_badge import create_price_source_badge
from components.page_header import page_header
from components.storytelling_cards import (
    build_performance_story_card,
    build_risk_story_card,
    build_flows_story_card,
)
from components.kpi_card import create_kpi_card

layout = html.Div([
    # --- HEADER ---
    page_header(
        title="Overview",
        icon="bi-house-door",
        subtitle="Portfolio summary and daily highlights"
    ),

    # Cash Settlement Alert (shown when CASH recon is auto-bridged)
    html.Div(id='settlement-alert-container'),

    # Data Status Note
    html.Div(id='data-status-container', style={'position': 'fixed', 'top': '15px', 'right': '20px', 'zIndex': 2000, 'maxWidth': '90vw'}),
    
    # Price Source Badge (Fixed position below data status)
    html.Div(id='price-source-badge-container', style={'position': 'fixed', 'top': '30px', 'right': '75px', 'zIndex': 1999}),

    # Morning Brief AI Card
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader([
                html.I(className="bi bi-robot me-2"),
                "Morning Brief (AI Summary)"
            ], className="section-header"),
            dbc.CardBody([
                dcc.Loading(dcc.Markdown(id='ai-brief-content', children="Generating summary..."))
            ])
        ], className="mb-4 shadow-sm border-primary"), width=12)
    ]),

    # ── STICKY KPI RIBBON ──
    html.Div([
        dbc.Row([
            dbc.Col(html.Div(id='kpi-val-card', style={'height': '100%'}), xs=6, sm=4, md=2),
            dbc.Col(html.Div(id='kpi-twr-card', style={'height': '100%'}), xs=6, sm=4, md=2),
            dbc.Col(html.Div(id='kpi-pl-card', style={'height': '100%'}), xs=6, sm=4, md=2),
            dbc.Col(html.Div(id='kpi-alpha-card', style={'height': '100%'}), xs=6, sm=4, md=2),
            dbc.Col(html.Div(id='kpi-mtd-card', style={'height': '100%'}), xs=6, sm=4, md=2),
            dbc.Col(html.Div(id='kpi-cashdrag-card', style={'height': '100%'}), xs=6, sm=4, md=2),
        ], className="g-2"),
    ], className="kpi-ribbon"),

    # ── HERO CHART + CONTEXT PANEL ──
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H5("Portfolio Value (Since Inception %)", className="card-title section-header p-2"),
            dcc.Graph(id='pv-chart', style={'height': '428px'})
        ], className="hero-chart-card"), xs=12, lg=7),
        dbc.Col(dbc.Card([
            html.H5("Portfolio Snapshot", className="card-title section-header p-2"),
            dcc.Loading(html.Div(id='snapshot-table-container', style={'height': '428px'}))
        ], className="context-panel"), xs=12, lg=5)
    ], className="mb-4 command-center-row"),
    
    # Highlights & Risk Row → Phase 3 Storytelling Cards
    html.Div(id='story-cards-container', className="story-card-row"),
], className="overview-page")

# AI Summary Callback (Independent)
@callback(
    Output('ai-brief-content', 'children'),
    [Input('data-signal', 'data')]
)
def update_ai_brief(signal):
    data = dw.get_data()
    if not data: return "Waiting for data..."
    return generate_ai_summary(data)

# Main Dashboard Callback
@callback(
    [Output('settlement-alert-container', 'children'),
     Output('data-status-container', 'children'),
     Output('price-source-badge-container', 'children'),
     Output('kpi-val-card', 'children'),
     Output('kpi-twr-card', 'children'),
     Output('kpi-pl-card', 'children'),
     Output('kpi-alpha-card', 'children'),
     Output('kpi-mtd-card', 'children'),
     Output('pv-chart', 'figure'),
     Output('snapshot-table-container', 'children'),
     Output('story-cards-container', 'children'),
     Output('kpi-cashdrag-card', 'children')],
    [Input('data-signal', 'data'),
     Input('chatbot-command', 'data'),
     Input('filter-store', 'data')]
)
def update_overview(signal, chat_cmd, _filters):
    data = dw.get_data()
    if not data:
        return None, None, None, "...", "...", "...", None, "...", {}, "Loading...", "Loading...", None
    
    # Cash Settlement Alert
    settlement_alert = None
    pv_series = data.get("pv")
    if pv_series is not None:
        bridge = getattr(pv_series, "attrs", {}).get("cash_settlement_bridge")
        if bridge:
            delta = bridge.get("amount", 0)
            direction = bridge.get("direction", "settling cash flow")
            settlement_alert = dbc.Alert(
                [
                    html.I(className="bi bi-clock-history me-2"),
                    html.Strong("Cash Settlement Notice: "),
                    f"Cash balance adjusted by ${abs(delta):,.2f} — likely {direction}. ",
                    "This resolves automatically on next E*TRADE sync.",
                ],
                color="warning",
                dismissable=True,
                className="mb-3",
            )

    # Price Source Badge
    price_source_meta = dw.get_price_source_summary(data)
    price_badge = create_price_source_badge(price_source_meta, "overview-price-badge") if price_source_meta else None
    
    # Data Status Note (handled globally in app.py)
    status_note = None

    # --- CHATBOT PARAMS ---
    chat_target = ""
    chat_action = None
    if chat_cmd:
        chat_action = chat_cmd.get("action")
        chat_target = chat_cmd.get("params", {}).get("target", "").lower()
        
    metrics = dw.get_snapshot_metrics(data)
    
    # KPI Values
    val = fmt_dollar_clean(metrics['current_mv'])
    twr = fmt_pct_clean(metrics['twr_si'])
    pl = fmt_dollar_clean(metrics['pl_si'])
    mtd = fmt_pct_clean(metrics['mtd_ret'])
    
    # Create Bloomberg-style KPI cards
    val_card = create_kpi_card("Current Value", val)
    
    # TWR card with indicator
    twr_is_positive = metrics['twr_si'] >= 0 if metrics['twr_si'] is not None else None
    
    # Dynamic Labeling for Annualization
    twr_label = "Inception TWR (Ann)" if metrics.get("is_annualized") else "Inception TWR"
    sub_label = "Ann. CAGR" if metrics.get("is_annualized") else "Cumulative"
    
    twr_card = create_kpi_card(twr_label, twr, subtext=sub_label, is_positive=twr_is_positive)
    
    # P/L card with indicator
    pl_is_positive = metrics['pl_si'] >= 0 if metrics['pl_si'] is not None else None
    pl_card = create_kpi_card("Inception P/L", pl, is_positive=pl_is_positive)
    
    # MTD card with indicator
    mtd_is_positive = metrics['mtd_ret'] >= 0 if metrics['mtd_ret'] is not None else None
    mtd_card = create_kpi_card("MTD Return", mtd, is_positive=mtd_is_positive)
            
    # Chart
    fig = dw.get_pv_mountain_chart(data, "dark")
    
    # 1. Snapshot Table
    snap_df = dw.get_horizon_analysis(data)
    
    # Remove Sharpe and Sortino columns as requested
    if 'Sharpe' in snap_df.columns:
        snap_df = snap_df.drop(columns=['Sharpe'])
    if 'Sortino' in snap_df.columns:
        snap_df = snap_df.drop(columns=['Sortino'])

    # Format
    snap_df['Return'] = snap_df['Return'].apply(fmt_pct_clean)
    snap_df['P/L'] = snap_df['P/L'].apply(fmt_dollar_clean)
    
    # All rows are now regular rows
    all_rows = snap_df.to_dict('records')
    
    # Chatbot Sort Check
    is_snap_target = "snapshot" in chat_target or (not chat_target and not any(x in chat_target for x in ["highlight", "risk", "flow"]))

    snap_column_defs = []
    for col in snap_df.columns:
        col_def = {"field": col, "headerName": col}
        
        # Hide Audit Meta Columns
        if col.startswith("meta_"):
            col_def["hide"] = True
            
        if col in ["Return", "P/L"]:
            col_def["comparator"] = {"function": "MoneyComparator"}
            
        if chat_action == "SORT" and is_snap_target:
             target_col = chat_cmd["params"].get("column", "").lower()
             if col.lower() == target_col or target_col in col.lower():
                 col_def["sort"] = chat_cmd["params"].get("direction", "desc")
                 
        snap_column_defs.append(col_def)
        
    snap_table = dag.AgGrid(
        id="overview-snapshot-grid",
        rowData=all_rows,
        columnDefs=snap_column_defs,
        defaultColDef={"flex": 1, "minWidth": 100, "sortable": True, "filter": True, "resizable": True},
        className="ag-theme-alpine-dark audit-target", # Added audit-target class
        dashGridOptions={
            "domLayout": "normal",
        },
        style={"height": "428px"}
    )
    
    # ── Phase 3: Storytelling Cards ──
    perf_card = build_performance_story_card(data, metrics, fmt_pct_clean, fmt_dollar_clean)
    risk_card = build_risk_story_card(data, metrics, fmt_pct_clean, fmt_dollar_clean)
    flows_card = build_flows_story_card(data, fmt_dollar_clean)
    story_cards = [perf_card, risk_card, flows_card]

    # Alpha vs S&P 500 (Since Inception)
    import numpy as np
    alpha_raw = metrics.get('alpha_vs_spy', np.nan)
    if pd.notna(alpha_raw):
        alpha_str = fmt_pct_clean(alpha_raw)
        alpha_pos = alpha_raw >= 0
        alpha_sub = "Ann." if metrics.get("is_annualized") else "Cumulative"
    else:
        alpha_str = "N/A"
        alpha_pos = None
        alpha_sub = ""
    alpha_card = create_kpi_card("Alpha vs S&P 500 (SI)", alpha_str, subtext=alpha_sub, is_positive=alpha_pos)

    # Cash Drag %
    cash_pct = metrics.get('cash_drag_pct', 0.0)
    cash_str = f"{cash_pct * 100:.1f}%"
    # Treat > 10% as warning (negative glow), otherwise neutral
    cash_accent = None
    cash_pos = None
    if cash_pct > 0.10:
        cash_pos = False  # red glow — high cash drag
    alpha_card_cash = create_kpi_card("Cash Drag", cash_str, subtext="% of portfolio in cash", is_positive=cash_pos)

    return settlement_alert, status_note, price_badge, val_card, twr_card, pl_card, alpha_card, mtd_card, fig, snap_table, story_cards, alpha_card_cash
