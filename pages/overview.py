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

def create_kpi_card(title, value, subtext=None, is_positive=None, accent=None):
    """
    Glassmorphism KPI card with frosted-glass surface, animated gradient
    border, and inner glow that shifts colour based on positive / negative.

    Args:
        title:       Label text (e.g. "Inception TWR")
        value:       Formatted value string (e.g. "+12.4%")
        subtext:     Optional secondary line (e.g. "Ann. CAGR")
        is_positive: True  → green glow / border
                     False → red glow / border
                     None  → neutral cyan glow
        accent:      Override glow flavour: "blue" | "green" (used for Sharpe / Sortino)
    """
    # ---- normalise is_positive to a plain Python bool (numpy.bool_ fails `is`) ----
    if is_positive is not None:
        is_positive = bool(is_positive)

    # ---- determine colour classes ----
    if accent == "blue":
        glass_mod = "kpi-glass--accent-blue"
        wrapper_mod = "kpi-glass-wrapper--accent-blue"
        value_color = "#6ea8fe"  # Bootstrap primary-light
        sub_color = "rgba(110,168,254,0.7)"
    elif accent == "green":
        glass_mod = "kpi-glass--accent-green"
        wrapper_mod = "kpi-glass-wrapper--accent-green"
        value_color = "#75b798"  # Bootstrap success-light
        sub_color = "rgba(117,183,152,0.7)"
    elif is_positive is True:
        glass_mod = "kpi-glass--positive"
        wrapper_mod = "kpi-glass-wrapper--positive"
        value_color = "#28a745"
        sub_color = "rgba(40,167,69,0.75)"
    elif is_positive is False:
        glass_mod = "kpi-glass--negative"
        wrapper_mod = "kpi-glass-wrapper--negative"
        value_color = "#dc3545"
        sub_color = "rgba(220,53,69,0.75)"
    else:
        glass_mod = "kpi-glass--neutral"
        wrapper_mod = ""
        value_color = "#f8f9fa"
        sub_color = "rgba(248,249,250,0.45)"

    # ---- arrow indicator ----
    arrow_span = None
    if is_positive is not None:
        symbol = "▲" if is_positive else "▼"
        arrow_span = html.Span(
            f"{symbol} ",
            style={
                "color": value_color,
                "fontSize": "1rem",
                "marginRight": "3px",
                "verticalAlign": "middle",
            },
        )

    value_children = [arrow_span, value] if arrow_span else value

    # ---- subtext line (always present to keep height stable) ----
    if subtext:
        sub_prefix = ("▲ " if is_positive else "▼ ") if is_positive is not None else ""
        sub_el = html.Div(
            f"{sub_prefix}{subtext}",
            className="kpi-glass-sub",
            style={"color": sub_color},
        )
    else:
        sub_el = html.Div("\u00a0", className="kpi-glass-sub", style={"color": "transparent"})

    # ---- assemble ----
    inner = html.Div(
        [
            html.Div(title, className="kpi-glass-label"),
            html.Div(
                value_children,
                className="kpi-glass-value",
                style={"color": value_color},
            ),
            sub_el,
        ],
        className=f"kpi-glass {glass_mod}",
    )

    return html.Div(inner, className=f"kpi-glass-wrapper {wrapper_mod}")

layout = html.Div([
    # --- HEADER ---
    page_header(
        title="Overview",
        icon="bi-house-door",
        subtitle="Portfolio summary and daily highlights"
    ),

    # Data Status Note
    html.Div(id='data-status-container', style={'position': 'fixed', 'top': '15px', 'right': '20px', 'zIndex': 2000, 'maxWidth': '90vw'}),
    
    # Price Source Badge (Fixed position below data status)
    html.Div(id='price-source-badge-container', style={'position': 'fixed', 'top': '30px', 'right': '75px', 'zIndex': 1999}),

    # Morning Brief AI Card (NEW)
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

    # Single Unified KPI Row
    dbc.Row([
        # Value
        dbc.Col(html.Div(id='kpi-val-card', style={'height': '100%'}), width=2),
        
        # TWR
        dbc.Col(html.Div(id='kpi-twr-card', style={'height': '100%'}), width=2),
        
        # P/L
        dbc.Col(html.Div(id='kpi-pl-card', style={'height': '100%'}), width=2),
        
        # Alpha vs S&P 500 (Since Inception)
        dbc.Col(html.Div(id='kpi-alpha-card', style={'height': '100%'}), width=2),

        # MTD
        dbc.Col(html.Div(id='kpi-mtd-card', style={'height': '100%'}), width=2),

        # Cash Drag
        dbc.Col(html.Div(id='kpi-cashdrag-card', style={'height': '100%'}), width=2),
    ], className="mb-4 g-2"),

    # Chart & Snapshot Row
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H5("Portfolio Value (Since Inception %)", className="card-title section-header p-2"),
            dcc.Graph(id='pv-chart', style={'height': '428px'})
        ]), width=7),
        dbc.Col(dbc.Card([
            html.H5("Portfolio Snapshot", className="card-title section-header p-2"),
            dcc.Loading(html.Div(id='snapshot-table-container', style={'height': '428px'}))
        ]), width=5)
    ], className="mb-4"),
    
    # Highlights & Risk Row
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H5("Performance Highlights", className="card-title section-header p-2"),
            dcc.Loading(html.Div(id='highlights-table-container'))
        ]), width=6),
        dbc.Col(dbc.Card([
            html.H5("Risk & Diversification", className="card-title section-header p-2"),
            dcc.Loading(html.Div(id='risk-table-container'))
        ]), width=6)
    ], className="mb-4"),
    
    # Flows Row
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H5("Flows Summary (YTD)", className="card-title section-header p-2"),
            dcc.Loading(html.Div(id='flows-table-container'))
        ]), width=12)
    ], className="mb-4")
])

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
    [Output('data-status-container', 'children'),
     Output('price-source-badge-container', 'children'),
     Output('kpi-val-card', 'children'),
     Output('kpi-twr-card', 'children'),
     Output('kpi-pl-card', 'children'),
     Output('kpi-alpha-card', 'children'),
     Output('kpi-mtd-card', 'children'),
     Output('pv-chart', 'figure'),
     Output('snapshot-table-container', 'children'),
     Output('highlights-table-container', 'children'),
     Output('risk-table-container', 'children'),
     Output('flows-table-container', 'children'),
     Output('kpi-cashdrag-card', 'children')],
    [Input('data-signal', 'data'),
     Input('chatbot-command', 'data'),
     Input('filter-store', 'data')]
)
def update_overview(signal, chat_cmd, _filters):
    data = dw.get_data()
    if not data:
        return None, None, "...", "...", "...", None, "...", {}, "Loading...", "Loading...", "Loading...", "Loading...", None
    
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
    
    # 2. Highlights Table
    high_df = dw.get_performance_highlights(data)
    is_high_target = "highlight" in chat_target
    
    high_column_defs = []
    for col in ["Metric", "Value"]:
        col_def = {"field": col, "headerName": "", "flex": 1, "minWidth": 150, "wrapText": True, "autoHeight": True}
        if col == "Value":
            col_def["comparator"] = {"function": "MoneyComparator"}
            
        if chat_action == "SORT" and is_high_target:
             target_col = chat_cmd["params"].get("column", "").lower()
             if col.lower() == target_col or target_col in col.lower():
                 col_def["sort"] = chat_cmd["params"].get("direction", "desc")
        high_column_defs.append(col_def)

    high_table = dag.AgGrid(
        id="overview-highlights-grid",
        rowData=high_df.to_dict('records'),
        columnDefs=high_column_defs,
        defaultColDef={"sortable": True, "filter": True, "resizable": True},
        className="ag-theme-alpine-dark audit-target",
        dashGridOptions={"domLayout": "autoHeight", "headerHeight": 0}
    )
    
    # 3. Risk Table
    risk_df = dw.get_risk_diversification(data)
    is_risk_target = "risk" in chat_target or "diversification" in chat_target
    
    risk_column_defs = []
    for col in ["Metric", "Value"]:
        col_def = {"field": col, "headerName": "", "flex": 1, "minWidth": 150, "wrapText": True, "autoHeight": True}
        if col == "Value":
            col_def["comparator"] = {"function": "MoneyComparator"}
            
        if chat_action == "SORT" and is_risk_target:
             target_col = chat_cmd["params"].get("column", "").lower()
             if col.lower() == target_col or target_col in col.lower():
                 col_def["sort"] = chat_cmd["params"].get("direction", "desc")
        risk_column_defs.append(col_def)

    risk_table = dag.AgGrid(
        id="overview-risk-grid",
        rowData=risk_df.to_dict('records'),
        columnDefs=risk_column_defs,
        defaultColDef={"sortable": True, "filter": True, "resizable": True},
        className="ag-theme-alpine-dark audit-target",
        dashGridOptions={"domLayout": "autoHeight", "headerHeight": 0}
    )
    
    # 4. Flows Table
    flows_df = dw.get_flows_summary_ytd(data)
    is_flow_target = "flow" in chat_target
    
    flows_column_defs = []
    for col in ["Metric", "Value"]:
        col_def = {"field": col, "headerName": "", "flex": 1, "minWidth": 150, "wrapText": True, "autoHeight": True}
        if col == "Value":
            col_def["comparator"] = {"function": "MoneyComparator"}
            
        if chat_action == "SORT" and is_flow_target:
             target_col = chat_cmd["params"].get("column", "").lower()
             if col.lower() == target_col or target_col in col.lower():
                 col_def["sort"] = chat_cmd["params"].get("direction", "desc")
        flows_column_defs.append(col_def)

    flows_table = dag.AgGrid(
        id="overview-flows-grid",
        rowData=flows_df.to_dict('records'),
        columnDefs=flows_column_defs,
        defaultColDef={"sortable": True, "filter": True, "resizable": True},
        className="ag-theme-alpine-dark audit-target",
        dashGridOptions={"domLayout": "autoHeight", "headerHeight": 0}
    )

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

    return status_note, price_badge, val_card, twr_card, pl_card, alpha_card, mtd_card, fig, snap_table, high_table, risk_table, flows_table, alpha_card_cash
