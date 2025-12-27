import dash
from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import dash_wrappers as dw
from report_formatting import fmt_pct_clean, fmt_dollar_clean
import pandas as pd
from components.ai_brief import generate_ai_summary

def create_kpi_card(title, value, subtext=None, is_positive=None):
    """
    Bloomberg-style KPI card with professional styling.
    """
    subtext_arrow = ""
    subtext_color = "#6c757d"  # Gray default
    main_arrow_span = None
    
    if is_positive is not None:
        if is_positive:
            color = "#28a745"  # Green
            symbol = "▲"
        else:
            color = "#dc3545"  # Red
            symbol = "▼"
            
        if subtext:
            subtext_arrow = f"{symbol} "
            subtext_color = color
            
        main_arrow_span = html.Span(
            f"{symbol} ", 
            style={
                'color': color, 
                'fontSize': '1.2rem', 
                'marginRight': '4px', 
                'verticalAlign': 'middle'
            }
        )
    
    h2_content = [main_arrow_span, value] if main_arrow_span else value
    
    card_content = [
        html.Div(title, className="text-muted small mb-1", style={'fontSize': '0.75rem', 'fontWeight': '500'}),
        html.H4(h2_content, className="mb-1", style={'fontWeight': '600', 'fontSize': '1.4rem'}),
    ]
    
    # Force a placeholder if subtext is missing to maintain height
    subtext_display = f"{subtext_arrow}{subtext}" if subtext else " "
    
    card_content.append(
        html.Div(
            subtext_display,
            style={
                'fontSize': '0.8rem',
                'fontWeight': '500',
                'color': subtext_color if subtext else 'transparent'
            }
        )
    )
    
    return dbc.Card(
        dbc.CardBody(card_content, className="p-2"),
        className="shadow-sm",
        style={
            'borderLeft': f'4px solid {subtext_color if is_positive is not None else "#4C6A92"}',
            'height': '100%'
        }
    )

layout = html.Div([
    # Data Status Note
    html.Div(id='data-status-container', style={'position': 'fixed', 'top': '15px', 'right': '20px', 'zIndex': 2000, 'maxWidth': '90vw'}),

    # Morning Brief AI Card (NEW)
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader([
                html.I(className="bi bi-robot me-2"),
                "Morning Brief (AI Summary)"
            ]),
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
        
        # MTD
        dbc.Col(html.Div(id='kpi-mtd-card', style={'height': '100%'}), width=2),

        # Sharpe
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.Div("Sharpe Ratio", className="text-muted small mb-1", style={'fontSize': '0.75rem', 'fontWeight': '500'}),
                html.H4(id="perf-sharpe-val", className="text-primary fw-bold mb-1", style={'fontWeight': '600', 'fontSize': '1.4rem'}),
                html.Div("Risk-Adj (Total)", className="text-muted", style={'fontSize': '0.8rem', 'fontWeight': '500'})
            ], className="p-2")
        ], className="shadow-sm border-start border-4 border-primary", style={'height': '100%'}), width=2),

        # Sortino
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.Div("Sortino Ratio", className="text-muted small mb-1", style={'fontSize': '0.75rem', 'fontWeight': '500'}),
                html.H4(id="perf-sortino-val", className="text-success fw-bold mb-1", style={'fontWeight': '600', 'fontSize': '1.4rem'}),
                html.Div("Risk-Adj (Down)", className="text-muted", style={'fontSize': '0.8rem', 'fontWeight': '500'})
            ], className="p-2")
        ], className="shadow-sm border-start border-4 border-success", style={'height': '100%'}), width=2),
    ], className="mb-4 g-2"),

    # Chart & Snapshot Row
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H5("Portfolio Value (Since Inception %)", className="card-title p-2"),
            dcc.Graph(id='pv-chart', style={'height': '400px'})
        ]), width=7),
        dbc.Col(dbc.Card([
            html.H5("Portfolio Snapshot", className="card-title p-2"),
            dcc.Loading(html.Div(id='snapshot-table-container'))
        ]), width=5)
    ], className="mb-4"),
    
    # Highlights & Risk Row
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H5("Performance Highlights", className="card-title p-2"),
            dcc.Loading(html.Div(id='highlights-table-container'))
        ]), width=6),
        dbc.Col(dbc.Card([
            html.H5("Risk & Diversification", className="card-title p-2"),
            dcc.Loading(html.Div(id='risk-table-container'))
        ]), width=6)
    ], className="mb-4"),
    
    # Flows Row
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H5("Flows Summary (YTD)", className="card-title p-2"),
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
     Output('kpi-val-card', 'children'),
     Output('kpi-twr-card', 'children'),
     Output('kpi-pl-card', 'children'),
     Output('kpi-mtd-card', 'children'),
     Output('pv-chart', 'figure'),
     Output('snapshot-table-container', 'children'),
     Output('highlights-table-container', 'children'),
     Output('risk-table-container', 'children'),
     Output('flows-table-container', 'children'),
     Output('perf-sharpe-val', 'children'),
     Output('perf-sortino-val', 'children')],
    [Input('data-signal', 'data'),
     Input('theme-store', 'data'),
     Input('chatbot-command', 'data'),
     Input('filter-store', 'data')]
)
def update_overview(signal, theme, chat_cmd, _filters):
    data = dw.get_data()
    if not data:
        return None, "...", "...", "...", "...", {}, "Loading...", "Loading...", "Loading...", "Loading...", "N/A", "N/A"
    
    # Data Status Note
    status_note = None
    # Robust error check: Look for explicit key first, then fallback to attrs
    errors = data.get('errors', [])
    if not errors and 'prices' in data and hasattr(data['prices'], 'attrs'):
        errors = data['prices'].attrs.get('errors', [])
        
    if errors:
        print(f"DEBUG: pages/overview.py received {len(errors)} errors.")
        # Show ALL errors to ensure the consolidated message is fully visible
        # Responsive & visible on mobile/iPad
        status_note = dbc.Alert(
            [
                html.Div([
                    html.I(className="bi bi-exclamation-triangle-fill me-2"), 
                    html.Span("Data Quality Warning", className="fw-bold")
                ], className="d-flex align-items-center mb-1"),
                html.Hr(className="my-1"),
            ] + 
            [html.Div(e, className="small mb-2", style={'whiteSpace': 'pre-wrap', 'wordBreak': 'break-word'}) for e in errors],
            color="warning",
            dismissable=True,
            className="py-2 px-3 shadow-sm",
            style={
                'width': 'auto',
                'minWidth': '300px',
                'maxWidth': '100%',
                'maxHeight': '80vh',
                'overflowY': 'auto',
                'border': '1px solid #ffc107', 
                'fontSize': '0.85rem',
                'opacity': '0.95'
            }
        )

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
    twr_card = create_kpi_card("Inception TWR (Ann)", twr, subtext=None, is_positive=twr_is_positive)
    
    # P/L card with indicator
    pl_is_positive = metrics['pl_si'] >= 0 if metrics['pl_si'] is not None else None
    pl_card = create_kpi_card("Inception P/L", pl, is_positive=pl_is_positive)
    
    # MTD card with indicator
    mtd_is_positive = metrics['mtd_ret'] >= 0 if metrics['mtd_ret'] is not None else None
    mtd_card = create_kpi_card("MTD Return", mtd, is_positive=mtd_is_positive)
            
    # Chart
    fig = dw.get_pv_mountain_chart(data, theme)
    
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
            "domLayout": "autoHeight",
        }
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

    # Efficiency Metrics
    sharpe_val = f"{metrics['sharpe']:.2f}" if isinstance(metrics['sharpe'], (int, float)) else "N/A"
    sortino_val = f"{metrics['sortino']:.2f}" if isinstance(metrics['sortino'], (int, float)) else "N/A"
    
    return status_note, val_card, twr_card, pl_card, mtd_card, fig, snap_table, high_table, risk_table, flows_table, sharpe_val, sortino_val
