import dash
from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import dash_wrappers as dw
import pandas as pd
from data_loader import fetch_etf_sectors
from report_formatting import fmt_pct_clean, fmt_dollar_clean, fmt_number_clean
import config

layout = html.Div([
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H5("Current Holdings", className="card-title p-2"),
            dcc.Loading(html.Div(id='holdings-table-container'))
        ]), width=12, className="mb-4"),
    ]),
    
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H5("Illustrative Monthly Contribution Schedule", className="card-title p-2"),
            dcc.Loading(html.Div(id='monthly-contrib-container'))
        ]), width=12, className="mb-4"),
    ]),
    
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H5("Ticker Allocation", className="card-title p-2"),
            dcc.Graph(id={'type': 'filter-chart', 'index': 'ticker-pie-chart'})
        ]), width=6, className="mb-4"),
        dbc.Col(dbc.Card([
            html.H5("Ticker Allocation vs Target", className="card-title p-2"),
            dcc.Graph(id={'type': 'filter-chart', 'index': 'ticker-bar-chart'})
        ]), width=6, className="mb-4"),
    ])
])

@callback(
    [Output('holdings-table-container', 'children'),
     Output('monthly-contrib-container', 'children'),
     Output({'type': 'filter-chart', 'index': 'ticker-pie-chart'}, 'figure'),
     Output({'type': 'filter-chart', 'index': 'ticker-bar-chart'}, 'figure')],
    [Input('data-signal', 'data'),
     Input('theme-store', 'data'),
     Input('filter-store', 'data'),
     Input('chatbot-command', 'data'),
     Input('include-exited-store', 'data')]
)
def update_holdings(signal, theme, filters, chat_cmd, include_exited):
    data = dw.get_data()
    if not data: return "Loading...", "Loading...", {}, {}
    
    # Logic: Toggle view based on include_exited
    if include_exited:
         df = data['sec_table'].copy()
    else:
         df = data['sec_table_current'].copy()

    # --- CHATBOT PARAMS ---
    chat_target = ""
    chat_action = None
    if chat_cmd:
        chat_action = chat_cmd.get("action")
        chat_target = chat_cmd.get("params", {}).get("target", "").lower()

    # --- CHATBOT FILTER ---
    if chat_action == "FILTER":
        val = chat_cmd["params"].get("value")
        if val:
            # Simple text search across all columns
            mask = df.astype(str).apply(lambda x: x.str.contains(val, case=False, na=False)).any(axis=1)
            df = df[mask]
    
    # Filter Logic
    if filters:
        # 1. Asset Class
        if filters.get("asset_class"):
            # Filter matches exact asset class string
            df = df[df["asset_class"] == filters["asset_class"]]
            
        # 2. Ticker
        if filters.get("ticker"):
            df = df[df["ticker"] == filters["ticker"]]
            
        # 3. Sector
        if filters.get("sector"):
            target = filters["sector"]
            # Normalization map (Must match dash_wrappers logic)
            SECTOR_NORMALIZATION = {
                "Comm Services": "Communication Services",
                "Consumer Disc.": "Consumer Discretionary",
                "Information Technology": "Tech",
                "Other": None,
            }
            
            valid_tickers = []
            for t in df["ticker"].unique():
                # Check Dynamic Sector Map
                sectors = fetch_etf_sectors(t)
                for s_raw in sectors.keys():
                    norm = SECTOR_NORMALIZATION.get(s_raw, s_raw)
                    if norm == target:
                        valid_tickers.append(t)
                        break
                            
            df = df[df["ticker"].isin(valid_tickers)]
    
    # Prepare column definitions for AG Grid
    return_cols = ["1D", "1W", "MTD", "1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "SI"]
    column_defs = []
    
    for col in df.columns:
        col_def = {
            "field": col, 
            "headerName": col, 
            "comparator": {"function": "MoneyComparator"}
        }
        
        # Hide Audit Meta Columns
        if col.startswith("meta_"):
            col_def["hide"] = True
        
        # --- CHATBOT SORT (HOLDINGS) ---
        # Apply if target is generic ("table") or specific ("holdings")
        is_holdings_target = not chat_target or any(x in chat_target for x in ["holding", "current", "table", "grid", "list"])
        # If specific keywords for other tables exist, exclude this one unless explicitly named
        if any(x in chat_target for x in ["contribution", "schedule", "monthly"]):
             is_holdings_target = False

        if chat_action == "SORT" and is_holdings_target:
             target = chat_cmd["params"].get("column", "").lower()
             direction = chat_cmd["params"].get("direction", "desc")
             
             # Fuzzy match or exact match
             # Map common terms
             if target == "return": target = "si" # Default to SI for "return"
             if target == "value": target = "market_value"
             
             if col.lower() == target or target in col.lower():
                 col_def["sort"] = direction
                 col_def["sortIndex"] = 0
        
        # Conditional styling for return columns
        if col in return_cols:
            col_def["cellStyle"] = {
                "styleConditions": [
                    {"condition": "params.value && params.value.includes('-')", "style": {"color": "#dc3545"}},
                    {"condition": "params.value && !params.value.includes('-') && params.value !== 'N/A'", "style": {"color": "#28a745"}}
                ]
            }
        
        column_defs.append(col_def)
    
    # Format data
    df_display = df.copy()
    cols_to_format = ["market_value", "weight", "1D", "1W", "MTD", "1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "SI", "shares"]
    for c in cols_to_format:
        if c in df_display.columns:
            if "value" in c:
                df_display[c] = df_display[c].apply(fmt_dollar_clean)
            elif c == "shares":
                df_display[c] = df_display[c].apply(fmt_number_clean)
            else:
                # GIPS: Returns require full measurement period.
                # Indicate consistent N/A if there is no valid return for period.
                df_display[c] = df_display[c].apply(lambda x: fmt_pct_clean(x) if pd.notna(x) else "N/A")
                
    table = html.Div(
        dag.AgGrid(
            id="holdings-grid",
            rowData=df_display.to_dict('records'),
            columnDefs=column_defs,
            defaultColDef={"minWidth": 100, "sortable": True, "filter": True, "resizable": True},
            columnSize="autoSize",
            className="ag-theme-alpine-dark audit-target",
            dashGridOptions={"domLayout": "autoHeight"}
        ), style={'overflowX': 'auto'}
    )
    
    # Monthly Contribution Schedule
    contrib_df, footer_text, is_empty = dw.get_monthly_contribution_schedule(data)
    
    if is_empty:
        monthly_contrib_content = html.Div([
            html.P("All holdings are at or above target allocation. No contribution schedule needed.", 
                   style={'fontStyle': 'italic', 'color': 'gray', 'padding': '10px'})
        ])
    else:
        # Build definitions manually to support Chatbot Sort
        contrib_column_defs = []
        is_contrib_target = any(x in chat_target for x in ["contribution", "schedule", "monthly"])

        for col in contrib_df.columns:
            col_def = {"field": col, "headerName": col}
            
            # Hide meta columns
            if col.startswith("meta_"):
                col_def["hide"] = True
            
            # --- CHATBOT SORT (CONTRIB) ---
            if chat_action == "SORT" and is_contrib_target:
                target = chat_cmd["params"].get("column", "").lower()
                direction = chat_cmd["params"].get("direction", "desc")
                
                if col.lower() == target or target in col.lower():
                    col_def["sort"] = direction
                    col_def["sortIndex"] = 0
            
            contrib_column_defs.append(col_def)

        monthly_contrib_table = dag.AgGrid(
            id="contrib-grid",
            rowData=contrib_df.to_dict('records'),
            columnDefs=contrib_column_defs,
            defaultColDef={"flex": 1, "minWidth": 120, "sortable": True, "filter": True, "resizable": True},
            className="ag-theme-alpine-dark audit-target",
            dashGridOptions={"domLayout": "autoHeight"}
        )
        
        monthly_contrib_content = html.Div([
            monthly_contrib_table,
            html.P(footer_text, style={
                'fontSize': '9pt',
                'fontStyle': 'italic',
                'color': 'gray',
                'marginTop': '10px',
                'marginBottom': '0px'
            })
        ])
    
    # Charts
    pie_fig, bar_fig = dw.get_ticker_allocation_charts(data, theme)
    
    return table, monthly_contrib_content, pie_fig, bar_fig
