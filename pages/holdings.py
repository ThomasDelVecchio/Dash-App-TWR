import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import dash_wrappers as dw
import pandas as pd
from data_loader import fetch_etf_sectors
from report_formatting import fmt_pct_clean, fmt_dollar_clean, fmt_number_clean
from components.page_header import page_header
import config

layout = html.Div([
    # --- HEADER ---
    page_header(
        title="Holdings",
        icon="bi-wallet2",
        subtitle="Security-level returns, weights, and classifications"
    ),
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H5("Current Holdings", className="card-title section-header p-2"),
            dcc.Loading(html.Div(id='holdings-table-container'))
        ]), width=12, className="mb-4"),
    ]),
    
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H5("Ticker Allocation", className="card-title section-header p-2"),
            dcc.Graph(id={'type': 'filter-chart', 'index': 'ticker-pie-chart'})
        ]), width=6, className="mb-4"),
        dbc.Col(dbc.Card([
            html.H5("Ticker Allocation vs Target", className="card-title section-header p-2"),
            dcc.Graph(id={'type': 'filter-chart', 'index': 'ticker-bar-chart'})
        ]), width=6, className="mb-4"),
    ])
])

@callback(
    [Output('holdings-table-container', 'children'),
     Output({'type': 'filter-chart', 'index': 'ticker-pie-chart'}, 'figure'),
     Output({'type': 'filter-chart', 'index': 'ticker-bar-chart'}, 'figure')],
    [Input('data-signal', 'data'),
     Input('filter-store', 'data'),
     Input('chatbot-command', 'data'),
     Input('include-exited-store', 'data')],
    [State('date-range-store', 'data')]
)
def update_holdings(signal, filters, chat_cmd, include_exited, dates):
    data = dw.get_data()
    if not data: return "Loading...", {}, {}
    
    # Dynamic 1D Label
    report_end_date = dates.get("end") if dates else None
    label_1d = dw.get_display_label_for_1d(report_end_date)
    
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
    
    # Calculate global annualization status
    is_port_annualized = dw.is_annualized(data["inception_date"], data.get("effective_as_of"))

    # Prepare column definitions for AG Grid
    return_cols = ["1D", "1W", "MTD", "1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "SI"]
    column_defs = []
    
    for col in df.columns:
        # Format Header Name: capitalize first letters and replace underscores/hyphens
        if col == "1D":
            header_name = label_1d
        elif col in return_cols:
            if col == "SI" and is_port_annualized:
                header_name = "SI (Ann.)"
            else:
                header_name = col
        else:
            header_name = col.replace('_', ' ').replace('-', ' ').title()
        
        col_def = {
            "field": col, 
            "headerName": header_name, 
            "comparator": {"function": "MoneyComparator"}
        }

        # Manual Column Sizing (Match Performance Page)
        if col == "asset_class":
             col_def["minWidth"] = 190
        elif col == "market_value":
             col_def["minWidth"] = 150
        elif col == "1D":
             col_def["minWidth"] = 130
        elif col in ["first_date", "last_date", "days_held"]:
             col_def["minWidth"] = 140
        
        # Freeze First Column and ensure mobile readability
        if col == "ticker":
            col_def["headerName"] = "Ticker"
            col_def["pinned"] = "left"
            col_def["lockPinned"] = True
            col_def["cellClass"] = "lock-pinned"
            col_def["width"] = 110
            col_def["suppressSizeToFit"] = True
        
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
            defaultColDef={"flex": 1, "minWidth": 110, "sortable": True, "filter": True, "resizable": True},
            className="ag-theme-alpine-dark audit-target",
            dashGridOptions={"domLayout": "autoHeight"}
        ), style={'overflowX': 'auto'}
    )
    
    # Charts
    pie_fig, bar_fig = dw.get_ticker_allocation_charts(data, "dark")
    
    return table, pie_fig, bar_fig
