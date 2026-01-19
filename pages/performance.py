import dash
from dash import dcc, html, callback, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import dash_wrappers as dw
from report_formatting import fmt_pct_clean, fmt_dollar_clean
from config import RISK_FREE_RATE
from components.data_source_badge import create_price_source_badge
import pandas as pd

layout = html.Div([
    
    # --- HEADER ---
    dbc.Row([
        dbc.Col([
            html.H2("Performance", className="fw-bold text-body"),
            html.P("Cumulative returns, horizon analysis, and benchmark comparison", className="text-muted small")
        ], width=12)
    ], className="mb-4"),
    
    # Price Source Badge (Fixed position)
    html.Div(id='perf-price-source-badge-container', style={'position': 'fixed', 'top': '15px', 'right': '20px', 'zIndex': 1999}),

    dbc.Row([
        dbc.Col(dbc.Card([
            html.H5("Cumulative Return vs Benchmarks", className="card-title p-2"),
            dcc.Graph(id='cum-ret-chart')
        ]), width=12, className="mb-4"),
    ]),
    
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H5("Excess Return vs Benchmarks", className="card-title p-2"),
            dcc.Graph(id='excess-ret-chart')
        ]), width=12, className="mb-4"),
    ]),
    
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader([
                dbc.Row([
                    dbc.Col(html.H5("Horizon Returns (Modified Dietz)", className="card-title p-0 m-0"), width=True),
                    dbc.Col([
                        dbc.Button("Expand All", id="btn-ret-expand", size="sm", color="light", outline=True, className="me-2", n_clicks=0),
                        dbc.Button("Collapse All", id="btn-ret-collapse", size="sm", color="light", outline=True, n_clicks=0),
                    ], width="auto")
                ], align="center")
            ]),
            dbc.CardBody(dcc.Loading([
                dbc.Accordion(id='perf-ret-accordion', always_open=True, active_item=[], flush=True),
                html.Div(id='perf-ret-footnote', className="text-muted small mt-2")
            ]))
        ]), width=12, className="mb-4"),
    ]),
    
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader([
                dbc.Row([
                    dbc.Col(html.H5("Horizon P/L (Economic)", className="card-title p-0 m-0"), width=True),
                    dbc.Col([
                        dbc.Button("Expand All", id="btn-pl-expand", size="sm", color="light", outline=True, className="me-2", n_clicks=0),
                        dbc.Button("Collapse All", id="btn-pl-collapse", size="sm", color="light", outline=True, n_clicks=0),
                    ], width="auto")
                ], align="center")
            ]),
            dbc.CardBody(dcc.Loading([
                dbc.Accordion(id='perf-pl-accordion', always_open=True, active_item=[], flush=True),
                html.Div(id='perf-pl-footnote', className="text-muted small mt-2")
            ]))
        ]), width=12, className="mb-4"),
    ]),
    
    # Growth of Invested Capital Section
    html.Hr(className="my-4"),
    dbc.Row([
        dbc.Col([
            html.H4("Growth of Invested Capital", className="mb-3 text-light"),
            html.P("Compare portfolio value vs cumulative cash invested by asset class", className="text-muted")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Label("Select Asset Class:"),
            dcc.Dropdown(
                id='growth-asset-class-filter',
                options=[{"label": "Total", "value": "Total"}],  # Will be populated dynamically
                value="Total",
                clearable=False,
                className="mb-3"
            )
        ], width=4)
    ]),
    
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H5("Portfolio Value vs Cash Invested", className="card-title p-2"),
            dcc.Loading(dcc.Graph(id='growth-of-capital-chart'))
        ]), width=12, className="mb-4"),
    ]),
    
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H5("Investment Summary by Asset Class", className="card-title p-2"),
            dcc.Loading(html.Div(id='growth-table-container'))
        ]), width=12, className="mb-4"),
    ]),
])

# Price Source Badge Callback
@callback(
    Output('perf-price-source-badge-container', 'children'),
    [Input('data-signal', 'data')]
)
def update_perf_price_badge(signal):
    data = dw.get_data()
    if not data:
        return None
    
    price_source_meta = dw.get_price_source_summary(data)
    return create_price_source_badge(price_source_meta, "perf-price-badge") if price_source_meta else None

@callback(
    [Output('cum-ret-chart', 'figure'),
     Output('excess-ret-chart', 'figure'),
     Output('perf-ret-accordion', 'children'),
     Output('perf-ret-footnote', 'children'),
     Output('perf-pl-accordion', 'children'),
     Output('perf-pl-footnote', 'children')],
    [Input('data-signal', 'data'),
     Input('date-range-store', 'data'),
     Input('benchmark-store', 'data'),
     Input('chatbot-command', 'data'),
     Input('filter-store', 'data'),
     Input('include-exited-store', 'data')]
)
def update_performance(signal, dates, benchmarks, chat_cmd, _filters, include_exited):
    data = dw.get_data()
    if not data: return {}, {}, [], "", [], ""
    
    # --- CHATBOT PARAMS ---
    chat_target = ""
    chat_action = None
    if chat_cmd:
        chat_action = chat_cmd.get("action")
        chat_target = chat_cmd.get("params", {}).get("target", "").lower()

    # Start date is always inception (handled by wrapper if None)
    start_date = None 
    bm_map = benchmarks if benchmarks else {"S&P 500": "SPY"}

    # Dynamic 1D Label
    report_end_date = dates.get("end") if dates else None
    label_1d = dw.get_display_label_for_1d(report_end_date)
    
    # 1. Charts
    cum_fig = dw.get_cumulative_return_chart(data, start_date, bm_map, "dark")
    exc_fig = dw.get_excess_return_chart(data, bm_map, "dark")
    
    # Initialize Data
    class_df = data['class_df']
    if include_exited:
         sec_table_display = data['sec_table']
    else:
         sec_table_display = data['sec_table_current']
    
    horizons = ["1D", "1W", "MTD", "1M", "3M", "6M", "YTD", "1Y", "SI"]
    cols = ["Asset Class / Ticker"] + horizons + ["Sharpe (SI)", "Vol (SI)"] 
    risk_data = data.get("risk_return", {}) # {AC: {return: %, vol: %}}
    ac_rank_map = {ac: i for i, ac in enumerate(class_df['asset_class'].unique())}

    # =========================================================================
    # 2. HORIZON RETURNS TABLE (ACCORDION)
    # =========================================================================
    
    # Define Columns (Shared across all accordion grids)
    ret_column_defs = []
    
    # Check Annualization
    is_port_annualized = dw.is_annualized(data['inception_date'], data.get('effective_as_of'))

    # Check Sort Target
    is_ret_target = "return" in chat_target or (not chat_target and not "p/l" in chat_target)
    
    for col in cols:
        header_text = col
        if col == "1D": header_text = label_1d
        
        # Dynamic SI Annualization Label
        if col == "SI" and is_port_annualized:
            header_text = "SI (Ann.)"

        col_def = {
            "field": col, 
            "headerName": header_text, 
            "comparator": {"function": "GroupedRowComparator"}
        }
        
        # Freeze First Column
        if col == "Asset Class / Ticker":
            col_def["pinned"] = "left"
            col_def["minWidth"] = 180
            col_def["lockPinned"] = True
            col_def["cellClass"] = "lock-pinned"
        
        # Chatbot Sort
        if chat_action == "SORT" and is_ret_target:
             target_col = chat_cmd["params"].get("column", "").lower()
             if col.lower() == target_col or target_col in col.lower():
                 col_def["sort"] = chat_cmd["params"].get("direction", "desc")

        # Hide Audit Meta Columns
        if col.startswith("meta_"):
            col_def["hide"] = True

        # Conditional Styling
        if col in horizons:
            col_def["cellStyle"] = {
                "styleConditions": [
                    {"condition": "params.value && params.value.includes('-')", "style": {"color": "#dc3545"}},
                    {"condition": "params.value && !params.value.includes('-') && params.value !== 'N/A'", "style": {"color": "#28a745"}}
                ]
            }
        
        # Risk Metrics Config
        if "Sharpe" in col or "Vol" in col:
            col_def["sortable"] = True
            col_def["minWidth"] = 100
            
        ret_column_defs.append(col_def)

    # Build Accordion Items
    ret_accordion_items = []
    
    for _, crow in class_df.iterrows():
        ac = crow['asset_class']
        rank = ac_rank_map.get(ac, 999)
        local_rows = []
        
        # --- A. Class Row ---
        ac_risk = risk_data.get(ac, {})
        sharpe_str = "N/A"
        vol_str = "N/A"
        
        if ac_risk:
            # Use pre-calculated metrics from backend (GIPS compliant)
            vol_val = ac_risk.get("vol", 0.0)
            sharpe_val = ac_risk.get("sharpe", None)
            
            # Retrieve arithmetic return for Audit Audit
            arith_ret = ac_risk.get("arith_return", ac_risk.get("return", 0.0))
            
            vol_str = f"{vol_val:.1f}%"
            
            if sharpe_val is not None:
                sharpe_str = f"{sharpe_val:.2f}"
            else:
                # Fallback if backend keys missing (legacy safety)
                rf_pct = RISK_FREE_RATE * 100.0
                if vol_val > 0:
                    sharpe_calc = (arith_ret - rf_pct) / vol_val
                    sharpe_str = f"{sharpe_calc:.2f}"
        
        r_vals = {
            "Asset Class / Ticker": " Total", # Indented slightly less than ticker? Space aligns text.
            "Type": "Class", 
            "_sort_rank": rank, 
            "_is_header": 1,
            "Sharpe (SI)": sharpe_str,
            "Vol (SI)": vol_str,
            
            # INJECT META DATA FOR AUDIT MODAL (Generic Keys)
            "meta_Sharpe_vol": ac_risk.get("vol", 0.0),
            "meta_Sharpe_ret": ac_risk.get("arith_return", 0.0),
            "meta_Sharpe_rf": RISK_FREE_RATE * 100.0,
            
            "meta_Vol_vol": ac_risk.get("vol", 0.0)
        }
        # Copy Meta
        for k, v in crow.items():
            if str(k).startswith("meta_"): r_vals[k] = v
        # Copy Horizon Returns
        for h in horizons:
            val = crow.get(h)
            r_vals[h] = fmt_pct_clean(val) if pd.notna(val) else "N/A"
        
        # --- B. Ticker Rows ---
        tickers = sec_table_display[sec_table_display['asset_class'] == ac]
        for _, trow in tickers.iterrows():
            t = trow['ticker']
            tr_vals = {
                "Asset Class / Ticker": f"  {t}", 
                "Type": "Ticker", 
                "_sort_rank": rank,
                "Sharpe (SI)": "",
                "Vol (SI)": ""
            }
            # Copy Meta
            for k, v in trow.items():
                if str(k).startswith("meta_"): tr_vals[k] = v
            # Copy Horizon Returns
            for h in horizons:
                val = trow.get(h)
                tr_vals[h] = fmt_pct_clean(val) if pd.notna(val) else "N/A"
            local_rows.append(tr_vals)
            
        # Insert Total Row at Top
        local_rows.insert(0, r_vals)

        # --- C. Create Grid for this Class ---
        # Add meta columns to defs if needed (dynamic check)
        local_defs = ret_column_defs[:]
        # Use r_vals or first local_row for meta check
        sample = r_vals if r_vals else (local_rows[0] if local_rows else {})
        if sample:
            meta_keys = [k for k in sample.keys() if k.startswith("meta_")]
            for mk in meta_keys:
                if mk not in [c["field"] for c in local_defs]:
                    local_defs.append({"field": mk, "hide": True})
        
        item_grid = dag.AgGrid(
            id={"type": "perf-ac-ret-grid", "index": str(rank)}, # Pattern Matching ID
            rowData=local_rows,
            columnDefs=local_defs,
            defaultColDef={"flex": 1, "minWidth": 100, "sortable": True, "filter": True, "resizable": True},
            className="ag-theme-alpine-dark audit-target",
            dashGridOptions={
                "domLayout": "autoHeight",
                "getRowStyle": {
                    "function": "params.data.Type === 'Class' ? {'fontWeight': 'bold', 'backgroundColor': 'rgba(255,255,255,0.05)'} : {}"
                }
            }
        )
        
        # --- D. Header Text ---
        # Include summary stats in the header
        si_ret = r_vals.get("SI", "N/A")
        header_text = f"{ac}  |  SI Return: {si_ret}  |  Vol: {vol_str}"
        
        ret_accordion_items.append(
            dbc.AccordionItem(
                item_grid,
                title=header_text,
                item_id=f"item-ret-{rank}"
            )
        )
        
    # Returns Accordion Items are ready in ret_accordion_items
    
    
    # =========================================================================
    # 3. HORIZON P/L TABLE (ACCORDION)
    # =========================================================================
    
    # Check Sort Target
    is_pl_target = "p/l" in chat_target or "profit" in chat_target

    pl_column_defs = []
    for col in cols:
        if "Sharpe" in col or "Vol" in col: continue # Exclude Risk
        
        header_text = col
        if col == "1D": header_text = label_1d
        
        col_def = {
            "field": col, 
            "headerName": header_text, 
            "comparator": {"function": "GroupedRowComparator"}
        }
        if col == "Asset Class / Ticker":
            col_def["pinned"] = "left"
            col_def["minWidth"] = 180
            col_def["lockPinned"] = True
            col_def["cellClass"] = "lock-pinned"
            
        if chat_action == "SORT" and is_pl_target:
             target_col = chat_cmd["params"].get("column", "").lower()
             if col.lower() == target_col or target_col in col.lower():
                 col_def["sort"] = chat_cmd["params"].get("direction", "desc")
        
        if col.startswith("meta_"): col_def["hide"] = True

        if col in horizons:
            col_def["cellStyle"] = {
                "styleConditions": [
                    {"condition": "params.value && params.value.includes('-')", "style": {"color": "#dc3545"}},
                    {"condition": "params.value && !params.value.includes('-') && params.value !== 'N/A' && params.value !== '$0'", "style": {"color": "#28a745"}}
                ]
            }
        pl_column_defs.append(col_def)

    # Pre-fetch ticker P/L
    ticker_pl_cache = {}
    for h in horizons:
        ticker_pl_cache[h] = dw.get_ticker_pl_df(data, h)

    pl_accordion_items = []
    
    for _, crow in class_df.iterrows():
        ac = crow['asset_class']
        rank = ac_rank_map.get(ac, 999)
        local_rows = []
        
        # --- A. Class Row ---
        r_vals = {"Asset Class / Ticker": " Total", "Type": "Class", "_sort_rank": rank, "_is_header": 1}
        for k, v in crow.items():
            if str(k).startswith("meta_"): r_vals[k] = v
            
        for h in horizons:
            res = dw.get_asset_class_pl(data, ac, h, return_components=True)
            if isinstance(res, dict):
                r_vals[h] = fmt_dollar_clean(res["pl"])
                r_vals[f"meta_{h}_start"] = res["start"]
                r_vals[f"meta_{h}_end"] = res["end"]
                r_vals[f"meta_{h}_flow"] = res["flow"]
                r_vals[f"meta_{h}_inc"] = res["inc"]
                r_vals[f"meta_{h}_denom"] = res["denom"]
                r_vals[f"meta_{h}_start_date"] = res.get("start_date")
                r_vals[f"meta_{h}_end_date"] = res.get("end_date")
            else:
                r_vals[h] = fmt_dollar_clean(res) if res is not None else "N/A"
        
        # --- B. Ticker Rows ---
        tickers = sec_table_display[sec_table_display['asset_class'] == ac]
        for _, t_row_full in tickers.iterrows():
            t = t_row_full['ticker']
            tr_vals = {"Asset Class / Ticker": f"  {t}", "Type": "Ticker", "_sort_rank": rank}
            for k, v in t_row_full.items():
                if str(k).startswith("meta_"): tr_vals[k] = v
            
            for h in horizons:
                df = ticker_pl_cache[h]
                if not df.empty and t in df.index:
                    pl_val = df.loc[t, 'pl']
                    tr_vals[h] = fmt_dollar_clean(pl_val) if pl_val is not None else "N/A"
                    for col in df.columns:
                        if col.startswith(f"meta_{h}_"): tr_vals[col] = df.loc[t, col]
                else:
                    tr_vals[h] = "N/A"
            local_rows.append(tr_vals)
            
        # Insert Total Row at Top
        if r_vals:
            local_rows.insert(0, r_vals)

        # --- C. Create Grid ---
        local_defs = pl_column_defs[:]
        # Use r_vals or first row for meta
        sample = r_vals if r_vals else (local_rows[0] if local_rows else {})
        if sample:
            meta_keys = [k for k in sample.keys() if k.startswith("meta_")]
            for mk in meta_keys:
                if mk not in [c["field"] for c in local_defs]:
                    local_defs.append({"field": mk, "hide": True})
        
        item_grid = dag.AgGrid(
            id={"type": "perf-ac-pl-grid", "index": str(rank)},
            rowData=local_rows,
            columnDefs=local_defs,
            defaultColDef={"flex": 1, "minWidth": 100, "sortable": True, "filter": True, "resizable": True},
            className="ag-theme-alpine-dark audit-target",
            dashGridOptions={
                "domLayout": "autoHeight",
                "getRowStyle": {
                    "function": "params.data.Type === 'Class' ? {'fontWeight': 'bold', 'backgroundColor': 'rgba(255,255,255,0.05)'} : {}"
                }
            }
        )
        
        si_pl = r_vals.get("SI", "N/A")
        header_text = f"{ac}  |  SI P/L: {si_pl}"
        
        pl_accordion_items.append(
            dbc.AccordionItem(item_grid, title=header_text, item_id=f"item-pl-{rank}")
        )
        
    
    # --- Cash / Recon (Outside Accordion) ---
    cash_recon_vals = dw.get_cash_recon_pl(data, horizons)
    recon_row = {"Asset Class / Ticker": "Cash / Recon", "Type": "Recon"}
    for h in horizons:
        val_obj = cash_recon_vals.get(h)
        if val_obj and isinstance(val_obj, dict):
            pl_val = val_obj.get("pl")
            recon_row[h] = fmt_dollar_clean(pl_val) if pl_val is not None else "N/A"
            recon_row[f"meta_{h}_start_date"] = val_obj.get("start_date")
            recon_row[f"meta_{h}_end_date"] = val_obj.get("end_date")
        else:
            recon_row[h] = "N/A"
    
    # Create distinct grid for Recon
    recon_grid = dag.AgGrid(
        id="perf-recon-grid",
        rowData=[recon_row],
        columnDefs=pl_column_defs, # Re-use pl defs
        defaultColDef={"flex": 1, "minWidth": 100, "resizable": True},
        className="ag-theme-alpine-dark audit-target",
        dashGridOptions={
            "domLayout": "autoHeight", 
            "headerHeight": 0, # Hide Header to blend in? Or keep it? Keeping it ensures column alignment visual.
                               # Actually header makes it look like a new table. 
                               # If we hide header, columns won't align visually with above tables if screen resizes.
                               # It's safer to show header or wrap in a card titled "Reconciliation".
        }
    )
    
    # Add Recon as the last Accordion Item?
    pl_accordion_items.append(
        dbc.AccordionItem(
            recon_grid, 
            title="Reconciliation (Cash & Fees)", 
            item_id="item-pl-recon"
        )
    )

    # Dynamic Footnote Text
    visibility_text = "Tables display ALL positions (active and exited) with valid history in the period." if include_exited else "Tables display currently active positions only."

    ret_footnote_text = f"Note: {visibility_text} Asset Class totals ALWAYS include historical contribution of closed positions (GIPS compliant). " \
                        "Returns require full measurement period (e.g., 1M return requires 30+ days of history)."
    
    pl_footnote_text = f"Note: {visibility_text} P/L shows actual economic gain/loss (MV_End - MV_Start - Net_Flows + Income). " \
                       "This reflects actual economic outcomes regardless of holding period."

    return cum_fig, exc_fig, ret_accordion_items, ret_footnote_text, pl_accordion_items, pl_footnote_text

# Growth of Invested Capital Callbacks

@callback(
    Output('growth-asset-class-filter', 'options'),
    [Input('data-signal', 'data')]
)
def update_growth_dropdown_options(signal):
    """Dynamically populate dropdown with asset classes that have non-zero shares."""
    data = dw.get_data()
    if not data:
        return [{"label": "Total", "value": "Total"}]
    
    sec_table_current = data.get('sec_table_current')
    if sec_table_current is None or sec_table_current.empty:
        return [{"label": "Total", "value": "Total"}]
    
    # Get unique asset classes with non-zero shares (excluding CASH for cleaner view)
    asset_classes = sec_table_current[
        (sec_table_current['shares'].abs() > 1e-6) & 
        (sec_table_current['asset_class'] != 'CASH')
    ]['asset_class'].unique().tolist()
    
    # Sort alphabetically
    asset_classes = sorted(asset_classes)
    
    # Build options list
    options = [
        {"label": "Total", "value": "Total"},
        {"label": "All Asset Classes", "value": "All"}
    ]
    
    for ac in asset_classes:
        options.append({"label": ac, "value": ac})
    
    return options

@callback(
    [Output('growth-of-capital-chart', 'figure'),
     Output('growth-table-container', 'children')],
    [Input('data-signal', 'data'),
     Input('date-range-store', 'data'),
     Input('growth-asset-class-filter', 'value'),
     Input('chatbot-command', 'data'),
     Input('filter-store', 'data')]
)
def update_growth_analysis(signal, dates, selected_ac, chat_cmd, _filters):
    """Update Growth of Invested Capital chart and table."""
    data = dw.get_data()
    if not data:
        return {}, html.Div("Loading...", className="p-3")
        
    # Parse End Date
    end_date = None
    if dates and isinstance(dates, dict):
        end_date = dates.get("end")
        
    # --- CHATBOT PARAMS ---
    chat_target = ""
    chat_action = None
    if chat_cmd:
        chat_action = chat_cmd.get("action")
        chat_target = chat_cmd.get("params", {}).get("target", "").lower()
    
    # Generate chart
    try:
        chart_fig = dw.get_growth_of_capital_chart(data, selected_ac, "dark", end_date=end_date)
    except Exception as e:
        chart_fig = {}
        print(f"Error generating growth chart: {e}")
    
    # Generate table
    try:
        table_df = dw.get_growth_of_capital_table_data(data)
        
        if table_df.empty:
            table_output = html.Div("No data available", className="p-3")
        else:
            # Extract Total Row for pinning
            total_mask = table_df["Asset Class"] == "Total"
            pinned_rows = table_df[total_mask].to_dict('records')
            main_rows = table_df[~total_mask].to_dict('records')

            # Check Sort Target (Growth Table)
            is_growth_target = "growth" in chat_target or "invested" in chat_target

            # Create AG Grid column definitions for growth table
            growth_column_defs = []
            for col in table_df.columns:
                col_def = {"field": col, "headerName": col}
                
                # Ensure minimum width for first column
                if col == "Asset Class":
                    col_def["minWidth"] = 180

                # Hide Meta Columns
                if col.startswith("meta_"):
                    col_def["hide"] = True
                
                # Chatbot Sort
                if chat_action == "SORT" and is_growth_target:
                     target_col = chat_cmd["params"].get("column", "").lower()
                     if col.lower() == target_col or target_col in col.lower():
                         col_def["sort"] = chat_cmd["params"].get("direction", "desc")

                # Add numerical comparator
                if col in ["Cash Invested", "Portfolio Value", "Growth", "Growth %"]:
                    col_def["comparator"] = {"function": "MoneyComparator"}
                
                # Add conditional styling for Growth columns (green for positive, red for negative)
                if col in ["Growth", "Growth %"]:
                    col_def["cellStyle"] = {
                        "styleConditions": [
                            {"condition": "params.value && params.value.includes('-')", "style": {"color": "#dc3545"}},
                            {"condition": "params.value && !params.value.includes('-') && params.value !== 'N/A'", "style": {"color": "#28a745"}}
                        ]
                    }
                # Right align numeric columns
                if col in ["Cash Invested", "Portfolio Value", "Growth", "Growth %"]:
                    col_def["cellClass"] = "text-end"
                growth_column_defs.append(col_def)
            
            table_output = html.Div(
                dag.AgGrid(
                    id="perf-growth-grid",
                    rowData=main_rows,
                    columnDefs=growth_column_defs,
                    defaultColDef={"flex": 1, "minWidth": 120, "sortable": True, "filter": True, "resizable": True},
                    className="ag-theme-alpine-dark audit-target",
                    dashGridOptions={
                        "domLayout": "autoHeight",
                        "pinnedBottomRowData": pinned_rows,
                        "getRowStyle": {
                            "function": "params.data['Asset Class'] === 'Total' ? {'fontWeight': 'bold', 'backgroundColor': 'rgba(255,255,255,0.05)', 'borderTop': '2px solid #888'} : {}"
                        }
                    }
                ), style={'overflowX': 'auto'}
            )
    except Exception as e:
        table_output = html.Div(f"Error loading table: {str(e)}", className="p-3 text-danger")
        print(f"Error generating growth table: {e}")
    
    return chart_fig, table_output

# =============================================================================
# ACCORDION TOGGLE CALLBACKS
# =============================================================================

@callback(
    Output("perf-ret-accordion", "active_item"),
    [Input("btn-ret-expand", "n_clicks"),
     Input("btn-ret-collapse", "n_clicks")],
    [State("perf-ret-accordion", "children")]
)
def toggle_ret_accordion(n_exp, n_col, children):
    ctx = dash.callback_context
    if not ctx.triggered: return dash.no_update
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if not children: return []
    
    # Normalize to list if single component
    if not isinstance(children, list):
        children = [children]
        
    if button_id == "btn-ret-expand":
        # Extract item_ids from component props
        ids = []
        for item in children:
            if isinstance(item, dict) and 'props' in item:
                i_id = item['props'].get('item_id')
                if i_id: ids.append(i_id)
        return ids
    
    elif button_id == "btn-ret-collapse":
        return []
        
    return dash.no_update


@callback(
    Output("perf-pl-accordion", "active_item"),
    [Input("btn-pl-expand", "n_clicks"),
     Input("btn-pl-collapse", "n_clicks")],
    [State("perf-pl-accordion", "children")]
)
def toggle_pl_accordion(n_exp, n_col, children):
    ctx = dash.callback_context
    if not ctx.triggered: return dash.no_update
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if not children: return []
    
    # Normalize to list if single component
    if not isinstance(children, list):
        children = [children]
        
    if button_id == "btn-pl-expand":
        ids = []
        for item in children:
            if isinstance(item, dict) and 'props' in item:
                i_id = item['props'].get('item_id')
                if i_id: ids.append(i_id)
        return ids
    
    elif button_id == "btn-pl-collapse":
        return []
        
    return dash.no_update

# =============================================================================
# GRID INTERACTION CALLBACKS
# =============================================================================

@callback(
    Output('audit-request-store', 'data', allow_duplicate=True),
    [Input({'type': 'perf-ac-ret-grid', 'index': ALL}, 'cellClicked'),
     Input({'type': 'perf-ac-ret-grid', 'index': ALL}, 'cellContextMenu')],
    [State('audit-request-store', 'data')],
    prevent_initial_call=True
)
def handle_perf_grid_clicks(cell_clicks, right_clicks, current_data):
    ctx = dash.callback_context
    if not ctx.triggered: return dash.no_update
    
    # Identify Trigger
    trigger_str = ctx.triggered[0]['prop_id']
    if not trigger_str: return dash.no_update
    
    # Extract ID part before the .property
    trigger_id_str = trigger_str.split('.')[0]
    
    # Parse JSON ID
    try:
        import json
        trigger_id = json.loads(trigger_id_str)
    except:
        return dash.no_update
        
    # Get Value
    click_data = ctx.triggered[0]['value']
    if not click_data: return dash.no_update
    
    # Construct Request
    request = {
        'gridId': trigger_id,
        'colId': click_data.get('colId'),
        'rowData': click_data.get('data'),
        'rowIndex': click_data.get('rowIndex'),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    return request

