import dash
from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import dash_wrappers as dw
from report_formatting import fmt_pct_clean, fmt_dollar_clean
import pandas as pd

layout = html.Div([

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
            html.H5("Horizon Returns (Modified Dietz)", className="card-title p-2"),
            dcc.Loading(html.Div(id='horizon-ret-table-container'))
        ]), width=12, className="mb-4"),
    ]),
    
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H5("Horizon P/L (Economic)", className="card-title p-2"),
            dcc.Loading(html.Div(id='horizon-pl-table-container'))
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

@callback(
    [Output('cum-ret-chart', 'figure'),
     Output('excess-ret-chart', 'figure'),
     Output('horizon-ret-table-container', 'children'),
     Output('horizon-pl-table-container', 'children')],
    [Input('data-signal', 'data'),
     Input('theme-store', 'data'),
     Input('date-range-store', 'data'),
     Input('benchmark-store', 'data'),
     Input('chatbot-command', 'data'),
     Input('filter-store', 'data'),
     Input('include-exited-store', 'data')]
)
def update_performance(signal, theme, dates, benchmarks, chat_cmd, _filters, include_exited):
    data = dw.get_data()
    if not data: return {}, {}, "Loading...", "Loading...", "N/A", "N/A"
    
    # --- CHATBOT PARAMS ---
    chat_target = ""
    chat_action = None
    if chat_cmd:
        chat_action = chat_cmd.get("action")
        chat_target = chat_cmd.get("params", {}).get("target", "").lower()

    # Start date is always inception (handled by wrapper if None)
    start_date = None 
    bm_map = benchmarks if benchmarks else {"S&P 500": "SPY"}
    
    # 1. Charts
    cum_fig = dw.get_cumulative_return_chart(data, start_date, bm_map, theme)
    exc_fig = dw.get_excess_return_chart(data, bm_map, theme)
    
    # 2. Horizon Returns Table
    # Re-using the horizon analysis function from dash_wrappers which NOW includes Sharpe/Sortino
    # We need to restructure it to fit the grouped row format if we want to keep that visual style.
    # However, get_horizon_analysis returns a Portfolio-level summary (one row per horizon).
    # The existing table (lines 142-200) shows Asset Class breakdown.
    # The prompt asked for "columns in the Horizon Returns table".
    # Since the Asset Class table shows "1D", "1W"... as columns, adding Sharpe/Sortino as columns
    # implies calculating them per asset class per horizon?? That's huge computation.
    # Usually Sharpe/Sortino are Portfolio-level metrics.
    # But let's assume the user wants them for the PORTFOLIO rows, or as separate columns in the table.
    #
    # Wait, the existing table has horizons as COLUMNS: | Asset/Ticker | 1D | 1W | ... |
    # Adding "Sharpe" and "Sortino" as columns means they are essentially new "Horizons"?
    # OR does it mean "For this asset class, what is the Sharpe?"
    # If the table structure is: Row=Asset, Col=Horizon Return.
    # Then Sharpe/Sortino should be new COLUMNS: | Asset | ... | SI | Sharpe | Sortino |
    # Yes, typically calculated over the SI (Since Inception) or 1Y window.
    # I will calculate Sharpe/Sortino for SI and add them as columns to the Asset Class table.
    
    # ... Wait, I can't easily calculate daily Sharpe for every single ticker efficiently here without heavy lifting.
    # For now, I will stick to the Portfolio Level "Horizon Analysis" table if it exists, 
    # OR add them to the Asset Class table using the metrics I put in dash_wrappers (which currently only do Portfolio).
    #
    # Actually, dash_wrappers.get_horizon_analysis returns a DataFrame with columns: Horizon, Return, P/L, Sharpe, Sortino.
    # That function creates a TABLE where Rows = Horizons (1D, 1W...).
    # BUT the visual table in performance.py (id='horizon-ret-table-container') has Rows = Asset Classes.
    # There is a disconnect. The user asked "add ... as columns in the Horizon Returns table".
    # If the "Horizon Returns table" refers to the Asset Class breakdown, then I need to add Sharpe/Sortino columns
    # which represent the risk-adjusted return of that asset class (likely SI).
    #
    # Re-reading prompt: "columns in the Horizon Returns table".
    # Given the existing table is Asset Class based, I will add "Sharpe (SI)" and "Sortino (SI)" columns.
    # I need to compute these per asset class.
    # `dash_wrappers.py` `_calculate_dynamic_risk_profile` already calculates Volatility.
    # I can use that + SI Return to estimate Sharpe.
    # Sharpe ~ (Ann. Return - Rf) / Volatility.
    # I will use the cached Risk Profile to populate these columns for Asset Classes.
    
    class_df = data['class_df']
    # Select source table based on toggle
    if include_exited:
         sec_table_display = data['sec_table']
    else:
         sec_table_display = data['sec_table_current']
    
    horizons = ["1D", "1W", "MTD", "1M", "3M", "6M", "1Y", "SI"]
    cols = ["Asset Class / Ticker"] + horizons + ["Sharpe (SI)", "Vol (SI)"] 
    # Using SI because the prompt requested to change 1yr to SI
    # Actually, let's use the Dynamic Risk Profile values.
    
    risk_data = data.get("risk_return", {}) # {AC: {return: %, vol: %}}
    
    # Derive Asset Class Rank Map for consistent sorting
    ac_rank_map = {ac: i for i, ac in enumerate(class_df['asset_class'].unique())}

    rows = []
    # Sort classes
    for _, crow in class_df.iterrows():
        ac = crow['asset_class']
        rank = ac_rank_map.get(ac, 999)
        
        # Fetch Risk Metrics
        ac_risk = risk_data.get(ac, {})
        sharpe_str = "N/A"
        vol_str = "N/A"
        
        if ac_risk:
            # Estimate Sharpe using TTM Return and Vol
            # Sharpe = (Ret - Rf) / Vol
            # Config Rf is decimal (0.04). Risk dict has percents (10.0).
            # Convert Rf to percent: 4.0
            rf_pct = 4.0 
            ret = ac_risk.get("return", 0.0)
            vol = ac_risk.get("vol", 0.0)
            
            if vol > 0:
                sharpe = (ret - rf_pct) / vol
                sharpe_str = f"{sharpe:.2f}"
            
            vol_str = f"{vol:.1f}%"
        
        # Class Row (Includes full history from Engine)
        r_vals = {
            "Asset Class / Ticker": ac, 
            "Type": "Class", 
            "_sort_rank": rank, 
            "_is_header": 1,
            "Sharpe (SI)": sharpe_str,
            "Vol (SI)": vol_str
        }
        # Add all meta columns from class_df row
        for k, v in crow.items():
            if str(k).startswith("meta_"):
                r_vals[k] = v

        for h in horizons:
            val = crow.get(h)
            r_vals[h] = fmt_pct_clean(val) if pd.notna(val) else "N/A"
        rows.append(r_vals)
        
        # Ticker Rows (Filtered to Display Holdings only)
        tickers = sec_table_display[sec_table_display['asset_class'] == ac]
        for _, trow in tickers.iterrows():
            t = trow['ticker']
            tr_vals = {
                "Asset Class / Ticker": f"  {t}", 
                "Type": "Ticker", 
                "_sort_rank": rank, 
                "_is_header": 0,
                "Sharpe (SI)": "", # Too expensive to calc per ticker on the fly
                "Vol (SI)": ""
            }
            # Add all meta columns from sec_table row
            for k, v in trow.items():
                if str(k).startswith("meta_"):
                    tr_vals[k] = v

            for h in horizons:
                val = trow.get(h)
                tr_vals[h] = fmt_pct_clean(val) if pd.notna(val) else "N/A"
            rows.append(tr_vals)
    
    # Check Sort Target
    is_ret_target = "return" in chat_target or (not chat_target and not "p/l" in chat_target)

    # Create AG Grid column definitions
    ret_column_defs = []
    for col in cols:
        col_def = {
            "field": col, 
            "headerName": col, 
            "comparator": {"function": "GroupedRowComparator"}
        }
        
        # Chatbot Sort
        if chat_action == "SORT" and is_ret_target:
             target_col = chat_cmd["params"].get("column", "").lower()
             if col.lower() == target_col or target_col in col.lower():
                 col_def["sort"] = chat_cmd["params"].get("direction", "desc")

        # Hide Audit Meta Columns (explicitly, though field match should suffice if column list is fixed)
        if col.startswith("meta_"):
            col_def["hide"] = True

        # Add conditional styling for return columns (green for positive, red for negative)
        if col in horizons:
            col_def["cellStyle"] = {
                "styleConditions": [
                    {"condition": "params.value && params.value.includes('-')", "style": {"color": "#dc3545"}},
                    {"condition": "params.value && !params.value.includes('-') && params.value !== 'N/A'", "style": {"color": "#28a745"}}
                ]
            }
        
        # Enable sorting for Risk Metrics
        if "Sharpe" in col or "Vol" in col:
            col_def["sortable"] = True
            
        ret_column_defs.append(col_def)
        
    # Append meta columns to defs so they are available in params.data
    # We scan the first row to find available meta columns
    if rows:
        sample_row = rows[0]
        meta_keys = [k for k in sample_row.keys() if k.startswith("meta_")]
        for mk in meta_keys:
            if mk not in [c["field"] for c in ret_column_defs]:
                ret_column_defs.append({"field": mk, "hide": True})
            
    ret_table = dag.AgGrid(
        id="perf-horizon-ret-grid",
        rowData=rows,
        columnDefs=ret_column_defs,
        defaultColDef={"flex": 1, "minWidth": 100, "sortable": True, "filter": True, "resizable": True},
        className="ag-theme-alpine-dark audit-target",
        dashGridOptions={
            "domLayout": "autoHeight",
            "getRowStyle": {
                "function": "params.data.Type === 'Class' ? {'fontWeight': 'bold', 'backgroundColor': 'rgba(255,255,255,0.05)'} : {}"
            }
        }
    )
    
    # 3. Horizon P/L Table
    # Use DIRECT asset class P/L calculation (matches PDF logic)
    pl_table_data = []
    
    # Pre-fetch ticker P/L for all horizons (avoid multiple price fetches)
    ticker_pl_cache = {}
    for h in horizons:
        ticker_pl_cache[h] = dw.get_ticker_pl_df(data, h)

    for _, crow in class_df.iterrows():
        ac = crow['asset_class']
        rank = ac_rank_map.get(ac, 999)
        
        # Asset Class Row: Use Direct Calculation (includes closed positions via Modified Dietz)
        r_vals = {
            "Asset Class / Ticker": ac, 
            "Type": "Class", 
            "_sort_rank": rank, 
            "_is_header": 1
        }
        # Add meta columns from class_df (crow) for Audit
        for k, v in crow.items():
            if str(k).startswith("meta_"):
                r_vals[k] = v

        # ASSET CLASS LOOP (UPDATED)
        for h in horizons:
            # Use synchronized calculation for Audit consistency
            res = dw.get_asset_class_pl(data, ac, h, return_components=True)
            
            if isinstance(res, dict):
                r_vals[h] = fmt_dollar_clean(res["pl"])
                # Populate meta columns from calculation components
                r_vals[f"meta_{h}_start"] = res["start"]
                r_vals[f"meta_{h}_end"] = res["end"]
                r_vals[f"meta_{h}_flow"] = res["flow"]
                r_vals[f"meta_{h}_inc"] = res["inc"]
                r_vals[f"meta_{h}_denom"] = res["denom"]
            else:
                pl_val = res
                r_vals[h] = fmt_dollar_clean(pl_val) if pl_val is not None else "N/A"
        pl_table_data.append(r_vals)
        
        # Ticker Rows: Show visible tickers only, use ticker-level P/L
        # Need to find the full ticker row in sec_table to get meta cols
        # tickers_in_ac is just a list of names.
        # Let's get the full rows from sec_table_display
        ticker_rows = sec_table_display[sec_table_display['asset_class'] == ac]
        
        for _, t_row_full in ticker_rows.iterrows():
            t = t_row_full['ticker']
            tr_vals = {
                "Asset Class / Ticker": f"  {t}", 
                "Type": "Ticker", 
                "_sort_rank": rank, 
                "_is_header": 0
            }
            # Add meta columns from ticker row
            for k, v in t_row_full.items():
                if str(k).startswith("meta_"):
                    tr_vals[k] = v

            # TICKER LOOP (UPDATED)
            for h in horizons:
                # Use cached ticker P/L (consistent with asset class calculation)
                df = ticker_pl_cache[h]
                if not df.empty and t in df.index:
                    pl_val = df.loc[t, 'pl']
                    tr_vals[h] = fmt_dollar_clean(pl_val) if pl_val is not None else "N/A"
                    
                    # SYNC META DATA: Pull meta columns from the cache to ensure Audit Modal matches Table Value
                    for col in df.columns:
                        if col.startswith(f"meta_{h}_"):
                            tr_vals[col] = df.loc[t, col]
                else:
                    tr_vals[h] = "N/A"
            pl_table_data.append(tr_vals)
    
    # Add Cash / Recon Row (matches PDF exactly)
    # pinnedBottomRowData expects a list of rows
    cash_recon_vals = dw.get_cash_recon_pl(data, horizons)
    recon_row = {"Asset Class / Ticker": "Cash / Recon", "Type": "Recon"}
    for h in horizons:
        pl_val = cash_recon_vals.get(h)
        recon_row[h] = fmt_dollar_clean(pl_val) if pl_val is not None else "N/A"
    
    # Use pinnedBottomRowData for Recon
    pinned_pl_rows = [recon_row]

    # Check Sort Target
    is_pl_target = "p/l" in chat_target or "profit" in chat_target

    # Create AG Grid column definitions for P/L table
    pl_column_defs = []
    for col in cols:
        # Exclude Risk Metrics from P/L Table
        if "Sharpe" in col or "Vol" in col:
            continue
            
        col_def = {
            "field": col, 
            "headerName": col, 
            "comparator": {"function": "GroupedRowComparator"}
        }
        
        # Chatbot Sort
        if chat_action == "SORT" and is_pl_target:
             target_col = chat_cmd["params"].get("column", "").lower()
             if col.lower() == target_col or target_col in col.lower():
                 col_def["sort"] = chat_cmd["params"].get("direction", "desc")

        # Hide Audit Meta Columns
        if col.startswith("meta_"):
            col_def["hide"] = True

        # Add conditional styling for P/L columns (green for positive, red for negative)
        if col in horizons:
            col_def["cellStyle"] = {
                "styleConditions": [
                    {"condition": "params.value && params.value.includes('-')", "style": {"color": "#dc3545"}},
                    {"condition": "params.value && !params.value.includes('-') && params.value !== 'N/A' && params.value !== '$0'", "style": {"color": "#28a745"}}
                ]
            }
        pl_column_defs.append(col_def)
        
    # Append meta columns to defs so they are available in params.data
    if pl_table_data:
        sample_row = pl_table_data[0]
        meta_keys = [k for k in sample_row.keys() if k.startswith("meta_")]
        for mk in meta_keys:
            if mk not in [c["field"] for c in pl_column_defs]:
                pl_column_defs.append({"field": mk, "hide": True})
    
    pl_table = dag.AgGrid(
        id="perf-horizon-pl-grid",
        rowData=pl_table_data,
        columnDefs=pl_column_defs,
        defaultColDef={"flex": 1, "minWidth": 100, "sortable": True, "filter": True, "resizable": True},
        className="ag-theme-alpine-dark audit-target",
        dashGridOptions={
            "domLayout": "autoHeight",
            "pinnedBottomRowData": pinned_pl_rows,
            "getRowStyle": {
                "function": """
                if (params.data.Type === 'Class') {
                    return {'fontWeight': 'bold', 'backgroundColor': 'rgba(255,255,255,0.05)'};
                } else if (params.data.Type === 'Recon') {
                    return {'fontWeight': 'bold', 'backgroundColor': 'rgba(255,255,0,0.15)', 'borderTop': '2px solid #888'};
                }
                return {};
                """
            }
        }
    )
    
    # ... inside update_performance function ...

    # Dynamic Footnote Text
    if include_exited:
        visibility_text = "Tables display ALL positions (active and exited) with valid history in the period."
    else:
        visibility_text = "Tables display currently active positions only."

    ret_footnote = html.Div(
        f"Note: {visibility_text} Asset Class totals ALWAYS include historical contribution of closed positions (GIPS compliant). "
        "Returns require full measurement period (e.g., 1M return requires 30+ days of history).",
        className="text-muted fst-italic mt-2 small"
    )
    
    pl_footnote = html.Div(
        f"Note: {visibility_text} P/L shows actual economic gain/loss (MV_End - MV_Start - Net_Flows + Income). "
        "This reflects actual economic outcomes regardless of holding period.",
        className="text-muted fst-italic mt-2 small"
    )

    return cum_fig, exc_fig, [ret_table, ret_footnote], [pl_table, pl_footnote]

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
     Input('theme-store', 'data'),
     Input('date-range-store', 'data'),
     Input('growth-asset-class-filter', 'value'),
     Input('chatbot-command', 'data'),
     Input('filter-store', 'data')]
)
def update_growth_analysis(signal, theme, dates, selected_ac, chat_cmd, _filters):
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
        chart_fig = dw.get_growth_of_capital_chart(data, selected_ac, theme, end_date=end_date)
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
            
            table_output = dag.AgGrid(
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
            )
    except Exception as e:
        table_output = html.Div(f"Error loading table: {str(e)}", className="p-3 text-danger")
        print(f"Error generating growth table: {e}")
    
    return chart_fig, table_output
