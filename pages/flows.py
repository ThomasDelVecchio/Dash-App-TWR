import dash
from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import dash_wrappers as dw
from report_formatting import fmt_dollar_clean
import pandas as pd

layout = html.Div([
    # External Flows Table
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H5("External Cash Flows (Since Inception)", className="card-title p-2"),
            html.Div(id='external-flows-table-container')
        ]), width=12, className="mb-4"),
    ]),
    
    # Internal Flows Summary Table
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H5("Internal Trading Summary", className="card-title p-2"),
            dcc.Loading(html.Div(id='internal-flows-table-container'))
        ]), width=12, className="mb-4"),
    ]),
    
    # Internal Flows Chart
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H5("Internal Flows by Asset Class", className="card-title p-2"),
            dcc.Graph(id={'type': 'filter-chart', 'index': 'flows-chart'})
        ]), width=12, className="mb-4"),
    ]),
])

@callback(
    [Output('external-flows-table-container', 'children'),
     Output('internal-flows-table-container', 'children'),
     Output({'type': 'filter-chart', 'index': 'flows-chart'}, 'figure')],
    [Input('data-signal', 'data'),
     Input('theme-store', 'data'),
     Input('chatbot-command', 'data'),
     Input('filter-store', 'data'),
     Input('include-exited-store', 'data')]
)
def update_flows(signal, theme, chat_cmd, _filters, include_exited):
    data = dw.get_data()
    if not data: return "Loading...", "Loading...", {}
    
    # --- CHATBOT PARAMS ---
    chat_target = ""
    chat_action = None
    if chat_cmd:
        chat_action = chat_cmd.get("action")
        chat_target = chat_cmd.get("params", {},).get("target", "").lower()
    
    # External Table
    cf = data['cf_ext']
    if cf.empty:
        ext_rows = [{"Metric": "No External Flows", "Value": ""}]
    else:
        dep = cf[cf['amount'] > 0]['amount'].sum()
        withd = cf[cf['amount'] < 0]['amount'].sum()
        net = cf['amount'].sum()
        recent = cf['date'].max().strftime('%Y-%m-%d')
        ext_rows = [
            {"Metric": "Total Deposits", "Value": fmt_dollar_clean(dep)},
            {"Metric": "Total Withdrawals", "Value": fmt_dollar_clean(withd)},
            {"Metric": "Net External Flow", "Value": fmt_dollar_clean(net)},
            {"Metric": "Most Recent", "Value": recent},
        ]
        
    is_ext_target = "external" in chat_target or (not chat_target and not "internal" in chat_target)

    ext_column_defs = []
    for col in ["Metric", "Value"]:
        col_def = {"field": col, "headerName": col, "flex": 1}
        if col == "Metric": col_def["flex"] = 2
        if col == "Value": col_def["comparator"] = {"function": "MoneyComparator"}
        
        if chat_action == "SORT" and is_ext_target:
             target_col = chat_cmd["params"].get("column", "").lower()
             if col.lower() == target_col or target_col in col.lower():
                 col_def["sort"] = chat_cmd["params"].get("direction", "desc")
        ext_column_defs.append(col_def)

    ext_table = dag.AgGrid(
        id="flows-ext-grid",
        rowData=ext_rows,
        columnDefs=ext_column_defs,
        defaultColDef={"sortable": True, "filter": True, "resizable": True},
        className="ag-theme-alpine-dark audit-target",
        dashGridOptions={"domLayout": "autoHeight"}
    )
    
    # Internal Table
    tx = data['tx_raw']
    divs = data['dividends']
    
    # Filter for "Include Exited" toggle
    # If False (default), only show tickers currently held
    if not include_exited:
        current_tickers = data['sec_table_current']['ticker'].unique()
        if not tx.empty:
            tx = tx[tx['ticker'].isin(current_tickers)]
        if not divs.empty:
            divs = divs[divs['ticker'].isin(current_tickers)]

    # Aggregate
    if tx.empty and divs.empty:
        int_rows = []
        total_row = []
    else:
        # Group logic
        tx_agg = pd.DataFrame()
        if not tx.empty:
            tx_agg = tx.groupby("ticker")["amount"].agg([
                ("buys", lambda s: s[s<0].sum()),
                ("sells", lambda s: s[s>0].sum())
            ]).reset_index()
        
        div_agg = pd.DataFrame()
        if not divs.empty:
            div_agg = divs.groupby("ticker")["amount"].sum().reset_index().rename(columns={"amount": "income"})
        
        if not tx_agg.empty and not div_agg.empty:
            merged = pd.merge(tx_agg, div_agg, on="ticker", how="outer").fillna(0)
        elif not tx_agg.empty:
            merged = tx_agg; merged['income'] = 0
        elif not div_agg.empty:
            merged = div_agg; merged['buys']=0; merged['sells']=0
        else:
            merged = pd.DataFrame()
            
        if not merged.empty:
            if 'buys' not in merged: merged['buys'] = 0
            if 'sells' not in merged: merged['sells'] = 0
            if 'income' not in merged: merged['income'] = 0
            
            merged['net'] = merged['buys'] + merged['sells'] + merged['income']
            merged = merged.sort_values('ticker')
            
            # Chatbot Filter (Internal Only as it has Tickers)
            if chat_action == "FILTER":
                val = chat_cmd["params"].get("value")
                if val:
                    mask = merged.astype(str).apply(lambda x: x.str.contains(val, case=False, na=False)).any(axis=1)
                    merged = merged[mask]

            int_rows = []
            for _, r in merged.iterrows():
                # Gather details for Audit
                t_tx = tx[tx["ticker"] == r['ticker']]
                t_buys = t_tx[t_tx["amount"] < 0].sort_values("date")
                t_sells = t_tx[t_tx["amount"] > 0].sort_values("date")
                
                t_divs = pd.DataFrame()
                if not divs.empty:
                    t_divs = divs[divs["ticker"] == r['ticker']].sort_values("date")
                
                def fmt_details(df):
                    dets = []
                    for _, row in df.iterrows():
                        amt = abs(row["amount"]) # Show absolute amounts
                        dets.append({"date": row["date"].strftime("%Y-%m-%d"), "amount": amt})
                    return dets

                int_rows.append({
                    "Ticker": r['ticker'],
                    "Buys": fmt_dollar_clean(r.get('buys', 0)),
                    "Sells": fmt_dollar_clean(r.get('sells', 0)),
                    "Income": fmt_dollar_clean(r.get('income', 0)),
                    "Net": fmt_dollar_clean(r['net']),
                    # Meta columns for Audit Modal
                    "meta_Buys_details": fmt_details(t_buys),
                    "meta_Sells_details": fmt_details(t_sells),
                    "meta_Income_details": fmt_details(t_divs)
                })

            # Calculate totals for the pinned row
            total_buys = merged['buys'].sum()
            total_sells = merged['sells'].sum()
            total_income = merged['income'].sum()
            total_net = merged['net'].sum()

            total_row = [{
                "Ticker": "Total",
                "Buys": fmt_dollar_clean(total_buys),
                "Sells": fmt_dollar_clean(total_sells),
                "Income": fmt_dollar_clean(total_income),
                "Net": fmt_dollar_clean(total_net)
            }]
        else:
            int_rows = []
            total_row = []
            
    is_int_target = "internal" in chat_target or "trading" in chat_target

    int_column_defs = []
    for col in ["Ticker", "Buys", "Sells", "Income", "Net"]:
        col_def = {"field": col, "headerName": col}
        if col != "Ticker":
            col_def["comparator"] = {"function": "MoneyComparator"}
            
        if chat_action == "SORT" and is_int_target:
             target_col = chat_cmd["params"].get("column", "").lower()
             if col.lower() == target_col or target_col in col.lower():
                 col_def["sort"] = chat_cmd["params"].get("direction", "desc")
        int_column_defs.append(col_def)

    int_table = dag.AgGrid(
        id="flows-int-grid",
        rowData=int_rows,
        columnDefs=int_column_defs,
        defaultColDef={"flex": 1, "minWidth": 100, "sortable": True, "filter": True, "resizable": True},
        className="ag-theme-alpine-dark audit-target",
        dashGridOptions={"domLayout": "autoHeight", "pinnedBottomRowData": total_row}
    )
    
    # Chart
    fig = dw.get_flows_chart(data, theme)
    
    return ext_table, int_table, fig
