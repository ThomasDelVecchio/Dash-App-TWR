import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import pandas as pd
import dash_wrappers as dw
from report_formatting import fmt_dollar_clean, fmt_pct_clean, fmt_number_clean
from tax_engine import build_tax_lots, simulate_sell, normalize_ticker
from pages.overview import create_kpi_card

# ============================================================
# LAYOUT
# ============================================================

layout = html.Div([
    # --- HEADER ---
    dbc.Row([
        dbc.Col(html.H2("Tax Authority", className="fw-bold text-body"), width=12)
    ], className="mb-4"),

    # --- KPI CARDS ---
    dbc.Row([
        dbc.Col(html.Div(id="tax-kpi-realized-container"), width=6, lg=3),
        dbc.Col(html.Div(id="tax-kpi-unrealized-container"), width=6, lg=3),
        dbc.Col(html.Div(id="tax-kpi-harvestable-container"), width=6, lg=3),
        dbc.Col(html.Div(id="tax-kpi-efficiency-container"), width=6, lg=3),
    ], className="mb-4 g-3"),

    # --- ALERTS SECTION ---
    dbc.Row([
        # Cliff Watch
        dbc.Col(dbc.Card([
            dbc.CardHeader([
                html.I(className="bi bi-hourglass-split me-2"),
                "The Cliff Watch (Turns LT in < 30 Days)"
            ]),
            dbc.CardBody([
                html.P("HOLD these lots! Waiting a few days drops tax rate from 35% to 15%.", className="text-muted small"),
                dcc.Loading(html.Div(id="cliff-watch-container"))
            ])
        ]), width=12, lg=6, className="mb-4"),

        # Harvesting Radar
        dbc.Col(dbc.Card([
            dbc.CardHeader([
                html.I(className="bi bi-bullseye me-2"),
                "Harvesting Radar (Unrealized Losses)"
            ]),
            dbc.CardBody([
                html.P("Sell these to offset gains. Watch for Wash Sales!", className="text-muted small"),
                dcc.Loading(html.Div(id="harvest-radar-container"))
            ])
        ]), width=12, lg=6, className="mb-4"),
    ]),

    # --- SIMULATOR ---
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Tax Simulator"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Ticker"),
                        dbc.Input(id="sim-ticker", placeholder="e.g. AAPL", type="text"),
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Shares to Sell"),
                        dbc.Input(id="sim-shares", placeholder="0", type="number"),
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Action"),
                        dbc.Button("Simulate Sale", id="btn-simulate", color="primary", className="w-100")
                    ], width=4, className="d-flex flex-column justify-content-end"),
                ]),
                html.Hr(),
                html.Div(id="sim-output", className="p-3 rounded border", style={"whiteSpace": "pre-wrap", "minHeight": "100px"})
            ])
        ]), width=12, className="mb-4"),
    ]),

    # --- LOT EXPLORER ---
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Tax Lot Explorer"),
            dbc.CardBody([
                dcc.Loading(html.Div(id="lot-explorer-container"))
            ])
        ]), width=12),
    ]),
])

# ============================================================
# CALLBACKS
# ============================================================

@callback(
    [Output("tax-kpi-realized-container", "children"),
     Output("tax-kpi-unrealized-container", "children"),
     Output("tax-kpi-harvestable-container", "children"),
     Output("tax-kpi-efficiency-container", "children"),
     Output("cliff-watch-container", "children"),
     Output("harvest-radar-container", "children"),
     Output("lot-explorer-container", "children")],
    [Input("data-signal", "data"),
     Input("theme-store", "data"),
     Input("chatbot-command", "data")]
)
def update_tax_dashboard(signal, theme, chat_cmd):
    # Load Fresh Data
    open_lots, realized_events = build_tax_lots()
    
    grid_theme = "ag-theme-alpine-dark" if theme == "dark" else "ag-theme-alpine"

    # Default Col Def with formatting
    default_col_def = {
        "flex": 1, 
        "minWidth": 100, 
        "sortable": True, 
        "filter": True, 
        "resizable": True,
        "valueFormatter": {"function": "d3.format(',.2f')(params.value) if typeof params.value === 'number' else params.value"}
    }
    
    # --- CHATBOT PARAMS ---
    chat_action = chat_cmd.get("action") if chat_cmd else None
    chat_target = chat_cmd.get("params", {}).get("target", "").lower() if chat_cmd else ""

    # 1. KPIs
    if not realized_events.empty:
        # Sum Tax Impact
        ytd_bill = realized_events["Tax Impact"].sum()
    else:
        ytd_bill = 0.0
        
    if not open_lots.empty:
        unrealized_liability = open_lots["Est Tax Liability"].sum()
        
        # New Metric: Harvestable Losses
        harvestable_losses = abs(open_lots[open_lots["Unrealized P/L"] < 0]["Unrealized P/L"].sum())
        
        # New Metric: Tax Efficiency (% LT)
        total_mv = open_lots["Market Value"].sum()
        lt_mv = open_lots[open_lots["Term"] == "Long-Term"]["Market Value"].sum()
        efficiency_pct = (lt_mv / total_mv * 100) if total_mv > 0 else 0.0
    else:
        unrealized_liability = 0.0
        harvestable_losses = 0.0
        efficiency_pct = 0.0
        
    kpi_realized = create_kpi_card("YTD Realized Tax Bill", fmt_dollar_clean(ytd_bill), is_positive=False if ytd_bill > 0 else None)
    kpi_unrealized = create_kpi_card("Unrealized Tax Liability", fmt_dollar_clean(unrealized_liability), is_positive=False if unrealized_liability > 0 else None)
    
    # Harvestable Losses KPI
    kpi_harvest = create_kpi_card(
        "Harvestable Losses", 
        fmt_dollar_clean(harvestable_losses), 
        subtext="Available to offset gains",
        is_positive=True if harvestable_losses > 0 else None
    )
    
    # Efficiency KPI
    eff_is_positive = True if efficiency_pct > 50 else (False if efficiency_pct < 20 else None)
    kpi_efficiency = create_kpi_card(
        "Tax Efficiency (% LT)", 
        fmt_pct_clean(efficiency_pct / 100), 
        subtext="Assets held > 1 year",
        is_positive=eff_is_positive
    )
    
    # 2. Cliff Watch
    if not open_lots.empty and "Is Near Cliff" in open_lots.columns:
        cliff_df = open_lots[open_lots["Is Near Cliff"] == True].copy()
        
        if not cliff_df.empty:
            # Format
            cliff_display = cliff_df[["Ticker", "Shares", "Date Acquired", "Unrealized P/L", "Days to LT"]].copy()
            cliff_display["Date Acquired"] = cliff_display["Date Acquired"].dt.strftime("%Y-%m-%d")
            
            col_defs = []
            for c in cliff_display.columns:
                cd = {"field": c, "headerName": c}
                if c in ["Unrealized P/L", "Shares"]:
                    cd["comparator"] = {"function": "MoneyComparator"}
                
                if c == "Unrealized P/L":
                    cd["valueFormatter"] = {"function": "d3.format('$,.2f')(params.value)"}
                elif c == "Shares":
                    cd["valueFormatter"] = {"function": "d3.format(',.2f')(params.value)"}

                # Chatbot Sort
                if chat_action == "SORT" and "cliff" in chat_target:
                    target_col = chat_cmd["params"].get("column", "").lower()
                    if c.lower() == target_col or target_col in c.lower():
                        cd["sort"] = chat_cmd["params"].get("direction", "desc")
                col_defs.append(cd)

            cliff_grid = dag.AgGrid(
                rowData=cliff_display.to_dict("records"),
                columnDefs=col_defs,
                defaultColDef=default_col_def,
                className=grid_theme,
                columnSize="responsiveSizeToFit",
                dashGridOptions={"domLayout": "autoHeight"}
            )
        else:
            cliff_grid = html.Div("No lots approaching the 1-year mark.", className="text-success p-2")
    else:
        cliff_grid = html.Div("No open lots found.", className="text-muted")
        
    # 3. Harvest Radar
    if not open_lots.empty:
        harvest_df = open_lots[open_lots["Unrealized P/L"] < 0].copy()
        
        if not harvest_df.empty:
            harvest_df = harvest_df.sort_values("Unrealized P/L", ascending=True) # Largest loss first
            
            # Format
            harvest_display = harvest_df[["Ticker", "Shares", "Cost Basis", "Market Value", "Unrealized P/L", "Term"]].copy()
            
            col_defs = []
            for c in harvest_display.columns:
                cd = {"field": c, "headerName": c}
                if c in ["Shares", "Cost Basis", "Market Value", "Unrealized P/L"]:
                    cd["comparator"] = {"function": "MoneyComparator"}
                
                if c in ["Cost Basis", "Market Value", "Unrealized P/L"]:
                    cd["valueFormatter"] = {"function": "d3.format('$,.2f')(params.value)"}
                elif c == "Shares":
                    cd["valueFormatter"] = {"function": "d3.format(',.2f')(params.value)"}

                # Chatbot Sort
                if chat_action == "SORT" and "harvest" in chat_target:
                    target_col = chat_cmd["params"].get("column", "").lower()
                    if c.lower() == target_col or target_col in c.lower():
                        cd["sort"] = chat_cmd["params"].get("direction", "desc")
                col_defs.append(cd)

            harvest_grid = dag.AgGrid(
                rowData=harvest_display.to_dict("records"),
                columnDefs=col_defs,
                defaultColDef=default_col_def,
                className=grid_theme,
                columnSize="responsiveSizeToFit",
                dashGridOptions={"domLayout": "autoHeight"}
            )
        else:
            harvest_grid = html.Div("No unrealized losses found. Great job!", className="text-success p-2")
    else:
        harvest_grid = html.Div("No open lots.", className="text-muted")

    # 4. Lot Explorer (TABS)
    
    # Tab 1: Open Lots
    if not open_lots.empty:
        explorer_df = open_lots.copy()
        explorer_df["Date Acquired"] = explorer_df["Date Acquired"].dt.strftime("%Y-%m-%d")
        
        cols_currency = ["Cost Basis", "Current Price", "Market Value", "Unrealized P/L", "Est Tax Liability"]
        cols_hide = ["Is Near Cliff", "Days to LT", "Cost Per Share"]
        col_defs = []
        for c in explorer_df.columns:
            hide = c in cols_hide
            cd = {"field": c, "headerName": c, "hide": hide}
            if c in cols_currency:
                cd["comparator"] = {"function": "MoneyComparator"}
                cd["valueFormatter"] = {"function": "d3.format('$,.2f')(params.value)"}
            elif c == "Shares":
                cd["comparator"] = {"function": "MoneyComparator"}
                cd["valueFormatter"] = {"function": "d3.format(',.2f')(params.value)"}
            
            # Chatbot Sort
            if chat_action == "SORT" and "open" in chat_target:
                target_col = chat_cmd["params"].get("column", "").lower()
                if c.lower() == target_col or target_col in c.lower():
                    cd["sort"] = chat_cmd["params"].get("direction", "desc")
            col_defs.append(cd)
            
        grid_open = dag.AgGrid(
            rowData=explorer_df.to_dict("records"),
            columnDefs=col_defs,
            defaultColDef=default_col_def,
            className=grid_theme,
            columnSize="responsiveSizeToFit",
            dashGridOptions={"domLayout": "autoHeight", "pagination": True, "paginationPageSize": 20}
        )
    else:
        grid_open = html.Div("No open lots.", className="p-3")

    # Tab 2: Realized Events
    if not realized_events.empty:
        realized_df = realized_events.copy()
        realized_df["Date Sold"] = realized_df["Date Sold"].dt.strftime("%Y-%m-%d")
        
        if "Is Wash Sale" in realized_df.columns:
            realized_df["Is Wash Sale"] = realized_df["Is Wash Sale"].apply(lambda x: "⚠️ YES" if x else "No")
            
        cols_currency = ["Realized P/L", "Tax Impact"]
        col_defs = []
        for c in realized_df.columns:
            cd = {"field": c, "headerName": c}
            if c in cols_currency:
                cd["comparator"] = {"function": "MoneyComparator"}
                cd["valueFormatter"] = {"function": "d3.format('$,.2f')(params.value)"}
            elif c == "Shares":
                cd["comparator"] = {"function": "MoneyComparator"}
                cd["valueFormatter"] = {"function": "d3.format(',.2f')(params.value)"}
            
            # Chatbot Sort
            if chat_action == "SORT" and "realized" in chat_target:
                target_col = chat_cmd["params"].get("column", "").lower()
                if c.lower() == target_col or target_col in c.lower():
                    cd["sort"] = chat_cmd["params"].get("direction", "desc")
            col_defs.append(cd)

        grid_realized = dag.AgGrid(
            rowData=realized_df.to_dict("records"),
            columnDefs=col_defs,
            defaultColDef=default_col_def,
            className=grid_theme,
            columnSize="responsiveSizeToFit",
            dashGridOptions={"domLayout": "autoHeight", "pagination": True, "paginationPageSize": 20}
        )
    else:
        grid_realized = html.Div("No realized events YTD.", className="p-3")

    explorer_tabs = dbc.Tabs([
        dbc.Tab(grid_open, label="Open Lots", tab_id="tab-open"),
        dbc.Tab(grid_realized, label="Realized History", tab_id="tab-realized"),
    ], active_tab="tab-open")

    return kpi_realized, kpi_unrealized, kpi_harvest, kpi_efficiency, cliff_grid, harvest_grid, explorer_tabs

@callback(
    Output("sim-output", "children"),
    Input("btn-simulate", "n_clicks"),
    [State("sim-ticker", "value"),
     State("sim-shares", "value")]
)
def run_simulation(n_clicks, ticker, shares):
    if not n_clicks:
        return "Enter parameters and click Simulate."
        
    if not ticker or not shares:
        return "Please provide both Ticker and Shares."
        
    try:
        shares_float = float(shares)
    except ValueError:
        return "Invalid shares amount."
        
    result = simulate_sell(ticker, shares_float)
    
    return result["summary_text"]
