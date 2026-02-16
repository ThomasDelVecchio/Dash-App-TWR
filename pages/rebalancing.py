import dash
from dash import dcc, html, callback, Input, Output, State, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import dash_wrappers as dw
from report_formatting import fmt_dollar_clean, fmt_pct_clean, fmt_number_clean
from tax_engine import build_tax_lots, _days_to_long_term, _classify_term, calculate_tax_optimized_sales
from components.page_header import page_header
from config import GLOBAL_PALETTE

# ============================================================
# LAYOUT
# ============================================================

layout = html.Div([
    # --- HEADER ---
    page_header(
        title="Rebalancing Tool",
        icon="bi-sliders",
        subtitle="Tax-aware drift analysis and cash deployment"
    ),
    # --- INPUT CONTROLS ---
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader([
                html.I(className="bi bi-cash-stack me-2"),
                "Deployment Parameters"
            ], className="section-header"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Cash to Deploy ($)", className="fw-bold"),
                        dbc.Input(
                            id="cash-to-deploy-input",
                            type="number",
                            placeholder="Enter amount...",
                            value=1000,
                            min=0,
                            step=100,
                            className="mb-2"
                        ),
                        html.Small("Enter the cash amount you want to invest.", className="text-muted"),
                        html.Small(id="available-cash-text", className="text-info d-block mt-1 fw-bold"),
                        
                        # --- NEW SWITCH ---
                        dbc.Switch(
                            id="allow-sales-switch",
                            label="Allow Sales for Rebalancing",
                            value=False,
                            className="mt-3 fw-bold text-warning"
                        ),
                        html.Small("Enable to sell overweight assets (Tax-Aware).", className="text-muted d-block ms-4")
                        
                    ], width=12, lg=4),
                    dbc.Col([
                        dbc.Label("Current Portfolio Value", className="fw-bold"),
                        html.H4(id="current-portfolio-value", className="text-success mb-0"),
                        html.Small("Excludes cash position", className="text-muted")
                    ], width=12, lg=4),
                    dbc.Col([
                        dbc.Label("Pro-Forma Portfolio Value", className="fw-bold"),
                        html.H4(id="proforma-portfolio-value", className="text-info mb-0"),
                        html.Small("After deployment & sales", className="text-muted")
                    ], width=12, lg=4),
                ], className="g-3"),
            ])
        ]), width=12, className="mb-4"),
    ]),

    # --- DEPLOYMENT TABLE ---
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader([
                html.I(className="bi bi-table me-2"),
                "Rebalancing Schedule"
            ], className="section-header"),
            dbc.CardBody([
                dcc.Loading(html.Div(id="deployment-table-container"))
            ])
        ]), width=12, className="mb-4"),
    ]),

    # --- DRIFT CHART & TAX IMPACT ---
    dbc.Row([
        # Drift Chart
        dbc.Col(dbc.Card([
            dbc.CardHeader([
                html.I(className="bi bi-bar-chart me-2"),
                "Weight Drift Analysis"
            ], className="section-header"),
            dbc.CardBody([
                dcc.Graph(id="drift-chart", config={"displayModeBar": False})
            ])
        ], className="h-100"), width=12, lg=7, className="mb-4"),

        # Tax Impact Summary
        dbc.Col(dbc.Card([
            dbc.CardHeader([
                html.I(className="bi bi-calculator me-2"),
                "Tax Impact Summary"
            ], className="section-header"),
            dbc.CardBody([
                dcc.Loading(html.Div(id="tax-impact-container"))
            ])
        ], className="h-100"), width=12, lg=5, className="mb-4"),
    ]),

    # --- CLIFF WATCH ---
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader([
                html.I(className="bi bi-hourglass-split me-2"),
                "Cliff Watch (New Shares Long-Term Transition)"
            ], className="section-header"),
            dbc.CardBody([
                html.P(
                    "Newly purchased shares will be short-term until held for > 365 days. "
                    "Track when these new lots transition to long-term for favorable tax treatment.",
                    className="text-muted small mb-3"
                ),
                dcc.Loading(html.Div(id="rebalancing-cliff-watch-container"))
            ])
        ]), width=12, className="mb-4"),
    ]),
], className="rebalancing-page", style={"padding": "20px"})


# ============================================================
# CALLBACK: Main Deployment Logic
# ============================================================

@callback(
    [Output("current-portfolio-value", "children"),
     Output("proforma-portfolio-value", "children"),
     Output("deployment-table-container", "children"),
     Output("drift-chart", "figure"),
     Output("tax-impact-container", "children"),
     Output("rebalancing-cliff-watch-container", "children"),
     Output("available-cash-text", "children")],
    [Input("cash-to-deploy-input", "value"),
     Input("allow-sales-switch", "value"),
     Input("data-signal", "data"),
     Input("tax-strategy-store", "data")]
)
def update_deployment(cash_to_deploy, allow_sales, signal, tax_strategy):
    """
    Main callback to calculate and display rebalancing recommendations.
    """
    # Default empty returns
    empty_fig = go.Figure()
    empty_fig.update_layout(template="plotly_dark")
    
    default_return = ("$0", "$0", "Loading...", empty_fig, "Loading...", "Loading...", "")
    
    # Validate inputs
    if cash_to_deploy is None: cash_to_deploy = 0
    if cash_to_deploy < 0: cash_to_deploy = 0
    
    # Get data
    data = dw.get_data()
    if not data:
        return default_return
    
    # Extract required data
    sec_table = data["sec_table_current"].copy()
    holdings = data["holdings"].copy()
    prices = data["prices"]
    
    if sec_table.empty:
        return default_return
    
    # ============================================================
    # STEP 1: Calculate Current State
    # ============================================================
    
    # Exclude CASH from investment calculations
    invested_df = sec_table[sec_table["ticker"] != "CASH"].copy()
    
    # Ensure target_pct is available
    if "target_pct" not in invested_df.columns:
        invested_df = invested_df.merge(
            holdings[["ticker", "target_pct"]],
            on="ticker",
            how="left"
        )
        invested_df["target_pct"] = invested_df["target_pct"].fillna(0.0)
    
    # Current portfolio value (excluding cash)
    current_total = invested_df["market_value"].sum()
    
    # Pro-forma total (current + new cash)
    # Note: If we sell, the portfolio value stays same (asset -> cash), 
    # but the 'investable' total (Assets + Cash) is basically (Current Market Value + New Cash Injection).
    proforma_total = current_total + cash_to_deploy
    
    # Current weights
    invested_df["current_weight_pct"] = (invested_df["market_value"] / current_total * 100) if current_total > 0 else 0
    
    # ============================================================
    # STEP 2: Logic - Sales & Buys
    # ============================================================
    
    # Filter to tickers with targets (or holdings that should be sold if target is 0)
    # We generally work with the invested_df which has current holdings.
    # If a holding has 0 target, it should be sold completely if sales are allowed.
    target_df = invested_df.copy() # All current holdings
    
    # Calculate Target Dollar
    target_df["target_dollar"] = (target_df["target_pct"] / 100) * proforma_total
    
    # Raw Difference (Target - Current)
    # Positive = Buy needed
    # Negative = Sell needed
    target_df["raw_diff"] = target_df["target_dollar"] - target_df["market_value"]
    target_df["drift"] = (target_df["target_pct"] - target_df["current_weight_pct"])
    
    # --- A. SALES LOGIC ---
    
    # Default values
    target_df["action"] = "Hold"
    target_df["recommend_amount"] = 0.0 # Signed amount (+Buy, -Sell)
    target_df["realized_pl"] = 0.0
    target_df["est_tax"] = 0.0
    
    sale_proceeds = 0.0
    total_realized_pl = 0.0
    total_est_tax = 0.0
    
    if allow_sales:
        # Identify Overweight Assets (raw_diff < 0)
        overweight_mask = target_df["raw_diff"] < -0.01
        
        candidates = {}
        for _, row in target_df[overweight_mask].iterrows():
            candidates[row["ticker"]] = abs(row["raw_diff"])
            
        # Calculate Optimized Sales
        # Use selected tax strategy (defaults to FIFO if not set)
        sales_res = calculate_tax_optimized_sales(candidates, avoid_st_gains=True, strategy=tax_strategy or "FIFO")
        
        # Unpack Results
        sales_df_res = sales_res["sales_df"]
        sale_proceeds = sales_res["total_proceeds"]
        total_realized_pl = sales_res["total_realized_pl"]
        total_est_tax = sales_res["est_tax_liability"]
        
        # Map back to target_df
        if not sales_df_res.empty:
            # Group by ticker in case multiple lots sold
            sales_grp = sales_df_res.groupby("Ticker")[["Proceeds", "Realized P/L", "Tax Impact"]].sum()
            
            for ticker, row in sales_grp.iterrows():
                # Update target_df
                mask = target_df["ticker"] == ticker
                if mask.any():
                    target_df.loc[mask, "recommend_amount"] = -row["Proceeds"] # Negative for Sell
                    target_df.loc[mask, "realized_pl"] = row["Realized P/L"]
                    target_df.loc[mask, "est_tax"] = row["Tax Impact"]
                    target_df.loc[mask, "action"] = "Sell"

    # --- B. BUYS LOGIC (Waterfall) ---
    
    # Total Cash Available = Input + Sale Proceeds
    total_cash_available = float(cash_to_deploy) + sale_proceeds
    remaining_cash = total_cash_available
    
    # Identify Underweight Assets (raw_diff > 0)
    # AND filter out anything we just sold (shouldn't happen if diff > 0, but safety check)
    buy_mask = (target_df["raw_diff"] > 0.01) & (target_df["recommend_amount"] == 0)
    
    # We only allocate to these rows
    target_df.loc[buy_mask, "full_buy_need"] = target_df.loc[buy_mask, "raw_diff"]
    target_df["buy_allocation"] = 0.0
    
    # Iterate for Waterfall
    for i in range(10):
        if remaining_cash < 0.01:
            break
            
        mask_room = (target_df["buy_allocation"] < target_df.get("full_buy_need", 0)) & buy_mask
        if not mask_room.any():
            break
            
        # Current Drift Sum for candidates
        # We use raw_diff as proxy for drift magnitude or actual drift
        current_drift_sum = target_df.loc[mask_room, "drift"].clip(lower=0).sum()
        
        allocation_weights = pd.Series(0.0, index=target_df.index)
        
        if current_drift_sum > 0:
            allocation_weights[mask_room] = target_df.loc[mask_room, "drift"].clip(lower=0) / current_drift_sum
        else:
            # Fallback to remaining capacity
            rem_cap = target_df.loc[mask_room, "full_buy_need"] - target_df.loc[mask_room, "buy_allocation"]
            cap_sum = rem_cap.sum()
            if cap_sum > 0:
                allocation_weights[mask_room] = rem_cap / cap_sum
            else:
                break
                
        tentative = allocation_weights * remaining_cash
        max_add = target_df.get("full_buy_need", 0) - target_df["buy_allocation"]
        actual = np.minimum(tentative, max_add)
        
        spent = actual.sum()
        target_df.loc[mask_room, "buy_allocation"] += actual[mask_room]
        remaining_cash -= spent
        
        if spent < 0.01: break
        
    # Apply Buys to results
    buy_rows = target_df["buy_allocation"] > 0
    target_df.loc[buy_rows, "recommend_amount"] = target_df.loc[buy_rows, "buy_allocation"]
    target_df.loc[buy_rows, "action"] = "Buy"
    
    # --- C. FINAL CALCS ---
    
    # Get available cash from data for display
    cash_row = sec_table[sec_table["ticker"] == "CASH"]
    curr_cash_balance = cash_row["market_value"].sum() if not cash_row.empty else 0
    cash_msg = f"Portfolio Cash: {fmt_dollar_clean(curr_cash_balance)} | New Input: {fmt_dollar_clean(cash_to_deploy)} | Sales: {fmt_dollar_clean(sale_proceeds)}"
    
    # Pro-Forma Weight
    # New Value = Current + Recommend Amount (Negative for sell)
    target_df["proforma_value"] = target_df["market_value"] + target_df["recommend_amount"]
    target_df["proforma_weight_pct"] = (target_df["proforma_value"] / proforma_total * 100)
    
    # Est Shares
    # Get latest prices
    latest_prices = {}
    if not prices.empty:
        for ticker in target_df["ticker"].unique():
            t_upper = str(ticker).upper()
            if t_upper in prices.columns:
                latest_prices[ticker] = prices[t_upper].dropna().iloc[-1]
            else:
                match = sec_table[sec_table["ticker"] == ticker]
                if not match.empty and match["shares"].iloc[0] != 0:
                    latest_prices[ticker] = match["market_value"].iloc[0] / match["shares"].iloc[0]
                else:
                    latest_prices[ticker] = 0
                    
    target_df["latest_price"] = target_df["ticker"].map(latest_prices)
    target_df["est_shares"] = np.where(
        target_df["latest_price"] > 0,
        abs(target_df["recommend_amount"]) / target_df["latest_price"],
        0
    )
    
    # ============================================================
    # STEP 3: Build AG Grid Table
    # ============================================================
    
    # Calculate Total Drift for Audit context (Sum of positive drifts)
    total_drift_val = target_df["drift"].clip(lower=0).sum()

    display_df = pd.DataFrame({
        "Ticker": target_df["ticker"],
        "Asset_Class": target_df["asset_class"],
        "Current_Pct": target_df["current_weight_pct"].apply(lambda x: f"{x:.2f}%"),
        "Target_Pct": target_df["target_pct"].apply(lambda x: f"{x:.2f}%"),
        "Drift": target_df["drift"].apply(lambda x: f"{x:.2f}%"),
        "Action": target_df["action"],
        "Amount": target_df["recommend_amount"].apply(fmt_dollar_clean),
        "Price": target_df["latest_price"].apply(lambda x: fmt_dollar_clean(x) if x > 0 else "N/A"),
        "Shares": target_df["est_shares"].apply(lambda x: f"{x:.2f}" if x > 0 else "0.00"),
        "Tax_Impact": target_df.apply(lambda r: fmt_dollar_clean(r["realized_pl"] * (0.15 if r["realized_pl"] > 0 else 0)) if r["action"] == "Sell" else "-", axis=1),
        "ProForma_Pct": target_df["proforma_weight_pct"].apply(lambda x: f"{x:.2f}%"),
        
        # Meta Columns for Audit
        "meta_price": target_df["latest_price"],  # Raw price for StageOrderButton
        "meta_shares": target_df["est_shares"],
        "meta_amount": target_df["recommend_amount"],
        "meta_tax": target_df["est_tax"],
        "meta_realized_pl": target_df["realized_pl"],
        "meta_current_weight": target_df["current_weight_pct"],
        "meta_target_weight": target_df["target_pct"],
        "meta_drift": target_df["drift"],
        "meta_full_target_buy": target_df["target_dollar"] - target_df["market_value"], # Raw gap
        "meta_market_value": target_df["market_value"],
        "meta_proforma_total": proforma_total,
        "meta_total_drift": total_drift_val,
        "meta_cash_to_deploy": total_cash_available,
        "meta_allow_sales": allow_sales
    })
    
    column_defs = [
        {"field": "Ticker", "headerName": "Ticker", "pinned": "left", "width": 140, "suppressSizeToFit": True, "lockPinned": True, "cellClass": "lock-pinned", "checkboxSelection": True, "headerCheckboxSelection": True},
        {"field": "Asset_Class", "headerName": "Asset Class", "minWidth": 185},
        {"field": "Current_Pct", "headerName": "Current %", "minWidth": 120, "comparator": {"function": "MoneyComparator"}},
        {"field": "Target_Pct", "headerName": "Target %", "minWidth": 120, "comparator": {"function": "MoneyComparator"}},
        {"field": "Drift", "headerName": "Drift", "minWidth": 120, "comparator": {"function": "MoneyComparator"},
         "cellStyle": {"styleConditions": [
             {"condition": "params.value.includes('-')", "style": {"color": "#ef4444", "backgroundColor": "rgba(239,68,68,0.08)"}}, # Negative drift
             {"condition": "!params.value.includes('-')", "style": {"color": "#ffc107"}}
         ]}},
        {"field": "Action", "headerName": "Action", "minWidth": 120,
         "cellStyle": {"styleConditions": [
             {"condition": "params.value == 'Buy'", "style": {"color": "#22c55e", "fontWeight": "bold"}},
             {"condition": "params.value == 'Sell'", "style": {"color": "#ef4444", "fontWeight": "bold"}}
         ]}},
        {"field": "Amount", "headerName": "Amount", "minWidth": 150, "comparator": {"function": "MoneyComparator"},
         "cellStyle": {"styleConditions": [
             {"condition": "params.data.meta_amount > 0", "style": {"color": "#22c55e", "backgroundColor": "rgba(34,197,94,0.08)"}},
             {"condition": "params.data.meta_amount < 0", "style": {"color": "#ef4444", "backgroundColor": "rgba(239,68,68,0.08)"}}
         ]}},
        {"field": "Shares", "headerName": "Shares", "minWidth": 120, "comparator": {"function": "MoneyComparator"}},
        {"field": "Tax_Impact", "headerName": "Est. Tax", "minWidth": 140, "comparator": {"function": "MoneyComparator"}},
        {"field": "ProForma_Pct", "headerName": "Pro-Forma %", "minWidth": 140, "comparator": {"function": "MoneyComparator"}},
        
        # Hidden Meta Columns
        {"field": "meta_price", "hide": True},
        {"field": "meta_shares", "hide": True},
        {"field": "meta_amount", "hide": True},
        {"field": "meta_tax", "hide": True},
        {"field": "meta_realized_pl", "hide": True},
        {"field": "meta_current_weight", "hide": True},
        {"field": "meta_target_weight", "hide": True},
        {"field": "meta_drift", "hide": True},
        {"field": "meta_full_target_buy", "hide": True},
        {"field": "meta_market_value", "hide": True},
        {"field": "meta_proforma_total", "hide": True},
        {"field": "meta_total_drift", "hide": True},
        {"field": "meta_cash_to_deploy", "hide": True},
        {"field": "meta_allow_sales", "hide": True}
    ]
    
    deployment_table = html.Div([
        dag.AgGrid(
            id="deployment-grid",
            rowData=display_df.to_dict("records"),
            columnDefs=column_defs,
            defaultColDef={"sortable": True, "filter": True, "resizable": True, "flex": 1, "minWidth": 110},
            className="ag-theme-alpine-dark audit-target",
            dashGridOptions={"domLayout": "autoHeight", "rowSelection": "multiple", "suppressRowClickSelection": True},
            style={"width": "100%"}
        ),
        html.Div([
            dbc.Button(
                [html.I(className="bi bi-cart-plus me-2"), "Stage Selected for Trade"],
                id="stage-selected-btn",
                color="success",
                className="mt-3",
                disabled=True
            ),
            html.Span(id="stage-selected-count", className="ms-3 text-muted small")
        ], className="d-flex align-items-center"),
        html.Div(
            html.Small(
                "Rebalancing ‘Current %’ excludes CASH; Holdings ‘Weight’ includes CASH.",
                className="text-muted fst-italic"
            ),
            className="mt-2 text-center"
        )
    ])
    
    # Check if all actions are Hold (diagnostic message)
    all_hold = (target_df["action"] == "Hold").all()
    if all_hold and len(target_df) > 0:
        hold_alert = dbc.Alert(
            [
                html.I(className="bi bi-info-circle me-2"),
                "All positions are at target weights. No rebalancing actions needed.",
                html.Br(),
                html.Small("Tip: Add cash to deploy or enable 'Allow Sales' to generate recommendations.", className="text-muted")
            ],
            color="info",
            className="mt-3"
        )
    else:
        hold_alert = None
    
    # ============================================================
    # STEP 4: Build Drift Chart
    # ============================================================
    
    drift_fig = build_drift_chart(target_df, "dark")
    
    # ============================================================
    # STEP 5: Tax Impact Summary
    # ============================================================
    
    tax_impact_content = build_tax_impact_summary(target_df, cash_to_deploy, total_realized_pl, total_est_tax, tax_strategy)
    
    # ============================================================
    # STEP 6: Cliff Watch
    # ============================================================
    
    cliff_watch_content = build_cliff_watch(target_df, "dark")
    
    # Wrap deployment table with optional hold alert
    if hold_alert:
        deployment_output = html.Div([hold_alert, deployment_table])
    else:
        deployment_output = deployment_table
    
    return (
        fmt_dollar_clean(current_total),
        fmt_dollar_clean(proforma_total),
        deployment_output,
        drift_fig,
        tax_impact_content,
        cliff_watch_content,
        cash_msg
    )


# ============================================================
# HELPER: Build Drift Chart
# ============================================================

def build_drift_chart(target_df, theme):
    """
    Bar chart showing current weight vs target vs pro-forma weight.
    """
    fig = go.Figure()
    
    # Sort by target weight descending for better visualization
    plot_df = target_df.sort_values("target_pct", ascending=True)
    
    # Current weight bars
    fig.add_trace(go.Bar(
        y=plot_df["ticker"],
        x=plot_df["current_weight_pct"],
        name="Current %",
        orientation="h",
        marker_color=GLOBAL_PALETTE[0] if len(GLOBAL_PALETTE) > 0 else "#636EFA",
        opacity=0.7,
        hovertemplate="<b>%{y}</b><br>Current: %{x:.2f}%<extra></extra>"
    ))
    
    # Pro-forma weight bars
    fig.add_trace(go.Bar(
        y=plot_df["ticker"],
        x=plot_df["proforma_weight_pct"],
        name="Pro-Forma %",
        orientation="h",
        marker_color=GLOBAL_PALETTE[2] if len(GLOBAL_PALETTE) > 2 else "#00CC96",
        opacity=0.7,
        hovertemplate="<b>%{y}</b><br>Pro-Forma: %{x:.2f}%<extra></extra>"
    ))
    
    # Target markers
    fig.add_trace(go.Scatter(
        y=plot_df["ticker"],
        x=plot_df["target_pct"],
        name="Target %",
        mode="markers",
        marker=dict(
            symbol="diamond",
            size=12,
            color="#FFD700",
            line=dict(width=1, color="white")
        ),
        hovertemplate="<b>%{y}</b><br>Target: %{x:.2f}%<extra></extra>"
    ))
    
    template = "plotly_dark"
    
    fig.update_layout(
        title="Weight Comparison: Current → Pro-Forma",
        xaxis_title="Weight %",
        yaxis_title="",
        barmode="group",
        template=template,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=max(300, len(plot_df) * 40),
        margin=dict(l=100, r=20, t=60, b=40)
    )
    
    return fig


# ============================================================
# HELPER: Build Tax Impact Summary
# ============================================================

def build_tax_impact_summary(target_df, cash_to_deploy, realized_pl, est_tax, tax_strategy):
    """
    Calculate and display tax impact of rebalancing.
    """
    # 1. New Investments (Cost Basis Addition)
    new_buys = target_df.loc[target_df["recommend_amount"] > 0, "recommend_amount"].sum()
    
    # 2. Sales (Realized P/L)
    # Passed in as realized_pl
    
    # Build summary cards
    summary = dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.Small("New Investments", className="text-muted"),
                html.H5(f"+{fmt_dollar_clean(new_buys)}", className="mb-0 text-success")
            ])
        ], className="bg-transparent border"), width=4, className="mb-2"),
        
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.Small("Realized Gain/Loss", className="text-muted"),
                html.H5(fmt_dollar_clean(realized_pl), 
                        className="mb-0 " + ("text-danger" if realized_pl < 0 else "text-warning"))
            ])
        ], className="bg-transparent border"), width=4, className="mb-2"),
        
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.Small("Est. Tax Liability", className="text-muted"),
                html.H5(fmt_dollar_clean(est_tax), className="mb-0 text-danger fw-bold")
            ])
        ], className="bg-transparent border"), width=4, className="mb-2"),
    ], className="g-2")
    
    # Additional insights
    tickers_to_buy = target_df[target_df["action"] == "Buy"]["ticker"].tolist()
    tickers_to_sell = target_df[target_df["action"] == "Sell"]["ticker"].tolist()
    
    insights = html.Div([
        html.Hr(),
        html.P([
            html.Strong("Buying: "),
            ", ".join(tickers_to_buy) if tickers_to_buy else "None"
        ], className="mb-1"),
        html.P([
            html.Strong("Selling: "),
            ", ".join(tickers_to_sell) if tickers_to_sell else "None"
        ], className="mb-1"),
        html.P([
            html.I(className="bi bi-info-circle me-1"),
            "Sales prioritize harvesting losses and avoiding short-term gains. Wash sales are excluded."
        ], className="text-muted small fst-italic")
    ])
    
    return html.Div([summary, insights])


# ============================================================
# HELPER: Build Cliff Watch
# ============================================================

def build_cliff_watch(target_df, theme):
    """
    Show when new purchases will transition to long-term.
    """
    # Filter to tickers with actual buys
    buys_df = target_df[target_df["recommend_amount"] > 0].copy()
    
    if buys_df.empty:
        return html.P("No purchases recommended.", className="text-muted fst-italic")
    
    # Calculate cliff dates (1 year from today)
    today = datetime.now()
    cliff_date = today + timedelta(days=366)
    cliff_date_str = cliff_date.strftime("%B %d, %Y")
    
    # Build display
    cliff_data = []
    for _, row in buys_df.iterrows():
        cliff_data.append({
            "Ticker": row["ticker"],
            "Buy_Amount": fmt_dollar_clean(row["recommend_amount"]),
            "Shares": f"{row['est_shares']:.2f}",
            "Purchase_Date": today.strftime("%Y-%m-%d"),
            "LT_Cliff_Date": cliff_date_str,
            "Days_to_LT": 366,
            "LT_Cliff_Date_Sort": cliff_date.strftime("%Y-%m-%d")
        })
    
    cliff_df = pd.DataFrame(cliff_data)
    
    column_defs = [
        {"field": "Ticker", "headerName": "Ticker", "width": 110, "suppressSizeToFit": True},
        {"field": "Buy_Amount", "headerName": "Investment", "minWidth": 120, "comparator": {"function": "MoneyComparator"}},
        {"field": "Shares", "headerName": "Shares", "minWidth": 110, "comparator": {"function": "MoneyComparator"}},
        {"field": "Purchase_Date", "headerName": "Buy Date", "minWidth": 120},
        {"field": "LT_Cliff_Date", "headerName": "Long-Term Date", "minWidth": 150,
         "valueGetter": {"function": "params.data.LT_Cliff_Date_Sort"},
         "valueFormatter": {"function": "params.data.LT_Cliff_Date"},
         "cellStyle": {"color": "#FFD700", "fontWeight": "bold"}},
        {"field": "Days_to_LT", "headerName": "Days to LT", "minWidth": 110, "comparator": {"function": "MoneyComparator"}},
        {"field": "LT_Cliff_Date_Sort", "hide": True}
    ]
    
    cliff_table = dag.AgGrid(
        id="cliff-watch-grid",
        rowData=cliff_df.to_dict("records"),
        columnDefs=column_defs,
        defaultColDef={"sortable": True, "filter": True, "resizable": True, "flex": 1, "minWidth": 110},
        className="ag-theme-alpine-dark",
        dashGridOptions={"domLayout": "autoHeight"},
        style={"width": "100%"}
    )
    
    summary_text = html.Div([
        html.P([
            html.I(className="bi bi-calendar-check me-2"),
            f"All new purchases will become long-term on ",
            html.Strong(cliff_date_str),
            " (366 days from today)."
        ], className="mb-2"),
        html.P([
            "Tax rate drops from ~35% (Short-Term) to ~15% (Long-Term) after this date."
        ], className="text-muted small mb-3"),
        cliff_table
    ])
    
    return summary_text


# ============================================================
# STAGE SELECTED ROWS CALLBACKS
# ============================================================

@callback(
    [Output("stage-selected-btn", "disabled"),
     Output("stage-selected-count", "children")],
    Input("deployment-grid", "selectedRows")
)
def update_stage_button_state(selected_rows):
    """Enable/disable the Stage Selected button based on row selection."""
    if not selected_rows:
        return True, ""
    
    # Filter out Hold actions - only actionable items can be staged
    actionable = [r for r in selected_rows if r.get("Action") in ["Buy", "Sell"]]
    count = len(actionable)
    
    if count == 0:
        return True, "No actionable items selected (Hold actions cannot be staged)"
    
    if count == 1:
        return False, f"1 order ready to stage"
    else:
        return False, f"{count} orders ready to stage"


@callback(
    [Output("staged-order-store", "data", allow_duplicate=True),
     Output("stage-selected-btn", "children")],
    Input("stage-selected-btn", "n_clicks"),
    State("deployment-grid", "selectedRows"),
    prevent_initial_call=True
)
def stage_selected_orders(n_clicks, selected_rows):
    """Stage the selected rows for trade execution."""
    if not n_clicks or not selected_rows:
        raise PreventUpdate
    
    # Filter only actionable rows
    actionable = [r for r in selected_rows if r.get("Action") in ["Buy", "Sell"]]
    
    if not actionable:
        raise PreventUpdate
    
    # Stage ALL selected actionable orders as a list
    action_map = {"Buy": "BUY", "Sell": "SELL"}
    
    staged_orders = []
    for row in actionable:
        raw_qty = row.get("meta_shares")
        quantity = float(raw_qty) if raw_qty not in (None, "") else 0.0
        if quantity <= 0:
            try:
                amount = float(row.get("meta_amount", 0) or 0)
                price = float(row.get("meta_price", 0) or 0)
                if price > 0:
                    quantity = abs(amount) / price
            except (TypeError, ValueError):
                quantity = 0.0

        staged_orders.append({
            "ticker": row.get("Ticker"),
            "action": action_map.get(row.get("Action"), "BUY"),
            "quantity": quantity,
            "amount": row.get("meta_amount", 0),
            "price": row.get("meta_price", 0),
            "source": "Rebalancing",
            "staged_at": datetime.now().isoformat()
        })
    
    # Return updated store and button text feedback
    count = len(staged_orders)
    btn_content = [html.I(className="bi bi-check-circle me-2"), f"Staged {count} Order{'s' if count > 1 else ''}!"]
    
    return staged_orders, btn_content
