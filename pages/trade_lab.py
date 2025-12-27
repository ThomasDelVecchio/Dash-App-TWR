import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import dash_wrappers as dw
from components.data_source_badge import create_data_source_badge
from components.monte_carlo import run_monte_carlo_simulation, calculate_trade_impact
from config import GLOBAL_PALETTE
import pandas as pd
import numpy as np

layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.H3("What-If Trade Lab", className="text-light mb-2"),
            html.P("Simulate the impact of hypothetical trades on your long-term success probability.", className="text-muted"),
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        # Input Panel
        dbc.Col(dbc.Card([
            html.Div([
                dbc.CardHeader("Trade Ticket", style={"display": "inline-block", "border": "none"}),
                html.Div(id="trade-data-source-header", style={"display": "inline-block", "paddingTop": "10px"})
            ], className="d-flex justify-content-between align-items-center pe-3"),
            dbc.CardBody([
                dbc.Label("Ticker Symbol"),
                dbc.Input(
                    id="trade-ticker", 
                    placeholder="e.g. SPY", 
                    type="text", 
                    className="mb-3",
                    persistence=True,
                    persistence_type='local'
                ),
                
                dbc.Label("Transaction Type"),
                dbc.RadioItems(
                    id="trade-side",
                    options=[
                        {"label": "Buy (From Cash)", "value": "Buy"},
                        {"label": "Sell (To Cash)", "value": "Sell"},
                        {"label": "Swap (Rebalance)", "value": "Swap"},
                    ],
                    value="Buy",
                    className="mb-3",
                    persistence=True,
                    persistence_type='local'
                ),
                
                dbc.Collapse([
                    dbc.Label("Target Ticker (Buy)"),
                    dbc.Input(
                        id="swap-target-ticker", 
                        placeholder="e.g. GLD", 
                        type="text", 
                        className="mb-3",
                        persistence=True,
                        persistence_type='local'
                    ),
                ], id="swap-target-container", is_open=False),

                dbc.Label("Trade Amount ($)"),
                dbc.Input(
                    id="trade-amount", 
                    placeholder="10000", 
                    type="number", 
                    className="mb-3",
                    persistence=True,
                    persistence_type='local'
                ),
                
                html.Hr(),
                dbc.Button("Run Simulation", id="btn-run-sim", color="primary", className="w-100", size="lg")
            ])
        ]), width=4),
        
        # Results Panel
        dbc.Col(dbc.Card([
            dbc.CardHeader("Simulation Results (10-Year Horizon)"),
            dbc.CardBody([
                dcc.Loading([
                    dcc.Graph(id="sim-overlay-chart", style={"height": "400px"}),
                    html.Div(id="sim-loading-dummy", style={"display": "none"})
                ]),
                html.Small(
                    "Projects future value using Historical Bootstrapping, resampling actual past returns to capture real-world risks. "
                    "Shaded cones represent the 10th-90th percentile range of probable outcomes.",
                    className="text-muted fst-italic d-block mt-2"
                ),
                html.Div(id="sim-stats-display", className="mt-3")
            ])
        ]), width=8)
    ]),

    # DISCLOSURE FOOTER
    html.Hr(),
    html.Div([
        html.Div([
            html.P("Data Sources & Methodology:", className="fw-bold mb-1", style={"display": "inline-block"}),
            html.Div(id="trade-data-source-footer-container", style={"display": "inline-block"})
        ]),
        html.Ul([
            html.Li("Sector Classifications: Sourced from Yahoo Finance / Equity Lookups.", id="trade-sector-source-desc"),
            html.Li("Price Data: Sourced from Yahoo Finance."),
            html.Li("Simulations: Historical Bootstrapping method uses 10-year realized daily returns. Past performance is not indicative of future results.")
        ], className="small text-muted")
    ], className="mb-4")
])

@callback(
    Output("swap-target-container", "is_open"),
    Input("trade-side", "value")
)
def toggle_swap_target(side):
    return side == "Swap"

@callback(
    [Output("trade-lab-state", "data"),
     Output("sim-loading-dummy", "children")],
    [Input("btn-run-sim", "n_clicks")],
    [State("trade-ticker", "value"),
     State("trade-side", "value"),
     State("trade-amount", "value"),
     State("swap-target-ticker", "value"),
     State("theme-store", "data")]
)
def update_trade_lab(n_clicks, ticker, side, amount, swap_target, theme):
    if not n_clicks:
        return dash.no_update, dash.no_update
        
    data = dw.get_data()
    if not data:
        return dash.no_update, dash.no_update
        
    if not ticker or not amount:
        return dash.no_update, dash.no_update
        
    ticker = ticker.upper()
    amount = float(amount)
    
    # 1. Get Current State
    sec_current = data["sec_table_current"]
    holdings = data["holdings"]
    current_value = sec_current["market_value"].sum()
    
    # Get Dynamic Risk Data
    risk_return = data.get("risk_return", {})
    correlation_matrix = data.get("correlation_matrix", {})
    
    # Current Weights {AssetClass: Weight}
    current_weights = sec_current.groupby("asset_class")["weight"].sum().to_dict()
    
    # 2. Calculate New State
    # Need holdings map for ticker lookup
    # Combine holdings file and current sec table to get map
    holdings_map = holdings.set_index("ticker")["asset_class"].to_dict()
    
    prices_df = data.get("prices", pd.DataFrame())
    
    # Handle Swap Target
    if side == "Swap" and swap_target:
        swap_target = swap_target.upper()
    else:
        swap_target = None

    # Calculate Asset Class Weights (Legacy/Fallback)
    new_weights = calculate_trade_impact(
        current_weights, 
        current_value, 
        ticker, 
        amount, 
        side, 
        holdings_map, 
        prices_df,
        swap_target_ticker=swap_target,
        risk_return=risk_return,
        correlation_matrix=correlation_matrix
    )

    # --- ADVANCED: Calculate Ticker-Level Weights for Historical Bootstrapping ---
    # This ensures "Swap Same Class" works and preserves specific asset history (e.g. FBTC vs SPY)
    
    # 1. Reconstruct Full Portfolio (Inc Cash)
    pv_series = data.get("pv", pd.Series())
    total_pv = pv_series.iloc[-1] if not pv_series.empty else current_value
    
    current_ticker_vals = sec_current.set_index("ticker")["market_value"].to_dict()
    invested_sum = sum(current_ticker_vals.values())
    current_cash = max(0, total_pv - invested_sum) # Derived cash
    
    if "CASH" not in current_ticker_vals:
        current_ticker_vals["CASH"] = current_cash
    else:
        current_ticker_vals["CASH"] += current_cash # If 'CASH' row exists, add derived residual
        
    # 2. Apply Trade Logic
    new_ticker_vals = current_ticker_vals.copy()
    
    # Ensure target ticker exists in map if new
    if side == "Buy" and ticker not in new_ticker_vals:
        new_ticker_vals[ticker] = 0.0
    if side == "Swap" and swap_target and swap_target not in new_ticker_vals:
        new_ticker_vals[swap_target] = 0.0

    if side == "Buy":
        # Check cash
        available_cash = new_ticker_vals.get("CASH", 0.0)
        # execute_trade_batch caps it. We should too.
        actual_buy = min(available_cash, amount) if amount > available_cash else amount
        
        new_ticker_vals["CASH"] = max(0, available_cash - actual_buy)
        new_ticker_vals[ticker] = new_ticker_vals.get(ticker, 0.0) + actual_buy
        
    elif side == "Sell":
        current_pos = new_ticker_vals.get(ticker, 0.0)
        actual_sell = min(current_pos, amount) if amount > current_pos else amount
        
        new_ticker_vals[ticker] = max(0, current_pos - actual_sell)
        new_ticker_vals["CASH"] = new_ticker_vals.get("CASH", 0.0) + actual_sell
        
    elif side == "Swap":
        current_pos = new_ticker_vals.get(ticker, 0.0)
        actual_swap = min(current_pos, amount) if amount > current_pos else amount
        
        new_ticker_vals[ticker] = max(0, current_pos - actual_swap)
        new_ticker_vals[swap_target] = new_ticker_vals.get(swap_target, 0.0) + actual_swap

    # 3. Normalize to Weights
    total_curr_val = sum(current_ticker_vals.values())
    if total_curr_val <= 0: total_curr_val = 1.0
    current_ticker_weights = {k: v/total_curr_val for k, v in current_ticker_vals.items()}
    
    total_new_val = sum(new_ticker_vals.values())
    if total_new_val <= 0: total_new_val = 1.0
    new_ticker_weights = {k: v/total_new_val for k, v in new_ticker_vals.items()}

    # 4. Generate CRN Seed
    # Same seed for both runs ensures apples-to-apples comparison of the portfolio change
    random_seed = np.random.randint(0, 1000000)
    
    # 3. Run Simulations
    # Current
    sim_current = run_monte_carlo_simulation(
        current_value, 
        current_weights, 
        correlation_matrix=correlation_matrix,
        risk_return=risk_return,
        prices_df=prices_df,
        random_seed=random_seed,
        ticker_weights=current_ticker_weights,
        holdings_map=holdings_map
    )
    # Hypothetical
    sim_new = run_monte_carlo_simulation(
        current_value, 
        new_weights, 
        correlation_matrix=correlation_matrix,
        risk_return=risk_return,
        prices_df=prices_df,
        random_seed=random_seed,
        ticker_weights=new_ticker_weights,
        holdings_map=holdings_map
    )
    
    if not sim_current or not sim_new:
        return dash.no_update, dash.no_update
        
    # 4. Plot Results (Probability Density / Distribution Overlay)
    # We'll plot the 10th-90th percentile cones over time
    
    fig = go.Figure()
    years = sim_current["years"]
    
    # Helper for adding traces
    def add_sim_traces(sim_data, name, color_line, color_fill):
        # Median Line
        fig.add_trace(go.Scatter(
            x=years, 
            y=sim_data["percentiles"]["50"],
            mode="lines",
            name=f"{name} Median",
            line=dict(color=color_line, width=2)
        ))
        # Cone (10th-90th)
        fig.add_trace(go.Scatter(
            x=years + years[::-1],
            y=sim_data["percentiles"]["90"] + sim_data["percentiles"]["10"][::-1],
            fill='toself',
            fillcolor=color_fill,
            line=dict(color='rgba(255,255,255,0)'),
            name=f"{name} 90% Conf.",
            showlegend=True
        ))

    # Current (Blue)
    add_sim_traces(sim_current, "Current", GLOBAL_PALETTE[0], dw._hex_to_rgba(GLOBAL_PALETTE[0], 0.2))
    
    # New (Orange/Gold)
    add_sim_traces(sim_new, "Hypothetical", GLOBAL_PALETTE[10], dw._hex_to_rgba(GLOBAL_PALETTE[10], 0.2))
    
    fig.update_layout(
        title="Projected Portfolio Value (10 Years)",
        xaxis_title="Years",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white" if theme == "light" else "plotly_dark",
        hovermode="x unified"
    )
    
    # Stats Text - Helper
    def get_stat_row(label, val_curr, val_new, inverse=False):
        diff = val_new - val_curr
        # Standard: Higher is Better (Green). Inverse: Lower is Better (Green).
        # For VaR/CVaR (Portfolio Value Floor), Higher is Better (Safer).
        is_better = diff > 0 if not inverse else diff < 0
        color = "green" if is_better else "red"
        
        return html.Div([
            html.H6(label, className="mt-3 mb-1"),
            html.Div([
                html.Span("Current: ", className="text-muted"),
                f"${val_curr:,.0f}"
            ]),
            html.Div([
                html.Span("Hypothetical: ", className="fw-bold"),
                f"${val_new:,.0f} ",
                html.Span(f"({diff:+,.0f})", style={"color": color, "font-weight": "bold"})
            ])
        ])

    # Extract Metrics
    curr_med = sim_current["percentiles"]["50"][-1]
    new_med = sim_new["percentiles"]["50"][-1]
    
    curr_var = sim_current["metrics"].get("var_95", 0)
    new_var = sim_new["metrics"].get("var_95", 0)
    
    curr_cvar = sim_current["metrics"].get("cvar_95", 0)
    new_cvar = sim_new["metrics"].get("cvar_95", 0)
    
    stats = html.Div([
        html.H5("Simulation Statistics", className="mb-3 text-primary"),
        
        get_stat_row("Median Outcome (50th Percentile)", curr_med, new_med),
        
        get_stat_row("95% Value at Risk (VaR)", curr_var, new_var),
        html.Small("Portfolio value in worst 5% of outcomes.", className="text-muted"),
        
        get_stat_row("Expected Shortfall (CVaR)", curr_cvar, new_cvar),
        html.Small("Average value if the worst-case scenario happens.", className="text-muted"),
    ])
    
    # Save results to store (CACHE OUTPUT)
    result_data = {
        "fig": fig,
        "stats": stats
    }
    
    return result_data, "loaded"

# Render results from Cache (Input/Output Caching)
@callback(
    [Output("sim-overlay-chart", "figure"),
     Output("sim-stats-display", "children"),
     Output("trade-data-source-footer-container", "children"),
     Output("trade-sector-source-desc", "children"),
     Output("trade-data-source-header", "children")],
    [Input("trade-lab-state", "data")]
)
def render_trade_lab_results(data):
    # Always try to show data source status even if sim hasn't run
    full_data = dw.get_data()
    source_summary = dw.get_data_source_summary(full_data)
    
    # Create two separate instances with unique IDs for tooltips
    def create_badge_with_suffix(summary, suffix):
        if not summary: return None
        badge_div = create_data_source_badge(summary)
        # badge_div is html.Div([badge, tooltip])
        badge = badge_div.children[0]
        tooltip = badge_div.children[1]
        
        badge.id = f"data-source-badge-{suffix}"
        tooltip.target = f"data-source-badge-{suffix}"
        return badge_div

    footer_badge = create_badge_with_suffix(source_summary, "footer")
    header_badge = create_badge_with_suffix(source_summary, "header")

    # Dynamic Sector Description
    if source_summary and source_summary.get('all_fmp'):
        sector_desc = "Sector Classifications: Sourced from Financial Modeling Prep (FMP)."
    elif source_summary and source_summary.get('sources', {}).get('FMP', 0) > 0:
        sector_desc = "Sector Classifications: Mixed sources (FMP and Yahoo Finance)."
    else:
        sector_desc = "Sector Classifications: Sourced from Yahoo Finance / Equity Lookups."

    if not data or not isinstance(data, dict):
        return go.Figure(), "Enter trade details and click Run Simulation.", footer_badge, sector_desc, header_badge
    
    # Reconstruct stats (Components are JSON serializable dictionaries in Dash)
    # If stats comes back as a dict, Dash can render it.
    
    return data.get("fig", go.Figure()), data.get("stats", ""), footer_badge, sector_desc, header_badge
