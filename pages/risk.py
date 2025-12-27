import dash
from dash import dcc, html, callback, Input, Output, State, ALL, MATCH
import dash_bootstrap_components as dbc
import dash_wrappers as dw
import plotly.graph_objects as go
from components.data_source_badge import create_data_source_badge
from config import TARGET_MONTHLY_CONTRIBUTION, GLOBAL_PALETTE
from components.monte_carlo import calculate_portfolio_sigma
import pandas as pd

layout = html.Div([
    # 1. RISK & CORRELATION ROW
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Risk vs Expected Return", className="card-title"),
                dcc.Graph(id={'type': 'filter-chart', 'index': 'risk-chart'}),
                html.Small(
                    "Plots annualized volatility (10Y) vs trailing 12-month return for each asset class. "
                    "Data is based on daily price history sourced from Yahoo Finance.",
                    className="text-muted fst-italic"
                )
            ])
        ]), width=6, className="mb-4"),
        
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Rolling Correlations (90-Day)", className="card-title"),
                dcc.Graph(id='correlation-heatmap'),
                html.Small(
                    "90-day correlation heat map where values near 1.0 indicate assets moving together, and values near 0 or negative indicate diversification benefits.",
                    className="text-muted fst-italic"
                )
            ])
        ]), width=6, className="mb-4"),
    ]),
    
    # 2. DRAWDOWN ROW (Underwater Chart)
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Drawdown Analysis (Underwater Chart)", className="card-title"),
                dcc.Graph(id='drawdown-chart'),
                html.Small(
                    "Shows the percentage decline from the historical peak (High Water Mark). "
                    "Useful for understanding the depth and duration of past losses.",
                    className="text-muted fst-italic"
                )
            ])
        ]), width=12, className="mb-4"),
    ]),

    # 3. PROJECTIONS ROW
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("20-Year Projections (Interactive)", className="card-title"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Expected Annual Return (%)"),
                        dcc.Slider(
                            id='proj-return-slider',
                            min=2, max=30, step=0.5, value=7,
                            marks={i: f'{i}%' for i in range(5, 31, 5)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            persistence=True,
                            persistence_type='local'
                        )
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Monthly Contribution ($)"),
                        dcc.Slider(
                            id='proj-contrib-slider',
                            min=0, max=5000, step=100, value=TARGET_MONTHLY_CONTRIBUTION,
                            marks={0: '$0', 1000: '$1k', 2500: '$2.5k', 5000: '$5k'},
                            tooltip={"placement": "bottom", "always_visible": True},
                            persistence=True,
                            persistence_type='local'
                        )
                    ], width=6),
                ], className="mb-3"),
                dcc.Graph(id='projections-chart'),
                html.Small(
                    "Projects future value using compound growth assumptions (CAGR). "
                    "This is a theoretical projection based on your input rate, distinct from the 'Realized' metrics below.",
                    className="text-muted fst-italic"
                )
            ])
        ]), width=12, className="mb-4"),
    ]),
    
    # 3. PORTFOLIO SIMULATOR (ASSET ALLOCATION)
    html.Hr(className="my-4"),
    dbc.Row([
        dbc.Col([
            html.H4("Asset Allocation Simulator", className="mb-3 text-light"),
            html.P("Adjust target weights to see impact on Portfolio Risk/Return profile.", className="text-muted")
        ])
    ]),
    
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.Div(id='simulator-sliders-container'),
                html.Hr(),
                html.Div(id='total-weight-display', className="text-center fw-bold mb-3"),
                dbc.Button("Recalculate Profile", id="btn-recalculate-sim", color="primary", className="w-100")
            ])
        ]), width=6),
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    dcc.Loading(dcc.Graph(id='sim-expected-return-gauge', config={'displayModeBar': False}))
                ], width=6),
                dbc.Col([
                    dcc.Loading(dcc.Graph(id='sim-volatility-gauge', config={'displayModeBar': False}))
                ], width=6)
            ]),
            html.Div(
                html.Small(
                    "Estimates impact of rebalancing on risk profile. "
                    "'Expected Return' is the weighted average of realized TTM returns; 'Volatility' is the weighted annualized standard deviation (10Y).",
                    className="text-muted fst-italic"
                ), className="mt-2 text-center"
            )
        ], width=6)
    ], className="mb-4"),
    
    # DISCLOSURE FOOTER
    html.Hr(),
    html.Div([
        html.Div([
            html.P("Data Sources & Methodology:", className="fw-bold mb-1", style={"display": "inline-block"}),
            html.Div(id="risk-data-source-container", style={"display": "inline-block"})
        ]),
        html.Ul([
            html.Li(id="risk-sector-source-desc"),
            html.Li("Price Data: Sourced from Yahoo Finance."),
            html.Li("Risk Metrics: Based on realized historical performance (Trailing 12-Month Return, 10-Year Volatility). Past performance is not indicative of future results.")
        ], className="small text-muted")
    ], className="mb-4")
])

# --- CALLBACKS ---

@callback(
    [Output({'type': 'filter-chart', 'index': 'risk-chart'}, 'figure'),
     Output('correlation-heatmap', 'figure'),
     Output('drawdown-chart', 'figure'),
     Output('projections-chart', 'figure'),
     Output('risk-data-source-container', 'children'),
     Output('risk-sector-source-desc', 'children')],
    [Input('data-signal', 'data'),
     Input('theme-store', 'data'),
     Input('proj-return-slider', 'value'),
     Input('proj-contrib-slider', 'value')]
)
def update_risk_page(signal, theme, proj_return, proj_contrib):
    data = dw.get_data()
    if not data: 
        return {}, {}, {}, {}
    
    # 1. Risk Scatter
    risk_fig = dw.get_risk_return_chart(data, theme)
    
    # 2. Correlation Heatmap
    corr_fig = dw.get_correlation_heatmap(data, theme)
    
    # 3. Drawdown Chart
    dd_fig = dw.get_drawdown_chart(data, theme)
    
    # 4. Projections
    proj_fig = dw.get_projections_chart(data, theme, rate_pct=proj_return, monthly_contrib=proj_contrib)
    
    # Data Source Badge
    source_summary = dw.get_data_source_summary(data)
    source_badge = create_data_source_badge(source_summary)
    
    # Dynamic Sector Description
    if source_summary and source_summary.get('all_fmp'):
        sector_desc = "Sector Classifications: Sourced from Financial Modeling Prep (FMP)."
    elif source_summary and source_summary.get('sources', {}).get('FMP', 0) > 0:
        sector_desc = "Sector Classifications: Mixed sources (FMP and Yahoo Finance)."
    else:
        sector_desc = "Sector Classifications: Sourced from Yahoo Finance / Equity Lookups."
        
    return risk_fig, corr_fig, dd_fig, proj_fig, source_badge, sector_desc

# --- SIMULATOR CALLBACKS (EXISTING LOGIC) ---

@callback(
    Output('simulator-sliders-container', 'children'),
    [Input('data-signal', 'data')],
    [State('asset-allocation-state', 'data')]
)
def populate_simulator_sliders(signal, saved_state):
    """Generate sliders for each asset class based on current portfolio."""
    data = dw.get_data()
    if not data:
        return html.Div("Loading portfolio data...", className="text-muted p-3")
    
    sec_table = data.get('sec_table_current')
    if sec_table is None or sec_table.empty:
        return html.Div("No portfolio data available", className="text-muted p-3")
        
    # Get Dynamic Risk Model Keys
    risk_return = data.get("risk_return", {})
    if not risk_return:
        return html.Div("Risk model unavailable", className="text-muted p-3")
    
    # Calculate current allocation percentages
    asset_weights = sec_table.groupby('asset_class')['weight'].sum() * 100
    
    # Filter to asset classes that either exist in RISK_RETURN or are CASH
    available_classes = [ac for ac in asset_weights.index if ac in risk_return or ac == 'CASH']
    
    if not available_classes:
        return html.Div("No asset classes found in risk model", className="text-muted p-3")
    
    asset_weights = asset_weights[available_classes]
    asset_weights = asset_weights.sort_values(ascending=False)
    
    sliders = []
    for asset_class in asset_weights.index:
        # Use saved state if available, otherwise use current portfolio weight
        if saved_state and asset_class in saved_state:
            current_weight = saved_state[asset_class]
        else:
            current_weight = asset_weights[asset_class]
        
        slider_component = html.Div([
            html.Label(f"{asset_class}: {current_weight:.1f}%", id={'type': 'slider-label', 'index': asset_class}, className="mb-1"),
            dbc.InputGroup([
                dbc.Button("-", id={'type': 'sim-btn-minus', 'index': asset_class}, n_clicks=0, color="secondary", outline=True, size="sm"),
                html.Div(
                    dcc.Slider(
                        id={'type': 'sim-slider', 'index': asset_class},
                        min=0, max=100, step=0.1,
                        value=current_weight,
                        marks={0: '', 100: ''}, 
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                    style={"flex": "1", "padding": "5px 10px"}
                ),
                dbc.Button("+", id={'type': 'sim-btn-plus', 'index': asset_class}, n_clicks=0, color="secondary", outline=True, size="sm"),
            ], className="mb-3 align-items-center")
        ])
        
        sliders.append(slider_component)
    
    return html.Div(sliders)

@callback(
    Output({'type': 'sim-slider', 'index': MATCH}, 'value'),
    [Input({'type': 'sim-btn-minus', 'index': MATCH}, 'n_clicks'),
     Input({'type': 'sim-btn-plus', 'index': MATCH}, 'n_clicks')],
    [State({'type': 'sim-slider', 'index': MATCH}, 'value')]
)
def update_slider_value(n_minus, n_plus, current_val):
    if current_val is None: return 0
    ctx = dash.callback_context
    if not ctx.triggered: return current_val
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Check if minus or plus
    if 'sim-btn-minus' in button_id:
        return max(0, round(current_val - 0.1, 1))
    elif 'sim-btn-plus' in button_id:
        return min(100, round(current_val + 0.1, 1))
    
    return current_val

@callback(
    [Output('sim-expected-return-gauge', 'figure'),
     Output('sim-volatility-gauge', 'figure'),
     Output('total-weight-display', 'children'),
     Output({'type': 'slider-label', 'index': ALL}, 'children')],
    [Input('btn-recalculate-sim', 'n_clicks'),
     Input({'type': 'sim-slider', 'index': ALL}, 'value')],
    [State({'type': 'sim-slider', 'index': ALL}, 'id'),
     State('data-signal', 'data'),
     State('theme-store', 'data')]
)
def update_simulator(n_clicks, slider_values, slider_ids, signal, theme):
    """Calculate portfolio statistics based on slider weights."""
    data = dw.get_data()
    
    if not data or not slider_ids or not slider_values:
        empty_fig = go.Figure()
        return empty_fig, empty_fig, "No data", []
    
    # Get Dynamic Risk Data
    risk_return = data.get("risk_return", {})
    correlation_matrix = data.get("correlation_matrix", {})
    
    asset_classes = [sid['index'] for sid in slider_ids]
    total_weight = sum(slider_values)
    labels = [f"{ac}: {val:.1f}%" for ac, val in zip(asset_classes, slider_values)]
    
    if abs(total_weight - 100) < 0.1:
        weight_color = "#28a745"
        weight_msg = f"✓ Total: {total_weight:.1f}%"
    else:
        weight_color = "#dc3545"
        weight_msg = f"⚠ Total: {total_weight:.1f}% (should be 100%)"
    
    total_display = html.Span(weight_msg, style={'color': weight_color})
    
    # Current Stats
    sec_table = data.get('sec_table_current')
    # Use full table to ensure Cash is included in weights for correct normalization
    current_weights_df = sec_table.groupby('asset_class')['weight'].sum()
    
    current_return = 0
    # Current Volatility (Enhanced Risk Model)
    current_weights_dict = current_weights_df.to_dict()
    current_vol = calculate_portfolio_sigma(current_weights_dict, correlation_matrix, risk_return) * 100.0
    
    for ac, weight in current_weights_df.items():
        if ac in risk_return:
            current_return += weight * risk_return[ac]['return']
            # Volatility is now calculated via Matrix above
    
    # Simulated Stats
    sim_return = 0
    
    # Construct Simulated Weights Dictionary
    sim_weights = {}
    for ac, weight_pct in zip(asset_classes, slider_values):
        weight = weight_pct / 100.0
        sim_weights[ac] = weight
        if ac in risk_return:
            sim_return += weight * risk_return[ac]['return']
            
    # Simulated Volatility (Enhanced Risk Model)
    sim_vol = calculate_portfolio_sigma(sim_weights, correlation_matrix, risk_return) * 100.0
            
    # Gauges
    template = "plotly_dark" if theme == "dark" else "plotly_white"
    
    # Dynamic Range for Return Gauge (Handle high TTM returns)
    # Ensure minimum range of 0-30 for visibility, or scale up if return is higher
    max_ret_range = max(30, current_return * 1.25, sim_return * 1.25)
    
    return_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=sim_return,
        number={'valueformat': '.2f'},
        delta={'reference': current_return, 'suffix': '%', 'valueformat': '.2f'},
        title={'text': "Expected Return (TTM) %"},
        gauge={
            'axis': {'range': [0, max_ret_range]},
            'bar': {'color': GLOBAL_PALETTE[0]},
            'steps': [
                {'range': [0, max_ret_range*0.33], 'color': dw._hex_to_rgba(GLOBAL_PALETTE[2], 0.3)},
                {'range': [max_ret_range*0.33, max_ret_range*0.66], 'color': dw._hex_to_rgba(GLOBAL_PALETTE[10], 0.3)},
                {'range': [max_ret_range*0.66, max_ret_range], 'color': dw._hex_to_rgba(GLOBAL_PALETTE[4], 0.3)}
            ],
            'threshold': {'line': {'color': GLOBAL_PALETTE[2], 'width': 4}, 'thickness': 0.75, 'value': current_return}
        }
    ))
    return_gauge.update_layout(template=template, height=300, margin=dict(l=20, r=20, t=50, b=20))
    
    vol_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=sim_vol,
        number={'valueformat': '.2f'},
        delta={'reference': current_vol, 'suffix': '%', 'valueformat': '.2f'},
        title={'text': "Volatility (%)"},
        gauge={
            'axis': {'range': [0, 30]},
            'bar': {'color': GLOBAL_PALETTE[6]},
            'steps': [
                {'range': [0, 10], 'color': dw._hex_to_rgba(GLOBAL_PALETTE[4], 0.3)},
                {'range': [10, 20], 'color': dw._hex_to_rgba(GLOBAL_PALETTE[10], 0.3)},
                {'range': [20, 30], 'color': dw._hex_to_rgba(GLOBAL_PALETTE[2], 0.3)}
            ],
            'threshold': {'line': {'color': GLOBAL_PALETTE[2], 'width': 4}, 'thickness': 0.75, 'value': current_vol}
        }
    ))
    vol_gauge.update_layout(template=template, height=300, margin=dict(l=20, r=20, t=50, b=20))
    
    return return_gauge, vol_gauge, total_display, labels

# Save asset allocation simulator state
@callback(
    Output('asset-allocation-state', 'data'),
    [Input({'type': 'sim-slider', 'index': ALL}, 'value')],
    [State({'type': 'sim-slider', 'index': ALL}, 'id')]
)
def save_allocation_state(slider_values, slider_ids):
    if not slider_ids or not slider_values:
        return dash.no_update
    
    asset_classes = [sid['index'] for sid in slider_ids]
    state_data = {ac: val for ac, val in zip(asset_classes, slider_values)}
    
    return state_data

# Note: Asset allocation slider state is saved to session storage.
# The sliders will restore to their last state within the current session,
# but will reset to portfolio weights when the session ends or app is restarted.
