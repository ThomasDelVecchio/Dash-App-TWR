import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL
import dash_bootstrap_components as dbc
from datetime import datetime
import pandas as pd

# Import wrappers
import dash_wrappers as dw

# Import Components
from components import chatbot
from components.audit_modal import get_audit_modal_content

# Import Pages
from pages import overview, performance, allocations, attribution, flows, holdings, risk, settings, trade_lab, help_index

# Initialize App
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    title="Portfolio Analytics"
)

# Initialize Data Cache
try:
    dw.refresh_data()
    print("Initial data load complete.")
except Exception as e:
    print(f"Initial data load failed: {e}")

# Sidebar Component
sidebar = html.Div(
    [
        html.H3("DELVEX", className="display-6"),
        html.P("Portfolio Analytics", className="lead"),
        html.Hr(),
        
        dbc.Nav(
            [
                dbc.NavLink("Overview", href="/", active="exact"),
                dbc.NavLink("Performance", href="/performance", active="exact"),
                dbc.NavLink("Allocations", href="/allocations", active="exact"),
                dbc.NavLink("Attribution", href="/attribution", active="exact"),
                dbc.NavLink("Flows", href="/flows", active="exact"),
                dbc.NavLink("Holdings", href="/holdings", active="exact"),
                dbc.NavLink("Risk & Proj", href="/risk", active="exact"),
                dbc.NavLink("Trade Lab", href="/trade-lab", active="exact"),
                dbc.NavLink("Settings", href="/settings", active="exact"),
                dbc.NavLink("Help Index", href="/help", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
        
        html.Hr(),
        
        # Controls
        html.Div([
            dbc.Label("Theme"),
            dbc.Switch(id="theme-switch", label="Dark Mode", value=True, className="mb-2"),
            
            dbc.Label("Analysis End Date"),
            dcc.DatePickerSingle(
                id="date-picker-end",
                date=datetime.now().date(),
                display_format="YYYY-MM-DD",
                className="mb-2 d-block",
                style={'zIndex': 100}
            ),
            
            dbc.Label("Benchmarks"),
            dcc.Dropdown(
                id="benchmark-dropdown",
                options=[
                    {"label": "S&P 500 (SPY)", "value": "SPY"},
                    {"label": "Global 60/40 (AOR)", "value": "AOR"},
                    {"label": "Conservative 40/60 (AOK)", "value": "AOK"},
                    {"label": "Nasdaq 100 (QQQ)", "value": "QQQ"},
                    {"label": "Total Int'l Stock (VXUS)", "value": "VXUS"},
                    {"label": "Total Bond Market (BND)", "value": "BND"},
                ],
                value=["SPY", "AOK", "AOR"],
                multi=True,
                className="mb-2 text-dark"
            ),
            
            dbc.Label("Include Exited Tickers", className="mt-2"),
            dbc.RadioItems(
                id="include-exited-radio",
                options=[
                    {"label": "Yes", "value": True},
                    {"label": "No", "value": False},
                ],
                value=False, # Default to No (Hidden)
                inline=True,
                className="mb-2"
            ),

            html.Hr(),
            dbc.Button("Clear All Filters", id="btn-clear-global", color="secondary", className="w-100"),
        ]),
    ],
    id="sidebar",
    className="sidebar",
)

# Content Container
content = html.Div(id="page-content", className="content")

# Main Layout
app.layout = html.Div(
    [
        dcc.Location(id="url"),
        
        # Stores for Global State
        dcc.Store(id="data-signal", data=datetime.now().isoformat()),
        dcc.Store(id="theme-store", data="dark"),
        dcc.Store(id="date-range-store"),
        dcc.Store(id="benchmark-store"),
        dcc.Store(id="filter-store", storage_type="memory"),
        dcc.Store(id="include-exited-store", data=False),
        
        # Global Audit Store
        dcc.Store(id="audit-request-store"),
        
        # Simulator State Stores (persisted in localStorage)
        dcc.Store(id="trade-lab-state", storage_type="local"),
        dcc.Store(id="asset-allocation-state", storage_type="session"),
        dcc.Store(id="projections-state", storage_type="local"),
        
        # Force Global MathJax Load
        dcc.Markdown(id="mathjax-preload", mathjax=True, style={"display": "none"}),
        
        # Global Audit Modal
        dbc.Modal(
            id="audit-modal",
            size="lg",
            centered=True,
            is_open=False,
            fade=False,
            style={"zIndex": 1050} # Ensure on top
        ),
        
        # Toggle Button
        html.Button(
            "☰", 
            id="btn-sidebar-toggle", 
            className="btn btn-secondary", 
            style={
                "position": "fixed", 
                "top": "10px", 
                "left": "10px", 
                "zIndex": 1100,
                "borderRadius": "50%",
                "width": "40px",
                "height": "40px",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "fontSize": "1.2rem",
                "paddingBottom": "4px"
            }
        ),
        
        sidebar,
        content,
        chatbot.layout
    ],
    id="main-container",
    **{"data-theme": "dark"}
)

# Validation Layout (Required for multi-page apps with global callbacks)
app.validation_layout = html.Div([
    app.layout,
    chatbot.layout,
    overview.layout,
    performance.layout,
    allocations.layout,
    attribution.layout,
    flows.layout,
    holdings.layout,
    risk.layout,
    settings.layout,
    trade_lab.layout,
    help_index.layout
])

# ============================================================
# CALLBACKS
# ============================================================

# 1. Router
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return overview.layout
    elif pathname == "/performance":
        return performance.layout
    elif pathname == "/allocations":
        return allocations.layout
    elif pathname == "/attribution":
        return attribution.layout
    elif pathname == "/flows":
        return flows.layout
    elif pathname == "/holdings":
        return holdings.layout
    elif pathname == "/risk":
        return risk.layout
    elif pathname == "/trade-lab":
        return trade_lab.layout
    elif pathname == "/settings":
        return settings.layout
    elif pathname == "/help":
        return help_index.layout
    return dbc.Container(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="py-3"
    )

# 2. Global State Updates
@app.callback(
    [Output("theme-store", "data"),
     Output("main-container", "data-theme"),
     Output("date-range-store", "data"),
     Output("benchmark-store", "data"),
     Output("data-signal", "data"),
     Output("include-exited-store", "data")],
    [Input("theme-switch", "value"),
     Input("date-picker-end", "date"),
     Input("benchmark-dropdown", "value"),
     Input("include-exited-radio", "value")]
)
def update_global_state(is_dark, end_date, benchmarks, include_exited):
    # Refresh data with new end date
    dw.refresh_data(end_date=end_date)
    
    theme = "dark" if is_dark else "light"
    
    dates = {"end": end_date} if end_date else None
    
    bm_map = {}
    if benchmarks:
        for b in benchmarks:
            # Simple label mapping
            label = b
            if b == "SPY": label = "S&P 500"
            elif b == "AOR": label = "Global 60/40"
            elif b == "AOK": label = "Cons 40/60"
            elif b == "QQQ": label = "Nasdaq 100"
            elif b == "VXUS": label = "Total Int'l"
            elif b == "BND": label = "Total Bond"
            bm_map[label] = b
            
    return theme, theme, dates, bm_map, datetime.now().isoformat(), include_exited

# 3. Global Filter Logic
@app.callback(
    Output("filter-store", "data"),
    [Input({'type': 'filter-chart', 'index': ALL}, 'clickData'),
     Input("btn-clear-global", "n_clicks")],
    [State("filter-store", "data")]
)
def update_filter_store(all_charts_click, clear_btn, current_filters):
    
    ctx = callback_context
    if not ctx.triggered:
        return current_filters or {}
    
    # Identify trigger
    triggered_id = ctx.triggered_id
    
    # Initialize store if None
    filters = current_filters or {}
    
    # Clear Logic
    if triggered_id == "btn-clear-global":
        return {}
        
    # Helper to update with toggle logic
    def update_key(key, value):
        if filters.get(key) == value:
            # Toggle off
            filters[key] = None
        else:
            # Set new value
            filters[key] = value
        return filters
    
    # Chart Click Logic
    # triggered_id is a dict for pattern matching callbacks: {'index': '...', 'type': '...'}
    if isinstance(triggered_id, dict) and triggered_id.get("type") == "filter-chart":
        chart_index = triggered_id["index"]
        
        # Get value from ctx.triggered (list of changed props)
        # We need the value of the component that triggered.
        # Since 'all_charts_click' is a list of ALL charts' clickData, finding the right one is tricky via args.
        # But ctx.triggered[0]['value'] gives the value of the trigger.
        click_data = ctx.triggered[0]["value"]
        
        if not click_data: return filters
        
        try:
            if chart_index == "asset-pie-chart":
                val = click_data["points"][0]["label"]
                return update_key("asset_class", val)
                
            elif chart_index == "asset-bar-chart":
                val = click_data["points"][0]["x"]
                return update_key("asset_class", val)
                
            elif chart_index == "sector-chart":
                val = click_data["points"][0]["y"] # Horizontal bar
                return update_key("sector", val)
                
            elif chart_index == "risk-chart":
                val = click_data["points"][0].get("hovertext")
                if val: return update_key("asset_class", val)
                    
            elif chart_index == "flows-chart":
                val = click_data["points"][0]["y"] # Horizontal bar (Asset Class)
                return update_key("asset_class", val)
                
            elif chart_index == "ticker-pie-chart":
                val = click_data["points"][0]["label"]
                return update_key("ticker", val)
                
            elif chart_index == "ticker-bar-chart":
                val = click_data["points"][0]["x"]
                return update_key("ticker", val)
                
        except Exception as e:
            print(f"Error updating filter store: {e}")
            return filters
            
    return filters

# 4. Sidebar Toggle Logic
@app.callback(
    [Output("sidebar", "className"),
     Output("page-content", "className")],
    [Input("btn-sidebar-toggle", "n_clicks")],
    [State("sidebar", "className"),
     State("page-content", "className")]
)
def toggle_sidebar(n, sidebar_class, content_class):
    if n:
        if "hidden" in sidebar_class:
            return sidebar_class.replace(" hidden", ""), content_class.replace(" expanded", "")
        else:
            return sidebar_class + " hidden", content_class + " expanded"
    return sidebar_class, content_class

# 5. Chatbot Callbacks
chatbot.register_callbacks(app)

# 6. Global Audit Callback
@app.callback(
    [Output("audit-modal", "is_open"),
     Output("audit-modal", "children")],
    [Input("audit-request-store", "data")],
    [State("audit-modal", "is_open")]
)
def toggle_audit_modal(request_data, is_open):
    if not request_data:
        return False, []
    
    # Fetch detailed data on demand
    try:
        detailed_request = dw.fetch_audit_details(request_data)
    except Exception as e:
        print(f"Error fetching audit details: {e}")
        detailed_request = request_data

    # Generate content
    body = get_audit_modal_content(detailed_request)
    
    # Add Header
    header = dbc.ModalHeader(dbc.ModalTitle("Explain This Number"), close_button=True)
    
    return True, [header, body]

if __name__ == "__main__":
    app.run(debug=True)
