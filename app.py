import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL
import dash_bootstrap_components as dbc
from datetime import datetime
import pandas as pd
import os
import hashlib

# Import wrappers
import dash_wrappers as dw

# Import Config
from config import NAV_MODULES, is_etrade_configured, ETRADE_AUTO_SYNC

# Import Components
from components import chatbot
from components.audit_modal import get_audit_modal_content

# Import Pages
from pages import overview, performance, allocations, attribution, flows, holdings, risk, settings, trade_lab, help_index, taxes, rebalancing, custom_report, trade_execution, strategy_backtesting

# ============================================================
# E*TRADE SYNC ON STARTUP
# ============================================================
_ETRADE_SYNC_STATUS = None
_LAST_REFRESHED_SYNC_TIME = None  # Track when we last refreshed for a sync

def _run_etrade_sync():
    """Run E*TRADE sync if configured. Updates global status and returns it."""
    global _ETRADE_SYNC_STATUS
    
    if not is_etrade_configured():
        print("⚠️ E*TRADE not configured. Skipping sync.")
        return None
        
    if not ETRADE_AUTO_SYNC:
        print("ℹ️ E*TRADE auto-sync disabled via ETRADE_AUTO_SYNC=false")
        return None
    
    try:
        from etrade_sync import sync_all
        
        print("🔄 Starting E*TRADE sync...")
        result = sync_all()
        
        # Check status field (not "success" boolean)
        status = result.get("status", "error")
        
        if status == "success":
            tx_count = result.get("transactions_added", 0)
            holdings_updated = result.get("holdings_updated", False)
            print(f"✅ E*TRADE sync complete: {tx_count} new transactions, holdings {'updated' if holdings_updated else 'unchanged'}")
        elif status == "partial":
            print(f"⚠️ E*TRADE sync partial: {result.get('message', 'Some issues')}")
        else:
            print(f"❌ E*TRADE sync error: {result.get('message', 'Unknown error')}")
        
        # Use sync result directly (not stale file data)
        _ETRADE_SYNC_STATUS = result
        return _ETRADE_SYNC_STATUS
        
    except Exception as e:
        print(f"❌ E*TRADE sync failed: {e}")
        print("ℹ️  Continuing with existing local data files (cashflows.csv, sample holdings.csv)")
        _ETRADE_SYNC_STATUS = {"status": "error", "message": str(e)}
        return _ETRADE_SYNC_STATUS

# Run E*TRADE sync BEFORE loading data cache
# Only run on the main process (not Flask reloader) to prevent double-sync
if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
    _run_etrade_sync()
else:
    # Reloader process: Load existing sync status from file
    from etrade_sync import get_sync_status
    _ETRADE_SYNC_STATUS = get_sync_status()

# Initialize App
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css"
    ],
    external_scripts=["https://cdnjs.cloudflare.com/ajax/libs/Sortable/1.15.0/Sortable.min.js"],
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
def _get_sync_status_badge():
    """Generate sync status badge for sidebar."""
    global _ETRADE_SYNC_STATUS
    
    if not is_etrade_configured():
        return html.Div(
            [
                html.I(className="bi bi-cloud-slash me-2"),
                html.Span("E*TRADE not configured", style={"fontSize": "0.8rem"})
            ],
            className="text-muted mb-2",
            style={"fontSize": "0.75rem"}
        )
    
    if _ETRADE_SYNC_STATUS is None:
        return html.Div()
    
    # Check status field (not "success" boolean)
    status = _ETRADE_SYNC_STATUS.get("status", "error")
    
    if status == "success":
        last_sync = _ETRADE_SYNC_STATUS.get("last_sync", "Unknown")
        tx_count = _ETRADE_SYNC_STATUS.get("transactions_added", 0)
        
        # Format the time nicely
        try:
            sync_dt = datetime.fromisoformat(last_sync)
            time_str = sync_dt.strftime("%I:%M %p")
        except:
            time_str = "Recently"
        
        return html.Div(
            [
                html.Div(
                    [
                        html.I(className="bi bi-cloud-check me-2 text-success"),
                        html.Span(f"Synced at {time_str}", style={"fontSize": "0.8rem"})
                    ]
                ),
                html.Div(
                    f"+{tx_count} new trades" if tx_count > 0 else "No new trades",
                    className="text-muted",
                    style={"fontSize": "0.7rem", "marginLeft": "1.25rem"}
                )
            ],
            className="mb-2"
        )
    else:
        error = _ETRADE_SYNC_STATUS.get("message", "Sync failed")
        return html.Div(
            [
                html.I(className="bi bi-exclamation-triangle me-2 text-warning"),
                html.Span("Sync issue", style={"fontSize": "0.8rem"}),
                html.Div(
                    error[:30] + "..." if len(error) > 30 else error,
                    className="text-muted",
                    style={"fontSize": "0.65rem", "marginLeft": "1.25rem"}
                )
            ],
            className="mb-2"
        )

sidebar = html.Div(
    [
        # Brand Header with integrated toggle
        html.Div([
            html.Div([
                html.I(className="bi bi-graph-up-arrow sidebar-brand-icon"),
                html.Span("DELVEX", className="display-6 sidebar-brand-text"),
            ], className="sidebar-brand-left"),
            html.Div(
                [
                    html.Button(
                        html.I(className="bi bi-x-lg"),
                        id="btn-sidebar-toggle",
                        className="btn btn-link sidebar-toggle-btn",
                        title="Close sidebar"
                    ),
                    html.Button(
                        html.I(
                            className="bi bi-layout-sidebar-inset",
                            id="sidebar-collapse-icon"
                        ),
                        id="btn-sidebar-collapse",
                        className="btn btn-link sidebar-toggle-btn",
                        title="Collapse sidebar"
                    ),
                    html.Button(
                        html.I(className="bi bi-arrow-repeat"),
                        id="btn-refresh-data",
                        className="btn btn-link sidebar-toggle-btn sidebar-refresh-btn",
                        title="Refresh data"
                    ),
                    html.Div(
                        "",
                        id="refresh-status",
                        className="refresh-status"
                    ),
                ],
                className="d-flex flex-column"
            ),
        ], className="sidebar-brand"),
        html.P("Portfolio Analytics", className="lead sidebar-subtitle"),
        
        # E*TRADE Sync Status Badge (dynamically updated via callback)
        html.Div(id="etrade-sync-badge", className="sidebar-sync-badge"),
        
        html.Hr(),
        
        dbc.Nav(
            [],
            id="sidebar-nav",
            vertical=True,
            pills=True,
        ),
        
        html.Hr(),
        
        # Controls (wrapped for collapse behavior)
        html.Div([
            dbc.Label("Tax Methodology"),
            dbc.Select(
                id="tax-strategy-select",
                options=[
                    {"label": "FIFO (First-In First-Out)", "value": "FIFO"},
                    {"label": "LIFO (Last-In First-Out)", "value": "LIFO"},
                    {"label": "HIFO (Highest-In First-Out)", "value": "HIFO"},
                ],
                value="FIFO",
                persistence=True,
                persistence_type="local",
                className="mb-2"
            ),
            
            dbc.Label("Analysis End Date"),
            dbc.Input(
                id="date-picker-end",
                type="date",
                value=datetime.now().date().isoformat(),
                className="mb-2",
                persistence=True,
                persistence_type="local"
            ),
            
            dbc.Label("Benchmarks"),
            html.Div([
                dbc.Button(
                    "3 Selected",
                    id="btn-benchmark-picker",
                    color="outline-light",
                    size="sm",
                    className="w-100 text-start benchmark-picker-btn",
                ),
                dbc.Offcanvas(
                    id="offcanvas-benchmark",
                    title="Select Benchmarks",
                    is_open=False,
                    placement="bottom",
                    close_button=True,
                    className="benchmark-offcanvas",
                    children=[
                        dbc.Checklist(
                            id="benchmark-dropdown",
                            options=[
                                {"label": "S&P 500 (SPY)", "value": "SPY"},
                                {"label": "Total Stock (VTI)", "value": "VTI"},
                                {"label": "Growth (VUG)", "value": "VUG"},
                                {"label": "Aggressive 80/20 (AOA)", "value": "AOA"},
                                {"label": "Global 60/40 (AOR)", "value": "AOR"},
                                {"label": "Cons 40/60 (AOK)", "value": "AOK"},
                                {"label": "Nasdaq 100 (QQQ)", "value": "QQQ"},
                            ],
                            value=["SPY", "VTI", "AOA"],
                            switch=True,
                            className="benchmark-checklist",
                            persistence=True,
                            persistence_type="local"
                        ),
                    ],
                ),
            ], className="mb-2"),
            
            dbc.Label("Include Exited Tickers", className="mt-2"),
            dbc.RadioItems(
                id="include-exited-radio",
                options=[
                    {"label": "Yes", "value": True},
                    {"label": "No", "value": False},
                ],
                value=False, # Default to No (Hidden)
                inline=True,
                className="mb-2",
                persistence=True,
                persistence_type="local"
            ),

            html.Hr(),
            dbc.Button("Clear All Filters", id="btn-clear-global", color="secondary", className="w-100"),
        ], className="sidebar-controls"),
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
        
        # Interval for E*TRADE sync status polling (every 30 seconds)
        dcc.Interval(id="sync-status-interval", interval=30*1000, n_intervals=0),

        # Global Toast Stack (prevents overlap)
        html.Div(
            [
                # Global Error Toast (dismissable)
                dbc.Toast(
                    id="app-error-toast",
                    header="Data Quality Warning",
                    is_open=False,
                    dismissable=True,
                    icon="warning",
                    duration=None,
                    className="toast-stack-item",
                    style={
                        "width": "420px",
                        "maxWidth": "90vw",
                    },
                ),

                # Price As-Of Toast (dismissable, non-persistent)
                dbc.Toast(
                    id="price-asof-toast",
                    header="Price Data Timestamp",
                    is_open=False,
                    dismissable=True,
                    icon="info",
                    duration=None,
                    className="toast-stack-item",
                    style={
                        "width": "420px",
                        "maxWidth": "90vw",
                    },
                ),
            ],
            className="toast-stack",
        ),
        
        # Stores for Global State
        dcc.Store(id="data-signal", data=datetime.now().isoformat()),
        dcc.Store(id="theme-store", data="dark"),
        dcc.Store(id="date-range-store"),
        dcc.Store(id="benchmark-store"),
        dcc.Store(id="filter-store", storage_type="memory"),
        dcc.Store(id="include-exited-store", data=False),
        dcc.Store(id="tax-strategy-store", data="FIFO", storage_type="local"),
        dcc.Store(id="active-modules-store", storage_type="local"),
        dcc.Store(id="error-toast-store", data={"dismissed": False, "hash": None}, storage_type="session"),
        dcc.Store(id="price-toast-store", data={"dismissed": False, "hash": None}, storage_type="session"),
        
        # Global Audit Store
        dcc.Store(id="audit-request-store"),
        
        # Simulator State Stores (persisted in localStorage)
        dcc.Store(id="trade-lab-state", storage_type="local"),
        dcc.Store(id="asset-allocation-state", storage_type="session"),
        dcc.Store(id="projections-state", storage_type="local"),
        
        # Staged Order Store (for Rebalancing -> Trade flow)
        dcc.Store(id="staged-order-store", storage_type="session"),
        dcc.Store(id="staged-order-index", data=0),
        
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
        
        # Floating Toggle Button (visible only when sidebar is hidden)
        html.Button(
            html.I(className="bi bi-list"),
            id="btn-sidebar-open",
            className="btn btn-secondary sidebar-open-btn",
            title="Open sidebar"
        ),

        # Floating Next Page Button
        html.Button(
            html.I(className="bi bi-chevron-right"),
            id="btn-next-page",
            className="btn btn-primary next-page-btn",
            title="Next Page"
        ),

        # Floating Previous Page Button (hidden on first page)
        html.Button(
            html.I(className="bi bi-chevron-left"),
            id="btn-prev-page",
            className="btn btn-secondary prev-page-btn",
            title="Previous Page"
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
    strategy_backtesting.layout,
    allocations.layout,
    attribution.layout,
    flows.layout,
    holdings.layout,
    rebalancing.layout,
    trade_execution.layout,
    risk.layout,
    settings.layout,
    trade_lab.layout,
    taxes.layout,
    help_index.layout,
    custom_report.layout
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
    elif pathname == "/strategy-backtesting":
        return strategy_backtesting.layout
    elif pathname == "/allocations":
        return allocations.layout
    elif pathname == "/attribution":
        return attribution.layout
    elif pathname == "/flows":
        return flows.layout
    elif pathname == "/holdings":
        return holdings.layout
    elif pathname == "/rebalancing":
        return rebalancing.layout
    elif pathname == "/trade":
        return trade_execution.layout
    elif pathname == "/risk":
        return risk.layout
    elif pathname == "/trade-lab":
        return trade_lab.layout
    elif pathname == "/taxes":
        return taxes.layout
    elif pathname == "/custom-report":
        return custom_report.layout
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
     Output("include-exited-store", "data"),
     Output("tax-strategy-store", "data")],
    [Input("date-picker-end", "value"),
     Input("benchmark-dropdown", "value"),
     Input("include-exited-radio", "value"),
     Input("tax-strategy-select", "value")],
    [State("data-signal", "data")]
)
def update_global_state(end_date, benchmarks, include_exited, tax_strategy, current_signal):
    ctx = callback_context
    refresh_triggered = ctx.triggered_id == "date-picker-end"

    # Refresh data ONLY when end date changes
    if refresh_triggered:
        dw.refresh_data(end_date=end_date)
    
    theme = "dark"
    
    dates = {"end": end_date} if end_date else None
    
    bm_map = {}
    if isinstance(benchmarks, str):
        benchmarks = [benchmarks]
    if benchmarks:
        for b in benchmarks:
            # Simple label mapping
            label = b
            if b == "SPY": label = "S&P 500 (SPY)"
            elif b == "VTI": label = "Total Stock (VTI)"
            elif b == "VUG": label = "Growth (VUG)"
            elif b == "AOA": label = "Aggressive (AOA)"
            elif b == "AOR": label = "Global 60/40 (AOR)"
            elif b == "AOK": label = "Cons 40/60 (AOK)"
            elif b == "QQQ": label = "Nasdaq 100 (QQQ)"
            bm_map[label] = b
            
    signal = datetime.now().isoformat() if refresh_triggered else current_signal

    return theme, theme, dates, bm_map, signal, include_exited, tax_strategy

# 2a-1. Benchmark picker offcanvas toggle
@app.callback(
    Output("offcanvas-benchmark", "is_open"),
    Input("btn-benchmark-picker", "n_clicks"),
    State("offcanvas-benchmark", "is_open"),
    prevent_initial_call=True
)
def toggle_benchmark_offcanvas(n, is_open):
    return not is_open

# 2a-2. Update benchmark picker button text
@app.callback(
    Output("btn-benchmark-picker", "children"),
    Input("benchmark-dropdown", "value")
)
def update_benchmark_btn_label(values):
    n = len(values) if values else 0
    return f"{n} Benchmark{'s' if n != 1 else ''} Selected"

# 2b. Manual Refresh Button (Force Price Re-fetch)
@app.callback(
    [Output("data-signal", "data", allow_duplicate=True),
     Output("refresh-status", "children")],
    [Input("btn-refresh-data", "n_clicks")],
    [State("date-picker-end", "value")],
    prevent_initial_call=True
)
def refresh_data_button(n_clicks, end_date):
    if not n_clicks:
        return dash.no_update, dash.no_update

    dw.refresh_data(end_date=end_date, force_price_refresh=True)
    return datetime.now().isoformat(), "Refresh complete"

# 3. Global Error Toast
@app.callback(
    [Output("app-error-toast", "children"),
     Output("app-error-toast", "is_open"),
     Output("error-toast-store", "data")],
    [Input("data-signal", "data")],
    [State("error-toast-store", "data")]
)
def update_error_toast(_signal, toast_state):
    data = dw.get_data()
    if not data:
        return "", False, {"dismissed": False, "hash": None}

    errors = data.get("errors", [])
    if not errors and "prices" in data and hasattr(data["prices"], "attrs"):
        errors = data["prices"].attrs.get("errors", [])

    if not errors:
        return "", False, {"dismissed": False, "hash": None}

    toast_state = toast_state or {"dismissed": False, "hash": None}
    last_hash = toast_state.get("hash")
    dismissed = toast_state.get("dismissed", False)

    error_hash = hashlib.md5("\n".join(errors).encode("utf-8")).hexdigest()

    toast_body = html.Div(
        [
            html.Hr(className="my-1"),
        ]
        + [
            html.Div(
                e,
                className="small mb-2",
                style={"whiteSpace": "pre-wrap", "wordBreak": "break-word"},
            )
            for e in errors
        ]
    )

    if error_hash != last_hash:
        return toast_body, True, {"dismissed": False, "hash": error_hash}

    if dismissed:
        return toast_body, False, {"dismissed": True, "hash": error_hash}

    return toast_body, True, {"dismissed": False, "hash": error_hash}


# 3b. Price As-Of Toast (Non-persistent)
@app.callback(
    [Output("price-asof-toast", "children"),
     Output("price-asof-toast", "is_open"),
     Output("price-toast-store", "data")],
    [Input("data-signal", "data")],
    [State("price-toast-store", "data")]
)
def update_price_asof_toast(_signal, toast_state):
    data = dw.get_data()
    if not data:
        return "", False, {"dismissed": False, "hash": None}

    price_fetched_at = data.get("price_fetched_at")
    benchmark_fetched_at = data.get("benchmark_fetched_at")
    price_cache_source = data.get("price_cache_source")
    benchmark_cache_source = data.get("benchmark_cache_source")
    price_cache_expiry_hours = data.get("price_cache_expiry_hours", 12)

    if price_fetched_at is None and benchmark_fetched_at is None:
        return "", False, {"dismissed": False, "hash": None}

    try:
        price_fetch_ts = pd.Timestamp(price_fetched_at) if price_fetched_at else None
        price_fetch_str = price_fetch_ts.strftime("%Y-%m-%d %I:%M %p") if price_fetch_ts else "N/A"
    except Exception:
        price_fetch_ts = None
        price_fetch_str = "N/A"

    try:
        bench_fetch_ts = pd.Timestamp(benchmark_fetched_at) if benchmark_fetched_at else None
        bench_fetch_str = bench_fetch_ts.strftime("%Y-%m-%d %I:%M %p") if bench_fetch_ts else "N/A"
    except Exception:
        bench_fetch_ts = None
        bench_fetch_str = "N/A"

    now_ts = pd.Timestamp.now()
    price_age = "N/A"
    if price_fetch_ts is not None:
        age_minutes = (now_ts - price_fetch_ts).total_seconds() / 60.0
        price_age = f"{age_minutes:.1f} min ago"

    bench_age = "N/A"
    if bench_fetch_ts is not None:
        age_minutes = (now_ts - bench_fetch_ts).total_seconds() / 60.0
        bench_age = f"{age_minutes:.1f} min ago"

    price_source = price_cache_source or "unknown"
    bench_source = benchmark_cache_source or "unknown"

    def _status_label(cache_source, fetch_ts):
        if fetch_ts is None:
            return "UNKNOWN"
        if cache_source == "live":
            return "LIVE"
        # Treat any memory source as cached
        if cache_source in {"memory", "memory-fallback"}:
            age_mins = (now_ts - fetch_ts).total_seconds() / 60.0
            expiry_mins = float(price_cache_expiry_hours) * 60.0
            return "STALE CACHED" if age_mins > expiry_mins else "CACHED"
        return "CACHED"

    price_status = _status_label(price_source, price_fetch_ts)
    bench_status = _status_label(bench_source, bench_fetch_ts)

    body = html.Div(
        [
            html.Div(f"Prices pulled at: {price_fetch_str} ({price_age}) — {price_status}"),
            html.Div(f"Benchmarks pulled at: {bench_fetch_str} ({bench_age}) — {bench_status}")
        ],
        className="small"
    )

    toast_state = toast_state or {"dismissed": False, "hash": None}
    last_hash = toast_state.get("hash")
    dismissed = toast_state.get("dismissed", False)

    toast_hash = hashlib.md5(
        f"{price_fetched_at}|{benchmark_fetched_at}|{price_cache_source}|{benchmark_cache_source}".encode("utf-8")
    ).hexdigest()

    if toast_hash != last_hash:
        return body, True, {"dismissed": False, "hash": toast_hash}

    if dismissed:
        return body, False, {"dismissed": True, "hash": toast_hash}

    return body, True, {"dismissed": False, "hash": toast_hash}


@app.callback(
    Output("price-toast-store", "data", allow_duplicate=True),
    [Input("price-asof-toast", "is_open")],
    [State("price-toast-store", "data")],
    prevent_initial_call=True
)
def track_price_toast_dismiss(is_open, toast_state):
    if is_open:
        return dash.no_update

    toast_state = toast_state or {"dismissed": False, "hash": None}
    if not toast_state.get("hash"):
        return dash.no_update

    toast_state["dismissed"] = True
    return toast_state


@app.callback(
    Output("error-toast-store", "data", allow_duplicate=True),
    [Input("app-error-toast", "is_open")],
    [State("error-toast-store", "data")],
    prevent_initial_call=True
)
def track_toast_dismiss(is_open, toast_state):
    if is_open:
        return dash.no_update

    toast_state = toast_state or {"dismissed": False, "hash": None}
    if not toast_state.get("hash"):
        return dash.no_update

    toast_state["dismissed"] = True
    return toast_state

# 3. Global Filter Logic
@app.callback(
    Output("filter-store", "data"),
    [Input({'type': 'filter-chart', 'index': ALL}, 'clickData'),
     Input("btn-clear-global", "n_clicks")],
    [State("filter-store", "data")]
)
def update_filter_store(all_charts_click, _clear_btn, current_filters):
    
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
     Output("page-content", "className"),
     Output("btn-sidebar-open", "style")],
    [Input("btn-sidebar-toggle", "n_clicks"),
     Input("btn-sidebar-open", "n_clicks")],
    [State("sidebar", "className"),
     State("page-content", "className")]
)
def toggle_sidebar(_n_close, _n_open, sidebar_class, content_class):
    ctx = callback_context
    
    # Style for hidden floating button
    hidden_style = {"display": "none"}
    # Style for visible floating button
    visible_style = {
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
        "fontSize": "1.2rem"
    }
    
    if not ctx.triggered:
        # Initial state: sidebar visible, floating button hidden
        return sidebar_class, content_class, hidden_style
    
    triggered_id = ctx.triggered_id
    
    if triggered_id == "btn-sidebar-toggle":
        # Close button clicked (inside sidebar)
        return sidebar_class + " hidden", content_class + " expanded", visible_style
    elif triggered_id == "btn-sidebar-open":
        # Open button clicked (floating button)
        return sidebar_class.replace(" hidden", ""), content_class.replace(" expanded", ""), hidden_style
    
    return sidebar_class, content_class, hidden_style

# 5. E*TRADE Sync Badge Callback (Dynamic Update + Data Refresh)
@app.callback(
    [Output("etrade-sync-badge", "children"),
     Output("data-signal", "data", allow_duplicate=True)],
    [Input("sync-status-interval", "n_intervals")],
    [State("data-signal", "data")],
    prevent_initial_call=True
)
def update_sync_badge(n_intervals, current_signal):
    """Dynamically update the E*TRADE sync status badge and refresh data if sync changed."""
    global _LAST_REFRESHED_SYNC_TIME
    
    # Reload sync status from file to detect changes from background sync
    from etrade_sync import get_sync_status
    current_status = get_sync_status()
    current_sync_time = current_status.get("last_sync")
    
    # Check if sync occurred since our last refresh
    if current_sync_time and current_sync_time != _LAST_REFRESHED_SYNC_TIME:
        print(f"🔄 Detected new sync ({current_sync_time}), refreshing data cache...")
        _LAST_REFRESHED_SYNC_TIME = current_sync_time
        dw.refresh_data()
        # Update global status for badge
        global _ETRADE_SYNC_STATUS
        _ETRADE_SYNC_STATUS = current_status
        # Return new signal to trigger UI updates
        return _get_sync_status_badge(), datetime.now().isoformat()
    
    return _get_sync_status_badge(), dash.no_update

# 6. Chatbot Callbacks
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

# 7. Sidebar Modules Callback
@app.callback(
    Output("sidebar-nav", "children"),
    Input("active-modules-store", "data")
)
def update_sidebar_modules(active_modules_ids):
    """Generate sidebar nav links with icons and section group dividers."""
    
    def create_nav_link(module):
        """Create a NavLink with icon and text for responsive sidebar."""
        icon_class = module.get("icon", "bi-circle")
        return dbc.NavLink(
            [
                html.I(className=f"{icon_class} nav-icon"),
                html.Span(module["label"], className="nav-text")
            ],
            href=module["href"],
            active="exact"
        )
    
    def create_section_label(label):
        """Create a subtle all-caps section divider label (Linear/Notion style)."""
        return html.Div(
            label,
            className="sidebar-section-label"
        )
    
    # Build the visible module list (respecting toggle state)
    visible_modules = []
    for m in NAV_MODULES:
        if not m["can_toggle"]:
            visible_modules.append(m)
        elif active_modules_ids is None or m["id"] in active_modules_ids:
            visible_modules.append(m)
    
    # Render links with section group dividers inserted between groups
    children = []
    current_group = None
    
    for m in visible_modules:
        group = m.get("group", "")
        
        # Insert a section label when the group changes (skip OVERVIEW — it's just the home link)
        if group != current_group:
            current_group = group
            if group and group != "OVERVIEW":
                children.append(create_section_label(group))
        
        children.append(create_nav_link(m))
            
    return children

# 8. Next Page Navigation Callback
@app.callback(
    Output("url", "pathname", allow_duplicate=True),
    Input("btn-next-page", "n_clicks"),
    [State("url", "pathname"),
     State("active-modules-store", "data")],
    prevent_initial_call=True
)
def navigate_to_next_page(n_clicks, current_path, active_modules_ids):
    if not n_clicks:
        return dash.no_update
        
    # Build list of active pages in order
    active_pages = []
    for m in NAV_MODULES:
        # If can_toggle is False, it's always active (e.g. Overview)
        if not m["can_toggle"]:
            active_pages.append(m["href"])
        # If can_toggle is True, check if it's enabled in active_modules_ids
        # If active_modules_ids is None (default), ALL toggleable modules are active
        elif active_modules_ids is None or m["id"] in active_modules_ids:
            active_pages.append(m["href"])
            
    # Find current index
    try:
        # Handle trailing slashes or exact matches
        clean_current = current_path.rstrip("/") if len(current_path) > 1 else current_path
        
        # Simple exact match first
        if clean_current in active_pages:
            idx = active_pages.index(clean_current)
        else:
            # Fallback: try matching with/without slash
            idx = -1
            for i, p in enumerate(active_pages):
                if p == clean_current or p == current_path:
                    idx = i
                    break
        
        if idx != -1:
            # Go to next page (loop back to start if at end)
            next_idx = (idx + 1) % len(active_pages)
            return active_pages[next_idx]
        else:
            # If current page not found in nav (e.g. 404), go to Home
            return "/"
            
    except Exception as e:
        print(f"Navigation error: {e}")
        return dash.no_update

# 9. Previous Page Navigation Callback
@app.callback(
    Output("url", "pathname", allow_duplicate=True),
    Input("btn-prev-page", "n_clicks"),
    [State("url", "pathname"),
     State("active-modules-store", "data")],
    prevent_initial_call=True
)
def navigate_to_prev_page(n_clicks, current_path, active_modules_ids):
    if not n_clicks:
        return dash.no_update

    # Build list of active pages in order
    active_pages = []
    for m in NAV_MODULES:
        if not m["can_toggle"]:
            active_pages.append(m["href"])
        elif active_modules_ids is None or m["id"] in active_modules_ids:
            active_pages.append(m["href"])

    try:
        clean_current = current_path.rstrip("/") if len(current_path) > 1 else current_path

        if clean_current in active_pages:
            idx = active_pages.index(clean_current)
        else:
            idx = -1
            for i, p in enumerate(active_pages):
                if p == clean_current or p == current_path:
                    idx = i
                    break

        if idx > 0:
            return active_pages[idx - 1]
        return dash.no_update

    except Exception as e:
        print(f"Navigation error: {e}")
        return dash.no_update

# 10. Back Button Visibility (hide on first page)
@app.callback(
    Output("btn-prev-page", "style"),
    [Input("url", "pathname"),
     Input("active-modules-store", "data")]
)
def toggle_prev_button_visibility(current_path, active_modules_ids):
    active_pages = []
    for m in NAV_MODULES:
        if not m["can_toggle"]:
            active_pages.append(m["href"])
        elif active_modules_ids is None or m["id"] in active_modules_ids:
            active_pages.append(m["href"])

    clean_current = current_path.rstrip("/") if current_path and len(current_path) > 1 else current_path

    if not active_pages or clean_current not in active_pages:
        return {"display": "none"}

    return {"display": "none"} if active_pages.index(clean_current) == 0 else {}

# 11. Next Button Visibility (hide on last page)
@app.callback(
    Output("btn-next-page", "style"),
    [Input("url", "pathname"),
     Input("active-modules-store", "data")]
)
def toggle_next_button_visibility(current_path, active_modules_ids):
    active_pages = []
    for m in NAV_MODULES:
        if not m["can_toggle"]:
            active_pages.append(m["href"])
        elif active_modules_ids is None or m["id"] in active_modules_ids:
            active_pages.append(m["href"])

    clean_current = current_path.rstrip("/") if current_path and len(current_path) > 1 else current_path

    if not active_pages or clean_current not in active_pages:
        return {"display": "none"}

    return {"display": "none"} if active_pages.index(clean_current) == (len(active_pages) - 1) else {}

if __name__ == "__main__":
    app.run(debug=True, dev_tools_ui=False)

