import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import dash_wrappers as dw
from components.page_header import page_header
from config import BENCHMARK_PRESETS, TARGET_WEIGHT_PRESET_NAME
from data_loader import load_holdings
import pandas as pd


def _preset_options():
    options = [{"label": TARGET_WEIGHT_PRESET_NAME, "value": TARGET_WEIGHT_PRESET_NAME}]
    options.extend([{"label": p["name"], "value": p["name"]} for p in BENCHMARK_PRESETS])
    return options


def _default_custom_benchmark_rows():
    try:
        holdings = load_holdings()
    except Exception:
        holdings = pd.DataFrame()

    if holdings.empty:
        return [
            {"Ticker": "QQQ", "Weight": 50},
            {"Ticker": "GLD", "Weight": 50},
        ]

    cols = [str(c).strip().lower() for c in holdings.columns]
    if "target_pct" in cols:
        target_col = holdings.columns[cols.index("target_pct")]
    else:
        target_col = None
        for key in cols:
            if "target" in key and "pct" in key:
                target_col = holdings.columns[cols.index(key)]
                break
        if target_col is None:
            for key in cols:
                if "target" in key:
                    target_col = holdings.columns[cols.index(key)]
                    break

    if target_col is None:
        return [
            {"Ticker": "QQQ", "Weight": 50},
            {"Ticker": "GLD", "Weight": 50},
        ]

    subset = holdings.copy()
    subset.columns = cols
    subset = subset[subset["ticker"].astype(str).str.upper() != "CASH"]

    rows = []
    for _, row in subset.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        try:
            weight = float(row.get(target_col, 0) or 0)
        except Exception:
            weight = 0.0
        if weight <= 0:
            continue
        rows.append({"Ticker": ticker, "Weight": weight})

    if rows:
        return rows

    return [
        {"Ticker": "QQQ", "Weight": 50},
        {"Ticker": "GLD", "Weight": 50},
    ]


layout = html.Div([
    page_header(
        title="Strategy Backtesting",
        icon="bi-activity",
        subtitle="Quarterly rebalanced strategy backtests against legendary benchmarks and custom rivals"
    ),

    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Lookback Period"),
                    dcc.Dropdown(
                        id="strategy-lookback",
                        options=[
                            {"label": "1Y", "value": "1Y"},
                            {"label": "3Y", "value": "3Y"},
                            {"label": "5Y", "value": "5Y"},
                            {"label": "10Y", "value": "10Y"},
                            {"label": "15Y", "value": "15Y"},
                            {"label": "Max", "value": "MAX"},
                        ],
                        value="MAX",
                        clearable=False,
                        className="mb-2"
                    )
                ], width=3),
                dbc.Col([
                    dbc.Label("Initial Investment ($)"),
                    dbc.Input(
                        id="strategy-initial",
                        type="number",
                        min=1000,
                        step=500,
                        value=10000,
                        className="mb-2"
                    )
                ], width=3),
                dbc.Col([
                    dbc.Label("Benchmark Presets"),
                    dcc.Dropdown(
                        id="strategy-preset-checklist",
                        options=_preset_options(),
                        value=[TARGET_WEIGHT_PRESET_NAME] + [p["name"] for p in BENCHMARK_PRESETS],
                        multi=True,
                        className="mb-2"
                    )
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "Custom Rival Builder",
                        id="btn-custom-toggle",
                        color="secondary",
                        className="me-2"
                    ),
                    dbc.Badge(id="strategy-backtest-badge", color="secondary", className="ms-2")
                ], width=12)
            ])
        ])
    ], className="mb-4"),

    dbc.Collapse([
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Custom Benchmark Name"),
                        dbc.Input(id="custom-benchmark-name", value="Custom Benchmark")
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Build Weights (Must Sum to 100%)"),
                        html.Div(id="custom-benchmark-error", className="text-warning small")
                    ], width=8)
                ], className="mb-2"),
                dag.AgGrid(
                    id="custom-benchmark-grid",
                    columnDefs=[
                        {"headerName": "Ticker", "field": "Ticker", "editable": True, "flex": 1},
                        {"headerName": "Weight %", "field": "Weight", "editable": True, "type": "numericColumn", "flex": 1},
                    ],
                    rowData=_default_custom_benchmark_rows(),
                    defaultColDef={"resizable": True, "sortable": True},
                    dashGridOptions={
                        "rowSelection": "multiple",
                        "animateRows": True,
                        "domLayout": "autoHeight"
                    },
                    className="ag-theme-alpine-dark"
                ),
                dbc.Row([
                    dbc.Col([
                        dbc.Button("Add Row", id="btn-custom-add", color="secondary", className="me-2"),
                        dbc.Button("Remove Selected", id="btn-custom-remove", color="secondary", className="me-2"),
                        dbc.Button("Apply Custom Benchmark", id="btn-custom-apply", color="primary")
                    ], width=12)
                ], className="mt-2")
            ])
        ], className="mb-4")
    ], id="custom-benchmark-collapse", is_open=False),

    dcc.Store(id="custom-benchmark-store"),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Backtest Portfolio Weights", className="card-title section-header"),
                html.Div(
                    dcc.Loading(dag.AgGrid(
                        id="strategy-weights-table",
                        columnDefs=[
                            {"headerName": "Portfolio", "field": "Portfolio", "flex": 1},
                            {"headerName": "Ticker", "field": "Ticker", "flex": 1},
                            {"headerName": "Weight", "field": "Weight", "type": "numericColumn", "flex": 1, "valueFormatter": {"function": "d3.format('.2%')(params.value)"}},
                        ],
                        rowData=[],
                        defaultColDef={"resizable": True, "sortable": True, "filter": True, "minWidth": 0},
                        dashGridOptions={"domLayout": "normal"},
                        className="ag-theme-alpine-dark audit-target strategy-weights-table",
                        style={"width": "100%", "height": "450px"}
                    )),
                    className="strategy-scorecard-flex"
                )
            ], className="strategy-scorecard-body")
        ], className="strategy-scorecard-card"), width=12, className="mb-4 strategy-scorecard-full"),
    ], className="strategy-scorecard-row"),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Growth of $ Investment", className="card-title section-header"),
                dcc.Loading(dcc.Graph(id="strategy-growth-chart")),
                html.P(
                    "* Growth curves assume quarterly rebalancing and show hypothetical value from the initial investment.",
                    className="footnote small text-muted px-3"
                )
            ])
        ]), width=12, className="mb-4"),
    ]),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Underwater Chart (Drawdown)", className="card-title section-header"),
                dcc.Loading(dcc.Graph(id="strategy-drawdown-chart")),
                html.P(
                    "* Drawdown shows peak-to-trough decline from each strategy’s prior high-water mark.",
                    className="footnote small text-muted px-3"
                )
            ])
        ]), width=12, className="mb-4"),
    ]),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Risk vs Return", className="card-title section-header"),
                dcc.Loading(dcc.Graph(id="strategy-risk-chart")),
                html.P(
                    "* Return is CAGR; volatility is annualized standard deviation over the selected lookback.",
                    className="footnote small text-muted px-3"
                )
            ])
        ]), width=12, className="mb-4"),
    ]),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Strategy Scorecard", className="card-title section-header"),
                html.Div(
                    dcc.Loading(dag.AgGrid(
                        id="strategy-scorecard",
                        columnDefs=[
                            {"headerName": "Strategy", "field": "Strategy", "pinned": "left", "minWidth": 220, "flex": 2, "lockPinned": True, "cellClass": "lock-pinned"},
                            {"headerName": "Overall Score", "field": "Overall Score", "type": "numericColumn", "flex": 1, "valueFormatter": {"function": "d3.format('.2f')(params.value)"}},
                            {"headerName": "CAGR", "field": "CAGR", "type": "numericColumn", "flex": 1, "valueFormatter": {"function": "d3.format('.2%')(params.value)"}},
                            {"headerName": "Volatility", "field": "Volatility", "type": "numericColumn", "flex": 1, "valueFormatter": {"function": "d3.format('.2%')(params.value)"}},
                            {"headerName": "Sharpe", "field": "Sharpe", "type": "numericColumn", "flex": 1, "valueFormatter": {"function": "d3.format('.2f')(params.value)"}},
                            {"headerName": "Sortino", "field": "Sortino", "type": "numericColumn", "flex": 1, "valueFormatter": {"function": "d3.format('.2f')(params.value)"}},
                            {"headerName": "Max Drawdown", "field": "Max Drawdown", "type": "numericColumn", "flex": 1, "valueFormatter": {"function": "d3.format('.2%')(params.value)"}},
                        ],
                        rowData=[],
                        defaultColDef={"resizable": True, "sortable": True, "filter": True, "minWidth": 110},
                        dashGridOptions={"domLayout": "autoHeight"},
                        className="ag-theme-alpine-dark audit-target",
                        style={"width": "100%"}
                    )),
                    className="strategy-scorecard-flex"
                )
            ], className="strategy-scorecard-body")
        ], className="strategy-scorecard-card"), width=12, className="mb-4 strategy-scorecard-full"),
    ], className="strategy-scorecard-row"),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(
                dbc.Button(
                    "Simulation Diagnostics / Proxy Log",
                    id="btn-proxy-toggle",
                    color="link",
                    className="p-0 text-decoration-none"
                )
            ),
            dbc.Collapse([
                dbc.CardBody([
                    html.Div(
                        dcc.Loading(dag.AgGrid(
                            id="strategy-proxy-log",
                            columnDefs=[
                                {"headerName": "Ticker", "field": "Ticker", "pinned": "left", "minWidth": 160, "flex": 1, "lockPinned": True, "cellClass": "lock-pinned"},
                                {"headerName": "Proxy", "field": "Proxy Used", "flex": 1, "minWidth": 160},
                                {
                                    "headerName": "Status",
                                    "field": "Status",
                                    "flex": 1,
                                    "minWidth": 140,
                                    "cellStyle": {
                                        "function": "params.value === 'Failed' ? {'color':'#dc3545','fontWeight':600} : params.value === 'Partial History' ? {'color':'#ffc107','fontWeight':600} : params.value === 'Spliced' ? {'color':'#f0ad4e','fontWeight':600} : {}"
                                    },
                                },
                                {"headerName": "Asset Class", "field": "Asset Class", "flex": 1, "minWidth": 180},
                                {"headerName": "Original Start", "field": "Original Start", "flex": 1, "minWidth": 160},
                                {"headerName": "Proxy Start", "field": "Proxy Start", "flex": 1, "minWidth": 160},
                                {"headerName": "Requested Start", "field": "Requested Start", "flex": 1, "minWidth": 160},
                            ],
                            rowData=[],
                            defaultColDef={"resizable": True, "sortable": True, "filter": True, "minWidth": 110},
                            dashGridOptions={"domLayout": "normal"},
                            className="ag-theme-alpine-dark audit-target",
                            style={"width": "100%", "height": "70vh"}
                        )),
                        className="strategy-scorecard-flex"
                    )
                ])
            ], id="strategy-proxy-collapse", is_open=False)
        ], className="strategy-scorecard-card"), width=12, className="mb-4 strategy-scorecard-full"),
    ], className="strategy-scorecard-row"),

    html.Div(id="strategy-error", className="text-warning small")
], className="strategy-backtesting-page")


@callback(
    Output("custom-benchmark-collapse", "is_open"),
    Input("btn-custom-toggle", "n_clicks"),
    State("custom-benchmark-collapse", "is_open")
)
def toggle_custom_benchmark(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


@callback(
    Output("strategy-proxy-collapse", "is_open"),
    Input("btn-proxy-toggle", "n_clicks"),
    State("strategy-proxy-collapse", "is_open")
)
def toggle_proxy_log(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


@callback(
    Output("custom-benchmark-grid", "rowData"),
    [Input("btn-custom-add", "n_clicks"), Input("btn-custom-remove", "n_clicks")],
    [State("custom-benchmark-grid", "rowData"), State("custom-benchmark-grid", "selectedRows")]
)
def update_custom_grid(add_clicks, remove_clicks, row_data, selected_rows):
    row_data = row_data or []
    ctx = dash.callback_context
    if not ctx.triggered:
        return row_data

    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger == "btn-custom-add":
        row_data.append({"Ticker": "", "Weight": 0})
        return row_data

    if trigger == "btn-custom-remove" and selected_rows:
        selected_keys = set()
        for row in selected_rows:
            ticker = str(row.get("Ticker", "")).strip().upper()
            try:
                weight = float(row.get("Weight", 0) or 0)
            except Exception:
                weight = 0.0
            selected_keys.add((ticker, weight))

        cleaned_rows = []
        for row in row_data:
            ticker = str(row.get("Ticker", "")).strip().upper()
            try:
                weight = float(row.get("Weight", 0) or 0)
            except Exception:
                weight = 0.0
            if (ticker, weight) not in selected_keys:
                cleaned_rows.append(row)

        return cleaned_rows

    return row_data


@callback(
    [Output("custom-benchmark-store", "data"), Output("custom-benchmark-error", "children")],
    Input("btn-custom-apply", "n_clicks"),
    [State("custom-benchmark-grid", "rowData"), State("custom-benchmark-name", "value")]
)
def apply_custom_benchmark(n_clicks, row_data, name):
    if not n_clicks:
        return dash.no_update, ""

    row_data = row_data or []
    cleaned = {}
    total = 0.0

    for row in row_data:
        ticker = str(row.get("Ticker", "")).strip().upper()
        if not ticker:
            continue
        try:
            weight = float(row.get("Weight", 0))
        except Exception:
            continue
        if weight <= 0:
            continue
        cleaned[ticker] = weight
        total += weight

    if not cleaned:
        return None, "Enter at least one valid ticker and weight."

    if abs(total - 100.0) > 0.5:
        return None, f"Weights must sum to 100%. Current total: {total:.2f}%"

    payload = {
        "name": name or "Custom Benchmark",
        "weights": cleaned
    }
    return payload, "Custom benchmark applied."


@callback(
    [Output("strategy-growth-chart", "figure"),
     Output("strategy-drawdown-chart", "figure"),
     Output("strategy-risk-chart", "figure"),
     Output("strategy-scorecard", "rowData"),
     Output("strategy-weights-table", "rowData"),
    Output("strategy-proxy-log", "rowData"),
     Output("strategy-backtest-badge", "children"),
     Output("strategy-error", "children")],
    [Input("data-signal", "data"),
     Input("strategy-lookback", "value"),
     Input("strategy-initial", "value"),
     Input("strategy-preset-checklist", "value"),
     Input("custom-benchmark-store", "data")]
)
def update_strategy_backtesting(signal, lookback, initial_value, presets, custom_benchmark):
    data = dw.get_data()
    if not data:
        return {}, {}, {}, [], [], [], "", "No data available."

    result = dw.get_strategy_backtest_results(
        data,
        lookback=lookback or "MAX",
        initial_value=initial_value or 10000.0,
        selected_presets=presets,
        custom_benchmark=custom_benchmark
    )

    if result.get("error"):
        return {}, {}, {}, [], [], [], "", result["error"]

    growth_fig = dw.get_strategy_backtest_growth_chart(result, initial_value or 10000.0)
    drawdown_fig = dw.get_strategy_backtest_drawdown_chart(result)
    risk_fig = dw.get_strategy_backtest_risk_return_chart(result)

    scorecard = result.get("scorecard", pd.DataFrame())
    score_rows = scorecard.to_dict("records") if not scorecard.empty else []

    weights_table = result.get("weights_table", pd.DataFrame())
    weights_rows = weights_table.to_dict("records") if not weights_table.empty else []

    proxy_log_df = result.get("proxy_log", pd.DataFrame())
    proxy_log_rows = proxy_log_df.to_dict("records") if not proxy_log_df.empty else []

    start_date = result.get("start_date")
    end_date = result.get("end_date")
    window_text = ""
    if start_date is not None and end_date is not None:
        window_text = f"{start_date.date()} → {end_date.date()} (shortest-history clipped)"

    return growth_fig, drawdown_fig, risk_fig, score_rows, weights_rows, proxy_log_rows, window_text, ""
