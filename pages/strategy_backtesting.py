import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import dash_wrappers as dw
from components.page_header import page_header
from config import BENCHMARK_PRESETS
import pandas as pd


def _preset_options():
    return [{"label": p["name"], "value": p["name"]} for p in BENCHMARK_PRESETS]


layout = html.Div([
    page_header(
        title="Strategy Arena",
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
                        value=[p["name"] for p in BENCHMARK_PRESETS],
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
                    rowData=[
                        {"Ticker": "QQQ", "Weight": 50},
                        {"Ticker": "GLD", "Weight": 50},
                    ],
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
                        dbc.Button("Apply Custom Benchmark", id="btn-custom-apply", color="primary", className="me-2"),
                        dbc.Button("Remove Custom Benchmark", id="btn-custom-clear", color="secondary")
                    ], width=12)
                ], className="mt-2")
            ])
        ], className="mb-4")
    ], id="custom-benchmark-collapse", is_open=False),

    dcc.Store(id="custom-benchmark-store"),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Portfolios & Weights", className="card-title section-header"),
                dcc.Loading(dag.AgGrid(
                    id="strategy-weights-table",
                    columnDefs=[
                        {"headerName": "Strategy", "field": "Strategy", "minWidth": 220, "flex": 2},
                        {"headerName": "Ticker", "field": "Ticker", "minWidth": 120, "flex": 1},
                        {"headerName": "Weight %", "field": "Weight", "type": "numericColumn", "minWidth": 120, "flex": 1,
                         "valueFormatter": {"function": "d3.format('.2f')(params.value) + '%'"}
                        },
                    ],
                    rowData=[],
                    defaultColDef={"resizable": True, "sortable": True, "filter": True},
                    dashGridOptions={
                        "domLayout": "normal",
                        "animateRows": True,
                        "suppressAggFuncInHeader": True,
                        "rowHeight": 32,
                        "headerHeight": 34,
                        "suppressHorizontalScroll": True
                    },
                    className="ag-theme-alpine-dark",
                    style={"width": "100%", "height": "360px"}
                ))
            ])
        ]), width=12, className="mb-4"),
    ]),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Growth of $ Investment", className="card-title section-header"),
                dcc.Loading(dcc.Graph(id="strategy-growth-chart"))
            ])
        ]), width=12, className="mb-4"),
    ]),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Underwater Chart (Drawdown)", className="card-title section-header"),
                dcc.Loading(dcc.Graph(id="strategy-drawdown-chart"))
            ])
        ]), width=12, className="mb-4"),
    ]),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Risk vs Return", className="card-title section-header"),
                dcc.Loading(dcc.Graph(id="strategy-risk-chart"))
            ])
        ]), width=12, className="mb-4"),
    ]),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Strategy Scorecard", className="card-title section-header"),
                dcc.Loading(dag.AgGrid(
                    id="strategy-scorecard",
                    columnDefs=[
                        {"headerName": "Strategy", "field": "Strategy", "pinned": "left", "minWidth": 200, "lockPinned": True, "cellClass": "lock-pinned"},
                        {"headerName": "CAGR", "field": "CAGR", "type": "numericColumn", "valueFormatter": {"function": "d3.format('.2%')(params.value)"}},
                        {"headerName": "Volatility", "field": "Volatility", "type": "numericColumn", "valueFormatter": {"function": "d3.format('.2%')(params.value)"}},
                        {"headerName": "Sharpe", "field": "Sharpe", "type": "numericColumn", "valueFormatter": {"function": "d3.format('.2f')(params.value)"}},
                        {"headerName": "Sortino", "field": "Sortino", "type": "numericColumn", "valueFormatter": {"function": "d3.format('.2f')(params.value)"}},
                        {"headerName": "Max Drawdown", "field": "Max Drawdown", "type": "numericColumn", "valueFormatter": {"function": "d3.format('.2%')(params.value)"}},
                    ],
                    rowData=[],
                    defaultColDef={"resizable": True, "sortable": True, "filter": True},
                    dashGridOptions={"domLayout": "autoHeight"},
                    className="ag-theme-alpine-dark"
                ))
            ])
        ]), width=12, className="mb-4"),
    ]),

    html.Div(id="strategy-error", className="text-warning small")
])


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
    [Input("btn-custom-apply", "n_clicks"), Input("btn-custom-clear", "n_clicks")],
    [State("custom-benchmark-grid", "rowData"), State("custom-benchmark-name", "value")]
)
def apply_custom_benchmark(apply_clicks, clear_clicks, row_data, name):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, ""

    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger == "btn-custom-clear":
        return None, "Custom benchmark removed."

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
    [Output("strategy-weights-table", "rowData"),
     Output("strategy-growth-chart", "figure"),
     Output("strategy-drawdown-chart", "figure"),
     Output("strategy-risk-chart", "figure"),
     Output("strategy-scorecard", "rowData"),
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
        return [], {}, {}, {}, [], "", "No data available."

    result = dw.get_strategy_backtest_results(
        data,
        lookback=lookback or "MAX",
        initial_value=initial_value or 10000.0,
        selected_presets=presets,
        custom_benchmark=custom_benchmark
    )

    if result.get("error"):
        return [], {}, {}, {}, [], "", result["error"]

    strategies = result.get("strategies", [])
    weights_rows = []
    for strat in strategies:
        name = strat.get("name", "")
        weights = strat.get("weights", {})
        for ticker, weight in weights.items():
            weights_rows.append({
                "Strategy": name,
                "Ticker": ticker,
                "Weight": float(weight) * 100.0
            })

    weights_rows = sorted(weights_rows, key=lambda r: (r["Strategy"], -r["Weight"]))

    growth_fig = dw.get_strategy_backtest_growth_chart(result, initial_value or 10000.0)
    drawdown_fig = dw.get_strategy_backtest_drawdown_chart(result)
    risk_fig = dw.get_strategy_backtest_risk_return_chart(result)

    scorecard = result.get("scorecard", pd.DataFrame())
    score_rows = scorecard.to_dict("records") if not scorecard.empty else []

    start_date = result.get("start_date")
    end_date = result.get("end_date")
    window_text = ""
    if start_date is not None and end_date is not None:
        window_text = f"{start_date.date()} → {end_date.date()} (shortest-history clipped)"

    return weights_rows, growth_fig, drawdown_fig, risk_fig, score_rows, window_text, ""
