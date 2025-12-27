import dash
from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc
import dash_wrappers as dw
import plotly.graph_objects as go
import dash_ag_grid as dag
from report_formatting import fmt_dollar_clean
from config import GLOBAL_PALETTE

layout = html.Div([
    # Active Strategy Section
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H5("Active Strategy vs Benchmarks", className="card-title p-2"),
            dcc.Loading(html.Div(id='active-strategy-table-container'))
        ]), width=12, className="mb-4"),
    ]),

    dbc.Row([
        dbc.Col(dbc.Card([
            html.H5("Attribution Since Inception", className="card-title p-2"),
            html.Div("Click on a bar to see the asset class breakdown for that period.", className="text-muted small px-2 mb-2"),
            dcc.Graph(id='attribution-chart')
        ]), width=12, className="mb-4"),
    ]),
    
    # Drill-down Section
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H5(id="attribution-detail-title", children="Breakdown", className="card-title p-2"),
            dcc.Loading(html.Div(id="attribution-detail-container"))
        ]), width=12, className="mb-4"),
    ]),

    # SI Attribution Summary Section
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H5("Lifetime SI Attribution Summary", className="card-title p-2"),
            dcc.Loading(html.Div(id='si-attribution-container'))
        ]), width=12, className="mb-4"),
    ])
])

# 1. Main Chart Callback
@callback(
    Output('attribution-chart', 'figure'),
    [Input('data-signal', 'data'),
     Input('theme-store', 'data')]
)
def update_attribution_main(signal, theme):
    data = dw.get_data()
    if not data: return {}
    return dw.get_smart_attribution_chart(data, theme=theme)

# 2. Drill-down Callback
@callback(
    [Output('attribution-detail-container', 'children'),
     Output('attribution-detail-title', 'children')],
    [Input('attribution-chart', 'clickData'),
     Input('attribution-chart', 'figure'),
     Input('theme-store', 'data')]
)
def update_attribution_detail(click_data, figure, theme):
    data = dw.get_data()
    if not data: return "", "Breakdown"
    
    if not click_data:
        return html.Div("Click a bar above to view details.", className="text-muted p-3"), "Breakdown"

    period_type = figure.get('layout', {}).get('meta', {}).get('period_type', 'daily')
    
    point = click_data['points'][0]
    date_str = point['x']
    
    if period_type == 'monthly':
        breakdown_df = dw.get_monthly_attribution_breakdown(data, date_str)
        title = f"Monthly Breakdown for {date_str}"
    elif period_type == 'weekly':
        breakdown_df = dw.get_weekly_attribution_breakdown(data, date_str)
        title = f"Weekly Breakdown for {date_str}"
    else:
        breakdown_df = dw.get_daily_attribution_breakdown(data, date_str)
        title = f"Daily Breakdown for {date_str}"

    
    if not breakdown_df.empty:
        # Create Waterfall Chart (Contribution %)
        total_contrib = breakdown_df["Contribution (%)"].sum()
        total_effect = breakdown_df["Effect"].sum()
        
        fig = go.Figure(go.Waterfall(
            name="Contribution",
            orientation="v",
            measure=["relative"] * len(breakdown_df) + ["total"],
            x=breakdown_df["Asset Class"].tolist() + ["Total"],
            y=breakdown_df["Contribution (%)"].tolist() + [0],
            text=[f"{x:+.2f}%" for x in breakdown_df["Contribution (%)"]] + [f"{total_contrib:+.2f}%"],
            textposition="auto",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": GLOBAL_PALETTE[2]}}, # Red
            increasing={"marker": {"color": GLOBAL_PALETTE[4]}}, # Green
            totals={"marker": {"color": GLOBAL_PALETTE[0]}},    # Blue
            customdata=breakdown_df["Effect"].tolist() + [total_effect],
            hovertemplate="<b>%{x}</b><br>Contribution: %{text}<br>Effect: %{customdata:$,.2f}<extra></extra>"
        ))
        # Fix clipping: Add vertical buffer and allow labels to bleed past axis
        fig.update_traces(textfont_size=12, cliponaxis=False)
        
        max_y = max(breakdown_df["Contribution (%)"].max(), total_contrib)
        min_y = breakdown_df["Contribution (%)"].min()
        y_range = [min_y * 1.2 if min_y < 0 else -0.05, max_y * 1.4] 

        fig.update_layout(
            title=f"Contribution to Return",
            yaxis_title="Contribution (%)",
            template="plotly_white" if theme == "light" else "plotly_dark",
            margin=dict(l=40, r=20, t=80, b=40),
            height=400,
            showlegend=False,
            yaxis=dict(range=y_range)
        )
        
        # Also a Grid
        breakdown_df["Effect Fmt"] = breakdown_df["Effect"].apply(fmt_dollar_clean)
        # Format Contribution % for Grid
        breakdown_df["Contrib Fmt"] = breakdown_df["Contribution (%)"].map(lambda x: f"{x:+.2f}%")
        
        grid = dag.AgGrid(
            id="attribution-drill-grid",
            rowData=breakdown_df.to_dict("records"),
            columnDefs=[
                {"field": "Asset Class"},
                {"field": "Effect Fmt", "headerName": "Effect ($)", "type": "rightAligned", "sort": "desc", "comparator": {"function": "MoneyComparator"}},
                {"field": "Contrib Fmt", "headerName": "Contribution (%)", "type": "rightAligned", "comparator": {"function": "MoneyComparator"},
                 "cellStyle": {"styleConditions": [
                    {"condition": "params.value.includes('+')", "style": {"color": "#9BBB59"}},
                    {"condition": "params.value.includes('-')", "style": {"color": "#C0504D"}}
                 ]}}
            ],
            defaultColDef={"flex": 1, "minWidth": 100, "sortable": True, "resizable": True},
            className=("ag-theme-alpine-dark" if theme == "dark" else "ag-theme-alpine") + " audit-target",
            dashGridOptions={"domLayout": "autoHeight"}
        )
        
        detail_content = dbc.Row([
            dbc.Col(dcc.Graph(figure=fig, responsive=True), width=12, lg=7),
            dbc.Col(grid, width=12, lg=5)
        ])
    else:
        detail_content = html.Div(f"No attribution data available for {date_str}.", className="text-warning p-3")
        
    if period_type == 'daily':
        footnote_text = "* This breakdown uses a simplified daily contribution calculation and is not geometrically linked. For multi-day attribution, see the Since Inception summary."
    else:
        footnote_text = f"* This {period_type} breakdown is calculated using the Frongello linking method to ensure geometric accuracy over the selected period."

    footnote = html.P(
        footnote_text,
        className="footnote small text-muted px-3"
    )

    return [detail_content, footnote], title

# 3. SI Attribution Callback
@callback(
    Output('si-attribution-container', 'children'),
    [Input('data-signal', 'data'),
     Input('theme-store', 'data')]
)
def update_si_attribution(signal, theme):
    data = dw.get_data()
    if not data:
        return {}

    df = dw.get_si_attribution_summary(data)

    if df.empty:
        return {}

    # Waterfall Chart
    total_contrib = df["Contribution (%)"].sum()
    total_effect = df["Effect"].sum()
    
    fig = go.Figure(go.Waterfall(
        name="Contribution",
        orientation="v",
        measure=["relative"] * len(df) + ["total"],
        x=df["Asset Class"].tolist() + ["Total"],
        y=df["Contribution (%)"].tolist() + [0],
        text=[f"{x:+.2f}%" for x in df["Contribution (%)"]] + [f"{total_contrib:+.2f}%"],
        textposition="auto",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": GLOBAL_PALETTE[2]}},
        increasing={"marker": {"color": GLOBAL_PALETTE[4]}},
        totals={"marker": {"color": GLOBAL_PALETTE[0]}},
        customdata=df["Effect"].tolist() + [total_effect],
        hovertemplate="<b>%{x}</b><br>Contribution: %{text}<br>Effect: %{customdata:$,.2f}<extra></extra>"
    ))
    # Fix clipping: Add vertical buffer and allow labels to bleed past axis
    fig.update_traces(textfont_size=12, cliponaxis=False)
    
    max_y = max(df["Contribution (%)"].max(), total_contrib)
    min_y = df["Contribution (%)"].min()
    y_range = [min_y * 1.2 if min_y < 0 else -0.5, max_y * 1.4]

    fig.update_layout(
        title="Lifetime Contribution to Return",
        yaxis_title="Contribution (%)",
        template="plotly_white" if theme == "light" else "plotly_dark",
        margin=dict(l=40, r=20, t=80, b=40),
        height=400,
        showlegend=False,
        yaxis=dict(range=y_range)
    )

    # Grid Data
    df["Effect Fmt"] = df["Effect"].apply(fmt_dollar_clean)
    df["Contrib Fmt"] = df["Contribution (%)"].map(lambda x: f"{x:+.2f}%")
    
    grid = dag.AgGrid(
        id="attribution-si-grid",
        rowData=df.to_dict("records"),
        columnDefs=[
            {"field": "Asset Class"},
            {"field": "Effect Fmt", "headerName": "Effect ($)", "type": "rightAligned", "sort": "desc", "comparator": {"function": "MoneyComparator"}},
            {"field": "Contrib Fmt", "headerName": "Contribution (%)", "type": "rightAligned", "comparator": {"function": "MoneyComparator"},
             "cellStyle": {"styleConditions": [
                {"condition": "params.value.includes('+')", "style": {"color": "#9BBB59"}},
                {"condition": "params.value.includes('-')", "style": {"color": "#C0504D"}}
             ]}}
        ],
        defaultColDef={"flex": 1, "minWidth": 100, "sortable": True, "resizable": True},
        className=("ag-theme-alpine-dark" if theme == "dark" else "ag-theme-alpine") + " audit-target",
        dashGridOptions={"domLayout": "autoHeight"}
    )

    footnote = html.P(
        "* Since Inception attribution is calculated using the Frongello linking method to ensure geometric accuracy over time.",
        className="footnote small text-muted px-3"
    )
    
    return dbc.Row([
        dbc.Col(dcc.Graph(figure=fig, responsive=True), width=12, lg=7),
        dbc.Col(grid, width=12, lg=5),
        dbc.Col(footnote, width=12)
    ])

@callback(
    Output('active-strategy-table-container', 'children'),
    [Input('data-signal', 'data'),
     Input('theme-store', 'data')]
)
def update_active_strategy_table(signal, theme):
    data = dw.get_data()
    if not data: return html.Div("Loading...", className="p-3")
    
    df = dw.get_active_strategy_table(data)
    
    if df.empty:
        return html.Div("Insufficient data for active strategy metrics.", className="text-warning p-3")
        
    grid = dag.AgGrid(
        id="active-strategy-grid",
        rowData=df.to_dict("records"),
        columnDefs=[
            {"field": "Benchmark", "headerName": "Benchmark"},
            {"field": "Beta", "headerName": "Beta (Sensitivity)", "type": "rightAligned"},
            {"field": "Tracking Error", "headerName": "Tracking Error (Active Risk)", "type": "rightAligned"}
        ],
        defaultColDef={"flex": 1, "minWidth": 150, "resizable": True},
        className=("ag-theme-alpine-dark" if theme == "dark" else "ag-theme-alpine"),
        dashGridOptions={"domLayout": "autoHeight"}
    )
    
    footnote = html.Div(
        [
            html.Span("Beta: ", className="fw-bold"), "Measures volatility relative to the benchmark (1.0 = same volatility). ",
            html.Br(),
            html.Span("Tracking Error: ", className="fw-bold"), "Standard deviation of excess returns. Higher values indicate more active deviation from the benchmark."
        ],
        className="text-muted small mt-2 px-2"
    )
    
    return html.Div([grid, footnote])
