import dash
from dash import dcc, html, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import dash_wrappers as dw
from components.data_source_badge import create_data_source_badge

layout = html.Div([
    # Top Row: Asset Class Pie & Bar
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H5("Asset Allocation", className="card-title p-2"),
            dcc.Store(id='alloc-drilldown-store'),
            html.Button("← Back to Overview", id="alloc-back-btn", className="btn btn-sm btn-outline-secondary mb-2 mx-2", style={'display': 'none', 'width': 'fit-content'}),
            dcc.Graph(id={'type': 'filter-chart', 'index': 'asset-pie-chart'})
        ]), width=6),
        dbc.Col(dbc.Card([
            html.H5("Allocation vs Target", className="card-title p-2"),
            dcc.Graph(id={'type': 'filter-chart', 'index': 'asset-bar-chart'})
        ]), width=6),
    ], className="mb-4"),
    
    # Asset Class Allocation Table
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H5("Asset Class Allocation", className="card-title p-2"),
            dcc.Loading(html.Div(id='asset-class-table-container'))
        ]), width=12, className="mb-4"),
    ]),
    
    # Sector Row
    dbc.Row([
        dbc.Col(dbc.Card([
            html.Div([
                html.H5("Sector Allocation (Look-through)", className="card-title p-2", style={"display": "inline-block"}),
                html.Div(id="sector-data-source-container", style={"display": "inline-block"})
            ]),
            dcc.Graph(id={'type': 'filter-chart', 'index': 'sector-chart'})
        ]), width=12, className="mb-4"),
    ]),
    
    # History Row
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H5("Allocation History", className="card-title p-2"),
            dcc.Loading(dcc.Graph(id='history-chart'))
        ]), width=12, className="mb-4"),
    ]),
])

@callback(
    [Output({'type': 'filter-chart', 'index': 'asset-bar-chart'}, 'figure'),
     Output('asset-class-table-container', 'children'),
     Output({'type': 'filter-chart', 'index': 'sector-chart'}, 'figure'),
     Output('history-chart', 'figure'),
     Output('sector-data-source-container', 'children')],
    [Input('data-signal', 'data'),
     Input('theme-store', 'data'),
     Input('chatbot-command', 'data'),
     Input('filter-store', 'data')]
)
def update_allocations(signal, theme, chat_cmd, _filters):
    data = dw.get_data()
    if not data: return {}, "Loading...", {}, {}
    
    # --- CHATBOT PARAMS ---
    chat_target = ""
    chat_action = None
    if chat_cmd:
        chat_action = chat_cmd.get("action")
        chat_target = chat_cmd.get("params", {}).get("target", "").lower()
    
    _, bar = dw.get_asset_allocation_charts(data, theme)
    
    # Asset Class Allocation Table
    asset_class_df = dw.get_asset_class_allocation_table(data)
    
    if asset_class_df.empty:
        asset_class_content = html.P("No data available.", 
                                     style={'fontStyle': 'italic', 'color': 'gray', 'padding': '10px'})
    else:
        # Extract TOTAL row for pinning
        total_mask = asset_class_df["Asset Class"] == "TOTAL"
        pinned_rows = asset_class_df[total_mask].to_dict('records')
        main_rows = asset_class_df[~total_mask].to_dict('records')
        
        # Chatbot Sort
        is_alloc_target = "asset" in chat_target or "allocation" in chat_target or not chat_target

        asset_class_column_defs = []
        for col in asset_class_df.columns:
            col_def = {"field": col, "headerName": col}
            
            # Hide Meta Columns
            if col.startswith("meta_"):
                col_def["hide"] = True
            
            if chat_action == "SORT" and is_alloc_target:
                 target_col = chat_cmd["params"].get("column", "").lower()
                 if col.lower() == target_col or target_col in col.lower():
                     col_def["sort"] = chat_cmd["params"].get("direction", "desc")

            # Add numerical comparator for value columns
            if any(k in col for k in ["Value", "%", "$"]):
                col_def["comparator"] = {"function": "MoneyComparator"}
            asset_class_column_defs.append(col_def)
            
        asset_class_table = dag.AgGrid(
            id="alloc-asset-class-grid",
            rowData=main_rows,
            columnDefs=asset_class_column_defs,
            defaultColDef={"flex": 1, "minWidth": 100, "sortable": True, "filter": True, "resizable": True},
            className="ag-theme-alpine-dark audit-target",
            dashGridOptions={
                "domLayout": "autoHeight",
                "pinnedBottomRowData": pinned_rows,
                "getRowStyle": {
                    "function": "params.data['Asset Class'] === 'TOTAL' ? {'fontWeight': 'bold', 'backgroundColor': 'rgba(255,255,255,0.05)', 'borderTop': '2px solid #888'} : {}"
                }
            }
        )
        
        asset_class_content = html.Div([
            asset_class_table,
            html.P("Delta % = actual allocation minus target allocation", style={
                'fontSize': '9pt',
                'fontStyle': 'italic',
                'color': 'gray',
                'marginTop': '10px',
                'marginBottom': '0px'
            })
        ])
    
    sector = dw.get_sector_allocation_chart(data, theme)
    hist = dw.get_allocation_history_chart(data, theme)
    
    # Data Source Badge
    source_summary = dw.get_data_source_summary(data)
    source_badge = create_data_source_badge(source_summary)
    
    return bar, asset_class_content, sector, hist, source_badge

# New Callbacks for Drilldown
@callback(
    [Output('alloc-drilldown-store', 'data'),
     Output('alloc-back-btn', 'style')],
    [Input({'type': 'filter-chart', 'index': 'asset-pie-chart'}, 'clickData'),
     Input('alloc-back-btn', 'n_clicks')],
    [State('alloc-drilldown-store', 'data')]
)
def handle_drilldown_interaction(clickData, back_clicks, current_state):
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update
    
    trigger_id = ctx.triggered[0]['prop_id']
    
    # Check if "Back" was clicked
    if 'alloc-back-btn' in trigger_id:
        return None, {'display': 'none', 'width': 'fit-content'}
        
    # Check if Pie Slice was clicked
    if clickData and not current_state:
        # Get label (Asset Class Short Name)
        label = clickData['points'][0]['label']
        return label, {'display': 'block', 'width': 'fit-content'}
    
    return no_update, no_update

@callback(
    Output({'type': 'filter-chart', 'index': 'asset-pie-chart'}, 'figure'),
    [Input('data-signal', 'data'),
     Input('alloc-drilldown-store', 'data'),
     Input('theme-store', 'data')]
)
def update_allocations_pie(signal, drilldown_target, theme):
    data = dw.get_data()
    if not data: return {}
    
    if drilldown_target:
        # Drilldown View
        return dw.get_asset_drilldown_chart(data, drilldown_target, theme)
    else:
        # Overview View
        pie, _ = dw.get_asset_allocation_charts(data, theme)
        return pie
