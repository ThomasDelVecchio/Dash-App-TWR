import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import dash_wrappers as dw
import base64
from datetime import datetime
from components.page_header import page_header
from config import NAV_MODULES

# Prepare options for the checklist
MODULE_OPTIONS = [
    {"label": m["label"], "value": m["id"]}
    for m in NAV_MODULES
    if m["can_toggle"]
]

layout = html.Div([
    # --- HEADER ---
    page_header(
        title="Settings",
        icon="bi-gear",
        subtitle="Manage configuration, modules, and integrations"
    ),
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H5("Data Management", className="card-title section-header p-2"),
            html.Div([
                html.P("Upload your latest data files to update the dashboard."),
                
                html.Label("Holdings CSV"),
                dcc.Upload(
                    id='upload-holdings',
                    children=html.Div(['Drag and Drop or ', html.A('Select File')]),
                    style={
                        'width': '100%', 'height': '60px', 'lineHeight': '60px',
                        'borderWidth': '1px', 'borderStyle': 'dashed',
                        'borderRadius': '5px', 'textAlign': 'center', 'marginBottom': '20px'
                    },
                    multiple=False
                ),
                
                html.Label("Cashflows CSV"),
                dcc.Upload(
                    id='upload-cashflows',
                    children=html.Div(['Drag and Drop or ', html.A('Select File')]),
                    style={
                        'width': '100%', 'height': '60px', 'lineHeight': '60px',
                        'borderWidth': '1px', 'borderStyle': 'dashed',
                        'borderRadius': '5px', 'textAlign': 'center', 'marginBottom': '20px'
                    },
                    multiple=False
                ),
                
                html.Div(id='upload-status', className="text-muted")
            ], className="p-3")
        ]), width=6),
        
        dbc.Col(dbc.Card([
            html.H5("Active Modules", className="card-title section-header p-2"),
            html.Div([
                html.P("Toggle features to customize your sidebar."),
                dbc.Checklist(
                    id="modules-checklist",
                    options=MODULE_OPTIONS,
                    value=[], # Populated by callback
                    switch=True,
                    persistence=False # We rely on the global store
                )
            ], className="p-3")
        ]), width=6),
    ]),
])

# 1. Sync Store -> Checklist (Load)
@callback(
    Output("modules-checklist", "value"),
    Input("active-modules-store", "data")
)
def load_modules_settings(store_data):
    if store_data is None:
        # Default to all if None
        return [m["value"] for m in MODULE_OPTIONS]
    return store_data

# 2. Sync Checklist -> Store (Save)
@callback(
    Output("active-modules-store", "data", allow_duplicate=True),
    Input("modules-checklist", "value"),
    State("active-modules-store", "data"),
    prevent_initial_call=True
)
def save_modules_settings(selected_modules, current_store):
    if selected_modules is None:
        return dash.no_update
        
    # Check for equality to break loop
    # Treat None as empty set for comparison, though usually None means "default to all" in our app logic
    # But here we are comparing what IS stored vs what IS selected.
    
    # If store is None, it means "All". 
    # If selected_modules is "All", and store is None, should we update store to "All"?
    # Yes, to make it explicit.
    
    current_set = set(current_store) if current_store else set()
    new_set = set(selected_modules)
    
    # Special case: if current_store is None (implicit ALL) and selected_modules is ALL, 
    # we might want to avoid writing if we want to keep it None. 
    # But writing the explicit list is safer for consistency.
    # However, if we write, we trigger the load callback again.
    
    # Let's just compare sets.
    # But we need to handle the "None = All" logic of the store side.
    if current_store is None:
        # If store is None, it effectively contains ALL options.
        current_set = set([m["value"] for m in MODULE_OPTIONS])
        
    if current_set == new_set:
        return dash.no_update
        
    return selected_modules

@callback(
    [Output('data-signal', 'data', allow_duplicate=True),
     Output('upload-status', 'children')],
    [Input('upload-holdings', 'contents'),
     Input('upload-cashflows', 'contents')],
    [State('upload-holdings', 'filename'),
     State('upload-cashflows', 'filename'),
    State('date-picker-end', 'date')],
    prevent_initial_call=True
)
def update_data_files(h_content, c_content, h_name, c_name, end_date):
    # This callback updates the signal store in app.py
    # Logic similar to original app.py
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, ""
        
    msg = []
    
    if h_content:
        content_type, content_string = h_content.split(',')
        decoded = base64.b64decode(content_string)
        with open("sample holdings.csv", "wb") as f:
            f.write(decoded)
        msg.append(f"Updated {h_name}")
        
    if c_content:
        content_type, content_string = c_content.split(',')
        decoded = base64.b64decode(content_string)
        with open("cashflows.csv", "wb") as f:
            f.write(decoded)
        msg.append(f"Updated {c_name}")
        
    if msg:
        try:
            dw.refresh_data(end_date=end_date)
            return datetime.now().isoformat(), " | ".join(msg) + " - Engine Re-run Complete"
        except Exception as e:
            return dash.no_update, f"Error: {str(e)}"
            
    return dash.no_update, ""
