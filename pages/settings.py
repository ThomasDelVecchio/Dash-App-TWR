import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import dash_wrappers as dw
import base64
from datetime import datetime

layout = html.Div([
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H5("Data Management", className="card-title p-2"),
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
    ]),
])

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
