import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from pages.help_content import HELP_TOPICS

# ============================================================
# LAYOUT
# ============================================================

layout = dbc.Container([
    
    # Header
    html.Div([
        html.H1("Help Index & Documentation", className="display-5 text-body"),
        html.P("Technical reference for calculations, methodologies, and configuration.", className="lead text-muted"),
        html.Hr()
    ], className="mb-4"),

    # Dynamic Content Generation
    html.Div([
        dbc.Card([
            dbc.CardHeader(html.H4(topic["title"], className="mb-0")),
            dbc.CardBody([
                dcc.Markdown(
                    topic["content"],
                    mathjax=True,
                    className="text-body"  # Inherits theme color (light/dark)
                )
            ])
        ], className="mb-4 shadow-sm") 
        for key, topic in HELP_TOPICS.items()
    ])

], fluid=True, className="py-4")
