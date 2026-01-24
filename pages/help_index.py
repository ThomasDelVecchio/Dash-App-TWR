import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from pages.help_content import HELP_TOPICS

# ============================================================
# LAYOUT
# ============================================================

layout = dbc.Container([
    
    # Header
    dbc.Row([
        dbc.Col([
            html.H2([
                html.I(className="bi bi-question-circle page-title-icon me-2"),
                "Help Index & Documentation"
            ], className="fw-bold text-body"),
            html.P("Technical reference for calculations, methodologies, and configuration.", className="subtitle")
        ], width=12)
    ], className="page-header mb-4"),

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
        ], className="mb-4 shadow-sm", id=key) 
        for key, topic in HELP_TOPICS.items()
    ])

], fluid=True, className="py-4")
