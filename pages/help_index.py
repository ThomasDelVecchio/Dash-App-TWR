import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from components.page_header import page_header
from pages.help_content import HELP_TOPICS

# ============================================================
# LAYOUT
# ============================================================

layout = dbc.Container([
    
    # Header
    page_header(
        title="Help Index & Documentation",
        icon="bi-question-circle",
        subtitle="Technical reference for calculations, methodologies, and configuration."
    ),

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

], fluid=True, className="help-page py-4")
