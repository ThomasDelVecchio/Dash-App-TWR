import dash_bootstrap_components as dbc
from dash import html

def create_data_source_badge(source_summary):
    """
    Creates a badge indicating the data source status.
    source_summary: {
        'all_fmp': bool,
        'sources': { 'FMP': count, 'YF': count, ... },
        'fallbacks': list of (ticker, source),
        'has_errors': bool
    }
    """
    if not source_summary:
        return html.Div()

    sources = source_summary.get('sources', {})
    has_fallbacks = any(s != 'FMP' for s in sources.keys())
    has_errors = source_summary.get('has_errors', False)

    if not has_fallbacks and not has_errors:
        label = "FMP Verified"
        color = "success"
        header = "All data successfully sourced from Financial Modeling Prep (FMP)."
    elif sources.get('FMP', 0) > 0:
        label = "Mixed Sources"
        color = "info"
        header = "Data sourced from both FMP and fallbacks (Yahoo/Equity)."
    else:
        label = "Yahoo Finance"
        color = "warning"
        header = "FMP unavailable. Using Yahoo Finance/Equity fallbacks."

    if has_errors:
        label = "Data Gaps"
        color = "danger"
        header = "Significant data gaps or errors detected."

    # Build detailed tooltip content
    summary_lines = []
    for s, count in sources.items():
        summary_lines.append(f"• {s}: {count} tickers")
    
    tooltip_content = html.Div([
        html.P(header, className="mb-2 fw-bold"),
        html.Div([html.P(line, className="mb-0") for line in summary_lines]),
        html.Hr(className="my-2") if source_summary.get('fallbacks') else None,
        html.P("Fallbacks:", className="mb-1 small") if source_summary.get('fallbacks') else None,
        html.P(", ".join([f"{t}({s})" for t, s in source_summary.get('fallbacks', [])]), className="small mb-0")
    ], style={"textAlign": "left", "padding": "5px"})

    badge = dbc.Badge(
        label,
        color=color,
        pill=True,
        id="data-source-badge",
        style={"cursor": "pointer", "fontSize": "0.8rem"}
    )

    return html.Div([
        badge,
        dbc.Tooltip(
            tooltip_content,
            target="data-source-badge",
            placement="bottom",
            className="source-tooltip"
        )
    ], style={"display": "inline-block", "marginLeft": "10px"})
