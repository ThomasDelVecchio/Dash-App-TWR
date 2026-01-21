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


def create_price_source_badge(price_source_metadata, badge_id="price-source-badge"):
    """
    Creates a badge indicating the price data source status.
    
    price_source_metadata: dict from prices.attrs['source_metadata']:
        {
            'FMP': [list of tickers],
            'yfinance': [list of tickers],
            'mixed': [list of tickers with data from both],
            'fmp_range': (start, end) or (None, None),
            'yf_range': (start, end) or (None, None),
            'stitch_date': pd.Timestamp
        }
    """
    if not price_source_metadata:
        return html.Div()
    
    fmp_tickers = price_source_metadata.get("FMP", [])
    yf_tickers = price_source_metadata.get("yfinance", [])
    mixed_tickers = price_source_metadata.get("mixed", [])
    fmp_range = price_source_metadata.get("fmp_range", (None, None))
    yf_range = price_source_metadata.get("yf_range", (None, None))
    
    total_fmp = len(fmp_tickers) + len(mixed_tickers)
    total_yf = len(yf_tickers) + len(mixed_tickers)
    
    # Determine badge status
    if total_fmp > 0 and total_yf == 0:
        label = "FMP Only"
        color = "success"
        header = "All price data from Financial Modeling Prep (FMP)."
    elif total_fmp > 0 and total_yf > 0:
        label = "Hybrid"
        color = "info"
        header = "Price data from FMP (recent) + Yahoo Finance (older)."
    elif total_yf > 0:
        label = "Yahoo Finance"
        color = "warning"
        header = "Price data from Yahoo Finance only. FMP unavailable or disabled."
    else:
        label = "No Data"
        color = "danger"
        header = "No price data available."
    
    # Build tooltip content
    tooltip_elements = [html.P(header, className="mb-2 fw-bold")]
    
    if total_fmp > 0:
        fmp_start, fmp_end = fmp_range
        range_str = ""
        if fmp_start and fmp_end:
            range_str = f" ({fmp_start.strftime('%Y-%m-%d')} to {fmp_end.strftime('%Y-%m-%d')})"
        tooltip_elements.append(html.P(f"• FMP: {total_fmp} tickers{range_str}", className="mb-0 small"))
    
    if total_yf > 0:
        yf_start, yf_end = yf_range
        range_str = ""
        if yf_start and yf_end:
            range_str = f" ({yf_start.strftime('%Y-%m-%d')} to {yf_end.strftime('%Y-%m-%d')})"
        tooltip_elements.append(html.P(f"• yfinance: {total_yf} tickers{range_str}", className="mb-0 small"))
        
        # Display specific tickers for YFinance fallbacks
        if yf_tickers:
            tooltip_elements.append(html.P(
                ", ".join(sorted(yf_tickers)), 
                className="mb-0 small text-muted", 
                style={"fontSize": "0.7rem", "marginLeft": "10px", "wordBreak": "break-word"}
            ))
    
    if mixed_tickers:
        tooltip_elements.append(html.P(f"• Stitched: {len(mixed_tickers)} tickers (FMP+YF)", className="mb-0 small"))
    
    tooltip_content = html.Div(tooltip_elements, style={"textAlign": "left", "padding": "5px", "maxWidth": "300px"})
    
    badge = dbc.Badge(
        [html.I(className="bi bi-database me-1"), label],
        color=color,
        pill=True,
        id=badge_id,
        style={"cursor": "pointer", "fontSize": "0.75rem"}
    )
    
    return html.Div([
        badge,
        dbc.Tooltip(
            tooltip_content,
            target=badge_id,
            placement="bottom"
        )
    ], style={"display": "inline-block", "marginLeft": "8px"})
