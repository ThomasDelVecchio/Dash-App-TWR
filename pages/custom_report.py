"""
Custom Report Builder Page
==========================
Institutional-quality report generator with Print Preview (Light Mode) toggle.
Allows configuring sections and generating print-ready PDF reports.

Key Features:
- "Print Preview" toggle switches charts to plotly_white template
- A4-proportion report container for consistent PDF output
- Ghost footer with disclaimers (hidden on screen, shown in print)
- Page break classes for multi-page PDF generation
"""

import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import dash_wrappers as dw
from report_formatting import fmt_pct_clean, fmt_dollar_clean, generate_word_report
from datetime import datetime
import pandas as pd
from tax_engine import build_tax_lots
from io import BytesIO


# ============================================================
# HELPER: Apply Light Theme to Figure
# ============================================================
def apply_print_theme(fig):
    """
    Deprecated. Returns figure as-is.
    """
    return fig


# ============================================================
# LAYOUT
# ============================================================
layout = html.Div(className="custom-report-page", children=[
    # Page Header
    html.Div([
        html.H2("Custom Report Builder", className="fw-bold text-body mb-2"),
        html.P("Configure and generate institutional-quality PDF reports", className="text-muted small mb-3")
    ]),
    
    dbc.Row([
        # Left Sidebar: Report Configuration
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="bi bi-gear-fill me-2"),
                    "Report Configuration"
                ]),
                dbc.CardBody([
                    # Report Title
                    dbc.Label("Report Title", className="fw-bold"),
                    dbc.Input(
                        id="report-title-input",
                        type="text",
                        value="Portfolio Performance Report",
                        placeholder="Enter report title...",
                        className="mb-3"
                    ),
                    
                    # Report Date Range
                    dbc.Label("Reporting Period", className="fw-bold"),
                    dbc.Select(
                        id="report-period-select",
                        options=[
                            {"label": "Since Inception", "value": "SI"},
                            {"label": "Year-to-Date", "value": "YTD"},
                            {"label": "Last 12 Months", "value": "12M"},
                            {"label": "Last Quarter", "value": "3M"},
                            {"label": "Last Month", "value": "1M"},
                        ],
                        value="SI",
                        className="mb-3 text-dark"
                    ),
                    
                    html.Hr(),
                    
                    # Section Toggles
                    dbc.Label("Report Sections", className="fw-bold mb-2"),
                    
                    html.Div([
                        dbc.Button(
                            "Select All",
                            id="toggle-all-sections",
                            color="link",
                            size="sm",
                            className="p-0 text-decoration-none",
                            style={"fontSize": "0.8rem"}
                        )
                    ], className="mb-2"),
                    
                    dbc.Checklist(
                        id="report-sections-checklist",
                        options=[
                            {"label": " Morning AI Summary", "value": "morning_brief"},
                            {"label": " Executive Summary", "value": "summary"},
                            {"label": " Performance Chart", "value": "performance_chart"},
                            {"label": " Horizon Analysis Table", "value": "horizon_table"},
                            {"label": " Asset Allocation", "value": "allocation"},
                            {"label": " Sector Breakdown", "value": "sector"},
                            {"label": " Top Holdings", "value": "holdings"},
                            {"label": " Risk Metrics", "value": "risk"},
                            {"label": " Flows Summary", "value": "flows"},
                            {"label": " Tax Lot Explorer (Open Lots)", "value": "tax_lots"},
                            {"label": " Risk Analysis Charts", "value": "risk_charts"},
                            {"label": " Performance Deep Dive", "value": "perf_deep_dive"},
                            {"label": " Attribution Analysis", "value": "attribution"},
                            {"label": " Asset Class Performance Table", "value": "ac_perf_table"},
                            {"label": " Asset Class P/L Table", "value": "ac_pl_table"},
                        ],
                        value=["summary", "performance_chart", "horizon_table", "allocation", "holdings"],
                        className="mb-3",
                        labelStyle={"display": "block", "marginBottom": "8px"},
                        inputStyle={"marginRight": "8px"}
                    ),
                    
                    html.Hr(),
                    
                    # Action Buttons
                    dbc.Button(
                        [html.I(className="bi bi-arrow-clockwise me-2"), "Refresh Preview"],
                        id="btn-refresh-report",
                        color="primary",
                        className="w-100 mb-2"
                    ),
                    dbc.Alert(
                        id="download-status-alert",
                        is_open=False,
                        dismissable=True,
                        className="mb-2 small"
                    ),
                    dbc.Button(
                        [html.I(className="bi bi-file-earmark-word me-2"), "Download Word Report"],
                        id="btn-download-word",
                        color="success",
                        className="w-100",
                        n_clicks=0
                    ),
                    dcc.Download(id="download-word-report")
                ])
            ], className="mb-3 position-sticky", style={"top": "20px"})
        ], width=3, className="report-config-sidebar d-print-none"),
        
        # Right Panel: Report Preview
        dbc.Col([
            # Report Container (A4 proportions)
            html.Div([
                # Report Header
                html.Div([
                    html.Div([
                        html.H1(id="report-header-title", className="report-title mb-1"),
                        html.P(id="report-header-subtitle", className="report-subtitle text-muted mb-0")
                    ], className="report-header-content"),
                    html.Div([
                        html.Img(
                            src="/assets/logo.png",
                            className="report-logo",
                            style={"height": "50px", "display": "none"}  # Hidden by default, enable if logo exists
                        )
                    ], className="report-logo-container")
                ], className="report-header d-flex justify-content-between align-items-center mb-4 pb-3 border-bottom"),
                
                # Report Date
                html.P(id="report-date-line", className="text-end text-muted small mb-4"),
                
                # Dynamic Report Content Container
                html.Div(id="report-content-container"),
                
                # Ghost Footer (Hidden on screen, visible in print)
                html.Div([
                    html.Hr(className="mt-4"),
                    html.P([
                        html.Strong("Important Disclosures: "),
                        "Past performance is not indicative of future results. Investment returns and principal value will fluctuate. ",
                        "This report is for informational purposes only and does not constitute investment advice. ",
                        "Data is sourced from third-party providers and may be delayed or inaccurate. ",
                        "Please consult a qualified financial advisor before making investment decisions."
                    ], className="small text-muted mb-2"),
                    html.P([
                        "Generated by DELVEX Portfolio Analytics | ",
                        html.Span(id="report-footer-timestamp")
                    ], className="small text-muted text-center")
                ], className="print-footer", id="report-ghost-footer")
                
            ], id="report-container", className="report-container")
        ], width=9, className="report-preview-col")
    ]),
    
    # Hidden Print Trigger Script
    dcc.Store(id="print-trigger-store")
])


# ============================================================
# CALLBACKS
# ============================================================

@callback(
    Output("report-sections-checklist", "value"),
    Input("toggle-all-sections", "n_clicks"),
    [State("report-sections-checklist", "value"),
     State("report-sections-checklist", "options")],
    prevent_initial_call=True
)
def toggle_all_report_sections(n_clicks, current_values, options):
    if n_clicks is None:
        return dash.no_update
    
    all_values = [opt["value"] for opt in options]
    
    # If not all are selected, select all. Otherwise, deselect all.
    if len(current_values) < len(all_values):
        return all_values
    return []


@callback(
    Output("toggle-all-sections", "children"),
    Input("report-sections-checklist", "value"),
    State("report-sections-checklist", "options")
)
def update_toggle_button_label(current_values, options):
    all_values = [opt["value"] for opt in options]
    if len(current_values) < len(all_values):
        return "Select All"
    return "Deselect All"


@callback(
    [Output("report-header-title", "children"),
     Output("report-header-subtitle", "children"),
     Output("report-date-line", "children"),
     Output("report-footer-timestamp", "children"),
     Output("report-content-container", "children"),
     Output("report-container", "className")],
    [Input("btn-refresh-report", "n_clicks"),
     Input("data-signal", "data"),
     Input("report-sections-checklist", "value"),
     Input("report-title-input", "value"),
     Input("report-period-select", "value")]
)
def update_report(n_clicks, signal, sections, title, period):
    print_preview = False # Defaults to dark mode for screen preview

    """
    Main callback to generate report content based on configuration.
    """
    data = dw.get_data()
    if not data:
        return "Loading...", "", "", "", html.Div("Loading data..."), "report-container"
    
    # Theme setup
    theme = "light" if print_preview else "dark"
    container_class = "report-container print-preview-mode" if print_preview else "report-container"
    
    # Header content
    report_title = title or "Portfolio Performance Report"
    
    period_labels = {
        "SI": "Since Inception",
        "YTD": "Year-to-Date",
        "12M": "Trailing 12 Months",
        "3M": "Last Quarter",
        "1M": "Last Month"
    }
    subtitle = f"{period_labels.get(period, 'Since Inception')} Analysis"
    
    # Date line
    now = datetime.now()
    date_line = f"Report Date: {now.strftime('%B %d, %Y')}"
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # Build report sections
    report_sections = []
    sections = sections or []
    
    # 0. Morning AI Summary
    if "morning_brief" in sections:
        # Generate summary text
        from components.ai_brief import generate_ai_summary
        summary_text = generate_ai_summary(data)
        
        brief_section = html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("Morning AI Summary", className="section-title mb-3"),
                    dcc.Markdown(summary_text, className="text-muted" if print_preview else "text-light")
                ], width=12)
            ])
        ], className="report-section no-break")
        report_sections.append(brief_section)
    
    # 1. Executive Summary
    if "summary" in sections:
        metrics = dw.get_snapshot_metrics(data)
        # Calculate MTD Return if missing
        mtd_ret = 0.0
        twr_df = data.get("twr_df", pd.DataFrame())
        if not twr_df.empty and "Horizon" in twr_df.columns:
             row = twr_df[twr_df["Horizon"] == "MTD"]
             if not row.empty:
                 mtd_ret = row["Return"].iloc[0]
                 
        summary_section = html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("Executive Summary", className="section-title mb-3"),
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div("Current Value", className="kpi-label-print"),
                        html.Div(fmt_dollar_clean(metrics.get('current_mv', 0) if metrics else 0), className="kpi-value-print")
                    ], className="kpi-box-print")
                ], width=3),
                dbc.Col([
                    html.Div([
                        html.Div("Total Return (TWR)", className="kpi-label-print"),
                        html.Div(fmt_pct_clean(metrics.get('twr_si', 0) if metrics else 0), className="kpi-value-print",
                                 style={"color": "#28a745" if (metrics.get('twr_si', 0) or 0) >= 0 else "#dc3545"})
                    ], className="kpi-box-print")
                ], width=3),
                dbc.Col([
                    html.Div([
                        html.Div("Total P/L", className="kpi-label-print"),
                        html.Div(fmt_dollar_clean(metrics.get('pl_si', 0) if metrics else 0), className="kpi-value-print",
                                 style={"color": "#28a745" if (metrics.get('pl_si', 0) or 0) >= 0 else "#dc3545"})
                    ], className="kpi-box-print")
                ], width=3),
                dbc.Col([
                    html.Div([
                        html.Div("MTD Return", className="kpi-label-print"),
                        html.Div(fmt_pct_clean(mtd_ret), className="kpi-value-print",
                                 style={"color": "#28a745" if (mtd_ret or 0) >= 0 else "#dc3545"})
                    ], className="kpi-box-print")
                ], width=3),
            ], className="mb-4"),
            
            # Risk Metrics Row
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div("Max Drawdown", className="kpi-label-print"),
                        html.Div(fmt_pct_clean(metrics.get('max_dd', 0) if metrics else 0), className="kpi-value-print",
                                 style={"color": "#dc3545"})
                    ], className="kpi-box-print")
                ], width=4),
                dbc.Col([
                    html.Div([
                        html.Div("Sharpe Ratio", className="kpi-label-print"),
                        html.Div(f"{metrics.get('sharpe', 'N/A'):.2f}" if isinstance(metrics.get('sharpe'), (int, float)) else str(metrics.get('sharpe', 'N/A')), className="kpi-value-print")
                    ], className="kpi-box-print")
                ], width=4),
                dbc.Col([
                    html.Div([
                        html.Div("Sortino Ratio", className="kpi-label-print"),
                        html.Div(f"{metrics.get('sortino', 'N/A'):.2f}" if isinstance(metrics.get('sortino'), (int, float)) else str(metrics.get('sortino', 'N/A')), className="kpi-value-print")
                    ], className="kpi-box-print")
                ], width=4),
            ], className="mb-4"),
        ], className="report-section no-break")
        report_sections.append(summary_section)
    
    # 2. Performance Chart
    if "performance_chart" in sections:
        fig = dw.get_pv_mountain_chart(data, theme)
        if print_preview:
            fig = apply_print_theme(fig)
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=30), autosize=True)
        
        perf_section = html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("Portfolio Performance", className="section-title mb-3"),
                    dcc.Graph(figure=fig, config={'displayModeBar': False}, style={'height': '500px', 'width': '100%'}, responsive=True),
                ], width=12)
            ])
        ], className="report-section page-break-before")
        report_sections.append(perf_section)
    
    # 3. Horizon Analysis Table
    if "horizon_table" in sections:
        horizon_df = dw.get_horizon_analysis(data)
        
        # Remove Sharpe/Sortino/Max DD columns if present
        for col in ['Sharpe', 'Sortino', 'Max DD']:
            if col in horizon_df.columns:
                horizon_df = horizon_df.drop(columns=[col])
        
        # Format
        if 'Return' in horizon_df.columns:
            horizon_df['Return'] = horizon_df['Return'].apply(fmt_pct_clean)
        if 'P/L' in horizon_df.columns:
            horizon_df['P/L'] = horizon_df['P/L'].apply(fmt_dollar_clean)
        
        # Filter meta columns
        display_cols = [c for c in horizon_df.columns if not c.startswith('meta_')]
        
        grid_class = "ag-theme-alpine" if print_preview else "ag-theme-alpine-dark"
        
        horizon_section = html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("Horizon Analysis", className="section-title mb-3"),
                    dag.AgGrid(
                        rowData=horizon_df[display_cols].to_dict('records'),
                        columnDefs=[{"field": col, "headerName": col, "flex": 1} for col in display_cols],
                        defaultColDef={"sortable": True, "resizable": True},
                        className=grid_class,
                        dashGridOptions={"domLayout": "autoHeight"},
                        style={"width": "100%"}
                    )
                ], width=12)
            ])
        ], className="report-section no-break")
        report_sections.append(horizon_section)
    
    # 4. Asset Allocation
    if "allocation" in sections:
        pie_fig, bar_fig = dw.get_asset_allocation_charts(data, theme)
        hist_fig = dw.get_allocation_history_chart(data, theme)
        
        if print_preview:
            pie_fig = apply_print_theme(pie_fig)
            bar_fig = apply_print_theme(bar_fig)
            hist_fig = apply_print_theme(hist_fig)
        
        pie_fig.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=20), autosize=True)
        bar_fig.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=20), autosize=True)
        hist_fig.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=40), autosize=True)
        
        alloc_section = html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("Asset Allocation", className="section-title mb-3"),
                ], width=12),
                dbc.Col([
                    dcc.Graph(figure=pie_fig, config={'displayModeBar': False}, style={'height': '450px', 'width': '100%'}, responsive=True)
                ], width=12, className="mb-4"),
                dbc.Col([
                    dcc.Graph(figure=bar_fig, config={'displayModeBar': False}, style={'height': '450px', 'width': '100%'}, responsive=True)
                ], width=12, className="mb-4"),
                dbc.Col([
                    html.H5("Allocation History", className="mt-2 mb-2"),
                    dcc.Graph(figure=hist_fig, config={'displayModeBar': False}, style={'height': '500px', 'width': '100%'}, responsive=True)
                ], width=12),
            ])
        ], className="report-section page-break-before")
        report_sections.append(alloc_section)
    
    # 5. Sector Breakdown
    if "sector" in sections:
        sector_fig = dw.get_sector_allocation_chart(data, theme)
        if print_preview:
            sector_fig = apply_print_theme(sector_fig)
        sector_fig.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=40), autosize=True)
        
        sector_section = html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("Sector Breakdown", className="section-title mb-3"),
                    dcc.Graph(figure=sector_fig, config={'displayModeBar': False}, style={'height': '500px', 'width': '100%'}, responsive=True)
                ], width=12)
            ])
        ], className="report-section no-break")
        report_sections.append(sector_section)
    
    # 6. Top Holdings
    if "holdings" in sections:
        sec_table = data.get('sec_table_current', pd.DataFrame())
        
        if not sec_table.empty:
            # Get top 10 by market_value
            holdings_df = sec_table[sec_table['ticker'] != 'CASH'].nlargest(10, 'market_value').copy()
            
            # Calculate weight
            total_value = sec_table['market_value'].sum()
            holdings_df['Weight'] = (holdings_df['market_value'] / total_value * 100).round(2).astype(str) + '%'
            holdings_df['Value'] = holdings_df['market_value'].apply(fmt_dollar_clean)
            holdings_df['Shares'] = holdings_df['shares'].round(2)
            
            display_df = holdings_df[['ticker', 'asset_class', 'Shares', 'Value', 'Weight']].copy()
            display_df.columns = ['Ticker', 'Asset Class', 'Shares', 'Value', 'Weight']
            
            grid_class = "ag-theme-alpine" if print_preview else "ag-theme-alpine-dark"
            
            holdings_section = html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H4("Top 10 Holdings", className="section-title mb-3"),
                        dag.AgGrid(
                            rowData=display_df.to_dict('records'),
                            columnDefs=[{"field": col, "headerName": col, "flex": 1} for col in display_df.columns],
                            defaultColDef={"sortable": True, "resizable": True},
                            className=grid_class,
                            dashGridOptions={"domLayout": "autoHeight"},
                            style={"width": "100%"}
                        )
                    ], width=12)
                ])
            ], className="report-section no-break")
            report_sections.append(holdings_section)
    
    # 7. Risk Metrics
    if "risk" in sections:
        risk_df = dw.get_risk_diversification(data)
        grid_class = "ag-theme-alpine" if print_preview else "ag-theme-alpine-dark"
        
        risk_section = html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("Risk & Diversification", className="section-title mb-3"),
                    dag.AgGrid(
                        rowData=risk_df.to_dict('records'),
                        columnDefs=[
                            {"field": "Metric", "headerName": "Metric", "flex": 2},
                            {"field": "Value", "headerName": "Value", "flex": 1}
                        ],
                        defaultColDef={"sortable": True, "resizable": True},
                        className=grid_class,
                        dashGridOptions={"domLayout": "autoHeight", "headerHeight": 0},
                        style={"width": "100%"}
                    )
                ], width=12)
            ])
        ], className="report-section no-break")
        report_sections.append(risk_section)
    
    # 8. Flows Summary
    if "flows" in sections:
        flows_df = dw.get_flows_summary_ytd(data)
        grid_class = "ag-theme-alpine" if print_preview else "ag-theme-alpine-dark"
        
        flows_section = html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("Cash Flow Summary (YTD)", className="section-title mb-3"),
                    dag.AgGrid(
                        rowData=flows_df.to_dict('records'),
                        columnDefs=[
                            {"field": "Metric", "headerName": "Metric", "flex": 2},
                            {"field": "Value", "headerName": "Value", "flex": 1}
                        ],
                        defaultColDef={"sortable": True, "resizable": True},
                        className=grid_class,
                        dashGridOptions={"domLayout": "autoHeight", "headerHeight": 0},
                        style={"width": "100%"}
                    )
                ], width=12)
            ])
        ], className="report-section no-break")
        report_sections.append(flows_section)

    # 9. Tax Lot Explorer (Open Lots)
    if "tax_lots" in sections:
        # Calculate tax lots
        open_lots, _ = build_tax_lots(strategy="FIFO", signal=signal)
        grid_class = "ag-theme-alpine" if print_preview else "ag-theme-alpine-dark"
        
        if not open_lots.empty:
            explorer_df = open_lots.copy()
            explorer_df["Date Acquired"] = explorer_df["Date Acquired"].dt.strftime("%Y-%m-%d")
            
            cols_currency = ["Cost Basis", "Current Price", "Market Value", "Unrealized P/L", "Est Tax Liability"]
            cols_hide = ["Is Near Cliff", "Days to LT", "Cost Per Share"]
            col_defs = []
            for c in explorer_df.columns:
                hide = c in cols_hide
                # Ensure currency and long header columns have extra width to prevent truncation
                min_w = 120 if print_preview else 150
                if c in cols_currency or len(c) > 12:
                    min_w = 140 if print_preview else 180
                
                cd = {"field": c, "headerName": c, "hide": hide, "minWidth": min_w, "flex": 1}
                
                if c in cols_currency:
                    cd["valueFormatter"] = {"function": "d3.format('$,.2f')(params.value)"}
                elif c == "Shares":
                    cd["valueFormatter"] = {"function": "d3.format(',.2f')(params.value)"}
                
                col_defs.append(cd)
                
            tax_lots_section = html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H4("Tax Lot Explorer (Open Lots)", className="section-title mb-3"),
                        dag.AgGrid(
                            rowData=explorer_df.to_dict("records"),
                            columnDefs=col_defs,
                            defaultColDef={"sortable": True, "resizable": True},
                            className=grid_class,
                            dashGridOptions={"domLayout": "autoHeight"},
                            style={"width": "100%"}
                        )
                    ], width=12)
                ])
            ], className="report-section page-break-before")
            report_sections.append(tax_lots_section)
        else:
            report_sections.append(html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H4("Tax Lot Explorer (Open Lots)", className="section-title mb-3"),
                        html.Div("No open lots found.", className="text-muted p-3")
                    ], width=12)
                ])
            ], className="report-section no-break"))

    # 11. Risk Analysis Charts
    if "risk_charts" in sections:
        risk_fig = dw.get_risk_return_chart(data, theme)
        corr_fig = dw.get_correlation_heatmap(data, theme)
        dd_fig = dw.get_drawdown_chart(data, theme)
        
        if print_preview:
            risk_fig = apply_print_theme(risk_fig)
            corr_fig = apply_print_theme(corr_fig)
            dd_fig = apply_print_theme(dd_fig)
            
        risk_fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=30), autosize=True)
        corr_fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=30), autosize=True)
        dd_fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=30), autosize=True)
            
        risk_section = html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("Risk Analysis", className="section-title mb-3"),
                ], width=12),
                dbc.Col(dcc.Graph(figure=risk_fig, config={'displayModeBar': False}, style={'height': '500px', 'width': '100%'}, responsive=True), width=12, className="mb-4"),
                dbc.Col(dcc.Graph(figure=corr_fig, config={'displayModeBar': False}, style={'height': '500px', 'width': '100%'}, responsive=True), width=12, className="mb-4"),
                dbc.Col(dcc.Graph(figure=dd_fig, config={'displayModeBar': False}, style={'height': '500px', 'width': '100%'}, responsive=True), width=12)
            ])
        ], className="report-section page-break-before")
        report_sections.append(risk_section)

    # 12. Performance Deep Dive
    if "perf_deep_dive" in sections:
        bm_map = {
            "S&P 500": "SPY",
            "Total US Market": "VTI",
            "Aggressive Alloc": "AOA"
        }
        cum_fig = dw.get_cumulative_return_chart(data, None, bm_map, theme)
        growth_fig = dw.get_growth_of_capital_chart(data, "Total", theme)
        
        if print_preview:
            cum_fig = apply_print_theme(cum_fig)
            growth_fig = apply_print_theme(growth_fig)
            
        cum_fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=30), autosize=True)
        growth_fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=30), autosize=True)
            
        perf_deep_section = html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("Performance Deep Dive", className="section-title mb-3"),
                    html.H5("Cumulative Return vs Benchmark", className="mb-2"),
                    dcc.Graph(figure=cum_fig, config={'displayModeBar': False}, style={'height': '500px', 'width': '100%'}, responsive=True),
                    html.H5("Growth of Invested Capital", className="mt-4 mb-2"),
                    dcc.Graph(figure=growth_fig, config={'displayModeBar': False}, style={'height': '500px', 'width': '100%'}, responsive=True)
                ], width=12)
            ])
        ], className="report-section page-break-before")
        report_sections.append(perf_deep_section)

    # 13. Attribution Analysis
    if "attribution" in sections:
        attr_fig = dw.get_smart_attribution_chart(data, theme)
        if print_preview:
            attr_fig = apply_print_theme(attr_fig)
        
        attr_fig.update_layout(height=600, margin=dict(l=0, r=0, t=30, b=30), autosize=True)
            
        attr_section = html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("Attribution Analysis", className="section-title mb-3"),
                    dcc.Graph(figure=attr_fig, config={'displayModeBar': False}, style={'height': '600px', 'width': '100%'}, responsive=True)
                ], width=12)
            ])
        ], className="report-section page-break-before")
        report_sections.append(attr_section)

    # 14. Asset Class Performance Table
    if "ac_perf_table" in sections:
        class_df = data['class_df']
        sec_table = data['sec_table_current'] # Default to current holdings
        horizons = ["1D", "1W", "MTD", "1M", "3M", "6M", "YTD", "1Y", "SI"]
        
        rows = []
        # Sort classes
        ac_rank_map = {ac: i for i, ac in enumerate(class_df['asset_class'].unique())}
        
        for _, crow in class_df.iterrows():
            ac = crow['asset_class']
            rank = ac_rank_map.get(ac, 999)
            
            # Class Row
            r_vals = {
                "Asset Class / Ticker": ac, 
                "Type": "Class", 
                "_sort_rank": rank
            }
            for h in horizons:
                val = crow.get(h)
                r_vals[h] = fmt_pct_clean(val) if pd.notna(val) else "N/A"
            rows.append(r_vals)
            
            # Ticker Rows
            tickers = sec_table[sec_table['asset_class'] == ac]
            for _, trow in tickers.iterrows():
                t = trow['ticker']
                tr_vals = {
                    "Asset Class / Ticker": f"  {t}", 
                    "Type": "Ticker", 
                    "_sort_rank": rank
                }
                for h in horizons:
                    val = trow.get(h)
                    tr_vals[h] = fmt_pct_clean(val) if pd.notna(val) else "N/A"
                rows.append(tr_vals)
        
        grid_class = "ag-theme-alpine" if print_preview else "ag-theme-alpine-dark"
        cols = ["Asset Class / Ticker"] + horizons
        
        perf_table_section = html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("Asset Class Performance", className="section-title mb-3"),
                    dag.AgGrid(
                        rowData=rows,
                        columnDefs=[
                            {
                                "field": c, 
                                "headerName": c, 
                                "flex": 1, 
                                "pinned": "left" if c == "Asset Class / Ticker" else None, 
                                "minWidth": (120 if print_preview else 150) if c == "Asset Class / Ticker" else (60 if print_preview else 80)
                            } 
                            for c in cols
                        ],
                        defaultColDef={"sortable": False, "resizable": True}, # Disable sort to keep hierarchy
                        className=grid_class,
                        dashGridOptions={
                            "domLayout": "autoHeight",
                            "getRowStyle": {
                                "function": "params.data.Type === 'Class' ? {'fontWeight': 'bold', 'backgroundColor': 'rgba(0,0,0,0.05)'} : {}"
                            }
                        },
                        style={"width": "100%"}
                    )
                ], width=12)
            ])
        ], className="report-section page-break-before")
        report_sections.append(perf_table_section)

    # 15. Asset Class P/L Table
    if "ac_pl_table" in sections:
        class_df = data['class_df']
        sec_table = data['sec_table_current']
        horizons = ["1D", "1W", "MTD", "1M", "3M", "6M", "YTD", "1Y", "SI"]
        
        # Pre-fetch ticker P/L for all horizons
        ticker_pl_cache = {}
        for h in horizons:
            ticker_pl_cache[h] = dw.get_ticker_pl_df(data, h)
            
        rows = []
        ac_rank_map = {ac: i for i, ac in enumerate(class_df['asset_class'].unique())}
        
        for _, crow in class_df.iterrows():
            ac = crow['asset_class']
            rank = ac_rank_map.get(ac, 999)
            
            # Class Row
            r_vals = {
                "Asset Class / Ticker": ac, 
                "Type": "Class", 
                "_sort_rank": rank
            }
            for h in horizons:
                res = dw.get_asset_class_pl(data, ac, h, return_components=False)
                r_vals[h] = fmt_dollar_clean(res) if res is not None else "N/A"
            rows.append(r_vals)
            
            # Ticker Rows
            tickers = sec_table[sec_table['asset_class'] == ac]
            for _, trow in tickers.iterrows():
                t = trow['ticker']
                tr_vals = {
                    "Asset Class / Ticker": f"  {t}", 
                    "Type": "Ticker", 
                    "_sort_rank": rank
                }
                for h in horizons:
                    pl_df = ticker_pl_cache[h]
                    if not pl_df.empty and t in pl_df.index:
                        val = pl_df.loc[t, "pl"]
                        tr_vals[h] = fmt_dollar_clean(val)
                    else:
                        tr_vals[h] = "N/A"
                rows.append(tr_vals)
                
        grid_class = "ag-theme-alpine" if print_preview else "ag-theme-alpine-dark"
        cols = ["Asset Class / Ticker"] + horizons
        
        pl_table_section = html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("Asset Class P/L (Economic)", className="section-title mb-3"),
                    dag.AgGrid(
                        rowData=rows,
                        columnDefs=[
                            {
                                "field": c, 
                                "headerName": c, 
                                "flex": 1, 
                                "pinned": "left" if c == "Asset Class / Ticker" else None, 
                                "minWidth": (120 if print_preview else 150) if c == "Asset Class / Ticker" else (60 if print_preview else 80)
                            } 
                            for c in cols
                        ],
                        defaultColDef={"sortable": False, "resizable": True},
                        className=grid_class,
                        dashGridOptions={
                            "domLayout": "autoHeight",
                            "getRowStyle": {
                                "function": "params.data.Type === 'Class' ? {'fontWeight': 'bold', 'backgroundColor': 'rgba(0,0,0,0.05)'} : {}"
                            }
                        },
                        style={"width": "100%"}
                    )
                ], width=12)
            ])
        ], className="report-section page-break-before")
        report_sections.append(pl_table_section)
    
    return report_title, subtitle, date_line, timestamp, report_sections, container_class


# ============================================================
# WORD REPORT GENERATION CALLBACK
# ============================================================
@callback(
    [Output("download-word-report", "data"),
     Output("download-status-alert", "children"),
     Output("download-status-alert", "color"),
     Output("download-status-alert", "is_open")],
    Input("btn-download-word", "n_clicks"),
    State("report-sections-checklist", "value"),
    State("report-title-input", "value"),
    State("report-period-select", "value"),
    prevent_initial_call=True
)
def download_word_report(n_clicks, sections, title, period):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update, False
        
    data = dw.get_data()
    sections = sections or []
    title = title or "Portfolio Value Report"
    
    period_labels = {
        "SI": "Since Inception",
        "YTD": "Year-to-Date",
        "12M": "Trailing 12 Months",
        "3M": "Last Quarter",
        "1M": "Last Month"
    }
    subtitle = f"{period_labels.get(period, 'Since Inception')} Analysis"
    
    # Generate Doc
    try:
        if not data:
            return dash.no_update, "No data available. Please load the portfolio first.", "warning", True
            
        doc = generate_word_report(data, sections, title, subtitle, period)
        
        # Save to buffer
        buffer = BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        
        return dcc.send_bytes(buffer.getvalue(), filename=f"Investment_Report_{datetime.now().strftime('%Y%m%d')}.docx"), "Report generated successfully!", "success", True
    except Exception as e:
        # In case of error (e.g. kaleido issue), maybe return a text file with error?
        # Or just let Dash handle it (it will show error in debug mode)
        # For now, simplistic error handling
        print(f"Error generating report: {e}")
        return dash.no_update, f"Error: {str(e)}", "danger", True
