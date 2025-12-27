import dash_bootstrap_components as dbc
from dash import dcc, html
import pandas as pd
import numpy as np
from report_formatting import fmt_dollar_clean, fmt_pct_clean
from config import RISK_FREE_RATE

def get_audit_modal_content(request_data):
    """
    Generates the Modal Content for the Audit Trail.
    
    request_data: dict containing:
        - gridId: str
        - colId: str (e.g., '1M')
        - rowIndex: int
        - rowData: dict (the full row data including hidden meta columns)
        - value: float/str (the clicked value)
    """
    if not request_data:
        return dbc.ModalBody("No data provided.")
        
    grid_id = request_data.get("gridId", "")
    col_id = request_data.get("colId")
    row_data = request_data.get("rowData", {})
    
    # ----------------------------------------------------
    # TYPE 1: Asset Class Value Breakdown (Allocation)
    # ----------------------------------------------------
    meta_key_breakdown = f"meta_{col_id}_breakdown"
    if meta_key_breakdown in row_data:
        ticker = row_data.get("Asset Class", "Unknown")
        breakdown = row_data[meta_key_breakdown] # List of dicts
        
        if not breakdown:
             return dbc.ModalBody(html.P("No breakdown available."))

        # Build Table
        tbl_rows = []
        total = 0.0
        for item in breakdown:
            val = item.get("value", 0.0)
            total += val
            tbl_rows.append(html.Tr([
                html.Td(item.get("ticker", "")),
                html.Td(fmt_dollar_clean(val), className="text-end")
            ]))
            
        tbl_rows.append(html.Tr([
            html.Td("Total", className="fw-bold"),
            html.Td(fmt_dollar_clean(total), className="text-end fw-bold", style={'borderTop': '2px solid white'})
        ]))
        
        content = [
            html.H4(f"Audit: {ticker} (Value Breakdown)", className="mb-3"),
            dbc.Table(html.Tbody(tbl_rows), bordered=False, size="sm", className="mt-3", style={'maxWidth': '400px'})
        ]
        return dbc.ModalBody(content)

    # ----------------------------------------------------
    # TYPE 2: Transaction Details (Internal Flows)
    # ----------------------------------------------------
    meta_key_details = f"meta_{col_id}_details"
    if meta_key_details in row_data:
        ticker = row_data.get("Ticker", "Unknown")
        details = row_data[meta_key_details]
        
        if not details:
            return dbc.ModalBody(html.P("No transactions found for this category."))
            
        # Build Table
        # details is list of {date, amount}
        tbl_rows = []
        total = 0.0
        for d in details:
            amt = d.get("amount", 0.0)
            total += amt
            tbl_rows.append(html.Tr([
                html.Td(d.get("date", "")),
                html.Td(fmt_dollar_clean(amt), className="text-end")
            ]))
            
        # Total Row
        tbl_rows.append(html.Tr([
            html.Td("Total", className="fw-bold"),
            html.Td(fmt_dollar_clean(total), className="text-end fw-bold", style={'borderTop': '2px solid white'})
        ]))
        
        content = [
            html.H4(f"Audit: {ticker} - {col_id}", className="mb-3"),
            html.P("Underlying transactions constituting this value:", className="text-muted"),
            dbc.Table(html.Tbody(tbl_rows), bordered=False, size="sm", className="mt-3", style={'maxWidth': '400px'})
        ]
        return dbc.ModalBody(content)

    # ----------------------------------------------------
    # TYPE 2.5: Contribution to Return (CTR)
    # ----------------------------------------------------
    if "Contrib" in str(col_id) and "meta_denominator" in row_data:
        ticker = row_data.get("Asset Class", "Unknown")
        effect = row_data.get("Effect", 0.0)
        denom = row_data.get("meta_denominator", 0.0)
        contrib_pct = row_data.get("Contribution (%)", 0.0)
        
        def fmt_num(n): return f"{n:,.2f}"

        formula_tex = r"""
        $$
        \text{Contribution} = \frac{\text{Effect}}{\text{Average Capital Invested}}
        $$
        """
        
        sub_tex = fr"""
        $$
        \text{{Contribution}} = \frac{{{fmt_dollar_clean(effect)}}}{{{fmt_dollar_clean(denom)}}}
        $$
        """
        
        result_tex = fr"""
        $$
        = {contrib_pct:+.2f}\%
        $$
        """
        
        content = []
        content.append(html.H4(f"Audit: {ticker} (Contribution to Return)", className="mb-3"))
        content.append(dcc.Markdown(formula_tex, mathjax=True, className="text-body"))
        content.append(html.Hr())
        content.append(html.H6("Applied Calculation", className="text-muted"))
        content.append(dcc.Markdown(sub_tex, mathjax=True, className="text-body"))
        content.append(dcc.Markdown(result_tex, mathjax=True, className="text-body"))
        
        rows = [
            html.Tr([html.Td("Asset Class Effect"), html.Td(fmt_dollar_clean(effect), className="text-end")]),
            html.Tr([html.Td("Avg Capital Invested"), html.Td(fmt_dollar_clean(denom), className="text-end")]),
            html.Tr([html.Td("Contribution", className="fw-bold"), html.Td(f"{contrib_pct:+.2f}%", className="text-end fw-bold", style={'borderTop': '1px solid white'})]),
        ]
        content.append(dbc.Table(html.Tbody(rows), bordered=False, size="sm", className="mt-3", style={'maxWidth': '400px'}))
        
        return dbc.ModalBody(content)

    # ----------------------------------------------------
    # TYPE 2.6: Frongello Attribution (Geometric Linking)
    # ----------------------------------------------------
    if "Contrib" in str(col_id) and "meta_frongello_sum_factors" in row_data:
        ticker = row_data.get("Asset Class", "Unknown")
        effect = row_data.get("Effect", 0.0)
        final_contrib = row_data.get("Contribution (%)", 0.0)
        avg_denom = row_data.get("meta_frongello_avg_denom", 0.0)
        
        # Calculate arithmetic proxy for comparison (What it would be without linking)
        arithmetic_proxy = (effect / avg_denom * 100.0) if avg_denom else 0
        interaction_effect = final_contrib - arithmetic_proxy
        
        formula_tex = r"""
        $$
        C_i = \sum_{t} \left( \frac{\text{Effect}_{i,t}}{\text{Portfolio Value}_{t-1}} \times \text{Link Factor}_t \right)
        $$
        """
        
        explanation = (
            "This metric uses Frongello Linking to account for the portfolio's compounding over time. "
            "It sums daily contributions scaled by the portfolio's growth factor (Link Factor) for that day."
        )
        
        rows = [
            html.Tr([html.Td("Total Dollar Effect (Numerator)"), html.Td(fmt_dollar_clean(effect), className="text-end")]),
            html.Tr([html.Td("Avg Capital Invested (Denominator)"), html.Td(fmt_dollar_clean(avg_denom), className="text-end")]),
            html.Tr([html.Td("Simple Arithmetic Return (Proxy)"), html.Td(f"{arithmetic_proxy:.2f}%", className="text-end", style={'fontStyle': 'italic', 'color': '#6c757d'})]),
            html.Tr([html.Td("Geometric Linking Impact"), html.Td(f"{interaction_effect:+.2f}%", className="text-end", style={'fontStyle': 'italic', 'color': '#6c757d'})]),
            html.Tr([
                html.Td("Frongello Linked Contribution", className="fw-bold"), 
                html.Td(f"{final_contrib:+.2f}%", className="text-end fw-bold", style={'borderTop': '1px solid white'})
            ]),
        ]
        
        content = [
            html.H4(f"Audit: {ticker} (Frongello)", className="mb-3"),
            dcc.Markdown(formula_tex, mathjax=True, className="text-body"),
            html.P(explanation, className="text-muted small mb-3"),
            dbc.Table(html.Tbody(rows), bordered=False, size="sm", className="mt-2", style={'maxWidth': '450px'})
        ]
        return dbc.ModalBody(content)

    # ----------------------------------------------------
    # TYPE 2.7: Risk Metrics (Sharpe/Vol)
    # ----------------------------------------------------
    if "Sharpe" in str(col_id) or "Vol" in str(col_id) or "Sortino" in str(col_id):
        ticker = row_data.get("Asset Class / Ticker", "Unknown")
        metric_val = request_data.get('value', 'N/A')
        
        content = []
        content.append(html.H4(f"Audit: {ticker} ({col_id})", className="mb-3"))
        
        if "Sharpe" in str(col_id):
            formula_tex = r"""
            $$
            \text{Sharpe} = \frac{R_p - R_f}{\sigma_p}
            $$
            """
            explanation = (
                f"Calculated using the annualized Return ($R_p$) and Volatility ($\sigma_p$). "
                f"Assumes a Risk-Free Rate ($R_f$) of **{RISK_FREE_RATE:.1%}**. "
                "Higher is better (more return per unit of total risk)."
            )
            content.append(dcc.Markdown(formula_tex, mathjax=True, className="text-body"))
            content.append(dcc.Markdown(explanation, mathjax=True, className="text-muted small"))
            
        elif "Sortino" in str(col_id):
            formula_tex = r"""
            $$
            \text{Sortino} = \frac{R_p - R_f}{\sigma_{down}}
            $$
            """
            explanation = fr"""
            Similar to Sharpe, but divides excess return by **Downside Deviation** ($\sigma_{{down}}$) only. 
            This penalizes only harmful volatility (negative returns). 
            Assumes a Risk-Free Rate ($R_f$) of **{RISK_FREE_RATE:.1%}**.
            """
            content.append(dcc.Markdown(formula_tex, mathjax=True, className="text-body"))
            content.append(dcc.Markdown(explanation, mathjax=True, className="text-muted small"))

        elif "Vol" in str(col_id):
            formula_tex = r"""
            $$
            \text{Volatility} = \sigma = \sqrt{\frac{\sum (R_i - \bar{R})^2}{N-1}} \times \sqrt{252}
            $$
            """
            explanation = r"""
            Annualized Standard Deviation of daily returns. 
            Represents the total variability of the asset's price path.
            """
            content.append(dcc.Markdown(formula_tex, mathjax=True, className="text-body"))
            content.append(dcc.Markdown(explanation, className="text-muted small"))
            
        rows = [
            html.Tr([html.Td("Metric Value"), html.Td(metric_val, className="text-end fw-bold")]),
        ]
        content.append(dbc.Table(html.Tbody(rows), bordered=False, size="sm", className="mt-3", style={'maxWidth': '300px'}))
        
        return dbc.ModalBody(content)

    # ----------------------------------------------------
    # TYPE 3: Contribution Schedule
    # ----------------------------------------------------

    if "contrib-grid" in str(grid_id) or "Monthly Contrib" in col_id or "Gap to Target" in col_id or "Share of Monthly" in col_id:
        ticker = row_data.get("Ticker", "Unknown")
        
        # Extract Meta Data (Fixed keys from dash_wrappers)
        gap = row_data.get("meta_Monthly Contrib_gap", 0.0)
        total_gap = row_data.get("meta_Monthly Contrib_total_gap", 0.0)
        total_monthly = row_data.get("meta_Monthly Contrib_total_monthly", 0.0)
        
        content = []
        content.append(html.H4(f"Audit: {ticker} ({col_id})", className="mb-3"))
        
        if col_id == "Gap to Target":
            formula_tex = r"""
        $$ 
        \text{Gap} = \text{Target Value} - \text{Current Value} 
        $$
        """
            
            content.append(dcc.Markdown(formula_tex, mathjax=True, className="text-body"))
            content.append(html.Hr())
            content.append(html.P("The gap is the difference between the target allocation value and the current market value for this holding.", className="text-muted"))
            
            rows = [
                html.Tr([html.Td("Gap to Target", className="fw-bold"), html.Td(fmt_dollar_clean(gap), className="text-end fw-bold", style={'borderTop': '1px solid white'})]),
            ]
            content.append(dbc.Table(html.Tbody(rows), bordered=False, size="sm", className="mt-3", style={'maxWidth': '400px'}))

        elif col_id == "Monthly Contrib":
            def fmt_num(n): return f"{n:,.2f}"
            
            formula_tex = r"""
        $$
        \text{Contribution} = \frac{\text{Gap}}{\text{Total Gap}} \times \text{Total Monthly Contribution}
        $$
        """
            
            sub_tex = fr"""
        $$
        \text{{Contribution}} = \frac{{{fmt_num(gap)}}}{{{fmt_num(total_gap)}}} \times {fmt_num(total_monthly)}
        $$
        """
            
            res = (gap / total_gap * total_monthly) if total_gap > 0 else 0
            pct = gap / total_gap if total_gap > 0 else 0

            content.append(dcc.Markdown(formula_tex, mathjax=True, className="text-body"))
            content.append(html.Hr())
            content.append(html.H6("Applied Calculation", className="text-muted"))
            content.append(dcc.Markdown(sub_tex, mathjax=True, className="text-body"))
            
            rows = [
                html.Tr([html.Td("This Position's Gap"), html.Td(fmt_dollar_clean(gap), className="text-end")]),
                html.Tr([html.Td("Total Portfolio Gap"), html.Td(fmt_dollar_clean(total_gap), className="text-end")]),
                html.Tr([html.Td("Total Monthly Contribution"), html.Td(fmt_dollar_clean(total_monthly), className="text-end")]),
                html.Tr([html.Td("Share of Total Gap"), html.Td(fmt_pct_clean(pct), className="text-end")]),
                html.Tr([html.Td("Monthly Contribution", className="fw-bold"), html.Td(fmt_dollar_clean(res), className="text-end fw-bold", style={'borderTop': '1px solid white'})]),
            ]
            content.append(dbc.Table(html.Tbody(rows), bordered=False, size="sm", className="mt-3", style={'maxWidth': '400px'}))

        elif col_id == "Share of Monthly":
            def fmt_num(n): return f"{n:,.2f}"
            
            formula_tex = r"""
        $$
        \text{Share} = \frac{\text{Monthly Contribution}}{\text{Total Monthly Contribution}}
        $$
        """
            
            contrib_val = (gap / total_gap * total_monthly) if total_gap > 0 else 0
            
            sub_tex = fr"""
        $$
        \text{{Share}} = \frac{{{fmt_num(contrib_val)}}}{{{fmt_num(total_monthly)}}}
        $$
        """
            
            share_result = contrib_val/total_monthly if total_monthly > 0 else 0

            content.append(dcc.Markdown(formula_tex, mathjax=True, className="text-body"))
            content.append(html.Hr())
            content.append(html.H6("Applied Calculation", className="text-muted"))
            content.append(dcc.Markdown(sub_tex, mathjax=True, className="text-body"))
            
            rows = [
                html.Tr([html.Td("Monthly Contribution"), html.Td(fmt_dollar_clean(contrib_val), className="text-end")]),
                html.Tr([html.Td("Total Monthly Contribution"), html.Td(fmt_dollar_clean(total_monthly), className="text-end")]),
                html.Tr([html.Td("Share of Monthly", className="fw-bold"), html.Td(fmt_pct_clean(share_result), className="text-end fw-bold", style={'borderTop': '1px solid white'})]),
            ]
            content.append(dbc.Table(html.Tbody(rows), bordered=False, size="sm", className="mt-3", style={'maxWidth': '400px'}))
            
        else:
            content.append(html.P("Calculation detail not available for this column."))
            
        return dbc.ModalBody(content)

    # ----------------------------------------------------
    # TYPE 5: TWR Audit (Monthly Schedule)
    # ----------------------------------------------------
    if "twr_monthly_breakdown" in request_data:
        monthly_data = request_data["twr_monthly_breakdown"]
        
        # Summary Values
        horizon = row_data.get("Horizon", "Period")
        
        # Get start/end/flow from meta if available, else from row
        v_start = request_data.get("meta_Return_start", 0.0)
        v_end = request_data.get("meta_Return_end", 0.0)
        flow = request_data.get("meta_Return_flow", 0.0)
        
        # Formula Header
        formula_tex = r"""
        $$
        TWR = \left( \prod_{t=1}^{n} (1 + r_t) \right) - 1
        $$
        """
        
        content = []
        content.append(html.H4(f"Audit: Portfolio Return ({horizon})", className="mb-3"))
        content.append(dcc.Markdown(formula_tex, mathjax=True, className="text-body"))
        
        # Summary Table
        summary_rows = [
            html.Tr([html.Td("Start Value"), html.Td(fmt_dollar_clean(v_start), className="text-end")]),
            html.Tr([html.Td("End Value"), html.Td(fmt_dollar_clean(v_end), className="text-end")]),
            html.Tr([html.Td("Net External Flows"), html.Td(fmt_dollar_clean(flow), className="text-end")]),
            html.Tr([html.Td("Final Return", className="fw-bold"), html.Td(str(request_data.get('value', 'N/A')), className="text-end fw-bold", style={'borderTop': '1px solid white'})]),
        ]
        content.append(dbc.Table(html.Tbody(summary_rows), bordered=False, size="sm", className="mt-3 mb-4", style={'maxWidth': '400px'}))
        
        # Handle Insufficient Data
        if not monthly_data:
             content.append(dbc.Alert("Insufficient historical data to calculate Time-Weighted Return for this horizon.", color="warning", className="mt-2"))
             return dbc.ModalBody(content)
        
        content.append(html.H5("Monthly Return Schedule", className="mb-2"))
        
        # Monthly Table
        tbl_header = html.Thead(html.Tr([
            html.Th("Month"),
            html.Th("Return", className="text-end"),
            html.Th("Factor", className="text-end"),
        ]))
        
        tbl_body_rows = []
        for item in monthly_data:
            m_ret = item.get("return", 0.0)
            m_fac = item.get("factor", 1.0)
            
            # Styling for negative returns
            style = {"color": "#dc3545"} if m_ret < 0 else {"color": "#28a745"}
            
            tbl_body_rows.append(html.Tr([
                html.Td(item.get("display_date")),
                html.Td(f"{m_ret*100:,.2f}%", className="text-end", style=style),
                html.Td(f"{m_fac:.6f}", className="text-end"),
            ]))
            
        content.append(html.Div(
            dbc.Table([tbl_header, html.Tbody(tbl_body_rows)], bordered=True, hover=True, size="sm", className="table-dark"),
            style={"maxHeight": "400px", "overflowY": "auto"}
        ))
        
        return dbc.ModalBody(content)

    # ----------------------------------------------------
    # TYPE 4: Standard Return / P/L (Horizon Analysis)
    # ----------------------------------------------------
    
    # Improved Title Logic
    ticker = row_data.get("ticker")
    if not ticker: ticker = row_data.get("Horizon")
    if not ticker: ticker = row_data.get("Asset Class / Ticker")
    if not ticker: ticker = row_data.get("Asset Class")
    if not ticker: ticker = "Unknown"
    
    # Meta Keys
    key_start = f"meta_{col_id}_start"
    key_end = f"meta_{col_id}_end"
    key_flow = f"meta_{col_id}_flow"
    key_inc = f"meta_{col_id}_inc"
    
    # Check availability
    if key_start not in row_data:
        return dbc.ModalBody(html.Div([
            html.H5("Audit Unavailable", className="text-danger"),
            html.P(f"No detailed audit data found for column '{col_id}'.")
        ]))
        
    # Extract Values (SAFE FLOAT CONVERSION)
    def safe_float(v):
        if v is None: return 0.0
        try: return float(v)
        except: return 0.0

    v_start = safe_float(row_data.get(key_start))
    v_end = safe_float(row_data.get(key_end))
    flow = safe_float(row_data.get(key_flow))
    inc = safe_float(row_data.get(key_inc))
    
    # Determine Logic Type
    is_pl_grid = "pl-grid" in str(grid_id)
    is_pl_col = "PL" in str(col_id).upper() or "P/L" in str(col_id).upper()
    
    is_pl = is_pl_grid or is_pl_col
    is_return = not is_pl
    
    content = []
    content.append(html.H4(f"Audit: {ticker} ({col_id})", className="mb-3"))
    
    def fmt_num(n): return f"{n:,.2f}"
    
    if is_return:
        # Modified Dietz Formula
        key_denom = f"meta_{col_id}_denom"
        denom = safe_float(row_data.get(key_denom, v_start + flow))
        
        formula_tex = r"""
        $$
        \text{Return} = \frac{V_{end} - V_{start} - \text{Net Flows} + \text{Income}}{\text{Average Capital Invested}}
        $$
        """
        
        sub_tex = fr"""
        $$
        \text{{Return}} = \frac{{{fmt_num(v_end)} - {fmt_num(v_start)} - ({fmt_num(flow)}) + {fmt_num(inc)}}}{{{fmt_num(denom)}}}
        $$
        """
        
        numerator = v_end - v_start - flow + inc
        result_tex = fr"""
        $$
        = \frac{{{fmt_num(numerator)}}}{{{fmt_num(denom)}}} = {request_data.get('value', 'N/A')}
        $$
        """
        
        content.append(dcc.Markdown(formula_tex, mathjax=True, className="text-body"))
        content.append(html.Hr())
        content.append(html.H6("Applied Calculation", className="text-muted"))
        content.append(dcc.Markdown(sub_tex, mathjax=True, className="text-body"))
        content.append(dcc.Markdown(result_tex, mathjax=True, className="text-body"))
        
        rows = [
            html.Tr([html.Td("Ending Value (V_end)"), html.Td(fmt_dollar_clean(v_end), className="text-end")]),
            html.Tr([html.Td("Starting Value (V_start)"), html.Td(fmt_dollar_clean(v_start), className="text-end")]),
            html.Tr([html.Td("Net Capital Flows"), html.Td(fmt_dollar_clean(flow), className="text-end")]),
            html.Tr([html.Td("Income (Dividends)"), html.Td(fmt_dollar_clean(inc), className="text-end")]),
            html.Tr([html.Td("Avg Capital (Denom)", className="fw-bold"), html.Td(fmt_dollar_clean(denom), className="text-end fw-bold", style={'borderTop': '1px solid white'})]),
        ]
        content.append(dbc.Table(html.Tbody(rows), bordered=False, size="sm", className="mt-3", style={'maxWidth': '400px'}))
        
    else:
        # P/L Logic
        # Hide Income line if 0 (Standard for Portfolio Level where Income is inside V_end)
        show_inc = abs(inc) >= 0.01

        if show_inc:
            formula_tex = r"""
            $$
            \text{P/L} = V_{end} - V_{start} - \text{Net Flows} + \text{Income}
            $$
            """
            
            sub_tex = fr"""
            $$
            \text{{P/L}} = {fmt_num(v_end)} - {fmt_num(v_start)} - ({fmt_num(flow)}) + {fmt_num(inc)}
            $$
            """
        else:
            formula_tex = r"""
            $$
            \text{P/L} = V_{end} - V_{start} - \text{Net Flows}
            $$
            """
            
            sub_tex = fr"""
            $$
            \text{{P/L}} = {fmt_num(v_end)} - {fmt_num(v_start)} - ({fmt_num(flow)})
            $$
            """
        
        content.append(dcc.Markdown(formula_tex, mathjax=True, className="text-body"))
        content.append(html.Hr())
        content.append(html.H6("Applied Calculation", className="text-muted"))
        content.append(dcc.Markdown(sub_tex, mathjax=True, className="text-body"))
        
        pl_calc = v_end - v_start - flow + inc
        
        rows = [
             html.Tr([html.Td("Ending Value"), html.Td(fmt_dollar_clean(v_end), className="text-end")]),
             html.Tr([html.Td("(-) Starting Value"), html.Td(f"-{fmt_dollar_clean(v_start)}", className="text-end")]),
             html.Tr([html.Td("(-) Net Flows"), html.Td(f"-{fmt_dollar_clean(flow)}", className="text-end")]),
        ]
        
        if show_inc:
            rows.append(html.Tr([html.Td("(+) Income"), html.Td(f"+{fmt_dollar_clean(inc)}", className="text-end")]))
            
        rows.append(html.Tr([html.Td("Total P/L", className="fw-bold"), html.Td(fmt_dollar_clean(pl_calc), className="text-end fw-bold", style={'borderTop': '1px solid white'})]))
        
        content.append(dbc.Table(html.Tbody(rows), bordered=False, size="sm", className="mt-3", style={'maxWidth': '400px'}))

    return dbc.ModalBody(content)
