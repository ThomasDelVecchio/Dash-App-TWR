import pandas as pd
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from io import BytesIO
import plotly.graph_objects as go
import dash_wrappers as dw 
from tax_engine import build_tax_lots

# =====================================================================
# Formatting Helpers
# =====================================================================

def fmt_pct_clean(x):
    try:
        if x is None or pd.isna(x):
            return "N/A"
        return f"{float(x)*100:.2f}%"
    except:
        return "N/A"

def fmt_dollar_clean(x):
    try:
        if x is None or pd.isna(x):
            return "N/A"
        return f"${float(x):,.2f}"
    except:
        return "N/A"

def fmt_number_clean(x):
    try:
        if x is None or pd.isna(x):
            return "N/A"
        return f"{float(x):,.2f}"
    except:
        return "N/A"

def safe(x):
    return "N/A" if x is None or pd.isna(x) else x

# =====================================================================
# Document Styling Helpers
# =====================================================================

def set_narrow_margins(doc):
    """Sets page margins to 0.5 inches."""
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.5)
        section.bottom_margin = Inches(0.5)
        section.left_margin = Inches(0.5)
        section.right_margin = Inches(0.5)

def add_header(doc, text, level=1):
    """Adds a centered header."""
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    # Keep with next paragraph (the table or chart)
    p.paragraph_format.keep_with_next = True

def add_paragraph_centered(doc, text, bold=False):
    """Adds a centered paragraph."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    if bold:
        run.bold = True

def add_page_break(doc):
    doc.add_page_break()

def add_markdown_paragraph(doc, text):
    """
    Parses simple markdown (bold only via **) and adds to doc.
    """
    if not text: return
    
    # Split into logical paragraphs by newlines
    paragraphs = text.split('\n')
    
    for paragraph_text in paragraphs:
        if not paragraph_text.strip():
            continue
            
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        
        # Split by ** for bolding
        parts = paragraph_text.split("**")
        for i, part in enumerate(parts):
            run = p.add_run(part)
            if i % 2 == 1: # Odd parts are inside **...**
                run.bold = True

# =====================================================================
# Table Helpers
# =====================================================================

def add_table(doc, headers, rows, right_align=None):
    """
    Adds a table with 'Light Grid Accent 1' style, autofitted to ~7.5 inches.
    """
    try:
        table = doc.add_table(rows=1, cols=len(headers))
        table.style = "Light Grid Accent 1"
        table.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Enable autofit
        table.autofit = True
        table.allow_autofit = True

        # Target width = 7.5" (8.5 - 0.5 - 0.5)
        max_width = Inches(7.5)
        col_width = max_width / len(headers)

        for col in table.columns:
            for cell in col.cells:
                cell.width = col_width

        # Header
        hdr_row = table.rows[0]
        # Repeat header row on new pages
        trPr = hdr_row._tr.get_or_add_trPr()
        trPr.append(OxmlElement("w:tblHeader"))
        
        hdr = hdr_row.cells
        for i, h in enumerate(headers):
            hdr[i].text = str(h)
            for p in hdr[i].paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:
                    r.bold = True

        # Rows
        for row in rows:
            row_cells = table.add_row().cells
            for i, val in enumerate(row):
                row_cells[i].text = str(val)
                
                # Align
                align = WD_ALIGN_PARAGRAPH.LEFT
                if right_align and i in right_align:
                    align = WD_ALIGN_PARAGRAPH.RIGHT
                elif i == 0 and len(headers) > 1: # Usually left align 1st col
                    align = WD_ALIGN_PARAGRAPH.LEFT
                else:
                    align = WD_ALIGN_PARAGRAPH.CENTER
                
                for p in row_cells[i].paragraphs:
                    p.alignment = align

        # Prevent row splitting
        for row in table.rows:
            tr = row._tr
            trPr = tr.get_or_add_trPr()
            cant = OxmlElement("w:cantSplit")
            trPr.append(cant)
            
        doc.add_paragraph() # Spacer
        return table
    except Exception as e:
        print(f"Error adding table: {e}")
        return None

# =====================================================================
# Image / Chart Helpers
# =====================================================================

def add_figure_to_doc(doc, fig, width_inches=7.5, height_inches=4.5):
    """
    Converts a Plotly fig to image and adds it to the Word doc.
    Uses kaleido.
    """
    if fig is None:
        return
    
    # 1. Update layout for static export (white background)
    fig.update_layout(
         template='plotly_white',
         paper_bgcolor='white',
         plot_bgcolor='white',
         font=dict(size=10, color='black'),
         margin=dict(l=20, r=20, t=40, b=20),
    )
    
    # 2. Convert to image bytes
    try:
        img_bytes = fig.to_image(format="png", width=width_inches*100, height=height_inches*100, scale=3)
        
        # 3. Add to doc
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(BytesIO(img_bytes), width=Inches(width_inches))
    except Exception as e:
        err_p = doc.add_paragraph()
        err_p.add_run(f"[Chart could not be generated: {str(e)}]")


# =====================================================================
# Main Generator
# =====================================================================

def generate_word_report(data, sections, report_title, subtitle, period_label):
    """
    Generates a Word Document based on selected sections.
    """
    doc = Document()
    set_narrow_margins(doc)
    
    # Title Page / Header
    add_header(doc, report_title, level=1)
    if subtitle:
        add_paragraph_centered(doc, subtitle)
    add_paragraph_centered(doc, f"Period: {period_label}", bold=True)
    add_paragraph_centered(doc, f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    doc.add_paragraph() # Spacer

    # -----------------------------------------------------------------
    # 0. Morning AI Summary (Optional)
    # -----------------------------------------------------------------
    if "morning_brief" in sections:
        # Avoid circular import at top level
        try:
             # Try absolute import which might work if not circular, but safe to keep as is
             from components.ai_brief import generate_ai_summary
        except:
             pass 

        # If data is missing or we need to calculate specific brief data:
        summary_text = generate_ai_summary(data) if 'generate_ai_summary' in locals() else "AI Summary not available."
        
        add_header(doc, "Morning AI Summary", level=2)
        add_markdown_paragraph(doc, summary_text)
        doc.add_paragraph() # Spacer

    # -----------------------------------------------------------------
    # 1. Executive Summary
    # -----------------------------------------------------------------
    if "summary" in sections:
        add_header(doc, "Executive Summary", level=2)
        
        metrics = dw.get_snapshot_metrics(data)
        metrics2 = data.get("snapshot_metrics", {})
        
        # Calculate MTD return from twr_df if available
        mtd_ret = 0.0
        twr_df = data.get("twr_df", pd.DataFrame())
        if not twr_df.empty and "Horizon" in twr_df.columns:
            row = twr_df[twr_df["Horizon"] == "MTD"]
            if not row.empty:
                mtd_ret = row["Return"].iloc[0]

        # Get Current Value (PV)
        pv = data.get("pv", pd.Series())
        curr_val = pv.iloc[-1] if not pv.empty else 0.0
        
        kpis = [
            ["Metric", "Value"],
            ["Total P/L (SI)", fmt_dollar_clean(metrics.get('pl_si'))],
            ["Cumulative Return (SI)", fmt_pct_clean(metrics.get('twr_si'))],
            ["Current Value", fmt_dollar_clean(curr_val)],
            ["MTD Return", fmt_pct_clean(mtd_ret)],
            ["Max Drawdown", fmt_pct_clean(metrics.get('max_dd'))],
            ["Sharpe Ratio", f"{metrics.get('sharpe', 'N/A'):.2f}" if isinstance(metrics.get('sharpe'), (int, float)) else str(metrics.get('sharpe', 'N/A'))],
            ["Sortino Ratio", f"{metrics.get('sortino', 'N/A'):.2f}" if isinstance(metrics.get('sortino'), (int, float)) else str(metrics.get('sortino', 'N/A'))],
        ]
        add_table(doc, kpis[0], kpis[1:], right_align=[1])

    # -----------------------------------------------------------------
    # 2. Performance Chart
    # -----------------------------------------------------------------
    if "performance_chart" in sections:
        add_header(doc, "Portfolio Performance", level=2)
        fig = dw.get_pv_mountain_chart(data, theme='light') 
        add_figure_to_doc(doc, fig)

    # -----------------------------------------------------------------
    # 3. Horizon Analysis Table
    # -----------------------------------------------------------------
    if "horizon_table" in sections:
        add_header(doc, "Horizon Analysis", level=2)
        
        horizon_df = dw.get_horizon_analysis(data)
        if not horizon_df.empty:
            headers = ["Horizon", "Return", "P/L ($)"]
            # Convert DF to list of lists including index
            rows = []
            for idx, row in horizon_df.iterrows():
                rows.append([
                    row.get('Horizon', str(idx)),
                    fmt_pct_clean(row.get('Return', 0)),
                    fmt_dollar_clean(row.get('P/L', 0))
                ])
            add_table(doc, headers, rows, right_align=[1, 2])

    # -----------------------------------------------------------------
    # 4. Asset Allocation
    # -----------------------------------------------------------------
    if "allocation" in sections:
        add_page_break(doc)
        add_header(doc, "Asset Allocation", level=2)
        
        # Unpack tuple
        res = dw.get_asset_allocation_charts(data, theme='light')
        if res:
            pie_fig = res[0] # Pie is first
            add_figure_to_doc(doc, pie_fig, width_inches=6, height_inches=4)
            
            bar_fig = res[1] 
            add_figure_to_doc(doc, bar_fig, width_inches=6, height_inches=4)

    # -----------------------------------------------------------------
    # 5. Sector Breakdown
    # -----------------------------------------------------------------
    if "sector" in sections:
        add_header(doc, "Sector Breakdown", level=2)
        sector_fig = dw.get_sector_allocation_chart(data, theme='light')
        add_figure_to_doc(doc, sector_fig)

    # -----------------------------------------------------------------
    # 6. Top Holdings
    # -----------------------------------------------------------------
    if "holdings" in sections:
        add_header(doc, "Top Holdings", level=2)
        sec_table = data.get('sec_table_current', pd.DataFrame()).copy()
        if not sec_table.empty:
            # Sort by weight if possible, else market value
            if 'weight' in sec_table.columns:
                sort_col = 'weight'
            else:
                sort_col = 'market_value'
                
            sec_table = sec_table.sort_values(sort_col, ascending=False).head(10)
            
            headers = ["Ticker", "Asset Class", "Shares", "Value", "Weight"]
            rows = []
            for _, row in sec_table.iterrows():
                rows.append([
                    str(row.get('ticker','')),
                    str(row.get('asset_class','')),
                    f"{row.get('shares', 0):,.2f}",
                    fmt_dollar_clean(row.get('market_value',0)),
                    fmt_pct_clean(row.get('weight',0))
                ])
                
            add_table(doc, headers, rows, right_align=[2, 3, 4])

    # -----------------------------------------------------------------
    # 7. Risk Metrics
    # -----------------------------------------------------------------
    if "risk" in sections:
        add_header(doc, "Risk Metrics", level=2)
        risk_df = dw.get_risk_diversification(data) 
        if not risk_df.empty:
             headers = ["Metric", "Value"]
             rows = []
             for _, row in risk_df.iterrows():
                 rows.append([
                     str(row.get('Metric', '')),
                     str(row.get('Value', ''))
                 ])
             add_table(doc, headers, rows, right_align=[1])

    # -----------------------------------------------------------------
    # 8. Net Flows
    # -----------------------------------------------------------------
    if "flows" in sections:
        add_header(doc, "Net Flows", level=2)
        flows_df = dw.get_flows_summary_ytd(data)
        if not flows_df.empty:
             headers = ["Metric", "Value"] # Match keys from diff
             rows = []
             # YTD summary returns keys like: "Start Value", "Net Inflow", "End Value" in 'Metric' col
             for _, row in flows_df.iterrows():
                 rows.append([
                     str(row.get('Metric', '')),
                     str(row.get('Value', '')) # Value is already formatted in get_flows_summary_ytd probably, or we check
                 ])
             # Check if we need formatting? 
             # Looking at dash_wrappers.py get_flows_summary_ytd, it returns formatted strings?
             # Let's assume it matches the UI which handles formatted strings or raw. 
             # Actually, looking at custom_report.py Section 8:
             # dag.AgGrid(rowData=flows_df.to_dict('records')...)
             # It displays exactly what's in the DF. So we just dump str(Value).
             
             add_table(doc, headers, rows, right_align=[1])

    # -----------------------------------------------------------------
    # 9. Tax Lot Explorer (Open Lots)
    # -----------------------------------------------------------------
    if "tax_lots" in sections:
        add_header(doc, "Tax Lot Explorer (Open Lots)", level=2)
        # Use simple FIFO strat since we don't have signal easy access
        open_lots, _ = build_tax_lots(strategy="FIFO")
        
        if not open_lots.empty:
            # Ensure Date formatting
            if "Date Acquired" in open_lots.columns:
                open_lots["Date Acquired"] = pd.to_datetime(open_lots["Date Acquired"]).dt.strftime("%Y-%m-%d")

            # Filter columns to match UI (hiding technical ones)
            cols_hide = ["Is Near Cliff", "Days to LT", "Cost Per Share"]
            display_cols = [c for c in open_lots.columns if c not in cols_hide]
            
            # Format rows
            rows = []
            # Max width constraint - limit columns if too many
            # For Word, keep core columns
            core_cols = ["Ticker", "Date Acquired", "Shares", "Cost Basis", "Market Value", "Unrealized P/L", "Term"]
            # Fallback if names differ
            final_cols = [c for c in core_cols if c in display_cols]
            if not final_cols: final_cols = display_cols[:7]
            
            for _, row in open_lots[final_cols].iterrows():
                r_vals = []
                for col in final_cols:
                    val = row[col]
                    # Apply formatting based on column name
                    if col in ["Cost Basis", "Current Price", "Market Value", "Unrealized P/L", "Est Tax Liability"]:
                        r_vals.append(fmt_dollar_clean(val))
                    elif col == "Shares":
                        r_vals.append(f"{val:,.2f}")
                    else:
                        r_vals.append(str(val))
                rows.append(r_vals)
                
            # Assume last few columns are numeric and should be right-aligned
            add_table(doc, final_cols, rows)
        else:
             doc.add_paragraph("No open tax lots found.")

    # -----------------------------------------------------------------
    # 10. Risk Analysis Charts
    # -----------------------------------------------------------------
    if "risk_charts" in sections:
        add_page_break(doc)
        add_header(doc, "Risk Analysis Charts", level=2)
        
        # Risk Return
        add_paragraph_centered(doc, "Risk vs Return Profile")
        fig_risk = dw.get_risk_return_chart(data, theme='light')
        add_figure_to_doc(doc, fig_risk, height_inches=4)
        
        # Correlation
        add_paragraph_centered(doc, "Correlation Matrix")
        fig_corr = dw.get_correlation_heatmap(data, theme='light')
        add_figure_to_doc(doc, fig_corr, height_inches=4)
        
        # Drawdown
        add_page_break(doc)
        add_paragraph_centered(doc, "Drawdown Analysis")
        fig_dd = dw.get_drawdown_chart(data, theme='light')
        add_figure_to_doc(doc, fig_dd, height_inches=4)

    # -----------------------------------------------------------------
    # 11. Performance Deep Dive
    # -----------------------------------------------------------------
    if "perf_deep_dive" in sections:
        add_page_break(doc)
        add_header(doc, "Performance Deep Dive", level=2)
        
        bm_map = {
            "S&P 500": "SPY",
            "Total US Market": "VTI",
            "Aggressive Alloc": "AOA"
        }
        add_paragraph_centered(doc, "Cumulative Return vs Benchmark")
        cum_fig = dw.get_cumulative_return_chart(data, None, bm_map, theme='light')
        add_figure_to_doc(doc, cum_fig, height_inches=4)
        
        add_paragraph_centered(doc, "Growth of Invested Capital")
        growth_fig = dw.get_growth_of_capital_chart(data, "Total", theme='light')
        add_figure_to_doc(doc, growth_fig, height_inches=4)

    # -----------------------------------------------------------------
    # 12. Attribution Analysis
    # -----------------------------------------------------------------
    if "attribution" in sections:
        add_header(doc, "Attribution Analysis", level=2)
        attr_fig = dw.get_smart_attribution_chart(data, theme='light')
        add_figure_to_doc(doc, attr_fig, height_inches=5)

    # -----------------------------------------------------------------
    # 13. Asset Class Performance Table
    # -----------------------------------------------------------------
    if "ac_perf_table" in sections:
        add_page_break(doc)
        add_header(doc, "Asset Class Performance", level=2)
        
        class_df = data.get('class_df', pd.DataFrame())
        sec_table = data.get('sec_table_current', pd.DataFrame())
        # Full list of horizons to match App
        horizons = ["1D", "1W", "MTD", "1M", "3M", "6M", "YTD", "1Y", "SI"]
        
        headers = ["Asset Class / Ticker"] + horizons
        rows = []
        
        # Unique classes from class_df
        unique_classes = class_df['asset_class'].unique() if not class_df.empty else []
        
        for ac in unique_classes:
            # 1. Class Row
            crow_df = class_df[class_df['asset_class'] == ac]
            if crow_df.empty: continue
            crow = crow_df.iloc[0]
            
            r_vals = [ac]
            for h in horizons:
                val = crow.get(h)
                r_vals.append(fmt_pct_clean(val))
            rows.append(r_vals)
            
            # 2. Ticker Rows
            if not sec_table.empty:
                tickers = sec_table[sec_table['asset_class'] == ac]
                for _, trow in tickers.iterrows():
                    t = trow.get('ticker', '')
                    tr_vals = [f"    {t}"] # Indent
                    for h in horizons:
                        val = trow.get(h)
                        tr_vals.append(fmt_pct_clean(val))
                    rows.append(tr_vals)
                    
        add_table(doc, headers, rows, right_align=list(range(1, len(headers))))

    # -----------------------------------------------------------------
    # 14. Asset Class P/L Table
    # -----------------------------------------------------------------
    if "ac_pl_table" in sections:
        add_header(doc, "Asset Class P/L (Economic)", level=2)
        
        class_df = data.get('class_df', pd.DataFrame())
        sec_table = data.get('sec_table_current', pd.DataFrame())
        horizons = ["1D", "1W", "MTD", "1M", "3M", "6M", "YTD", "1Y", "SI"]
        
        # Pre-fetch ticker P/L for all horizons to avoid re-calc in loop
        ticker_pl_cache = {}
        for h in horizons:
            try:
                ticker_pl_cache[h] = dw.get_ticker_pl_df(data, h)
            except:
                ticker_pl_cache[h] = pd.DataFrame() # Fallback
            
        rows = []
        headers = ["Asset Class / Ticker"] + horizons
        
        # Determine order of asset classes (can rely on class_df order)
        unique_classes = class_df['asset_class'].unique() if not class_df.empty else []
        
        for ac in unique_classes:
            # 1. Class Row
            r_vals = [ac]
            for h in horizons:
                res = dw.get_asset_class_pl(data, ac, h, return_components=False)
                r_vals.append(fmt_dollar_clean(res))
            rows.append(r_vals)
            
            # 2. Ticker Rows belonging to this class
            if not sec_table.empty:
                tickers = sec_table[sec_table['asset_class'] == ac]
                for _, trow in tickers.iterrows():
                    t = trow['ticker']
                    tr_vals = [f"    {t}"] # Indent
                    for h in horizons:
                        pl_df = ticker_pl_cache.get(h)
                        val = None
                        if pl_df is not None and not pl_df.empty and t in pl_df.index:
                            val = pl_df.loc[t, "pl"]
                        tr_vals.append(fmt_dollar_clean(val))
                    rows.append(tr_vals)
                    
        add_table(doc, headers, rows, right_align=list(range(1, len(headers))))

    return doc
