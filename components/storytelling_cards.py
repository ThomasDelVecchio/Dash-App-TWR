"""
Storytelling Cards — Phase 3

Magazine-style editorial cards for executive-grade analytics display.
Each card composes:
  - Lead metric (hero number) with animated count-up
  - Annotated mini-trend (inline sparkline via Plotly)
  - Delta badge (vs prior period)
  - "What changed" narrative zone
  - Context chips / badges (auto-warnings)
  - Click-to-drill navigation

Desktop: asymmetric, clean visual composition.
Mobile: stacked, readable cards with collapsible details.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
from portfolio_engine import compute_drawdown_series


# ── Colour tokens (match styles.css design system) ──────────────
_GREEN = "#22c55e"
_RED = "#ef4444"
_ACCENT = "#00d4ff"
_NEUTRAL = "#94a3b8"
_WARNING = "#f59e0b"
_MUTED_BG = "rgba(255,255,255,0.03)"
_TEAL = "#14b8a6"
_VIOLET = "#a78bfa"
_AMBER = "#f59e0b"


def _polarity_color(value):
    """Return green/red/neutral based on sign."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return _NEUTRAL
    return _GREEN if value >= 0 else _RED


def _arrow(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return "▲" if value >= 0 else "▼"


def _mini_trend_figure(series: pd.Series, color: str = _ACCENT, height: int = 48):
    """
    Build a tiny Plotly sparkline figure for embedding inside a card.
    Expects a numeric Series (e.g. daily prices or cumulative returns).
    Y-axis is auto-ranged to the data so small variations are visible.
    """
    if series is None or series.empty or len(series) < 2:
        return go.Figure().update_layout(
            height=height, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )

    # Determine line colour from trend direction
    trend_color = _GREEN if series.iloc[-1] >= series.iloc[0] else _RED
    fill_rgba = f"rgba({int(trend_color[1:3],16)},{int(trend_color[3:5],16)},{int(trend_color[5:7],16)},0.10)"

    vals = series.values.astype(float)
    y_min, y_max = float(vals.min()), float(vals.max())
    y_pad = max((y_max - y_min) * 0.10, abs(y_max) * 0.001, 0.01)

    # Baseline trace at the series minimum (invisible) so fill="tonexty"
    # shades only the area between the baseline and the line, not down to 0.
    xs = list(range(len(series)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=[y_min - y_pad] * len(xs),
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=list(vals),
        mode="lines",
        line=dict(color=trend_color, width=1.5),
        fill="tonexty",
        fillcolor=fill_rgba,
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, range=[y_min - y_pad, y_max + y_pad]),
    )
    return fig


def _mini_bar_chart(labels, values, height: int = 56):
    """
    Horizontal bar chart for showing per-ticker 1M returns or similar.
    Bars are coloured green/red by sign.
    """
    if not labels or not values:
        return go.Figure().update_layout(
            height=height, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )

    colors = [_GREEN if v >= 0 else _RED for v in values]

    fig = go.Figure(go.Bar(
        y=labels,
        x=values,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in values],
        textposition="inside",
        insidetextanchor="end",
        textfont=dict(size=9, color="rgba(255,255,255,0.85)"),
        hoverinfo="skip",
    ))
    # Symmetric x-range so bars are balanced around zero
    abs_max = max(abs(v) for v in values) * 1.15
    fig.update_layout(
        height=height,
        margin=dict(l=40, r=8, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[-abs_max, abs_max],
                   zeroline=True, zerolinecolor="rgba(255,255,255,0.15)", zerolinewidth=1),
        yaxis=dict(
            tickfont=dict(size=9, color="rgba(255,255,255,0.6)"),
            automargin=True,
        ),
        bargap=0.30,
    )
    return fig


def _monthly_volume_chart(tx: pd.DataFrame, year: int, height: int = 56):
    """
    Grouped bar chart showing monthly buy vs sell $ volume for a given year.
    tx must have 'date' and 'amount' columns (buys negative, sells positive).
    """
    if tx is None or tx.empty:
        return go.Figure().update_layout(
            height=height, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )

    ytd = tx[tx["date"].dt.year == year].copy()
    if ytd.empty:
        return go.Figure().update_layout(
            height=height, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )

    ytd["month"] = ytd["date"].dt.month
    buys_by_m = ytd[ytd["amount"] < 0].groupby("month")["amount"].apply(lambda s: s.abs().sum())
    sells_by_m = ytd[ytd["amount"] > 0].groupby("month")["amount"].sum()

    # Only show months up to the current month (no empty future bars)
    now = pd.Timestamp.now()
    current_year = now.year
    last_month = now.month if year == current_year else 12
    all_month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    months = list(range(1, last_month + 1))
    month_labels = [all_month_labels[m - 1] for m in months]

    buy_vals = [buys_by_m.get(m, 0) for m in months]
    sell_vals = [sells_by_m.get(m, 0) for m in months]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Buys", x=month_labels, y=buy_vals,
                         marker_color="rgba(239,68,68,0.65)", hoverinfo="skip"))
    fig.add_trace(go.Bar(name="Sells", x=month_labels, y=sell_vals,
                         marker_color="rgba(34,197,94,0.65)", hoverinfo="skip"))
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="group",
        bargap=0.3,
        bargroupgap=0.05,
        showlegend=False,
        xaxis=dict(tickfont=dict(size=8, color="rgba(255,255,255,0.5)"),
                   showgrid=False),
        yaxis=dict(visible=False),
    )
    return fig


# ── Context chips ────────────────────────────────────────────────
def context_chip(label: str, accent: str = "neutral"):
    """Small tag/badge for contextual metadata."""
    color_map = {
        "positive": _GREEN,
        "negative": _RED,
        "accent": _ACCENT,
        "neutral": _NEUTRAL,
        "warning": _WARNING,
        "teal": _TEAL,
        "violet": _VIOLET,
        "amber": _AMBER,
    }
    clr = color_map.get(accent, _NEUTRAL)
    return html.Span(
        label,
        className="story-chip",
        style={
            "color": clr,
            "borderColor": clr,
        },
    )


def warning_chip(label: str):
    """Alert/warning chip with pulsing icon."""
    return html.Span(
        [html.I(className="bi bi-exclamation-triangle-fill me-1"), label],
        className="story-chip story-chip--warning",
        style={"color": _WARNING, "borderColor": _WARNING},
    )


# ── Delta badge ──────────────────────────────────────────────────
def _delta_badge(label: str, current: float, previous: float, mode: str = "ratio"):
    """
    Small badge showing period-over-period change.

    mode="ratio"  → percentage change: (curr-prev)/|prev| * 100  (for $ amounts)
    mode="pp"     → percentage-point diff: (curr-prev) * 100     (for return %)
    """
    if current is None or previous is None:
        return None
    if isinstance(current, float) and np.isnan(current):
        return None
    if isinstance(previous, float) and np.isnan(previous):
        return None

    diff = current - previous

    if mode == "pp":
        # Percentage-point difference (both inputs are decimals like 0.05)
        display_val = abs(diff) * 100
        suffix = "%"
    else:
        # Ratio-based percentage change
        if previous == 0:
            return None
        display_val = abs((diff / abs(previous)) * 100) if abs(previous) > 1e-9 else 0
        suffix = "%"

    arrow = "↑" if diff >= 0 else "↓"
    clr = _GREEN if diff >= 0 else _RED

    return html.Div([
        html.Span(f"{arrow} ", style={"color": clr, "fontWeight": "600"}),
        html.Span(f"{display_val:.1f}{suffix}", style={"color": clr, "fontWeight": "600"}),
        html.Span(f"  {label}", style={"color": "rgba(255,255,255,0.4)"}),
    ], className="story-delta-badge")


# ── Main storytelling card builder ───────────────────────────────
def storytelling_card(
    card_id: str,
    title: str,
    lead_value: str,
    lead_raw: float = None,
    subtitle: str = None,
    narrative: str = None,
    trend_series: pd.Series = None,
    trend_figure: go.Figure = None,
    trend_label: str = None,
    chips: list = None,
    icon: str = None,
    accent: str = None,
    extra_rows: list = None,
    delta: dict = None,
    drill_href: str = None,
):
    """
    Build a magazine-style storytelling card.

    Args:
        card_id:      Unique HTML id
        title:        Card header label (e.g. "Performance Highlights")
        lead_value:   Hero metric string (e.g. "+12.4%")
        lead_raw:     Raw numeric for polarity colouring (optional)
        subtitle:     Secondary descriptor (e.g. "Since Inception, Annualized")
        narrative:    Short "what changed" paragraph
        trend_series: pd.Series for mini sparkline (auto-generates figure)
        trend_figure: Pre-built Plotly figure (takes precedence over trend_series)
        trend_label:  Label under the chart (e.g. "30-Day Trend")
        chips:        List of (label, accent) tuples for context badges
        icon:         Bootstrap icon class (e.g. "bi-graph-up-arrow")
        accent:       Override accent colour key ("positive", "negative", "accent")
        extra_rows:   List of dicts {"label": str, "value": str, "raw": float}
        delta:        Dict {"label": str, "current": float, "previous": float}
        drill_href:   URL path for click-to-drill (e.g. "/performance")
    """
    polarity = accent or ("positive" if (lead_raw is not None and lead_raw >= 0) else
                          "negative" if (lead_raw is not None and lead_raw < 0) else "neutral")
    clr = {"positive": _GREEN, "negative": _RED, "accent": _ACCENT,
           "teal": _TEAL, "violet": _VIOLET, "amber": _AMBER}.get(polarity, _NEUTRAL)

    # ── Header row ──
    header_children = []
    if icon:
        header_children.append(html.I(className=f"{icon} me-2", style={"color": clr, "fontSize": "1.1rem"}))
    header_children.append(html.Span(title, className="story-card-title"))
    # Drill-through arrow hint (desktop only)
    if drill_href:
        header_children.append(html.I(className="bi bi-arrow-right story-drill-arrow"))

    header = html.Div(header_children, className="story-card-header")

    # ── Hero metric (with countUp data attribute for JS animation) ──
    arrow = _arrow(lead_raw)
    hero = html.Div([
        html.Span(f"{arrow} " if arrow else "", className="story-hero-arrow", style={"color": clr}),
        html.Span(
            lead_value,
            className="story-hero-value story-countup",
            style={"color": clr},
            **{"data-final": lead_value},
        ),
    ], className="story-hero")

    if subtitle:
        hero_sub = html.Div(subtitle, className="story-hero-sub")
    else:
        hero_sub = None

    # ── Delta badge ──
    delta_block = None
    if delta:
        delta_block = _delta_badge(
            delta.get("label", ""), delta.get("current"), delta.get("previous"),
            mode=delta.get("mode", "ratio"),
        )

    # ── Mini trend / chart ──
    trend_block = None
    _trend_fig = None
    _trend_height = "48px"

    # Pre-built figure takes precedence
    if trend_figure is not None:
        _trend_fig = trend_figure
        # Extract height from the figure layout if set
        h = getattr(trend_figure, 'layout', None)
        if h and getattr(h, 'height', None):
            _trend_height = f"{h.height}px"
    elif trend_series is not None and not trend_series.empty and len(trend_series) >= 2:
        _trend_fig = _mini_trend_figure(trend_series)

    if _trend_fig is not None:
        trend_block = html.Div([
            dcc.Graph(
                figure=_trend_fig,
                config={"displayModeBar": False, "staticPlot": True},
                style={"height": _trend_height, "width": "100%"},
                className="story-mini-trend-graph",
            ),
            html.Div(trend_label or "", className="story-trend-label"),
        ], className="story-trend-block")

    # ── Extra metric rows ──
    extra_block = None
    if extra_rows:
        rows = []
        for row in extra_rows:
            raw = row.get("raw")
            row_clr = _polarity_color(raw) if raw is not None else _NEUTRAL
            rows.append(html.Div([
                html.Span(row["label"], className="story-extra-label"),
                html.Span(row["value"], className="story-extra-value", style={"color": row_clr}),
            ], className="story-extra-row"))
        extra_block = html.Div(rows, className="story-extra-section")

    # ── Narrative ──
    narrative_block = None
    if narrative:
        narrative_block = html.Div([
            html.Div(className="story-narrative-rule"),
            html.P(narrative, className="story-narrative"),
        ], className="story-narrative-block")

    # ── Chips ──
    chips_block = None
    if chips:
        chip_elements = []
        for item in chips:
            if len(item) == 3 and item[2] == "warning":
                chip_elements.append(warning_chip(item[0]))
            else:
                chip_elements.append(context_chip(item[0], item[1]))
        chips_block = html.Div(chip_elements, className="story-chips")

    # ── Assemble ──
    body_children = [c for c in [header, hero, hero_sub, delta_block, trend_block, extra_block, narrative_block, chips_block] if c is not None]

    card = html.Div(
        body_children,
        id=card_id,
        className=f"story-card story-card--{polarity}",
    )

    # Wrap in click-to-drill link
    if drill_href:
        card = dcc.Link(card, href=drill_href, className="story-card-link")

    return card


# ── Pre-built card factories for common analytics ───────────────

def _safe_twr_curve(data):
    """Get daily TWR curve from cached data, import-free."""
    from dash_wrappers import _get_daily_twr_curve
    try:
        return _get_daily_twr_curve(data)
    except Exception:
        return pd.Series(dtype=float)


def _calc_prior_month_return(data):
    """
    Calculate the prior calendar month's TWR from the daily TWR curve.
    Returns float or None.
    """
    try:
        twr_curve = _safe_twr_curve(data)
        if twr_curve is None or twr_curve.empty or len(twr_curve) < 2:
            return None
        now = pd.Timestamp.now().normalize()
        month_start = pd.Timestamp(now.year, now.month, 1)
        prev_month_start = (month_start - pd.DateOffset(months=1))
        # Get the growth-of-$1 value at boundaries
        before_prev = twr_curve[twr_curve.index < prev_month_start]
        before_curr = twr_curve[twr_curve.index < month_start]
        if before_prev.empty or before_curr.empty:
            return None
        start_val = before_prev.iloc[-1]
        end_val = before_curr.iloc[-1]
        if start_val == 0:
            return None
        return (end_val / start_val) - 1.0
    except Exception:
        return None


def build_performance_story_card(data, metrics, fmt_pct, fmt_dollar):
    """
    Builds a Performance Highlights storytelling card from engine data.
    Includes: delta badge (MTD vs prior month), threshold alerts, drill link.
    """
    sec = data.get("sec_table_current", pd.DataFrame())
    perf = sec[sec["ticker"] != "CASH"] if not sec.empty else pd.DataFrame()

    hero_val = "N/A"
    hero_raw = None
    narrative_parts = []
    extra = []
    chips = []
    delta = None
    trend = None         # will hold a figure (not a pd.Series)
    trend_label = None

    if not perf.empty and "1M" in perf.columns:
        valid = perf.dropna(subset=["1M"])
        if not valid.empty:
            top = valid.loc[valid["1M"].idxmax()]
            bot = valid.loc[valid["1M"].idxmin()]
            hero_val = f"{top['ticker']}  {top['1M']*100:+.2f}%"
            hero_raw = top["1M"]
            extra.append({"label": "Weakest 1M", "value": f"{bot['ticker']}  {bot['1M']*100:+.2f}%", "raw": bot["1M"]})
            narrative_parts.append(
                f"{top['ticker']} led the portfolio this month at {top['1M']*100:+.2f}%, "
                f"while {bot['ticker']} lagged at {bot['1M']*100:+.2f}%."
            )

            # ── Mini bar chart: 1M returns by ticker (sorted) ──
            sorted_perf = valid.sort_values("1M", ascending=True)
            # Cap at 10 for very large portfolios
            if len(sorted_perf) > 10:
                sorted_perf = pd.concat([sorted_perf.head(5), sorted_perf.tail(5)])
            bar_labels = sorted_perf["ticker"].tolist()
            bar_values = (sorted_perf["1M"] * 100).tolist()
            bar_height = max(56, len(bar_labels) * 20)
            trend = _mini_bar_chart(bar_labels, bar_values, height=bar_height)
            trend_label = "1-Month Return by Position"

    # 1D best & worst performer
    if not perf.empty and "1D" in perf.columns:
        valid_1d = perf.dropna(subset=["1D"])
        if not valid_1d.empty:
            top_1d = valid_1d.loc[valid_1d["1D"].idxmax()]
            extra.append({"label": "Best Today", "value": f"{top_1d['ticker']}  {top_1d['1D']*100:+.2f}%", "raw": top_1d["1D"]})
            bot_1d = valid_1d.loc[valid_1d["1D"].idxmin()]
            extra.append({"label": "Worst Today", "value": f"{bot_1d['ticker']}  {bot_1d['1D']*100:+.2f}%", "raw": bot_1d["1D"]})

    # Delta badge: MTD vs prior calendar month return (percentage-point diff)
    mtd = metrics.get("mtd_ret")
    prev_month_ret = _calc_prior_month_return(data)
    if mtd is not None and prev_month_ret is not None:
        delta = {"label": "vs prior month", "current": mtd, "previous": prev_month_ret, "mode": "pp"}

    # MTD chip
    if mtd is not None and not np.isnan(mtd):
        chips.append((f"MTD {mtd*100:+.1f}%", "positive" if mtd >= 0 else "negative"))

    # Position count chip
    pos_count = metrics.get("position_count", 0)
    if pos_count:
        chips.append((f"{pos_count} positions", "accent"))

    # ⚠ Threshold alert: any single holding > 30% of portfolio
    if not perf.empty and "weight" in perf.columns:
        big = perf[perf["weight"] > 0.30]
        if not big.empty:
            ticker = big.iloc[0]["ticker"]
            wt = big.iloc[0]["weight"] * 100
            chips.append((f"{ticker} is {wt:.0f}% of portfolio", "warning", "warning"))

    narrative = " ".join(narrative_parts) if narrative_parts else None

    return storytelling_card(
        card_id="story-performance",
        title="Performance Highlights",
        lead_value=hero_val,
        lead_raw=hero_raw,
        subtitle="Top 1-Month Performer",
        narrative=narrative,
        trend_figure=trend,
        trend_label=trend_label,
        chips=chips,
        icon="bi-graph-up-arrow",
        extra_rows=extra,
        delta=delta,
        drill_href="/performance",
    )


def build_risk_story_card(data, metrics, fmt_pct, fmt_dollar):
    """
    Builds a Risk & Diversification storytelling card.
    Includes: drawdown sparkline, threshold alerts, drill link.
    """
    from dash_wrappers import calculate_efficiency_metrics, calculate_active_metrics, _get_daily_twr_curve

    sec = data.get("sec_table_current", pd.DataFrame())
    sec_no_cash = sec[sec["ticker"] != "CASH"] if not sec.empty else pd.DataFrame()
    holdings = data.get("holdings", pd.DataFrame())

    # Hero: Max Drawdown
    max_dd = metrics.get("max_dd")
    hero_val = f"{max_dd*100:.2f}%" if max_dd is not None and not np.isnan(max_dd) else "N/A"
    hero_raw = max_dd

    extra = []
    chips = []
    narrative_parts = []

    # Sharpe
    sharpe = metrics.get("sharpe", "N/A")
    if sharpe != "N/A":
        extra.append({"label": "Sharpe Ratio", "value": f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else str(sharpe), "raw": None})
        # ⚠ Poor risk-adjusted return
        if isinstance(sharpe, (int, float)) and sharpe < 0.5:
            chips.append(("Low Sharpe", "warning", "warning"))

    # Sortino
    sortino = metrics.get("sortino", "N/A")
    if sortino != "N/A":
        extra.append({"label": "Sortino Ratio", "value": f"{sortino:.2f}" if isinstance(sortino, (int, float)) else str(sortino), "raw": None})

    # Annualized Volatility
    try:
        twr_curve = _get_daily_twr_curve(data)
        eff = calculate_efficiency_metrics(twr_curve)
        vol = eff.get("vol", 0.0)
        if isinstance(vol, (int, float)) and vol > 0:
            extra.append({"label": "Ann. Volatility", "value": f"{vol*100:.1f}%", "raw": None})
    except Exception:
        pass

    # Calmar Ratio (Ann. Return / |Max DD|)
    twr_si = data.get("twr_si_ann")
    if twr_si is None or (isinstance(twr_si, float) and np.isnan(twr_si)):
        twr_si = data.get("twr_si")
    if (twr_si is not None and not (isinstance(twr_si, float) and np.isnan(twr_si))
            and max_dd is not None and not np.isnan(max_dd) and max_dd < 0):
        calmar = twr_si / abs(max_dd)
        extra.append({"label": "Calmar Ratio", "value": f"{calmar:.2f}", "raw": None})

    # Beta vs SPY
    try:
        active = calculate_active_metrics(data)
        beta = active.get("beta", "N/A")
        if beta != "N/A" and isinstance(beta, (int, float)):
            extra.append({"label": "Beta (vs SPY)", "value": f"{beta:.2f}", "raw": None})
    except Exception:
        pass

    # Top-3 concentration
    if not sec_no_cash.empty and "weight" in sec_no_cash.columns:
        top3_pct = sec_no_cash.nlargest(3, "weight")["weight"].sum() * 100
        extra.append({"label": "Top 3 Concentration", "value": f"{top3_pct:.1f}%", "raw": None})
        if top3_pct > 70:
            chips.append((f"High Concentration ({top3_pct:.0f}%)", "warning", "warning"))
            narrative_parts.append(f"Top 3 holdings represent {top3_pct:.1f}% of the portfolio — consider diversification.")
        elif top3_pct > 60:
            chips.append(("Concentrated", "negative"))
        else:
            chips.append(("Diversified", "positive"))

    # Largest asset class
    if not sec_no_cash.empty and "asset_class" in sec_no_cash.columns:
        ac_weights = sec_no_cash.groupby("asset_class")["weight"].sum() * 100
        if not ac_weights.empty:
            largest_ac = ac_weights.idxmax()
            chips.append((f"{largest_ac} {ac_weights.max():.0f}%", "accent"))

    # ⚠ Deep drawdown alert
    if max_dd is not None and not np.isnan(max_dd) and max_dd < -0.15:
        chips.append((f"Drawdown > 15%", "warning", "warning"))

    # Max DD narrative
    if max_dd is not None and not np.isnan(max_dd):
        narrative_parts.insert(0, f"Maximum drawdown of {max_dd*100:.2f}% since inception.")

    narrative = " ".join(narrative_parts) if narrative_parts else None

    # Drawdown sparkline (underwater chart)
    dd_series = None
    try:
        twr_curve = _safe_twr_curve(data)
        if not twr_curve.empty and len(twr_curve) >= 2:
            dd_raw, _, _ = compute_drawdown_series(twr_curve)
            if dd_raw is not None and not dd_raw.empty:
                dd_series = dd_raw.tail(90)  # Last 90 days
    except Exception:
        dd_series = None

    return storytelling_card(
        card_id="story-risk",
        title="Risk & Diversification",
        lead_value=hero_val,
        lead_raw=hero_raw,
        subtitle="Max Drawdown (Since Inception)",
        narrative=narrative,
        trend_series=dd_series,
        trend_label="90-Day Drawdown (Underwater)",
        chips=chips,
        icon="bi-shield-check",
        accent="negative" if (hero_raw is not None and hero_raw < -0.10) else "accent",
        extra_rows=extra,
        drill_href="/risk",
    )


def build_flows_story_card(data, fmt_dollar):
    """
    Builds a Cash Flow Activity storytelling card (internal trades + external flows).
    Includes: delta badge (this month vs last month), threshold alerts, drill link.
    """
    tx = data.get("tx_raw", pd.DataFrame())
    cf_ext = data.get("cf_ext", pd.DataFrame())
    pv = data.get("pv")

    now = pd.Timestamp.now()
    year_start = pd.Timestamp(now.year, 1, 1)
    month_start = pd.Timestamp(now.year, now.month, 1)
    prev_month_start = (month_start - pd.DateOffset(months=1))

    hero_val = "$0"
    hero_raw = 0.0
    extra = []
    chips = []
    narrative_parts = []
    delta = None
    trend_fig = None
    trend_label = None

    # ── External Flows (Since Inception) ──
    has_ext = not cf_ext.empty and "date" in cf_ext.columns and "amount" in cf_ext.columns
    if has_ext:
        deposits_si = cf_ext.loc[cf_ext["amount"] > 0, "amount"].sum()
        withdrawals_si = cf_ext.loc[cf_ext["amount"] < 0, "amount"].sum()
        net_ext_si = cf_ext["amount"].sum()

        extra.append({"label": "Deposits (SI)", "value": fmt_dollar(deposits_si), "raw": deposits_si})
        extra.append({"label": "Withdrawals (SI)", "value": fmt_dollar(withdrawals_si), "raw": withdrawals_si})
        extra.append({"label": "Net External Flow", "value": fmt_dollar(net_ext_si), "raw": net_ext_si})

        flow_count = len(cf_ext)
        chips.append((f"{flow_count} external flow{'s' if flow_count != 1 else ''}", "accent"))

        if net_ext_si > 0:
            narrative_parts.append(f"Net {fmt_dollar(net_ext_si)} deposited since inception.")
        elif net_ext_si < 0:
            narrative_parts.append(f"Net {fmt_dollar(abs(net_ext_si))} withdrawn since inception.")

    # ── Internal Trading (YTD) ──
    if not tx.empty and "date" in tx.columns and "amount" in tx.columns:
        ytd = tx[tx["date"] >= year_start]
        buys = ytd[ytd["amount"] < 0]
        sells = ytd[ytd["amount"] > 0]
        total_bought = buys["amount"].abs().sum() if not buys.empty else 0
        total_sold = sells["amount"].sum() if not sells.empty else 0
        net = total_sold - total_bought

        # Hero: net trading activity YTD
        hero_val = fmt_dollar(net)
        hero_raw = net

        buy_count = len(buys)
        sell_count = len(sells)
        extra.append({"label": "Bought (YTD)", "value": fmt_dollar(total_bought), "raw": None})
        extra.append({"label": "Sold (YTD)", "value": fmt_dollar(total_sold), "raw": total_sold})

        total_trades = buy_count + sell_count
        chips.append((f"{total_trades} trades YTD", "accent"))
        if buy_count:
            chips.append((f"{buy_count} buys", "neutral"))
        if sell_count:
            chips.append((f"{sell_count} sells", "neutral"))

        # Delta badge: this month vs last month activity (sum of absolute values)
        this_month = tx[(tx["date"] >= month_start)]
        last_month = tx[(tx["date"] >= prev_month_start) & (tx["date"] < month_start)]
        this_month_vol = this_month["amount"].abs().sum() if not this_month.empty else 0
        last_month_vol = last_month["amount"].abs().sum() if not last_month.empty else 0
        if last_month_vol > 0:
            delta = {"label": "vs last month volume", "current": this_month_vol, "previous": last_month_vol}

        # ⚠ High turnover alert (standard: min(buys, sells) / avg portfolio value)
        if pv is not None and not pv.empty:
            ytd_pv = pv[pv.index >= year_start]
            avg_mv = ytd_pv.mean() if not ytd_pv.empty else pv.iloc[-1]
            if avg_mv > 0:
                turnover = min(total_bought, total_sold) / avg_mv
                turnover_pct = turnover * 100
                if turnover > 0.50:
                    chips.append((f"\u26A0 High Turnover ({turnover_pct:.0f}%)", "warning", "warning"))

        if net > 0:
            narrative_parts.append(f"Net selling activity of {fmt_dollar(net)} year-to-date — proceeds exceeded purchases.")
        elif net < 0:
            narrative_parts.append(f"Net buying activity of {fmt_dollar(abs(net))} year-to-date — deploying capital into positions.")
        else:
            narrative_parts.append("No net internal trading activity year-to-date.")

        # ── Mini chart: monthly buy vs sell volume ──
        trend_fig = _monthly_volume_chart(tx, now.year, height=56)
        trend_label = "Monthly Buy / Sell Volume (YTD)"
    else:
        narrative_parts.append("No internal transactions recorded.")

    narrative = " ".join(narrative_parts) if narrative_parts else None

    return storytelling_card(
        card_id="story-flows",
        title="Cash Flow Activity",
        lead_value=hero_val,
        lead_raw=hero_raw,
        subtitle="Net Trading Activity (YTD)",
        narrative=narrative,
        trend_figure=trend_fig,
        trend_label=trend_label,
        chips=chips,
        icon="bi-arrow-left-right",
        extra_rows=extra,
        delta=delta,
        drill_href="/flows",
    )


# ── Card 4: Tax Efficiency ──────────────────────────────────────

def build_tax_efficiency_story_card(data, fmt_dollar):
    """
    Builds a Tax Efficiency storytelling card using FIFO lot-building.
    Hero: total unrealized gain/loss. Sparkline: unrealized P/L by top holdings.
    """
    try:
        from tax_engine import build_tax_lots

        # Build open lots via FIFO (uses cached data)
        open_lots, _ = build_tax_lots(strategy="FIFO")

        if open_lots is None or open_lots.empty or "Unrealized P/L" not in open_lots.columns:
            return _tax_fallback_card()

        # ── Aggregate metrics ──
        total_unrealized = open_lots["Unrealized P/L"].sum()

        # Term classification
        has_term = "Term" in open_lots.columns
        if has_term:
            st_lots = open_lots[open_lots["Term"] == "Short-Term"]
            lt_lots = open_lots[open_lots["Term"] == "Long-Term"]
            n_st = len(st_lots)
            n_lt = len(lt_lots)
            st_unrealized = st_lots["Unrealized P/L"].sum()
            lt_unrealized = lt_lots["Unrealized P/L"].sum()
        else:
            n_st, n_lt = 0, 0
            st_unrealized, lt_unrealized = 0.0, 0.0

        # Hero
        hero_val = fmt_dollar(total_unrealized)
        hero_raw = total_unrealized

        # Subtitle
        subtitle = f"{n_st} short-term lot{'s' if n_st != 1 else ''}, {n_lt} long-term lot{'s' if n_lt != 1 else ''}"

        # ── Narrative ──
        narrative_parts = []
        if total_unrealized >= 0:
            if lt_unrealized > st_unrealized and lt_unrealized > 0:
                narrative_parts.append(
                    "Most of your unrealized gains are long-term, which means lower tax rates if you sell."
                )
            elif st_unrealized > lt_unrealized and st_unrealized > 0:
                narrative_parts.append(
                    "Heads up — the majority of your unrealized gains are short-term. "
                    "Selling now would trigger higher ordinary-income tax rates."
                )
            else:
                narrative_parts.append(
                    "Unrealized gains are balanced between short- and long-term lots."
                )
        else:
            narrative_parts.append(
                "Your portfolio is sitting on net unrealized losses — "
                "consider tax-loss harvesting to offset gains elsewhere."
            )

        # Add dollar context
        if has_term:
            narrative_parts.append(
                f"Long-term: {fmt_dollar(lt_unrealized)}.  Short-term: {fmt_dollar(st_unrealized)}."
            )

        narrative = " ".join(narrative_parts)

        # ── Sparkline: unrealized P/L by ticker (top holdings) ──
        ticker_pl = open_lots.groupby("Ticker")["Unrealized P/L"].sum().sort_values()
        # Cap at 10 for readability
        if len(ticker_pl) > 10:
            ticker_pl = pd.concat([ticker_pl.head(5), ticker_pl.tail(5)])
        bar_labels = ticker_pl.index.tolist()
        bar_values = [round(v, 2) for v in ticker_pl.values]
        # Convert to % of cost basis for better chart readability
        ticker_cost = open_lots.groupby("Ticker")["Cost Basis"].sum()
        bar_pct_values = []
        for t, upl in zip(bar_labels, bar_values):
            cb = ticker_cost.get(t, 1)
            pct = (upl / cb * 100) if abs(cb) > 0.01 else 0
            bar_pct_values.append(round(pct, 1))

        bar_height = max(56, len(bar_labels) * 20)
        trend_fig = _mini_bar_chart(bar_labels, bar_pct_values, height=bar_height)
        trend_label = "Unrealized P/L % by Position"

        # ── Extra rows ──
        extra = []
        extra.append({"label": "LT Unrealized", "value": fmt_dollar(lt_unrealized), "raw": lt_unrealized})
        extra.append({"label": "ST Unrealized", "value": fmt_dollar(st_unrealized), "raw": st_unrealized})

        # Estimated tax liability if all gains sold
        if "Est Tax Liability" in open_lots.columns:
            est_tax = open_lots["Est Tax Liability"].sum()
            extra.append({"label": "Est Tax if Sold", "value": fmt_dollar(est_tax), "raw": -est_tax})

        # Near-cliff lots
        if "Is Near Cliff" in open_lots.columns:
            cliff_count = open_lots["Is Near Cliff"].sum()
            if cliff_count > 0:
                extra.append({"label": "Near LT Cliff", "value": f"{int(cliff_count)} lot{'s' if cliff_count != 1 else ''}", "raw": None})

        # ── Chips ──
        chips = []
        if n_st > 0:
            st_color = "negative" if st_unrealized > 0 else "teal"
            chips.append((f"{n_st} short-term", st_color))
        if n_lt > 0:
            lt_color = "positive" if lt_unrealized >= 0 else "negative"
            chips.append((f"{n_lt} long-term", lt_color))

        # Near-cliff warning
        if "Is Near Cliff" in open_lots.columns:
            cliff_count = int(open_lots["Is Near Cliff"].sum())
            if cliff_count > 0:
                chips.append((f"{cliff_count} lot{'s' if cliff_count != 1 else ''} near LT cliff", "warning", "warning"))

        # Wash sale flag
        if "Is Wash Sale" in open_lots.columns and open_lots.get("Is Wash Sale", pd.Series()).any():
            chips.append(("Wash sale adjustments applied", "warning", "warning"))

        return storytelling_card(
            card_id="story-tax-efficiency",
            title="Tax Efficiency",
            lead_value=hero_val,
            lead_raw=hero_raw,
            subtitle=subtitle,
            narrative=narrative,
            trend_figure=trend_fig,
            trend_label=trend_label,
            chips=chips,
            icon="bi-receipt",
            accent="teal",
            extra_rows=extra,
            drill_href="/taxes",
        )

    except Exception:
        return _tax_fallback_card()


def _tax_fallback_card():
    """Graceful fallback when tax lot data is unavailable."""
    return storytelling_card(
        card_id="story-tax-efficiency",
        title="Tax Efficiency",
        lead_value="N/A",
        subtitle="Tax lot data unavailable",
        narrative="Unable to build tax lots — check that transaction history is loaded.",
        icon="bi-receipt",
        accent="teal",
        drill_href="/taxes",
    )


# ── Card 5: Momentum & Trend ────────────────────────────────────

def build_momentum_story_card(data):
    """
    Builds a Momentum & Trend storytelling card.
    Hero: % of holdings above 200-day MA.
    Sparkline: distance from 200-day MA by ticker.
    """
    try:
        prices = data.get("prices", pd.DataFrame())
        sec = data.get("sec_table_current", pd.DataFrame())

        if prices is None or prices.empty or sec is None or sec.empty:
            return _momentum_fallback_card()

        # Only non-cash equity tickers
        tickers = sec[sec["ticker"] != "CASH"]["ticker"].unique().tolist()
        if not tickers:
            return _momentum_fallback_card()

        ma_distances = {}  # ticker -> % distance from 200-day MA
        above_count = 0
        total_count = 0

        for t in tickers:
            if t not in prices.columns:
                continue
            series = prices[t].dropna()
            if len(series) < 50:
                # Need reasonable history; skip if too short
                continue
            total_count += 1

            # Calculate 200-day MA (or use all available if < 200 days)
            ma_window = min(200, len(series))
            ma_200 = series.iloc[-ma_window:].mean()

            if ma_200 == 0 or pd.isna(ma_200):
                continue

            current_price = series.iloc[-1]
            pct_distance = ((current_price - ma_200) / ma_200) * 100
            ma_distances[t] = round(pct_distance, 1)

            if current_price > ma_200:
                above_count += 1

        if total_count == 0:
            return _momentum_fallback_card()

        pct_above = (above_count / total_count) * 100
        below_count = total_count - above_count

        # Hero
        hero_val = f"{pct_above:.0f}%"
        hero_raw = pct_above / 100  # normalize for polarity

        # Subtitle
        subtitle = f"{above_count} of {total_count} holdings above 200-day MA"

        # ── Narrative ──
        if pct_above >= 75:
            narrative = (
                "Strong bullish momentum — the vast majority of your holdings are trading "
                "above their 200-day moving averages. Trend conditions favor staying invested."
            )
        elif pct_above >= 50:
            narrative = (
                "Mixed signals — roughly half your portfolio is in an uptrend. "
                "Monitor the laggards for potential weakness, but the overall picture is constructive."
            )
        elif pct_above >= 25:
            narrative = (
                "Caution warranted — most holdings have slipped below their 200-day moving averages. "
                "Broad downtrend conditions suggest reviewing risk exposure."
            )
        else:
            narrative = (
                "Bearish momentum across the board — very few holdings are above their long-term trend line. "
                "Consider defensive positioning or tightening stop-losses."
            )

        # ── Sparkline: MA distance by ticker (sorted) ──
        if ma_distances:
            sorted_ma = dict(sorted(ma_distances.items(), key=lambda x: x[1]))
            # Cap at 10
            items = list(sorted_ma.items())
            if len(items) > 10:
                items = items[:5] + items[-5:]
            bar_labels = [t for t, _ in items]
            bar_values = [v for _, v in items]
            bar_height = max(56, len(bar_labels) * 20)
            trend_fig = _mini_bar_chart(bar_labels, bar_values, height=bar_height)
            trend_label = "Distance from 200-Day MA (%)"
        else:
            trend_fig = None
            trend_label = None

        # ── Extra rows ──
        extra = []
        if ma_distances:
            strongest = max(ma_distances, key=ma_distances.get)
            weakest = min(ma_distances, key=ma_distances.get)
            extra.append({"label": "Strongest", "value": f"{strongest}  {ma_distances[strongest]:+.1f}%", "raw": ma_distances[strongest]})
            extra.append({"label": "Weakest", "value": f"{weakest}  {ma_distances[weakest]:+.1f}%", "raw": ma_distances[weakest]})

        # ── Chips ──
        chips = []
        chips.append((f"{above_count} above MA", "positive" if above_count > below_count else "neutral"))
        chips.append((f"{below_count} below MA", "negative" if below_count >= above_count else "neutral"))

        # Highlight strongest and weakest by MA distance
        if ma_distances:
            strongest = max(ma_distances, key=ma_distances.get)
            weakest = min(ma_distances, key=ma_distances.get)
            chips.append((f"{strongest} {ma_distances[strongest]:+.0f}%", "violet"))
            chips.append((f"{weakest} {ma_distances[weakest]:+.0f}%", "negative"))

        # Choose accent by overall sentiment
        if pct_above >= 60:
            accent = "positive"
        elif pct_above <= 40:
            accent = "negative"
        else:
            accent = "violet"

        return storytelling_card(
            card_id="story-momentum",
            title="Momentum & Trend",
            lead_value=hero_val,
            lead_raw=hero_raw,
            subtitle=subtitle,
            narrative=narrative,
            trend_figure=trend_fig,
            trend_label=trend_label,
            chips=chips,
            icon="bi-arrow-up-right",
            accent=accent,
            extra_rows=extra,
            drill_href="/risk",
        )

    except Exception:
        return _momentum_fallback_card()


def _momentum_fallback_card():
    """Graceful fallback when momentum data is unavailable."""
    return storytelling_card(
        card_id="story-momentum",
        title="Momentum & Trend",
        lead_value="N/A",
        subtitle="Insufficient price history",
        narrative="Unable to calculate moving averages — need at least 50 days of price data.",
        icon="bi-arrow-up-right",
        accent="violet",
        drill_href="/risk",
    )


# ── Card 6: Rebalancing Health ──────────────────────────────────

def build_rebalancing_story_card(data, fmt_dollar):
    """
    Builds a Rebalancing Health storytelling card.
    Hero: total absolute drift. Sparkline: drift per holding.
    """
    try:
        sec = data.get("sec_table_current", pd.DataFrame())
        holdings = data.get("holdings", pd.DataFrame())

        if sec is None or sec.empty:
            return _rebalancing_fallback_card()

        invested = sec[sec["ticker"] != "CASH"].copy()
        if invested.empty:
            return _rebalancing_fallback_card()

        # Ensure weight and target_pct columns exist
        if "weight" not in invested.columns or "target_pct" not in invested.columns:
            # Try to merge from holdings
            if not holdings.empty and "target_pct" in holdings.columns:
                invested = invested.merge(
                    holdings[["ticker", "target_pct"]], on="ticker", how="left", suffixes=("", "_h")
                )
                if "target_pct_h" in invested.columns:
                    invested["target_pct"] = invested["target_pct_h"].fillna(invested.get("target_pct", 0))
                    invested.drop(columns=["target_pct_h"], inplace=True)
            if "weight" not in invested.columns or "target_pct" not in invested.columns:
                return _rebalancing_fallback_card()

        invested["target_pct"] = invested["target_pct"].fillna(0.0)

        # Current weight in percentage points (0-100 scale)
        invested["current_pct"] = invested["weight"] * 100

        # Drift = current weight - target weight (percentage points)
        invested["drift"] = invested["current_pct"] - invested["target_pct"]

        # Total absolute drift
        total_abs_drift = invested["drift"].abs().sum()

        # Identify most over- and under-weight
        most_over_idx = invested["drift"].idxmax()
        most_under_idx = invested["drift"].idxmin()
        most_over = invested.loc[most_over_idx]
        most_under = invested.loc[most_under_idx]

        # Hero
        hero_val = f"{total_abs_drift:.1f}%"
        hero_raw = -total_abs_drift  # negative = higher drift = worse

        # Subtitle
        subtitle = (
            f"Most over: {most_over['ticker']} ({most_over['drift']:+.1f}pp)  ·  "
            f"Most under: {most_under['ticker']} ({most_under['drift']:+.1f}pp)"
        )

        # ── Narrative ──
        if total_abs_drift < 5:
            narrative = (
                "Your portfolio is well-balanced — drift from targets is minimal. "
                "No rebalancing action needed at this time."
            )
        elif total_abs_drift < 15:
            narrative = (
                "Moderate drift detected across a few positions. "
                "Consider a light rebalance on your next deployment of cash to bring weights back in line."
            )
        elif total_abs_drift < 30:
            narrative = (
                "Several positions have drifted significantly from their targets. "
                "A rebalancing trade is recommended to restore your intended allocation."
            )
        else:
            narrative = (
                "Major allocation drift — your portfolio looks quite different from your target model. "
                "Strongly recommend rebalancing soon to control unintended risk exposures."
            )

        # ── Sparkline: drift per holding (sorted) ──
        drift_sorted = invested[["ticker", "drift"]].sort_values("drift")
        if len(drift_sorted) > 10:
            drift_sorted = pd.concat([drift_sorted.head(5), drift_sorted.tail(5)])
        bar_labels = drift_sorted["ticker"].tolist()
        bar_values = [round(v, 1) for v in drift_sorted["drift"].values]
        bar_height = max(56, len(bar_labels) * 20)
        trend_fig = _mini_bar_chart(bar_labels, bar_values, height=bar_height)
        trend_label = "Weight Drift (Current − Target, pp)"

        # ── Extra rows ──
        extra = []
        extra.append({
            "label": f"{most_over['ticker']} Weight",
            "value": f"{most_over['current_pct']:.1f}% (target {most_over['target_pct']:.0f}%)",
            "raw": most_over["drift"],
        })
        extra.append({
            "label": f"{most_under['ticker']} Weight",
            "value": f"{most_under['current_pct']:.1f}% (target {most_under['target_pct']:.0f}%)",
            "raw": most_under["drift"],
        })

        # Positions with 0% target but nonzero weight (unplanned exposure)
        no_target = invested[(invested["target_pct"] == 0) & (invested["current_pct"] > 0.5)]
        if not no_target.empty:
            tickers_str = ", ".join(no_target["ticker"].tolist()[:3])
            extra.append({"label": "No Target Set", "value": tickers_str, "raw": None})

        # ── Chips ──
        chips = []
        # Drift severity chip
        if total_abs_drift < 5:
            chips.append(("On Target", "positive"))
        elif total_abs_drift < 15:
            chips.append(("Moderate Drift", "amber"))
        else:
            chips.append(("High Drift", "warning", "warning"))

        # Individual position chips for biggest offenders
        top_offenders = invested.reindex(invested["drift"].abs().nlargest(2).index)
        for _, row in top_offenders.iterrows():
            direction = "over" if row["drift"] > 0 else "under"
            color = "negative" if abs(row["drift"]) > 5 else "amber"
            chips.append((f"{row['ticker']} {row['drift']:+.1f}pp {direction}", color))

        # No-target positions warning
        if not no_target.empty:
            chips.append((f"{len(no_target)} position{'s' if len(no_target) != 1 else ''} w/o target", "warning", "warning"))

        return storytelling_card(
            card_id="story-rebalancing",
            title="Rebalancing Health",
            lead_value=hero_val,
            lead_raw=hero_raw,
            subtitle=subtitle,
            narrative=narrative,
            trend_figure=trend_fig,
            trend_label=trend_label,
            chips=chips,
            icon="bi-sliders",
            accent="amber",
            extra_rows=extra,
            drill_href="/rebalancing",
        )

    except Exception:
        return _rebalancing_fallback_card()


def _rebalancing_fallback_card():
    """Graceful fallback when rebalancing data is unavailable."""
    return storytelling_card(
        card_id="story-rebalancing",
        title="Rebalancing Health",
        lead_value="N/A",
        subtitle="Target allocation data unavailable",
        narrative="Set target percentages in your holdings file to enable drift analysis.",
        icon="bi-sliders",
        accent="amber",
        drill_href="/rebalancing",
    )
