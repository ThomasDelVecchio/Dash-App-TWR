from dash import html


def create_kpi_card(title, value, subtext=None, is_positive=None, accent=None):
    """
    Glassmorphism KPI card with frosted-glass surface, animated gradient
    border, and inner glow that shifts colour based on positive / negative.

    Args:
        title:       Label text (e.g. "Inception TWR")
        value:       Formatted value string (e.g. "+12.4%")
        subtext:     Optional secondary line (e.g. "Ann. CAGR")
        is_positive: True  → green glow / border
                     False → red glow / border
                     None  → neutral cyan glow
        accent:      Override glow flavour: "blue" | "green" (used for Sharpe / Sortino)
    """
    # ---- normalise is_positive to a plain Python bool (numpy.bool_ fails `is`) ----
    if is_positive is not None:
        is_positive = bool(is_positive)

    # ---- determine colour classes ----
    if accent == "blue":
        glass_mod = "kpi-glass--accent-blue"
        wrapper_mod = "kpi-glass-wrapper--accent-blue"
        value_color = "#6ea8fe"  # Bootstrap primary-light
        sub_color = "rgba(110,168,254,0.7)"
    elif accent == "green":
        glass_mod = "kpi-glass--accent-green"
        wrapper_mod = "kpi-glass-wrapper--accent-green"
        value_color = "#75b798"  # Bootstrap success-light
        sub_color = "rgba(117,183,152,0.7)"
    elif is_positive is True:
        glass_mod = "kpi-glass--positive"
        wrapper_mod = "kpi-glass-wrapper--positive"
        value_color = "#22c55e"
        sub_color = "rgba(34,197,94,0.75)"
    elif is_positive is False:
        glass_mod = "kpi-glass--negative"
        wrapper_mod = "kpi-glass-wrapper--negative"
        value_color = "#ef4444"
        sub_color = "rgba(239,68,68,0.75)"
    else:
        glass_mod = "kpi-glass--neutral"
        wrapper_mod = ""
        value_color = "#f8f9fa"
        sub_color = "rgba(248,249,250,0.45)"

    # ---- arrow indicator ----
    arrow_span = None
    if is_positive is not None:
        symbol = "▲" if is_positive else "▼"
        arrow_span = html.Span(
            f"{symbol} ",
            style={
                "color": value_color,
                "fontSize": "1rem",
                "marginRight": "3px",
                "verticalAlign": "middle",
            },
        )

    value_children = [arrow_span, value] if arrow_span else value

    # ---- subtext line (always present to keep height stable) ----
    if subtext:
        sub_prefix = ("▲ " if is_positive else "▼ ") if is_positive is not None else ""
        sub_el = html.Div(
            f"{sub_prefix}{subtext}",
            className="kpi-glass-sub",
            style={"color": sub_color},
        )
    else:
        sub_el = html.Div("\u00a0", className="kpi-glass-sub", style={"color": "transparent"})

    # ---- assemble ----
    inner = html.Div(
        [
            html.Div(title, className="kpi-glass-label"),
            html.Div(
                value_children,
                className="kpi-glass-value",
                style={"color": value_color},
            ),
            sub_el,
        ],
        className=f"kpi-glass {glass_mod}",
    )

    return html.Div(inner, className=f"kpi-glass-wrapper {wrapper_mod}")
