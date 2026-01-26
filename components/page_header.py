"""
Page Header Component

Provides a consistent header component for all pages with:
- Title
- Subtitle (optional)
- Breadcrumb navigation (optional)
- Right-aligned action buttons (optional)
"""

from dash import html
import dash_bootstrap_components as dbc


def page_header(
    title: str,
    subtitle: str = None,
    breadcrumb: list = None,
    actions: list = None,
    icon: str = None
):
    """
    Creates a standardized page header with optional subtitle, breadcrumb, and actions.
    
    Args:
        title: Main page title
        subtitle: Optional descriptive subtitle
        breadcrumb: Optional list of tuples [(label, href), ...] for breadcrumb trail
        actions: Optional list of Dash components (buttons, etc.) for right side
        icon: Optional Bootstrap icon class (e.g., "bi bi-house") to display inline with title
    
    Returns:
        html.Div with page-header class
    
    Example:
        page_header(
            title="Performance",
            subtitle="Portfolio returns and benchmark comparison",
            breadcrumb=[("Home", "/"), ("Performance", None)],
            actions=[dbc.Button("Export", color="primary", size="sm")],
            icon="bi bi-graph-up"
        )
    """
    header_content = []
    
    # Left side: Title, subtitle, breadcrumb
    left_content = []
    
    # Breadcrumb
    if breadcrumb:
        crumb_items = []
        for i, (label, href) in enumerate(breadcrumb):
            is_last = i == len(breadcrumb) - 1
            if href and not is_last:
                crumb_items.append(
                    html.A(label, href=href, className="text-decoration-none")
                )
                crumb_items.append(html.Span(" / ", className="mx-2 text-muted"))
            else:
                crumb_items.append(html.Span(label, className="text-muted"))
        
        left_content.append(
            html.Div(crumb_items, className="breadcrumb mb-1")
        )
    
    # Title with optional Icon
    title_components = []
    if icon:
        title_components.append(html.I(className=f"{icon} page-title-icon me-3"))
    title_components.append(html.Span(title))
    
    left_content.append(html.H2(title_components, className="mb-0 d-flex align-items-center"))
    
    # Subtitle
    if subtitle:
        left_content.append(
            html.P(subtitle, className="subtitle text-muted mb-0")
        )
    
    header_content.append(html.Div(left_content, className="flex-grow-1"))
    
    # Right side: Actions
    if actions:
        header_content.append(
            html.Div(
                actions,
                className="d-flex align-items-center gap-2"
            )
        )
    
    return html.Div(
        header_content,
        className="page-header d-flex align-items-start justify-content-between"
    )


def create_skeleton_loader(type: str = "card", count: int = 1):
    """
    Creates skeleton loading placeholders.
    
    Args:
        type: "card", "chart", "table", "text", "kpi"
        count: Number of skeleton elements to create
    
    Returns:
        html.Div or list of skeleton elements
    """
    skeletons = []
    
    for _ in range(count):
        if type == "card":
            skeletons.append(
                html.Div(className="skeleton skeleton-card mb-3")
            )
        elif type == "chart":
            skeletons.append(
                html.Div(className="skeleton skeleton-chart mb-3")
            )
        elif type == "table":
            skeletons.append(
                html.Div(className="skeleton skeleton-table mb-3")
            )
        elif type == "text":
            skeletons.append(
                html.Div([
                    html.Div(className="skeleton skeleton-text"),
                    html.Div(className="skeleton skeleton-text"),
                    html.Div(className="skeleton skeleton-text-sm"),
                ], className="mb-3")
            )
        elif type == "kpi":
            skeletons.append(
                dbc.Card(
                    dbc.CardBody([
                        html.Div(className="skeleton skeleton-text-sm mb-2", style={"width": "60%"}),
                        html.Div(className="skeleton skeleton-title", style={"width": "80%"}),
                        html.Div(className="skeleton skeleton-text-sm", style={"width": "40%"}),
                    ], className="p-2"),
                    className="shadow-sm"
                )
            )
    
    if count == 1:
        return skeletons[0]
    
    return html.Div(skeletons)
