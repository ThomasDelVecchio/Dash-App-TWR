"""
Trade Execution Page

Standalone execution hub for placing orders via E*TRADE API.
Supports:
- Pre-filled form from staged orders (via staged-order-store)
- Manual order entry
- Lot picker for sells (FIFO/LIFO/HIFO or manual selection)
- Order preview with estimated costs
- Confirmation modal before execution
- Sandbox/Live environment indicator
"""

import dash
from dash import dcc, html, Input, Output, State, callback, callback_context, ALL, MATCH
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import pandas as pd
import numpy as np
import threading
from datetime import datetime

# Local Imports
import dash_wrappers as dw
from components.page_header import page_header
from config import ETRADE_SANDBOX, is_etrade_configured, TAX_RATE_ST, TAX_RATE_LT
from tax_engine import build_tax_lots, simulate_sell
from etrade_orders import (
    get_environment_info,
    preview_order,
    place_order,
    get_order_status,
    cancel_order,
    get_recent_orders,
    fetch_position_lots
)
from report_formatting import fmt_dollar_clean, fmt_pct_clean

# ============================================================
# HELPER: Build Environment Badge
# ============================================================

def build_environment_badge():
    """Creates a badge indicating Sandbox vs Production mode."""
    env_info = get_environment_info()
    
    if not env_info.get("configured", False):
        return dbc.Badge(
            [html.I(className="bi bi-exclamation-triangle me-1"), "E*TRADE Not Configured"],
            color="warning",
            className="me-2"
        )
    
    if env_info.get("is_sandbox", True):
        return dbc.Badge(
            [html.I(className="bi bi-box me-1"), "SANDBOX MODE"],
            color="info",
            className="me-2",
            id="env-badge"
        )
    else:
        return dbc.Badge(
            [html.I(className="bi bi-lightning-charge me-1"), "LIVE TRADING"],
            color="danger",
            className="me-2",
            id="env-badge"
        )

# ============================================================
# HELPER: Build Ticker Suggestions (Owned Holdings)
# ============================================================

def get_owned_tickers():
    """Get list of tickers from current holdings for autocomplete suggestions."""
    try:
        data = dw.get_data()
        if not data:
            return []
        
        sec_table = data.get("sec_table_current", pd.DataFrame())
        if sec_table.empty:
            return []
        
        tickers = sec_table[sec_table["ticker"] != "CASH"]["ticker"].unique().tolist()
        return sorted(tickers)
    except:
        return []


def validate_ticker(symbol: str) -> dict:
    """
    Validate a ticker symbol using yfinance.
    
    Returns:
        dict with keys:
            - valid: bool
            - message: str (error message if invalid)
            - price: float (current price if valid)
            - name: str (company name if valid)
    """
    if not symbol or not symbol.strip():
        return {"valid": False, "message": "Please enter a ticker symbol", "price": None, "name": None}
    
    symbol = symbol.strip().upper()
    
    # Quick format check
    if not symbol.isalnum() or len(symbol) > 10:
        return {"valid": False, "message": f"'{symbol}' is not a valid ticker format", "price": None, "name": None}
    
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        
        # Try to get current price - this confirms the ticker exists
        hist = ticker.history(period="1d")
        
        if hist.empty:
            # Fallback: try info (slower but more reliable for some tickers)
            info = ticker.info
            if not info or info.get("regularMarketPrice") is None:
                return {"valid": False, "message": f"'{symbol}' not found on market", "price": None, "name": None}
            
            price = info.get("regularMarketPrice", 0)
            name = info.get("shortName", symbol)
        else:
            price = float(hist["Close"].iloc[-1])
            # Get name from fast_info if available
            try:
                name = ticker.info.get("shortName", symbol)
            except:
                name = symbol
        
        return {"valid": True, "message": "", "price": price, "name": name}
        
    except Exception as e:
        return {"valid": False, "message": f"Could not validate '{symbol}': {str(e)[:50]}", "price": None, "name": None}

# ============================================================
# LAYOUT
# ============================================================

layout = dbc.Container([
    # Header Row
    page_header(
        title="Trade Execution",
        icon="bi-cart-check",
        subtitle="Place and manage orders via E*TRADE",
        actions=[
            build_environment_badge(),
            dbc.Button(
                [html.I(className="bi bi-arrow-clockwise me-1"), "Refresh"],
                id="btn-refresh-trade",
                color="secondary",
                size="sm"
            )
        ]
    ),
    
    # Alert for staged order notification
    dbc.Alert(
        id="staged-order-alert",
        is_open=False,
        dismissable=True,
        color="info",
        className="mb-3"
    ),
    dbc.Button(
        [html.I(className="bi bi-skip-forward me-2"), "Next Staged Order"],
        id="btn-next-staged-order",
        color="primary",
        size="sm",
        disabled=True,
        className="mb-4"
    ),
    
    # Main Content: Order Form + Preview
    dbc.Row([
        # Left Column: Order Entry Form
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="bi bi-pencil-square me-2"),
                    "Order Entry"
                ]),
                dbc.CardBody([
                    # Action (Buy/Sell)
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Action", className="fw-bold"),
                            dbc.RadioItems(
                                id="trade-action",
                                options=[
                                    {"label": "Buy", "value": "BUY"},
                                    {"label": "Sell", "value": "SELL"},
                                ],
                                value="BUY",
                                inline=True,
                                className="mb-3"
                            ),
                        ], width=12),
                    ]),
                    
                    # Ticker Selection
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Ticker", className="fw-bold"),
                            # Datalist provides autocomplete suggestions from owned tickers
                            html.Datalist(
                                id="ticker-suggestions",
                                children=[html.Option(value=t) for t in get_owned_tickers()]
                            ),
                            dbc.Input(
                                id="trade-ticker",
                                type="text",
                                placeholder="Enter any ticker symbol...",
                                list="ticker-suggestions",  # Links to datalist for autocomplete
                                debounce=True,  # Triggers callback after typing stops
                                className="mb-1",
                                style={"backgroundColor": "#333", "color": "#fff", "border": "1px solid #555"}
                            ),
                            # Validation error/success message
                            html.Div(id="ticker-validation-msg", className="small mb-2"),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Quantity", className="fw-bold"),
                            dbc.Input(
                                id="trade-quantity",
                                type="number",
                                min=1,
                                step=1,
                                placeholder="Shares",
                                className="mb-3"
                            ),
                        ], width=6),
                    ]),
                    
                    # Order Type
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Order Type", className="fw-bold"),
                            dbc.Select(
                                id="trade-order-type",
                                options=[
                                    {"label": "Market", "value": "MARKET"},
                                    {"label": "Limit", "value": "LIMIT"},
                                    {"label": "Stop", "value": "STOP"},
                                    {"label": "Stop Limit", "value": "STOP_LIMIT"},
                                ],
                                value="MARKET",
                                className="mb-3"
                            ),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Duration", className="fw-bold"),
                            dbc.Select(
                                id="trade-duration",
                                options=[
                                    {"label": "Day", "value": "GOOD_FOR_DAY"},
                                    {"label": "GTC (60 Days)", "value": "GOOD_UNTIL_CANCEL"},
                                    {"label": "Fill or Kill", "value": "FILL_OR_KILL"},
                                    {"label": "Immediate or Cancel", "value": "IMMEDIATE_OR_CANCEL"},
                                ],
                                value="GOOD_FOR_DAY",
                                className="mb-3"
                            ),
                        ], width=6),
                    ]),
                    
                    # Price Fields (conditional)
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Limit Price", className="fw-bold"),
                            dbc.Input(
                                id="trade-limit-price",
                                type="number",
                                min=0.01,
                                step=0.01,
                                placeholder="$0.00",
                                disabled=True,
                                className="mb-3"
                            ),
                        ], width=6, id="limit-price-col"),
                        dbc.Col([
                            dbc.Label("Stop Price", className="fw-bold"),
                            dbc.Input(
                                id="trade-stop-price",
                                type="number",
                                min=0.01,
                                step=0.01,
                                placeholder="$0.00",
                                disabled=True,
                                className="mb-3"
                            ),
                        ], width=6, id="stop-price-col"),
                    ]),
                    
                    # Lot Selection (for sells only)
                    html.Div([
                        html.Hr(),
                        dbc.Label([
                            html.I(className="bi bi-layers me-1"),
                            "Tax Lot Selection"
                        ], className="fw-bold"),
                        dbc.RadioItems(
                            id="lot-strategy",
                            options=[
                                {"label": "FIFO (First In, First Out)", "value": "FIFO"},
                                {"label": "LIFO (Last In, First Out)", "value": "LIFO"},
                                {"label": "HIFO (Highest Cost First)", "value": "HIFO"},
                                {"label": "Specific Lots (Manual)", "value": "MANUAL"},
                            ],
                            value="FIFO",
                            className="mb-2"
                        ),
                        dbc.Button(
                            [html.I(className="bi bi-list-ul me-1"), "Select Specific Lots"],
                            id="btn-open-lot-picker",
                            color="outline-secondary",
                            size="sm",
                            disabled=True,
                            className="mb-3"
                        ),
                    ], id="lot-selection-section", style={"display": "none"}),
                    
                    # Action Buttons
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                [html.I(className="bi bi-eye me-1"), "Preview Order"],
                                id="btn-preview-order",
                                color="primary",
                                className="w-100"
                            ),
                        ], width=6),
                        dbc.Col([
                            dbc.Button(
                                [html.I(className="bi bi-x-circle me-1"), "Clear Form"],
                                id="btn-clear-trade-form",
                                color="outline-secondary",
                                className="w-100"
                            ),
                        ], width=6),
                    ]),
                ])
            ], className="mb-4"),
        ], md=6),
        
        # Right Column: Order Preview
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="bi bi-receipt me-2"),
                    "Order Preview"
                ]),
                dbc.CardBody([
                    # Preview Content (populated by callback)
                    html.Div(id="order-preview-content", children=[
                        html.P("Enter order details and click 'Preview Order' to see estimated costs.",
                               className="text-muted text-center my-5")
                    ]),
                ])
            ], className="mb-4"),
            
            # Tax Impact Summary (for sells)
            html.Div(id="trade-tax-impact-section"),
            
        ], md=6),
    ]),
    
    # Recent Orders Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="bi bi-clock-history me-2"),
                    "Recent Orders"
                ]),
                dbc.CardBody([
                    html.Div(id="recent-orders-content")
                ])
            ])
        ])
    ], className="mt-4"),
    
    # Store for selected ticker in Lot Picker
    dcc.Store(id="lot-picker-ticker-store"),

    # Lot Picker Modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Select Tax Lots")),
        dbc.ModalBody([
            html.P("Select specific lots to sell. Check the lots you want to include:", className="mb-3"),
            html.Div(id="lot-picker-grid-container"),
            html.Div(id="lot-picker-summary", className="mt-3"),
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="btn-lot-picker-cancel", color="secondary"),
            dbc.Button("Confirm Selection", id="btn-lot-picker-confirm", color="primary"),
        ])
    ], id="lot-picker-modal", size="xl", is_open=False, style={"maxWidth": "90vw", "width": "90vw"}),
    
    # Confirmation Modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle([
            html.I(className="bi bi-exclamation-triangle text-warning me-2"),
            "Confirm Order Execution"
        ])),
        dbc.ModalBody([
            html.Div(id="confirm-modal-body"),
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="btn-cancel-execution", color="secondary"),
            dbc.Button(
                [html.I(className="bi bi-check2-circle me-1"), "Execute Order"],
                id="btn-confirm-execution",
                color="success"
            ),
        ])
    ], id="confirm-execution-modal", is_open=False),
    
    # Hidden stores for state management
    dcc.Store(id="preview-data-store"),
    dcc.Store(id="selected-lots-store", data=[]),
    dcc.Store(id="trade-page-ready", data=1),
    dcc.Store(id="trade-page-refresh", data=0),
    
    # Hidden placeholder for dynamically created Execute Order button
    # Required because Dash callbacks must reference IDs that exist in initial layout
    html.Div(
        dbc.Button(id="btn-execute-order", style={"display": "none"}),
        style={"display": "none"}
    ),
    
], fluid=True, className="py-4")


# ============================================================
# CALLBACKS
# ============================================================

# 1. Handle staged order from rebalancing page
@callback(
    [Output("staged-order-alert", "children", allow_duplicate=True),
     Output("staged-order-alert", "is_open", allow_duplicate=True),
     Output("trade-action", "value", allow_duplicate=True),
     Output("trade-ticker", "value", allow_duplicate=True),
     Output("trade-quantity", "value", allow_duplicate=True)],
    [Input("trade-page-ready", "data"),
     Input("trade-page-refresh", "data")],
    [State("staged-order-store", "data"),
     State("staged-order-index", "data")],
    prevent_initial_call="initial_duplicate"
)
def load_staged_order(_page_ready, _page_refresh, staged_data, staged_index):
    """Pre-fill form with staged order(s) from rebalancing page."""
    if not staged_data:
        return "", False, "BUY", None, None
    
    # Handle both single order (dict) and multiple orders (list)
    if isinstance(staged_data, dict):
        orders = [staged_data]
    else:
        orders = staged_data
    
    if not orders:
        return "", False, "BUY", None, None
    
    # Pre-fill form with the FIRST order
    index = staged_index or 0
    if index < 0:
        index = 0
    if index >= len(orders):
        index = 0

    current_order = orders[index]
    ticker = current_order.get("ticker", "")
    action = current_order.get("action", "BUY")
    quantity = current_order.get("quantity", 0)
    source = current_order.get("source", "")

    display_qty = None
    try:
        qty_val = float(quantity)
        if qty_val > 0:
            display_qty = int(round(qty_val))
            if display_qty <= 0:
                display_qty = 1
    except (TypeError, ValueError):
        display_qty = None
    
    # Build alert message showing all staged orders
    if len(orders) == 1:
        alert_msg = [
            html.I(className="bi bi-info-circle me-2"),
            f"Order staged from {source}: {action} {display_qty or 0} shares of {ticker}"
        ]
    else:
        # Multiple orders - show summary
        order_lines = [html.Div([
            html.I(className="bi bi-info-circle me-2"),
            f"{len(orders)} orders staged from {source}:"
        ])]
        for i, order in enumerate(orders, 1):
            order_lines.append(html.Div(
                f"  {i}. {order.get('action')} {order.get('quantity', 0):.0f} shares of {order.get('ticker')}",
                className="ms-4 small"
            ))
        order_lines.append(html.Div(
            f"Form pre-filled with order {index + 1}. Execute orders one at a time.",
            className="mt-2 text-muted small"
        ))
        alert_msg = order_lines
    
    return alert_msg, True, action, ticker, display_qty


# 1b. Reset staged index when new staged orders arrive
@callback(
    Output("staged-order-index", "data", allow_duplicate=True),
    Input("staged-order-store", "data"),
    prevent_initial_call=True
)
def reset_staged_order_index(staged_data):
    if not staged_data:
        raise dash.exceptions.PreventUpdate
    return 0


# 1c. Advance to next staged order
@callback(
    [Output("staged-order-index", "data"),
     Output("trade-page-refresh", "data")],
    Input("btn-next-staged-order", "n_clicks"),
    [State("staged-order-index", "data"),
     State("staged-order-store", "data"),
     State("trade-page-refresh", "data")],
    prevent_initial_call=True
)
def advance_staged_order(n_clicks, staged_index, staged_data, page_refresh):
    if not n_clicks or not staged_data:
        raise dash.exceptions.PreventUpdate

    orders = [staged_data] if isinstance(staged_data, dict) else staged_data
    if not orders:
        raise dash.exceptions.PreventUpdate

    current_index = staged_index or 0
    next_index = current_index + 1
    if next_index >= len(orders):
        next_index = 0

    refresh_val = (page_refresh or 0) + 1
    return next_index, refresh_val


# 1d. Update Next button state and label
@callback(
    [Output("btn-next-staged-order", "disabled"),
     Output("btn-next-staged-order", "children")],
    [Input("trade-page-ready", "data"),
     Input("trade-page-refresh", "data")],
    [State("staged-order-store", "data"),
     State("staged-order-index", "data")],
    prevent_initial_call=False
)
def update_next_staged_button(_page_ready, _page_refresh, staged_data, staged_index):
    icon = html.I(className="bi bi-skip-forward me-2")

    if not staged_data:
        return True, [icon, "Next Staged Order"]

    orders = [staged_data] if isinstance(staged_data, dict) else staged_data
    if not orders or len(orders) <= 1:
        return True, [icon, "Next Staged Order"]

    index = staged_index or 0
    if index < 0 or index >= len(orders):
        index = 0

    return False, [icon, f"Next Staged Order ({index + 1}/{len(orders)})"]


# 2. Toggle price fields based on order type
@callback(
    [Output("trade-limit-price", "disabled"),
     Output("trade-stop-price", "disabled")],
    [Input("trade-order-type", "value")]
)
def toggle_price_fields(order_type):
    if order_type == "MARKET":
        return True, True
    elif order_type == "LIMIT":
        return False, True
    elif order_type == "STOP":
        return True, False
    elif order_type == "STOP_LIMIT":
        return False, False
    return True, True


# 3. Show/hide lot selection section for sells
@callback(
    [Output("lot-selection-section", "style"),
     Output("btn-open-lot-picker", "disabled")],
    [Input("trade-action", "value"),
     Input("lot-strategy", "value")]
)
def toggle_lot_selection(action, strategy):
    if action == "SELL":
        show = {"display": "block"}
        manual_disabled = strategy != "MANUAL"
        return show, manual_disabled
    return {"display": "none"}, True


# 4. Preview Order
@callback(
    [Output("order-preview-content", "children"),
     Output("trade-tax-impact-section", "children"),
     Output("preview-data-store", "data")],
    [Input("btn-preview-order", "n_clicks")],
    [State("trade-action", "value"),
     State("trade-ticker", "value"),
     State("trade-quantity", "value"),
     State("trade-order-type", "value"),
     State("trade-duration", "value"),
     State("trade-limit-price", "value"),
     State("trade-stop-price", "value"),
     State("lot-strategy", "value"),
     State("selected-lots-store", "data"),
     State("tax-strategy-store", "data")]
)
def preview_order_callback(n_clicks, action, ticker, quantity, order_type, duration,
                           limit_price, stop_price, lot_strategy, selected_lots, global_tax_strategy):
    if not n_clicks or not ticker or not quantity:
        return [html.P("Enter order details and click 'Preview Order'.", 
                       className="text-muted text-center my-5")], None, None
    
    # Validate quantity
    try:
        quantity = int(quantity)
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
    except:
        return [dbc.Alert("Please enter a valid quantity.", color="danger")], None, None
    
    # Get lot IDs for sells
    lot_ids = None
    if action == "SELL":
        if lot_strategy == "MANUAL" and selected_lots:
            
            # Logic: We take selected lots in order (sorted by selected order? or just list order) 
            # and allocate 'quantity' to them.
            
            qty_remaining = quantity
            mapped_lots = []
            
            # Sort lots? Or trust user selection order?
            # selected_lots is usually in grid order or selection order.
            
            for lot in selected_lots:
                if qty_remaining <= 0:
                    break
                    
                available = float(lot.get("Shares", 0))  # AG Grid column name
                to_sell_from_lot = min(qty_remaining, available)
                
                mapped_lots.append({
                    "id": lot.get("positionLotId"),
                    "size": to_sell_from_lot
                })
                
                qty_remaining -= to_sell_from_lot
            
            # If we didn't satisfy full quantity, should we warn?
            # The API will probably error if sum(lots) != quantity
            
            lot_ids = mapped_lots
            
        # For FIFO/LIFO/HIFO, lot selection is handled by the API or our logic
    
    # Validate manual lot quantity matches order quantity
    if lot_ids and action == "SELL":
        total_selected = sum(float(lot.get("size", 0)) for lot in lot_ids)
        if abs(total_selected - quantity) > 0.01:
            return [dbc.Alert(
                [
                    html.I(className="bi bi-exclamation-triangle-fill me-2"),
                    f"Mismatch: Order Quantity is {quantity}, but you selected lots totaling {total_selected:,.2f}.",
                    html.Br(),
                    "Please adjust Quantity or select more lots."
                ],
                color="danger"
            )], None, None
    
    # Call preview API
    preview_result = preview_order(
        ticker=ticker,
        quantity=quantity,
        action=action,
        price_type=order_type,
        limit_price=limit_price,
        stop_price=stop_price,
        order_term=duration,
        lot_ids=lot_ids
    )
    
    if not preview_result.get("success", False):
        error_msg = preview_result.get("error", "Unknown error occurred")
        return [dbc.Alert(f"Preview failed: {error_msg}", color="danger")], None, None
    
    # Build preview display
    est_value = preview_result.get("estimated_value", 0)
    est_commission = preview_result.get("estimated_commission", 0)
    est_total = preview_result.get("estimated_total", 0)
    preview_id = preview_result.get("preview_id")
    messages = preview_result.get("messages", [])
    
    preview_content = [
        dbc.Row([
            dbc.Col([
                html.H4(f"{action} {quantity} {ticker}", className="mb-3"),
            ])
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.Div("Order Type", className="text-muted small"),
                html.Div(order_type.replace("_", " ").title(), className="fw-bold"),
            ], width=6),
            dbc.Col([
                html.Div("Duration", className="text-muted small"),
                html.Div(duration.replace("_", " ").title(), className="fw-bold"),
            ], width=6),
        ], className="mb-3"),
        
        dbc.Row([
            dbc.Col([
                html.Div("Estimated Value (Gross)", className="text-muted small"),
                html.Div(fmt_dollar_clean(est_value), className="fw-bold fs-5"),
            ], width=6),
            dbc.Col([
                html.Div("Commission", className="text-muted small"),
                html.Div(fmt_dollar_clean(est_commission), className="fw-bold"),
            ], width=6),
        ], className="mb-3"),
        
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.Div("Estimated Total (Net of fees)", className="text-muted small"),
                html.Div(
                    fmt_dollar_clean(est_total), 
                    className="fw-bold fs-4 text-success" if action == "SELL" else "fw-bold fs-4 text-danger"
                ),
            ])
        ], className="mb-3"),
    ]
    
    # Show selected lots in preview (for MANUAL mode verification)
    if lot_ids and action == "SELL":
        preview_content.append(html.Hr())
        preview_content.append(html.Div("Selected Tax Lots (Manual)", className="text-muted small mb-2"))
        
        lot_details = []
        for i, lot in enumerate(lot_ids):
            lot_id = lot.get("id")
            lot_size = lot.get("size", 0)
            
            # Try to find the original lot data for more context
            original_lot = None
            if selected_lots:
                for sl in selected_lots:
                    if sl.get("positionLotId") == lot_id:
                        original_lot = sl
                        break
            
            if original_lot:
                acq_date = original_lot.get("Date Acquired", "Unknown")  # AG Grid column name
                term = original_lot.get("Term", "")  # AG Grid column name
                cost_basis = original_lot.get("Cost Basis", 0)  # AG Grid column name
                lot_details.append(
                    html.Div([
                        html.I(className="bi bi-check-circle-fill text-success me-2"),
                        html.Span(f"{lot_size:.0f} shares from {acq_date} ({term})", className="fw-bold"),
                        html.Span(f" — Lot ID: {lot_id}", className="text-muted small ms-2"),
                    ], className="mb-1")
                )
            else:
                lot_details.append(
                    html.Div([
                        html.I(className="bi bi-check-circle-fill text-success me-2"),
                        html.Span(f"{lot_size:.0f} shares", className="fw-bold"),
                        html.Span(f" — Lot ID: {lot_id}", className="text-muted small ms-2"),
                    ], className="mb-1")
                )
        
        preview_content.append(html.Div(lot_details))
        preview_content.append(
            dbc.Alert(
                "✓ These specific lots will be sold (not FIFO default).",
                color="success",
                className="py-2 mt-2 mb-0"
            )
        )
    
    # Add any messages from preview
    if messages:
        preview_content.append(html.Hr())
        for msg in messages:
            preview_content.append(
                dbc.Alert(msg, color="info", className="py-2 mb-1")
            )
    
    # Execute button
    preview_content.append(html.Hr())
    preview_content.append(
        dbc.Button(
            [html.I(className="bi bi-lightning-charge me-1"), "Execute Order"],
            id="btn-execute-order",
            color="success" if action == "SELL" else "primary",
            className="w-100",
            size="lg"
        )
    )
    
    # Build tax impact section for sells
    tax_section = None
    if action == "SELL":
        if lot_strategy == "MANUAL" and lot_ids:
            # Construct manual lot impact from selected lot details
            manual_lots_info = []
            for l_alloc in lot_ids:
                l_id = l_alloc.get("id")
                l_size = l_alloc.get("size")
                # find in selected_lots
                match = next((s for s in selected_lots if str(s.get("positionLotId")) == str(l_id)), None)
                if match:
                    # Calculate per-share basis from the original lot totals
                    orig_shares = float(match.get("Shares", 0))
                    orig_cost = float(match.get("Cost Basis", 0))
                    per_share_basis = orig_cost / orig_shares if orig_shares > 0 else 0
                    
                    manual_lots_info.append({
                        "size": l_size,
                        "per_share_basis": per_share_basis,
                        "term": match.get("Term", "Unknown"),
                        "acq_date": match.get("Date Acquired", "Unknown")
                    })
            
            tax_section = build_manual_lot_impact_card(manual_lots_info, est_total)
            
        else:
            # Use global tax strategy if not manual
            strategy = global_tax_strategy or "FIFO"
            if lot_strategy != "MANUAL" and lot_strategy:
                strategy = lot_strategy
            
            # Get tax simulation
            try:
                tax_result = simulate_sell(ticker, quantity, strategy=strategy)
                if tax_result:
                    tax_section = build_tax_impact_card(tax_result, ticker, quantity)
            except Exception as e:
                print(f"Tax simulation error: {e}")
    
    # Store preview data for execution
    preview_data = {
        "action": action,
        "ticker": ticker,
        "quantity": quantity,
        "order_type": order_type,
        "duration": duration,
        "limit_price": limit_price,
        "stop_price": stop_price,
        "lot_strategy": lot_strategy,
        "lot_ids": lot_ids,
        "preview_id": preview_id,
        "client_order_id": preview_result.get("client_order_id"),  # CRITICAL: Pass to place_order
        "estimated_total": est_total
    }
    
    return preview_content, tax_section, preview_data


def build_tax_impact_card(tax_result, ticker, quantity):
    """Build tax impact summary card for sell preview."""
    total_gain = tax_result.get("total_gain", 0)
    est_tax = tax_result.get("est_tax", 0)
    breakdown = tax_result.get("breakdown", [])
    
    gain_color = "text-success" if total_gain >= 0 else "text-danger"
    
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="bi bi-bank me-2"),
            "Estimated Tax Impact"
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div("Total Gain/Loss", className="text-muted small"),
                    html.Div(fmt_dollar_clean(total_gain), className=f"fw-bold fs-5 {gain_color}"),
                ], width=6),
                dbc.Col([
                    html.Div("Estimated Tax", className="text-muted small"),
                    html.Div(fmt_dollar_clean(est_tax), className="fw-bold fs-5"),
                ], width=6),
            ], className="mb-3"),
            
            # Lot breakdown if available
            html.Div([
                html.Hr(),
                html.Div("Lot Breakdown", className="text-muted small mb-2"),
                html.Div([
                    html.Div(
                        f"{lot.get('shares', 0):.0f} shares @ ${lot.get('cost_basis', 0):.2f} → "
                        f"{lot.get('term', 'Unknown')} ({fmt_dollar_clean(lot.get('gain', 0))})",
                        className="small"
                    ) for lot in breakdown[:5]  # Show max 5 lots
                ])
            ]) if breakdown else None
        ])
    ], className="mt-3")

    
def build_manual_lot_impact_card(lots, estimated_total):
    """Build tax impact summary for MANUAL lot selection."""
    if not lots: return None
    
    total_gain = 0
    st_gain = 0  # Short-term gains (taxed at higher rate)
    lt_gain = 0  # Long-term gains (taxed at lower rate)
    est_total_val = abs(estimated_total) if estimated_total else 0
    total_shares_to_sell = sum(lot.get("size", 0) for lot in lots)
    
    # Infer execution price per share
    est_price = est_total_val / total_shares_to_sell if total_shares_to_sell > 0 else 0
    
    breakdown_display = []
    
    for lot_info in lots:
        sold_shares = lot_info.get("size", 0)
        per_share_basis = lot_info.get("per_share_basis", 0)
        
        # Calculate cost of sold portion
        cost_of_sold = sold_shares * per_share_basis
        proceeds = sold_shares * est_price
        gain = proceeds - cost_of_sold
        total_gain += gain
        
        # Track gains by term for proper tax calculation
        term = lot_info.get('term', 'Short-Term')
        if term == 'Long-Term':
            lt_gain += max(0, gain)
        else:
            st_gain += max(0, gain)
        
        breakdown_display.append(
            html.Div(
                f"{sold_shares:.0f} shares from {lot_info.get('acq_date')} @ ${per_share_basis:.2f} → "
                f"{lot_info.get('term')} ({fmt_dollar_clean(gain)})",
                className="small"
            )
        )
        
    gain_color = "text-success" if total_gain >= 0 else "text-danger"
    
    # Apply correct tax rates by term (GIPS-compliant)
    est_tax = (st_gain * TAX_RATE_ST) + (lt_gain * TAX_RATE_LT) 
    
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="bi bi-bank me-2"),
            "Estimated Tax Impact (Manual Lots)"
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div("Total Gain/Loss", className="text-muted small"),
                    html.Div(fmt_dollar_clean(total_gain), className=f"fw-bold fs-5 {gain_color}"),
                ], width=6),
                dbc.Col([
                    html.Div("Est. Tax Impact", className="text-muted small"),
                    html.Div(f"~{fmt_dollar_clean(est_tax)}", className="fw-bold fs-5"),
                ], width=6),
            ], className="mb-3"),
            
            html.Div([
                html.Hr(),
                html.Div("Selected Lot Breakdown", className="text-muted small mb-2"),
                html.Div(breakdown_display)
            ])
        ])
    ], className="mt-3")


# 5. Open confirmation modal
@callback(
    [Output("confirm-execution-modal", "is_open"),
     Output("confirm-modal-body", "children")],
    [Input("btn-execute-order", "n_clicks"),
     Input("btn-cancel-execution", "n_clicks"),
     Input("btn-confirm-execution", "n_clicks")],
    [State("preview-data-store", "data"),
     State("confirm-execution-modal", "is_open")]
)
def toggle_confirm_modal(execute_click, cancel_click, confirm_click, preview_data, is_open):
    ctx = callback_context
    if not ctx.triggered:
        return False, []
    
    triggered_id = ctx.triggered_id
    
    if triggered_id == "btn-execute-order" and preview_data:
        # Show confirmation
        action = preview_data.get("action", "")
        ticker = preview_data.get("ticker", "")
        quantity = preview_data.get("quantity", 0)
        est_total = preview_data.get("estimated_total", 0)
        
        env_info = get_environment_info()
        is_sandbox = env_info.get("is_sandbox", True)
        
        body = [
            html.P([
                "You are about to execute the following order:"
            ], className="mb-3"),
            
            dbc.Card([
                dbc.CardBody([
                    html.H5(f"{action} {quantity} shares of {ticker}", className="mb-2"),
                    html.P(f"Estimated Total (Net of fees): {fmt_dollar_clean(est_total)}", className="mb-0"),
                ])
            ], className="mb-3"),
        ]
        
        if not is_sandbox:
            body.append(
                dbc.Alert([
                    html.I(className="bi bi-exclamation-triangle-fill me-2"),
                    "WARNING: You are in LIVE TRADING mode. This order will execute with real money."
                ], color="danger")
            )
        else:
            body.append(
                dbc.Alert([
                    html.I(className="bi bi-info-circle me-2"),
                    "You are in SANDBOX mode. This is a simulated order."
                ], color="info")
            )
        
        return True, body
    
    return False, []


# 6. Execute order
@callback(
    [Output("order-preview-content", "children", allow_duplicate=True),
     Output("confirm-execution-modal", "is_open", allow_duplicate=True)],
    [Input("btn-confirm-execution", "n_clicks")],
    [State("preview-data-store", "data")],
    prevent_initial_call=True
)
def execute_order_callback(n_clicks, preview_data):
    if not n_clicks or not preview_data:
        return dash.no_update, dash.no_update
    
    # Call place_order API
    result = place_order(
        ticker=preview_data.get("ticker"),
        quantity=preview_data.get("quantity"),
        action=preview_data.get("action"),
        price_type=preview_data.get("order_type"),
        limit_price=preview_data.get("limit_price"),
        stop_price=preview_data.get("stop_price"),
        order_term=preview_data.get("duration"),
        lot_ids=preview_data.get("lot_ids"),
        preview_id=preview_data.get("preview_id"),
        client_order_id=preview_data.get("client_order_id")
    )
    
    if result.get("success", False):
        order_id = result.get("order_id", "Unknown")
        status = result.get("status", "Submitted")
        
        # Trigger E*TRADE sync after 15 seconds to refresh holdings/transactions
        # This runs in a background thread to not block the UI
        def trigger_sync():
            try:
                from etrade_sync import sync_all
                print("[POST-TRADE] Running automatic sync after order execution...")
                sync_all()
                print("[POST-TRADE] Sync complete.")
                
                # Persist updated files to Google Drive for cross-platform access
                persist_to_drive()
                
            except Exception as e:
                print(f"[POST-TRADE] Sync failed (non-critical): {e}")
        
        def persist_to_drive():
            """
            Persist cashflows.csv and sample holdings.csv to Google Drive.
            Supports both:
            - Colab environment (Drive mounted at /content/drive/)
            - Desktop environment (using Google Drive API via token.json)
            """
            import os
            import shutil
            
            files_to_persist = ["cashflows.csv", "sample holdings.csv", "etrade_sync_status.json"]
            
            # Check if running in Colab (Drive mounted)
            colab_drive_path = "/content/drive/MyDrive/Dash-App-TWR"
            if os.path.exists("/content/drive/MyDrive"):
                print("[POST-TRADE] Colab detected - persisting to mounted Drive...")
                try:
                    os.makedirs(colab_drive_path, exist_ok=True)
                    for f in files_to_persist:
                        if os.path.exists(f):
                            shutil.copy(f, os.path.join(colab_drive_path, f))
                            print(f"[POST-TRADE] ✅ Persisted {f} to Drive")
                    print("[POST-TRADE] Drive persistence complete.")
                except Exception as e:
                    print(f"[POST-TRADE] Drive persistence failed: {e}")
                return
            
            # Desktop environment - use Google Drive API
            token_file = "token.json"
            if os.path.exists(token_file):
                print("[POST-TRADE] Desktop detected - uploading to Drive via API...")
                try:
                    from dash_wrappers import send_to_drive
                    for f in files_to_persist:
                        if os.path.exists(f):
                            with open(f, "r") as fp:
                                content = fp.read()
                            result = send_to_drive(content, f, mimetype="text/csv")
                            print(f"[POST-TRADE] {f}: {result}")
                    print("[POST-TRADE] Drive API upload complete.")
                except Exception as e:
                    print(f"[POST-TRADE] Drive API upload failed: {e}")
            else:
                print("[POST-TRADE] No Drive credentials found (token.json). Skipping Drive persistence.")
        
        threading.Timer(15.0, trigger_sync).start()
        
        success_content = [
            dbc.Alert([
                html.I(className="bi bi-check-circle-fill me-2"),
                html.Strong("Order Executed Successfully!")
            ], color="success", className="mb-3"),
            
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div("Order ID", className="text-muted small"),
                            html.Div(order_id, className="fw-bold"),
                        ], width=6),
                        dbc.Col([
                            html.Div("Status", className="text-muted small"),
                            html.Div(status, className="fw-bold"),
                        ], width=6),
                    ]),
                ])
            ], className="mb-3"),
            
            dbc.Button(
                [html.I(className="bi bi-plus-circle me-1"), "New Order"],
                id="btn-new-order",
                color="primary",
                href="/trade"
            )
        ]
        
        return success_content, False
    else:
        error_msg = result.get("error", "Unknown error occurred")
        
        error_content = [
            dbc.Alert([
                html.I(className="bi bi-x-circle-fill me-2"),
                f"Order Failed: {error_msg}"
            ], color="danger"),
            
            dbc.Button(
                [html.I(className="bi bi-arrow-clockwise me-1"), "Try Again"],
                id="btn-retry-order",
                color="primary"
            )
        ]
        
        return error_content, False


# 7. Load recent orders
@callback(
    Output("recent-orders-content", "children"),
    [Input("btn-refresh-trade", "n_clicks"),
     Input("url", "pathname")]
)
def load_recent_orders(n_clicks, pathname):
    if pathname != "/trade":
        return dash.no_update
    
    orders = get_recent_orders(limit=10)
    
    if not orders:
        return html.P("No recent orders found.", className="text-muted text-center my-3")
    
    # Build orders table
    rows = []
    for order in orders:
        status_color = {
            "EXECUTED": "success",
            "OPEN": "primary",
            "PENDING": "warning",
            "CANCELLED": "secondary",
            "REJECTED": "danger"
        }.get(order.get("status", ""), "secondary")
        
        rows.append(
            html.Tr([
                html.Td(order.get("timestamp", "")[:10]),
                html.Td(order.get("action", "")),
                html.Td(order.get("ticker", "")),
                html.Td(f"{order.get('quantity', 0):,.0f}"),
                html.Td(order.get("order_type", "")),
                html.Td(dbc.Badge(order.get("status", ""), color=status_color)),
            ])
        )
    
    return dbc.Table([
        html.Thead(html.Tr([
            html.Th("Date"),
            html.Th("Action"),
            html.Th("Ticker"),
            html.Th("Qty"),
            html.Th("Type"),
            html.Th("Status"),
        ])),
        html.Tbody(rows)
    ], bordered=True, hover=True, responsive=True, className="mb-0")


# 8. Lot Picker Modal
@callback(
    [Output("lot-picker-modal", "is_open"),
     Output("lot-picker-grid-container", "children"),
     Output("lot-picker-ticker-store", "data"),
     Output("btn-lot-picker-confirm", "disabled")],
    [Input("btn-open-lot-picker", "n_clicks"),
     Input("btn-lot-picker-cancel", "n_clicks"),
     Input("btn-lot-picker-confirm", "n_clicks")],
    [State("trade-ticker", "value"),
     State("lot-picker-modal", "is_open")]
)
def toggle_lot_picker(open_click, cancel_click, confirm_click, ticker, is_open):
    ctx = callback_context
    if not ctx.triggered:
        return False, [], None, False
    
    triggered_id = ctx.triggered_id
    
    if triggered_id == "btn-open-lot-picker" and ticker:
        # Build lot picker grid using LIVE E*TRADE DATA
        try:
            # Clean ticker
            ticker = ticker.strip().upper()
            lots = fetch_position_lots(ticker)
            
            if not lots:
                # If no lots found, check if it's an error or just empty
                msg = html.Div([
                    html.I(className="bi bi-exclamation-triangle text-warning display-4 mb-3"),
                    html.H5("No Tax Lots Found"),
                    html.P([
                        f"Could not retrieve cost basis details for {ticker}. ",
                        html.Br(),
                        "This may happen if:",
                        html.Ul([
                            html.Li("The position was just acquired today"),
                            html.Li("The E*TRADE API is currently unavailable"),
                            html.Li("The position is held in a non-brokerage account")
                        ]),
                        "You can still place the order, but it will use the default FIFO method."
                    ], className="text-muted")
                ], className="text-center py-5")
                
                return True, msg, ticker, True # Disable confirm button
            
            # Format lots for AG Grid
            grid_data = []
            for lot in lots:
                # Convert epoch date to string
                acq_date = datetime.fromtimestamp(lot.get("acquiredDate", 0) / 1000).strftime("%m/%d/%Y")
                
                # Check for wash sale
                is_wash = lot.get("shortType", 0) == 1
                date_display = f"{acq_date} ⚠️" if is_wash else acq_date
                
                item = {
                    "Date Acquired": date_display,
                    "Shares": lot.get("remainingQty", 0),
                    "Cost Basis": lot.get("pricePaid", 0), # Total cost for remaining shares
                    "Current Price": lot.get("marketValue", 0) / lot.get("remainingQty", 1) if lot.get("remainingQty", 0) > 0 else 0,
                    "Gain/Loss": float(lot.get("marketValue", 0)) - float(lot.get("pricePaid", 0)),
                    "Term": "Short" if lot.get("termCode") == 1 else "Long",
                    "positionLotId": lot.get("positionLotId"), # HIDDEN KEY
                    "Select": False
                }
                grid_data.append(item)
            
            column_defs = [
                {
                    "field": "Select", "headerName": "", 
                    "checkboxSelection": True, "headerCheckboxSelection": True, 
                    "width": 50, "maxWidth": 50, "pinned": "left", 
                    "suppressMenu": True, "suppressMovable": True, "lockPosition": True
                },
                {"field": "Date Acquired", "minWidth": 140, "flex": 1},
                {"field": "Shares", "minWidth": 90, "flex": 1, "type": "numericColumn"},
                {
                    "field": "Cost Basis", "headerName": "Total Cost", 
                    "minWidth": 110, "flex": 1, 
                    "valueFormatter": {"function": "'$' + (value ? value.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) : '0.00')"}
                },
                {
                    "field": "Current Price", "headerName": "Price", 
                    "minWidth": 90, "flex": 1, 
                    "valueFormatter": {"function": "'$' + (value ? value.toFixed(2) : '0.00')"}
                },
                {
                    "field": "Gain/Loss", "minWidth": 110, "flex": 1,
                    "valueFormatter": {"function": "(value > 0 ? '+' : '') + '$' + (value ? value.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) : '0.00')"},
                    "cellStyle": {"function": "params.value >= 0 ? {'color': '#28a745'} : {'color': '#dc3545'}"}
                },
                {"field": "Term", "minWidth": 90, "flex": 1},
                # HIDDEN ID COLUMN
                {"field": "positionLotId", "hide": True}
            ]
            
            grid = dag.AgGrid(
                id="lot-picker-grid",
                rowData=grid_data,
                columnDefs=column_defs,
                defaultColDef={"sortable": True, "filter": True, "resizable": True},
                dashGridOptions={"rowSelection": "multiple", "suppressRowClickSelection": True},
                style={"height": "60vh", "width": "100%"}
            )
            
            return True, [grid], ticker, False
            
        except Exception as e:
            err_msg = html.Div([
                html.I(className="bi bi-x-circle text-danger display-4 mb-3"),
                html.H5("Error Loading Lots"),
                html.P(str(e), className="text-danger"),
                html.P("Please check your connection and try again.", className="text-muted")
            ], className="text-center py-5")
            return True, err_msg, ticker, True
    
    return False, [], None, False


# 8b. Save selected lots when Confirm is clicked
@callback(
    [Output("selected-lots-store", "data"),
     Output("lot-picker-modal", "is_open", allow_duplicate=True)], # Close modal on confirm
    Input("btn-lot-picker-confirm", "n_clicks"),
    State("lot-picker-grid", "selectedRows"),
    prevent_initial_call=True
)
def save_selected_lots(n_clicks, selected_rows):
    """Save manually selected tax lots to the store."""
    if not n_clicks:
        return dash.no_update, dash.no_update
    
    if not selected_rows:
        return [], False
    
    # Return the selected lot records
    return selected_rows, False


# 8c. Update lot picker summary as user selects rows
@callback(
    Output("lot-picker-summary", "children"),
    Input("lot-picker-grid", "selectedRows"),
    prevent_initial_call=True
)
def update_lot_picker_summary(selected_rows):
    """Show a live count of selected lots and shares."""
    if not selected_rows:
        return html.P("No lots selected", className="text-muted")
    
    total_shares = sum(row.get("Shares", 0) for row in selected_rows)
    lot_count = len(selected_rows)
    
    return html.Div([
        html.Strong(f"{lot_count} lot(s) selected"),
        html.Span(f" — {total_shares:,.2f} shares total", className="text-muted ms-2")
    ], className="text-success")


# 9. Clear form
@callback(
    [Output("trade-action", "value", allow_duplicate=True),
     Output("trade-ticker", "value", allow_duplicate=True),
     Output("trade-quantity", "value", allow_duplicate=True),
     Output("trade-order-type", "value"),
     Output("trade-limit-price", "value"),
     Output("trade-stop-price", "value"),
     Output("staged-order-alert", "is_open", allow_duplicate=True),
     Output("ticker-validation-msg", "children", allow_duplicate=True),
     Output("ticker-validation-msg", "className", allow_duplicate=True)],
    [Input("btn-clear-trade-form", "n_clicks")],
    prevent_initial_call=True
)
def clear_trade_form(n_clicks):
    if not n_clicks:
        return dash.no_update
    return "BUY", None, None, "MARKET", None, None, False, "", "small mb-2"


# 10. Validate ticker symbol on input change
@callback(
    Output("ticker-validation-msg", "children"),
    Output("ticker-validation-msg", "className"),
    Input("trade-ticker", "value"),
    prevent_initial_call=True
)
def validate_ticker_input(ticker_value):
    """
    Validate the ticker symbol when user finishes typing.
    Shows success message with price, or error if invalid.
    """
    if not ticker_value:
        return "", "small mb-2"
    
    # Clean the input
    symbol = ticker_value.strip().upper()
    
    if len(symbol) < 1 or len(symbol) > 10:
        return "Invalid ticker format", "small mb-2 text-danger"
    
    # Validate using yfinance
    result = validate_ticker(symbol)
    
    if result["valid"]:
        # Success: Show ticker name and current price
        price_str = f"${result['price']:.2f}" if result['price'] else "N/A"
        msg = f"✓ {result['name']} — Last: {price_str}"
        return msg, "small mb-2 text-success"
    else:
        # Error: Show the error message
        return f"✗ {result['message']}", "small mb-2 text-danger"
