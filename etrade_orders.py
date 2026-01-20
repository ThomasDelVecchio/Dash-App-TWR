"""
E*TRADE Order Placement Module

Handles order preview and execution via E*TRADE API.
Supports both sandbox and production environments based on ETRADE_SANDBOX config.

Usage:
    from etrade_orders import preview_order, place_order, get_order_status
    
    # Preview before placing
    preview = preview_order("AAPL", 10, "BUY")
    
    # Place order
    result = place_order("AAPL", 10, "BUY", price_type="MARKET")
"""

import os
import json
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from enum import Enum

from etrade_auth import get_etrade_session, get_etrade_session_safe, get_base_url
from config import ETRADE_SANDBOX, ETRADE_ACCOUNT_ID, is_etrade_configured


# ============================================================
# ENUMS & CONSTANTS
# ============================================================

class OrderAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    BUY_TO_COVER = "BUY_TO_COVER"
    SELL_SHORT = "SELL_SHORT"


class PriceType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderTerm(Enum):
    GOOD_FOR_DAY = "GOOD_FOR_DAY"
    GOOD_UNTIL_CANCEL = "GOOD_UNTIL_CANCEL"
    IMMEDIATE_OR_CANCEL = "IMMEDIATE_OR_CANCEL"
    FILL_OR_KILL = "FILL_OR_KILL"


# Order status tracking file (for confirmation display)
ORDER_HISTORY_FILE = "order_history.json"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _get_account_id_key(session) -> Tuple[str, str]:
    """
    Get the E*TRADE account ID KEY (encrypted key for API calls).
    
    Returns:
        Tuple[str, str]: (accountIdKey, accountNumber)
    """
    base_url = get_base_url()
    
    headers = {"Accept": "application/json"}
    response = session.get(
        f"{base_url}/v1/accounts/list",
        headers=headers,
        timeout=(10, 30)
    )
    
    if response.status_code != 200:
        raise RuntimeError(f"Failed to get accounts: HTTP {response.status_code}")
    
    data = response.json()
    accounts = data.get("AccountListResponse", {}).get("Accounts", {}).get("Account", [])
    
    if not accounts:
        raise RuntimeError("No E*TRADE accounts found")
    
    if not isinstance(accounts, list):
        accounts = [accounts]
    
    # Find matching account if specified
    if ETRADE_ACCOUNT_ID:
        for acct in accounts:
            acct_num = acct.get("accountId", "")
            acct_key = acct.get("accountIdKey", "")
            
            if acct_num == ETRADE_ACCOUNT_ID or acct_key == ETRADE_ACCOUNT_ID:
                return acct_key, acct_num
        
        available = [a.get("accountId", "unknown") for a in accounts]
        raise RuntimeError(f"Account '{ETRADE_ACCOUNT_ID}' not found. Available: {available}")
    
    # Default to first account
    first_acct = accounts[0]
    return first_acct.get("accountIdKey", ""), first_acct.get("accountId", "")


def _build_order_payload(
    ticker: str,
    quantity: int,
    action: str,
    price_type: str = "MARKET",
    limit_price: Optional[float] = None,
    stop_price: Optional[float] = None,
    order_term: str = "GOOD_FOR_DAY",
    lot_ids: Optional[List[str]] = None,
    account_id_key: str = None,
) -> Dict:
    """
    Builds the E*TRADE order XML/JSON payload.
    
    Args:
        ticker: Stock symbol
        quantity: Number of shares
        action: BUY, SELL, etc.
        price_type: MARKET, LIMIT, STOP, STOP_LIMIT
        limit_price: Required for LIMIT orders
        stop_price: Required for STOP orders
        order_term: GOOD_FOR_DAY, GOOD_UNTIL_CANCEL, etc.
        lot_ids: Specific lot IDs for tax lot selection (SELL orders)
        account_id_key: E*TRADE account key
    
    Returns:
        Dict: Order payload for API
    """
    # E*TRADE uses a nested structure
    instrument = {
        "Product": {
            "securityType": "EQ",  # Equity
            "symbol": ticker.upper()
        },
        "orderAction": action.upper(),
        "quantityType": "QUANTITY",
        "quantity": int(quantity)
    }
    
    # Add lot details if specified (for tax lot selection on sells)
    if lot_ids and action.upper() == "SELL":
        instrument["Lots"] = {
            "Lot": [{"id": lot_id} for lot_id in lot_ids]
        }
    
    order_detail = {
        "allOrNone": False,
        "priceType": price_type.upper(),
        "orderTerm": order_term.upper(),
        "marketSession": "REGULAR",
        "Instrument": [instrument]
    }
    
    # Add price constraints based on order type
    if price_type.upper() == "LIMIT" and limit_price:
        order_detail["limitPrice"] = float(limit_price)
    elif price_type.upper() == "STOP" and stop_price:
        order_detail["stopPrice"] = float(stop_price)
    elif price_type.upper() == "STOP_LIMIT":
        if limit_price:
            order_detail["limitPrice"] = float(limit_price)
        if stop_price:
            order_detail["stopPrice"] = float(stop_price)
    
    return {
        "PlaceOrderRequest": {
            "orderType": "EQ",
            "clientOrderId": f"DELVEX_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "Order": [order_detail]
        }
    }


def _save_order_history(order_result: Dict):
    """Save order to local history for confirmation display."""
    history = []
    
    if os.path.exists(ORDER_HISTORY_FILE):
        try:
            with open(ORDER_HISTORY_FILE, "r") as f:
                history = json.load(f)
        except:
            history = []
    
    # Flatten order_result into history entry (avoid nesting under "order" key)
    history_entry = {
        "timestamp": datetime.now().isoformat(),
        "environment": "SANDBOX" if ETRADE_SANDBOX else "PRODUCTION"
    }
    # Merge order_result fields at top level
    if isinstance(order_result, dict):
        history_entry.update(order_result)
    else:
        history_entry["order"] = order_result  # Fallback for non-dict
    history.append(history_entry)
    
    # Keep last 100 orders
    history = history[-100:]
    
    with open(ORDER_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


# ============================================================
# PUBLIC API FUNCTIONS
# ============================================================

def get_environment_info() -> Dict:
    """
    Returns current E*TRADE environment configuration.
    
    Returns:
        Dict with environment details for UI display
    """
    return {
        "configured": is_etrade_configured(),
        "is_sandbox": ETRADE_SANDBOX,
        "environment_label": "SANDBOX" if ETRADE_SANDBOX else "PRODUCTION",
        "base_url": get_base_url(),
        "account_id": ETRADE_ACCOUNT_ID or "Auto-detect",
        "warning": "Orders will execute with REAL money!" if not ETRADE_SANDBOX else None
    }


def preview_order(
    ticker: str,
    quantity: int,
    action: str,
    price_type: str = "MARKET",
    limit_price: Optional[float] = None,
    stop_price: Optional[float] = None,
    order_term: str = "GOOD_FOR_DAY",
    lot_ids: Optional[List[str]] = None,
) -> Dict:
    """
    Preview an order WITHOUT placing it.
    
    E*TRADE's preview endpoint returns estimated costs, commissions,
    and validates the order before execution.
    
    Args:
        ticker: Stock symbol (e.g., "AAPL")
        quantity: Number of shares
        action: "BUY" or "SELL"
        price_type: "MARKET", "LIMIT", "STOP", "STOP_LIMIT"
        limit_price: Required for LIMIT orders
        stop_price: Required for STOP orders
        order_term: "GOOD_FOR_DAY", "GOOD_UNTIL_CANCEL", etc.
        lot_ids: Specific lot IDs for SELL orders (tax lot selection)
    
    Returns:
        Dict with preview details:
            - estimated_value: Estimated order value
            - estimated_commission: Commission/fees
            - estimated_total: Total including fees
            - order_id: Preview order ID (needed for placement)
            - messages: Any warnings from E*TRADE
            - success: True if preview succeeded
    """
    try:
        session = get_etrade_session_safe()
        if session is None:
            return {
                "success": False,
                "error": "Not authenticated. Run 'python etrade_auth.py' to authenticate.",
                "estimated_value": 0,
                "estimated_commission": 0,
                "estimated_total": 0
            }
        
        account_id_key, account_num = _get_account_id_key(session)
        
        payload = _build_order_payload(
            ticker=ticker,
            quantity=quantity,
            action=action,
            price_type=price_type,
            limit_price=limit_price,
            stop_price=stop_price,
            order_term=order_term,
            lot_ids=lot_ids,
            account_id_key=account_id_key
        )
        
        # Convert to PreviewOrderRequest
        payload["PreviewOrderRequest"] = payload.pop("PlaceOrderRequest")
        
        base_url = get_base_url()
        url = f"{base_url}/v1/accounts/{account_id_key}/orders/preview"
        
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        # DEBUG: Log request details
        import json as json_module
        print("\n" + "="*60)
        print("DEBUG: E*TRADE Preview Order Request")
        print("="*60)
        print(f"URL: {url}")
        print(f"Headers: {headers}")
        print(f"Payload:\n{json_module.dumps(payload, indent=2)}")
        print("="*60)
        
        response = session.post(url, json=payload, headers=headers, timeout=(10, 30))
        
        # DEBUG: Log response
        print(f"\nDEBUG: Response Status: {response.status_code}")
        print(f"DEBUG: Response Body: {response.text[:1000] if response.text else 'EMPTY'}")
        print("="*60 + "\n")
        
        if response.status_code != 200:
            error_text = response.text[:500] if response.text else "Unknown error"
            return {
                "success": False,
                "error": f"Preview failed: HTTP {response.status_code} - {error_text}",
                "estimated_value": 0,
                "estimated_commission": 0,
                "estimated_total": 0
            }
        
        data = response.json()
        preview = data.get("PreviewOrderResponse", {})
        order_info = preview.get("Order", [{}])[0]
        
        # Extract preview details
        estimated_value = float(order_info.get("estimatedOrderValue", 0))
        estimated_commission = float(order_info.get("estimatedCommission", 0))
        estimated_total = float(order_info.get("estimatedTotalAmount", estimated_value + estimated_commission))
        
        # Get preview ID (required for placement)
        preview_ids = preview.get("PreviewIds", {}).get("previewId", [])
        preview_id = preview_ids[0] if preview_ids else None
        
        # Get any messages/warnings
        messages = []
        msg_list = order_info.get("messages", {}).get("Message", [])
        if not isinstance(msg_list, list):
            msg_list = [msg_list]
        for msg in msg_list:
            if isinstance(msg, dict):
                messages.append(msg.get("description", str(msg)))
            else:
                messages.append(str(msg))
        
        return {
            "success": True,
            "preview_id": preview_id,
            "ticker": ticker.upper(),
            "action": action.upper(),
            "quantity": quantity,
            "price_type": price_type.upper(),
            "estimated_value": estimated_value,
            "estimated_commission": estimated_commission,
            "estimated_total": estimated_total,
            "messages": messages,
            "account": account_num,
            "environment": "SANDBOX" if ETRADE_SANDBOX else "PRODUCTION"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "estimated_value": 0,
            "estimated_commission": 0,
            "estimated_total": 0
        }


def place_order(
    ticker: str,
    quantity: int,
    action: str,
    price_type: str = "MARKET",
    limit_price: Optional[float] = None,
    stop_price: Optional[float] = None,
    order_term: str = "GOOD_FOR_DAY",
    lot_ids: Optional[List[str]] = None,
    preview_id: Optional[str] = None,
) -> Dict:
    """
    Place an order with E*TRADE.
    
    IMPORTANT: This will execute a REAL trade in production mode!
    Always call preview_order() first to validate.
    
    Args:
        ticker: Stock symbol
        quantity: Number of shares
        action: "BUY" or "SELL"
        price_type: "MARKET", "LIMIT", "STOP", "STOP_LIMIT"
        limit_price: Required for LIMIT orders
        stop_price: Required for STOP orders
        order_term: "GOOD_FOR_DAY", "GOOD_UNTIL_CANCEL", etc.
        lot_ids: Specific lot IDs for SELL orders
        preview_id: Preview ID from preview_order() (recommended)
    
    Returns:
        Dict with order result:
            - success: True if order placed
            - order_id: E*TRADE order ID
            - status: Order status
            - filled_quantity: Shares filled (for market orders, often immediate)
            - error: Error message if failed
    """
    try:
        session = get_etrade_session_safe()
        if session is None:
            return {
                "success": False,
                "error": "Not authenticated. Run 'python etrade_auth.py' to authenticate."
            }
        
        account_id_key, account_num = _get_account_id_key(session)
        
        payload = _build_order_payload(
            ticker=ticker,
            quantity=quantity,
            action=action,
            price_type=price_type,
            limit_price=limit_price,
            stop_price=stop_price,
            order_term=order_term,
            lot_ids=lot_ids,
            account_id_key=account_id_key
        )
        
        # Add preview ID if available (validates against preview)
        if preview_id:
            payload["PlaceOrderRequest"]["PreviewIds"] = {
                "previewId": [preview_id]
            }
        
        base_url = get_base_url()
        url = f"{base_url}/v1/accounts/{account_id_key}/orders/place"
        
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        response = session.post(url, json=payload, headers=headers, timeout=(10, 30))
        
        if response.status_code not in (200, 201):
            error_text = response.text[:500] if response.text else "Unknown error"
            return {
                "success": False,
                "error": f"Order failed: HTTP {response.status_code} - {error_text}"
            }
        
        data = response.json()
        order_response = data.get("PlaceOrderResponse", {})
        order_info = order_response.get("Order", [{}])[0]
        
        # Extract order details
        order_id = order_info.get("orderId")
        
        # Get order status
        order_detail = order_info.get("OrderDetail", [{}])[0]
        status = order_detail.get("status", "UNKNOWN")
        
        # Get filled info
        filled_qty = int(order_detail.get("filledQuantity", 0))
        executed_price = float(order_detail.get("executedPrice", 0))
        
        result = {
            "success": True,
            "order_id": order_id,
            "ticker": ticker.upper(),
            "action": action.upper(),
            "quantity": quantity,
            "status": status,
            "filled_quantity": filled_qty,
            "executed_price": executed_price,
            "account": account_num,
            "environment": "SANDBOX" if ETRADE_SANDBOX else "PRODUCTION",
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to history
        _save_order_history(result)
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def get_order_status(order_id: str) -> Dict:
    """
    Get the current status of an order.
    
    Args:
        order_id: E*TRADE order ID
    
    Returns:
        Dict with order status details
    """
    try:
        session = get_etrade_session_safe()
        if session is None:
            return {"success": False, "error": "Not authenticated"}
        
        account_id_key, _ = _get_account_id_key(session)
        
        base_url = get_base_url()
        url = f"{base_url}/v1/accounts/{account_id_key}/orders/{order_id}"
        
        headers = {"Accept": "application/json"}
        response = session.get(url, headers=headers, timeout=(10, 30))
        
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"Failed to get order status: HTTP {response.status_code}"
            }
        
        data = response.json()
        order = data.get("OrdersResponse", {}).get("Order", [{}])[0]
        order_detail = order.get("OrderDetail", [{}])[0]
        
        return {
            "success": True,
            "order_id": order_id,
            "status": order_detail.get("status", "UNKNOWN"),
            "filled_quantity": int(order_detail.get("filledQuantity", 0)),
            "ordered_quantity": int(order_detail.get("orderedQuantity", 0)),
            "executed_price": float(order_detail.get("executedPrice", 0)),
            "order_type": order.get("orderType"),
            "placed_time": order_detail.get("placedTime")
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def cancel_order(order_id: str) -> Dict:
    """
    Cancel a pending order.
    
    Args:
        order_id: E*TRADE order ID
    
    Returns:
        Dict with cancellation result
    """
    try:
        session = get_etrade_session_safe()
        if session is None:
            return {"success": False, "error": "Not authenticated"}
        
        account_id_key, _ = _get_account_id_key(session)
        
        base_url = get_base_url()
        url = f"{base_url}/v1/accounts/{account_id_key}/orders/cancel"
        
        payload = {
            "CancelOrderRequest": {
                "orderId": order_id
            }
        }
        
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        response = session.put(url, json=payload, headers=headers, timeout=(10, 30))
        
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"Cancel failed: HTTP {response.status_code}"
            }
        
        return {
            "success": True,
            "order_id": order_id,
            "message": "Order cancelled successfully"
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_recent_orders(limit: int = 25) -> List[Dict]:
    """
    Get recent order history from local file.
    
    Args:
        limit: Maximum number of orders to return
    
    Returns:
        List of recent order records
    """
    if not os.path.exists(ORDER_HISTORY_FILE):
        return []
    
    try:
        with open(ORDER_HISTORY_FILE, "r") as f:
            history = json.load(f)
        
        # Migration: Unwrap nested "order" keys from old format
        migrated = []
        for record in history:
            if "order" in record and isinstance(record["order"], dict):
                # Old format: flatten nested order data to top level
                flat = {k: v for k, v in record.items() if k != "order"}
                flat.update(record["order"])
                migrated.append(flat)
            else:
                migrated.append(record)
        
        return migrated[-limit:][::-1]  # Most recent first
    except:
        return []


# ============================================================
# CLI TEST
# ============================================================

if __name__ == "__main__":
    """
    Test order preview (does NOT place order):
        python etrade_orders.py
    """
    print("=" * 60)
    print("E*TRADE Order Module Test")
    print("=" * 60)
    
    env = get_environment_info()
    print(f"\nEnvironment: {env['environment_label']}")
    print(f"Base URL: {env['base_url']}")
    print(f"Account: {env['account_id']}")
    
    if env['warning']:
        print(f"\n⚠️  WARNING: {env['warning']}")
    
    print("\n" + "-" * 60)
    print("Testing Order Preview (no execution)...")
    print("-" * 60)
    
    # Test preview only
    preview = preview_order(
        ticker="AAPL",
        quantity=1,
        action="BUY",
        price_type="MARKET"
    )
    
    if preview["success"]:
        print(f"\n✅ Preview successful!")
        print(f"   Ticker: {preview['ticker']}")
        print(f"   Action: {preview['action']}")
        print(f"   Quantity: {preview['quantity']}")
        print(f"   Estimated Value: ${preview['estimated_value']:,.2f}")
        print(f"   Commission: ${preview['estimated_commission']:,.2f}")
        print(f"   Total: ${preview['estimated_total']:,.2f}")
        
        if preview.get("messages"):
            print(f"   Messages: {preview['messages']}")
    else:
        print(f"\n❌ Preview failed: {preview.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 60)
