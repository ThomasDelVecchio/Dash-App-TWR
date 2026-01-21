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

USER_AGENT = "DELVEX-Portfolio-Analytics/1.0"

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
    
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
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


def fetch_position_lots(ticker: str) -> List[Dict]:
    """
    Fetch tax lots for a specific position from E*TRADE Portfolio API.
    
    This returns E*TRADE's internal positionLotId for each lot, which is
    required to specify specific lots when placing sell orders.
    
    Args:
        ticker: Stock symbol (e.g., "AAPL")
    
    Returns:
        List of lot dicts with keys:
            - positionLotId: E*TRADE's internal lot ID (required for order API)
            - acquiredDate: Date lot was acquired (epoch ms)
            - originalQty: Original shares in lot
            - remainingQty: Shares remaining in lot
            - price: Cost basis per share
            - marketValue: Current market value
            - pricePaid: Total cost basis
            - termCode: 1 = Short-Term, 2 = Long-Term
            - daysGain: Dollar gain/loss
            - daysGainPct: Percent gain/loss
            - shortType: 0 = Normal, 1 = Wash Sale
    """
    try:
        from etrade_auth import get_etrade_session_safe, get_base_url
        
        session = get_etrade_session_safe()
        if session is None:
            print("[fetch_position_lots] Not authenticated")
            return []
        
        account_id_key, account_num = _get_account_id_key(session)
        base_url = get_base_url()
        
        # Request portfolio with lot details
        url = f"{base_url}/v1/accounts/{account_id_key}/portfolio"
        params = {
            "lotsRequired": "true",
            "view": "COMPLETE"
        }
        headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
        
        response = session.get(url, params=params, headers=headers, timeout=(10, 30))
        
        if response.status_code != 200:
            print(f"[fetch_position_lots] API error: HTTP {response.status_code}")
            return []
        
        data = response.json()
        
        # Navigate E*TRADE's nested response
        portfolio = data.get("PortfolioResponse", {}).get("AccountPortfolio", [])
        if isinstance(portfolio, dict):
            portfolio = [portfolio]
        
        # Find positions and their lots
        ticker_upper = ticker.upper()
        lots = []
        
        for account in portfolio:
            positions = account.get("Position", [])
            if isinstance(positions, dict):
                positions = [positions]
            
            for position in positions:
                product = position.get("Product", {})
                symbol = product.get("symbol", "").upper()
                
                if symbol != ticker_upper:
                    continue
                
                # Extract lots from PositionLot array
                # E*TRADE API requires a second call to the lotsDetails URL
                # The portfolio response contains a URL, not inline lot data
                lots_url = position.get("lotsDetails")
                
                if not lots_url:
                    print(f"[fetch_position_lots] No lotsDetails URL for {ticker}")
                    return []
                
                # Fetch the actual lot data from the lotsDetails URL
                print(f"[fetch_position_lots] Fetching lots from: {lots_url}")
                lots_response = session.get(
                    lots_url,
                    headers={"Accept": "application/json"},
                    timeout=(10, 30)
                )
                
                if lots_response.status_code != 200:
                    print(f"[fetch_position_lots] Lots API returned {lots_response.status_code}")
                    return []
                
                lots_data = lots_response.json()
                pos_lots = lots_data.get("PositionLotsResponse", {}).get("PositionLot", [])
                
                if isinstance(pos_lots, dict):
                    pos_lots = [pos_lots]
                
                print(f"[fetch_position_lots] Found {len(pos_lots)} lot(s) for {ticker}")
                
                for lot in pos_lots:
                    # Robust field extraction
                    price = float(lot.get("price", 0))
                    qty = float(lot.get("remainingQty", 0))
                    price_paid = float(lot.get("pricePaid", 0))
                    
                    # Fallback for Total Cost if API returns 0
                    if price_paid == 0 and price > 0 and qty > 0:
                        price_paid = price * qty
                        
                    lot_data = {
                        "positionLotId": lot.get("positionLotId"),
                        "acquiredDate": lot.get("acquiredDate"),
                        "originalQty": float(lot.get("originalQty", 0)),
                        "remainingQty": qty,
                        "price": price,
                        "marketValue": float(lot.get("marketValue", 0)),
                        "pricePaid": price_paid,
                        "termCode": lot.get("termCode", 0),
                        "daysGain": float(lot.get("daysGain", 0)),
                        "daysGainPct": float(lot.get("daysGainPct", 0)),
                        "totalGain": float(lot.get("totalGain", 0)),
                        "totalGainPct": float(lot.get("totalGainPct", 0)),
                        "shortType": lot.get("shortType", 0),  # 0=Normal, 1=Wash Sale
                    }
                    
                    # Convert acquiredDate from epoch ms to readable format
                    if lot_data["acquiredDate"]:
                        try:
                            from datetime import datetime
                            ts = lot_data["acquiredDate"] / 1000
                            lot_data["acquiredDateStr"] = datetime.fromtimestamp(ts).strftime("%m/%d/%Y")
                        except:
                            lot_data["acquiredDateStr"] = "Unknown"
                    else:
                        lot_data["acquiredDateStr"] = "Unknown"
                    
                    # Determine term label
                    lot_data["termLabel"] = "Long-Term" if lot_data["termCode"] == 2 else "Short-Term"
                    
                    # Wash sale indicator
                    lot_data["isWashSale"] = lot_data["shortType"] == 1
                    
                    lots.append(lot_data)
        
        return lots
        
    except Exception as e:
        print(f"[fetch_position_lots] Error: {e}")
        return []


def _generate_client_order_id() -> str:
    """Generate a unique 20-char max client order ID."""
    import uuid
    return str(uuid.uuid4())[:20]


def _build_preview_payload(
    ticker: str,
    quantity: int,
    action: str,
    price_type: str = "MARKET",
    limit_price: Optional[float] = None,
    stop_price: Optional[float] = None,
    order_term: str = "GOOD_FOR_DAY",
    client_order_id: str = None,
    lots: Optional[List[Dict]] = None,
) -> Dict:
    """
    Builds the E*TRADE PreviewOrderRequest payload.
    
    EXACT structure that works (from test_order_api.py).
    
    Args:
        lots: Optional list of lot specifications for SELL orders.
              Each dict should have: {"id": positionLotId, "size": shares_to_sell}
    """
    if client_order_id is None:
        client_order_id = _generate_client_order_id()
    
    # Build base Instrument object
    instrument = {
        "Product": {
            "securityType": "EQ",
            "symbol": ticker.upper()
        },
        "orderAction": action.upper(),
        "quantityType": "QUANTITY",
        "quantity": str(quantity)
    }
    
    # Add lot specification for SELL orders with specific lots
    if action.upper() == "SELL" and lots:
        instrument["Lots"] = {
            "Lot": [{"id": lot["id"], "size": lot["size"]} for lot in lots]
        }
    
    return {
        "PreviewOrderRequest": {
            "orderType": "EQ",
            "clientOrderId": client_order_id,
            "Order": [
                {
                    "allOrNone": "false",
                    "priceType": price_type.upper(),
                    "orderTerm": order_term.upper(),
                    "marketSession": "REGULAR",
                    "stopPrice": f"{stop_price:.2f}" if stop_price is not None else "",
                    "limitPrice": f"{limit_price:.2f}" if limit_price is not None else "",
                    "Instrument": [instrument]
                }
            ]
        }
    }


def _build_place_payload(
    ticker: str,
    quantity: int,
    action: str,
    preview_id: str,
    client_order_id: str,
    price_type: str = "MARKET",
    limit_price: Optional[float] = None,
    stop_price: Optional[float] = None,
    order_term: str = "GOOD_FOR_DAY",
    lots: Optional[List[Dict]] = None,
) -> Dict:
    """
    Builds the E*TRADE PlaceOrderRequest payload.
    
    EXACT structure that works (from test_order_api.py).
    
    Args:
        lots: Optional list of lot specifications for SELL orders.
              Each dict should have: {"id": positionLotId, "size": shares_to_sell}
    """
    # Build base Instrument object
    instrument = {
        "Product": {
            "securityType": "EQ",
            "symbol": ticker.upper()
        },
        "orderAction": action.upper(),
        "quantityType": "QUANTITY",
        "quantity": str(quantity)
    }
    
    # Add lot specification for SELL orders with specific lots
    if action.upper() == "SELL" and lots:
        instrument["Lots"] = {
            "Lot": [{"id": lot["id"], "size": lot["size"]} for lot in lots]
        }
    
    return {
        "PlaceOrderRequest": {
            "orderType": "EQ",
            "clientOrderId": client_order_id,
            "PreviewIds": [
                {
                    "previewId": preview_id
                }
            ],
            "Order": [
                {
                    "allOrNone": "false",
                    "priceType": price_type.upper(),
                    "orderTerm": order_term.upper(),
                    "marketSession": "REGULAR",
                    "stopPrice": f"{stop_price:.2f}" if stop_price is not None else "",
                    "limitPrice": f"{limit_price:.2f}" if limit_price is not None else "",
                    "Instrument": [instrument]
                }
            ]
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


def _build_order_xml(
    ticker: str,
    quantity: int,
    action: str,
    price_type: str = "MARKET",
    limit_price: Optional[float] = None,
    stop_price: Optional[float] = None,
    order_term: str = "GOOD_FOR_DAY",
    lot_ids: Optional[List[str]] = None,
    account_id_key: str = None,
    is_preview: bool = True,
) -> str:
    """
    Build XML payload that EXACTLY matches E*TRADE's expected format.
    
    Based on working XML template:
    - NO OrderDetail wrapper - fields go directly under Order
    - routingDestination MUST be present
    - stopPrice and limitPrice ALWAYS present (empty if not used)
    - Single Order element (not array)
    - Single Instrument element (not array)
    """
    client_order_id = f"DELVEX_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Format price values
    limit_price_str = f"{limit_price:.2f}" if limit_price is not None else ""
    stop_price_str = f"{stop_price:.2f}" if stop_price is not None else ""
    
    # Determine the root element based on preview vs. place
    root_element = "PreviewOrderRequest" if is_preview else "PlaceOrderRequest"
    
    xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<{root_element}>
    <orderType>EQ</orderType>
    <clientOrderId>{client_order_id}</clientOrderId>
    <Order>
        <allOrNone>false</allOrNone>
        <priceType>{price_type.upper()}</priceType>
        <orderTerm>{order_term.upper()}</orderTerm>
        <marketSession>REGULAR</marketSession>
        <stopPrice>{stop_price_str}</stopPrice>
        <limitPrice>{limit_price_str}</limitPrice>
        <routingDestination>AUTO</routingDestination>
        <Instrument>
            <Product>
                <securityType>EQ</securityType>
                <symbol>{ticker.upper()}</symbol>
            </Product>
            <orderAction>{action.upper()}</orderAction>
            <quantityType>QUANTITY</quantityType>
            <quantity>{quantity}</quantity>
        </Instrument>
    </Order>
</{root_element}>'''
    
    return xml


def _parse_preview_response(response_text: str, content_type: str = "json") -> Dict:
    """Parse preview response from E*TRADE (JSON or XML)."""
    import xml.etree.ElementTree as ET
    
    result = {
        "estimated_value": 0.0,
        "estimated_commission": 0.0,
        "estimated_total": 0.0,
        "preview_id": None,
        "messages": []
    }
    
    # Try JSON first
    if "json" in content_type.lower() or response_text.strip().startswith("{"):
        try:
            data = json.loads(response_text)
            preview = data.get("PreviewOrderResponse", {})
            
            # Order might be an array or single object
            order_info = preview.get("Order", {})
            if isinstance(order_info, list):
                order_info = order_info[0] if order_info else {}
            
            result["estimated_value"] = float(order_info.get("estimatedOrderValue", 0))
            result["estimated_commission"] = float(order_info.get("estimatedCommission", 0))
            result["estimated_total"] = float(order_info.get("estimatedTotalAmount", 
                result["estimated_value"] + result["estimated_commission"]))
            
            # Preview ID
            preview_ids = preview.get("PreviewIds", {}).get("previewId", [])
            if isinstance(preview_ids, list):
                result["preview_id"] = preview_ids[0] if preview_ids else None
            else:
                result["preview_id"] = preview_ids
            
            # Messages
            msg_container = order_info.get("messages", {})
            msg_list = msg_container.get("Message", []) if isinstance(msg_container, dict) else []
            if not isinstance(msg_list, list):
                msg_list = [msg_list]
            for msg in msg_list:
                msg_text = ""
                if isinstance(msg, dict):
                    msg_text = msg.get("description", str(msg))
                else:
                    msg_text = str(msg)
                
                # Strip status code prefix (e.g., "200|Your order...")
                if "|" in msg_text:
                    parts = msg_text.split("|", 1)
                    # specific heuristic: if left side is digits, remove it
                    # Also strip whitespace to handle " 200 " etc
                    if parts[0].strip().isdigit():
                        msg_text = parts[1]
                
                result["messages"].append(msg_text)
            
            return result
        except json.JSONDecodeError:
            pass  # Fall through to XML parsing
    
    # Try XML parsing
    try:
        root = ET.fromstring(response_text)
        
        # Find values with flexible path searching
        result["estimated_value"] = float(root.findtext(".//estimatedOrderValue", "0"))
        result["estimated_commission"] = float(root.findtext(".//estimatedCommission", "0"))
        result["estimated_total"] = float(root.findtext(".//estimatedTotalAmount", 
            str(result["estimated_value"] + result["estimated_commission"])))
        result["preview_id"] = root.findtext(".//previewId")
        
        # Messages
        for msg_elem in root.findall(".//Message"):
            desc = msg_elem.findtext("description")
            if desc:
                result["messages"].append(desc)
                
    except ET.ParseError:
        pass
    
    return result


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
        Dict with preview details including preview_id and client_order_id
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
        base_url = get_base_url()
        url = f"{base_url}/v1/accounts/{account_id_key}/orders/preview"
        
        # Generate client order ID (needed for both preview and place)
        client_order_id = _generate_client_order_id()
        
        # Build payload using exact working structure
        payload = _build_preview_payload(
            ticker=ticker,
            quantity=quantity,
            action=action,
            price_type=price_type,
            limit_price=limit_price,
            stop_price=stop_price,
            order_term=order_term,
            client_order_id=client_order_id,
            lots=lot_ids
        )
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        response = session.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"Preview failed: HTTP {response.status_code} - {response.text[:300]}",
                "estimated_value": 0,
                "estimated_commission": 0,
                "estimated_total": 0
            }
        
        data = response.json()
        preview_response = data.get("PreviewOrderResponse", {})
        
        # Extract previewId
        preview_ids = preview_response.get("PreviewIds", [])
        if isinstance(preview_ids, dict):
            preview_ids = [preview_ids]
        preview_id = preview_ids[0].get("previewId") if preview_ids else None
        
        # Extract order details
        orders = preview_response.get("Order", [])
        if isinstance(orders, dict):
            orders = [orders]
        order_info = orders[0] if orders else {}
        
        # Extract messages
        messages = []
        msg_container = order_info.get("messages", {})
        if isinstance(msg_container, dict):
            msg_list = msg_container.get("Message", [])
            if isinstance(msg_list, dict):
                msg_list = [msg_list]
            for msg in msg_list:
                msg_text = ""
                if isinstance(msg, dict):
                    msg_text = msg.get("description", "")
                else:
                    msg_text = str(msg)
                
                # Strip status code prefix (e.g., "200|Your order...")
                if "|" in msg_text:
                    parts = msg_text.split("|", 1)
                    # specific heuristic: if left side is digits, remove it
                    if parts[0].strip().isdigit():
                        msg_text = parts[1]
                
                if msg_text:
                    messages.append(msg_text)
        
        est_commission = float(order_info.get("estimatedCommission", 0) or 0)
        est_total = float(order_info.get("estimatedTotalAmount", 0) or 0)
        est_value = float(order_info.get("estimatedOrderValue", 0) or 0)
        if (not est_value) and est_total:
            est_value = max(abs(est_total) - est_commission, 0.0)

        return {
            "success": True,
            "preview_id": preview_id,
            "client_order_id": client_order_id,  # CRITICAL: Pass to place_order
            "ticker": ticker.upper(),
            "action": action.upper(),
            "quantity": quantity,
            "price_type": price_type.upper(),
            "limit_price": limit_price,
            "stop_price": stop_price,
            "order_term": order_term.upper(),
            "estimated_value": est_value,
            "estimated_commission": est_commission,
            "estimated_total": est_total,
            "messages": messages,
            "account": account_num,
            "environment": "SANDBOX" if ETRADE_SANDBOX else "PRODUCTION"
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"{str(e)}\n{traceback.format_exc()}",
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
    client_order_id: Optional[str] = None,
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
        preview_id: Preview ID from preview_order() (REQUIRED)
        client_order_id: Client order ID from preview_order() (REQUIRED)
    
    Returns:
        Dict with order result:
            - success: True if order placed
            - order_id: E*TRADE order ID
            - status: Order status
            - filled_quantity: Shares filled (for market orders, often immediate)
            - error: Error message if failed
    """
    try:
        # Validate required params
        if not preview_id:
            return {
                "success": False,
                "error": "preview_id is required. Call preview_order() first."
            }
        if not client_order_id:
            return {
                "success": False,
                "error": "client_order_id is required. Pass the value from preview_order()."
            }
        
        session = get_etrade_session_safe()
        if session is None:
            return {
                "success": False,
                "error": "Not authenticated. Run 'python etrade_auth.py' to authenticate."
            }
        
        account_id_key, account_num = _get_account_id_key(session)
        base_url = get_base_url()
        url = f"{base_url}/v1/accounts/{account_id_key}/orders/place"
        
        # Build payload using exact working structure
        payload = _build_place_payload(
            ticker=ticker,
            quantity=quantity,
            action=action,
            preview_id=preview_id,
            client_order_id=client_order_id,
            price_type=price_type,
            limit_price=limit_price,
            stop_price=stop_price,
            order_term=order_term,
            lots=lot_ids
        )
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        response = session.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code != 200:
            error_text = response.text[:500] if response.text else "Unknown error"
            return {
                "success": False,
                "error": f"Order failed: HTTP {response.status_code} - {error_text}"
            }
        
        data = response.json()
        place_response = data.get("PlaceOrderResponse", {})
        
        # Extract order ID from OrderIds array
        order_ids = place_response.get("OrderIds", [])
        if isinstance(order_ids, dict):
            order_ids = [order_ids]
        order_id = order_ids[0].get("orderId") if order_ids else None
        
        # Get order details
        orders = place_response.get("Order", [])
        if isinstance(orders, dict):
            orders = [orders]
        order_info = orders[0] if orders else {}
        
        # For market orders, usually fills immediately
        # But status may still be OPEN until confirmed
        status = "PLACED"
        
        result = {
            "success": True,
            "order_id": order_id,
            "ticker": ticker.upper(),
            "action": action.upper(),
            "quantity": quantity,
            "price_type": price_type.upper(),
            "status": status,
            "estimated_commission": float(order_info.get("estimatedCommission", 0) or 0),
            "estimated_total": float(order_info.get("estimatedTotalAmount", 0) or 0),
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
        
        headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
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
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT
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
