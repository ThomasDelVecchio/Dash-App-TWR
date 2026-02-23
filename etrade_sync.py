"""
E*TRADE Sync Module

Fetches transactions and holdings from E*TRADE API and merges them
with existing local CSV files.

Features:
- Appends new transactions to cashflows.csv (preserving history)
- Updates holdings.csv with current positions (preserving metadata)
- Deduplication to prevent duplicate entries
- Sync status tracking for UI notifications

Usage:
    from etrade_sync import sync_all, get_sync_status
    
    result = sync_all()
    print(result["message"])
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import time
import functools

import re

# ============================================================
# API RESILIENCE CONFIGURATION
# ============================================================
USER_AGENT = "DELVEX-Portfolio-Analytics/1.0"
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds

def retry_on_server_error(max_retries=MAX_RETRIES, backoff_base=RETRY_BACKOFF_BASE):
    """
    Decorator to retry API calls on 5xx errors with exponential backoff.
    E*TRADE occasionally returns 500 errors that resolve on retry.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except RuntimeError as e:
                    error_msg = str(e)
                    # Only retry on server errors (5xx)
                    if "500" in error_msg or "502" in error_msg or "503" in error_msg or "504" in error_msg:
                        wait_time = backoff_base ** attempt
                        print(f"   ⚠️  E*TRADE server error (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        last_exception = e
                    else:
                        raise  # Non-5xx errors should not be retried
            # All retries exhausted
            print(f"   ❌ E*TRADE server error persisted after {max_retries} attempts.")
            print(f"   This is an E*TRADE-side issue. Try again later or check E*TRADE status.")
            raise last_exception
        return wrapper
    return decorator

from etrade_auth import get_etrade_session, get_etrade_session_safe, get_base_url
from config import ETRADE_ACCOUNT_ID, ETRADE_HEADLESS, ETRADE_SKIP_TRANSACTIONS, ETRADE_SYNC_TIMEOUT


def _strip_html(text: str) -> str:
    """Strip HTML tags from error response text for cleaner console output."""
    if not text:
        return text
    # Remove style/script blocks entirely (including content)
    clean = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r'<script[^>]*>.*?</script>', '', clean, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML tags
    clean = re.sub(r'<[^>]+>', ' ', clean)  # Replace with space to avoid word-mashing
    # Collapse whitespace
    clean = re.sub(r'\s+', ' ', clean).strip()
    # Truncate if too long
    if len(clean) > 150:
        clean = clean[:150] + "..."
    return clean


def _format_api_error(status_code: int, raw_text: str) -> str:
    """Format API error into a clean, actionable message."""
    if status_code >= 500:
        return (
            f"E*TRADE server error (HTTP {status_code}). "
            "This is an E*TRADE-side issue, not your code. Try again in a few minutes."
        )
    elif status_code == 401:
        return "Authentication failed. Token may be expired - run `python etrade_auth.py` to re-authenticate."
    elif status_code == 403:
        return "Access denied. Check your E*TRADE API permissions or re-authenticate."
    elif status_code == 404:
        return "Resource not found. Check your account ID configuration."
    elif status_code == 429:
        return "Rate limited by E*TRADE. Wait a few minutes before trying again."
    else:
        # For other errors, include stripped text for context
        stripped = _strip_html(raw_text)
        return f"HTTP {status_code}: {stripped}"


# ============================================================
# CONFIGURATION
# ============================================================
CASHFLOWS_FILE = "cashflows.csv"
HOLDINGS_FILE = "sample holdings.csv"
HOLDINGS_EXTERNAL_FILE = "holdings_external.csv"  # For positions not in E*TRADE (stock plans, other brokers)
SYNC_STATUS_FILE = "etrade_sync_status.json"
ASSET_CLASS_CACHE_FILE = "asset_class_cache.json"

# Sync lookback periods
# First sync needs extended history to capture older positions (E*TRADE allows ~2 years)
# Incremental syncs only need recent activity
FIRST_SYNC_LOOKBACK_DAYS = 730    # ~2 years (E*TRADE API max)
INCREMENTAL_SYNC_LOOKBACK_DAYS = 90

# Transaction type mapping from E*TRADE to our schema
TRANSACTION_TYPE_MAP = {
    # Trades
    "Bought": "TRADE",
    "Sold": "TRADE",
    "Buy": "TRADE",
    "Sell": "TRADE",
    "Misc Trade": "TRADE",
    # Dividends (E*TRADE uses various names)
    "Dividend": "DIVIDEND",
    "Qualified Dividend": "DIVIDEND",
    "Non-Qualified Div": "DIVIDEND",
    "Reinvested Dividend": "DIVIDEND",
    # Interest
    "Interest": "INTEREST",
    "Interest Income": "INTEREST",
    # Cash Flows (Deposits/Withdrawals/Transfers)
    "Deposit": "FLOW",
    "Withdrawal": "FLOW",
    "Transfer": "FLOW",
    "Online Transfer": "FLOW",
    "Funds Received": "FLOW",
    "ACH Deposit": "FLOW",
    "Wire Transfer": "FLOW",
    "Debit Card": "FLOW",
    "Service Fee": "FLOW",
}


# ============================================================
# SYNC STATUS TRACKING
# ============================================================

def load_sync_status() -> Dict:
    """Load last sync status from disk."""
    if not os.path.exists(SYNC_STATUS_FILE):
        return {
            "last_sync": None,
            "last_sync_display": "Never",
            "transactions_added": 0,
            "holdings_updated": False,
            "status": "never_synced",
            "message": "E*TRADE sync has not been run yet"
        }
    
    try:
        with open(SYNC_STATUS_FILE, "r") as f:
            return json.load(f)
    except:
        return {"status": "error", "message": "Could not load sync status"}


def save_sync_status(status: Dict):
    """Save sync status to disk."""
    with open(SYNC_STATUS_FILE, "w") as f:
        json.dump(status, f, indent=2)


def get_sync_status() -> Dict:
    """
    Get current sync status for UI display.
    
    Returns:
        dict with keys:
            - last_sync: ISO timestamp or None
            - last_sync_display: Human-readable string
            - status: "success", "error", "never_synced"
            - message: Status message
            - transactions_added: Count of new transactions
    """
    return load_sync_status()


def load_asset_class_cache() -> Dict[str, Dict]:
    """Load historical asset_class/target_pct cache from disk."""
    if not os.path.exists(ASSET_CLASS_CACHE_FILE):
        return {}
    try:
        with open(ASSET_CLASS_CACHE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_asset_class_cache(cache: Dict[str, Dict]):
    """Persist asset_class/target_pct cache to disk."""
    try:
        with open(ASSET_CLASS_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


# ============================================================
# ACCOUNT DISCOVERY
# ============================================================

def get_account_id_key(session) -> Tuple[str, str]:
    """
    Get the E*TRADE account ID KEY (not the account number).
    
    E*TRADE has two identifiers:
    - Account Number: Human-readable (e.g., "12345678")
    - Account ID Key: Internal encrypted key for API calls
    
    If ETRADE_ACCOUNT_ID is set, finds matching account's ID key.
    Otherwise, uses first brokerage account.
    
    Returns:
        Tuple[str, str]: (accountIdKey, accountNumber) for API calls
    """
    base_url = get_base_url()
    
    # Always fetch account list to get proper accountIdKey
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    response = session.get(
        f"{base_url}/v1/accounts/list",
        headers=headers,
        timeout=(10, 30)  # 10s connect, 30s read
    )
    
    if response.status_code != 200:
        raise RuntimeError(_format_api_error(response.status_code, response.text))
    
    data = response.json()
    
    # Navigate E*TRADE's nested response structure
    accounts = data.get("AccountListResponse", {}).get("Accounts", {}).get("Account", [])
    
    if not accounts:
        raise RuntimeError("No E*TRADE accounts found")
    
    # Ensure accounts is a list
    if not isinstance(accounts, list):
        accounts = [accounts]
    
    # If user specified an account, find it
    if ETRADE_ACCOUNT_ID:
        for acct in accounts:
            acct_num = acct.get("accountId", "")
            acct_key = acct.get("accountIdKey", "")
            
            # Match by account number OR accountIdKey (user might have either)
            if acct_num == ETRADE_ACCOUNT_ID or acct_key == ETRADE_ACCOUNT_ID:
                print(f"   Found account: {acct_num} (key: {acct_key[:8]}...)")
                return acct_key, acct_num
        
        # Account not found - list available accounts
        available = [a.get("accountId", "unknown") for a in accounts]
        raise RuntimeError(
            f"Account '{ETRADE_ACCOUNT_ID}' not found. "
            f"Available accounts: {available}"
        )
    
    # No account specified - use first brokerage account
    first_acct = accounts[0]
    acct_key = first_acct.get("accountIdKey", "")
    acct_num = first_acct.get("accountId", "")
    print(f"   Using first account: {acct_num}")
    return acct_key, acct_num


# ============================================================
# TRANSACTION SYNC
# ============================================================

def fetch_etrade_transactions(session, account_id: str, start_date: datetime = None) -> List[Dict]:
    """
    Fetch transactions from E*TRADE API.
    
    Args:
        session: Authenticated OAuth session
        account_id: E*TRADE account ID key (NOT account number)
        start_date: Fetch transactions after this date (default: 90 days ago)
    
    Returns:
        List of transaction dicts in E*TRADE format
    """
    base_url = get_base_url()
    
    if start_date is None:
        start_date = datetime.now() - timedelta(days=90)
    
    # E*TRADE expects dates in MMDDYYYY format (no slashes)
    start_str = start_date.strftime("%m%d%Y")
    end_str = datetime.now().strftime("%m%d%Y")
    
    url = f"{base_url}/v1/accounts/{account_id}/transactions"
    params = {
        "startDate": start_str,
        "endDate": end_str,
    }
    
    # Request JSON explicitly (E*TRADE defaults to XML sometimes)
    # Use configurable timeout (default 45s) to prevent hanging on slow/broken API
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    response = session.get(url, params=params, headers=headers, timeout=(10, ETRADE_SYNC_TIMEOUT))
    
    if response.status_code == 204:
        # No content - no transactions in range
        return []
    
    if response.status_code != 200:
        raise RuntimeError(_format_api_error(response.status_code, response.text))
    
    data = response.json()
    
    # Navigate E*TRADE's nested response
    transactions = (
        data.get("TransactionListResponse", {})
        .get("Transaction", [])
    )
    
    # Ensure it's a list (single transaction comes as dict)
    if isinstance(transactions, dict):
        transactions = [transactions]
    
    return transactions


def transform_etrade_transaction(tx: Dict) -> Optional[Dict]:
    """
    Transform E*TRADE transaction to cashflows.csv format.
    
    Args:
        tx: E*TRADE transaction dict
        
    Returns:
        Dict with keys: date, ticker, shares, amount, type
        Or None if transaction should be skipped
    """
    try:
        # Extract fields from E*TRADE format
        # Note: E*TRADE structure varies by transaction type
        
        tx_date = tx.get("transactionDate")
        tx_type = tx.get("transactionType", "")
        description = str(tx.get("description", "") or "")
        desc_lower = description.lower()
        
        # Parse date (E*TRADE uses epoch milliseconds or string)
        # CRITICAL: Epoch timestamps are in UTC; convert to local for display
        if isinstance(tx_date, (int, float)):
            # E*TRADE returns epoch milliseconds in UTC
            from datetime import timezone
            date_obj = datetime.fromtimestamp(tx_date / 1000, tz=timezone.utc).replace(tzinfo=None)
        else:
            date_obj = datetime.strptime(str(tx_date), "%Y-%m-%d")
        
        date_str = date_obj.strftime("%m/%d/%Y")
        
        # Special case: "Misc Trade" that moves proceeds to a cash-hold bucket
        # (not part of brokerage cash). Treat as external cash flow OUT/IN.
        # DISABLED (2026-01-21): This causes double-counting of cash outflow during settlement.
        # We prefer to track Trade Date accounting, so we ignore these holds.
        # if tx_type == "Misc Trade" and "cash hold" in desc_lower:
        #     return {
        #         "date": date_str,
        #         "ticker": "CASH",
        #         "shares": 0.0,
        #         "amount": float(tx.get("amount", 0) or 0),
        #         "type": "FLOW"
        #     }

        # Map transaction type
        our_type = TRANSACTION_TYPE_MAP.get(tx_type, None)
        if our_type is None:
            print(f"⚠️  Unknown transaction type: {tx_type}")
            return None
        
        # Extract brokerage-specific fields
        brokerage = tx.get("brokerage", {})
        
        # Get symbol
        product = brokerage.get("product", {})
        ticker = product.get("symbol", "CASH")
        
        # Get quantity and amount
        # NOTE: E*TRADE returns 'amount' at TOP LEVEL of transaction, NOT inside brokerage object
        quantity = float(brokerage.get("quantity", 0) or 0)
        amount = float(tx.get("amount", 0) or 0)

        # Guard: E*TRADE sometimes emits a CASH "trade" entry that mirrors
        # the cash leg of a security trade. This would double-count cash.
        if our_type == "TRADE" and ticker == "CASH" and abs(quantity) < 1e-9:
            return None
        
        # Apply sign conventions for our schema:
        # - Shares: positive for buy, negative for sell
        # - Amount: negative for buy (cash out), positive for sell (cash in)
        
        if tx_type in ["Sold", "Sell"]:
            quantity = -abs(quantity)  # Negative shares for sell
            amount = abs(amount)       # Positive amount (cash in)
        elif tx_type in ["Bought", "Buy"]:
            quantity = abs(quantity)   # Positive shares for buy
            amount = -abs(amount)      # Negative amount (cash out)
        elif tx_type == "Dividend":
            quantity = 0               # No shares change
            amount = abs(amount)       # Positive (income)
            our_type = "DIVIDEND"
        elif tx_type == "Interest":
            ticker = "CASH"
            quantity = 0
            amount = abs(amount)
            our_type = "INTEREST"
        elif our_type == "FLOW":
            ticker = "CASH"
            quantity = 0
            # Keep original sign for deposits/withdrawals
        
        return {
            "date": date_str,
            "ticker": ticker.upper(),
            "shares": quantity,
            "amount": amount,
            "type": our_type
        }
        
    except Exception as e:
        print(f"⚠️  Error transforming transaction: {e}")
        print(f"   Raw data: {tx}")
        return None


def get_last_transaction_date() -> Optional[datetime]:
    """Get the most recent transaction date from cashflows.csv."""
    if not os.path.exists(CASHFLOWS_FILE):
        return None
    
    try:
        df = pd.read_csv(CASHFLOWS_FILE)
        df["date"] = pd.to_datetime(df["date"])
        return df["date"].max()
    except:
        return None


def deduplicate_transactions(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove transactions that already exist in the file.
    Uses composite key: date + ticker + amount (rounded to 2 decimals)
    """
    if existing_df.empty:
        return new_df
    
    if new_df.empty:
        return new_df
    
    # Normalize dates for comparison
    existing_df = existing_df.copy()
    new_df = new_df.copy()
    
    existing_df["_date_norm"] = pd.to_datetime(existing_df["date"]).dt.date
    new_df["_date_norm"] = pd.to_datetime(new_df["date"]).dt.date
    
    # Create composite keys
    # Include 'type' to distinguish dividends, trades, etc. on same ticker/date
    def make_key(row):
        tx_type = row.get('type', 'UNKNOWN')
        return f"{row['_date_norm']}_{row['ticker']}_{round(row['amount'], 2)}_{tx_type}"
    
    existing_keys = set(existing_df.apply(make_key, axis=1))
    
    mask = new_df.apply(make_key, axis=1).apply(lambda k: k not in existing_keys)
    
    result = new_df[mask].drop(columns=["_date_norm"])
    
    return result


@retry_on_server_error()
def sync_transactions() -> Tuple[int, str]:
    """
    Sync transactions from E*TRADE to cashflows.csv.
    
    Returns:
        Tuple of (count of new transactions, status message)
    """
    # Check if transaction sync is disabled (E*TRADE API may be slow/broken)
    if ETRADE_SKIP_TRANSACTIONS:
        print("\n📊 Skipping E*TRADE transactions sync (ETRADE_SKIP_TRANSACTIONS=true)")
        return 0, "Transaction sync skipped (using existing cashflows.csv)"
    
    print("\n📊 Syncing E*TRADE transactions...")
    
    try:
        # Use safe session in headless mode (Colab) to avoid browser prompts
        if ETRADE_HEADLESS:
            session = get_etrade_session_safe()
            if session is None:
                return -1, "Token expired. Run etrade_auth.py to re-authenticate."
        else:
            session = get_etrade_session()
        
        account_id_key, account_num = get_account_id_key(session)
        
        print(f"   Account ID: {account_num}")
        
        # Determine start date based on whether this is first sync or incremental
        last_date = get_last_transaction_date()
        today = datetime.now().date()
        
        if last_date:
            # Incremental sync: start from last known transaction
            start_date = last_date + timedelta(days=1)
            
            # CRITICAL FIX: Guard against future dates
            # If last transaction is today (or somehow in the future), start_date
            # would be tomorrow+, which E*TRADE rejects with HTTP 500.
            # Solution: Re-fetch from today to catch any new same-day transactions.
            # Deduplication will filter out already-synced ones.
            if start_date.date() > today:
                start_date = datetime.combine(today, datetime.min.time())
                print(f"   Re-syncing today ({today}) to catch new transactions...")
            else:
                print(f"   Fetching transactions since: {start_date.date()}")
        else:
            # First sync: fetch maximum available history to capture older positions
            # This prevents reconciliation errors for users with positions > 90 days old
            start_date = datetime.now() - timedelta(days=FIRST_SYNC_LOOKBACK_DAYS)
            print(f"   🆕 First sync detected. Fetching last {FIRST_SYNC_LOOKBACK_DAYS} days (~2 years).")
            print(f"   This ensures older positions are properly captured.")
        
        # Fetch from E*TRADE (use accountIdKey, not account number)
        raw_transactions = fetch_etrade_transactions(session, account_id_key, start_date)
        print(f"   Found {len(raw_transactions)} transactions from E*TRADE")
        
        if not raw_transactions:
            return 0, "No new transactions found"
        
        # Transform to our format
        transformed = []
        for tx in raw_transactions:
            result = transform_etrade_transaction(tx)
            if result:
                transformed.append(result)
        
        if not transformed:
            return 0, "No compatible transactions found"
        
        new_df = pd.DataFrame(transformed)
        
        # Load existing and deduplicate
        if os.path.exists(CASHFLOWS_FILE):
            existing_df = pd.read_csv(CASHFLOWS_FILE)
        else:
            existing_df = pd.DataFrame()
        
        unique_new = deduplicate_transactions(existing_df, new_df)
        
        if unique_new.empty:
            return 0, "All transactions already exist in file"
        
        # Append to file
        if existing_df.empty:
            unique_new.to_csv(CASHFLOWS_FILE, index=False)
        else:
            combined = pd.concat([existing_df, unique_new], ignore_index=True)
            combined.to_csv(CASHFLOWS_FILE, index=False)
        
        count = len(unique_new)
        print(f"   ✅ Added {count} new transactions")
        
        return count, f"Added {count} new transactions"
        
    except Exception as e:
        error_msg = f"Transaction sync failed: {e}"
        print(f"   ❌ {error_msg}")
        return -1, error_msg


# ============================================================
# HOLDINGS SYNC
# ============================================================

def fetch_etrade_holdings(session, account_id: str) -> List[Dict]:
    """
    Fetch current portfolio positions from E*TRADE.
    
    Args:
        account_id: E*TRADE account ID key (NOT account number)
    
    Returns:
        List of position dicts
    """
    base_url = get_base_url()
    url = f"{base_url}/v1/accounts/{account_id}/portfolio"
    
    # Request JSON explicitly (E*TRADE defaults to XML sometimes)
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    response = session.get(url, headers=headers, timeout=(10, 30))  # 10s connect, 30s read
    
    if response.status_code == 204:
        return []  # Empty portfolio
    
    if response.status_code != 200:
        raise RuntimeError(_format_api_error(response.status_code, response.text))
    
    data = response.json()
    
    # Navigate E*TRADE's nested response
    portfolio = data.get("PortfolioResponse", {}).get("AccountPortfolio", [])
    
    if isinstance(portfolio, dict):
        portfolio = [portfolio]
    
    positions = []
    for account in portfolio:
        pos_list = account.get("Position", [])
        if isinstance(pos_list, dict):
            pos_list = [pos_list]
        positions.extend(pos_list)
    
    return positions


def transform_etrade_position(pos: Dict) -> Optional[Dict]:
    """
    Transform E*TRADE position to holdings.csv format.
    
    Preserves only ticker and shares from E*TRADE.
    asset_class and target_pct must be merged from existing file.
    """
    try:
        product = pos.get("Product", {})
        ticker = product.get("symbol", "").upper()
        
        quantity = float(pos.get("quantity", 0) or 0)
        
        if not ticker:
            return None
        
        return {
            "ticker": ticker,
            "shares": quantity
        }
        
    except Exception as e:
        print(f"⚠️  Error transforming position: {e}")
        return None


@retry_on_server_error()
def sync_holdings() -> Tuple[bool, str]:
    """
    Sync holdings from E*TRADE to sample holdings.csv.
    
    HYBRID MODE: Preserves existing asset_class and target_pct columns.
    
    Returns:
        Tuple of (success, status message)
    """
    print("\n📈 Syncing E*TRADE holdings...")
    
    try:
        # Use safe session in headless mode (Colab) to avoid browser prompts
        if ETRADE_HEADLESS:
            session = get_etrade_session_safe()
            if session is None:
                return False, "Token expired. Run etrade_auth.py to re-authenticate."
        else:
            session = get_etrade_session()
        
        account_id_key, account_num = get_account_id_key(session)
        
        # Fetch positions (use accountIdKey, not account number)
        raw_positions = fetch_etrade_holdings(session, account_id_key)
        print(f"   Found {len(raw_positions)} positions")
        
        # Transform
        positions = []
        for pos in raw_positions:
            result = transform_etrade_position(pos)
            if result:
                positions.append(result)
        
        new_holdings = pd.DataFrame(positions)
        
        # Load existing metadata from both sample holdings AND external holdings
        metadata = {}
        asset_class_cache = load_asset_class_cache()
        if os.path.exists(HOLDINGS_FILE):
            existing = pd.read_csv(HOLDINGS_FILE)
            existing["ticker"] = existing["ticker"].str.upper()
            
            # Build metadata lookup
            for _, row in existing.iterrows():
                ticker = row["ticker"]
                metadata[ticker] = {
                    "asset_class": row.get("asset_class", "Unknown"),
                    "target_pct": row.get("target_pct", 0)
                }

        # Merge cached asset_class metadata (persists exited tickers)
        for t, meta in asset_class_cache.items():
            if t not in metadata:
                metadata[t] = {
                    "asset_class": meta.get("asset_class", "Unknown"),
                    "target_pct": meta.get("target_pct", 0)
                }
        
        # ALSO load metadata from external holdings (for new tickers check)
        if os.path.exists(HOLDINGS_EXTERNAL_FILE):
            try:
                ext_existing = pd.read_csv(HOLDINGS_EXTERNAL_FILE)
                ext_existing["ticker"] = ext_existing["ticker"].str.upper()
                for _, row in ext_existing.iterrows():
                    ticker = row["ticker"]
                    if ticker not in metadata:  # Don't overwrite if already exists
                        metadata[ticker] = {
                            "asset_class": row.get("asset_class", "Unknown"),
                            "target_pct": row.get("target_pct", 0)
                        }
            except Exception:
                pass  # External file may not exist or be malformed
        
        # Merge metadata with new positions
        def get_asset_class(ticker):
            return metadata.get(ticker, {}).get("asset_class", "Unknown")
        
        def get_target_pct(ticker):
            return metadata.get(ticker, {}).get("target_pct", 0)
        
        new_holdings["asset_class"] = new_holdings["ticker"].apply(get_asset_class)
        new_holdings["target_pct"] = new_holdings["ticker"].apply(get_target_pct)
        
        # Fetch cash balance from account
        cash_balance = fetch_cash_balance(session, account_id_key)
        
        # Add CASH row
        cash_row = pd.DataFrame([{
            "ticker": "CASH",
            "shares": cash_balance,
            "asset_class": "CASH",
            "target_pct": metadata.get("CASH", {}).get("target_pct", 0)
        }])
        
        new_holdings = pd.concat([new_holdings, cash_row], ignore_index=True)
        
        # ============================================================
        # MERGE EXTERNAL HOLDINGS (Stock Plans, Other Brokers, etc.)
        # ============================================================
        # External holdings are tracked in a separate file and merged
        # at sync time. This ensures GIPS-compliant reconciliation with
        # cashflows.csv which contains the full transaction history.
        
        if os.path.exists(HOLDINGS_EXTERNAL_FILE):
            try:
                print(f"   📁 Loading external holdings from {HOLDINGS_EXTERNAL_FILE}")
                external_df = pd.read_csv(HOLDINGS_EXTERNAL_FILE)
                external_df.columns = [c.lower() for c in external_df.columns]
                external_df["ticker"] = external_df["ticker"].str.upper()
                
                # Ensure required columns exist
                if "asset_class" not in external_df.columns:
                    external_df["asset_class"] = "Unknown"
                if "target_pct" not in external_df.columns:
                    external_df["target_pct"] = 0
                
                # Check for duplicates (position in both E*TRADE and external)
                etrade_tickers = set(new_holdings["ticker"].unique())
                external_tickers = set(external_df["ticker"].unique())
                duplicates = etrade_tickers & external_tickers
                
                if duplicates:
                    print(f"   ⚠️  Warning: Tickers in both E*TRADE and external: {duplicates}")
                    print(f"       E*TRADE positions will be used for: {duplicates}")
                    # Remove duplicates from external (E*TRADE takes precedence)
                    external_df = external_df[~external_df["ticker"].isin(duplicates)]
                
                # Merge external positions
                if not external_df.empty:
                    new_holdings = pd.concat([new_holdings, external_df], ignore_index=True)
                    print(f"   ✅ Merged {len(external_df)} external position(s)")
            except Exception:
                # Optional file; ignore malformed/empty data
                pass
        
        # ============================================================
        # PRESERVE EXITED TICKERS (0-share positions with metadata)
        # ============================================================
        # E*TRADE API only returns active positions. To preserve historical
        # asset_class and target_pct for exited tickers, we carry forward
        # any 0-share rows from the previous holdings file.
        
        if os.path.exists(HOLDINGS_FILE):
            try:
                prev_holdings = pd.read_csv(HOLDINGS_FILE)
                prev_holdings["ticker"] = prev_holdings["ticker"].str.upper()
                
                # Find exited tickers (0 shares) in previous file
                exited_mask = prev_holdings["shares"].abs() < 1e-6
                exited_rows = prev_holdings[exited_mask].copy()
                
                if not exited_rows.empty:
                    # Only keep exited tickers not already in new_holdings
                    current_tickers = set(new_holdings["ticker"].unique())
                    exited_rows = exited_rows[~exited_rows["ticker"].isin(current_tickers)]
                    
                    if not exited_rows.empty:
                        new_holdings = pd.concat([new_holdings, exited_rows], ignore_index=True)
                        print(f"   📜 Preserved {len(exited_rows)} exited ticker(s)")
            except Exception as e:
                print(f"   ⚠️  Could not preserve exited tickers: {e}")

        # ============================================================
        # PRESERVE HISTORICAL TICKERS FROM CASHFLOWS
        # ============================================================
        # If a ticker was fully exited today, it may no longer appear
        # in E*TRADE holdings and might not exist in the previous holdings
        # file yet. Ensure any ticker present in cashflows.csv is retained
        # as a 0-share row for attribution and audit consistency.
        if os.path.exists(CASHFLOWS_FILE):
            try:
                flows_df = pd.read_csv(CASHFLOWS_FILE)
                flows_df.columns = [c.lower() for c in flows_df.columns]
                if "ticker" in flows_df.columns:
                    flow_tickers = set(
                        flows_df["ticker"].astype(str).str.upper().unique()
                    )
                    flow_tickers.discard("CASH")

                    current_tickers = set(new_holdings["ticker"].unique())
                    missing = sorted(flow_tickers - current_tickers)

                    if missing:
                        add_rows = []
                        for t in missing:
                            add_rows.append({
                                "ticker": t,
                                "shares": 0.0,
                                "asset_class": metadata.get(t, {}).get("asset_class", "Unknown"),
                                "target_pct": metadata.get(t, {}).get("target_pct", 0),
                            })
                        new_holdings = pd.concat([new_holdings, pd.DataFrame(add_rows)], ignore_index=True)
                        print(f"   📜 Preserved {len(add_rows)} historical ticker(s) from cashflows")
            except Exception as e:
                print(f"   ⚠️  Could not preserve historical tickers: {e}")
        
        # Save combined holdings
        new_holdings.to_csv(HOLDINGS_FILE, index=False)
        
        print(f"   ✅ Updated holdings ({len(new_holdings)} total positions)")
        
        # Flag new tickers that need asset_class assignment
        new_tickers = [t for t in new_holdings["ticker"] if t not in metadata and t != "CASH"]
        if new_tickers:
            print(f"   ⚠️  New tickers need asset_class: {new_tickers}")

        # Update asset class cache with any known classifications
        updated_cache = asset_class_cache.copy()
        for _, row in new_holdings.iterrows():
            t = row.get("ticker")
            ac = row.get("asset_class")
            if t and ac and ac != "Unknown":
                updated_cache[str(t).upper()] = {
                    "asset_class": ac,
                    "target_pct": row.get("target_pct", 0)
                }
        save_asset_class_cache(updated_cache)
        
        return True, f"Updated {len(new_holdings)} positions"
        
    except Exception as e:
        error_msg = f"Holdings sync failed: {e}"
        print(f"   ❌ {error_msg}")
        return False, error_msg


def fetch_cash_balance(session, account_id_key: str) -> float:
    """Fetch cash balance from account using accountIdKey."""
    base_url = get_base_url()
    url = f"{base_url}/v1/accounts/{account_id_key}/balance"
    
    params = {"instType": "BROKERAGE", "realTimeNAV": "true"}
    headers = {"Accept": "application/json"}
    
    response = session.get(url, params=params, headers=headers, timeout=(10, 30))  # 10s connect, 30s read
    
    if response.status_code != 200:
        print(f"   ⚠️  Could not fetch cash balance: {response.status_code}")
        return 0.0
    
    data = response.json()
    
    # Navigate to cash balance
    computed = data.get("BalanceResponse", {}).get("Computed", {})
    cash = computed.get("cashAvailableForInvestment", 0) or computed.get("cashBalance", 0)
    
    return float(cash or 0)


# ============================================================
# SETTLEMENT BRIDGE CLEANUP
# ============================================================

SETTLEMENT_BRIDGE_FILE = "settlement_bridges.json"


def _cleanup_settlement_bridges() -> int:
    """
    Remove settlement bridge FLOW rows from cashflows.csv when the
    real transaction has been synced from E*TRADE.

    Logic:
    - For each active bridge entry in settlement_bridges.json:
      1. Search cashflows.csv for a REAL matching row (same amount ±$1,
         same date ±5 days, type=FLOW) that is NOT the bridge row itself.
      2. If found: the API caught up → remove the bridge FLOW row from
         cashflows.csv and mark the bridge entry as "retired".
      3. If bridge is >10 days old with no match: log a warning.

    Returns count of bridges cleaned up.
    """
    if not os.path.exists(SETTLEMENT_BRIDGE_FILE):
        return 0

    try:
        with open(SETTLEMENT_BRIDGE_FILE, "r") as f:
            bridge_data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return 0

    active = [b for b in bridge_data.get("bridges", []) if b.get("status") == "active"]
    if not active:
        return 0

    if not os.path.exists(CASHFLOWS_FILE):
        return 0

    df = pd.read_csv(CASHFLOWS_FILE)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = df["amount"].astype(float)
    if "type" in df.columns:
        df["type"] = df["type"].fillna("").astype(str).str.upper()

    cleaned = 0
    now = datetime.now()

    for bridge in active:
        b_date = pd.to_datetime(bridge["date"])
        b_amount = float(bridge["amount"])
        b_created = datetime.fromisoformat(bridge.get("created_at", now.isoformat()))
        age_days = (now - b_created).days

        # Find the bridge's own row in cashflows (to remove it)
        bridge_mask = (
            (df["type"] == "FLOW") &
            (df["ticker"].astype(str).str.upper() == "CASH") &
            (abs(df["amount"] - b_amount) < 0.01) &
            (abs((df["date"] - b_date).dt.days) <= 1)
        )

        # Now check if there's ALSO a real matching row with different index
        # A "real" row would have been synced by the API — same amount ±$1
        # but we look for ANY FLOW matching in a 5-day window
        real_match_mask = (
            (df["type"] == "FLOW") &
            (df["ticker"].astype(str).str.upper() == "CASH") &
            (abs(df["amount"] - b_amount) <= 1.00) &
            (abs((df["date"] - b_date).dt.days) <= 5)
        )

        # Count: if there are 2+ matching rows, the real one is here
        bridge_indices = df[bridge_mask].index.tolist()
        real_indices = df[real_match_mask].index.tolist()

        # The API has synced a matching row if we find more matches
        # than just the bridge itself
        if len(real_indices) > len(bridge_indices) and bridge_indices:
            # Remove the bridge row (keep the API-synced one)
            df = df.drop(index=bridge_indices)
            bridge["status"] = "retired"
            bridge["retired_at"] = now.isoformat()
            bridge["retired_reason"] = "Matched synced transaction from E*TRADE"
            cleaned += 1
            print(f"🔄 Settlement bridge retired: ${b_amount:,.2f} on {bridge['date']} — API caught up")

        elif age_days > 10:
            print(
                f"⚠️  Settlement bridge is {age_days} days old with no API match: "
                f"${b_amount:,.2f} on {bridge['date']}. Check E*TRADE manually."
            )

    if cleaned > 0:
        # Write cleaned cashflows back
        df.to_csv(CASHFLOWS_FILE, index=False)
        # Save updated bridge statuses
        with open(SETTLEMENT_BRIDGE_FILE, "w") as f:
            json.dump(bridge_data, f, indent=2, default=str)

    return cleaned


# ============================================================
# MAIN SYNC FUNCTION
# ============================================================

def sync_all(sync_holdings_flag: bool = True) -> Dict:
    """
    Run full E*TRADE sync (transactions + optionally holdings).
    
    Args:
        sync_holdings_flag: If True, also sync holdings (default: True)
    
    Returns:
        Dict with sync results for UI notification:
            - status: "success", "partial", "error"
            - message: Human-readable summary
            - last_sync: ISO timestamp
            - last_sync_display: Formatted time string
            - transactions_added: Count
            - holdings_updated: Boolean
    """
    print("\n" + "="*60)
    print("E*TRADE SYNC")
    print("="*60)
    
    now = datetime.now()
    result = {
        "last_sync": now.isoformat(),
        "last_sync_display": now.strftime("%I:%M %p"),
        "transactions_added": 0,
        "holdings_updated": False,
        "status": "success",
        "message": ""
    }
    
    messages = []
    has_error = False
    
    # Sync transactions
    tx_count, tx_msg = sync_transactions()
    result["transactions_added"] = max(tx_count, 0)
    
    if tx_count < 0:
        has_error = True
        messages.append(f"❌ {tx_msg}")
    elif tx_count == 0:
        messages.append("✓ Transactions up to date")
    else:
        messages.append(f"✓ {tx_msg}")
    
    # Sync holdings
    if sync_holdings_flag:
        holdings_ok, holdings_msg = sync_holdings()
        result["holdings_updated"] = holdings_ok
        
        if not holdings_ok:
            has_error = True
            messages.append(f"❌ {holdings_msg}")
        else:
            messages.append(f"✓ {holdings_msg}")
    
    # Clean up settlement bridges that the API now covers
    bridge_cleaned = _cleanup_settlement_bridges()
    if bridge_cleaned > 0:
        messages.append(f"✓ Retired {bridge_cleaned} settlement bridge(s)")

    # Set final status
    if has_error:
        if result["transactions_added"] > 0 or result["holdings_updated"]:
            result["status"] = "partial"
        else:
            result["status"] = "error"
    
    result["message"] = " | ".join(messages)
    
    # Save status
    save_sync_status(result)
    
    print("\n" + "="*60)
    print(f"Sync completed: {result['message']}")
    print("="*60 + "\n")
    
    return result


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    """
    Run sync directly:
        python etrade_sync.py
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Sync E*TRADE data")
    parser.add_argument("--transactions-only", action="store_true", 
                        help="Only sync transactions, not holdings")
    parser.add_argument("--status", action="store_true",
                        help="Show last sync status and exit")
    
    args = parser.parse_args()
    
    if args.status:
        status = get_sync_status()
        print(f"Last sync: {status.get('last_sync_display', 'Never')}")
        print(f"Status: {status.get('status', 'unknown')}")
        print(f"Message: {status.get('message', 'N/A')}")
    else:
        result = sync_all(sync_holdings_flag=not args.transactions_only)
        print(f"\nResult: {result['status']}")
