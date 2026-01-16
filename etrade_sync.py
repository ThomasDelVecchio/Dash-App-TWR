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

import re

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
    clean = re.sub(r'<[^>]+>', '', clean)
    # Collapse whitespace
    clean = re.sub(r'\s+', ' ', clean).strip()
    # Truncate if too long
    if len(clean) > 200:
        clean = clean[:200] + "..."
    return clean


# ============================================================
# CONFIGURATION
# ============================================================
CASHFLOWS_FILE = "cashflows.csv"
HOLDINGS_FILE = "sample holdings.csv"
HOLDINGS_EXTERNAL_FILE = "holdings_external.csv"  # For positions not in E*TRADE (stock plans, other brokers)
SYNC_STATUS_FILE = "etrade_sync_status.json"

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
    response = session.get(
        f"{base_url}/v1/accounts/list",
        headers={"Accept": "application/json"},
        timeout=(10, 30)  # 10s connect, 30s read
    )
    
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch accounts: {response.status_code} - {_strip_html(response.text)}")
    
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
    # Use configurable timeout (default 15s) to prevent hanging on slow/broken API
    response = session.get(url, params=params, headers={"Accept": "application/json"}, timeout=(10, ETRADE_SYNC_TIMEOUT))
    
    if response.status_code == 204:
        # No content - no transactions in range
        return []
    
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch transactions: {response.status_code} - {_strip_html(response.text)}")
    
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
        
        # Parse date (E*TRADE uses epoch milliseconds or string)
        if isinstance(tx_date, (int, float)):
            date_obj = datetime.fromtimestamp(tx_date / 1000)
        else:
            date_obj = datetime.strptime(str(tx_date), "%Y-%m-%d")
        
        date_str = date_obj.strftime("%m/%d/%Y")
        
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
    def make_key(row):
        return f"{row['_date_norm']}_{row['ticker']}_{round(row['amount'], 2)}"
    
    existing_keys = set(existing_df.apply(make_key, axis=1))
    
    mask = new_df.apply(make_key, axis=1).apply(lambda k: k not in existing_keys)
    
    result = new_df[mask].drop(columns=["_date_norm"])
    
    return result


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
        if last_date:
            # Incremental sync: start from last known transaction
            start_date = last_date + timedelta(days=1)
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
    response = session.get(url, headers={"Accept": "application/json"}, timeout=(10, 30))  # 10s connect, 30s read
    
    if response.status_code == 204:
        return []  # Empty portfolio
    
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch portfolio: {response.status_code} - {_strip_html(response.text)}")
    
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
        
        # Save combined holdings
        new_holdings.to_csv(HOLDINGS_FILE, index=False)
        
        print(f"   ✅ Updated holdings ({len(new_holdings)} total positions)")
        
        # Flag new tickers that need asset_class assignment
        new_tickers = [t for t in new_holdings["ticker"] if t not in metadata and t != "CASH"]
        if new_tickers:
            print(f"   ⚠️  New tickers need asset_class: {new_tickers}")
        
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
