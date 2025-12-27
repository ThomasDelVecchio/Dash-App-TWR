import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import requests
import json
import os
from config import FMP_API_KEY

# ============================================================
# CONFIG
# ============================================================
HOLDINGS_FILE = "sample holdings.csv"
CASHFLOWS_FILE = "cashflows.csv"
PRICE_LOOKBACK_YEARS = 10
METADATA_CACHE_FILE = "metadata_cache.json"

# Simple in-memory cache for price history to keep horizons consistent
_PRICE_CACHE = {}
_METADATA_CACHE = {}

# Global set to track tickers we've already warned about to avoid console spam
_REPORTED_MISSING = set()

# ------------------------------------------------------------
# Metadata Cache Management
# ------------------------------------------------------------

def load_metadata_cache():
    global _METADATA_CACHE
    if os.path.exists(METADATA_CACHE_FILE):
        try:
            with open(METADATA_CACHE_FILE, "r") as f:
                _METADATA_CACHE = json.load(f)
        except Exception:
            _METADATA_CACHE = {}

def save_metadata_cache():
    try:
        with open(METADATA_CACHE_FILE, "w") as f:
            json.dump(_METADATA_CACHE, f, indent=2)
    except Exception:
        pass

# Initialize cache on module load
load_metadata_cache()

# ------------------------------------------------------------
# Sector Loading Logic (FMP -> YF -> Equity)
# ------------------------------------------------------------

def fetch_etf_sectors(ticker: str) -> dict:
    """
    Fetches sector weightings for a ticker with priority:
    1. Cache
    2. FMP API (Primary for ETFs)
    3. yfinance (Secondary for ETFs)
    4. Equity Fallback (Single Sector = 100%)
    
    Returns:
        dict: { "SectorName": percent_float, ... }
    """
    ticker = ticker.upper()
    if ticker in _METADATA_CACHE:
        return _METADATA_CACHE[ticker]

    weights = {}
    
    # 1. Try FMP API
    weights = fetch_fmp_sector_weights(ticker)
    
    # 2. Try yfinance
    if not weights:
        weights = fetch_yf_sector_weights(ticker)
        
    # 3. Equity Fallback
    if not weights:
        sector = get_equity_sector(ticker)
        if sector:
            weights = {sector: 100.0}
            
    # Cache result (even if empty, to avoid re-fetching)
    _METADATA_CACHE[ticker] = weights
    save_metadata_cache()
    
    return weights

def fetch_fmp_sector_weights(ticker: str) -> dict:
    """Fetch ETF sector weights from Financial Modeling Prep."""
    if not FMP_API_KEY or FMP_API_KEY == "demo":
        return {}
        
    try:
        url = f"https://financialmodelingprep.com/api/v3/etf-sector-weightings/{ticker}?apikey={FMP_API_KEY}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            # FMP returns list of dicts: [{'sector': 'Technology', 'weightPercentage': '25.5%'}, ...]
            weights = {}
            for item in data:
                sector = item.get("sector", "")
                pct_str = item.get("weightPercentage", "0").replace("%", "")
                try:
                    pct = float(pct_str)
                    if sector and pct > 0:
                        weights[sector] = pct
                except ValueError:
                    continue
            return weights
    except Exception as e:
        print(f"FMP fetch failed for {ticker}: {e}")
        
    return {}

def fetch_yf_sector_weights(ticker: str) -> dict:
    """Fetch ETF sector weights from yfinance."""
    try:
        t = yf.Ticker(ticker)
        # funds_data.sector_weightings returns a dict like {'technology': 0.25, ...}
        # Note: yfinance returns decimals (0.25), FMP returns percents (25.0)
        # We need to normalize to percents (0-100)
        info = t.funds_data
        if info and hasattr(info, 'sector_weightings'):
            raw = info.sector_weightings
            if raw:
                # Convert keys to Title Case and values to Percent
                weights = {}
                for k, v in raw.items():
                    sector_name = k.replace("_", " ").title()
                    weights[sector_name] = v * 100.0
                return weights
    except Exception:
        pass
        
    return {}

def get_equity_sector(ticker: str) -> str:
    """Fetch single sector for equity fallback."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        # Try 'sector' field
        return info.get("sector", "Other")
    except Exception:
        return "Other"

# ------------------------------------------------------------
# Load holdings (your schema)
# ------------------------------------------------------------

def load_holdings(path: str = HOLDINGS_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    required = {"ticker", "shares"}
    if not required.issubset(df.columns):
        raise ValueError(f"Holdings must contain columns: {required}")

    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["shares"] = df["shares"].astype(float)

    if "asset_class" not in df.columns:
        df["asset_class"] = "Unknown"
    if "target_pct" not in df.columns:
        df["target_pct"] = np.nan

    return df


# ------------------------------------------------------------
# Load cashflows for PORTFOLIO TWR (external flows only)
# ------------------------------------------------------------

def load_cashflows_external(path: str = CASHFLOWS_FILE) -> pd.DataFrame:
    """
    For portfolio TWR we ONLY want external flows:
      - Deposits/withdrawals (CASH)
      - Or rows with shares == 0 (if you encode flows that way)

    Trades (buys/sells) MUST be excluded from TWR.
    """
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    if "date" not in df.columns or "amount" not in df.columns:
        raise ValueError("cashflows.csv must have at least columns: date, amount")

    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = df["amount"].astype(float)

    if "type" in df.columns:
        df["type"] = df["type"].fillna("").astype(str).str.upper()
        # Only keep explicit FLOW types (deposits/withdrawals)
        external = df[df["type"] == "FLOW"].copy()
        df = external[["date", "amount"]]
    elif "ticker" in df.columns and "shares" in df.columns:
        df["ticker"] = df["ticker"].fillna("").astype(str).str.upper()
        df["shares"] = df["shares"].fillna(0.0).astype(float)
        # External flows: CASH or zero-share rows
        external = df[(df["ticker"] == "CASH") | (df["shares"] == 0.0)].copy()
        df = external[["date", "amount"]]
    else:
        df = df[["date", "amount"]]

    df = df.sort_values("date").reset_index(drop=True)
    return df


# ------------------------------------------------------------
# Load RAW transactions for SECURITY-LEVEL Dietz (ticker flows)
# ------------------------------------------------------------

def load_transactions_raw(path: str = CASHFLOWS_FILE) -> pd.DataFrame:
    """
    For security-level Modified Dietz we want ALL ticker flows:
      - Buys (negative amounts)
      - Sells (positive amounts)
    CASH rows are stripped out here.
    """
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    required = {"date", "ticker", "shares", "amount"}
    if not required.issubset(df.columns):
        # If not present, we simply skip security MD
        return pd.DataFrame(columns=["date", "ticker", "shares", "amount"])

    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["shares"] = df["shares"].astype(float)
    df["amount"] = df["amount"].astype(float)

    # Drop external CASH flows: they are for portfolio TWR, not security-level
    if "type" in df.columns:
        df["type"] = df["type"].fillna("").astype(str).str.upper()
        # Keep only TRADES for MD (exclude FLOW and DIVIDEND to avoid double counting with Adj Close)
        df = df[df["type"] == "TRADE"].copy()
    else:
        df = df[df["ticker"] != "CASH"].copy()

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


# ------------------------------------------------------------
# Load DIVIDENDS for Reporting (Income)
# ------------------------------------------------------------

def load_dividends(path: str = CASHFLOWS_FILE) -> pd.DataFrame:
    """
    Load rows marked as 'DIVIDEND' to report as Income.
    """
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    # If no 'type' column, no dividends to load
    if "type" not in df.columns:
        return pd.DataFrame(columns=["date", "ticker", "shares", "amount"])

    df["type"] = df["type"].fillna("").astype(str).str.upper()
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = df["amount"].astype(float)
    df["ticker"] = df["ticker"].fillna("").astype(str).str.upper()

    divs = df[df["type"] == "DIVIDEND"].copy()
    divs = divs.sort_values("date").reset_index(drop=True)
    return divs


# ------------------------------------------------------------
# Download price history and extract adjusted closes robustly
# ------------------------------------------------------------

def fetch_price_history(tickers, years_back: int = PRICE_LOOKBACK_YEARS, use_adj_close: bool = False) -> pd.DataFrame:
    global _REPORTED_MISSING

    # Normalize tickers to a hashable, order-independent cache key
    # FIX: Deduplicate tickers list to safely check len() later
    # FIX 2: Ensure Default Benchmark (SPY) is ALWAYS fetched to support active risk metrics
    raw_set = set(str(t).upper() for t in tickers)
    if "SPY" not in raw_set:
        raw_set.add("SPY")
        
    unique_tickers = sorted(list(raw_set))
    key = (tuple(unique_tickers), int(years_back), use_adj_close)

    if key in _PRICE_CACHE:
        # Return a copy so callers can't mutate the cached DataFrame in-place
        cached = _PRICE_CACHE[key]
        res = cached.copy()
        res.attrs = cached.attrs
        # DEBUG: Confirm errors are attached on cache hit
        errors = res.attrs.get('errors', [])
        if errors:
            print(f"DEBUG: data_loader returning CACHED prices with {len(errors)} errors.")
        return res

    start_date = (datetime.today() - timedelta(days=365 * years_back)).strftime("%Y-%m-%d")

    # Retry logic to handle occasional network/data gaps
    raw = pd.DataFrame()
    for attempt in range(3):
        try:
            raw = yf.download(
                unique_tickers,
                start=start_date,
                progress=False,
                auto_adjust=False,
                group_by="column",
            )
            if not raw.empty:
                break
        except Exception:
            pass
        
    if raw.empty:
        # This raises error if ALL failed. 
        # If partial failed, we continue and check active_holdings logic below.
        raise RuntimeError("yfinance returned no data after 3 attempts. Check tickers or network.")

    # FIX 1: Strip timezones immediately (Yahoo sends UTC, your CSVs are naive)
    if isinstance(raw.index, pd.DatetimeIndex) and raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)

    # Handle both MultiIndex and flat columns cases
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)

        if use_adj_close:
            # Prioritize Adj Close
            if "Adj Close" in level0:
                prices = raw.xs("Adj Close", axis=1, level=0)
            elif "Close" in level0:
                # Fallback
                prices = raw.xs("Close", axis=1, level=0)
            else:
                first_field = level0[0]
                prices = raw.xs(first_field, axis=1, level=0)
        else:
            # Prioritize Close (Standard)
            if "Close" in level0:
                prices = raw.xs("Close", axis=1, level=0)
            elif "Adj Close" in level0:
                prices = raw.xs("Adj Close", axis=1, level=0)
            else:
                first_field = level0[0]
                prices = raw.xs(first_field, axis=1, level=0)
    else:
        cols = list(raw.columns)
        if use_adj_close:
            # Prioritize Adj Close
            if "Adj Close" in cols:
                prices = raw["Adj Close"]
            elif "Close" in cols:
                prices = raw["Close"]
            else:
                prices = raw
        else:
            # Prioritize Close (Standard)
            if "Close" in cols:
                prices = raw["Close"]
            elif "Adj Close" in cols:
                prices = raw["Adj Close"]
            else:
                prices = raw

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    # FIX 2: If we have a single ticker, force the column name to be the ticker.
    # Otherwise yfinance leaves it as "Adj Close" and your engine can't find the price.
    if len(unique_tickers) == 1:
        prices.columns = [unique_tickers[0]]
    else:
        # Normalize column names to uppercase tickers
        prices.columns = [str(c).upper() for c in prices.columns]

    # Ensure datetime index
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index)
        if prices.index.tz is not None:
             prices.index = prices.index.tz_localize(None)

    prices = prices.sort_index()

    # -------------------------------
    # WARN IF MISSING PRICES (Before ffill)
    # -------------------------------
    errors = []
    
    # Check 1: Data Freshness
    last_price_date = prices.index.max().date()
    today = datetime.today().date()
    if last_price_date < today:
        days_gap = (today - last_price_date).days
        # Ignore small gaps on weekends (e.g. checked on Sunday, last data Friday -> gap 2)
        is_weekend = today.weekday() >= 5
        if not (is_weekend and days_gap <= 2):
            msg = f"Data stale: Latest {last_price_date} ({days_gap}d ago)"
            print(f"[WARNING] {msg}")
            errors.append(msg)

    # Check 2: Missing Data (Comprehensive & Deduplicated)
    start_check = pd.Timestamp("2025-11-01")
    
    # Load holdings to identify liquidated positions
    holdings_df = load_holdings()
    liquidated_holdings = set(holdings_df[holdings_df["shares"].abs() <= 1e-6]["ticker"].str.upper().tolist())
    
    # Universe of Concern:
    # Check ALL requested tickers, EXCEPT those that are explicitly liquidated holdings.
    # This ensures we check:
    # 1. Active holdings
    # 2. Benchmarks (which are not in holdings at all)
    # But we ignore:
    # 3. Liquidated holdings (history)
    requested_set = set(unique_tickers)
    universe_of_concern = requested_set - liquidated_holdings
    
    # Identify tickers completely missing from download (requested but not returned)
    downloaded_cols = set(prices.columns)
    missing_entirely = universe_of_concern - downloaded_cols
    
    # Identify tickers with missing days (partial data)
    check_tickers = [t for t in universe_of_concern if t in downloaded_cols]
    missing_days_map = {} # date -> list of tickers
    
    if check_tickers:
        full_index = pd.bdate_range(start=start_check, end=prices.index.max())
        # Reindex to identify missing days (creates NaNs)
        check_prices = prices[check_tickers].reindex(full_index)
        
        for date in check_prices.index:
            row = check_prices.loc[date]
            missing_t = row[row.isna()].index.tolist()
            if missing_t:
                # Holiday Heuristic: If > 75% tickers missing, likely holiday
                if len(missing_t) > len(check_tickers) * 0.75:
                    continue
                missing_days_map[date] = missing_t

    # --- Deduplication & Reporting Logic ---
    
    # 1. Clear reported status for tickers that are now clean in this call
    #    A ticker is clean if: it IS in columns AND it is NOT in missing_days_map
    tickers_with_missing_days = set()
    for t_list in missing_days_map.values():
        tickers_with_missing_days.update(t_list)
        
    clean_tickers = set(check_tickers) - tickers_with_missing_days
    _REPORTED_MISSING -= clean_tickers
    
    # 2. Collect ALL problems for the UI (Unfiltered)
    all_problems_msg = []
    
    # A. Entirely Missing
    if missing_entirely:
        t_str = ", ".join(sorted(missing_entirely))
        all_problems_msg.append(f"Tickers with NO data: {t_str}")
        
    # B. Missing Days
    for date in sorted(missing_days_map.keys()):
        t_list = missing_days_map[date]
        # Show ALL tickers for the UI notification (no truncation)
        t_str = ", ".join(t_list)
        all_problems_msg.append(f"{date.strftime('%Y-%m-%d')}: {t_str}")
        
    # 3. Filter for Console Output (Only new problems)
    console_msg_lines = []
    newly_reported = set()
    
    # Check entirely missing
    new_missing_entirely = missing_entirely - _REPORTED_MISSING
    if new_missing_entirely:
         t_str = ", ".join(sorted(new_missing_entirely))
         console_msg_lines.append(f"Tickers with NO data: {t_str}")
         newly_reported.update(new_missing_entirely)
         
    # Check missing days
    # Identify dates that contain at least one NEW missing ticker
    dates_with_new_problems = []
    for date in sorted(missing_days_map.keys()):
        t_list = missing_days_map[date]
        if any(t not in _REPORTED_MISSING for t in t_list):
            dates_with_new_problems.append(date)
            newly_reported.update(t_list)
            
    for date in dates_with_new_problems:
        t_list = missing_days_map[date]
        t_str = ", ".join(t_list[:4])
        if len(t_list) > 4: t_str += f" (+{len(t_list)-4})"
        console_msg_lines.append(f"{date.strftime('%Y-%m-%d')}: {t_str}")

    # Print to console ONLY if we have new information
    if console_msg_lines:
        summary_msg = "Missing Price Data:\n  " + "\n  ".join(console_msg_lines)
        print(f"[WARNING] {summary_msg}")
        _REPORTED_MISSING.update(newly_reported)
        
    # Append FULL list to errors for UI display
    if all_problems_msg:
        full_summary = "Missing Price Data:\n  " + "\n  ".join(all_problems_msg)
        errors.append(full_summary)

    # Fill forward AFTER checking for gaps
    prices = prices.ffill()
    
    prices.attrs['errors'] = errors
    
    # DEBUG: Confirm errors are attached
    if errors:
        print(f"DEBUG: data_loader returning {len(errors)} errors attached to prices.")

    # Store in cache and return a copy
    _PRICE_CACHE[key] = prices
    
    # Explicitly ensure attrs are preserved in the copy
    res = prices.copy()
    res.attrs = prices.attrs
    return res
