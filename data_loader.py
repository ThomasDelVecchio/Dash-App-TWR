import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import requests
import json
import os
import pickle
import time
from config import FMP_API_KEY, FMP_PRICE_ENABLED, FMP_PRICE_LOOKBACK_YEARS

# ============================================================
# CONFIG
# ============================================================
HOLDINGS_FILE = "sample holdings.csv"
CASHFLOWS_FILE = "cashflows.csv"
COMPOSITE_MAPPING_FILE = "composite_mapping.csv"
PRICE_LOOKBACK_YEARS = 20
METADATA_CACHE_FILE = "metadata_cache.json"
PRICE_CACHE_FILE = "price_cache.pkl"
PRICE_CACHE_EXPIRY_HOURS = 12

# Simple in-memory cache for price history to keep horizons consistent
_PRICE_CACHE = {}
_METADATA_CACHE = {}

# Global set to track tickers we've already warned about to avoid console spam
_REPORTED_MISSING = set()

# ------------------------------------------------------------
# Price Cache Management (Persistent)
# ------------------------------------------------------------

def load_price_cache_from_disk():
    global _PRICE_CACHE
    if os.path.exists(PRICE_CACHE_FILE):
        try:
            # Check expiry
            modified_time = datetime.fromtimestamp(os.path.getmtime(PRICE_CACHE_FILE))
            age = datetime.now() - modified_time
            if age > timedelta(hours=PRICE_CACHE_EXPIRY_HOURS):
                 print(f"[CACHE] Price cache expired ({age}). Refreshing from API...")
                 _PRICE_CACHE = {}
                 return

            with open(PRICE_CACHE_FILE, "rb") as f:
                _PRICE_CACHE = pickle.load(f)
            print(f"[CACHE] Loaded {len(_PRICE_CACHE)} price entries from disk.")
        except Exception as e:
            print(f"[CACHE] Error loading price cache: {e}")
            _PRICE_CACHE = {}

def save_price_cache_to_disk():
    try:
        with open(PRICE_CACHE_FILE, "wb") as f:
            pickle.dump(_PRICE_CACHE, f)
        # print("[CACHE] Price cache saved to disk.")
    except Exception as e:
        print(f"[CACHE] Error saving price cache: {e}")

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
# load_price_cache_from_disk()

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
        # Compatibility check: if old format (direct dict), return it
        cached = _METADATA_CACHE[ticker]
        if isinstance(cached, dict) and "weights" in cached:
            cached_weights = cached["weights"]
            cached_source = cached.get("source", "Unknown")
        else:
            cached_weights = cached
            cached_source = "Unknown"

        # If FMP is enabled and cached source is not FMP, refresh from FMP
        if FMP_PRICE_ENABLED and FMP_API_KEY and FMP_API_KEY != "demo" and cached_source != "FMP":
            fmp_weights = fetch_fmp_sector_weights(ticker)
            if fmp_weights:
                _METADATA_CACHE[ticker] = {
                    "weights": fmp_weights,
                    "source": "FMP",
                    "timestamp": datetime.now().isoformat()
                }
                save_metadata_cache()
                return fmp_weights

        return cached_weights

    weights = {}
    source = "Unknown"
    
    # 1. Try FMP API
    weights = fetch_fmp_sector_weights(ticker)
    if weights:
        source = "FMP"
    
    # 2. Try yfinance
    if not weights:
        weights = fetch_yf_sector_weights(ticker)
        if weights:
            source = "YF"
        
    # 3. Equity Fallback
    if not weights:
        sector = get_equity_sector(ticker)
        if sector:
            weights = {sector: 100.0}
            source = "Equity Fallback"
            
    # Cache result with metadata
    _METADATA_CACHE[ticker] = {
        "weights": weights,
        "source": source,
        "timestamp": datetime.now().isoformat()
    }
    save_metadata_cache()
    
    return weights

def fetch_fmp_sector_weights(ticker: str) -> dict:
    """Fetch ETF sector weights from Financial Modeling Prep."""
    if not FMP_API_KEY or FMP_API_KEY == "demo":
        return {}
        
    try:
        # UPDATED: Use /stable/ endpoint (Jan 2026)
        # Old: /api/v3/etf-sector-weightings/{ticker}
        url = f"https://financialmodelingprep.com/stable/etf/sector-weightings?symbol={ticker}&apikey={FMP_API_KEY}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            # FMP returns list of dicts: [{'sector': 'Technology', 'weightPercentage': '25.5%'}, ...]
            weights = {}
            for item in data:
                sector = item.get("sector", "")
                pct_val = item.get("weightPercentage", 0)
                try:
                    if isinstance(pct_val, str):
                        pct = float(pct_val.replace("%", ""))
                    else:
                        pct = float(pct_val)
                        
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
# FMP Price History Helpers (Hybrid Mode)
# ------------------------------------------------------------

def fetch_fmp_price_history_single(ticker: str, start_date: str, end_date: str, use_adj_close: bool = False) -> pd.DataFrame:
    """
    Fetch daily price history for a single ticker from FMP API.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        use_adj_close: If True, uses the dividend-adjusted endpoint
    
    Returns:
        DataFrame with DatetimeIndex and 'Close' (or 'Adj Close') column
    """
    if not FMP_API_KEY or FMP_API_KEY == "demo":
        return pd.DataFrame()
    
    try:
        # UPDATED: Use /stable/ endpoint (Jan 2026) for non-legacy users
        # Select endpoint based on adjusted close requirement
        if use_adj_close:
            # Dividend Adjusted Endpoint: returns 'adjClose' which is split & dividend adjusted
            endpoint = "historical-price-eod/dividend-adjusted"
        else:
            # Standard Endpoint: returns 'close' which is split-adjusted only
            endpoint = "historical-price-eod/full"
            
        url = f"https://financialmodelingprep.com/stable/{endpoint}?symbol={ticker}&from={start_date}&to={end_date}&apikey={FMP_API_KEY}"
        resp = requests.get(url, timeout=15)
        
        if resp.status_code != 200:
            print(f"[FMP] HTTP {resp.status_code} for {ticker}")
            return pd.DataFrame()
        
        data = resp.json()
        
        # Handle dict response (v3) vs list response (stable)
        if isinstance(data, dict) and "historical" in data:
            historical = data["historical"]
        elif isinstance(data, list):
            historical = data
        else:
            historical = []

        if not historical:
            return pd.DataFrame()
        
        df = pd.DataFrame(historical)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        
        if use_adj_close:
            # The dividend-adjusted endpoint usually returns 'adjClose'
            # We map this to 'Adj Close' for our internal logic
            if "adjClose" in df.columns:
                df = df.rename(columns={"adjClose": "Adj Close"})
            elif "close" in df.columns:
                # Fallback if adjClose not present (rare)
                df = df.rename(columns={"close": "Adj Close"})
            
            return df[["Adj Close"]]
        else:
            # Standard endpoint returns 'close'
            df = df.rename(columns={"close": "Close"})
            return df[["Close"]]
        
    except Exception as e:
        print(f"[FMP] Error fetching {ticker}: {e}")
        return pd.DataFrame()


def _stitch_price_dataframes(fmp_df: pd.DataFrame, yf_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Stitch FMP (recent) and yfinance (historical) price data for a single ticker.
    
    FMP data takes priority for overlapping dates (more reliable for recent data).
    
    Args:
        fmp_df: DataFrame from FMP (recent years)
        yf_df: DataFrame from yfinance (full history)
        ticker: Ticker symbol for column naming
    
    Returns:
        Combined DataFrame with continuous price history
    """
    # Handle edge cases
    if fmp_df.empty and yf_df.empty:
        return pd.DataFrame()
    if fmp_df.empty:
        return yf_df
    if yf_df.empty:
        return fmp_df
    
    # Find the boundary date (earliest FMP date)
    fmp_start = fmp_df.index.min()
    
    # Take yfinance data BEFORE FMP starts (exclusive to avoid overlap issues)
    yf_historical = yf_df[yf_df.index < fmp_start]
    
    # Concatenate: yfinance historical + FMP recent
    combined = pd.concat([yf_historical, fmp_df])
    combined = combined.sort_index()
    
    # Remove any duplicate indices (prefer later entry = FMP)
    combined = combined[~combined.index.duplicated(keep='last')]
    
    return combined


# ------------------------------------------------------------
# Load holdings (your schema)
# ------------------------------------------------------------

HOLDINGS_EXTERNAL_FILE = "holdings_external.csv"

def load_holdings(path: str = HOLDINGS_FILE) -> pd.DataFrame:
    """
    Load holdings from primary file and merge external holdings if present.
    
    External holdings (holdings_external.csv) contain positions not tracked
    by the primary broker API (e.g., stock plans, other brokers). This merge
    ensures GIPS-compliant reconciliation with the full transaction history
    in cashflows.csv.
    """
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

    # ============================================================
    # MERGE EXTERNAL HOLDINGS (Fallback if not merged during sync)
    # ============================================================
    # This ensures the app works correctly even if E*TRADE sync
    # hasn't run or failed. External holdings are merged if they
    # don't already exist in the primary holdings file.
    
    if os.path.exists(HOLDINGS_EXTERNAL_FILE):
        ext_df = pd.read_csv(HOLDINGS_EXTERNAL_FILE)
        ext_df.columns = [c.lower() for c in ext_df.columns]
        
        if "ticker" in ext_df.columns and "shares" in ext_df.columns:
            ext_df["ticker"] = ext_df["ticker"].astype(str).str.upper()
            ext_df["shares"] = ext_df["shares"].astype(float)
            
            if "asset_class" not in ext_df.columns:
                ext_df["asset_class"] = "Unknown"
            if "target_pct" not in ext_df.columns:
                ext_df["target_pct"] = np.nan
            
            # Only add external positions not already in primary holdings
            existing_tickers = set(df["ticker"].unique())
            ext_new = ext_df[~ext_df["ticker"].isin(existing_tickers)]
            
            if not ext_new.empty:
                df = pd.concat([df, ext_new], ignore_index=True)

    return df

# ------------------------------------------------------------
# Load Composite Mappings
# ------------------------------------------------------------

def load_composite_mappings(path: str = COMPOSITE_MAPPING_FILE) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["composite_name", "ticker"])
        
    df = pd.read_csv(path)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    
    # Expected columns: composite_name, ticker
    required = {"composite_name", "ticker"}
    if not required.issubset(df.columns):
        # Fallback if names are slightly different but order is correct? 
        # Or just return empty to avoid crashing
        return pd.DataFrame(columns=["composite_name", "ticker"])
        
    df["composite_name"] = df["composite_name"].astype(str)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    
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

    divs = df[df["type"].isin(["DIVIDEND", "INTEREST"])].copy()
    divs = divs.sort_values("date").reset_index(drop=True)
    return divs


# ------------------------------------------------------------
# Download price history and extract adjusted closes robustly
# ------------------------------------------------------------

def fetch_price_history(tickers, years_back: int = PRICE_LOOKBACK_YEARS, use_adj_close: bool = False) -> pd.DataFrame:
    """
    Fetch price history using hybrid FMP/yfinance mode or yfinance-only.
    
    HYBRID MODE (FMP_PRICE_ENABLED=True):
        - FMP: Last FMP_PRICE_LOOKBACK_YEARS years (default 5)
        - yfinance: Years 5-10 (older historical data)
        - Stitched together for complete 10-year history
    
    YFINANCE-ONLY MODE (FMP_PRICE_ENABLED=False):
        - yfinance: Full 10-year history (free, unlimited)
    
    This function is the SINGLE SOURCE OF TRUTH for all price data in the app.
    All pages, simulators, and analytics use this centralized function.
    """
    global _REPORTED_MISSING

    # Normalize tickers to a hashable, order-independent cache key
    # FIX: Deduplicate tickers list to safely check len() later
    # FIX 2: Ensure Default Benchmark (SPY) is ALWAYS fetched to support active risk metrics
    raw_set = set(str(t).upper() for t in tickers)
    if "SPY" not in raw_set:
        raw_set.add("SPY")
        
    unique_tickers = sorted(list(raw_set))
    
    # Include FMP flag in cache key to ensure correct invalidation when toggling modes
    key = (tuple(unique_tickers), int(years_back), use_adj_close, FMP_PRICE_ENABLED)

    # SMART CACHING:
    # If we have this exact key in memory, and it was fetched "recently" (e.g. within this session/engine run),
    # use it to prevent the Engine and Wrappers from double-hitting the API for the same tickers in the same second.
    # However, if 'force_refresh' is passed (implied if not in cache or if user explicitly asks - strict logic below), we fetch.
    
    # Simple logic: If in memory cache, assume it's fresh enough for this specific function call.
    # The external trigger (app.py or polling) determines when to clear/reload the cache.
    if key in _PRICE_CACHE:
        # Return a copy so callers can't mutate the cached DataFrame in-place
        cached = _PRICE_CACHE[key]
        res = cached.copy()
        res.attrs = dict(cached.attrs)
        res.attrs["cache_source"] = "memory"
        return res

    # Calculate date boundaries
    today = datetime.today()
    full_start_date = (today - timedelta(days=365 * years_back)).strftime("%Y-%m-%d")
    # FIX: Explicitly set end date to TOMORROW to ensure today's data is included
    # yfinance end date is exclusive, so we add 1 day to capture today
    end_date = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    end_date_display = today.strftime("%Y-%m-%d")
    
    # ================================================================
    # HYBRID MODE: FMP for recent years + yfinance for older history
    # ================================================================
    if FMP_PRICE_ENABLED and FMP_API_KEY and FMP_API_KEY != "demo":
        print(f"[PRICE] Hybrid mode: FMP ({FMP_PRICE_LOOKBACK_YEARS}yr) + yfinance ({years_back - FMP_PRICE_LOOKBACK_YEARS}yr)")
        
        # FMP covers last N years
        fmp_start = (today - timedelta(days=365 * FMP_PRICE_LOOKBACK_YEARS)).strftime("%Y-%m-%d")
        
        # yfinance covers the older portion (full history, we'll filter later)
        yf_start = full_start_date
        
        all_prices = {}
        fmp_success = 0
        fmp_fallback = 0
        
        # Track which tickers came from which source for metadata
        fmp_ticker_list = []
        yf_ticker_list = []
        
        for ticker in unique_tickers:
            # 1. Fetch FMP (recent years)
            fmp_df = fetch_fmp_price_history_single(ticker, fmp_start, end_date_display, use_adj_close=use_adj_close)
            
            # 2. Fetch yfinance (full history as fallback/older data)
            try:
                yf_raw = yf.download(
                    ticker,
                    start=yf_start,
                    end=end_date,
                    progress=False,
                    auto_adjust=False
                )
                
                if not yf_raw.empty:
                    # Strip timezone if present
                    if isinstance(yf_raw.index, pd.DatetimeIndex) and yf_raw.index.tz is not None:
                        yf_raw.index = yf_raw.index.tz_localize(None)
                    
                    # Handle MultiIndex columns (yfinance returns ('Close', 'TICKER') for single ticker)
                    if isinstance(yf_raw.columns, pd.MultiIndex):
                        # Flatten to just the price field names
                        yf_raw.columns = yf_raw.columns.get_level_values(0)
                    
                    # Select price column
                    col_name = "Adj Close" if use_adj_close else "Close"
                    if col_name in yf_raw.columns:
                        yf_series = yf_raw[col_name]
                        yf_df = pd.DataFrame({"Close": yf_series})
                    elif "Close" in yf_raw.columns:
                        yf_series = yf_raw["Close"]
                        yf_df = pd.DataFrame({"Close": yf_series})
                    else:
                        yf_df = pd.DataFrame()
                else:
                    yf_df = pd.DataFrame()
            except Exception as e:
                print(f"[YF] Error fetching {ticker}: {e}")
                yf_df = pd.DataFrame()
            
            # 3. Stitch together
            if not fmp_df.empty:
                # Select correct price column from FMP
                fmp_col = "Adj Close" if use_adj_close and "Adj Close" in fmp_df.columns else "Close"
                if fmp_col in fmp_df.columns:
                    fmp_prices = fmp_df[[fmp_col]].rename(columns={fmp_col: "Close"})
                else:
                    fmp_prices = pd.DataFrame()
                
                combined = _stitch_price_dataframes(fmp_prices, yf_df, ticker)
                fmp_success += 1
                fmp_ticker_list.append(ticker)
            else:
                # FMP failed, use yfinance only for this ticker
                combined = yf_df
                if not yf_df.empty:
                    fmp_fallback += 1
                    yf_ticker_list.append(ticker)
            
            if not combined.empty:
                all_prices[ticker] = combined["Close"]
        
        print(f"[PRICE] FMP success: {fmp_success}/{len(unique_tickers)}, yfinance fallback: {fmp_fallback}")
        
        # Combine all tickers into single DataFrame
        if all_prices:
            prices = pd.DataFrame(all_prices)
        elif key in _PRICE_CACHE:
            print(f"[CACHE] Hybrid fetch failed. Using cached data (last successful fetch).")
            cached = _PRICE_CACHE[key]
            res = cached.copy()
            res.attrs = dict(cached.attrs)
            res.attrs["cache_source"] = "memory-fallback"
            return res
        else:
            raise RuntimeError("Hybrid fetch returned no data. Check API keys and network.")
        
        # Store source metadata for UI display
        prices.attrs['source'] = 'hybrid'
        prices.attrs['fmp_tickers'] = fmp_success
        prices.attrs['yf_fallback'] = fmp_fallback
        
        # Store detailed source_metadata dictionary for badge display
        prices.attrs['source_metadata'] = {
            "FMP": fmp_ticker_list,
            "yfinance": yf_ticker_list,
            "mixed": [],  # Hybrid mode: tickers are either FMP or YF, not mixed
            "fmp_range": (pd.Timestamp(fmp_start), pd.Timestamp(end_date_display)),
            "yf_range": (pd.Timestamp(full_start_date), pd.Timestamp(fmp_start)),
        }
    
    # ================================================================
    # YFINANCE-ONLY MODE: Full history from yfinance
    # ================================================================
    else:
        if not FMP_PRICE_ENABLED:
            print(f"[PRICE] yfinance-only mode ({years_back}yr history)")
        else:
            print(f"[PRICE] yfinance-only mode (FMP API key missing or demo)")
        
        start_date = full_start_date

        # Retry logic to handle occasional network/data gaps
        raw = pd.DataFrame()
        for attempt in range(3):
            try:
                raw = yf.download(
                    unique_tickers,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=False,
                    group_by="column",
                )
                if not raw.empty:
                    break
            except Exception as e:
                wait = 2 ** attempt
                print(f"[YF] Attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
        
        if raw.empty:
            if key in _PRICE_CACHE:
                print(f"[CACHE] yfinance fetch failed. Using cached data (last successful fetch).")
                cached = _PRICE_CACHE[key]
                res = cached.copy()
                res.attrs = dict(cached.attrs)
                res.attrs["cache_source"] = "memory-fallback"
                return res
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
        
        # Store source metadata for UI display
        prices.attrs['source'] = 'yfinance'
        
        # Store detailed source_metadata dictionary for badge display
        prices.attrs['source_metadata'] = {
            "FMP": [],
            "yfinance": list(prices.columns),
            "mixed": [],
            "fmp_range": (None, None),
            "yf_range": (pd.Timestamp(start_date), prices.index.max() if not prices.empty else None),
        }

    # ================================================================
    # COMMON POST-PROCESSING (Both Modes)
    # ================================================================
    
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

    # Attach fetch timestamp for diagnostics
    prices.attrs['fetched_at'] = datetime.now().isoformat()
    prices.attrs['cache_source'] = "live"

    # Store in cache and return a copy
    _PRICE_CACHE[key] = prices
    save_price_cache_to_disk()
    
    # Explicitly ensure attrs are preserved in the copy
    res = prices.copy()
    res.attrs = prices.attrs
    return res
