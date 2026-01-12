import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from functools import lru_cache
from data_loader import load_transactions_raw, fetch_price_history
from config import TAX_RATE_ST, TAX_RATE_LT

# ============================================================
# HELPERS: NORMALIZATION & TERM
# ============================================================

def normalize_ticker(ticker):
    """
    Standardizes ticker strings to prevent mismatch errors.
    Logic: Strip whitespace, Upper case, Replace dots with dashes (e.g., BRK.B -> BRK-B).
    """
    if not isinstance(ticker, str):
        return str(ticker)
    return ticker.strip().upper().replace(".", "-")

def _classify_term(date_acquired, date_sold=None):
    """
    Determines if a lot is Short-Term or Long-Term.
    Threshold: > 365 days is Long-Term.
    """
    if date_sold is None:
        date_sold = datetime.now()
        
    held_duration = date_sold - date_acquired
    
    if held_duration.days > 365:
        return "Long-Term"
    else:
        return "Short-Term"

def _is_near_cliff(date_acquired):
    """
    Returns True if the lot will turn Long-Term in the next 30 days.
    Range: 335 < Days Held <= 365
    """
    days_held = (datetime.now() - date_acquired).days
    return 335 < days_held <= 365

def _days_to_long_term(date_acquired):
    """Returns days remaining until Long-Term status."""
    days_held = (datetime.now() - date_acquired).days
    remaining = 366 - days_held
    return max(0, remaining)

# ============================================================
# CORE ENGINE: MULTI-STRATEGY MATCHING
# ============================================================

@lru_cache(maxsize=32)
def build_tax_lots(strategy="FIFO", signal=None, as_of_date=None):
    """
    Reconstructs the tax lot history from raw transactions using specified strategy.
    Includes Wash Sale detection.
    
    Args:
        strategy (str): 'FIFO', 'LIFO', or 'HIFO'. Default 'FIFO'.
        signal (str): Optional data timestamp to invalidate cache when data changes.
        as_of_date (str/datetime): Optional cutoff date for the analysis.
        
    Returns:
        open_lots_df (pd.DataFrame): Currently held lots.
        realized_events_df (pd.DataFrame): Closed tax events.
    """
    raw_tx = load_transactions_raw()
    if raw_tx.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    # 1. Data Hygiene
    raw_tx["date"] = pd.to_datetime(raw_tx["date"])

    # NEW: Filter by as_of_date if provided
    if as_of_date:
        cutoff = pd.Timestamp(as_of_date)
        raw_tx = raw_tx[raw_tx["date"] <= cutoff]

    raw_tx["ticker"] = raw_tx["ticker"].apply(normalize_ticker)
    
    open_lots = []      # List of {ticker, date_acquired, shares, cost_basis, cost_per_share}
    realized_events = [] # List of {ticker, date_sold, shares, realized_pl, term, tax_impact, is_wash_sale}
    
    # Process by Ticker
    tickers = raw_tx["ticker"].unique()
    
    for ticker in tickers:
        # Filter and Sort Chronologically
        tx_history = raw_tx[raw_tx["ticker"] == ticker].sort_values("date")
        
        # Lot Queue for this ticker (List of dicts)
        # Each item: {date_acquired, shares, cost_per_share}
        lot_queue = []
        
        for _, row in tx_history.iterrows():
            date = row["date"]
            shares = float(row["shares"])
            amount = float(row["amount"]) # Negative for Buy, Positive for Sell
            
            if abs(shares) < 1e-6:
                continue
                
            if shares > 0:
                # ==========================
                # BUY EVENT (Acquisition)
                # ==========================
                cost_basis = -amount
                cost_per_share = cost_basis / shares if shares > 0 else 0
                
                lot_queue.append({
                    "date_acquired": date,
                    "shares": shares,
                    "cost_per_share": cost_per_share
                })
                
            else:
                # ==========================
                # SELL EVENT (Disposal)
                # ==========================
                shares_to_sell = abs(shares)
                total_proceeds = amount 
                proceeds_per_share = total_proceeds / shares_to_sell if shares_to_sell > 0 else 0
                
                # Apply Strategy Sorting before matching
                if strategy == "LIFO":
                    # Last-In, First-Out: Sort by Date Descending
                    lot_queue.sort(key=lambda x: x["date_acquired"], reverse=True)
                elif strategy == "HIFO":
                    # Highest-In, First-Out: Sort by Cost Descending, then Date Ascending (FIFO tie-break)
                    # Tuple sort: (-cost, date) -> min() of this sorts by max cost, then min date
                    # OR simply sort normally with key
                    lot_queue.sort(key=lambda x: (-x["cost_per_share"], x["date_acquired"]))
                else:
                    # FIFO (Default): Sort by Date Ascending
                    lot_queue.sort(key=lambda x: x["date_acquired"])
                
                while shares_to_sell > 1e-6 and lot_queue:
                    # Take from top of sorted queue
                    current_lot = lot_queue[0]
                    
                    available = current_lot["shares"]
                    sold_qty = min(available, shares_to_sell)
                    
                    # Update Lot
                    if available > shares_to_sell:
                         current_lot["shares"] -= sold_qty
                    else:
                         lot_queue.pop(0)
                    
                    # Record Event
                    cost_basis_sold = sold_qty * current_lot["cost_per_share"]
                    proceeds_sold = sold_qty * proceeds_per_share
                    gain_loss = proceeds_sold - cost_basis_sold
                    term = _classify_term(current_lot["date_acquired"], date)
                    
                    # Initial Tax Impact (Wash Sales processed later)
                    tax_rate = TAX_RATE_LT if term == "Long-Term" else TAX_RATE_ST
                    tax_impact = gain_loss * tax_rate if gain_loss > 0 else 0
                    
                    realized_events.append({
                        "Date Sold": date,
                        "Ticker": ticker,
                        "Shares": sold_qty,
                        "Realized P/L": gain_loss,
                        "Term": term,
                        "Tax Impact": tax_impact,
                        "Is Wash Sale": False 
                    })
                    
                    shares_to_sell -= sold_qty

        # Collect Open Lots
        for lot in lot_queue:
            open_lots.append({
                "Ticker": ticker,
                "Date Acquired": lot["date_acquired"],
                "Shares": lot["shares"],
                "Cost Basis": lot["shares"] * lot["cost_per_share"],
                "Cost Per Share": lot["cost_per_share"]
            })

    # Convert to DataFrames
    open_lots_df = pd.DataFrame(open_lots)
    realized_df = pd.DataFrame(realized_events)
    
    # ==========================
    # WASH SALE GUARD (Robust)
    # ==========================
    # Rule: If Loss, check for Buy within +/- 30 days.
    # If found, disallow loss (Zero Tax Impact) AND add loss to basis of replacement lot.
    
    if not realized_df.empty:
        # Initialize default columns if not present
        if "Is Wash Sale" not in realized_df.columns:
            realized_df["Is Wash Sale"] = False
        
        # Iterate over LOSSES only
        loss_indices = realized_df[realized_df["Realized P/L"] < -0.01].index
        
        for idx in loss_indices:
            loss_row = realized_df.loc[idx]
            ticker = loss_row["Ticker"]
            sold_date = loss_row["Date Sold"]
            shares_sold = loss_row["Shares"]
            loss_amount = -loss_row["Realized P/L"] # Positive magnitude of loss
            
            # Define Window: [Sold - 30, Sold + 30]
            start_window = sold_date - pd.Timedelta(days=30)
            end_window = sold_date + pd.Timedelta(days=30)
            
            # Find candidate replacement lots in OPEN LOTS
            # (Matches Ticker + Acquired in Window)
            if not open_lots_df.empty:
                candidates_mask = (
                    (open_lots_df["Ticker"] == ticker) & 
                    (open_lots_df["Date Acquired"] >= start_window) & 
                    (open_lots_df["Date Acquired"] <= end_window)
                )
                
                if candidates_mask.any():
                    # FOUND REPLACEMENT
                    
                    # 1. Mark as Wash Sale (Disallow Loss)
                    realized_df.at[idx, "Is Wash Sale"] = True
                    realized_df.at[idx, "Tax Impact"] = 0.0 # Loss cannot be claimed
                    
                    # 2. Adjust Basis of Replacement Lot
                    # Applying entire loss to the first matching replacement lot found
                    # (Simplified "All-or-Nothing" approach sufficient for this analytics scope)
                    repl_idx = open_lots_df[candidates_mask].index[0]
                    
                    current_basis = open_lots_df.at[repl_idx, "Cost Basis"]
                    new_basis = current_basis + loss_amount
                    shares_repl = open_lots_df.at[repl_idx, "Shares"]
                    
                    # Update Basis and derived Cost Per Share
                    open_lots_df.at[repl_idx, "Cost Basis"] = new_basis
                    if shares_repl > 0:
                        open_lots_df.at[repl_idx, "Cost Per Share"] = new_basis / shares_repl

    # Enrich Open Lots with Market Data
    if not open_lots_df.empty:
        # Fetch current prices
        unique_tickers = open_lots_df["Ticker"].unique().tolist()
        try:
            prices = fetch_price_history(unique_tickers)
            if as_of_date:
                cutoff = pd.Timestamp(as_of_date)
                # Find last price on or before cutoff
                prices_hist = prices[prices.index <= cutoff]
                latest_prices = prices_hist.iloc[-1] if not prices_hist.empty else pd.Series(dtype=float)
            else:
                latest_prices = prices.iloc[-1]
        except Exception:
            latest_prices = pd.Series(dtype=float)
            
        def get_market_metrics(row):
            ticker = row["Ticker"]
            shares = row["Shares"]
            cost = row["Cost Basis"]
            date_acq = row["Date Acquired"]
            
            curr_price = 0.0
            if ticker in latest_prices:
                curr_price = float(latest_prices[ticker])
            
            mkt_val = shares * curr_price
            unrealized_pl = mkt_val - cost
            
            term = _classify_term(date_acq)
            near_cliff = _is_near_cliff(date_acq)
            days_left = _days_to_long_term(date_acq) if term == "Short-Term" else 0
            
            return pd.Series([curr_price, mkt_val, unrealized_pl, term, near_cliff, days_left])
            
        metrics = open_lots_df.apply(get_market_metrics, axis=1)
        metrics.columns = ["Current Price", "Market Value", "Unrealized P/L", "Term", "Is Near Cliff", "Days to LT"]
        
        open_lots_df = pd.concat([open_lots_df, metrics], axis=1)
        
        def calc_liability(row):
            gain = row["Unrealized P/L"]
            if gain <= 0: return 0.0
            rate = TAX_RATE_LT if row["Term"] == "Long-Term" else TAX_RATE_ST
            return gain * rate
            
        open_lots_df["Est Tax Liability"] = open_lots_df.apply(calc_liability, axis=1)
        
    return open_lots_df, realized_df

# ============================================================
# SIMULATOR
# ============================================================

def simulate_sell(ticker, shares_to_sell, strategy="FIFO"):
    """
    Simulates selling shares using specified strategy.
    Alerts on Wash Sales (Buying in last 30 days).
    """
    ticker = normalize_ticker(ticker)
    
    if shares_to_sell <= 0:
        return {"summary_text": "Invalid share quantity.", "total_gain": 0, "est_tax": 0, "breakdown": []}

    open_lots_df, _ = build_tax_lots(strategy=strategy) # Rebuild to get fresh state using strategy
    
    if open_lots_df.empty or ticker not in open_lots_df["Ticker"].values:
        return {"summary_text": f"No open lots found for {ticker}.", "total_gain": 0, "est_tax": 0, "breakdown": []}
        
    # Get lots for ticker
    lots_df = open_lots_df[open_lots_df["Ticker"] == ticker].copy()
    
    # Sort lots based on Strategy for the *next* sell simulation
    # Note: open_lots_df is already the result of past strategy application. 
    # We now simulate the next step.
    if strategy == "LIFO":
        lots_df = lots_df.sort_values("Date Acquired", ascending=False)
    elif strategy == "HIFO":
        lots_df = lots_df.sort_values(["Cost Per Share", "Date Acquired"], ascending=[False, True])
    else:
        # FIFO
        lots_df = lots_df.sort_values("Date Acquired", ascending=True)
    
    total_avail = lots_df["Shares"].sum()
    if shares_to_sell > total_avail:
        return {
            "summary_text": f"Cannot sell {shares_to_sell} shares. Only {total_avail:.2f} available.",
            "total_gain": 0, "est_tax": 0, "breakdown": []
        }
    
    # Check for Wash Sale Risk (Recent Buy)
    # Check raw tx for buys in last 30 days
    raw_tx = load_transactions_raw()
    if not raw_tx.empty:
        raw_tx["ticker"] = raw_tx["ticker"].apply(normalize_ticker)
        recent_buys = raw_tx[
            (raw_tx["ticker"] == ticker) & 
            (raw_tx["shares"] > 0) & 
            (raw_tx["date"] >= datetime.now() - timedelta(days=30))
        ]
        wash_sale_risk = not recent_buys.empty
    else:
        wash_sale_risk = False

    # Simulation Loop
    remaining_to_sell = shares_to_sell
    impact_records = []
    
    total_st_gain = 0.0
    total_lt_gain = 0.0
    
    curr_price = lots_df.iloc[0]["Current Price"]
    summary_parts = []
    
    for _, lot in lots_df.iterrows():
        if remaining_to_sell <= 1e-6:
            break
            
        available = lot["Shares"]
        date_str = lot["Date Acquired"].strftime("%b %Y")
        term = lot["Term"]
        
        sold = min(available, remaining_to_sell)
        cost = sold * lot["Cost Per Share"]
        proceeds = sold * curr_price
        gain = proceeds - cost
        
        impact_records.append({
            "Date": date_str,
            "Shares": sold,
            "Term": term,
            "Gain": gain
        })
        
        if sold == available:
            summary_parts.append(f"deplete your oldest lot from {date_str} ({term})")
        else:
            summary_parts.append(f"partially tap into your {date_str} lot ({term})")

        if term == "Short-Term": total_st_gain += gain
        else: total_lt_gain += gain
        
        remaining_to_sell -= sold
            
    # Calculate Tax
    tax_st = max(0, total_st_gain) * TAX_RATE_ST
    tax_lt = max(0, total_lt_gain) * TAX_RATE_LT
    total_tax = tax_st + tax_lt
    
    # Narrative
    action_text = f"Selling {shares_to_sell} shares will " + " and ".join(summary_parts) + "."
    result_text = f"Result: ${total_lt_gain:,.2f} LT Gain, ${total_st_gain:,.2f} ST Gain."
    tax_text = f"Est Tax: ${total_tax:,.2f}."
    
    full_text = f"{action_text}\n\n{result_text} {tax_text}"
    
    if wash_sale_risk:
        full_text = "⚠️ STOP: Recent buy detected (last 30 days). Selling now triggers a Wash Sale on any losses.\n\n" + full_text
    
    return {
        "summary_text": full_text,
        "total_gain": total_st_gain + total_lt_gain,
        "est_tax": total_tax,
        "breakdown": impact_records
    }

# ============================================================
# OPTIMIZATION: TAX-AWARE SALES
# ============================================================

def calculate_tax_optimized_sales(candidates, avoid_st_gains=True):
    """
    Calculates recommended sales for multiple tickers to meet target sell amounts,
    prioritizing tax efficiency (Losses First, Gains Last).
    
    Priority:
    1. Long-Term Losses (Harvest)
    2. Short-Term Losses
    3. Long-Term Gains
    4. Short-Term Gains (Optional: Avoid)
    
    Args:
        candidates (dict): {ticker: amount_to_sell_usd}
        avoid_st_gains (bool): If True, will stop selling if only ST Gains remain.
        
    Returns:
        dict: {
            "sales_df": pd.DataFrame, # Columns: [Ticker, Shares, Proceeds, Realized_PL, Tax_Impact, Term]
            "total_proceeds": float,
            "total_realized_pl": float,
            "est_tax_liability": float
        }
    """
    if not candidates:
        return {
            "sales_df": pd.DataFrame(), 
            "total_proceeds": 0.0, 
            "total_realized_pl": 0.0, 
            "est_tax_liability": 0.0
        }
        
    # 1. Get All Lots
    open_lots_df, _ = build_tax_lots(strategy="FIFO")
    if open_lots_df.empty:
         return {
            "sales_df": pd.DataFrame(), 
            "total_proceeds": 0.0, 
            "total_realized_pl": 0.0, 
            "est_tax_liability": 0.0
        }
        
    # 2. Check Wash Sale Risks (Recent Buys in last 30 days)
    raw_tx = load_transactions_raw()
    wash_risks = set()
    if not raw_tx.empty:
        raw_tx["ticker"] = raw_tx["ticker"].apply(normalize_ticker)
        cutoff = datetime.now() - timedelta(days=30)
        # Any buy in last 30 days creates risk
        recent_buys = raw_tx[(raw_tx["shares"] > 0) & (raw_tx["date"] >= cutoff)]
        wash_risks = set(recent_buys["ticker"].unique())
        
    recommended_sales = []
    
    for ticker, target_usd in candidates.items():
        ticker = normalize_ticker(ticker)
        if target_usd <= 0: continue
        
        # Get lots for this ticker
        t_lots = open_lots_df[open_lots_df["Ticker"] == ticker].copy()
        if t_lots.empty: continue
        
        # Current Price needed to convert USD to Shares
        # t_lots has "Current Price" from build_tax_lots
        curr_price = t_lots.iloc[0]["Current Price"]
        if curr_price <= 0: continue
        
        # We need to sell $X worth. 
        # But we select by LOTS. 
        
        # Classify Buckets
        # Bucket 1: LT Loss
        # Bucket 2: ST Loss
        # Bucket 3: LT Gain
        # Bucket 4: ST Gain
        
        def assign_bucket(row):
            # WASH SALE FILTER: If Loss AND Ticker in Risk Set -> Exclude (Bucket 99)
            if row["Unrealized P/L"] < 0 and ticker in wash_risks:
                return 99 # Do not sell
                
            if row["Unrealized P/L"] < 0:
                return 1 if row["Term"] == "Long-Term" else 2
            else:
                return 3 if row["Term"] == "Long-Term" else 4
        
        t_lots["Bucket"] = t_lots.apply(assign_bucket, axis=1)
        
        # Sort: Bucket asc, then maybe magnitude of loss desc?
        # Requirement says "Sell first to harvest losses". 
        # Within buckets, HIFO (Highest Cost) is generally best for losses (Maximize Loss) and best for Gains (Minimize Gain).
        # HIFO = Sort by Cost Basis Per Share DESC.
        t_lots = t_lots.sort_values(by=["Bucket", "Cost Per Share"], ascending=[True, False])
        
        remaining_target_usd = target_usd
        
        for _, lot in t_lots.iterrows():
            if remaining_target_usd < 0.01:
                break
                
            bucket = lot["Bucket"]
            if bucket == 99: continue # Wash Sale Risk
            
            if bucket == 4 and avoid_st_gains:
                # We hit ST Gains and want to avoid them. Stop selling this ticker.
                break
                
            # How much can we sell from this lot?
            lot_value = lot["Market Value"]
            lot_shares = lot["Shares"]
            
            sell_value = min(lot_value, remaining_target_usd)
            sell_shares = sell_value / curr_price
            
            # Record Sale
            cost_basis_sold = sell_shares * lot["Cost Per Share"]
            realized_pl = sell_value - cost_basis_sold
            
            # Tax
            if realized_pl > 0:
                rate = TAX_RATE_LT if bucket == 3 else TAX_RATE_ST # Bucket 3 is LT Gain
                tax_impact = realized_pl * rate
            else:
                tax_impact = 0.0 # Loss doesn't pay tax
                
            recommended_sales.append({
                "Ticker": ticker,
                "Shares": sell_shares,
                "Proceeds": sell_value,
                "Realized P/L": realized_pl,
                "Tax Impact": tax_impact,
                "Term": lot["Term"],
                "Bucket": bucket
            })
            
            remaining_target_usd -= sell_value
            
    # Compile Results
    if not recommended_sales:
         return {
            "sales_df": pd.DataFrame(), 
            "total_proceeds": 0.0, 
            "total_realized_pl": 0.0, 
            "est_tax_liability": 0.0
        }
        
    sales_df = pd.DataFrame(recommended_sales)
    
    total_proceeds = sales_df["Proceeds"].sum()
    total_realized_pl = sales_df["Realized P/L"].sum()
    est_tax_liability = sales_df["Tax Impact"].sum()
    
    return {
        "sales_df": sales_df,
        "total_proceeds": total_proceeds,
        "total_realized_pl": total_realized_pl,
        "est_tax_liability": est_tax_liability
    }
