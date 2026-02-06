import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from data_loader import CASHFLOWS_FILE

# Holidays for NYSE Calendar Logic
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, Holiday, nearest_workday, 
    USMartinLutherKingJr, USPresidentsDay, USMemorialDay, 
    USLaborDay, USThanksgivingDay, GoodFriday
)
from pandas.tseries.offsets import DateOffset

# ============================================================
# CONFIG / CONSTANTS
# ============================================================
HORIZONS = ["1D", "1W", "MTD", "1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y"]

# Multi-year horizons to annualize when presenting results
ANNUALIZE_HORIZONS = {
    "3Y": 3,
    "5Y": 5,
}

# ------------------------------------------------------------
# NYSE Holiday Calendar (GIPS Compliance)
# ------------------------------------------------------------

class NYSECalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('NewYearsDay', month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        # Juneteenth (Observed since 2021, but good to have)
        Holiday('Juneteenth', month=6, day=19, observance=nearest_workday),
        Holiday('USIndependenceDay', month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday('Christmas', month=12, day=25, observance=nearest_workday)
    ]

def get_market_calendar():
    return NYSECalendar()

def is_market_holiday(date: pd.Timestamp) -> bool:
    """
    Checks if a date is a non-trading day (Weekend OR NYSE Holiday).
    Returns True if the market is closed.
    """
    if pd.isna(date):
        return False
        
    # 1. Check Weekend (Saturday=5, Sunday=6)
    if date.weekday() >= 5:
        return True
        
    # 2. Check Holiday
    cal = get_market_calendar()
    # holidays() returns a DatetimeIndex of holidays between start/end
    # We strictly check if the date IS a holiday
    holidays = cal.holidays(start=date, end=date)
    return not holidays.empty

def get_effective_anchor_date(date: pd.Timestamp) -> pd.Timestamp:
    """
    Returns the last valid trading day on or before 'date'.
    If 'date' is a holiday/weekend, snaps backward to the most recent trading close.
    Used for rolling horizon calculations to ensure weekend reports match Friday reports.
    """
    d = pd.Timestamp(date)
    # Simple loop back (max 10 days to be safe)
    for _ in range(10):
        if not is_market_holiday(d):
            return d
        d -= pd.Timedelta(days=1)
    return d # Fallback

def get_horizon_target_date(as_of: pd.Timestamp, label: str) -> pd.Timestamp:
    """
    Returns the ideal TARGET calendar date for a given horizon, before snapping to any trading calendar.
    Centralized logic for Portfolio, Asset Class, and Security calculations.
    
    Logic:
      - 1D:  one day prior to effective anchor (to capture prev close)
      - 1W:  7 days prior to effective anchor
      - MTD: Last day of prior month
      - 1M:  1 month prior (calendar offset)
      - YTD: Last day of prior year
      - 3M..5Y: Fixed day counts
    """
    effective_as_of = get_effective_anchor_date(as_of)

    if label == "1D":
        # Target is strictly 1 day before the effective anchor
        # (e.g. Fri -> Thu). Snapping <= Thu will catch Thu Close.
        return effective_as_of - pd.Timedelta(days=1)

    if label == "1W":
        return effective_as_of - pd.Timedelta(days=7)

    if label == "MTD":
        return as_of.replace(day=1) - pd.Timedelta(days=1)

    if label == "1M":
        return effective_as_of - pd.DateOffset(months=1)

    if label == "YTD":
        return as_of.replace(month=1, day=1) - pd.Timedelta(days=1)
    
    days_map = {
        "3M": 90, "6M": 180, "1Y": 365, "3Y": 365 * 3, "5Y": 365 * 5
    }
    if label in days_map:
        return effective_as_of - timedelta(days=days_map[label])
        
    return None

# ------------------------------------------------------------
# Universal Annualization Logic Gate
# ------------------------------------------------------------
def annualize_return(r_cum: float, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    """
    Universal Logic Gate for Return Calculation:
    - Calculates exact duration in years using 365.25 day basis.
    - If duration > 1.0 years (strictly > 365 days approx), Annualize (CAGR).
    - If duration <= 1.0 years, keep Cumulative.
    """
    if pd.isna(r_cum):
        return np.nan
        
    days = (end_date - start_date).days
    if days <= 0:
        return r_cum

    years = days / 365.25
    
    # Strictly greater than 1.0 years triggers annualization
    # 365 days / 365.25 = 0.9993 <= 1.0 -> Cumulative
    # 366 days / 365.25 = 1.0020 > 1.0 -> Annualized
    if years > 1.0:
        # Prevent math domain error for -100% loss
        if r_cum <= -1.0:
            return -1.0
        return (1.0 + r_cum) ** (1.0 / years) - 1.0
        
    return r_cum

def is_annualized(start_date: pd.Timestamp, end_date: pd.Timestamp) -> bool:
    """
    Helper to check if a period triggers annualization logic.
    Matches the logic in annualize_return exactly.
    """
    if pd.isna(start_date) or pd.isna(end_date):
        return False
        
    days = (end_date - start_date).days
    if days <= 0:
        return False

    years = days / 365.25
    return years > 1.0

# ------------------------------------------------------------
# Build portfolio value series (including CASH)
# ------------------------------------------------------------

def build_portfolio_value_series_from_flows(
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    cashflows_path: str = CASHFLOWS_FILE,
    cashflows_df: pd.DataFrame = None,
) -> pd.Series:
    """
    Build daily portfolio value from flows:

      - Start from zero positions and zero cash.
      - Apply all rows in cashflows.csv (or cashflows_df) in chronological order.
      - Treat CASH rows as pure external cash flows.
      - For each trading day in `prices.index`:
          * PV(t) = cash_before_flows_on_t + Σ shares_i * price_i(t)
          * then apply any flows dated exactly t for use on the next day.

    This guarantees:
      - Sum(amount) path builds to final CASH exactly.
      - Sum(shares) path builds to final holdings exactly.
    """

    # ----- Load raw cashflows -----
    if cashflows_df is not None:
        raw = cashflows_df.copy()
    else:
        raw = pd.read_csv(cashflows_path)

    raw.columns = [c.lower() for c in raw.columns]

    required = {"date", "ticker", "shares", "amount"}
    if not required.issubset(raw.columns):
        raise ValueError(
            f"cashflows file must contain columns {required} for flow-based PV."
        )

    raw["date"] = pd.to_datetime(raw["date"])
    raw["ticker"] = raw["ticker"].astype(str).str.upper()
    raw["shares"] = raw["shares"].astype(float)
    raw["amount"] = raw["amount"].astype(float)
    raw = raw.sort_values("date").reset_index(drop=True)

    # ----- Normalize price index -----
    pv_index = prices.index
    if not isinstance(pv_index, pd.DatetimeIndex):
        pv_index = pd.to_datetime(pv_index)
        prices.index = pv_index
    pv_index = pv_index.sort_values()

    # Universe of tickers we expect to see prices for
    holdings_tickers = set(holdings["ticker"].astype(str).str.upper())
    flow_tickers = set(raw["ticker"].unique())
    track_tickers = (holdings_tickers | flow_tickers) - {"CASH"}

    # Initialize positions and cash
    positions = {t: 0.0 for t in track_tickers}
    cash_balance = 0.0

    # Pre-apply flows strictly before the first price date
    flow_idx = 0
    n_flows = len(raw)
    first_price_date = pv_index.min()

    while flow_idx < n_flows and raw.loc[flow_idx, "date"] < first_price_date:
        row = raw.loc[flow_idx]
        t = row["ticker"]
        if t == "CASH":
            cash_balance += row["amount"]
        else:
            if t not in positions:
                positions[t] = 0.0
            positions[t] += row["shares"]
            cash_balance += row["amount"]
        flow_idx += 1

    # ----- Build PV series day by day -----
    pv = pd.Series(index=pv_index, dtype=float)
    cash_trace = pd.Series(index=pv_index, dtype=float)

    for current_date in pv_index:
        # Apply any flows dated strictly before current_date (but after prior dates)
        while flow_idx < n_flows and raw.loc[flow_idx, "date"] < current_date:
            row = raw.loc[flow_idx]
            t = row["ticker"]
            if t == "CASH":
                cash_balance += row["amount"]
            else:
                if t not in positions:
                    positions[t] = 0.0
                positions[t] += row["shares"]
                cash_balance += row["amount"]
            flow_idx += 1

        # ------------------------------------------------------------
        # GIPS-CORRECT ORDER:
        # 1. Apply flows dated on current_date (start-of-day)
        # 2. Snapshot PV after flows (end-of-day)
        # ------------------------------------------------------------

        # 1. Apply flows that occur exactly on current_date
        while flow_idx < n_flows and raw.loc[flow_idx, "date"] == current_date:
            row = raw.loc[flow_idx]
            t = row["ticker"]
            if t == "CASH":
                cash_balance += row["amount"]
            else:
                if t not in positions:
                    positions[t] = 0.0
                positions[t] += row["shares"]
                cash_balance += row["amount"]
            flow_idx += 1

        # 2. Snapshot PV AFTER today's flows using end-of-day prices
        total = cash_balance
        for t, qty in positions.items():
            if abs(qty) < 1e-9:
                continue
            if t not in prices.columns:
                raise ValueError(
                    f"Missing price data for ticker '{t}' while building PV from flows."
                )
            px = prices.at[current_date, t]
            if pd.isna(px):
                continue
            total += qty * float(px)

        pv.loc[current_date] = total
        cash_trace.loc[current_date] = cash_balance


    # ----- Sanity check against final holdings -----
    holdings_map = {
        str(t).upper(): float(s)
        for t, s in zip(holdings["ticker"], holdings["shares"])
    }

    mismatches = []

    for t, target_shares in holdings_map.items():
        if t == "CASH":
            continue
        model_shares = positions.get(t, 0.0)
        if abs(model_shares - target_shares) > 1e-6:
            mismatches.append((t, model_shares, target_shares))

    target_cash = holdings_map.get("CASH", None)
    if target_cash is not None and abs(cash_balance - target_cash) > 1e-6:
        mismatches.append(("CASH", cash_balance, target_cash))

    # Allow tiny rounding drift for CASH only (≤ $0.50)
    filtered = []
    for (tkr, flows_val, hold_val) in mismatches:
        if tkr == "CASH" and abs(flows_val - hold_val) <= 0.50:
            continue
        filtered.append((tkr, flows_val, hold_val))

    if filtered:
        # ------------------------------------------------------------
        # Optional Settlement Bridge for External Holdings
        # ------------------------------------------------------------
        # If the ONLY mismatch is CASH and we recently sold external
        # holdings, allow a temporary cash delta to reconcile while
        # broker cash settles. This prevents PV failure during T+2.
        if len(filtered) == 1 and filtered[0][0] == "CASH":
            cash_delta = cash_balance - float(target_cash)

            # Only apply when flows show MORE cash than holdings
            if cash_delta > 0:
                external_tickers = set()
                external_file = "holdings_external.csv"
                if os.path.exists(external_file):
                    try:
                        ext_df = pd.read_csv(external_file)
                        if "ticker" in ext_df.columns:
                            external_tickers = set(
                                ext_df["ticker"].astype(str).str.upper().tolist()
                            )
                    except Exception:
                        external_tickers = set()

                if external_tickers:
                    tx_external = raw[raw["ticker"].isin(external_tickers)].copy()
                    if not tx_external.empty:
                        if "type" in tx_external.columns:
                            tx_external["type"] = tx_external["type"].fillna("").astype(str).str.upper()
                            tx_external = tx_external[tx_external["type"] == "TRADE"]

                        # Settlement window (T+2 with weekend buffer)
                        settlement_window_days = 5
                        as_of = pv_index.max()
                        window_start = as_of - pd.Timedelta(days=settlement_window_days)
                        tx_recent = tx_external[tx_external["date"] >= window_start]

                        recent_net_cash = float(tx_recent["amount"].sum()) if not tx_recent.empty else 0.0

                        # Allow small rounding tolerance
                        if recent_net_cash >= cash_delta - 0.50:
                            print(
                                f"⚠️  CASH reconciliation bridged for unsettled external sales: +{cash_delta:.2f}"
                            )
                            pv.attrs["cash_settlement_bridge"] = {
                                "amount": float(cash_delta),
                                "as_of": pv_index.max(),
                            }
                            cash_trace.attrs["cash_settlement_bridge"] = pv.attrs["cash_settlement_bridge"]
                            return pv, cash_trace

        raise ValueError(
            f"Flow-based PV reconciliation failed. Final positions from flows do not match holdings: {filtered}"
        )


    return pv, cash_trace


# ------------------------------------------------------------
# Portfolio TWR computation (institutional)
# ------------------------------------------------------------

def compute_period_twr(
    pv: pd.Series,
    cf: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    return_breakdown: bool = False,
) -> float:
    """
    True TWR over [start_date, end_date].

    Assumptions:
      - pv: daily portfolio value series (END-of-day values)
      - cf: external cashflows (date, amount)
      - flows with date D are applied at START-of-day D
      - flows on the start_date are part of opening capital (not adjusted again)
    """
    # 1) Restrict PV to horizon
    pv_window = pv[(pv.index >= start_date) & (pv.index <= end_date)].sort_index()
    if pv_window.empty:
        if return_breakdown:
            return np.nan, []
        return np.nan

    # 2) Restrict flows: strictly AFTER start_date, up to and including end_date
    # GIPS FIX: Guard against empty DataFrame to prevent numpy array comparison error
    if cf.empty:
        cf_window = cf.copy()
    else:
        cf_window = cf[
            (cf["date"] > start_date)
            & (cf["date"] <= end_date)
        ].copy()



    # 3) Aggregate flows by date
    cf_agg = (
        cf_window.groupby("date", as_index=False)["amount"]
        .sum()
        .sort_values("date")
    )

    # 4) Align flows to PV dates:
    #    flow on date d is applied at the start of the FIRST PV date >= d.
    #    This ensures flows on non-trading days (weekends/holidays) are
    #    picked up on the next valuation date instead of being ignored.
    flow_series = pd.Series(0.0, index=pv_window.index, dtype=float)

    pv_idx = pv_window.index
    for _, row in cf_agg.iterrows():
        d = row["date"]
        amt = float(row["amount"])

        # Find first PV date >= cashflow date
        pos = pv_idx.searchsorted(d)
        if pos >= len(pv_idx):
            # Flow occurs after our horizon end -> ignore in this window
            continue

        flow_date = pv_idx[pos]
        flow_series.loc[flow_date] += amt

    pv_prev = float(pv_window.iloc[0])
    factors: list[float] = []
    
    # Store daily breakdown if requested
    breakdown = []

    # 5) Walk day-by-day and chain daily TWRs
    
    # Corrected TWR logic to handle Day 0 return.
    # The pv_window is EOD values. pv_window[0] is EOD on start_date.
    # The base for the first day is the sum of flows that occurred ON start_date.
    
    base_0 = 0.0
    flows_on_start = cf[(cf["date"] == start_date)].copy()
    if not flows_on_start.empty:
        base_0 = flows_on_start["amount"].sum()

    # GIPS COMPLIANCE FIX: Capture Day 1 (Inception) Return.
    # If funding (base_0) happened on start_date, calculate the return: (EndVal - Funding) / Funding.
    if base_0 > 1e-6 and not pv_window.empty:
        pv_0 = float(pv_window.iloc[0])
        r_0 = (pv_0 - base_0) / base_0
        factors.append(1.0 + r_0)
        if return_breakdown:
             breakdown.append({"date": pv_window.index[0], "return": r_0})

    if pv_window.empty:
        if return_breakdown:
            return np.nan, []
        return np.nan
        
    pv_prev = float(pv_window.iloc[0])
    loop_window = pv_window.iloc[1:]

    # Standard Chain Loop
    for curr_date, pv_curr in loop_window.items():
        flow_today = float(flow_series.loc[curr_date])

        # Capital invested for this day
        base = pv_prev + flow_today
        if base <= 0:
            # Skip pathological segments (zero/negative base)
            pv_prev = pv_curr
            continue

        r = (float(pv_curr) - base) / base
        factors.append(1.0 + r)
        if return_breakdown:
            breakdown.append({"date": curr_date, "return": r})

        pv_prev = float(pv_curr)

    if not factors:
        if return_breakdown:
            return np.nan, []
        return np.nan

    twr_val = float(np.prod(factors) - 1.0)
    
    if return_breakdown:
        return twr_val, breakdown
        
    return twr_val


def get_portfolio_horizon_start(
    pv: pd.Series,
    inception_date: pd.Timestamp,
    label: str,
):
    """
    Canonical horizon start logic used for both:
      - compute_horizon_twr (portfolio TWR)
      - build_report horizon anchoring (P/L, charts, etc.)
      
    UPDATED: Uses 'get_effective_anchor_date' to ensure that if as_of is a 
    Weekend/Holiday, the lookback anchors to the last valid trading day (e.g. Friday),
    ensuring "1 Week" means "Friday to Friday" even if report is run on Sunday.
    """
    as_of = pv.index.max()
    
    # SI Special Case: Always return inception
    if label == "SI":
        return inception_date

    # Centralized Calculation of Target Date
    target_date = get_horizon_target_date(as_of, label)
    
    if target_date is None:
        return None

    # Backward Snap: Find last valid PV date <= target_date
    pv_idx = pv.index
    prev_dates = pv_idx[pv_idx <= target_date]
    
    if len(prev_dates) == 0:
        # If no history before the target date, we cannot calculate a valid horizon return
        return None

    start = prev_dates.max()

    # VALIDATION: Strict Duration Check
    # For rolling horizons (1D, 1W, 1M, MTD, etc.) AND YTD, we require full history.
    # If Inception > Target, we don't have the full period.
    # Strictly enforce gating for all metrics.
    
    if inception_date > target_date:
         return None
             
    # Ensure start is not after as_of
    if start >= as_of:
        return None

    # Clamp start to inception if it somehow precedes it
    if start < inception_date:
        start = inception_date

    return start


def compute_horizon_twr(
    pv: pd.Series,
    cf: pd.DataFrame,
    inception_date: pd.Timestamp,
    label: str,
    effective_as_of: pd.Timestamp = None,
) -> float:
    """
    Compute TWR for labeled horizon: 1D, 1W, MTD, 1M, 3M, 6M, YTD, 1Y, 3Y, 5Y.

    Conventions:
      - MTD: from calendar month start or inception (whichever is later).
      - YTD: ONLY valid if portfolio existed at calendar year start.
             If inception_date > Jan 1 => YTD = NaN.
      - 1D/1W/1M/3M/6M/1Y/3Y/5Y: require full horizon length of live history.
    """
    as_of = pv.index.max()
    
    # GIPS FIX: Snap to last valid trading day (e.g. Friday if today is Sunday)
    # This prevents artificial expanding of the denominator (days) on weekends
    if effective_as_of is not None:
        calc_end = effective_as_of
    else:
        calc_end = get_effective_anchor_date(as_of)

    start = get_portfolio_horizon_start(pv, inception_date, label)
    if start is None:
        return np.nan
        
    # Ensure start is not after calc_end
    if start >= calc_end:
        return np.nan

    r_cum = compute_period_twr(pv, cf, start, calc_end)
    return annualize_return(r_cum, start, calc_end)


# ------------------------------------------------------------
# Security-level Modified Dietz helpers
# ------------------------------------------------------------

def modified_dietz_for_ticker_window(
    ticker: str,
    price_series: pd.Series,
    tx_all: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    dividends: pd.DataFrame = None,
    return_components: bool = False,
) -> float:
    """
    Modified Dietz return for a single security over [start, end],
    using ticker-level cashflows and prices.
    
    Now includes dividends for Total Return calculation (GIPS Requirement).

    tx_all: all transactions for this ticker with columns:
            date, shares, amount  (amount: negative for buys, positive for sells)
    price_series: price history for this ticker (index = dates, values = prices)
    dividends: optional DataFrame of dividends for this ticker (date, amount)
    """
    series = price_series.dropna()
    if series.empty:
        return np.nan

    # DO NOT clamp start date. If the horizon starts before the first price,
    # V0 should be 0, and the initial purchase should be a flow.
    
    # FIX: Allow start == end (0 days) for "Bought Today" scenarios
    if end < start:
        return np.nan

    total_days = (end - start).days + 1
    if total_days == 0:
        return 0.0
    
    if total_days < 0:
        return np.nan

    # Price at start: on or strictly BEFORE start date (asof)
    p_start = series.asof(start)
    if pd.isna(p_start):
        p_start = 0.0

    # Price at end: last on or before end
    end_idx = series.index.searchsorted(end)
    if end_idx == 0:
        return np.nan
    if end_idx == len(series) or series.index[end_idx] > end:
        end_idx -= 1
    p_end = series.iloc[end_idx]

    # Build cumulative shares from transactions
    tx_sorted = tx_all.sort_values("date").copy()
    tx_sorted["cum_shares"] = tx_sorted["shares"].cumsum()

    def shares_on(date: pd.Timestamp) -> float:
        mask = tx_sorted["date"] <= date
        if not mask.any():
            return 0.0
        return float(tx_sorted.loc[mask, "cum_shares"].iloc[-1])

    shares_start = shares_on(start)
    shares_end = shares_on(end)

    # --- FIX: Inception Logic ---
    # If the window starts on/before the first trade, V0 must be 0 (Cash to Asset).
    # The initial purchase will be captured in sum_weighted_flows.
    first_tx_date = tx_all["date"].min()
    if start <= first_tx_date:
        V0 = 0.0
    else:
        V0 = shares_start * p_start
        
    V1 = shares_end * p_end
    
    # Calculate Total Dividends in window
    total_divs = 0.0
    if dividends is not None and not dividends.empty:
        # Dividends strictly inside the window (start, end]
        # (Assuming dividends paid on 'start' are reflected in p_start or prior cash balance,
        #  so we only count those received strictly after start)
        mask_div = (dividends["date"] > start) & (dividends["date"] <= end)
        total_divs = dividends.loc[mask_div, "amount"].sum()

    # Delayed Entry Logic (GIPS compliance)
    # If we start with 0 position, but buy later in the window,
    # we should measure return from the first purchase to avoid denominator dilution.
    if V0 < 1e-6:
        # Find first transaction inside the window
        first_tx_in_window = tx_sorted[(tx_sorted["date"] >= start) & (tx_sorted["date"] <= end)]
        if not first_tx_in_window.empty:
            first_date = first_tx_in_window["date"].min()
            
            # If the first trade is strictly later than start, shift start
            if first_date > start:
                start = first_date
                total_days = (end - start).days + 1
                # V0 remains 0.0 (since we didn't hold it before this trade)

    # Cashflows inside (start, end]
    # If V0 is 0 (likely inception), we must include flows ON the start date to fund the position
    if V0 < 1e-6:
        tx_window = tx_sorted[(tx_sorted["date"] >= start) & (tx_sorted["date"] <= end)].copy()
    else:
        tx_window = tx_sorted[(tx_sorted["date"] > start) & (tx_sorted["date"] <= end)].copy()
    
    # Our file uses amount negative for buys (cash out), positive for sells (cash in)
    # Contributions C_i should be positive for cash INTO the security (BUY).
    # Withdrawals W_i should be negative for cash OUT OF the security (SELL).
    # In transactions file:
    #   Buy: amount < 0 (e.g. -1000 cash). Security receives +1000 value.
    #   Sell: amount > 0 (e.g. +1000 cash). Security releases -1000 value.
    # Therefore, Net Capital Flow C = -amount.
    
    sum_weighted_flows = 0.0
    net_external_flows = 0.0

    if not tx_window.empty:
        tx_window["C"] = -tx_window["amount"]
        
        dates = tx_window["date"].tolist()
        Cs = tx_window["C"].tolist()
        
        # Weights w_i = fraction of period the cashflow is present
        # Consistent with SOD flows (TWR assumption):
        # Flow on End Date (d=end) participates for 1 day -> Weight = 1/total_days
        # Flow on Start+1 (d=start+1) participates for (Total) days -> Weight = 1.0
        weights = [((end - d).days + 1) / total_days for d in dates]
        
        sum_weighted_flows = sum(w * c for w, c in zip(weights, Cs))
        net_external_flows = sum(Cs)

    denom = V0 + sum_weighted_flows
    
    if denom <= 0:
        # Handle small denominators or zero-basis cases
        return np.nan


    gain = V1 - V0 - net_external_flows + total_divs
    
    if return_components:
        return {
            "return": gain / denom,
            "start_val": V0,
            "end_val": V1,
            "net_flow": net_external_flows,
            "income": total_divs,
            "weighted_flow": sum_weighted_flows,
            "denom": denom,
            "start_date": start,
            "end_date": end
        }
    
    return gain / denom

def modified_dietz_for_asset_class_window(
    tickers: list[str],
    prices: pd.DataFrame,
    tx_all: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    dividends: pd.DataFrame = None,
    return_components: bool = False,
    is_si: bool = False,
) -> float:
    """
    Modified Dietz return for an asset class over [start, end],
    using aggregated cashflows and market values.
    """
    if not tickers:
        return np.nan

    total_days = (end - start).days + 1
    if total_days <= 0:
        return 0.0

    V0 = 0.0
    V1 = 0.0

    for ticker in tickers:
        if ticker not in prices.columns:
            continue
        
        price_series = prices[ticker].dropna()
        if price_series.empty:
            continue

        effective_start = start
        if end < effective_start:
            continue
            
        # Price at start: on or strictly BEFORE start date (asof)
        p_start = price_series.asof(effective_start)
        if pd.isna(p_start):
            p_start = 0.0

        # Price at end: last on or before end
        end_idx = price_series.index.searchsorted(end)
        if end_idx == 0 and price_series.index[0] > end:
            continue
        if end_idx == len(price_series) or price_series.index[end_idx] > end:
            end_idx -= 1
        p_end = price_series.iloc[end_idx]


        tx_ticker = tx_all[tx_all["ticker"] == ticker].sort_values("date").copy()
        if tx_ticker.empty:
            continue

        tx_ticker["cum_shares"] = tx_ticker["shares"].cumsum()

        def shares_on(date: pd.Timestamp) -> float:
            mask = tx_ticker["date"] <= date
            if not mask.any():
                return 0.0
            return float(tx_ticker.loc[mask, "cum_shares"].iloc[-1])

        shares_start = shares_on(start)
        shares_end = shares_on(end)

        # --- FIX: Inception Logic ---
        # If the window starts on/before the first trade of this ticker,
        # its contribution to the portfolio V0 is 0.
        first_tx_t = tx_ticker["date"].min()
        if start <= first_tx_t:
            val_start_t = 0.0
        else:
            val_start_t = shares_start * p_start

        V0 += val_start_t
        V1 += shares_end * p_end

    # Aggregate dividends for the asset class
    total_divs = 0.0
    if dividends is not None and not dividends.empty:
        mask_div = (
            (dividends["ticker"].isin(tickers)) &
            (dividends["date"] > start) &
            (dividends["date"] <= end)
        )
        total_divs = dividends.loc[mask_div, "amount"].sum()

    # Delayed Entry Logic (GIPS compliance) for Asset Class
    # If the asset class has 0 value at start, but receives inflows later,
    # shift start to the first inflow date to avoid denominator dilution.
    # GIPS FIX: Guard against empty DataFrame to prevent numpy array comparison error
    if V0 < 1e-6 and not tx_all.empty:
        # Check for any flows in this asset class >= start
        first_tx_in_window = tx_all[
            (tx_all["ticker"].isin(tickers)) &
            (tx_all["date"] >= start) &
            (tx_all["date"] <= end)
        ]
        if not first_tx_in_window.empty:
            first_date = first_tx_in_window["date"].min()
            
            if first_date > start:
                start = first_date
                total_days = (end - start).days + 1
                # V0 remains 0.0

    # GIPS FIX: Do NOT clamp end to exit date for fully liquidated positions.
    # Modified Dietz flow weighting naturally handles V1=0 cases:
    #   - The sell flow reduces the denominator proportionally to time remaining
    #   - Post-exit dividends (earned while holding) are captured through full end date
    #   - This ensures consistency between ticker-level and class-level returns
    # Previously, clamping distorted the denominator and created mismatches
    # between the "Total" row and individual ticker returns.

    # Aggregate cashflows for the asset class
    # CRITICAL FIX: For SI calculations where V0=0, include flows ON the start date
    # This ensures initial purchases are properly counted in the denominator
    # GIPS FIX: Guard against empty DataFrame to prevent numpy array comparison error
    if tx_all.empty:
        tx_window = tx_all.copy()
    elif is_si or V0 < 1e-6:
        # Include flows >= start (inception day flows counted)
        tx_window = tx_all[
            (tx_all["ticker"].isin(tickers)) &
            (tx_all["date"] >= start) &
            (tx_all["date"] <= end)
        ].copy()
    else:
        # Standard: flows after start (V0 already established)
        tx_window = tx_all[
            (tx_all["ticker"].isin(tickers)) &
            (tx_all["date"] > start) &
            (tx_all["date"] <= end)
        ].copy()

    # Net Capital Flows C = -(Buy Amount + Sell Amount)
    # Our file: Buy < 0, Sell > 0.
    # So sum(amount) is Net Cash Out/In.
    # C = -sum(amount).
    
    sum_weighted_flows = 0.0
    net_external_flows = 0.0

    if not tx_window.empty:
        tx_window["C"] = -tx_window["amount"]
        dates = tx_window["date"].tolist()
        Cs = tx_window["C"].tolist()
        
        # Consistent with SOD flows (TWR assumption)
        weights = [((end - d).days + 1) / total_days for d in dates]
        
        sum_weighted_flows = sum(w * c for w, c in zip(weights, Cs))
        net_external_flows = sum(Cs)

    denom = V0 + sum_weighted_flows

    if denom <= 0:
        if V0 <= 0 and sum_weighted_flows == 0:
            return np.nan
        # If denom is near-zero but we have valid ops, usually handle as nan or big number.
        return np.nan

    # GIPS Total Return: (V1 - V0 - C + I) / (V0 + W*C)
    # FIX: Using Unadjusted Close implies dividends are NOT accounted for in the price change.
    # We must add total_divs explicitly (Income).
    gain = V1 - V0 - net_external_flows + total_divs

    if return_components:
        return {
            "return": gain / denom,
            "start_val": V0,
            "end_val": V1,
            "net_flow": net_external_flows,
            "income": total_divs,
            "weighted_flow": sum_weighted_flows,
            "denom": denom,
            "start_date": start,
            "end_date": end
        }

    return gain / denom

def compute_security_modified_dietz(
    transactions: pd.DataFrame,
    prices: pd.DataFrame,
    holdings: pd.DataFrame,
    dividends: pd.DataFrame = None,
    horizons=HORIZONS,
    effective_as_of: pd.Timestamp = None,
) -> pd.DataFrame:

    if transactions.empty:
        return pd.DataFrame(columns=["ticker"] + list(horizons))

    as_of = prices.index.max()

    # GIPS FIX: Calculate Calculation End Date
    if effective_as_of is not None:
        calc_end = effective_as_of
    else:
        calc_end = get_effective_anchor_date(as_of)

    rows = []

    for t in sorted(transactions["ticker"].unique()):
        if t == "CASH":
            continue
        if t not in prices.columns:
            continue

        tx_all = transactions[transactions["ticker"] == t].copy()
        if tx_all.empty:
            continue

        tx_all = tx_all.sort_values("date")
        first_tx_date = tx_all["date"].min()
        price_series = prices[t].dropna()
        if price_series.empty:
            continue

        earliest_price_ticker = price_series.index.min()

        # Determine if currently held
        # Calculate net shares from all transactions
        net_shares = tx_all["shares"].sum()
        
        # Default to as_of (Open)
        last_held_date = as_of
        
        # If exited (shares ~ 0), find exit date
        if abs(net_shares) < 1e-6:
            # Use last transaction date as the end of holding period
            last_tx_date = tx_all["date"].max()
            last_held_date = last_tx_date
            
        row = {
            "ticker": t,
            "first_date": first_tx_date.date(),
            "last_date": last_held_date.date(),
            "days_held": (last_held_date - first_tx_date).days,
        }

        for h in horizons:

            # ------------------------------
            # Step 1 — Horizon window logic (Unified with Portfolio TWR)
            # ------------------------------
            if h == "SI":
                start = first_tx_date - pd.Timedelta(days=1)
                effective_start = start
            else:
                target_date = get_horizon_target_date(as_of, h)
                if target_date is None:
                    row[h] = np.nan
                    continue

                # Backward Snap: Find last valid Price Date <= target_date
                # Ensures we align with the specific ticker's trading history
                price_idx = price_series.index
                prev_dates = price_idx[price_idx <= target_date]

                if len(prev_dates) == 0:
                    row[h] = np.nan
                    continue
                else:
                    start = prev_dates.max()
                    # GIPS GATE: Security must have existed before horizon start
                    # Exception for 1D: allow first trade ON the horizon start date
                    if (h != "1D" and first_tx_date >= start) or (h == "1D" and first_tx_date > start):
                        row[h] = np.nan
                        continue

                # Step 2: Clamp to available data
                # We need prices and transaction history to compute return
                effective_start = max(start, earliest_price_ticker, first_tx_date)

                # Step 3: Strict Duration Check (Except YTD)
                # If clamping pushed start forward (e.g. missing prices), we lack full history
                if h != "YTD" and effective_start > start:
                    row[h] = np.nan
                    continue

            # ------------------------------
            # Step 4 — Safe MD computation
            # ------------------------------
            
            # Filter dividends for this ticker
            divs_t = None
            if dividends is not None and not dividends.empty:
                divs_t = dividends[dividends["ticker"] == t].copy()

            # GIPS FIX: Do NOT clamp effective_end to exit date.
            # Modified Dietz flow weighting naturally handles partial-period
            # positions (V1=0, sell flow reduces denominator proportionally).
            # Clamping distorts the denominator and drops post-exit dividends
            # (e.g., bond ETF interest paid after shares are sold).
            effective_end = calc_end

            md_ret = modified_dietz_for_ticker_window(
                t,
                price_series,
                tx_all,
                effective_start,
                effective_end,
                dividends=divs_t,
                return_components=True,
            )
            
            if isinstance(md_ret, dict):
                r_val = md_ret["return"]
                # Apply Universal Gate
                r_final = annualize_return(r_val, effective_start, calc_end)
                
                row[h] = r_final
                # Add Audit Meta Columns
                row[f"meta_{h}_start"] = md_ret["start_val"]
                row[f"meta_{h}_end"] = md_ret["end_val"]
                row[f"meta_{h}_flow"] = md_ret["net_flow"]
                row[f"meta_{h}_inc"] = md_ret["income"]
                row[f"meta_{h}_denom"] = md_ret["denom"]
                row[f"meta_{h}_is_annualized"] = is_annualized(effective_start, calc_end)
                row[f"meta_{h}_days"] = (calc_end - effective_start).days
                row[f"meta_{h}_start_date"] = md_ret.get("start_date", effective_start)
                row[f"meta_{h}_end_date"] = md_ret.get("end_date", effective_end)
            else:
                # Apply Universal Gate
                row[h] = annualize_return(md_ret, effective_start, calc_end)
                row[f"meta_{h}_is_annualized"] = is_annualized(effective_start, calc_end)
                row[f"meta_{h}_days"] = (calc_end - effective_start).days
                row[f"meta_{h}_start_date"] = effective_start
                row[f"meta_{h}_end_date"] = effective_end


        rows.append(row)

    return pd.DataFrame(rows)

# ------------------------------------------------------------
# Future Value Helpers (from build_report.py)
# ------------------------------------------------------------

def fv_lump(pv0, r, yr):
    return pv0 * ((1 + r) ** yr)

def fv_contrib(c, r, yr):
    monthly_r = r / 12.0
    n = yr * 12
    if monthly_r == 0:
        return c * n
    return c * (( (1 + monthly_r) ** n - 1 ) / monthly_r)


# ------------------------------------------------------------
# Cash Yield Logic
# ------------------------------------------------------------

def compute_cash_yield(
    cash_trace: pd.Series,
    interest_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    return_components: bool = False
):
    """
    Computes annualized cash yield based on interest income and average daily balance.
    Return = Sum(Interest) / Average(Cash Balance).
    Annualized if > 1 year.
    """
    if cash_trace.empty:
        if return_components:
            return {"return": 0.0, "start_val": 0.0, "end_val": 0.0, "net_flow": 0.0, "income": 0.0, "denom": 0.0}
        return 0.0
        
    # Filter Cash Trace to window [start, end]
    # Includes start date balance as base
    sub_trace = cash_trace[(cash_trace.index >= start_date) & (cash_trace.index <= end_date)]
    if sub_trace.empty:
        if return_components:
            return {"return": 0.0, "start_val": 0.0, "end_val": 0.0, "net_flow": 0.0, "income": 0.0, "denom": 0.0}
        return 0.0
        
    avg_balance = sub_trace.mean()
    if avg_balance <= 0.01: # Avoid div/0 or negative base issues
        if return_components:
            return {"return": 0.0, "start_val": 0.0, "end_val": 0.0, "net_flow": 0.0, "income": 0.0, "denom": 0.0}
        return 0.0
        
    # Filter Interest to window (start, end]
    # Interest paid ON start date usually belongs to prior period
    total_interest = 0.0
    if not interest_df.empty:
        mask = (interest_df["date"] > start_date) & (interest_df["date"] <= end_date)
        total_interest = interest_df.loc[mask, "amount"].sum()
        
    yield_val = total_interest / avg_balance
    
    # Universal Annualization Gate
    final_ret = annualize_return(yield_val, start_date, end_date)
    
    if return_components:
        # Determine Start/End Vals for Audit
        # start_val: Balance on or before start_date
        # end_val: Balance on or before end_date
        # trace indices are dates. Use asof logic via searchsorted.
        sorted_dates = cash_trace.index # assumed sorted
        
        start_idx = sorted_dates.searchsorted(start_date, side='right') - 1
        start_val = 0.0
        if start_idx >= 0:
            start_val = float(cash_trace.iloc[start_idx])
            
        end_idx = sorted_dates.searchsorted(end_date, side='right') - 1
        end_val = 0.0
        if end_idx >= 0:
            end_val = float(cash_trace.iloc[end_idx])
            
        # Net Flow for Audit Formula: (End - Start - Flow + Inc) / Denom
        # We want Gain = Income(Inc)
        # Inc = End - Start - Flow + Inc
        # Flow = End - Start
        net_flow = end_val - start_val
    
        return {
            "return": final_ret,
            "start_val": start_val,
            "end_val": end_val,
            "net_flow": net_flow,
            "income": total_interest,
            "denom": avg_balance,
            "start_date": start_date,
            "end_date": end_date
        }
        
    return final_ret


