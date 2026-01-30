import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import BENCHMARK_PRESETS, GLOBAL_PALETTE, RISK_FREE_RATE, TARGET_WEIGHT_PRESET_NAME
from data_loader import fetch_price_history, load_holdings
from portfolio_engine import compute_drawdown_series
from components.monte_carlo import ASSET_CLASS_BENCHMARKS

# ============================================================
# STRATEGY BACKTESTING ENGINE (SSOT for Backtesting Logic)
# ============================================================

LOOKBACK_YEARS = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10, "15Y": 15}
MAX_BACKTEST_YEARS = 20
GAP_BUFFER_DAYS = 30

# Known benchmark classifications (fallback when ticker not in holdings)
KNOWN_BENCHMARK_CLASSES = {
    # US Equity
    "SPY": "US Large Cap",
    "VOO": "US Large Cap",
    "IVV": "US Large Cap",
    "VTI": "US Large Cap",
    "ITOT": "US Large Cap",
    "SCHB": "US Large Cap",
    "SPTM": "US Large Cap",
    "DIA": "US Large Cap",
    "SPLG": "US Large Cap",
    "VV": "US Large Cap",
    "SCHX": "US Large Cap",
    "IWB": "US Large Cap",
    "RSP": "US Large Cap",
    "VONE": "US Large Cap",
    "QQQ": "US Growth",
    "QQQM": "US Growth",
    "SCHG": "US Growth",
    "VUG": "US Growth",
    "IWF": "US Growth",
    "IVW": "US Growth",
    "IUSG": "US Growth",
    "MGK": "US Growth",
    "VONG": "US Growth",
    "SPYG": "US Growth",
    "IWY": "US Growth",
    "ONEQ": "US Growth",
    "IWM": "US Small Cap",
    "AVUV": "US Small Cap",
    "VBR": "US Small Cap",
    "IJR": "US Small Cap",
    "VB": "US Small Cap",
    "SCHA": "US Small Cap",
    "IWN": "US Small Cap",
    "IWO": "US Small Cap",
    "SLY": "US Small Cap",
    "SPSM": "US Small Cap",
    "VBK": "US Small Cap",
    "VIOO": "US Small Cap",
    "VIOV": "US Small Cap",
    "IJS": "US Small Cap",
    "VTWO": "US Small Cap",
    "VTWV": "US Small Cap",
    # International Equity
    "VXUS": "International Equity",
    "VEU": "International Equity",
    "VEA": "International Equity",
    "IEFA": "International Equity",
    "EFA": "International Equity",
    "EEM": "International Equity",
    "VWO": "International Equity",
    "IEMG": "International Equity",
    "ACWX": "International Equity",
    "IXUS": "International Equity",
    "SCHF": "International Equity",
    "SPDW": "International Equity",
    "SPEM": "International Equity",
    "SCZ": "International Equity",
    "VSS": "International Equity",
    "IDEV": "International Equity",
    "EEMV": "International Equity",
    # Fixed Income
    "BND": "Fixed Income",
    "BNDX": "Fixed Income",
    "AGG": "Fixed Income",
    "SCHZ": "Fixed Income",
    "IUSB": "Fixed Income",
    "IAGG": "Fixed Income",
    "LQD": "Fixed Income",
    "VCIT": "Fixed Income",
    "VCSH": "Fixed Income",
    "VCLT": "Fixed Income",
    "HYG": "Fixed Income",
    "JNK": "Fixed Income",
    "TLT": "Fixed Income",
    "IEF": "Fixed Income",
    "SHY": "Fixed Income",
    "IEI": "Fixed Income",
    "TIP": "Fixed Income",
    "SCHP": "Fixed Income",
    "VGIT": "Fixed Income",
    "VGSH": "Fixed Income",
    "VGLT": "Fixed Income",
    "GOVT": "Fixed Income",
    "MUB": "Fixed Income",
    "VTEB": "Fixed Income",
    "BIV": "Fixed Income",
    "BLV": "Fixed Income",
    "EDV": "Fixed Income",
    "BNDW": "Fixed Income",
    "SHV": "Fixed Income",
    "BIL": "Fixed Income",
    "IGSB": "Fixed Income",
    "BSV": "Fixed Income",
    "VMBS": "Fixed Income",
    "TLH": "Fixed Income",
    "EMB": "Fixed Income",
    # Commodities / Real Assets
    "GLD": "Gold / Precious Metals",
    "IAU": "Gold / Precious Metals",
    "SGOL": "Gold / Precious Metals",
    "SIVR": "Gold / Precious Metals",
    "SLV": "Gold / Precious Metals",
    "PPLT": "Gold / Precious Metals",
    "GLDM": "Gold / Precious Metals",
    "DBC": "Commodities",
    "PDBC": "Commodities",
    "GSG": "Commodities",
    "COMT": "Commodities",
    "CMDY": "Commodities",
    "USCI": "Commodities",
    # Digital Assets
    "BTC-USD": "Digital Assets",
    "IBIT": "Digital Assets",
    "FBTC": "Digital Assets",
    "BTCO": "Digital Assets",
    "HODL": "Digital Assets",
    "EZBC": "Digital Assets",
    "BITO": "Digital Assets",
    "BTF": "Digital Assets",
    "XBTF": "Digital Assets",
    "ETHA": "Digital Assets",
    "FETH": "Digital Assets",
    "QETH": "Digital Assets",
    "ETHV": "Digital Assets",
    "EZET": "Digital Assets",
    "ARKB": "Digital Assets",
}


def _normalize_weights(raw_weights: dict) -> dict:
    cleaned = {}
    for k, v in (raw_weights or {}).items():
        if k is None:
            continue
        t = str(k).strip().upper()
        try:
            w = float(v)
        except Exception:
            continue
        if not t or w <= 0:
            continue
        cleaned[t] = w

    total = sum(cleaned.values())
    if total <= 0:
        return {}

    return {t: w / total for t, w in cleaned.items()}


def _get_portfolio_weights(data) -> dict:
    sec_table = data.get("sec_table_current", pd.DataFrame())
    if sec_table.empty:
        return {}

    subset = sec_table[sec_table["ticker"] != "CASH"].copy()
    subset = subset[subset["market_value"] > 0]
    if subset.empty:
        return {}

    total_mv = subset["market_value"].sum()
    if total_mv <= 0:
        return {}

    weights = (subset.set_index("ticker")["market_value"] / total_mv).to_dict()
    return _normalize_weights(weights)


def _get_preset_map() -> dict:
    preset_map = {}
    for preset in BENCHMARK_PRESETS:
        name = preset.get("name")
        weights = preset.get("weights", {})
        if name and weights:
            preset_map[name] = _normalize_weights(weights)
    return preset_map


def _get_target_weight_column(holdings: pd.DataFrame) -> str | None:
    if holdings.empty:
        return None

    cols = [str(c).strip().lower() for c in holdings.columns]
    if "target_pct" in cols:
        return holdings.columns[cols.index("target_pct")]

    for key in cols:
        if "target" in key and "pct" in key:
            return holdings.columns[cols.index(key)]

    for key in cols:
        if "target" in key:
            return holdings.columns[cols.index(key)]

    return None


def _get_target_weights(data) -> dict:
    holdings = data.get("holdings", pd.DataFrame()) if data else pd.DataFrame()
    if holdings.empty:
        holdings = load_holdings()

    if holdings.empty or "ticker" not in [c.lower() for c in holdings.columns]:
        return {}

    holdings = holdings.copy()
    holdings.columns = [str(c).strip().lower() for c in holdings.columns]

    target_col = _get_target_weight_column(holdings)
    if not target_col:
        return {}

    subset = holdings[holdings["ticker"].astype(str).str.upper() != "CASH"].copy()
    if subset.empty:
        return {}

    raw_weights = subset.set_index("ticker")[target_col].to_dict()
    return _normalize_weights(raw_weights)


def _get_common_start_date(prices: pd.DataFrame, tickers: list[str]) -> pd.Timestamp | None:
    if prices.empty:
        return None

    starts = []
    for t in tickers:
        if t not in prices.columns:
            return None
        series = prices[t].dropna()
        if series.empty:
            return None
        starts.append(series.index.min())

    if not starts:
        return None

    return max(starts)


def _get_requested_start_date(prices: pd.DataFrame, lookback: str) -> pd.Timestamp | None:
    if prices.empty:
        return None

    end_date = prices.index.max()
    if lookback in LOOKBACK_YEARS:
        return end_date - pd.DateOffset(years=LOOKBACK_YEARS[lookback])

    return end_date - pd.DateOffset(years=MAX_BACKTEST_YEARS)


def _snap_to_trading_start(index: pd.DatetimeIndex, requested_start: pd.Timestamp) -> pd.Timestamp:
    if index.empty:
        return requested_start

    pos = index.searchsorted(requested_start)
    if pos >= len(index):
        return index[-1]
    return index[pos]


def _build_rebalance_dates(index: pd.DatetimeIndex) -> set[pd.Timestamp]:
    if index.empty:
        return set()

    q_ends = pd.date_range(start=index.min(), end=index.max(), freq="QE")
    rebalance_dates = set()

    for q_end in q_ends:
        pos = index.searchsorted(q_end, side="right") - 1
        if pos >= 0:
            rebalance_dates.add(index[pos])

    return rebalance_dates


def _simulate_rebalanced_portfolio(returns_df: pd.DataFrame, target_weights: np.ndarray) -> pd.Series:
    if returns_df.empty:
        return pd.Series(dtype=float)

    w = np.array(target_weights, dtype=float)
    if w.sum() <= 0:
        return pd.Series(dtype=float)

    w = w / w.sum()
    rebalance_dates = _build_rebalance_dates(returns_df.index)

    port_returns = []
    for date, row in returns_df.iterrows():
        r = row.to_numpy(dtype=float)
        port_ret = float(np.dot(w, r))
        port_returns.append(port_ret)

        w = w * (1.0 + r)
        total = w.sum()
        if total > 0:
            w = w / total

        if date in rebalance_dates:
            w = target_weights / target_weights.sum()

    return pd.Series(port_returns, index=returns_df.index)


def _compute_backtest_metrics(returns: pd.Series, growth_of_one: pd.Series) -> dict:
    if returns.empty or growth_of_one.empty:
        return {
            "cagr": np.nan,
            "vol": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "max_drawdown": np.nan,
            "downside_dev": np.nan,
            "dd_peak": np.nan,
            "dd_trough": np.nan,
        }

    start_date = growth_of_one.index.min()
    end_date = growth_of_one.index.max()
    days = max((end_date - start_date).days, 1)
    years = days / 365.25

    ending = growth_of_one.iloc[-1]
    cagr = (ending ** (1.0 / years) - 1.0) if years > 0 else np.nan

    vol = returns.std() * np.sqrt(252)
    mean_ret = returns.mean() * 252
    rf = RISK_FREE_RATE
    sharpe = (mean_ret - rf) / vol if vol > 0 else np.nan

    downside = returns[returns < 0]
    if not downside.empty:
        downside_dev = np.sqrt((downside ** 2).mean()) * np.sqrt(252)
        sortino = (mean_ret - rf) / downside_dev if downside_dev > 0 else np.nan
    else:
        sortino = np.nan
        downside_dev = np.nan

    drawdown_series, max_dd, _ = compute_drawdown_series(growth_of_one)
    if not drawdown_series.empty:
        hwm = growth_of_one.cummax()
        drawdown = (growth_of_one - hwm) / hwm
        trough_date = drawdown.idxmin()
        dd_peak = float(hwm.loc[trough_date])
        dd_trough = float(growth_of_one.loc[trough_date])
    else:
        dd_peak = np.nan
        dd_trough = np.nan

    return {
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd / 100.0 if max_dd is not None else np.nan,
        "downside_dev": downside_dev,
        "dd_peak": dd_peak,
        "dd_trough": dd_trough,
    }


def _build_holdings_map(data) -> dict:
    holdings_df = data.get("holdings", pd.DataFrame()) if isinstance(data, dict) else pd.DataFrame()
    if holdings_df.empty:
        holdings_df = load_holdings()
    if holdings_df.empty:
        return {}
    holdings_df = holdings_df.copy()
    holdings_df["ticker"] = holdings_df["ticker"].astype(str).str.upper()
    return holdings_df.set_index("ticker")["asset_class"].to_dict()


def _resolve_asset_class(ticker: str, holdings_map: dict) -> str:
    if ticker in holdings_map:
        return holdings_map[ticker]

    if ticker in KNOWN_BENCHMARK_CLASSES:
        return KNOWN_BENCHMARK_CLASSES[ticker]

    for ac, proxy in ASSET_CLASS_BENCHMARKS.items():
        if proxy and proxy.upper() == ticker:
            return ac

    return "US Large Cap"


def _resolve_proxy_ticker(asset_class: str) -> str:
    proxy = ASSET_CLASS_BENCHMARKS.get(asset_class)
    if proxy:
        return proxy
    return ASSET_CLASS_BENCHMARKS.get("US Large Cap", "SPY")


def _collect_proxy_tickers(tickers: list[str], holdings_map: dict) -> set[str]:
    proxies = set()
    for t in tickers:
        ac = _resolve_asset_class(t, holdings_map)
        proxy = _resolve_proxy_ticker(ac)
        if proxy and proxy != "CASH":
            proxies.add(proxy)
    return proxies


def _splice_proxy_returns(
    prices: pd.DataFrame,
    requested_start_date: pd.Timestamp,
    tickers: list[str],
    holdings_map: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if prices.empty:
        return prices, pd.DataFrame()

    full_index = prices.index[prices.index >= requested_start_date]
    if full_index.empty:
        return prices, pd.DataFrame()

    healed = {}
    log_records = []

    buffer_cutoff = requested_start_date + pd.Timedelta(days=GAP_BUFFER_DAYS)

    for ticker in tickers:
        if ticker not in prices.columns:
            continue

        series = prices[ticker].reindex(full_index)
        first_valid = series.first_valid_index()

        needs_splice = False
        if first_valid is None:
            needs_splice = True
        else:
            if first_valid > buffer_cutoff:
                needs_splice = True

        if needs_splice:
            asset_class = _resolve_asset_class(ticker, holdings_map)
            proxy_ticker = _resolve_proxy_ticker(asset_class)
            if proxy_ticker == ticker:
                proxy_ticker = _resolve_proxy_ticker("US Large Cap")

            proxy_series = prices[proxy_ticker].reindex(full_index) if proxy_ticker in prices.columns else None
            proxy_first_valid = proxy_series.first_valid_index() if proxy_series is not None else None

            status = "Spliced"
            if proxy_series is None or proxy_series.dropna().empty:
                status = "Failed"
                healed[ticker] = series.ffill()
                proxy_ticker = None
            else:
                if proxy_first_valid is None or proxy_first_valid > buffer_cutoff:
                    status = "Partial History"

                orig_ret = series.pct_change()
                proxy_ret = proxy_series.pct_change()
                combined_ret = orig_ret.fillna(proxy_ret).fillna(0.0)
                synthetic = (1.0 + combined_ret).cumprod()
                if not synthetic.empty:
                    synthetic.iloc[0] = 1.0
                healed[ticker] = synthetic * 100.0

            log_records.append({
                "Ticker": ticker,
                "Status": status,
                "Proxy Used": proxy_ticker,
                "Asset Class": asset_class,
                "Original Start": first_valid.date() if first_valid is not None else None,
                "Proxy Start": proxy_first_valid.date() if proxy_first_valid is not None else None,
                "Requested Start": requested_start_date.date() if requested_start_date is not None else None,
            })
        else:
            healed[ticker] = series.ffill()

    log_df = pd.DataFrame(
        log_records,
        columns=[
            "Ticker",
            "Status",
            "Proxy Used",
            "Asset Class",
            "Original Start",
            "Proxy Start",
            "Requested Start",
        ],
    )

    if not healed:
        return prices.reindex(full_index), log_df

    healed_df = pd.DataFrame(healed, index=full_index)
    return healed_df, log_df


def get_strategy_backtest_results(
    data,
    lookback: str = "MAX",
    initial_value: float = 10000.0,
    selected_presets: list[str] | None = None,
    custom_benchmark: dict | None = None,
):
    if not data:
        return {"error": "No portfolio data available."}

    strategies = []
    portfolio_weights = _get_portfolio_weights(data)
    if portfolio_weights:
        strategies.append({"name": "My Portfolio", "weights": portfolio_weights, "primary": True})

    preset_map = _get_preset_map()
    target_weights = _get_target_weights(data)
    if target_weights:
        preset_map[TARGET_WEIGHT_PRESET_NAME] = target_weights

    if selected_presets is None:
        selected_presets = list(preset_map.keys())

    for name in selected_presets:
        weights = preset_map.get(name)
        if weights:
            strategies.append({"name": name, "weights": weights, "primary": False})

    if custom_benchmark:
        custom_name = custom_benchmark.get("name", "Custom Benchmark")
        weights = _normalize_weights(custom_benchmark.get("weights", {}))
        if weights:
            strategies.append({"name": custom_name, "weights": weights, "primary": False})

    if not strategies:
        return {"error": "No valid strategies available for backtesting."}

    all_tickers = sorted({t for s in strategies for t in s["weights"].keys()})
    if not all_tickers:
        return {"error": "No valid tickers available for backtesting."}

    holdings_map = _build_holdings_map(data)
    proxy_tickers = _collect_proxy_tickers(all_tickers, holdings_map)

    fetch_universe = sorted(set(all_tickers) | proxy_tickers)
    prices = fetch_price_history(fetch_universe, use_adj_close=True)
    if prices.empty:
        return {"error": "Price history unavailable for selected strategies."}

    # Ensure we only keep requested tickers + proxies
    prices = prices[[c for c in prices.columns if c in fetch_universe]]

    common_start = _get_common_start_date(prices, fetch_universe)
    if common_start is None:
        return {"error": "Insufficient price history for selected tickers."}

    requested_start = _get_requested_start_date(prices, lookback)
    if requested_start is None:
        return {"error": "Unable to determine requested backtest start date."}

    requested_start = _snap_to_trading_start(prices.index, requested_start)

    healed_prices, proxy_log = _splice_proxy_returns(prices, requested_start, all_tickers, holdings_map)
    if healed_prices.empty:
        return {"error": "Insufficient overlap for requested lookback window."}

    healed_prices = healed_prices[healed_prices.index >= requested_start]
    if healed_prices.empty:
        return {"error": "Insufficient overlap for requested lookback window."}

    healed_prices = healed_prices.ffill()
    returns = healed_prices.pct_change().dropna(how="all")
    if returns.empty:
        return {"error": "Unable to compute returns for selected strategies."}

    end_date = returns.index.max()
    start_date = returns.index.min()

    curves = {}
    drawdowns = {}
    metrics_rows = []
    risk_points = []

    for strat in strategies:
        name = strat["name"]
        weights = strat["weights"]
        tickers = list(weights.keys())
        strat_returns = returns[tickers].copy() if set(tickers).issubset(returns.columns) else pd.DataFrame()
        if strat_returns.empty:
            continue

        target_weights = np.array([weights[t] for t in tickers], dtype=float)
        if target_weights.sum() <= 0:
            continue

        port_returns = _simulate_rebalanced_portfolio(strat_returns, target_weights)
        if port_returns.empty:
            continue

        growth = (1.0 + port_returns).cumprod() * float(initial_value)
        growth_of_one = growth / float(initial_value)
        end_value = float(growth.iloc[-1]) if not growth.empty else float(initial_value)

        mean_ret = port_returns.mean() * 252
        vol_annual = port_returns.std() * np.sqrt(252)
        vol_daily = port_returns.std()
        rf_pct = RISK_FREE_RATE * 100.0
        days = (end_date - start_date).days if start_date is not None and end_date is not None else 0

        dd_series, max_dd, _ = compute_drawdown_series(growth_of_one)

        dd_peak_value = np.nan
        dd_trough_value = np.nan
        if not growth.empty:
            hwm_val = growth.cummax()
            drawdown_val = (growth - hwm_val) / hwm_val
            trough_date_val = drawdown_val.idxmin()
            dd_peak_value = float(hwm_val.loc[trough_date_val])
            dd_trough_value = float(growth.loc[trough_date_val])

        curves[name] = growth
        drawdowns[name] = dd_series

        metrics = _compute_backtest_metrics(port_returns, growth_of_one)
        metrics_rows.append({
            "Strategy": name,
            "CAGR": metrics["cagr"],
            "Volatility": metrics["vol"],
            "Sharpe": metrics["sharpe"],
            "Sortino": metrics["sortino"],
            "Max Drawdown": metrics["max_drawdown"],
            "Asset Class / Ticker": name,

            "meta_Sharpe_ret": mean_ret * 100.0,
            "meta_Sharpe_vol": vol_annual * 100.0,
            "meta_Sharpe_rf": rf_pct,

            "meta_Vol_vol": vol_annual * 100.0,
            "meta_Vol_daily": vol_daily * 100.0,

            "meta_Sortino_ret": mean_ret * 100.0,
            "meta_Sortino_rf": rf_pct,
            "meta_Sortino_down": metrics.get("downside_dev", np.nan) * 100.0,

            "meta_Drawdown_peak": metrics.get("dd_peak", np.nan),
            "meta_Drawdown_trough": metrics.get("dd_trough", np.nan),
            "meta_Drawdown_peak_value": dd_peak_value,
            "meta_Drawdown_trough_value": dd_trough_value,
            "meta_Drawdown_pct": metrics.get("max_drawdown", np.nan) * 100.0,

            "meta_CAGR_start": float(initial_value),
            "meta_CAGR_end": end_value,
            "meta_CAGR_flow": 0.0,
            "meta_CAGR_inc": 0.0,
            "meta_CAGR_denom": float(initial_value),
            "meta_CAGR_is_annualized": True,
            "meta_CAGR_days": days,
            "meta_CAGR_start_date": start_date,
            "meta_CAGR_end_date": end_date,
        })

        risk_points.append({
            "Strategy": name,
            "Return": metrics["cagr"],
            "Volatility": metrics["vol"],
            "primary": strat.get("primary", False)
        })

    scorecard = pd.DataFrame(metrics_rows)
    if not scorecard.empty:
        scorecard["_rank_cagr"] = scorecard["CAGR"].rank(pct=True)
        scorecard["_rank_sharpe"] = scorecard["Sharpe"].rank(pct=True)
        scorecard["_rank_sortino"] = scorecard["Sortino"].rank(pct=True)
        scorecard["_rank_vol"] = (1.0 - scorecard["Volatility"].rank(pct=True))
        scorecard["_rank_dd"] = (1.0 - scorecard["Max Drawdown"].rank(pct=True))

        scorecard["Overall Score"] = scorecard[[
            "_rank_cagr",
            "_rank_sharpe",
            "_rank_sortino",
            "_rank_vol",
            "_rank_dd",
        ]].mean(axis=1)

        scorecard = scorecard.sort_values("Overall Score", ascending=False).reset_index(drop=True)

        scorecard = scorecard.drop(columns=[
            "_rank_cagr",
            "_rank_sharpe",
            "_rank_sortino",
            "_rank_vol",
            "_rank_dd",
        ])
    risk_df = pd.DataFrame(risk_points)

    used_names = set(curves.keys())
    weights_rows = []
    for strat in strategies:
        name = strat.get("name", "")
        if name not in used_names:
            continue
        weights = strat.get("weights", {})
        for ticker, weight in weights.items():
            weights_rows.append({
                "Portfolio": name,
                "Ticker": ticker,
                "Weight": float(weight)
            })

    return {
        "curves": curves,
        "drawdowns": drawdowns,
        "scorecard": scorecard,
        "risk": risk_df,
        "weights_table": pd.DataFrame(weights_rows),
        "proxy_log": proxy_log,
        "start_date": start_date,
        "end_date": end_date,
        "error": None
    }


def get_strategy_backtest_growth_chart(backtest_data, initial_value: float = 10000.0):
    fig = go.Figure()
    curves = backtest_data.get("curves", {})

    if not curves:
        return fig

    primary_color = GLOBAL_PALETTE[0]
    muted_colors = [GLOBAL_PALETTE[1], GLOBAL_PALETTE[2], GLOBAL_PALETTE[4], GLOBAL_PALETTE[6], GLOBAL_PALETTE[8]]
    color_idx = 0

    for name, series in curves.items():
        is_primary = name == "My Portfolio"
        color = primary_color if is_primary else muted_colors[color_idx % len(muted_colors)]
        width = 3 if is_primary else 1.5
        opacity = 1.0 if is_primary else 0.7
        if not is_primary:
            color_idx += 1

        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name=name,
            line=dict(color=color, width=width),
            opacity=opacity,
            hovertemplate="<b>%{text}</b><br>Value: %{y:$,.2f}<extra></extra>",
            text=[name] * len(series)
        ))

    fig.update_layout(
        template="plotly_dark",
        yaxis_title="Portfolio Value ($)",
        height=450,
        margin=dict(l=40, r=20, t=40, b=40),
        hovermode="x unified"
    )
    return fig


def get_strategy_backtest_drawdown_chart(backtest_data):
    fig = go.Figure()
    drawdowns = backtest_data.get("drawdowns", {})
    if not drawdowns:
        return fig

    primary_color = GLOBAL_PALETTE[2]
    muted_colors = [GLOBAL_PALETTE[3], GLOBAL_PALETTE[5], GLOBAL_PALETTE[7], GLOBAL_PALETTE[9]]
    color_idx = 0

    for name, series in drawdowns.items():
        is_primary = name == "My Portfolio"
        color = primary_color if is_primary else muted_colors[color_idx % len(muted_colors)]
        width = 2.5 if is_primary else 1.2
        opacity = 1.0 if is_primary else 0.6
        if not is_primary:
            color_idx += 1

        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name=name,
            line=dict(color=color, width=width),
            opacity=opacity,
            hovertemplate="<b>%{text}</b><br>Drawdown: %{y:.2f}%<extra></extra>",
            text=[name] * len(series)
        ))

    fig.update_layout(
        template="plotly_dark",
        yaxis_title="Drawdown (%)",
        height=320,
        margin=dict(l=40, r=20, t=30, b=40),
        hovermode="x unified",
        yaxis=dict(autorange="reversed")
    )
    return fig


def get_strategy_backtest_risk_return_chart(backtest_data):
    fig = go.Figure()
    risk_df = backtest_data.get("risk", pd.DataFrame())
    if risk_df.empty:
        return fig

    for _, row in risk_df.iterrows():
        name = row["Strategy"]
        is_primary = row.get("primary", False)
        color = GLOBAL_PALETTE[0] if is_primary else GLOBAL_PALETTE[4]
        size = 14 if is_primary else 10
        fig.add_trace(go.Scatter(
            x=[row["Volatility"] * 100.0],
            y=[row["Return"] * 100.0],
            mode="markers+text",
            text=[name],
            textposition="top center",
            marker=dict(color=color, size=size, line=dict(width=1, color="#222")),
            name=name,
            hovertemplate="<b>%{text}</b><br>Return: %{y:.2f}%<br>Volatility: %{x:.2f}%<extra></extra>",
        ))

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Volatility (%)",
        yaxis_title="Annualized Return (%)",
        height=420,
        margin=dict(l=40, r=20, t=40, b=60)
    )
    return fig
