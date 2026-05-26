"""
Core calculations for the Mochi Portfolio Analytics dashboard.

Public function groups:
  - Trade-level utilities: extract_ticker, profit_factor, max_drawdown_path
  - Per-strategy metrics: get_cagr, get_max_drawdown, get_max_duration,
                          get_risk_ratios, get_mdd_info, get_div_ratio,
                          get_signed_position_series
  - Time-series helpers: get_monthly_returns, get_yearly_returns, rolling_sharpe
  - Tail-risk: get_var_cvar
  - Monte Carlo: monte_carlo, simulate_year, block_bootstrap_values,
                 find_max_safe_leverage
  - Portfolio aggregation: process_portfolio
  - Vol-targeted sizing: mc_vol_targeted_allocation, vt_max_load
  - Walk-forward: split_into_folds, fold_metrics, walk_forward_analysis,
                  robustness_score
  - Selection-bias correction: deflated_sharpe
  - Regime analysis: classify_regimes, regime_performance, regime_segments,
                     per_strategy_regime_pnl
  - BTC benchmark auto-fetch: fetch_btc_daily (Binance public REST, no auth)
  - Live monitoring: segment_metrics, distribution_drift_test,
                     strategy_health_status, per_strategy_live_table,
                     live_monitoring_analysis (master)
  - Extended metrics (quantstats-style): skew_kurtosis, tail_ratio, omega_ratio,
                     common_sense_ratio, ulcer_index, ulcer_performance_index,
                     recovery_factor, smart_sharpe, kelly_criterion,
                     beta_alpha_correlation, information_ratio, treynor_ratio,
                     period_returns, worst_drawdowns, consecutive_streaks,
                     period_win_rates, best_worst_extremes, time_in_market
  - Stress correlations: stress_correlation
"""

import glob
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from scipy import stats


# ============================================================================
# TRADE-LEVEL UTILITIES
# ============================================================================

COMMON_EXCHANGE = 'BINANCE|BYBIT'
COMMON_TICKERS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT']

# TradingView changed export column names in 2026 (e.g. "Date/Time" → "Date and time",
# "Position size (value)" → "Size (value)"). Normalize incoming CSVs so the rest of
# the pipeline can use a single canonical schema.
TV_COLUMN_ALIASES = {
    'Date and time': 'Date/Time',
    'Size (value)': 'Position size (value)',
}


def _normalize_tv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename new-format TradingView columns to canonical names. Idempotent."""
    rename_map = {old: new for old, new in TV_COLUMN_ALIASES.items() if old in df.columns}
    return df.rename(columns=rename_map) if rename_map else df


def extract_direction(filename: str) -> str:
    """Detect strategy direction from filename. Returns 'LONG' | 'SHORT' | 'BOTH'.

    Convention: ``BOTH`` (long+short capable), ``LONG`` (long-only),
    ``SHORT`` (short-only). Defaults to 'BOTH' when ambiguous since most
    crypto algos can short — only override if explicitly tagged.
    """
    up = filename.upper()
    # Order matters: BOTH must check first, otherwise 'LONG' substring matches
    if 'BOTH' in up:
        return 'BOTH'
    if '_LONG_' in up or up.endswith('_LONG') or up.startswith('LONG_'):
        return 'LONG'
    if '_SHORT_' in up or up.endswith('_SHORT') or up.startswith('SHORT_'):
        return 'SHORT'
    return 'BOTH'


def _fetch_ticker_regime(ticker: str, start: str, end: str,
                         lookback: int = 30,
                         bull_threshold: float = 0.05,
                         bear_threshold: float = -0.05) -> pd.Series:
    """Fetch a ticker's daily price from Binance, classify each day into
    Bull/Bear/Chop using a rolling-return lookback. Returns empty series on
    fetch failure.

    Critical for monitoring altcoin strategies: SOL/BNB regime can be very
    different from BTC regime, and a LONG-only strategy in a falling altcoin
    is not 'broken' — it's regime-headwinded.
    """
    try:
        df = fetch_btc_daily(start, end, symbol=ticker)
        if df.empty:
            return pd.Series(dtype=object)
        price = pd.Series(df['close'].values, index=df['time'])
        return classify_regimes(
            price, lookback=lookback,
            bull_threshold=bull_threshold,
            bear_threshold=bear_threshold,
        )
    except Exception as e:
        print(f"Failed to fetch {ticker} regime: {e}")
        return pd.Series(dtype=object)


def extract_family(filename: str) -> str:
    """Extract the strategy-family ID — everything before the exchange/ticker tail.

    Convention: ``FAMILY_NAME_<EXCHANGE>_<TICKER>_<EXPORT_DATE>``. The family ID
    is what identifies a unique parameter set; the same family deployed across
    multiple tickers is a cross-asset robustness pair.

    Examples:
      ``DON_CHANNEL_BOTH_1H_BINANCE_ETHUSDT_2026-05-23`` → ``DON_CHANNEL_BOTH_1H``
      ``MR_VOTING_SYSTEM_BOTH_6H_BINANCE_BTCUSDT_2025-12-18``
        → ``MR_VOTING_SYSTEM_BOTH_6H``
    """
    m = re.match(rf'^(.+?)_(?:{COMMON_EXCHANGE})_', filename)
    if m:
        return m.group(1)
    # Fallback: strip a trailing _TICKER_DATE if pattern is recognisable
    m = re.match(r'^(.+?)_(?:BTC|ETH|SOL|XRP|BNB)USDT', filename)
    if m:
        return m.group(1)
    return filename


def extract_ticker(filename: str) -> str:
    """Extract ticker symbol from strategy filename."""
    m = re.search(rf'(?:{COMMON_EXCHANGE})_([A-Z0-9\.]+)_', filename)
    if m:
        return m.group(1)
    for t in COMMON_TICKERS:
        if t in filename:
            return t
    return "UNKNOWN"


def profit_factor(trade_pnls: np.ndarray) -> float:
    """gains / |losses|; inf when no losses."""
    gains = trade_pnls[trade_pnls > 0].sum()
    losses = trade_pnls[trade_pnls < 0].sum()
    denom = abs(losses)
    return np.inf if denom == 0 else float(gains / denom)


def max_drawdown_path(equity_path: np.ndarray) -> Tuple[float, float]:
    """Return (mdd_dollars, mdd_pct_of_peak) for a single equity path."""
    if len(equity_path) == 0:
        return 0.0, 0.0
    peaks = np.maximum.accumulate(equity_path)
    dd_dollars = peaks - equity_path
    mdd_dollars = float(dd_dollars.max())
    with np.errstate(divide='ignore', invalid='ignore'):
        dd_pcts = dd_dollars / peaks
        dd_pcts = np.nan_to_num(dd_pcts, nan=0.0, posinf=0.0, neginf=0.0)
    return mdd_dollars, float(dd_pcts.max())


# ============================================================================
# PER-STRATEGY METRICS
# ============================================================================

def get_cagr(start_val: float, end_val: float, days: int) -> float:
    """CAGR; returns simple ROI for periods under 0.5y to avoid distortion."""
    if days < 1 or start_val <= 0 or end_val <= 0:
        return 0.0
    years = days / 365.25
    if years < 0.5:
        return (end_val / start_val) - 1
    try:
        res = (end_val / start_val) ** (1 / years) - 1
        return float(res) if np.isfinite(res) else 0.0
    except Exception:
        return 0.0


def get_max_drawdown(equity_series: pd.Series, starting_capital: float) -> Tuple[float, pd.Series]:
    """Return (mdd_pct, dd_series) where dd_pct = drop / peak (standard MDD).

    Uses peak-based denominator (industry-standard MDD definition):
    a $50 drop from a $200 peak is a 25% drawdown — regardless of starting capital.

    starting_capital is used as a floor for the running max, so before equity ever
    exceeds starting_capital the DD is measured from starting_capital itself
    (avoids spurious 0% DD when the very first observation is already underwater).
    """
    if equity_series.empty:
        return 0.0, equity_series
    running_max = np.maximum(equity_series.cummax(), starting_capital)
    nominal_dd = equity_series - running_max
    # Guard against division by zero if running_max is ever ≤ 0
    safe_denom = np.where(running_max > 0, running_max, starting_capital if starting_capital > 0 else 1.0)
    dd_series = pd.Series(nominal_dd.values / safe_denom, index=equity_series.index)
    return float(dd_series.min()), dd_series


def get_max_duration(dates: pd.Series, dd_series: pd.Series) -> int:
    """Max drawdown duration in days."""
    if dd_series.empty or len(dates) == 0:
        return 0
    dd = pd.Series(dd_series.values, index=pd.to_datetime(dates.values))
    is_peak = (dd == 0)
    last_peak_date = pd.Series(index=dd.index, dtype='datetime64[ns]')
    last_peak_date[is_peak] = dd.index[is_peak]
    last_peak_date = last_peak_date.ffill()
    durations = dd.index - last_peak_date
    return int(durations.max().days) if len(durations) else 0


def get_risk_ratios(daily_returns: pd.Series, risk_free_annual: float) -> Tuple[float, float]:
    """Annualized Sharpe and Sortino."""
    if daily_returns.empty or daily_returns.std() == 0:
        return 0.0, 0.0
    rf_daily = risk_free_annual / 365.0
    excess = daily_returns - rf_daily
    sharpe = float((excess.mean() / daily_returns.std()) * np.sqrt(365))
    downside = np.minimum(0, excess)
    downside_dev = np.sqrt(np.mean(downside ** 2))
    sortino = float((excess.mean() / downside_dev) * np.sqrt(365)) if downside_dev > 0 else 0.0
    return sharpe, sortino


def get_signed_position_series(df: pd.DataFrame) -> pd.Series:
    """Time-series of signed position value (long +, short -)."""
    mask = df['Type'].str.contains('Entry|Exit', case=False, na=False)
    ops = df[mask].copy()
    ops['Date/Time'] = pd.to_datetime(ops['Date/Time'])
    ops['SortRank'] = ops['Type'].apply(lambda x: 0 if 'Exit' in str(x) else 1)
    ops.sort_values(by=['Date/Time', 'SortRank'], inplace=True)

    current = 0.0
    history = {}
    open_trades = {}
    for _, row in ops.iterrows():
        ts = row['Date/Time']
        t = str(row['Type']).lower()
        trade_id = row.get('Trade #')
        val = float(row.get('Position size (value)', 0))
        is_short = 'short' in t
        signed = -val if is_short else val
        if 'entry' in t:
            open_trades[trade_id] = signed
            current += signed
        elif 'exit' in t:
            if trade_id in open_trades:
                current -= open_trades.pop(trade_id)
            else:
                current -= signed
        if abs(current) < 1e-9:
            current = 0.0
        history[ts] = current
    return pd.Series(history)


def get_div_ratio(all_daily_pnl: pd.DataFrame, port_daily_pnl: pd.Series) -> float:
    """Diversification ratio: sum of per-strategy stds / portfolio std."""
    indiv = all_daily_pnl.std().sum()
    port = port_daily_pnl.std()
    return float(indiv / port) if port > 0 else 1.0


def get_mdd_info(equity_series: pd.Series, starting_capital: float,
                 dd_series: Optional[pd.Series] = None):
    """Return (mdd_pct, peak_date_str, trough_date_str).

    If ``dd_series`` is provided, peak/trough are located against it so the
    dates align with whichever MaxDD convention produced it (typically the
    dd_series returned by ``get_max_drawdown`` — starting-cap relative). If
    omitted, falls back to peak-relative drawdown.
    """
    if equity_series.empty:
        return 0.0, "N/A", "N/A"
    if dd_series is not None and len(dd_series) == len(equity_series):
        dd_use = dd_series.copy()
        dd_use.index = equity_series.index
    else:
        running_max = np.maximum(equity_series.cummax(), starting_capital)
        dd_use = (equity_series - running_max) / running_max
    mdd_pct = dd_use.min()
    if mdd_pct == 0:
        return 0.0, "N/A", "N/A"
    trough_date = dd_use.idxmin()
    pre_trough = dd_use.loc[:trough_date]
    zero_days = pre_trough[pre_trough >= -1e-9]
    peak_date = zero_days.index[-1] if not zero_days.empty else pre_trough.index[0]
    return float(mdd_pct), peak_date.strftime('%Y-%m-%d'), trough_date.strftime('%Y-%m-%d')


# ============================================================================
# TIME-SERIES HELPERS
# ============================================================================

def get_monthly_returns(equity_series: pd.Series) -> pd.DataFrame:
    """Year x Month percent-return heatmap matrix."""
    if equity_series.empty:
        return pd.DataFrame()
    eq = equity_series.copy()
    eq.index = pd.DatetimeIndex(eq.index)
    monthly = eq.resample('ME').last().pct_change().fillna(0) * 100
    df = pd.DataFrame({'ret': monthly.values}, index=monthly.index)
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    pivot = df.pivot_table(index='Year', columns='Month', values='ret', aggfunc='sum')
    pivot.columns = [pd.Timestamp(2000, m, 1).strftime('%b') for m in pivot.columns]
    return pivot


def get_yearly_returns(equity_series: pd.Series) -> pd.Series:
    """Per-year percent returns."""
    if equity_series.empty:
        return pd.Series(dtype=float)
    eq = equity_series.copy()
    eq.index = pd.DatetimeIndex(eq.index)
    yearly = eq.resample('YE').last().pct_change().fillna(0) * 100
    yearly.index = yearly.index.year
    return yearly


def rolling_sharpe(daily_returns: pd.Series, window: int = 30,
                   risk_free_annual: float = 0.04) -> pd.Series:
    """Rolling annualized Sharpe."""
    rf_daily = risk_free_annual / 365.0
    excess = daily_returns - rf_daily
    return (excess.rolling(window).mean() / daily_returns.rolling(window).std()) * np.sqrt(365)


def rolling_calmar(daily_pnl: pd.Series, starting_capital: float,
                    window: int = 60) -> pd.Series:
    """Rolling Calmar over a trailing window: window-CAGR / |window-MDD|.

    Calmar tells you how much annualized return you got per unit of worst-case
    drawdown — a single ratio that captures both upside and tail risk. Live
    rolling Calmar dropping below backtest reference = strategy is generating
    less return per unit of pain.
    """
    if daily_pnl is None or len(daily_pnl) < window:
        return pd.Series(dtype=float, index=getattr(daily_pnl, 'index', None))
    eq = daily_pnl.fillna(0).cumsum() + starting_capital
    out = pd.Series(np.nan, index=eq.index, dtype=float)
    years_per_window = window / 365.25
    for i in range(window - 1, len(eq)):
        win = eq.iloc[i - window + 1 : i + 1]
        start_v = float(win.iloc[0])
        end_v = float(win.iloc[-1])
        # Skip if any operand makes CAGR undefined:
        # - start_v <= 0: division by zero / negative base
        # - end_v <= 0: (end/start)^(1/T) goes complex when ratio is negative,
        #   triggering "float() argument must be a string or a real number, not 'complex'"
        if start_v <= 0 or end_v <= 0 or years_per_window <= 0:
            continue
        try:
            cagr = (end_v / start_v) ** (1 / years_per_window) - 1
        except (ZeroDivisionError, ValueError):
            continue
        peaks = win.cummax()
        # Guard against zero/negative peaks (would give inf/nan/negative DDs)
        safe_peaks = peaks.where(peaks > 0, np.nan)
        dd = (win - safe_peaks) / safe_peaks
        mdd_val = dd.min()
        if pd.isna(mdd_val):
            continue
        mdd = abs(float(mdd_val))
        if mdd > 1e-9:
            out.iloc[i] = float(cagr / mdd)
    return out


# ============================================================================
# TAIL-RISK
# ============================================================================

def get_var_cvar(returns: np.ndarray, alpha: float = 0.05):
    """Return (VaR, CVaR) at the alpha quantile (default 5% left tail)."""
    if len(returns) == 0:
        return 0.0, 0.0
    sorted_r = np.sort(returns)
    var = float(np.quantile(sorted_r, alpha))
    tail = sorted_r[sorted_r <= var]
    cvar = float(tail.mean()) if tail.size else var
    return var, cvar


# ============================================================================
# MONTE CARLO
# ============================================================================

def block_bootstrap_values(vals: np.ndarray, n: int, B: int,
                           rng: np.random.Generator) -> np.ndarray:
    """Block bootstrap n values using blocks of length B (circular)."""
    if len(vals) == 0:
        return np.array([], dtype=float)
    B = max(1, int(B))
    out = []
    N = len(vals)
    while len(out) < n:
        s = int(rng.integers(0, N))
        e = s + B
        if e <= N:
            out.extend(vals[s:e])
        else:
            out.extend(vals[s:])
            wrap = e - N
            if wrap > 0:
                out.extend(vals[:wrap])
    return np.array(out[:n], dtype=float)


def simulate_year(trade_pnls: np.ndarray, start_equity: float, ruin_equity: float,
                  trades_per_year: int, rng: np.random.Generator,
                  block_len: int = 3) -> Tuple[float, float, float, float, bool, float, np.ndarray]:
    """Simulate one year of trading using block bootstrap.
    Returns (final_equity, return_pct, mdd_dollars, mdd_pct, ruined, run_pf, path)."""
    equity = start_equity
    path = [equity]
    ruined = False
    sampled = block_bootstrap_values(trade_pnls, trades_per_year, B=block_len, rng=rng)
    executed = []
    for pnl in sampled:
        equity += pnl
        executed.append(pnl)
        path.append(equity)
        if equity <= ruin_equity:
            ruined = True
            break
    path = np.array(path, dtype=float)
    mdd_dollars, mdd_pct = max_drawdown_path(path)
    ret = (path[-1] / start_equity) - 1.0
    executed = np.array(executed, dtype=float)
    run_pf = profit_factor(executed) if executed.size else np.inf
    return path[-1], ret, mdd_dollars, mdd_pct, ruined, run_pf, path


def monte_carlo(trade_pnls: np.ndarray, start_equity: float, ruin_equity: float,
                trades_per_year: int, n_runs: int = 2500,
                seed: Optional[int] = 42, block_len: int = 3) -> dict:
    """Run n_runs simulations of 1-year horizons."""
    rng = np.random.default_rng(seed)
    finals, rets, mdds_d, mdds_p, ruins, pfs, paths = [], [], [], [], [], [], []
    for _ in range(n_runs):
        fe, r, mdd_d, mdd_p, ruined, pf, path = simulate_year(
            trade_pnls, start_equity, ruin_equity, trades_per_year, rng, block_len=block_len
        )
        finals.append(fe); rets.append(r); mdds_d.append(mdd_d); mdds_p.append(mdd_p)
        ruins.append(ruined); pfs.append(pf); paths.append(path)
    max_len = max((len(p) for p in paths), default=1)
    padded = np.full((n_runs, max_len), np.nan, dtype=float)
    for i, p in enumerate(paths):
        padded[i, :len(p)] = p
    return {
        'final_equity': np.array(finals),
        'return': np.array(rets),
        'mdd_dollars': np.array(mdds_d),
        'mdd_pct': np.array(mdds_p),
        'ruined': np.array(ruins, dtype=bool),
        'profit_factor': np.array(pfs, dtype=float),
        'equity_paths': padded,
    }


def find_max_safe_leverage(pct_returns_net: np.ndarray, trades_per_year: int,
                           target_ror: float = 0.10, ruin_fraction: float = 0.60,
                           n_runs: int = 500, block_len: int = 5,
                           seed: Optional[int] = 42,
                           leverage_min: float = 0.01, leverage_max: float = 10.0,
                           tolerance: float = 0.02) -> Tuple[float, float]:
    """Binary-search the max leverage where RoR <= target.

    pct_returns_net: per-trade percentage returns, already net of costs.
    For leverage L on $1 capital, each trade contributes L * pct_return to equity.
    RoR is invariant to absolute capital level; only leverage drives it.
    Returns (safe_leverage, achieved_ror)."""
    if len(pct_returns_net) == 0 or pct_returns_net.std() == 0:
        return leverage_min, 0.0
    effective_block = max(1, min(block_len, max(len(pct_returns_net) // 5, 1)))

    def ror_at(L):
        if L <= 0:
            return 0.0
        scaled = pct_returns_net * L
        mc = monte_carlo(
            trade_pnls=scaled, start_equity=1.0, ruin_equity=ruin_fraction,
            trades_per_year=trades_per_year, n_runs=n_runs,
            seed=seed, block_len=effective_block,
        )
        return float(mc['ruined'].mean())

    # Bracket check
    if ror_at(leverage_min) > target_ror:
        return leverage_min, ror_at(leverage_min)
    if ror_at(leverage_max) <= target_ror:
        return leverage_max, ror_at(leverage_max)
    # Binary search
    lo, hi = leverage_min, leverage_max
    while hi - lo > tolerance:
        mid = (lo + hi) / 2
        if ror_at(mid) <= target_ror:
            lo = mid
        else:
            hi = mid
    return lo, ror_at(lo)


# ============================================================================
# PORTFOLIO AGGREGATION (load all strategy CSVs → portfolio metrics)
# ============================================================================

def process_portfolio(folder: str, total_cap: float, risk_free_rate: float,
                     oos_start: str, oos_end: str,
                     ) -> Tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    """Load every strategy CSV from `folder` and aggregate into a portfolio view.

    Each CSV is a TradingView backtest export (Exit rows carry Net P&L USDT).
    Capital is equal-weighted across N strategies. BTC benchmark is auto-fetched
    from Binance public API for B&H comparison.

    Args:
        folder: directory containing *.csv strategy exports
        total_cap: total starting capital (e.g., 8000 = $8k)
        risk_free_rate: annualized risk-free rate (e.g., 0.04 = 4%)
        oos_start: ISO date 'YYYY-MM-DD' — filter trades to ≥ this date
        oos_end: ISO date 'YYYY-MM-DD' — filter trades to ≤ this date

    Returns:
        Tuple of (metrics_df, port_stats, plot_data, exposure_df):
          • metrics_df: per-strategy stats DataFrame (Sharpe, MDD, PF, etc.)
          • port_stats: portfolio aggregate dict (TotalReturn, MaxDD, CAGR, ...)
          • plot_data: daily P&L per strategy + Portfolio Equity + B&H BTC columns
          • exposure_df: daily exposure per strategy (for max-load analysis)
    """
    files = glob.glob(f"{folder}/*.csv")
    if not files:
        return pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame()

    initial_cap_per_strat = total_cap / max(len(files), 1)
    all_daily_pnl = pd.DataFrame()
    all_daily_exposure = pd.DataFrame()
    all_position_series = []
    stats_list = []
    failed_files: List[Tuple[str, str]] = []  # (filename, error_msg)

    for file in files:
        name = Path(file).stem
        ticker = extract_ticker(name)
        try:
            df = pd.read_csv(file)
            df = _normalize_tv_columns(df)
            if 'Date/Time' not in df.columns:
                raise KeyError(f"missing 'Date/Time' (or 'Date and time') column. Got: {list(df.columns)[:6]}")
            df['Date/Time'] = pd.to_datetime(df['Date/Time'], errors='coerce')
            df.dropna(subset=['Date/Time'], inplace=True)
            df.sort_values('Date/Time', inplace=True)
            df = df[(df['Date/Time'] >= oos_start) & (df['Date/Time'] <= f"{oos_end} 23:59:59")].copy()
            if df.empty:
                continue

            exits = df[df['Type'].str.startswith('Exit', na=False)].copy()
            exits['Net P&L USDT'] = pd.to_numeric(exits['Net P&L USDT'], errors='coerce').fillna(0)
            daily_pnl = exits.set_index('Date/Time')['Net P&L USDT'].resample('D').sum().fillna(0)
            daily_pnl.name = name
            daily_equity = daily_pnl.cumsum() + initial_cap_per_strat
            daily_rets = daily_equity.pct_change().fillna(0)
            equity_curve = exits['Net P&L USDT'].cumsum() + initial_cap_per_strat

            pos_series = get_signed_position_series(df)
            pos_series.name = ticker
            all_position_series.append(pos_series)
            if not pos_series.empty:
                daily_exp = pos_series.resample('h').ffill().resample('D').mean().fillna(0)
                daily_exp.name = name
                all_daily_exposure = pd.concat([all_daily_exposure, daily_exp], axis=1, sort=False)
            indiv_max_load = pos_series.abs().max() if not pos_series.empty else 0.0

            if not exits.empty:
                start_dt = pd.to_datetime(oos_start)
                end_dt = exits['Date/Time'].max().tz_localize(None)
                days = max((end_dt - start_dt).days, 1)
                first_trade = exits['Date/Time'].min()
                last_trade = exits['Date/Time'].max()
                total_pnl = exits['Net P&L USDT'].sum()
                final_equity = initial_cap_per_strat + total_pnl
                cagr = get_cagr(initial_cap_per_strat, final_equity, days)
                mdd, dd_series = get_max_drawdown(equity_curve, initial_cap_per_strat)
                mdd_duration = get_max_duration(
                    exits['Date/Time'].reset_index(drop=True),
                    dd_series.reset_index(drop=True)
                )
                calmar = cagr / abs(mdd) if mdd != 0 else 0.0
                sharpe, sortino = get_risk_ratios(daily_rets, risk_free_rate)

                # Trade-level
                exit_pnls = exits['Net P&L USDT'].values
                wins = exit_pnls[exit_pnls > 0]
                losses = exit_pnls[exit_pnls < 0]
                n_trades = int(len(exit_pnls))
                trades_per_year = (n_trades / max(days, 1)) * 365.25
                win_rate = float(len(wins) / n_trades) if n_trades > 0 else 0.0
                avg_win = float(wins.mean()) if len(wins) else 0.0
                avg_loss = float(losses.mean()) if len(losses) else 0.0
                expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
                pf_strat = profit_factor(exit_pnls) if n_trades > 0 else 0.0

                # Position size from entries
                entries_only = df[df['Type'].str.contains('Entry', case=False, na=False)]
                pos_values = pd.to_numeric(
                    entries_only.get('Position size (value)', pd.Series(dtype=float)),
                    errors='coerce',
                ).dropna()
                avg_pos = float(pos_values.abs().mean()) if len(pos_values) > 0 else 0.0
                median_pos = float(pos_values.abs().median()) if len(pos_values) > 0 else 0.0

                stats_list.append({
                    'Strategy': name, 'Ticker': ticker,
                    'CAGR': cagr, 'Sharpe': sharpe, 'Sortino': sortino,
                    'MaxDD': mdd, 'Calmar': calmar,
                    'PF': pf_strat if np.isfinite(pf_strat) else 999.0,
                    'Trades': n_trades, 'Trades/Yr': trades_per_year,
                    'Win Rate': win_rate, 'Avg Win': avg_win, 'Avg Loss': avg_loss,
                    'Expectancy': expectancy,
                    'Avg Position $': avg_pos, 'Median Position $': median_pos,
                    'Max Load': indiv_max_load, 'Net Profit': total_pnl,
                    'start': first_trade, 'end': last_trade,
                })

            all_daily_pnl = pd.concat([all_daily_pnl, daily_pnl], axis=1, sort=False)
        except Exception as e:
            failed_files.append((name, str(e)))
            print(f"Error reading {name}: {e}")

    all_daily_pnl.fillna(0, inplace=True)
    all_daily_pnl.sort_index(inplace=True)
    if all_daily_pnl.empty:
        err_stats = {'failed_files': failed_files, 'files_scanned': len(files)}
        return pd.DataFrame(), err_stats, pd.DataFrame(), pd.DataFrame()

    port_daily_pnl = all_daily_pnl.sum(axis=1)
    port_equity = port_daily_pnl.cumsum() + total_cap
    port_daily_rets = port_equity.pct_change().fillna(0)

    p_start = pd.to_datetime(oos_start)
    p_end = port_equity.index[-1].tz_localize(None)
    p_days = max((p_end - p_start).days, 1)
    p_final = float(port_equity.iloc[-1])
    p_cagr = get_cagr(total_cap, p_final, p_days)
    p_mdd, p_dd_series = get_max_drawdown(port_equity, total_cap)
    p_duration = get_max_duration(
        port_equity.index.to_series().reset_index(drop=True),
        p_dd_series.reset_index(drop=True)
    )
    p_sharpe, p_sortino = get_risk_ratios(port_daily_rets, risk_free_rate)
    p_calmar = p_cagr / abs(p_mdd) if p_mdd != 0 else 0
    div_ratio = get_div_ratio(all_daily_pnl, port_daily_pnl)
    _, p_peak_date, p_trough_date = get_mdd_info(port_equity, total_cap)

    # Portfolio max load (sum of |signed exposures| over time)
    portfolio_load_curve = pd.Series(dtype=float)
    if all_position_series:
        full_pos = pd.concat(all_position_series, axis=1, sort=False)
        full_pos.sort_index(inplace=True)
        full_pos.ffill(inplace=True)
        full_pos.fillna(0, inplace=True)
        net_ticker_pos = full_pos.T.groupby(level=0).sum().T
        portfolio_load_curve = net_ticker_pos.abs().sum(axis=1)
        true_max_load = float(portfolio_load_curve.max())
    else:
        true_max_load = 0.0

    metrics_df = pd.DataFrame(stats_list).set_index('Strategy') if stats_list else pd.DataFrame()

    portfolio_stats = {
        'Days': p_days, 'CAGR': p_cagr, 'Sharpe': p_sharpe, 'Sortino': p_sortino,
        'MaxDD': p_mdd, 'DD Duration (Days)': p_duration, 'Calmar': p_calmar,
        'Div Ratio': div_ratio, 'Max Load': true_max_load,
        'Total Capital': total_cap, 'Final Equity': p_final,
        'MDD Peak Date': p_peak_date, 'MDD Trough Date': p_trough_date,
        'Net Profit': p_final - total_cap,
        'files_scanned': len(files),
        'files_loaded': len(stats_list),
        'failed_files': failed_files,
    }

    plot_data = all_daily_pnl.copy()
    plot_data['Portfolio Equity'] = port_equity
    plot_data['Portfolio Daily P&L'] = plot_data['Portfolio Equity'].diff().fillna(0)
    plot_data['Portfolio DD'] = p_dd_series
    if not portfolio_load_curve.empty:
        load_daily = portfolio_load_curve.resample('D').max().reindex(
            plot_data.index, method='ffill'
        ).fillna(0)
        plot_data['Portfolio Load'] = load_daily

    # BTC benchmark — auto-fetched from Binance public API (no auth, no CSV needed)
    plot_data['B&H BTC Equity'] = total_cap  # default if fetch fails
    try:
        btc_df = fetch_btc_daily(oos_start, oos_end)
        if not btc_df.empty:
            btc_close = pd.Series(btc_df['close'].values, index=btc_df['time'])
            port_idx = pd.DatetimeIndex(plot_data.index).tz_localize(None).normalize()
            btc_aligned = btc_close.reindex(port_idx, method='ffill')
            if not btc_aligned.empty and pd.notna(btc_aligned.iloc[0]):
                start_price = float(btc_aligned.iloc[0])
                plot_data['B&H BTC Equity'] = (btc_aligned / start_price) * total_cap
    except Exception as e:
        print(f"Binance BTC auto-fetch failed: {e}")

    return metrics_df, portfolio_stats, plot_data, all_daily_exposure


# ============================================================================
# VOLATILITY TARGETING (per-strategy + portfolio aggregate)
# ============================================================================
# Costs are applied INSIDE mc_vol_targeted_allocation (both at the per-trade
# pct-return level for RoR computation, and as a daily drag on the portfolio
# equity curve). No standalone cost-drag helper is needed.

def mc_vol_targeted_allocation(plot_data: pd.DataFrame, metrics_df: pd.DataFrame,
                               total_cap: float,
                               target_ror: float = 0.10, ruin_fraction: float = 0.60,
                               max_leverage_cap: float = 1.0,
                               target_portfolio_vol: float = 0.20,
                               n_runs: int = 1000, block_len: int = 5,
                               seed: Optional[int] = 42,
                               cost_bps_per_round_trip: float = 0.0,
                               slippage_bps: float = 0.0,
                               funding_bps_per_day: float = 0.0,
                               normalize_backtest_pos: bool = False) -> dict:
    """Monte Carlo + Vol Targeting combined sizing workflow.

    Two-stage leverage sizing: (1) per-strategy MC sizing to keep individual
    risk of ruin under a target, then (2) uniform portfolio scale to hit a
    target portfolio vol. Result: a leverage/position allocation that respects
    BOTH single-strategy tail risk and portfolio-level vol budget.

    Steps:
        1. Per-strategy MC binary-search for max leverage where RoR ≤ target_ror
        2. Pre-scale position $ = MC_leverage × equal_cap (per strategy)
        3. Aggregate to portfolio; measure realized portfolio annualized vol
        4. Uniform portfolio_scale = target_portfolio_vol / realized_vol
        5. Final position $ = pre × portfolio_scale  (capped at max_leverage_cap)
        6. Verify post-scale RoR per strategy (returns full MC distributions)

    Args:
        plot_data: daily P&L per strategy (from process_portfolio)
        metrics_df: per-strategy stats (Trades/Yr, Profit Factor, etc.)
        total_cap: total starting capital
        target_ror: max acceptable per-strategy risk of ruin (default 10%)
        ruin_fraction: ruin = drawdown ≥ this fraction of starting cap (default 60%)
        max_leverage_cap: hard ceiling on portfolio_scale (default 1.0)
        target_portfolio_vol: target annualized portfolio volatility (default 20%)
        n_runs: MC simulations per leverage search step (default 1000)
        block_len: block-bootstrap block length (default 5)
        seed: RNG seed for reproducibility (default 42)
        cost_bps_per_round_trip: trading cost (e.g., 11 = 11bps RT)
        slippage_bps: per-trade slippage (e.g., 2 = 2bps)
        funding_bps_per_day: perp funding rate (e.g., 0.5 = 0.5bps/day)
        normalize_backtest_pos: rescale BT positions to 1× for MC sizing

    Returns:
        Dict with keys:
          • allocation: {strategy: margin $}
          • position_sizes: {strategy: leveraged position $}
          • portfolio_pnl_pre: daily portfolio P&L before portfolio scale
          • portfolio_pnl: daily portfolio P&L after portfolio scale (NET of costs)
          • ror_pre / ror_post: per-strategy RoR before/after portfolio scale
          • mc_distributions: MC return / DD samples for each strategy (post-scale)
          • realized_vol / portfolio_scale / max_load: diagnostic metadata
    """
    ignore = {'Portfolio Equity', 'Portfolio DD', 'Portfolio Daily P&L',
              'B&H BTC Equity', 'Portfolio Load'}
    strategy_cols = [c for c in plot_data.columns if c not in ignore]
    n = max(len(strategy_cols), 1)
    equal_cap = total_cap / n

    cost_pct = (cost_bps_per_round_trip + slippage_bps) / 10000.0
    funding_pct_per_day = funding_bps_per_day / 10000.0
    annual_funding_pct = funding_pct_per_day * 365.0

    safe_leverage = {}
    achieved_rors = {}
    strategy_vols = {}
    backtest_positions = {}

    # Step 1: per-strategy MC for safe leverage
    for col in strategy_cols:
        backtest_pos = equal_cap if normalize_backtest_pos else (
            float(metrics_df.loc[col, 'Avg Position $'])
            if col in metrics_df.index and 'Avg Position $' in metrics_df.columns
            else 0.0
        )
        backtest_positions[col] = backtest_pos

        daily_pnl = plot_data[col].fillna(0).values
        trade_pnls = daily_pnl[daily_pnl != 0]
        if len(trade_pnls) == 0 or trade_pnls.std() == 0 or backtest_pos <= 0:
            safe_leverage[col] = 0.0
            achieved_rors[col] = 0.0
            strategy_vols[col] = 0.0
            continue
        tpy = max(int(metrics_df.loc[col, 'Trades/Yr']), 1) if col in metrics_df.index else 100
        pct_gross = trade_pnls / backtest_pos
        funding_per_trade_pct = annual_funding_pct / tpy
        pct_net = pct_gross - cost_pct - funding_per_trade_pct
        strategy_vols[col] = float(pct_gross.std()) * np.sqrt(tpy)

        L_safe, achieved = find_max_safe_leverage(
            pct_returns_net=pct_net, trades_per_year=tpy,
            target_ror=target_ror, ruin_fraction=ruin_fraction,
            n_runs=n_runs, block_len=block_len, seed=seed,
            leverage_min=0.01, leverage_max=max_leverage_cap,
        )
        safe_leverage[col] = L_safe
        achieved_rors[col] = achieved

    # Step 2: equal capital, initial positions
    allocation = {col: equal_cap for col in strategy_cols}
    position_pre = {col: safe_leverage[col] * equal_cap for col in strategy_cols}

    def _bt_pos(col):
        return backtest_positions.get(col, 0.0)

    # Step 3: pre-scale portfolio vol
    port_pnl_pre = pd.Series(0.0, index=plot_data.index)
    for col in strategy_cols:
        bt_pos = _bt_pos(col)
        if bt_pos <= 0:
            continue
        port_pnl_pre = port_pnl_pre + plot_data[col].fillna(0) * (position_pre[col] / bt_pos)
    port_returns_pre = port_pnl_pre / total_cap
    portfolio_vol_pre = float(port_returns_pre.std() * np.sqrt(365))

    # Step 4: portfolio scale
    if target_portfolio_vol > 0 and portfolio_vol_pre > 1e-9:
        portfolio_scale = target_portfolio_vol / portfolio_vol_pre
    else:
        portfolio_scale = 1.0

    # Step 5: final positions
    position_sizes = {col: position_pre[col] * portfolio_scale for col in strategy_cols}
    port_pnl = pd.Series(0.0, index=plot_data.index)
    for col in strategy_cols:
        bt_pos = _bt_pos(col)
        if bt_pos <= 0:
            continue
        port_pnl = port_pnl + plot_data[col].fillna(0) * (position_sizes[col] / bt_pos)

    # Apply cost drag at the LEVERAGED position scale so the equity curve,
    # CAGR, Sharpe, MaxDD, and portfolio-level MC are all net-of-cost (consistent
    # with the per-trade pct_net used in RoR computation above).
    # Daily-amortized: trade cost spread evenly across 365 days, funding charged daily.
    if cost_pct > 0 or funding_pct_per_day > 0:
        n_days = max(len(port_pnl), 1)
        for col in strategy_cols:
            if col not in metrics_df.index:
                continue
            pos = position_sizes[col]
            if pos <= 0:
                continue
            tpy = max(int(metrics_df.loc[col, 'Trades/Yr']), 1) if 'Trades/Yr' in metrics_df.columns else 0
            daily_trade_cost = (tpy * cost_pct * pos) / 365.0
            daily_funding_cost = funding_pct_per_day * pos
            port_pnl = port_pnl - (daily_trade_cost + daily_funding_cost)

    portfolio_returns = port_pnl / total_cap
    portfolio_vol = float(portfolio_returns.std() * np.sqrt(365))

    # Sum-of-weighted-strategy-vols (= portfolio vol if ρ=1)
    sum_strat_vol_contrib = sum(
        strategy_vols[col] * (position_pre[col] / total_cap) * portfolio_scale
        for col in strategy_cols
    )
    div_ratio = sum_strat_vol_contrib / portfolio_vol if portfolio_vol > 1e-9 else 1.0

    # Step 6: post-scale RoR verification (per strategy at FINAL leverage)
    achieved_ror_post = {}
    return_distributions = {}
    mdd_distributions = {}
    for col in strategy_cols:
        bt_pos = _bt_pos(col)
        if bt_pos <= 0 or safe_leverage[col] <= 0:
            achieved_ror_post[col] = 0.0
            return_distributions[col] = np.array([])
            mdd_distributions[col] = np.array([])
            continue
        daily_pnl = plot_data[col].fillna(0).values
        trade_pnls = daily_pnl[daily_pnl != 0]
        if len(trade_pnls) == 0 or trade_pnls.std() == 0:
            achieved_ror_post[col] = 0.0
            return_distributions[col] = np.array([])
            mdd_distributions[col] = np.array([])
            continue
        tpy = max(int(metrics_df.loc[col, 'Trades/Yr']), 1) if col in metrics_df.index else 100
        pct_gross = trade_pnls / bt_pos
        funding_per_trade_pct = annual_funding_pct / tpy
        pct_net = pct_gross - cost_pct - funding_per_trade_pct
        eff_leverage = safe_leverage[col] * portfolio_scale
        scaled = pct_net * eff_leverage
        block = max(1, min(block_len, max(len(scaled) // 5, 1)))
        mc = monte_carlo(
            trade_pnls=scaled, start_equity=1.0, ruin_equity=ruin_fraction,
            trades_per_year=tpy, n_runs=n_runs, seed=seed, block_len=block,
        )
        achieved_ror_post[col] = float(mc['ruined'].mean())
        return_distributions[col] = mc['return']
        mdd_distributions[col] = mc['mdd_pct']

    return {
        'safe_leverage': safe_leverage,
        'achieved_ror': achieved_rors,
        'achieved_ror_post': achieved_ror_post,
        'return_distributions': return_distributions,
        'mdd_distributions': mdd_distributions,
        'strategy_vols': strategy_vols,
        'backtest_positions': backtest_positions,
        'allocation': allocation,
        'position_sizes_pre': position_pre,
        'position_sizes': position_sizes,
        'portfolio_scale': portfolio_scale,
        'portfolio_vol_pre': portfolio_vol_pre,
        'portfolio_vol': portfolio_vol,
        'diversification_ratio': div_ratio,
        'sum_strat_vol_contrib': sum_strat_vol_contrib,
        'portfolio_returns': portfolio_returns,
        'target_ror': target_ror,
        'target_portfolio_vol': target_portfolio_vol,
        'max_leverage_cap': max_leverage_cap,
    }


def vt_max_load(exposure_df: pd.DataFrame,
                position_sizes: dict,
                backtest_positions: dict) -> Tuple[float, pd.Series]:
    """Recompute peak gross exposure (Max Load) under vol-targeted sizing.

    Scales each strategy's signed exposure by (vt_position / backtest_position),
    nets long+short on the same TICKER (since opposite positions cancel at the
    exchange), then takes max over time of sum(|netted|).

    Returns (max_load_dollars, daily_load_curve).
    """
    if exposure_df is None or exposure_df.empty:
        return 0.0, pd.Series(dtype=float)
    scaled_cols = {}
    ticker_for = {}
    for col in exposure_df.columns:
        if col not in position_sizes or col not in backtest_positions:
            continue
        bt = backtest_positions[col]
        if bt <= 0:
            continue
        scaled_cols[col] = exposure_df[col].fillna(0).values * (position_sizes[col] / bt)
        ticker_for[col] = extract_ticker(col)
    if not scaled_cols:
        return 0.0, pd.Series(dtype=float)
    scaled_df = pd.DataFrame(scaled_cols, index=exposure_df.index)
    # Net by ticker (long + short on same ticker cancel)
    scaled_df.columns = [ticker_for[c] for c in scaled_df.columns]
    netted = scaled_df.T.groupby(level=0).sum().T
    load_curve = netted.abs().sum(axis=1)
    return float(load_curve.max()), load_curve


# ============================================================================
# WALK-FORWARD / K-FOLD OOS ROBUSTNESS
# ============================================================================

def split_into_folds(daily_pnl: pd.Series, n_folds: int
                     ) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Series]]:
    """Split a daily P&L series into N equal-time folds by calendar days."""
    if daily_pnl.empty or n_folds < 1:
        return []
    dates = pd.DatetimeIndex(daily_pnl.index)
    start, end = dates[0], dates[-1]
    total_days = (end - start).days + 1
    fold_days = total_days / n_folds
    folds = []
    for i in range(n_folds):
        fold_start = start + pd.Timedelta(days=fold_days * i)
        fold_end = start + pd.Timedelta(days=fold_days * (i + 1))
        if i == n_folds - 1:
            fold_end = end + pd.Timedelta(days=1)
        mask = (dates >= fold_start) & (dates < fold_end)
        folds.append((fold_start, fold_end - pd.Timedelta(days=1), daily_pnl[mask]))
    return folds


def fold_metrics(fold_pnl: pd.Series, capital: float, rfr: float) -> dict:
    """Key performance metrics for a single fold."""
    empty = {
        'sharpe': 0.0, 'sortino': 0.0, 'cagr': 0.0, 'mdd': 0.0,
        'pf': 0.0, 'win_rate': 0.0, 'n_active_days': 0,
        'total_pnl': 0.0, 'final_equity': float(capital),
    }
    if fold_pnl.empty or capital <= 0:
        return empty
    equity = fold_pnl.cumsum() + capital
    if equity.empty:
        return empty
    rets = equity.pct_change().fillna(0)
    sharpe, sortino = get_risk_ratios(rets, rfr)
    days = max((equity.index[-1] - equity.index[0]).days, 1)
    final = float(equity.iloc[-1])
    cagr = get_cagr(capital, final, days)
    mdd, _ = get_max_drawdown(equity, capital)
    nonzero = fold_pnl[fold_pnl != 0]
    pf = profit_factor(nonzero.values) if len(nonzero) else 0.0
    if not np.isfinite(pf):
        pf = 999.0
    win_rate = float((nonzero > 0).mean()) if len(nonzero) else 0.0
    return {
        'sharpe': float(sharpe), 'sortino': float(sortino),
        'cagr': float(cagr), 'mdd': float(mdd),
        'pf': float(pf), 'win_rate': win_rate,
        'n_active_days': int(len(nonzero)),
        'total_pnl': float(fold_pnl.sum()), 'final_equity': final,
    }


def walk_forward_analysis(plot_data: pd.DataFrame, capital: float, rfr: float,
                          n_folds: int,
                          portfolio_pnl: Optional[pd.Series] = None) -> dict:
    """K-fold OOS robustness across all strategies + portfolio.

    ``portfolio_pnl`` (optional) overrides the daily portfolio P&L used for
    fold splits + portfolio metrics — e.g. pass the vol-targeted P&L Series
    so the portfolio row reflects deployed sizing rather than the raw
    equal-weight backtest. Per-strategy fold metrics still use ``plot_data``
    columns (raw P&L per strategy).
    """
    ignore = {'Portfolio Equity', 'Portfolio DD', 'Portfolio Daily P&L',
              'B&H BTC Equity', 'Portfolio Load'}
    strategy_cols = [c for c in plot_data.columns if c not in ignore]
    if not strategy_cols or plot_data.empty:
        return {'folds': [], 'strategies': {}, 'portfolio': []}
    per_strat_cap = capital / max(len(strategy_cols), 1)
    port_pnl_use = portfolio_pnl if portfolio_pnl is not None else plot_data['Portfolio Daily P&L']
    fold_splits = split_into_folds(port_pnl_use, n_folds)
    fold_info = [(s, e) for s, e, _ in fold_splits]
    strategies = {}
    for col in strategy_cols:
        strategies[col] = []
        for fs, fe, _ in fold_splits:
            mask = (plot_data.index >= fs) & (plot_data.index <= fe)
            strategies[col].append(fold_metrics(plot_data.loc[mask, col], per_strat_cap, rfr))
    portfolio = [fold_metrics(fp, capital, rfr) for _, _, fp in fold_splits]
    return {'folds': fold_info, 'strategies': strategies, 'portfolio': portfolio}


def robustness_score(strat_folds: list, sharpe_threshold: float = 0.0) -> dict:
    """Aggregate per-fold metrics into a robustness scorecard."""
    if not strat_folds:
        return {
            'mean_sharpe': 0.0, 'std_sharpe': 0.0, 'min_sharpe': 0.0,
            'pct_positive_sharpe': 0.0, 'pct_pf_above_1': 0.0,
            'consistency': 0.0, 'mean_cagr': 0.0, 'total_pnl': 0.0,
        }
    sharpes = np.array([f['sharpe'] for f in strat_folds])
    cagrs = np.array([f['cagr'] for f in strat_folds])
    pfs = np.array([f['pf'] for f in strat_folds])
    pnls = np.array([f['total_pnl'] for f in strat_folds])
    n = len(sharpes)
    pct_positive = float((sharpes > sharpe_threshold).sum() / n)
    pct_pf_above_1 = float((pfs > 1.0).sum() / n)
    std_s = float(sharpes.std())
    consistency = float(sharpes.mean() / std_s) if std_s > 1e-9 else (
        float(sharpes.mean()) if sharpes.mean() > 0 else 0.0
    )
    return {
        'mean_sharpe': float(sharpes.mean()),
        'std_sharpe': std_s,
        'min_sharpe': float(sharpes.min()),
        'pct_positive_sharpe': pct_positive,
        'pct_pf_above_1': pct_pf_above_1,
        'consistency': consistency,
        'mean_cagr': float(cagrs.mean()),
        'total_pnl': float(pnls.sum()),
    }


# ============================================================================
# DEFLATED SHARPE RATIO (Bailey & López de Prado, 2014)
# ============================================================================

def deflated_sharpe(observed_sharpe: float, n_trials: int, n_obs: int,
                    skewness: float = 0.0, kurtosis: float = 3.0) -> dict:
    """Probabilistic Sharpe Ratio adjusted for selection bias.
    Returns dict with psr, expected_max_sharpe, is_significant, sharpe_std."""
    if n_obs < 10 or n_trials < 1:
        return {'psr': 0.0, 'expected_max_sharpe': 0.0, 'is_significant': False, 'sharpe_std': 0.0}
    sr_daily = observed_sharpe / np.sqrt(365)
    e_max_daily = (
        (1 - np.euler_gamma) * stats.norm.ppf(1 - 1.0 / n_trials)
        + np.euler_gamma * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    ) / np.sqrt(n_obs)
    e_max_annual = e_max_daily * np.sqrt(365)
    sr_std = np.sqrt(
        max((1 - skewness * sr_daily + ((kurtosis - 1) / 4.0) * sr_daily ** 2) / (n_obs - 1), 1e-12)
    ) * np.sqrt(365)
    psr = float(stats.norm.cdf((observed_sharpe - e_max_annual) / sr_std))
    return {
        'psr': psr,
        'expected_max_sharpe': float(e_max_annual),
        'is_significant': psr >= 0.95,
        'sharpe_std': float(sr_std),
    }


# ============================================================================
# REGIME ANALYSIS (BTC-trend conditional)
# ============================================================================

def classify_regimes(btc_equity: pd.Series, lookback: int = 30,
                     bull_threshold: float = 0.05,
                     bear_threshold: float = -0.05) -> pd.Series:
    """Bucket each day into Bull / Bear / Chop based on BTC rolling return."""
    if btc_equity.empty:
        return pd.Series(dtype=object)
    roc = btc_equity.pct_change(lookback).fillna(0)
    regime = pd.Series('Chop', index=btc_equity.index)
    regime[roc >= bull_threshold] = 'Bull'
    regime[roc <= bear_threshold] = 'Bear'
    return regime


def regime_performance(plot_data: pd.DataFrame, regime: pd.Series,
                       capital: float, rfr: float) -> pd.DataFrame:
    """Per-regime portfolio performance stats."""
    if 'Portfolio Daily P&L' not in plot_data.columns or regime.empty:
        return pd.DataFrame()
    aligned = regime.reindex(plot_data.index, method='ffill')
    port_pnl = plot_data['Portfolio Daily P&L']
    rows = []
    for r in ['Bull', 'Bear', 'Chop']:
        mask = aligned == r
        days = int(mask.sum())
        if days == 0:
            continue
        sub = port_pnl[mask]
        eq = sub.cumsum() + capital
        rets = eq.pct_change().fillna(0)
        sh, _ = get_risk_ratios(rets, rfr)
        rows.append({
            'Regime': r, 'Days': days, 'Pct of Time': days / len(aligned),
            'Total P&L ($)': float(sub.sum()),
            'Avg Daily P&L ($)': float(sub.mean()),
            'Daily Win Rate': float((sub > 0).mean()),
            'Sharpe (annualized)': float(sh),
            'Best Day ($)': float(sub.max()),
            'Worst Day ($)': float(sub.min()),
        })
    return pd.DataFrame(rows)


def regime_segments(regime: pd.Series) -> List[Tuple[str, pd.Timestamp, pd.Timestamp, int]]:
    """Group consecutive same-regime days into [(regime, start, end, n_days), ...]."""
    if regime.empty:
        return []
    segs = []
    current = regime.iloc[0]
    start = regime.index[0]
    for i in range(1, len(regime)):
        if regime.iloc[i] != current:
            segs.append((current, start, regime.index[i - 1],
                         (regime.index[i - 1] - start).days + 1))
            current = regime.iloc[i]
            start = regime.index[i]
    segs.append((current, start, regime.index[-1], (regime.index[-1] - start).days + 1))
    return segs


def per_strategy_regime_pnl(plot_data: pd.DataFrame, regime: pd.Series) -> pd.DataFrame:
    """Total P&L by strategy × regime."""
    ignore = {'Portfolio Equity', 'Portfolio DD', 'Portfolio Daily P&L',
              'B&H BTC Equity', 'Portfolio Load'}
    strategy_cols = [c for c in plot_data.columns if c not in ignore]
    if not strategy_cols or regime.empty:
        return pd.DataFrame()
    aligned = regime.reindex(plot_data.index, method='ffill')
    rows = {}
    for r in ['Bull', 'Bear', 'Chop']:
        mask = aligned == r
        rows[r] = plot_data.loc[mask, strategy_cols].sum()
    return pd.DataFrame(rows)


# ============================================================================
# LIVE INCUBATION MONITORING
# ============================================================================
# Honest live-vs-backtest monitoring.
#
# The "honest test" is to derive sizing (leverage, position $) on BACKTEST-ONLY
# data, then apply that sizing to the live segment — replicating what a trader
# would actually have deployed at the backtest cut-off. Using full-period vol
# targeting (that includes live data in the sizing decision) leaks information
# and overstates live performance.
#
# Helpers below produce per-segment metrics, statistical drift tests
# (Kolmogorov-Smirnov, Mann-Whitney U), and rule-based health classification.

# Default split date — strategies went OOS live on this date.
LIVE_START_DEFAULT = '2025-12-03'

# ============================================================================
# KILL-RULE CONSTANTS (single source of truth — referenced in app.py captions
# and column tooltips. Change here, propagate everywhere.)
# ============================================================================

# MC envelope thresholds (percent units, 0–100)
MC_TAIL_PCT = 5        # ≤5% on either MC %ile fires KILL
MC_WARN_PCT = 15       # ≤15% on either MC %ile fires WARN (subordinate to KILL)

# Sample-size floor — guard against small-sample false positives
MIN_LIVE_TRADES = 20   # min live active-trading days before kill rule applies

# KS test sample-size floors and significance level (Edge Diagnosis)
KS_ALPHA = 0.05               # KS p-value below this → "shift" signal
MIN_BT_TRADES_FOR_KS = 20     # need ≥20 BT trades for KS to have any power
MIN_LIVE_TRADES_FOR_KS = 5    # need ≥5 live trades to compare distributions

# Rolling-window length for Sharpe / Calmar charts (days)
ROLLING_WINDOW_DAYS = 60

# Default MC bootstrap params for per-strategy evaluation
MC_DEFAULT_RUNS = 1000
MC_DEFAULT_BLOCK_LEN = 5
MC_DEFAULT_SEED = 42

# Verdict color palette (single source — duplicated 89× across files before)
VERDICT_COLORS = {
    'kill':       '#c0392b',   # red — kill / disqualified / broken edge
    'warn':       '#f39c12',   # amber — borderline
    'watch':      '#f1c40f',   # yellow — secondary attention
    'keep':       '#27ae60',   # green — qualified / stable
    'neutral':    '#7f8c8d',   # grey — BT reference / informational
    'incubating': '#95a5a6',   # grey — insufficient data
    'edge_drift': '#e67e22',   # orange — distribution drifting
    'axis':       '#34495e',   # dark grey — axes / reference lines
}


# ============================================================================
# StrategyEvaluation — typed result container for per_strategy_evaluation()
# ============================================================================
# Backwards-compatible: supports both `ev.attribute` and `ev['key']` access, so
# existing dict-style consumers in app.py keep working without migration. New
# code should prefer attribute access for IDE autocomplete and type checks.

@dataclass
class StrategyEvaluation:
    """Typed result of per_strategy_evaluation(). 22 fields.

    Access either way:
      ev.combined_verdict   (preferred — IDE autocomplete + type check)
      ev['combined_verdict'] (legacy dict-style — preserved for migration)
    """
    # Segment stats (dict shape from segment_metrics())
    bt_metrics: dict
    live_metrics: dict
    # Live observation ($)
    live_final_pnl: float
    live_dd_dollars: float
    live_trades: int
    min_live_trades: int
    # Efficiency (% comparisons)
    return_eff_pct: Optional[float]
    dd_eff_pct: Optional[float]
    # MC envelope
    mc: dict
    mc_return_percentile: Optional[float]
    mc_dd_percentile: Optional[float]
    # Per-stat verdict (kill rule input #1 + #2 visualized separately)
    return_verdict: str
    return_verdict_color: str
    dd_verdict: str
    dd_verdict_color: str
    # Combined kill rule
    combined_verdict: str
    combined_color: str
    # KS / MW edge diagnosis
    ks_p: Optional[float]
    mw_p: Optional[float]
    edge_diagnosis: str
    edge_diagnosis_color: str
    # Rolling charts
    rolling_sharpe: pd.Series
    rolling_calmar: pd.Series
    # Raw segments (for downstream chart construction)
    bt_pnl: pd.Series
    live_pnl: pd.Series
    split_date: pd.Timestamp

    # --- Dict-style accessors (backwards compat) ----------------------------

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError as e:
            raise KeyError(key) from e

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self):
        return asdict(self).keys()

    def to_dict(self) -> dict:
        return asdict(self)


def segment_metrics(daily_pnl: pd.Series, starting_capital: float,
                    rfr: float = 0.04) -> dict:
    """Comprehensive metrics for one daily-P&L segment (backtest OR live).

    Returns dict with: n_days, n_active_days, total_pnl, final_equity,
    sharpe, sortino, cagr, mdd, win_rate (%), pf, avg_win, avg_loss,
    best_day, worst_day, first_trade, last_trade, days_since_last_trade,
    trades_per_year (extrapolated from active days)."""
    empty = {
        'n_days': 0, 'n_active_days': 0, 'total_pnl': 0.0,
        'final_equity': float(starting_capital),
        'sharpe': 0.0, 'sortino': 0.0, 'cagr': 0.0, 'mdd': 0.0,
        'win_rate': 0.0, 'pf': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0,
        'best_day': 0.0, 'worst_day': 0.0,
        'first_trade': None, 'last_trade': None,
        'days_since_last_trade': None, 'trades_per_year': 0.0,
    }
    if daily_pnl is None or daily_pnl.empty:
        return empty
    pnl = daily_pnl.fillna(0)
    equity = pnl.cumsum() + starting_capital
    n_days = len(pnl)
    rets = equity.pct_change().fillna(0)
    sharpe, sortino = get_risk_ratios(rets, rfr)
    days_span = max((equity.index[-1] - equity.index[0]).days, 1)
    cagr = get_cagr(starting_capital, float(equity.iloc[-1]), days_span)
    mdd, _ = get_max_drawdown(equity, starting_capital)
    nonzero = pnl[pnl != 0]
    n_active = int(len(nonzero))
    pf = profit_factor(nonzero.values) if n_active else 0.0
    if not np.isfinite(pf):
        pf = 999.0
    win_rate = float((nonzero > 0).mean() * 100) if n_active else 0.0
    avg_win = float(nonzero[nonzero > 0].mean()) if (nonzero > 0).any() else 0.0
    avg_loss = float(nonzero[nonzero < 0].mean()) if (nonzero < 0).any() else 0.0
    first_trade = nonzero.index[0] if n_active else None
    last_trade = nonzero.index[-1] if n_active else None
    days_since = (pnl.index[-1] - last_trade).days if last_trade is not None else None
    tpy = (n_active / max(days_span, 1)) * 365.25
    return {
        'n_days': n_days, 'n_active_days': n_active,
        'total_pnl': float(pnl.sum()),
        'final_equity': float(equity.iloc[-1]),
        'sharpe': float(sharpe), 'sortino': float(sortino),
        'cagr': float(cagr), 'mdd': float(mdd),
        'win_rate': win_rate, 'pf': float(pf),
        'avg_win': avg_win, 'avg_loss': avg_loss,
        'best_day': float(pnl.max()), 'worst_day': float(pnl.min()),
        'first_trade': first_trade, 'last_trade': last_trade,
        'days_since_last_trade': days_since,
        'trades_per_year': float(tpy),
    }


def distribution_drift_test(bt_returns: pd.Series, live_returns: pd.Series,
                             alpha: float = 0.05) -> dict:
    """Kolmogorov-Smirnov + Mann-Whitney U tests for distribution shift
    between backtest and live segments. Uses non-zero P&L observations only
    (zeros from no-trade days would dilute the signal).

    Returns dict with ks_stat, ks_pvalue, mw_pvalue, verdict, n_bt, n_live.
    Verdicts: 'distribution_shifted' (KS p<0.01), 'weak_drift' (p<0.05),
              'consistent' (p>=0.05), 'insufficient_data' (n_live<5)."""
    bt = pd.Series(bt_returns).fillna(0)
    lv = pd.Series(live_returns).fillna(0)
    bt = bt[bt != 0]
    lv = lv[lv != 0]
    if len(bt) < 10 or len(lv) < 5:
        return {
            'ks_stat': None, 'ks_pvalue': None, 'mw_pvalue': None,
            'verdict': 'insufficient_data',
            'n_bt': int(len(bt)), 'n_live': int(len(lv)),
        }
    try:
        ks_stat, ks_p = stats.ks_2samp(bt.values, lv.values)
    except Exception:
        ks_stat, ks_p = float('nan'), 1.0
    try:
        _, mw_p = stats.mannwhitneyu(bt.values, lv.values, alternative='two-sided')
    except Exception:
        mw_p = 1.0
    if ks_p < 0.01:
        verdict = 'distribution_shifted'
    elif ks_p < alpha:
        verdict = 'weak_drift'
    else:
        verdict = 'consistent'
    return {
        'ks_stat': float(ks_stat) if ks_stat is not None else None,
        'ks_pvalue': float(ks_p),
        'mw_pvalue': float(mw_p),
        'verdict': verdict,
        'n_bt': int(len(bt)), 'n_live': int(len(lv)),
    }


def strategy_health_status(bt_m: dict, live_m: dict,
                           drift: dict) -> Tuple[str, str, str]:
    """Rule-based health classification for a single strategy's live performance.

    Returns (status_label, color_hex, action_hint). Priority order:
      1. Insufficient live data  → ⏳  (need ≥10 active days)
      2. Inactive               → 💤  (no trades in 2× expected gap)
      3. Broken                 → 🔴  (live PF < 0.5× backtest PF OR live total P&L < 0 with backtest > 0)
      4. Watch                  → 🟡  (PF 0.5–0.85× backtest OR significant distribution drift)
      5. In-spec                → 🟢
    """
    if live_m['n_active_days'] == 0:
        return ('💤 No live trades', '#7f8c8d', 'Check whether the strategy is firing in deployment.')
    if live_m['n_active_days'] < 10:
        return ('⏳ Insufficient live data',
                '#95a5a6',
                f"Only {live_m['n_active_days']} active live days — wait for ≥10 trades to judge.")

    # Inactivity check: expected gap = ~365 / TPY days, alert if 2× exceeded
    bt_tpy = max(bt_m['trades_per_year'], 1.0)
    expected_gap_days = 365.0 / bt_tpy
    days_since = live_m.get('days_since_last_trade') or 0
    if days_since > expected_gap_days * 2.5:
        return ('💤 Inactive',
                '#e67e22',
                f"Last trade was {days_since} days ago (backtest expected ~{expected_gap_days:.0f}).")

    bt_pf = bt_m['pf'] if bt_m['pf'] > 0 else 0.001
    pf_ratio = live_m['pf'] / bt_pf
    bt_pnl = bt_m['total_pnl']
    live_pnl = live_m['total_pnl']

    # Broken: huge PF collapse OR sign flip on P&L
    if pf_ratio < 0.5 or (bt_pnl > 0 and live_pnl < -abs(bt_pnl) * 0.1):
        return ('🔴 Broken',
                '#c0392b',
                f"Live PF {live_m['pf']:.2f} vs backtest {bt_m['pf']:.2f} ({pf_ratio:.0%}). Consider pausing.")

    # Watch: drift OR moderate PF decay
    drift_v = drift.get('verdict', 'consistent')
    if drift_v == 'distribution_shifted' or pf_ratio < 0.85:
        return ('🟡 Watch',
                '#f39c12',
                f"Live PF {live_m['pf']:.2f} vs backtest {bt_m['pf']:.2f} ({pf_ratio:.0%}); drift={drift_v}.")

    return ('🟢 In-spec',
            '#27ae60',
            f"Live PF {live_m['pf']:.2f} ≈ backtest {bt_m['pf']:.2f}; drift={drift_v}.")


def per_strategy_regime_metrics(daily_pnl: pd.Series, regime: pd.Series,
                                  starting_capital: float, rfr: float) -> dict:
    """Compute per-regime metrics (sharpe, pf, mean_daily) for ONE strategy's daily P&L.

    Returns dict keyed by regime → metrics. Regimes with <5 active days return
    None (insufficient data). Used to derive a regime-conditional expectation
    of how the strategy SHOULD perform under a given live regime mix.
    """
    aligned = regime.reindex(daily_pnl.index, method='ffill')
    out = {}
    for r in ['Bull', 'Bear', 'Chop']:
        sub = daily_pnl[aligned == r].fillna(0)
        nonzero = sub[sub != 0]
        if len(nonzero) < 5 or sub.std() == 0:
            out[r] = None
            continue
        sh = (sub.mean() / sub.std()) * np.sqrt(365)
        pf = profit_factor(nonzero.values)
        if not np.isfinite(pf):
            pf = 999.0
        out[r] = {
            'n_days': int(len(sub)),
            'n_active_days': int(len(nonzero)),
            'sharpe': float(sh),
            'pf': float(pf),
            'mean_daily': float(sub.mean()),
            'win_rate': float((nonzero > 0).mean() * 100),
        }
    return out


def regime_conditional_expected(bt_regime_metrics: dict,
                                 live_regime_mix: dict,
                                 fallback_pf: float = 1.0,
                                 fallback_sharpe: float = 0.0) -> dict:
    """Weight backtest per-regime stats by the live period's regime composition
    to compute the EXPECTED live metrics under no decay.

    bt_regime_metrics: {'Bull': {...}, 'Bear': {...}, 'Chop': {...}} (some may be None)
    live_regime_mix:   {'Bull': 0.26, 'Bear': 0.32, 'Chop': 0.42} (must sum to ~1)
    fallback_pf/sharpe: used for regimes with insufficient bt data (so we don't
        falsely assume strategy will perform like fallback in unseen regime).

    Returns {'expected_pf': float, 'expected_sharpe': float, 'covered_share': float}
    where covered_share is the % of live days for which we had bt regime data.
    """
    weighted_pf = 0.0
    weighted_sh = 0.0
    covered = 0.0
    for r, share in live_regime_mix.items():
        m = bt_regime_metrics.get(r)
        if m is None or share == 0:
            # Fall back to neutral assumption for unseen regimes
            weighted_pf += share * fallback_pf
            weighted_sh += share * fallback_sharpe
        else:
            weighted_pf += share * m['pf']
            weighted_sh += share * m['sharpe']
            covered += share
    return {
        'expected_pf': float(weighted_pf),
        'expected_sharpe': float(weighted_sh),
        'covered_share': float(covered),
    }


def sharpe_standard_error(sharpe_annualized: float, n_obs: int,
                           periods_per_year: int = 365) -> float:
    """Lo (2002) closed-form SE of an ANNUALISED Sharpe ratio.

    The base Lo formula `Var(SR) = (1 + 0.5·SR²)/T` is in per-period units.
    For an annualised Sharpe (SR_ann = SR_period · √k):
       Var(SR_ann) = k · Var(SR_period) = k · (1 + 0.5·SR_period²)/T
                   = k/T + 0.5·SR_ann²/T
       SE(SR_ann)  = √(k/T + 0.5·SR_ann²/T)

    Where T = number of observations Sharpe was computed on, k = periods/year.

    Common pitfall: feeding annualized SR into the per-period formula
    underestimates SE by ~√k (≈19× for daily). Always use this annualized form.

    Caveat: assumes IID returns; serial correlation widens the true SE.
    """
    if n_obs < 2:
        return float('inf')
    k = float(periods_per_year)
    return float(np.sqrt(k / n_obs + 0.5 * sharpe_annualized ** 2 / n_obs))


def regime_conditional_health_status(live_m: dict, expected: dict,
                                       bt_m: dict, drift: dict,
                                       min_live_trades: int = 10,
                                       direction: str = 'BOTH',
                                       ticker_live_bear_tilt: Optional[float] = None,
                                       ticker_live_price_change: Optional[float] = None) -> Tuple[str, str, str]:
    """Regime-conditional + sample-size-aware health classifier.

    Compares live metrics to the BACKTEST EXPECTATION UNDER LIVE'S REGIME MIX,
    not the regime-averaged backtest. Also gates on Sharpe standard error so
    we don't classify noisy small-sample strategies as "broken".

    Returns (status_label, color, hint).
    """
    n_live = live_m['n_active_days']
    if n_live == 0:
        return ('💤 No live trades', '#7f8c8d',
                'Check whether the strategy is firing in deployment.')
    if n_live < min_live_trades:
        return ('⏳ Insufficient',
                '#95a5a6',
                f"Only {n_live} active live days — need ≥{min_live_trades} for a verdict.")

    # Inactivity gate
    bt_tpy = max(bt_m['trades_per_year'], 1.0)
    expected_gap_days = 365.0 / bt_tpy
    days_since = live_m.get('days_since_last_trade') or 0
    if days_since > expected_gap_days * 2.5:
        return ('💤 Inactive',
                '#e67e22',
                f"Last trade {days_since}d ago (BT expected gap ~{expected_gap_days:.0f}d).")

    expected_pf = expected['expected_pf']
    expected_sh = expected['expected_sharpe']
    live_pf = live_m['pf']
    live_sh = live_m['sharpe']

    # Sample-size gate: if live & expected Sharpe ranges overlap, we cannot
    # statistically distinguish decay from noise. BUT we still subdivide by
    # secondary signals (PF ratio, P&L sign) so the user gets an actionable
    # hint even when Sharpe is statistically inconclusive.
    live_se = sharpe_standard_error(live_sh, n_live)
    overlap = (live_sh + 2 * live_se) >= expected_sh
    pf_ratio = live_pf / expected_pf if expected_pf > 0 else 0.0
    drift_v = drift.get('verdict', 'consistent')
    if overlap:
        # Severe non-statistical red flag: PF collapsed AND P&L deeply negative
        # — call this Broken even with wide CI, because the trade-level signal is loud.
        if (pf_ratio < 0.4 and live_m['total_pnl'] < 0 and
                bt_m['total_pnl'] > 0):
            return ('🔴 Broken (small sample)',
                    '#c0392b',
                    f"PF ratio {pf_ratio:.0%} + live P&L flipped negative despite small sample "
                    f"({n_live} trade days). Sharpe is wide ({live_sh:+.2f}±{2*live_se:.2f}) but the "
                    f"trade-level signal is unambiguous.")
        # Mild concern: PF down, P&L negative — within noise but worth watching
        if pf_ratio < 0.7 or live_m['total_pnl'] < 0:
            return ('🟡 Within noise — watch P&L',
                    '#f39c12',
                    f"Sharpe statistically inconclusive ({live_sh:+.2f}±{2*live_se:.2f} overlaps "
                    f"expected {expected_sh:+.2f}). But PF ratio {pf_ratio:.0%} & P&L ${live_m['total_pnl']:+.0f} "
                    f"are mildly concerning — gather more live trade days before judging.")
        # Looks fine even though sample is small
        return ('🟢 Within noise',
                '#27ae60',
                f"Live Sharpe {live_sh:+.2f}±{2*live_se:.2f} overlaps regime-expected {expected_sh:+.2f}. "
                f"PF ratio {pf_ratio:.0%}. No evidence of decay; sample size is the limit.")

    if pf_ratio < 0.5 or (expected_pf > 1.0 and live_pf < 0.7):
        # Directional override: LONG-only in bearish ticker (or SHORT in bullish) = headwind, not decay
        tilt_bearish = (ticker_live_bear_tilt is not None and ticker_live_bear_tilt >= 0.15)
        price_bearish = (ticker_live_price_change is not None and ticker_live_price_change <= -0.10)
        tilt_bullish = (ticker_live_bear_tilt is not None and ticker_live_bear_tilt <= -0.15)
        price_bullish = (ticker_live_price_change is not None and ticker_live_price_change >= 0.10)
        if direction == 'LONG' and (tilt_bearish or price_bearish):
            pc_s = f", price {ticker_live_price_change*100:+.1f}% live" if ticker_live_price_change is not None else ""
            return ('📉 Directional headwind',
                    '#3498db',
                    f"LONG-only strategy on bearish ticker (bear tilt {ticker_live_bear_tilt:+.2f}{pc_s}). "
                    f"Live PF {live_pf:.2f} vs expected {expected_pf:.2f} ({pf_ratio:.0%}). "
                    f"Underperformance is regime-explained, NOT decay. Wait for regime change.")
        if direction == 'SHORT' and (tilt_bullish or price_bullish):
            pc_s = f", price {ticker_live_price_change*100:+.1f}% live" if ticker_live_price_change is not None else ""
            return ('📈 Directional headwind',
                    '#3498db',
                    f"SHORT-only strategy on bullish ticker (bear tilt {ticker_live_bear_tilt:+.2f}{pc_s}). "
                    f"Live PF {live_pf:.2f} vs expected {expected_pf:.2f} ({pf_ratio:.0%}). "
                    f"Underperformance is regime-explained, NOT decay.")
        return ('🔴 Broken',
                '#c0392b',
                f"Live PF {live_pf:.2f} vs regime-expected {expected_pf:.2f} ({pf_ratio:.0%}); "
                f"Sharpe {live_sh:+.2f} vs {expected_sh:+.2f}. Real decay, not just regime.")

    if pf_ratio < 0.85 or drift_v == 'distribution_shifted':
        return ('🟡 Watch',
                '#f39c12',
                f"Live PF {live_pf:.2f} vs regime-expected {expected_pf:.2f} ({pf_ratio:.0%}); "
                f"drift={drift_v}. Monitor.")

    return ('🟢 In-spec',
            '#27ae60',
            f"Live PF {live_pf:.2f} matches regime-expected {expected_pf:.2f} ({pf_ratio:.0%}).")


def per_strategy_live_table(plot_data: pd.DataFrame, metrics_df: pd.DataFrame,
                            total_cap: float, rfr: float,
                            live_start: str,
                            regime: Optional[pd.Series] = None,
                            ticker_regimes: Optional[dict] = None) -> pd.DataFrame:
    """Build a per-strategy backtest-vs-live comparison table.

    If `regime` is provided, classification uses regime-conditional expectations
    (live PF vs what backtest predicts under live's regime mix). Otherwise
    falls back to the simple regime-blind classifier.
    """
    ignore = {'Portfolio Equity', 'Portfolio DD', 'Portfolio Daily P&L',
              'B&H BTC Equity', 'Portfolio Load'}
    strategy_cols = [c for c in plot_data.columns if c not in ignore]
    if not strategy_cols:
        return pd.DataFrame()
    split = pd.Timestamp(live_start)
    per_strat_cap = total_cap / max(len(strategy_cols), 1)

    # Live regime mix (used by regime-conditional classifier)
    live_regime_mix = {'Bull': 1/3, 'Bear': 1/3, 'Chop': 1/3}
    if regime is not None and not regime.empty:
        live_regime = regime[regime.index >= split]
        if len(live_regime) > 0:
            live_regime_mix = {
                'Bull': float((live_regime == 'Bull').mean()),
                'Bear': float((live_regime == 'Bear').mean()),
                'Chop': float((live_regime == 'Chop').mean()),
            }

    rows = []
    for col in strategy_cols:
        col_pnl = plot_data[col].fillna(0)
        bt_pnl = col_pnl[col_pnl.index < split]
        live_pnl = col_pnl[col_pnl.index >= split]
        bt_m = segment_metrics(bt_pnl, per_strat_cap, rfr)
        live_m = segment_metrics(live_pnl, per_strat_cap, rfr)
        drift = distribution_drift_test(bt_pnl, live_pnl)

        # Regime-conditional expectation (or zero if no regime data)
        if regime is not None and not regime.empty:
            bt_regime_m = per_strategy_regime_metrics(bt_pnl, regime, per_strat_cap, rfr)
            expected = regime_conditional_expected(
                bt_regime_m, live_regime_mix,
                fallback_pf=max(bt_m['pf'], 0.5),
                fallback_sharpe=bt_m['sharpe'],
            )
            # Per-ticker bear tilt + price change (for directional override)
            tk = extract_ticker(col)
            direction = extract_direction(col)
            tk_bear_tilt = None
            tk_price_chg = None
            if ticker_regimes and tk in ticker_regimes:
                tk_regime = ticker_regimes[tk]
                tk_live_regime = tk_regime[tk_regime.index >= split]
                if len(tk_live_regime) > 0:
                    tk_bear_tilt = float(
                        (tk_live_regime == 'Bear').mean() -
                        (tk_live_regime == 'Bull').mean()
                    )
                    tk_price_chg = _ticker_live_price_change(tk, split)
            status, color, hint = regime_conditional_health_status(
                live_m, expected, bt_m, drift,
                direction=direction,
                ticker_live_bear_tilt=tk_bear_tilt,
                ticker_live_price_change=tk_price_chg,
            )
        else:
            expected = {'expected_pf': bt_m['pf'], 'expected_sharpe': bt_m['sharpe']}
            status, color, hint = strategy_health_status(bt_m, live_m, drift)

        pf_ratio = (live_m['pf'] / expected['expected_pf']) if expected['expected_pf'] > 0 else 0.0
        sharpe_delta = live_m['sharpe'] - expected['expected_sharpe']
        live_se = sharpe_standard_error(live_m['sharpe'], live_m['n_active_days'])
        # Efficiency metrics (use TOTAL calendar days for a fair daily-rate comparison)
        ret_eff = _return_efficiency(
            live_m['total_pnl'], live_m['n_days'],
            bt_m['total_pnl'], bt_m['n_days'],
        )
        dd_budget = _drawdown_budget_used(live_m['mdd'], bt_m['mdd'])

        rows.append({
            'Strategy': col,
            'Family': extract_family(col),
            'Ticker': extract_ticker(col),
            'Status': status,
            '_color': color,
            'Hint': hint,
            'Live Trade Days': live_m['n_active_days'],
            'Days Since Last Trade': live_m['days_since_last_trade'] or 0,
            'Live Sharpe': live_m['sharpe'],
            'Live Sharpe SE': live_se,
            'Expected Sharpe (regime-adj)': expected['expected_sharpe'],
            'ΔSharpe vs expected': sharpe_delta,
            'Live PF': live_m['pf'],
            'Expected PF (regime-adj)': expected['expected_pf'],
            'PF Ratio %': pf_ratio * 100,
            'Return Eff %': ret_eff if ret_eff is not None else 0.0,
            'DD Budget %': dd_budget if dd_budget is not None else 0.0,
            'BT Sharpe (regime-avg)': bt_m['sharpe'],
            'BT PF (regime-avg)': bt_m['pf'],
            'Live Win %': live_m['win_rate'],
            'BT Win %': bt_m['win_rate'],
            'Live P&L $': live_m['total_pnl'],
            'BT P&L $': bt_m['total_pnl'],
            'Live MDD %': live_m['mdd'] * 100,
            'BT MDD %': bt_m['mdd'] * 100,
            'KS p-value': drift.get('ks_pvalue'),
            'Drift': drift.get('verdict'),
        })
    return pd.DataFrame(rows)


def strategy_family_table(strategy_table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-strategy live table into family-level decisions.

    A "family" is one parameter set deployed across multiple tickers. The
    family-level decision is sharper than per-ticker because it leverages the
    cross-ticker robustness design: if a parameter set works in BOTH BTC and
    ETH live, that's stronger evidence than either alone. Conversely, if it
    breaks in ONE ticker but holds in the other, the issue is likely ticker
    microstructure, not the strategy itself.

    Decision rules (priority order, first match wins):
      • Any leg INSUFFICIENT → ⏳ INSUFFICIENT (need more live trades)
      • Single leg (n=1) → ⚠️ UNVALIDATED (no cross-ticker confirmation)
      • All legs broken → 🔴 KILL (real strategy decay, not ticker-specific)
      • All legs healthy → 🟢 KEEP
      • Mixed → 🟡 INVESTIGATE / 📉 REGIME HOLD / PARTIAL HEADWIND
        (depends on direction tag + per-ticker regime)

    Args:
        strategy_table: per-strategy DataFrame from per_strategy_live_table.
                        Must contain Family, Ticker, Verdict, Direction columns.

    Returns:
        DataFrame indexed by Family with aggregated columns:
          • Tickers: comma-joined ticker list
          • N: leg count
          • Verdict: family-level decision (emoji + label)
          • Worst Leg / Best Leg: extremes for quick review
          • Pair Notes: direction/regime context when relevant
    """
    if strategy_table is None or strategy_table.empty:
        return pd.DataFrame()
    rows = []
    for family, group in strategy_table.groupby('Family'):
        tickers = sorted(group['Ticker'].tolist())
        n_inst = len(group)
        statuses = group['Status'].tolist()
        n_healthy = sum(1 for s in statuses if s.startswith('🟢'))
        n_watch = sum(1 for s in statuses if s.startswith('🟡'))
        n_broken = sum(1 for s in statuses if s.startswith('🔴'))
        n_inactive = sum(1 for s in statuses if s.startswith('💤'))
        n_insufficient = sum(1 for s in statuses if s.startswith('⏳'))
        n_headwind = sum(1 for s in statuses if s.startswith('📉') or s.startswith('📈'))

        # Family-level decision (priority order: structural > behavioral > insufficient)
        if n_inst == 1:
            # Single-ticker family = PILOT phase. User's workflow: validate on
            # one ticker, then promote to a 2nd ticker if it works.
            single_status = statuses[0]
            single_live_days = int(group['Live Trade Days'].iloc[0])
            single_live_pnl = float(group['Live P&L $'].iloc[0])
            single_pf_ratio = float(group['PF Ratio %'].iloc[0])

            if single_status.startswith('⏳'):
                decision = '📋 PILOT — INSUFFICIENT'
                color = '#95a5a6'
                hint = (f"Pilot has only {single_live_days} live trades — "
                        "need ≥10–30 for a promote/terminate decision.")
            elif single_status.startswith('💤'):
                decision = '💤 PILOT INACTIVE'
                color = '#e67e22'
                hint = ("Pilot hasn't traded recently — check deployment before "
                        "scaling decision.")
            elif single_status.startswith('📉') or single_status.startswith('📈'):
                decision = '📉 PILOT IN HEADWIND'
                color = '#3498db'
                hint = ("Pilot facing directional headwind (long-only in bear / "
                        "short-only in bull). Wait for regime change before "
                        "promote/terminate decision — current loss is regime-explained, "
                        "not strategy failure.")
            elif single_status.startswith('🔴'):
                decision = '🔴 TERMINATE PILOT'
                color = '#c0392b'
                hint = (f"Pilot failing live (PF ratio {single_pf_ratio:.0f}%). "
                        "Do NOT scale to additional tickers — kill or rebuild before "
                        "wasting capital on a 2nd instance.")
            elif single_status.startswith('🟢'):
                if single_live_days >= 30:
                    decision = '🟢 PROMOTE — scale to 2nd ticker'
                    color = '#27ae60'
                    hint = (f"Pilot is in-spec with sufficient sample ({single_live_days} "
                            "live trades). Ready to add a 2nd uncorrelated ticker — "
                            "the cross-asset run validates parameters are not curve-fit.")
                else:
                    decision = '🟢 PILOT ON TRACK'
                    color = '#16a085'
                    hint = (f"Pilot is in-spec ({single_live_days} live trades). "
                            "Continue to 30+ trades before promoting to a 2nd ticker.")
            elif single_status.startswith('🟡'):
                decision = '🟡 CONTINUE PILOT'
                color = '#f39c12'
                hint = (f"Mixed signal ({single_status}, PF ratio {single_pf_ratio:.0f}%). "
                        "Continue pilot and gather more data before promote/terminate.")
            else:
                decision = '📋 PILOT'
                color = '#bdc3c7'
                hint = "Single-ticker pilot phase. Evaluate before scaling."
        elif n_insufficient >= n_inst / 2:
            decision = '⏳ INSUFFICIENT'
            color = '#95a5a6'
            hint = "Most instances lack live trade volume — wait for more data."
        elif n_broken == n_inst:
            decision = '🔴 KILL'
            color = '#c0392b'
            hint = ("ALL tickers broken under regime-adjusted expectation — real strategy decay. "
                    "Cross-asset failure is the strongest signal that the parameter set is dead.")
        elif n_healthy == n_inst:
            decision = '🟢 KEEP'
            color = '#27ae60'
            hint = "All tickers performing within regime-adjusted expectation."
        elif n_headwind == n_inst:
            # All instances flagged as directional headwind — strategy is on hold
            # until regime changes, NOT broken. Critical to not kill these.
            decision = '📉 REGIME HOLD'
            color = '#3498db'
            hint = ("ALL tickers are facing directional headwind (e.g. LONG-only strategy in "
                    "bear market). Underperformance is regime-explained, NOT decay. "
                    "Reduce size or pause IF you want to skip the regime, but DO NOT kill "
                    "permanently — wait for the next regime cycle to fairly evaluate.")
        elif n_headwind > 0 and n_broken == 0 and n_healthy > 0:
            decision = '🟡 PARTIAL HEADWIND'
            color = '#3498db'
            hint = (f"{n_headwind}/{n_inst} tickers face directional headwind, {n_healthy}/{n_inst} healthy. "
                    "The healthy ticker validates the strategy; the headwind ticker is regime-blocked.")
        elif n_broken > 0 and n_healthy > 0:
            decision = '🟡 INVESTIGATE'
            color = '#f39c12'
            hint = ("Asymmetric performance: some tickers fine, others broken. "
                    "Likely ticker-specific issue (microstructure, MM competition, liquidity) "
                    "rather than strategy decay. The healthy ticker validates the parameter set.")
        elif n_broken > 0:
            decision = '🟡 PARTIAL CONCERN'
            color = '#f39c12'
            hint = f"{n_broken}/{n_inst} broken — investigate ticker-specific factors."
        else:
            decision = '🟡 WATCH'
            color = '#f39c12'
            hint = "Mixed signals — monitor."

        # Aggregate metrics across instances (capital-weighted)
        live_pnl_sum = float(group['Live P&L $'].sum())
        live_sh_mean = float(group['Live Sharpe'].mean())
        expected_sh_mean = float(group['Expected Sharpe (regime-adj)'].mean())
        live_pf_mean = float(group['Live PF'].mean())
        expected_pf_mean = float(group['Expected PF (regime-adj)'].mean())
        live_days_total = int(group['Live Trade Days'].sum())
        worst_drift = group['KS p-value'].min(skipna=True)
        avg_ret_eff = float(group['Return Eff %'].mean()) if 'Return Eff %' in group.columns else 0.0
        worst_dd_budget = float(group['DD Budget %'].max()) if 'DD Budget %' in group.columns else 0.0

        # Get the strategy direction from any instance (all share same direction)
        family_direction = extract_direction(group['Strategy'].iloc[0]) if 'Strategy' in group.columns else 'BOTH'

        rows.append({
            'Family': family,
            'Direction': family_direction,
            'Decision': decision,
            '_color': color,
            'Hint': hint,
            'Tickers': ', '.join(tickers),
            '# Instances': n_inst,
            '🟢': n_healthy, '🟡': n_watch, '🔴': n_broken,
            '📉': n_headwind, '💤': n_inactive, '⏳': n_insufficient,
            'Σ Live P&L $': live_pnl_sum,
            'Avg Live Sharpe': live_sh_mean,
            'Avg Expected Sharpe': expected_sh_mean,
            'Avg Live PF': live_pf_mean,
            'Avg Expected PF': expected_pf_mean,
            'Avg Return Eff %': avg_ret_eff,
            'Worst DD Budget %': worst_dd_budget,
            'Σ Live Trades': live_days_total,
            'Worst KS p': worst_drift if pd.notna(worst_drift) else None,
        })
    # Sort by urgency / actionability:
    #   KILL & TERMINATE first (stop the bleeding)
    #   INVESTIGATE / CONCERN next (decide ticker-specific vs decay)
    #   PROMOTE next (good news, actionable upward — scale to 2nd ticker)
    #   WATCH / CONTINUE PILOT (gather more data)
    #   REGIME HOLD / HEADWIND / INACTIVE / INSUFFICIENT (passive wait)
    #   KEEP / PILOT ON TRACK last (no action needed today)
    decision_order = {
        '🔴 KILL': 0,
        '🔴 TERMINATE PILOT': 1,
        '🟡 INVESTIGATE': 2,
        '🟡 PARTIAL CONCERN': 3,
        '🟡 PARTIAL HEADWIND': 4,
        '🟢 PROMOTE — scale to 2nd ticker': 5,
        '🟡 WATCH': 6,
        '🟡 CONTINUE PILOT': 7,
        '📉 REGIME HOLD': 8,
        '📉 PILOT IN HEADWIND': 9,
        '💤 PILOT INACTIVE': 10,
        '⏳ INSUFFICIENT': 11,
        '📋 PILOT — INSUFFICIENT': 12,
        '📋 PILOT': 13,
        '🟢 KEEP': 14,
        '🟢 PILOT ON TRACK': 15,
    }
    df = pd.DataFrame(rows)
    if not df.empty:
        df['_order'] = df['Decision'].map(decision_order).fillna(99)
        df = df.sort_values(['_order', 'Σ Live P&L $']).drop(columns=['_order'])
    return df


def _return_efficiency(live_pnl: float, live_days: int,
                       bt_pnl: float, bt_days: int) -> Optional[float]:
    """Live daily-P&L rate as % of backtest daily-P&L rate.
    100% = live matches backtest pace · <100% = underperforming · >100% = outperforming.

    Uses TOTAL calendar days (not active days) for a fair rate-of-return comparison.
    """
    if bt_days <= 0 or live_days <= 0 or bt_pnl == 0:
        return None
    bt_rate = bt_pnl / bt_days
    if bt_rate == 0:
        return None
    live_rate = live_pnl / live_days
    return float(live_rate / bt_rate * 100)


def _drawdown_budget_used(live_mdd: float, bt_mdd: float) -> Optional[float]:
    """Live max-drawdown as % of backtest max-drawdown.
    <50% = healthy DD usage · 50-100% = within historical worst-case ·
    >100% = exceeded historical worst (concerning) · >150% = severe (likely broken).
    """
    if bt_mdd == 0:
        return None
    return float(abs(live_mdd) / abs(bt_mdd) * 100)


def bootstrap_equity_envelope(bt_daily_pnl: pd.Series, n_live_days: int,
                                starting_equity: float = 0.0,
                                n_sims: int = 1000, seed: Optional[int] = 42,
                                block_len: int = 5) -> pd.DataFrame:
    """Block-bootstrap simulated equity paths for `n_live_days` using BT daily P&L.

    Generates `n_sims` paths by sampling blocks of length `block_len` from the
    backtest daily-P&L distribution (preserves short-term autocorrelation).
    Returns a DataFrame indexed by day-offset (0..n_live_days) with columns
    P5, P25, P50, P75, P95 — the percentile bands of the simulated outcomes.

    This is the proper hypothesis-test framing for live monitoring: under H0
    'live behaves like backtest', P5–P95 is the ±2σ envelope. Live curves
    drifting outside the envelope are statistical evidence of regime change.
    """
    if bt_daily_pnl is None or len(bt_daily_pnl) == 0 or n_live_days < 1:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    bt_array = pd.Series(bt_daily_pnl).fillna(0).values
    n_sims = int(n_sims)
    paths = np.zeros((n_sims, n_live_days + 1))
    paths[:, 0] = starting_equity
    for i in range(n_sims):
        sample = block_bootstrap_values(bt_array, n_live_days, B=block_len, rng=rng)
        paths[i, 1:] = starting_equity + np.cumsum(sample)
    bands = {f'P{p}': np.percentile(paths, p, axis=0) for p in [5, 25, 50, 75, 95]}
    return pd.DataFrame(bands)


def live_within_envelope(live_equity: pd.Series,
                          envelope: pd.DataFrame,
                          low_pct: int = 5, high_pct: int = 95) -> dict:
    """Check what fraction of live days fall within the [P{low}, P{high}] envelope.

    Returns dict with: within_pct (%), days_below, days_above, total_days,
    compliance_label, compliance_color. Compliance buckets:
      ≥90% within → 🟢 Within envelope (statistically consistent w/ backtest)
      70-90% within, mostly above → 🟢 Above envelope (outperforming)
      70-90% within, mostly below → 🟡 Borderline below
      <70% within, mostly below → 🔴 Outside expected envelope (regime change)
      <70% within, mostly above → 🟢 Outperforming envelope
    """
    if envelope is None or envelope.empty or live_equity is None or len(live_equity) == 0:
        return {
            'within_pct': None, 'days_below': 0, 'days_above': 0, 'total_days': 0,
            'compliance_label': '⏳ n/a', 'compliance_color': '#95a5a6',
        }
    n = min(len(live_equity), len(envelope))
    low = envelope[f'P{low_pct}'].iloc[:n].values
    high = envelope[f'P{high_pct}'].iloc[:n].values
    live = pd.Series(live_equity).iloc[:n].values
    above = live > high
    below = live < low
    within = ~(above | below)
    within_pct = float(within.mean() * 100)
    days_below = int(below.sum())
    days_above = int(above.sum())

    if within_pct >= 90:
        label, color = '🟢 Within envelope', '#27ae60'
    elif days_above > days_below:
        label, color = '🟢 Outperforming envelope', '#16a085'
    elif within_pct >= 70:
        label, color = '🟡 Borderline below envelope', '#f39c12'
    else:
        label, color = '🔴 Outside expected envelope', '#c0392b'
    return {
        'within_pct': within_pct, 'days_below': days_below, 'days_above': days_above,
        'total_days': n, 'compliance_label': label, 'compliance_color': color,
    }


def _fisher_z_correlation_test(r1: float, n1: int, r2: float, n2: int) -> Optional[float]:
    """Fisher z-transformation test for difference between two Pearson correlations.

    Tests H0: ρ_bt == ρ_live (i.e. pair dynamics unchanged). Returns two-sided
    p-value. None if either sample is too small or a correlation is invalid.
    """
    if (pd.isna(r1) or pd.isna(r2) or n1 < 10 or n2 < 5 or
            abs(r1) >= 0.9999 or abs(r2) >= 0.9999):
        # Bound for numerical stability; if past bound, can't run the test
        if pd.notna(r1) and pd.notna(r2) and n1 >= 10 and n2 >= 5:
            r1 = float(np.clip(r1, -0.999, 0.999))
            r2 = float(np.clip(r2, -0.999, 0.999))
        else:
            return None
    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))
    se = np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    if se == 0:
        return None
    z = (z1 - z2) / se
    return float(2.0 * (1.0 - stats.norm.cdf(abs(z))))


def _pair_divergence_verdict(bt_corr: float, live_corr: float,
                              corr_pvalue: Optional[float],
                              live_pnls: dict, n_live_active: int,
                              direction: str = 'BOTH',
                              ticker_live_regime_mix: Optional[dict] = None,
                              min_live_active: int = 10) -> Tuple[str, str, str]:
    """Interpret pair divergence + per-ticker live outcomes into a verdict.

    Critical: considers strategy DIRECTION + each ticker's LIVE REGIME mix.
    A LONG-only strategy on tickers that crashed isn't decayed — it's regime-blocked.

    Returns (label, color, explanation). Verdict priority:
      0. Directional headwind (long-only in bear / short-only in bull)
      1. Synchronised decay (BOTH-direction strategies losing in lockstep) → KILL
      2. Pair decoupled (correlation collapsed, asymmetric outcomes) → ticker-specific
      3. Both healthy, pair stable → strategy validated
      4. Asymmetric within historical variance → sample size effect
    """
    if n_live_active < min_live_active:
        return ('⏳ Insufficient live data',
                '#95a5a6',
                f"Only {n_live_active} co-active live days — need ≥{min_live_active} "
                f"to evaluate pair dynamics reliably.")
    if pd.isna(bt_corr) or pd.isna(live_corr):
        return ('⏳ Insufficient correlation data',
                '#95a5a6',
                "Not enough overlapping active days to estimate pair correlation.")

    delta = live_corr - bt_corr
    significant = (corr_pvalue is not None and corr_pvalue < 0.05)
    n_pos = sum(1 for v in live_pnls.values() if v > 0)
    n_neg = sum(1 for v in live_pnls.values() if v < 0)
    asymmetric = (n_pos > 0 and n_neg > 0)
    all_negative = (n_neg == len(live_pnls) and len(live_pnls) > 0)
    all_positive = (n_pos == len(live_pnls) and len(live_pnls) > 0)

    # --- 0. Directional headwind detection ------------------------------------
    # LONG-only on tickers with bearish tilt or down-trending price = expected loss.
    # SHORT-only on bullish tilt = expected loss. We use TWO signals:
    #   (a) Net regime tilt: bear_share - bull_share (catches slow bleeds where
    #       most days classify as chop but the asymmetric tilt is bearish)
    #   (b) Actual cumulative price change during live (the most direct evidence)
    if ticker_live_regime_mix and direction in ('LONG', 'SHORT') and all_negative:
        # ticker_live_regime_mix may contain 'price_change_pct' (from caller)
        bear_tilt = {  # bear − bull share per ticker
            t: m.get('Bear', 0) - m.get('Bull', 0)
            for t, m in ticker_live_regime_mix.items()
        }
        price_changes = {
            t: m.get('price_change_pct')
            for t, m in ticker_live_regime_mix.items()
        }
        # Strong directional headwind: all tickers either bearish-tilted OR price down
        # Use net tilt ≥ 15% OR price down ≥ 10% (whichever is more informative)
        def is_bearish(t):
            tilt = bear_tilt.get(t, 0)
            pc = price_changes.get(t)
            tilt_bearish = tilt >= 0.15
            price_bearish = (pc is not None and pc <= -0.10)
            return tilt_bearish or price_bearish

        def is_bullish(t):
            tilt = bear_tilt.get(t, 0)
            pc = price_changes.get(t)
            tilt_bullish = tilt <= -0.15
            price_bullish = (pc is not None and pc >= 0.10)
            return tilt_bullish or price_bullish

        regime_desc_parts = []
        for t, m in ticker_live_regime_mix.items():
            pc = m.get('price_change_pct')
            pc_s = f", price {pc*100:+.1f}%" if pc is not None else ""
            regime_desc_parts.append(
                f"{t}: {m.get('Bull', 0):.0%}🟢/{m.get('Bear', 0):.0%}🔴/{m.get('Chop', 0):.0%}🟡{pc_s}"
            )
        regime_desc = '; '.join(regime_desc_parts)

        if direction == 'LONG' and all(is_bearish(t) for t in ticker_live_regime_mix):
            return ('📉 Directional headwind (LONG-only in bear)',
                    '#3498db',
                    f"Strategy is LONG-only. All tickers have bearish tilt during live "
                    f"({regime_desc}). Losses are EXPECTED for long-only strategies in falling "
                    f"markets — NOT strategy decay. Pair correlation BT {bt_corr:+.2f} → "
                    f"Live {live_corr:+.2f}. **Do not kill on this evidence alone** — wait "
                    f"for regime change to fairly evaluate.")
        if direction == 'SHORT' and all(is_bullish(t) for t in ticker_live_regime_mix):
            return ('📈 Directional headwind (SHORT-only in bull)',
                    '#3498db',
                    f"Strategy is SHORT-only. All tickers have bullish tilt during live "
                    f"({regime_desc}). Losses are EXPECTED for short-only strategies in rising "
                    f"markets — NOT strategy decay. Pair correlation BT {bt_corr:+.2f} → "
                    f"Live {live_corr:+.2f}. **Do not kill on this evidence alone**.")

    # --- 1. Synchronised decay = strongest KILL signal (for BOTH-direction) ---
    if all_negative and (delta > 0.15 or (live_corr > 0.5 and bt_corr > 0.3)):
        dir_note = (" [BOTH-direction strategy — cannot blame regime]"
                    if direction == 'BOTH' else "")
        return ('🔴 Synchronised decay confirmed',
                '#c0392b',
                f"BT correlation {bt_corr:+.2f} → Live {live_corr:+.2f} (Δ {delta:+.2f}"
                f"{', sig' if significant else ''}).{dir_note} All tickers losing IN LOCKSTEP — "
                f"the textbook signature of parameter decay, not regime headwind. "
                f"Confirms KILL: the strategy is dead, not just out of regime.")

    # Pair decoupled with asymmetric outcome = ticker-specific
    if asymmetric and (delta < -0.2 or significant):
        good = [t for t, v in live_pnls.items() if v > 0]
        bad = [t for t, v in live_pnls.items() if v < 0]
        return ('🟡 Pair decoupled — ticker-specific issue',
                '#f39c12',
                f"BT correlation {bt_corr:+.2f} → Live {live_corr:+.2f} (Δ {delta:+.2f}"
                f"{f', sig p={corr_pvalue:.3f}' if significant else ''}). "
                f"Pair dynamics broke down AND outcomes diverged ({', '.join(good)} positive, "
                f"{', '.join(bad)} negative). The healthy ticker validates the parameter set; "
                f"investigate exchange/microstructure changes on the broken ticker "
                f"(liquidity, MM competition, listing changes, fee structure).")

    if all_negative:
        return ('🔴 Correlated breakdown',
                '#c0392b',
                f"BT correlation {bt_corr:+.2f} → Live {live_corr:+.2f}. Both tickers losing "
                f"with similar pair dynamics as backtest — systemic loss, not ticker issue. "
                f"Consistent with KILL.")

    if all_positive and abs(delta) < 0.15:
        return ('🟢 Pair stable, both healthy',
                '#27ae60',
                f"BT correlation {bt_corr:+.2f} → Live {live_corr:+.2f} (stable). Both tickers "
                f"performing — strategy fully validated across the asset pair.")

    if asymmetric and abs(delta) < 0.15:
        return ('🟡 Within historical pair variance',
                '#f39c12',
                f"BT correlation {bt_corr:+.2f} → Live {live_corr:+.2f} (stable). "
                f"One ticker outperforming the other is within the historical pair variance — "
                f"likely sample-size effect rather than a real decoupling. Wait for more data.")

    return ('🟡 Mixed signal',
            '#7f8c8d',
            f"BT corr {bt_corr:+.2f}, Live corr {live_corr:+.2f}. Mixed outcomes — needs more data.")


_TICKER_PRICE_CACHE: dict = {}  # {(ticker, start_iso, end_iso): pd.Series}


def fetch_ticker_prices(ticker: str, start: str, end: str) -> pd.Series:
    """Return the daily close price Series for a ticker (full history). Cached.

    Used by the per-strategy equity-vs-price chart so we can overlay the
    underlying ticker movement on each strategy's equity curve.
    """
    if not ticker or ticker == 'UNKNOWN':
        return pd.Series(dtype=float)
    try:
        start_iso = str(pd.Timestamp(start).date())
        end_iso = str(pd.Timestamp(end).date())
    except Exception:
        return pd.Series(dtype=float)
    cache_key = (ticker, start_iso, end_iso)
    if cache_key in _TICKER_PRICE_CACHE:
        return _TICKER_PRICE_CACHE[cache_key]
    try:
        df = fetch_btc_daily(start_iso, end_iso, symbol=ticker)
        if df.empty:
            _TICKER_PRICE_CACHE[cache_key] = pd.Series(dtype=float)
            return _TICKER_PRICE_CACHE[cache_key]
        prices = pd.Series(df['close'].values, index=df['time'], name=ticker)
        _TICKER_PRICE_CACHE[cache_key] = prices
        return prices
    except Exception:
        return pd.Series(dtype=float)


def strategy_monte_carlo(bt_daily_pnl: pd.Series, n_horizon_days: int,
                          n_runs: int = 1000, block_len: int = 5,
                          seed: Optional[int] = 42) -> dict:
    """Block-bootstrap N paths of `n_horizon_days` from backtest daily P&L.

    For each simulated path computes:
      - final cumulative P&L (= total return in $)
      - max drawdown along the path (in $, negative number)

    Returns dict with distributions and percentile bands suitable for both
    statistical hypothesis testing (where does live fall in the MC distribution?)
    and envelope plotting (P5/P25/P50/P75/P95 paths).
    """
    empty = {
        'paths': np.zeros((0, 0)),
        'final_pnls': np.array([]),
        'max_dds': np.array([]),
        'p5_path': np.array([]), 'p25_path': np.array([]),
        'p50_path': np.array([]), 'p75_path': np.array([]),
        'p95_path': np.array([]),
        'p5_final': 0.0, 'p25_final': 0.0, 'p50_final': 0.0,
        'p75_final': 0.0, 'p95_final': 0.0,
        'p5_dd': 0.0, 'p25_dd': 0.0, 'p50_dd': 0.0,
        'p75_dd': 0.0, 'p95_dd': 0.0,
        'n_runs': 0, 'n_horizon_days': int(n_horizon_days),
    }
    if bt_daily_pnl is None or len(bt_daily_pnl) == 0 or n_horizon_days < 1:
        return empty
    bt_array = pd.Series(bt_daily_pnl).fillna(0).values
    if len(bt_array) < 5 or float(np.std(bt_array)) == 0:
        return empty
    rng = np.random.default_rng(seed)
    n_runs = int(n_runs)
    paths = np.zeros((n_runs, n_horizon_days + 1))  # +1 for starting 0
    final_pnls = np.zeros(n_runs)
    max_dds = np.zeros(n_runs)
    for i in range(n_runs):
        sample = block_bootstrap_values(bt_array, n_horizon_days, B=block_len, rng=rng)
        cum = np.cumsum(sample)
        paths[i, 1:] = cum
        final_pnls[i] = float(cum[-1])
        # Floor peaks at 0: starting equity baseline is 0 in cum-PnL space.
        # Without this, a path that opens with a loss understates its true MDD
        # (peak would be set to first negative value instead of starting 0).
        peaks = np.maximum(np.maximum.accumulate(cum), 0)
        dd = cum - peaks  # ≤ 0
        max_dds[i] = float(dd.min())
    return {
        'paths': paths,
        'final_pnls': final_pnls,
        'max_dds': max_dds,
        'p5_path': np.percentile(paths, 5, axis=0),
        'p25_path': np.percentile(paths, 25, axis=0),
        'p50_path': np.percentile(paths, 50, axis=0),
        'p75_path': np.percentile(paths, 75, axis=0),
        'p95_path': np.percentile(paths, 95, axis=0),
        'p5_final': float(np.percentile(final_pnls, 5)),
        'p25_final': float(np.percentile(final_pnls, 25)),
        'p50_final': float(np.percentile(final_pnls, 50)),
        'p75_final': float(np.percentile(final_pnls, 75)),
        'p95_final': float(np.percentile(final_pnls, 95)),
        'p5_dd': float(np.percentile(max_dds, 5)),
        'p25_dd': float(np.percentile(max_dds, 25)),
        'p50_dd': float(np.percentile(max_dds, 50)),
        'p75_dd': float(np.percentile(max_dds, 75)),
        'p95_dd': float(np.percentile(max_dds, 95)),
        'n_runs': n_runs,
        'n_horizon_days': int(n_horizon_days),
    }


def _mc_percentile(observed: float, distribution: np.ndarray) -> Optional[float]:
    """Where does `observed` fall in the MC distribution (0-100 percentile)?
    Returns None if distribution is empty.

    Convention: pct = % of MC paths with value ≤ `observed`. For DDs (negative),
    a LOW percentile means live DD is in the worse tail. For returns (mixed sign),
    a LOW percentile means live return is in the worse tail.
    """
    if distribution is None or len(distribution) == 0:
        return None
    return float((distribution <= observed).mean() * 100)


def per_strategy_capital(total_cap: float, n_strats: int) -> float:
    """Equal-weight per-strategy capital allocation.

    Used across the live monitoring tab to derive per-strategy dollar bases for
    Live% / Live MDD% / chart equity calculations. Returns total_cap when N=0.
    """
    return float(total_cap) / max(int(n_strats), 1)


def live_pct_bt_end_based(ev, per_strat_cap: float) -> float:
    """Live cumulative P&L as a % of equity at the START of the live segment.

    Equity baseline for the live segment is BT-end equity, not the original
    starting capital — using starting_capital here would inflate Live% by a
    factor of (1 + bt_return), which is misleading when BT generated material
    P&L before live started (the live segment is a CONTINUATION of BT).

    Args:
        ev: StrategyEvaluation (preferred) or dict with bt_metrics, live_metrics keys.
        per_strat_cap: per-strategy starting capital in $.

    Returns 0.0 when bt_end_equity ≤ 0 (defensive — shouldn't happen in normal use).
    """
    bt_end_eq = float(per_strat_cap) + float(ev['bt_metrics']['total_pnl'])
    if bt_end_eq <= 0:
        return 0.0
    return float(ev['live_metrics']['total_pnl']) / bt_end_eq * 100.0


def _eval_segments(daily_pnl: pd.Series, starting_capital: float, rfr: float,
                    split_date: pd.Timestamp) -> dict:
    """Split daily_pnl at split_date, compute segment_metrics for BT and Live.

    Live segment uses BT-end-equity as its starting baseline (Live is a continuation
    of BT, not a fresh restart from `starting_capital`).

    Returns dict with: bt_pnl, live_pnl, bt_m, live_m, bt_end_equity, return_eff_pct,
    dd_eff_pct, live_final_pnl, live_dd_dollars, live_trades.
    """
    pnl = daily_pnl.fillna(0)
    bt_pnl = pnl[pnl.index < split_date]
    live_pnl = pnl[pnl.index >= split_date]

    bt_m = segment_metrics(bt_pnl, starting_capital, rfr)
    bt_end_equity = float(starting_capital + bt_pnl.sum()) if len(bt_pnl) else float(starting_capital)
    live_m = segment_metrics(live_pnl, bt_end_equity, rfr)

    # Return rate efficiency (live $/day rate vs BT $/day rate)
    bt_rate = bt_m['total_pnl'] / max(bt_m['n_days'], 1) if bt_m['n_days'] else 0.0
    live_rate = live_m['total_pnl'] / max(live_m['n_days'], 1) if live_m['n_days'] else 0.0
    ret_eff_pct = (live_rate / bt_rate * 100.0) if bt_rate != 0 else None

    # DD efficiency: |live MDD| / |BT MDD|
    dd_eff_pct = (abs(live_m['mdd']) / abs(bt_m['mdd']) * 100.0) if bt_m['mdd'] != 0 else None

    # Live cumulative P&L + peak-to-trough $ excursion (with 0 floor on peaks)
    live_cum = live_pnl.cumsum().values if len(live_pnl) > 0 else np.array([])
    live_final_pnl = float(live_cum[-1]) if len(live_cum) > 0 else 0.0
    if len(live_cum) > 0:
        peaks = np.maximum(np.maximum.accumulate(live_cum), 0)
        live_dd_dollars = float((live_cum - peaks).min())
    else:
        live_dd_dollars = 0.0

    return {
        'bt_pnl': bt_pnl, 'live_pnl': live_pnl,
        'bt_m': bt_m, 'live_m': live_m, 'bt_end_equity': bt_end_equity,
        'return_eff_pct': ret_eff_pct, 'dd_eff_pct': dd_eff_pct,
        'live_final_pnl': live_final_pnl,
        'live_dd_dollars': live_dd_dollars,
        'live_trades': int(live_m['n_active_days']),
    }


def _eval_mc_envelope(bt_pnl: pd.Series, live_final_pnl: float, live_dd_dollars: float,
                       n_live_days: int, n_mc_runs: int) -> dict:
    """Bootstrap MC envelope from BT and compute live percentiles.

    Returns dict with: mc (full bootstrap dist + percentile bands),
    mc_return_percentile, mc_dd_percentile.
    """
    mc = strategy_monte_carlo(bt_pnl, max(n_live_days, 1), n_runs=n_mc_runs)
    return {
        'mc': mc,
        'mc_return_percentile': _mc_percentile(live_final_pnl, mc['final_pnls']),
        'mc_dd_percentile': _mc_percentile(live_dd_dollars, mc['max_dds']),
    }


def _verdict_for_return(mc_ret_pct: Optional[float]) -> Tuple[str, str]:
    """Per-stat verdict for the Return %ile column."""
    if mc_ret_pct is None:
        return '⏳ n/a', VERDICT_COLORS['incubating']
    if mc_ret_pct < MC_TAIL_PCT:
        return f'🔴 Below MC P{MC_TAIL_PCT}', VERDICT_COLORS['kill']
    if mc_ret_pct < 25:
        return '🟡 Lower quartile', VERDICT_COLORS['warn']
    if mc_ret_pct > 95:
        return '🟢 Above MC P95', VERDICT_COLORS['keep']
    if mc_ret_pct > 75:
        return '🟢 Upper quartile', VERDICT_COLORS['keep']
    return '🟢 Within MC', VERDICT_COLORS['keep']


def _verdict_for_dd(mc_dd_pct: Optional[float]) -> Tuple[str, str]:
    """Per-stat verdict for the DD %ile column (lower = worse for DDs)."""
    if mc_dd_pct is None:
        return '⏳ n/a', VERDICT_COLORS['incubating']
    if mc_dd_pct < MC_TAIL_PCT:
        return f'🔴 Worse than MC P{MC_TAIL_PCT}', VERDICT_COLORS['kill']
    if mc_dd_pct < 25:
        return '🟡 DD worse than typical', VERDICT_COLORS['warn']
    if mc_dd_pct > 75:
        return '🟢 DD better than typical', VERDICT_COLORS['keep']
    return '🟢 DD within MC', VERDICT_COLORS['keep']


def _kill_verdict(mc_dd_pct: Optional[float], mc_ret_pct: Optional[float],
                   live_trades: int) -> Tuple[str, str]:
    """Combined kill rule verdict (see per_strategy_evaluation docstring for rules).

    Returns (verdict_string, color).
    """
    if mc_dd_pct is None or mc_ret_pct is None:
        return '⏳ Insufficient data', VERDICT_COLORS['incubating']
    if live_trades < MIN_LIVE_TRADES:
        return (f'⏳ Incubating ({live_trades}/{MIN_LIVE_TRADES} trades)',
                VERDICT_COLORS['incubating'])
    if mc_dd_pct < MC_TAIL_PCT and mc_ret_pct < MC_TAIL_PCT:
        return f'🔴 KILL (DD & Return < P{MC_TAIL_PCT})', VERDICT_COLORS['kill']
    if mc_dd_pct < MC_TAIL_PCT:
        return f'🔴 KILL (DD crash < P{MC_TAIL_PCT})', VERDICT_COLORS['kill']
    if mc_ret_pct < MC_TAIL_PCT:
        return f'🔴 KILL (slow bleed: Return < P{MC_TAIL_PCT})', VERDICT_COLORS['kill']
    if mc_dd_pct < MC_WARN_PCT or mc_ret_pct < MC_WARN_PCT:
        return f'🟡 WARN (near P{MC_WARN_PCT})', VERDICT_COLORS['warn']
    return '🟢 KEEP (within MC envelope)', VERDICT_COLORS['keep']


def _ks_edge_diagnosis(bt_pnl: pd.Series, live_pnl: pd.Series,
                        mc_dd_pct: Optional[float], mc_ret_pct: Optional[float]
                        ) -> Tuple[Optional[float], Optional[float], str, str]:
    """KS + Mann-Whitney on per-trade PnL → edge diagnosis verdict.

    Returns (ks_p, mw_p, edge_diagnosis_string, color). KS is direction-blind;
    used only as a diagnostic to distinguish broken-edge from bad-luck cases
    when the kill rule fires.
    """
    bt_active = bt_pnl[bt_pnl != 0].values
    live_active = live_pnl[live_pnl != 0].values
    ks_p: Optional[float] = None
    mw_p: Optional[float] = None
    if len(bt_active) >= MIN_BT_TRADES_FOR_KS and len(live_active) >= MIN_LIVE_TRADES_FOR_KS:
        try:
            ks_p = float(stats.ks_2samp(bt_active, live_active).pvalue)
            mw_p = float(stats.mannwhitneyu(bt_active, live_active,
                                            alternative='two-sided').pvalue)
        except Exception:
            ks_p, mw_p = None, None

    mc_fires = ((mc_dd_pct is not None and mc_dd_pct < MC_TAIL_PCT) or
                (mc_ret_pct is not None and mc_ret_pct < MC_TAIL_PCT))
    ks_fires = (ks_p is not None and ks_p < KS_ALPHA)

    if ks_p is None or mc_dd_pct is None or mc_ret_pct is None:
        return ks_p, mw_p, '⏳ n/a', VERDICT_COLORS['incubating']
    if mc_fires and ks_fires:
        return ks_p, mw_p, '🔴 BROKEN EDGE', VERDICT_COLORS['kill']
    if mc_fires and not ks_fires:
        return ks_p, mw_p, '🟠 UNLUCKY (edge intact)', VERDICT_COLORS['edge_drift']
    if not mc_fires and ks_fires:
        return ks_p, mw_p, '🟡 EDGE DRIFTING', VERDICT_COLORS['watch']
    return ks_p, mw_p, '🟢 STABLE', VERDICT_COLORS['keep']


def _rolling_metrics(pnl: pd.Series, starting_capital: float, rfr: float
                      ) -> Tuple[pd.Series, pd.Series]:
    """Rolling Sharpe + Calmar on full history (BT + Live combined)."""
    rets = (pnl.cumsum() + starting_capital).pct_change().fillna(0)
    roll_sh = rolling_sharpe(rets, window=ROLLING_WINDOW_DAYS, risk_free_annual=rfr)
    roll_cm = rolling_calmar(pnl, starting_capital, window=ROLLING_WINDOW_DAYS)
    return roll_sh, roll_cm


def per_strategy_evaluation(daily_pnl: pd.Series, starting_capital: float,
                              rfr: float, split_date: pd.Timestamp,
                              n_mc_runs: int = MC_DEFAULT_RUNS) -> 'StrategyEvaluation':
    """Master per-strategy evaluation for the Live Monitoring tab.

    Computes EVERY signal needed to render one strategy's drill-down panel,
    including the kill-rule verdict and edge-diagnosis verdict. The UI just
    reads from the returned dict — no further analysis happens downstream.

    Splits daily_pnl at `split_date` into BT (in-sample) and Live (out-of-sample)
    segments, runs all stats on each, then computes the live-vs-MC envelope
    comparison that drives the kill rule.

    Args:
        daily_pnl: full daily-P&L series (BT + Live combined, indexed by date)
        starting_capital: original portfolio capital allocated to this strategy
        rfr: annualized risk-free rate (for Sharpe / Sortino)
        split_date: timestamp where Live segment begins (typically 2025-12-03)
        n_mc_runs: number of bootstrap paths for MC envelope (default 1000)

    Returns:
        Dict with 22 keys covering:
          Segment stats:    bt_metrics, live_metrics
          Live observation: live_final_pnl, live_dd_dollars, live_trades
          MC envelope:      mc (full bootstrap dist), mc_return_percentile,
                            mc_dd_percentile
          Per-stat verdict: return_verdict, dd_verdict (+ colors)
          Kill rule:        combined_verdict, combined_color, min_live_trades
          Edge diagnosis:   ks_p, mw_p, edge_diagnosis, edge_diagnosis_color
          Rolling charts:   rolling_sharpe, rolling_calmar
          Efficiency:       return_eff_pct, dd_eff_pct
          Raw segments:     bt_pnl, live_pnl, split_date

    Kill rule (combined_verdict):
        KILL  if (MC_DD_%ile ≤ MC_TAIL_PCT OR MC_Ret_%ile ≤ MC_TAIL_PCT)
                 AND live_trades ≥ MIN_LIVE_TRADES
        WARN  if (MC_DD_%ile ≤ MC_WARN_PCT OR MC_Ret_%ile ≤ MC_WARN_PCT)
                 AND live_trades ≥ MIN_LIVE_TRADES
        INCUBATE  if live_trades < MIN_LIVE_TRADES
        KEEP  otherwise

    Edge diagnosis (edge_diagnosis):
        MC fires + KS fires  → 🔴 BROKEN EDGE (archive permanently)
        MC fires + KS quiet  → 🟠 UNLUCKY (suspend not archive)
        MC quiet + KS fires  → 🟡 EDGE DRIFTING (watch)
        Otherwise            → 🟢 STABLE
    """
    # ── 1. Segment metrics (BT / Live split, all derived $ stats) ──────────
    seg = _eval_segments(daily_pnl, starting_capital, rfr, split_date)

    # ── 2. MC envelope (bootstrap from BT, percentile of live values) ──────
    env = _eval_mc_envelope(
        seg['bt_pnl'], seg['live_final_pnl'], seg['live_dd_dollars'],
        n_live_days=len(seg['live_pnl']), n_mc_runs=n_mc_runs,
    )

    # ── 3. Per-stat verdicts (informational, displayed in drill-down banner)
    ret_verdict, ret_color = _verdict_for_return(env['mc_return_percentile'])
    dd_verdict, dd_color = _verdict_for_dd(env['mc_dd_percentile'])

    # ── 4. Combined kill verdict + KS-based edge diagnosis ─────────────────
    combined, combined_color = _kill_verdict(
        env['mc_dd_percentile'], env['mc_return_percentile'], seg['live_trades'],
    )
    ks_p, mw_p, edge_diagnosis, edge_color = _ks_edge_diagnosis(
        seg['bt_pnl'], seg['live_pnl'],
        env['mc_dd_percentile'], env['mc_return_percentile'],
    )

    # ── 5. Rolling Sharpe & Calmar on full history (BT + Live combined) ────
    roll_sh, roll_cm = _rolling_metrics(daily_pnl.fillna(0), starting_capital, rfr)

    return StrategyEvaluation(
        bt_metrics=seg['bt_m'], live_metrics=seg['live_m'],
        live_final_pnl=seg['live_final_pnl'],
        live_dd_dollars=seg['live_dd_dollars'],
        live_trades=seg['live_trades'], min_live_trades=MIN_LIVE_TRADES,
        return_eff_pct=seg['return_eff_pct'], dd_eff_pct=seg['dd_eff_pct'],
        mc=env['mc'],
        mc_return_percentile=env['mc_return_percentile'],
        mc_dd_percentile=env['mc_dd_percentile'],
        return_verdict=ret_verdict, return_verdict_color=ret_color,
        dd_verdict=dd_verdict, dd_verdict_color=dd_color,
        combined_verdict=combined, combined_color=combined_color,
        ks_p=ks_p, mw_p=mw_p,
        edge_diagnosis=edge_diagnosis, edge_diagnosis_color=edge_color,
        rolling_sharpe=roll_sh, rolling_calmar=roll_cm,
        bt_pnl=seg['bt_pnl'], live_pnl=seg['live_pnl'],
        split_date=split_date,
    )


def _ticker_live_price_change(ticker: str, split: pd.Timestamp,
                                 end: Optional[pd.Timestamp] = None) -> Optional[float]:
    """Cumulative price change of `ticker` from `split` to `end` (or today).
    Returns fraction (e.g. -0.22 = -22%). None on fetch failure or no data."""
    if not ticker or ticker == 'UNKNOWN':
        return None
    end_dt = end or pd.Timestamp.utcnow().tz_localize(None)
    cache_key = (ticker, str(split.date()), str(end_dt.date()))
    if cache_key in _TICKER_PRICE_CACHE:
        prices = _TICKER_PRICE_CACHE[cache_key]
    else:
        try:
            df = fetch_btc_daily(str(split.date()), str(end_dt.date()), symbol=ticker)
            if df.empty:
                _TICKER_PRICE_CACHE[cache_key] = pd.Series(dtype=float)
                return None
            prices = pd.Series(df['close'].values, index=df['time'])
            _TICKER_PRICE_CACHE[cache_key] = prices
        except Exception:
            return None
    if prices.empty or len(prices) < 2:
        return None
    return float(prices.iloc[-1] / prices.iloc[0] - 1.0)


def pair_divergence_analysis(plot_data: pd.DataFrame,
                              family_table: pd.DataFrame,
                              strategy_table: pd.DataFrame,
                              live_start: str,
                              rolling_window: int = 60,
                              ticker_regimes: Optional[dict] = None) -> dict:
    """For each multi-ticker family, analyse pair P&L dynamics: backtest vs live.

    Returns dict keyed by family name → analysis dict with:
      tickers, strategies (CSV stems), n_tickers
      bt_correlation, live_correlation, correlation_delta, correlation_pvalue
        (Fisher z-test for H0: correlations equal)
      bt_sign_agreement, live_sign_agreement (% of co-active days where signs match)
      rolling_correlation (pd.Series, rolling_window-day)
      bt_pnl, live_pnl (per-strategy DataFrames for plotting)
      ticker_data (dict of per-ticker stats + cumulative P&L)
      verdict, verdict_color, explanation

    For 3+ ticker families, only the first pairing is analysed; the rest are listed.
    """
    if family_table is None or family_table.empty or strategy_table is None or strategy_table.empty:
        return {}
    split = pd.Timestamp(live_start)
    multi = family_table[family_table['# Instances'] >= 2]
    if multi.empty:
        return {}

    out = {}
    for family in multi['Family']:
        instances = strategy_table[strategy_table['Family'] == family]
        cols = instances['Strategy'].tolist()
        if len(cols) < 2:
            continue

        pair_pnl = plot_data[cols].fillna(0)
        bt_pnl = pair_pnl[pair_pnl.index < split]
        live_pnl = pair_pnl[pair_pnl.index >= split]

        # Use first pair (ticker 0 vs ticker 1) for primary analysis
        c1, c2 = cols[0], cols[1]
        bt_co_active = bt_pnl[(bt_pnl[c1] != 0) | (bt_pnl[c2] != 0)]
        live_co_active = live_pnl[(live_pnl[c1] != 0) | (live_pnl[c2] != 0)]

        bt_corr = (bt_co_active[c1].corr(bt_co_active[c2])
                   if len(bt_co_active) >= 10 else float('nan'))
        live_corr = (live_co_active[c1].corr(live_co_active[c2])
                     if len(live_co_active) >= 5 else float('nan'))

        corr_pvalue = _fisher_z_correlation_test(
            bt_corr, len(bt_co_active), live_corr, len(live_co_active)
        )

        # Sign agreement on days when BOTH tickers actually traded
        bt_both = bt_pnl[(bt_pnl[c1] != 0) & (bt_pnl[c2] != 0)]
        live_both = live_pnl[(live_pnl[c1] != 0) & (live_pnl[c2] != 0)]
        bt_sign = (float((np.sign(bt_both[c1]) == np.sign(bt_both[c2])).mean() * 100)
                   if len(bt_both) >= 5 else float('nan'))
        live_sign = (float((np.sign(live_both[c1]) == np.sign(live_both[c2])).mean() * 100)
                     if len(live_both) >= 3 else float('nan'))

        # Rolling correlation (full series)
        rolling = pair_pnl[c1].rolling(
            rolling_window, min_periods=max(rolling_window // 3, 10)
        ).corr(pair_pnl[c2])

        # Per-ticker stats + cumulative P&L for plotting
        ticker_data = {}
        for col in cols:
            full_pnl = plot_data[col].fillna(0)
            full_cum = full_pnl.cumsum()
            ticker_data[col] = {
                'ticker': extract_ticker(col),
                'bt_total': float(bt_pnl[col].sum()),
                'live_total': float(live_pnl[col].sum()),
                'bt_active_days': int((bt_pnl[col] != 0).sum()),
                'live_active_days': int((live_pnl[col] != 0).sum()),
                'cum_pnl_series': full_cum,
            }

        live_pnls_for_verdict = {
            extract_ticker(c): ticker_data[c]['live_total'] for c in cols
        }

        # --- Direction + per-ticker live regime mix + price change %  --------
        # The most intuitive "is this regime bearish" signal is the cumulative
        # price change of the ticker during the live segment. We piggy-back on
        # ticker_regimes (which carries the index from a price-derived series),
        # plus a separately-fetched price series via `_ticker_live_price_change`.
        direction = extract_direction(cols[0])
        ticker_live_regime_mix = {}
        if ticker_regimes:
            for col in cols:
                tk = extract_ticker(col)
                tk_regime = ticker_regimes.get(tk)
                if tk_regime is None or tk_regime.empty:
                    continue
                live_tk_regime = tk_regime[tk_regime.index >= split]
                if len(live_tk_regime) == 0:
                    continue
                ticker_live_regime_mix[tk] = {
                    'Bull': float((live_tk_regime == 'Bull').mean()),
                    'Bear': float((live_tk_regime == 'Bear').mean()),
                    'Chop': float((live_tk_regime == 'Chop').mean()),
                    'price_change_pct': _ticker_live_price_change(tk, split),
                }

        verdict, color, explanation = _pair_divergence_verdict(
            bt_corr, live_corr, corr_pvalue,
            live_pnls_for_verdict, len(live_co_active),
            direction=direction,
            ticker_live_regime_mix=ticker_live_regime_mix,
        )

        out[family] = {
            'tickers': [extract_ticker(c) for c in cols],
            'strategies': cols,
            'n_tickers': len(cols),
            'direction': direction,
            'ticker_live_regime_mix': ticker_live_regime_mix,
            'bt_correlation': float(bt_corr) if pd.notna(bt_corr) else None,
            'live_correlation': float(live_corr) if pd.notna(live_corr) else None,
            'correlation_delta': (
                float(live_corr - bt_corr) if (pd.notna(bt_corr) and pd.notna(live_corr)) else None
            ),
            'correlation_pvalue': corr_pvalue,
            'bt_sign_agreement': bt_sign if pd.notna(bt_sign) else None,
            'live_sign_agreement': live_sign if pd.notna(live_sign) else None,
            'rolling_correlation': rolling,
            'bt_pnl': bt_pnl, 'live_pnl': live_pnl,
            'ticker_data': ticker_data,
            'verdict': verdict, 'verdict_color': color, 'explanation': explanation,
            'n_bt_co_active': int(len(bt_co_active)),
            'n_live_co_active': int(len(live_co_active)),
        }
    return out


def live_monitoring_analysis(plot_data: pd.DataFrame, metrics_df: pd.DataFrame,
                              total_cap: float, rfr: float,
                              live_start: str,
                              vt_kwargs: Optional[dict] = None,
                              regime_lookback: int = 60,
                              regime_bull_thr: float = 0.10,
                              regime_bear_thr: float = -0.10) -> dict:
    """Master live-monitoring orchestrator — produces every signal the live tab renders.

    Honest OOS workflow: vol-targeting is derived from BACKTEST-ONLY data, then
    that sizing is applied to the live segment. No look-ahead leakage. This
    answers "would the sizing we'd have chosen at deployment have survived live?"

    Pipeline:
        1. Re-run vol-targeting on backtest-only slice → honest deployment sizing
        2. Apply that sizing to the live segment → live portfolio P&L (no leakage)
        3. Compute segment_metrics(bt) and segment_metrics(live) at portfolio level
        4. Classify regimes for portfolio (BTC-based) AND per-ticker
        5. Build per_strategy_live_table (one row per strategy, regime-conditional)
        6. Build strategy_family_table (cross-ticker family roll-up)
        7. Run pair_divergence_analysis (Fisher z-test for paired strategies)
        8. Compute per_strategy_evaluation for each strategy (kill rule, MC, KS, edge)
        9. Build portfolio-level bootstrap equity envelope (P5–P95 hypothesis test)

    Args:
        plot_data: daily P&L per strategy (from process_portfolio)
        metrics_df: per-strategy stats (used to derive vt sizing params)
        total_cap: total portfolio capital
        rfr: annualized risk-free rate (for Sharpe / Sortino)
        live_start: ISO date 'YYYY-MM-DD' where live segment begins
        vt_kwargs: optional kwargs forwarded to mc_vol_targeted_allocation
        regime_lookback: rolling window (days) for regime classification (default 60)
        regime_bull_thr: BTC rolling return above this → BULL (default +10%)
        regime_bear_thr: BTC rolling return below this → BEAR (default -10%)

    Returns:
        Dict with:
          vt_bt_only          – mc_vol_targeted_allocation result on BT-only data
          bt_portfolio_pnl    – daily portfolio $ P&L during backtest
          live_portfolio_pnl  – daily portfolio $ P&L during live (BT-derived sizing)
          bt_metrics          – segment_metrics for backtest portfolio
          live_metrics        – segment_metrics for live portfolio
          portfolio_drift     – KS + Mann-Whitney distribution drift test
          strategy_table      – per-strategy live comparison DataFrame
          family_table        – cross-ticker family roll-up DataFrame
          pair_analysis       – pair-divergence analysis (Fisher z-test)
          per_strategy_evals  – {col: per_strategy_evaluation dict} for every strategy
          envelope_df         – portfolio-level MC bootstrap envelope P5/P25/P50/P75/P95
          envelope_status     – within/outside envelope verdict
          regime              – regime classification time series (BTC-derived)
          ticker_regimes      – {ticker: regime series} per-ticker classification
          split_date          – the LIVE_START timestamp
          backtest_days       – integer day count
          live_days           – integer day count
        On error: {'error': str} with diagnostic message.
    """
    split = pd.Timestamp(live_start)
    bt_plot = plot_data[plot_data.index < split]
    live_plot = plot_data[plot_data.index >= split]

    # If we don't have both segments, gracefully bail out
    if bt_plot.empty or live_plot.empty:
        return {
            'error': (
                f"Cannot split at {live_start}: "
                f"backtest={len(bt_plot)} days, live={len(live_plot)} days. "
                f"Check `Live incubation start` and ensure the date range includes both segments."
            ),
            'split_date': split,
            'backtest_days': len(bt_plot), 'live_days': len(live_plot),
        }

    # --- Step 1: Backtest-only metrics for backtest-only vol-targeting ---
    # We re-derive per-strategy metrics from the bt_plot (so Trades/Yr, Avg Position $
    # reflect BACKTEST-ONLY behavior — no live leakage into the sizing model).
    bt_metrics_df = _metrics_from_pnl_only(bt_plot, metrics_df, total_cap, rfr)

    # --- Step 2: Vol-target on backtest only ---
    vt_kwargs = vt_kwargs or {}
    vt_kwargs.setdefault('target_ror', 0.10)
    vt_kwargs.setdefault('ruin_fraction', 0.60)
    vt_kwargs.setdefault('max_leverage_cap', 1.0)
    vt_kwargs.setdefault('target_portfolio_vol', 0.20)
    vt_kwargs.setdefault('n_runs', 1000)
    vt_kwargs.setdefault('block_len', 5)
    vt_kwargs.setdefault('seed', 42)
    vt_kwargs.setdefault('normalize_backtest_pos', False)

    vt_bt = mc_vol_targeted_allocation(
        plot_data=bt_plot, metrics_df=bt_metrics_df, total_cap=total_cap, **vt_kwargs
    )

    # --- Step 3: Apply backtest-derived position sizes to BOTH segments ---
    # Backtest segment uses vt_bt['portfolio_returns'] (already net of costs).
    bt_port_pnl = vt_bt['portfolio_returns'] * total_cap
    bt_port_pnl.name = 'BT Portfolio P&L'

    live_port_pnl = pd.Series(0.0, index=live_plot.index, name='Live Portfolio P&L')
    cost_pct = (vt_kwargs.get('cost_bps_per_round_trip', 0) +
                vt_kwargs.get('slippage_bps', 0)) / 10000.0
    funding_pct_day = vt_kwargs.get('funding_bps_per_day', 0) / 10000.0
    ignore = {'Portfolio Equity', 'Portfolio DD', 'Portfolio Daily P&L',
              'B&H BTC Equity', 'Portfolio Load'}
    for col in [c for c in live_plot.columns if c not in ignore]:
        if col not in vt_bt['position_sizes']:
            continue
        bt_pos = vt_bt['backtest_positions'].get(col, 0)
        target_pos = vt_bt['position_sizes'].get(col, 0)
        if bt_pos <= 0 or target_pos <= 0:
            continue
        scaler = target_pos / bt_pos
        live_port_pnl = live_port_pnl + live_plot[col].fillna(0) * scaler
        # Apply same daily-amortized cost as the backtest segment
        if cost_pct > 0 or funding_pct_day > 0:
            tpy = (
                int(bt_metrics_df.loc[col, 'Trades/Yr'])
                if col in bt_metrics_df.index and 'Trades/Yr' in bt_metrics_df.columns
                else 0
            )
            daily_trade_cost = (max(tpy, 1) * cost_pct * target_pos) / 365.0
            daily_funding = funding_pct_day * target_pos
            live_port_pnl = live_port_pnl - (daily_trade_cost + daily_funding)

    # --- Step 4: Per-segment portfolio metrics ---
    bt_m = segment_metrics(bt_port_pnl, total_cap, rfr)
    live_m = segment_metrics(live_port_pnl, total_cap, rfr)
    drift = distribution_drift_test(bt_port_pnl, live_port_pnl)

    # --- Step 5: Regime classification (full series, then split) ---
    regime = None
    live_regime_mix = {'Bull': 0.0, 'Bear': 0.0, 'Chop': 0.0}
    bt_regime_mix = {'Bull': 0.0, 'Bear': 0.0, 'Chop': 0.0}
    if 'B&H BTC Equity' in plot_data.columns and plot_data['B&H BTC Equity'].std() > 0:
        regime = classify_regimes(
            plot_data['B&H BTC Equity'], lookback=regime_lookback,
            bull_threshold=regime_bull_thr, bear_threshold=regime_bear_thr,
        )
        live_regime = regime[regime.index >= split]
        bt_regime = regime[regime.index < split]
        if len(live_regime) > 0:
            live_regime_mix = {
                'Bull': float((live_regime == 'Bull').mean()),
                'Bear': float((live_regime == 'Bear').mean()),
                'Chop': float((live_regime == 'Chop').mean()),
            }
        if len(bt_regime) > 0:
            bt_regime_mix = {
                'Bull': float((bt_regime == 'Bull').mean()),
                'Bear': float((bt_regime == 'Bear').mean()),
                'Chop': float((bt_regime == 'Chop').mean()),
            }

    # --- Step 6a: Fetch PER-TICKER regimes UP-FRONT (classified by each ticker's own price) ---
    # Critical for long-only altcoin strategies: a falling SOL doesn't show in BTC regime.
    # Done before per-strategy table so the directional override has the data it needs.
    ticker_regimes: dict = {}
    unique_tickers = set()
    strategy_cols_for_tickers = [c for c in plot_data.columns if c not in ignore]
    for col in strategy_cols_for_tickers:
        tk = extract_ticker(col)
        if tk and tk != 'UNKNOWN':
            unique_tickers.add(tk)
    data_start = str(plot_data.index.min().date()) if not plot_data.empty else live_start
    data_end = str(plot_data.index.max().date()) if not plot_data.empty else live_start
    for tk in unique_tickers:
        tk_regime = _fetch_ticker_regime(
            tk, data_start, data_end,
            lookback=regime_lookback,
            bull_threshold=regime_bull_thr,
            bear_threshold=regime_bear_thr,
        )
        if not tk_regime.empty:
            ticker_regimes[tk] = tk_regime

    # --- Step 6b: Per-strategy comparison (regime + per-ticker direction-aware) ---
    strat_table = per_strategy_live_table(
        plot_data, metrics_df, total_cap, rfr, live_start, regime=regime,
        ticker_regimes=ticker_regimes,
    )

    # --- Step 7: Family-level roll-up ---
    family_table = strategy_family_table(strat_table)

    # --- Step 8: Pair-divergence analysis (uses ticker_regimes from Step 6a) ---
    pair_analysis = pair_divergence_analysis(
        plot_data, family_table, strat_table, live_start,
        ticker_regimes=ticker_regimes,
    )

    # --- Step 9: Per-strategy evaluations (Return/DD Eff, MC, rolling Sharpe/Calmar) ---
    # Computed upfront for all 16 strategies so the per-strategy table loads instantly.
    n_strats = len(strategy_cols_for_tickers)
    per_strat_cap = total_cap / max(n_strats, 1)
    per_strategy_evals: dict = {}
    for col in strategy_cols_for_tickers:
        per_strategy_evals[col] = per_strategy_evaluation(
            plot_data[col], per_strat_cap, rfr, split, n_mc_runs=1000,
        )

    # --- Step 9: Bootstrap equity envelope (portfolio level) ---
    # Build the ±2σ envelope for the live segment using BT daily P&L. This is
    # the "should live be here?" hypothesis test: under H0 of no decay, the
    # live equity curve should stay within the P5-P95 band ~90% of the time.
    n_live = len(live_port_pnl)
    envelope_df = bootstrap_equity_envelope(
        bt_port_pnl, n_live_days=n_live, starting_equity=float(total_cap),
        n_sims=1000, seed=42, block_len=5,
    )
    live_equity_actual = live_port_pnl.cumsum() + total_cap
    envelope_status = live_within_envelope(live_equity_actual.values, envelope_df)

    return {
        'vt_bt_only': vt_bt,
        'bt_portfolio_pnl': bt_port_pnl,
        'live_portfolio_pnl': live_port_pnl,
        'bt_metrics': bt_m, 'live_metrics': live_m,
        'portfolio_drift': drift,
        'strategy_table': strat_table,
        'family_table': family_table,
        'pair_analysis': pair_analysis,
        'per_strategy_evals': per_strategy_evals,  # per-strategy MC + rolling + verdicts
        'live_regime_mix': live_regime_mix,
        'bt_regime_mix': bt_regime_mix,
        'regime': regime,                  # BTC-based portfolio regime
        'ticker_regimes': ticker_regimes,  # per-ticker independent regimes
        'envelope_df': envelope_df,        # bootstrap ±2σ equity bands
        'envelope_status': envelope_status,  # within/outside dict
        'live_equity_actual': live_equity_actual,
        'split_date': split,
        'backtest_days': len(bt_plot),
        'live_days': len(live_plot),
    }


def _metrics_from_pnl_only(plot_data_subset: pd.DataFrame,
                            full_metrics_df: pd.DataFrame,
                            total_cap: float, rfr: float) -> pd.DataFrame:
    """Build a metrics_df-shaped frame for a TIME SUBSET of plot_data.

    Some columns (Avg Position $) come from the original metrics_df since we
    don't have access to per-trade position data here; others (Trades/Yr) are
    re-computed from the subset's active days. This is the input required by
    mc_vol_targeted_allocation when re-running on a date sub-range.
    """
    ignore = {'Portfolio Equity', 'Portfolio DD', 'Portfolio Daily P&L',
              'B&H BTC Equity', 'Portfolio Load'}
    strategy_cols = [c for c in plot_data_subset.columns if c not in ignore]
    n = max(len(strategy_cols), 1)
    per_strat_cap = total_cap / n
    if plot_data_subset.empty:
        return pd.DataFrame()
    days_span = max(
        (plot_data_subset.index[-1] - plot_data_subset.index[0]).days, 1
    )
    rows = []
    for col in strategy_cols:
        col_pnl = plot_data_subset[col].fillna(0)
        nonzero = col_pnl[col_pnl != 0]
        n_active = int(len(nonzero))
        if n_active < 1:
            continue
        tpy = (n_active / days_span) * 365.25
        # Avg Position $ is taken from full metrics_df (per-trade size is regime-stable)
        avg_pos = (
            float(full_metrics_df.loc[col, 'Avg Position $'])
            if col in full_metrics_df.index and 'Avg Position $' in full_metrics_df.columns
            else per_strat_cap
        )
        rows.append({
            'Strategy': col,
            'Trades/Yr': tpy,
            'Avg Position $': avg_pos,
        })
    return pd.DataFrame(rows).set_index('Strategy') if rows else pd.DataFrame()


# ============================================================================
# EXTENDED PORTFOLIO METRICS (quantstats-style)
# ============================================================================

def skew_kurtosis(daily_returns: pd.Series) -> Tuple[float, float]:
    """Skew and Fisher kurtosis (+3) of return distribution.
    Kurtosis > 3 = fatter than normal (tail risk)."""
    r = pd.Series(daily_returns).dropna()
    if len(r) < 3:
        return 0.0, 0.0
    return float(stats.skew(r)), float(stats.kurtosis(r) + 3)


def tail_ratio(daily_returns: pd.Series, cutoff: float = 0.95) -> float:
    """P(cutoff) / |P(1-cutoff)| of returns. >1 = right tail dominates."""
    r = np.asarray(daily_returns)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return 0.0
    p_hi = np.percentile(r, cutoff * 100)
    p_lo = np.percentile(r, (1 - cutoff) * 100)
    return float(p_hi / abs(p_lo)) if p_lo != 0 else float('inf')


def common_sense_ratio(daily_returns: pd.Series) -> float:
    """Profit factor × Tail Ratio. >1.5 = robust edge per quantstats."""
    r = np.asarray(daily_returns)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return 0.0
    pf = profit_factor(r)
    tr = tail_ratio(r)
    if not np.isfinite(pf):
        return float('inf')
    return float(pf * tr) if np.isfinite(tr) else float('inf')


def omega_ratio(daily_returns: pd.Series, threshold: float = 0.0) -> float:
    """Sum of returns above threshold / |sum below|. >1 = positive bias."""
    r = np.asarray(daily_returns)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return 0.0
    above = r[r > threshold].sum()
    below = -r[r < threshold].sum()
    return float(above / below) if below > 0 else float('inf')


def ulcer_index(equity_series: pd.Series) -> float:
    """RMS drawdown — penalizes both depth AND duration of drawdowns.
    Lower = smoother equity curve."""
    if len(equity_series) == 0:
        return 0.0
    eq = np.asarray(equity_series.values, dtype=float)
    peaks = np.maximum.accumulate(eq)
    with np.errstate(divide='ignore', invalid='ignore'):
        dd = np.where(peaks > 0, (eq - peaks) / peaks, 0)
    return float(np.sqrt(np.mean(dd ** 2)) * 100)  # %


def ulcer_performance_index(equity_series: pd.Series, rfr: float = 0.04) -> float:
    """(CAGR - rfr) / Ulcer Index. Sharpe-like but uses Ulcer not vol."""
    if len(equity_series) < 2:
        return 0.0
    u = ulcer_index(equity_series)
    if u == 0:
        return 0.0
    days = max((equity_series.index[-1] - equity_series.index[0]).days, 1)
    start_v = float(equity_series.iloc[0])
    end_v = float(equity_series.iloc[-1])
    cagr_val = get_cagr(start_v, end_v, days)
    return float((cagr_val - rfr) / (u / 100.0))


def recovery_factor(total_return: float, max_dd: float) -> float:
    """Total return / |MaxDD|. >5 = strong recovery from drawdowns."""
    if max_dd == 0:
        return 0.0
    return float(total_return / abs(max_dd))


def smart_sharpe(daily_returns: pd.Series, rfr: float = 0.04,
                 periods: int = 365) -> float:
    """Sharpe penalized by 1-lag autocorrelation. Smooth-return strategies
    (often a backtest artifact) get downweighted."""
    r = pd.Series(daily_returns).dropna()
    if len(r) < 10 or r.std() == 0:
        return 0.0
    rf_daily = rfr / periods
    excess = r - rf_daily
    sharpe = (excess.mean() / r.std()) * np.sqrt(periods)
    try:
        rho = r.autocorr(lag=1)
        if rho is None or np.isnan(rho):
            rho = 0
    except Exception:
        rho = 0
    # Penalty factor: sqrt(1 + 2ρ) inflates std when ρ>0 (returns are smoother than IID)
    penalty = np.sqrt(max(1 + 2 * rho, 0.01))
    return float(sharpe / penalty)


def kelly_criterion(daily_returns: pd.Series) -> float:
    """Optimal bet fraction = mean / variance. Returned as fraction (0.15 = 15%)."""
    r = pd.Series(daily_returns).dropna()
    if len(r) == 0 or r.var() == 0:
        return 0.0
    return float(r.mean() / r.var())


def beta_alpha_correlation(returns: pd.Series, bench_returns: pd.Series,
                            rfr: float = 0.04) -> Tuple[float, float, float]:
    """Returns (beta, annualized alpha, correlation) vs benchmark."""
    if len(returns) < 2 or len(bench_returns) < 2:
        return 0.0, 0.0, 0.0
    aligned = pd.concat([returns, bench_returns], axis=1, join='inner').dropna()
    if len(aligned) < 2:
        return 0.0, 0.0, 0.0
    aligned.columns = ['r', 'b']
    var_b = aligned['b'].var()
    if var_b == 0:
        return 0.0, 0.0, 0.0
    beta_val = float(aligned['r'].cov(aligned['b']) / var_b)
    rf_d = rfr / 365.0
    alpha_daily = (aligned['r'].mean() - rf_d) - beta_val * (aligned['b'].mean() - rf_d)
    alpha_ann = float(alpha_daily * 365)
    corr = float(aligned['r'].corr(aligned['b']))
    return beta_val, alpha_ann, corr


def information_ratio(returns: pd.Series, bench_returns: pd.Series) -> float:
    """Annualized active return / tracking error. >0.5 = consistent alpha."""
    aligned = pd.concat([returns, bench_returns], axis=1, join='inner').dropna()
    if len(aligned) < 2:
        return 0.0
    aligned.columns = ['r', 'b']
    active = aligned['r'] - aligned['b']
    if active.std() == 0:
        return 0.0
    return float((active.mean() / active.std()) * np.sqrt(365))


def treynor_ratio(returns: pd.Series, bench_returns: pd.Series,
                  rfr: float = 0.04) -> float:
    """(Annualized return - rfr) / Beta. Like Sharpe but uses systematic risk."""
    beta_val, _, _ = beta_alpha_correlation(returns, bench_returns, rfr)
    if beta_val == 0:
        return 0.0
    aligned = pd.concat([returns, bench_returns], axis=1, join='inner').dropna()
    if len(aligned) < 2:
        return 0.0
    annual_ret = float(aligned.iloc[:, 0].mean() * 365)
    return float((annual_ret - rfr) / beta_val)


def period_returns(equity_series: pd.Series) -> dict:
    """Return % for MTD, 3M, 6M, YTD, 1Y, 3Y, 5Y, All-time."""
    if len(equity_series) < 2:
        return {}
    eq = equity_series.copy()
    eq.index = pd.DatetimeIndex(eq.index)
    last_date = eq.index[-1]
    last_val = float(eq.iloc[-1])
    res = {}

    def ret_from(start_date):
        sub = eq[eq.index >= start_date]
        if len(sub) == 0 or sub.iloc[0] == 0:
            return None
        return (last_val / float(sub.iloc[0]) - 1) * 100

    res['MTD'] = ret_from(last_date.replace(day=1))
    res['3M'] = ret_from(last_date - pd.DateOffset(months=3))
    res['6M'] = ret_from(last_date - pd.DateOffset(months=6))
    res['YTD'] = ret_from(pd.Timestamp(year=last_date.year, month=1, day=1))
    res['1Y'] = ret_from(last_date - pd.DateOffset(years=1))
    res['3Y'] = ret_from(last_date - pd.DateOffset(years=3))
    res['5Y'] = ret_from(last_date - pd.DateOffset(years=5))
    res['All-time'] = (last_val / float(eq.iloc[0]) - 1) * 100 if eq.iloc[0] != 0 else None
    # Replace Nones
    return {k: (v if v is not None else 0.0) for k, v in res.items()}


def worst_drawdowns(equity_series: pd.Series, n: int = 5,
                    dd_series: Optional[pd.Series] = None) -> pd.DataFrame:
    """Top N worst drawdown episodes, with Start / Valley / End / Days / MaxDD%.

    If ``dd_series`` is provided, episodes are detected against THAT series so
    the depths and dates match whichever convention computed it (e.g. pass the
    output of ``get_max_drawdown`` for consistency with the headline MaxDD).
    Otherwise defaults to standard peak-relative drawdown.
    """
    if len(equity_series) < 2:
        return pd.DataFrame()
    eq = equity_series.copy()
    eq.index = pd.DatetimeIndex(eq.index)
    if dd_series is not None and len(dd_series) == len(eq):
        dd = dd_series.copy()
        dd.index = eq.index
    else:
        peaks = eq.cummax()
        dd = (eq - peaks) / peaks
    in_dd = (dd < -1e-9).values  # bool array
    episodes = []
    start_pos = None
    for i in range(len(dd)):
        if in_dd[i] and start_pos is None:
            start_pos = i
        elif not in_dd[i] and start_pos is not None:
            seg = dd.iloc[start_pos:i]
            if len(seg) > 0:
                valley_pos = int(seg.values.argmin())
                episodes.append({
                    'Start': dd.index[start_pos],
                    'Valley': dd.index[start_pos + valley_pos],
                    'End': dd.index[i],
                    'Days': int((dd.index[i] - dd.index[start_pos]).days),
                    'MaxDD %': float(seg.min() * 100),
                })
            start_pos = None
    if start_pos is not None:
        seg = dd.iloc[start_pos:]
        valley_pos = int(seg.values.argmin())
        episodes.append({
            'Start': dd.index[start_pos],
            'Valley': dd.index[start_pos + valley_pos],
            'End': dd.index[-1],
            'Days': int((dd.index[-1] - dd.index[start_pos]).days),
            'MaxDD %': float(seg.min() * 100),
        })
    if not episodes:
        return pd.DataFrame()
    return pd.DataFrame(episodes).sort_values('MaxDD %').head(n).reset_index(drop=True)


def consecutive_streaks(daily_returns: pd.Series) -> Tuple[int, int]:
    """Max consecutive winning days and max consecutive losing days."""
    r = np.asarray(daily_returns)
    if len(r) == 0:
        return 0, 0
    max_win = max_loss = cur_win = cur_loss = 0
    for v in r:
        if v > 0:
            cur_win += 1
            cur_loss = 0
            if cur_win > max_win:
                max_win = cur_win
        elif v < 0:
            cur_loss += 1
            cur_win = 0
            if cur_loss > max_loss:
                max_loss = cur_loss
        else:
            cur_win = cur_loss = 0
    return int(max_win), int(max_loss)


def period_win_rates(equity_series: pd.Series) -> Tuple[float, float, float]:
    """Win % at monthly / quarterly / yearly bucket level."""
    if len(equity_series) < 2:
        return 0.0, 0.0, 0.0
    eq = equity_series.copy()
    eq.index = pd.DatetimeIndex(eq.index)
    m = eq.resample('ME').last().pct_change().dropna()
    q = eq.resample('QE').last().pct_change().dropna()
    y = eq.resample('YE').last().pct_change().dropna()
    return (
        float((m > 0).mean() * 100) if len(m) else 0.0,
        float((q > 0).mean() * 100) if len(q) else 0.0,
        float((y > 0).mean() * 100) if len(y) else 0.0,
    )


def best_worst_extremes(equity_series: pd.Series) -> dict:
    """Best/worst day, month, year returns as %."""
    if len(equity_series) < 2:
        return {}
    eq = equity_series.copy()
    eq.index = pd.DatetimeIndex(eq.index)
    daily = eq.pct_change().dropna() * 100
    monthly = eq.resample('ME').last().pct_change().dropna() * 100
    yearly = eq.resample('YE').last().pct_change().dropna() * 100
    return {
        'best_day': float(daily.max()) if len(daily) else 0.0,
        'worst_day': float(daily.min()) if len(daily) else 0.0,
        'best_month': float(monthly.max()) if len(monthly) else 0.0,
        'worst_month': float(monthly.min()) if len(monthly) else 0.0,
        'best_year': float(yearly.max()) if len(yearly) else 0.0,
        'worst_year': float(yearly.min()) if len(yearly) else 0.0,
    }


def time_in_market(daily_returns: pd.Series) -> float:
    """% of days with non-zero P&L. 100% = always trading; lower = idle days."""
    r = pd.Series(daily_returns).fillna(0)
    if len(r) == 0:
        return 0.0
    return float((r != 0).mean() * 100)


# ============================================================================
# BTC BENCHMARK AUTO-FETCH (Binance public API — no auth, no rate limits)
# ============================================================================

def fetch_btc_daily(start: str, end: str, symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Pull daily OHLCV from Binance's public REST API. Returns DataFrame with
    'time' (UTC, normalized to date) and 'close' columns, empty on failure.

    No API key required. Free for public market data. Paginates if range > 1000d.
    """
    try:
        start_ms = int(pd.Timestamp(start).tz_localize('UTC').timestamp() * 1000)
        end_ms = int(pd.Timestamp(end).tz_localize('UTC').timestamp() * 1000)
    except Exception:
        return pd.DataFrame()

    rows = []
    cursor = start_ms
    # Binance returns up to 1000 candles per call → paginate
    while cursor < end_ms:
        try:
            r = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={
                    'symbol': symbol, 'interval': '1d',
                    'startTime': cursor, 'endTime': end_ms, 'limit': 1000,
                },
                timeout=15,
            )
            r.raise_for_status()
            batch = r.json()
        except Exception:
            return pd.DataFrame(rows, columns=['time', 'close']) if rows else pd.DataFrame()
        if not batch:
            break
        rows.extend(batch)
        last_open_ms = batch[-1][0]
        # Advance one day past the last open time we received
        cursor = last_open_ms + 24 * 60 * 60 * 1000
        if len(batch) < 1000:
            break  # got everything available

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        't_open', 'o', 'h', 'l', 'c', 'v', 't_close', 'q', 'n', 'tb', 'tq', 'i',
    ])
    df['time'] = pd.to_datetime(df['t_open'], unit='ms', utc=True).dt.tz_localize(None).dt.normalize()
    df['close'] = pd.to_numeric(df['c'], errors='coerce')
    out = df[['time', 'close']].dropna().drop_duplicates(subset='time').reset_index(drop=True)
    return out


# ============================================================================
# STRESS CORRELATIONS
# ============================================================================

def stress_correlation(plot_data: pd.DataFrame, percentile: int = 10) -> pd.DataFrame:
    """Correlation matrix on the worst N% portfolio days."""
    if 'Portfolio Daily P&L' not in plot_data.columns:
        return pd.DataFrame()
    port_pnl = plot_data['Portfolio Daily P&L']
    threshold = port_pnl.quantile(percentile / 100.0)
    bad_days = plot_data[port_pnl <= threshold]
    ignore = {'Portfolio Equity', 'Portfolio DD', 'Portfolio Daily P&L',
              'B&H BTC Equity', 'Portfolio Load'}
    strategy_cols = [c for c in plot_data.columns if c not in ignore]
    return bad_days[strategy_cols].corr()
