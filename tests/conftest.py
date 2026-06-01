"""Shared pytest fixtures — synthetic P&L series for every test scenario.

Synthetic data lets us exhaustively test math without depending on real CSVs
(which would tie tests to a specific portfolio snapshot). Each fixture is
named after the regime/pathology it represents so tests read clearly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent directory so `import calculations` works from tests/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Shared constants
LIVE_START = pd.Timestamp("2025-12-03")
START_CAP = 1000.0


def _index(n_days: int, end: pd.Timestamp = pd.Timestamp("2026-05-22")) -> pd.DatetimeIndex:
    """Daily date index ending at `end` with `n_days` entries."""
    return pd.date_range(end=end, periods=n_days, freq="D")


@pytest.fixture
def synthetic_bull_pnl() -> pd.Series:
    """Steady positive-edge strategy. Mean +$0.5/day, std $5. 400 days total."""
    rng = np.random.default_rng(0)
    idx = _index(400)
    return pd.Series(rng.normal(0.5, 5.0, size=400), index=idx)


@pytest.fixture
def synthetic_bear_pnl() -> pd.Series:
    """Steady losing strategy. Mean -$0.5/day, std $5."""
    rng = np.random.default_rng(1)
    idx = _index(400)
    return pd.Series(rng.normal(-0.5, 5.0, size=400), index=idx)


@pytest.fixture
def synthetic_chop_pnl() -> pd.Series:
    """Zero-edge random walk. Mean 0, std $5. Used for null-hypothesis testing."""
    rng = np.random.default_rng(2)
    idx = _index(400)
    return pd.Series(rng.normal(0.0, 5.0, size=400), index=idx)


@pytest.fixture
def synthetic_broken_pnl() -> pd.Series:
    """BT good, live BAD — edge truly broken. Tests KS BROKEN_EDGE path."""
    rng = np.random.default_rng(3)
    idx = _index(400)
    pnl = np.empty(400)
    # 250 days BT with positive mean & low vol
    pnl[:250] = rng.normal(1.0, 3.0, size=250)
    # 150 days live with negative mean & high vol (clearly different distribution)
    pnl[250:] = rng.normal(-2.0, 8.0, size=150)
    return pd.Series(pnl, index=idx)


@pytest.fixture
def synthetic_unlucky_pnl() -> pd.Series:
    """Same distribution BT and live, but live happens to draw a bad sequence.

    Same mean & std in both segments — KS should NOT fire. But live cumsum may
    fall to MC P5 by pure luck. Used to validate kill-vs-suspend logic.
    """
    rng = np.random.default_rng(4)
    idx = _index(400)
    pnl = rng.normal(0.3, 4.0, size=400)
    # Force the last 30 days to have an extreme cluster of losses
    pnl[-30:] -= 5.0
    return pd.Series(pnl, index=idx)


@pytest.fixture
def synthetic_sparse_pnl() -> pd.Series:
    """Low-frequency strategy: < 20 active days in BT (sparse trader).

    Used to verify the KS sample-size floor and Edge Diagnosis fallback.
    """
    rng = np.random.default_rng(5)
    idx = _index(400)
    pnl = pd.Series(0.0, index=idx)
    # 15 BT active days (below MIN_BT_TRADES_FOR_KS=20)
    bt_active = rng.choice(range(250), size=15, replace=False)
    pnl.iloc[bt_active] = rng.normal(0, 8.0, size=15)
    # 8 live active days
    lv_active = rng.choice(range(250, 400), size=8, replace=False)
    pnl.iloc[lv_active] = rng.normal(-3, 5.0, size=8)
    return pnl


@pytest.fixture
def synthetic_empty_live_pnl() -> pd.Series:
    """All trades before live_start — empty live segment. Tests crash safety."""
    rng = np.random.default_rng(6)
    # Index ends BEFORE LIVE_START
    idx = pd.date_range("2024-01-01", "2025-11-01", freq="D")
    return pd.Series(rng.normal(0.3, 4.0, size=len(idx)), index=idx)


@pytest.fixture
def synthetic_equity_curve() -> pd.Series:
    """Equity curve with a known peak and trough — for MDD formula tests.

    Equity goes: 1000 → 1500 (peak) → 1200 (trough) → 1400 → 1300 (final).
    Peak-to-trough drop = 1500 - 1200 = 300. Standard MDD = 300/1500 = 20%.
    """
    idx = _index(5, end=pd.Timestamp("2024-01-05"))
    return pd.Series([1000.0, 1500.0, 1200.0, 1400.0, 1300.0], index=idx)


@pytest.fixture
def synthetic_underwater_equity() -> pd.Series:
    """Equity starts above seed, dips below seed, never recovers. Tests floor logic."""
    idx = _index(5, end=pd.Timestamp("2024-01-05"))
    return pd.Series([1000.0, 1200.0, 800.0, 700.0, 600.0], index=idx)


@pytest.fixture
def starting_capital() -> float:
    return START_CAP


@pytest.fixture
def split_date() -> pd.Timestamp:
    return LIVE_START


@pytest.fixture
def tz_aware_tv_csv(tmp_path: Path) -> Path:
    """Write a fake TradingView CSV with tz-aware timestamps to disk.

    Regression for the bug where tz-aware Date/Time columns silently dropped
    strategies from `process_portfolio` (would compare tz-aware vs tz-naive
    string oos_start → TypeError → caught by outer try/except → strategy
    excluded with no warning to user).
    """
    csv_path = tmp_path / "TZ_AWARE_BINANCE_BTCUSDT_2026-05-23.csv"
    rows = []
    rows.append("Trade #,Type,Date/Time,Net P&L USDT,Signal,Cumulative P&L,Position size (value)")
    # 30 trades, half BT half live, all tz-aware UTC timestamps
    start = pd.Timestamp("2024-01-01", tz="UTC")
    for i in range(30):
        dt = start + pd.Timedelta(days=i * 10)
        rows.append(f"{i+1},Exit long,{dt.isoformat()},{(i % 5) * 2 - 4}.00,exit,,500")
    csv_path.write_text("\n".join(rows))
    return csv_path
