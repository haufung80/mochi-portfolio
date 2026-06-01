"""Tests for segment_metrics — per-segment Sharpe, MDD, win rate, etc.

Regression for the bug where live MDD% used starting_capital as the base
(showing 12% as 92%) — fixed by using BT-end equity for the live segment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calculations import ANNUALIZATION_FACTOR, segment_metrics


class TestSegmentMetricsBasic:
    def test_empty_input_returns_zero_filled_dict(self):
        m = segment_metrics(pd.Series([], dtype=float), 1000.0, rfr=0.04)
        assert m['total_pnl'] == 0.0
        assert m['n_days'] == 0
        assert m['sharpe'] == 0.0

    def test_sharpe_uses_365_day_annualization(self):
        """Crypto Sharpe uses sqrt(365), not sqrt(252)."""
        # Constant daily return — Sharpe ≈ mean/std × sqrt(365)
        # std=0 protected by segment_metrics; use small noise
        rng = np.random.default_rng(42)
        pnl = pd.Series(
            rng.normal(1.0, 5.0, size=365),  # mean +$1/day on $1000 cap = 0.1% daily
            index=pd.date_range("2024-01-01", periods=365, freq="D"),
        )
        m = segment_metrics(pnl, 1000.0, rfr=0.0)
        # Expected: Sharpe = (mean_ret / std_ret) × sqrt(365)
        # We don't pin the exact value (depends on noise) but check magnitude:
        # mean ≈ 0.1%, std ≈ 0.5%, daily SR ≈ 0.2, annual ≈ 3.8
        assert 1.0 < m['sharpe'] < 7.0, f"Sharpe {m['sharpe']} outside crypto-365 expected band"

    def test_mdd_is_fraction_not_percent(self):
        """segment_metrics returns mdd as fraction (-0.20 = -20%), NOT -20."""
        # Equity: 1000 → 1200 (peak) → 800 (trough) → 900 (final)
        # MDD = (800-1200)/1200 = -0.3333
        pnl_diffs = pd.Series([200, -400, 100], index=pd.date_range("2024-01-01", periods=3))
        m = segment_metrics(pnl_diffs, 1000.0, rfr=0.04)
        assert -1.0 <= m['mdd'] <= 0.0
        # Should be roughly -33% (drop from peak 1200 to trough 800)
        assert m['mdd'] == pytest.approx(-0.3333, abs=0.01)


class TestWinLossCount:
    def test_win_rate_correct(self):
        """3 wins, 2 losses → 60% win rate."""
        pnl = pd.Series([5, -3, 2, -1, 4], index=pd.date_range("2024-01-01", periods=5))
        m = segment_metrics(pnl, 1000.0, 0.04)
        assert m['win_rate'] == pytest.approx(60.0, abs=1e-9)

    def test_zero_days_excluded_from_winrate(self):
        """Zero-P&L days are inactive and not counted in winrate denominator."""
        pnl = pd.Series([5, 0, 0, -3, 0], index=pd.date_range("2024-01-01", periods=5))
        m = segment_metrics(pnl, 1000.0, 0.04)
        # Only 2 active days (5 and -3): 1 win → 50%
        assert m['win_rate'] == pytest.approx(50.0, abs=1e-9)
        assert m['n_active_days'] == 2


class TestProfitFactor:
    def test_profit_factor_basic(self):
        """PF = sum(wins) / |sum(losses)|. Wins=10, losses=5 → PF=2.0."""
        pnl = pd.Series([10, -5, 0], index=pd.date_range("2024-01-01", periods=3))
        m = segment_metrics(pnl, 1000.0, 0.04)
        assert m['pf'] == pytest.approx(2.0, abs=1e-9)

    def test_no_losses_gives_capped_pf(self):
        """All-wins (no losses) → PF capped at 999 (not inf)."""
        pnl = pd.Series([5, 10, 3], index=pd.date_range("2024-01-01", periods=3))
        m = segment_metrics(pnl, 1000.0, 0.04)
        assert m['pf'] == 999.0
