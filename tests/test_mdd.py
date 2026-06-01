"""Tests for get_max_drawdown — peak-based MDD formula.

Regression for the bug where MDD was computed as drop/starting_capital instead
of drop/peak. The old formula could produce >100% MDD when per-strategy capital
was small relative to equity excursion (e.g., $115 drop on $125 cap → -92% MDD).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calculations import get_max_drawdown


class TestMddFormula:
    """Verify the peak-based MDD math against hand-computed values."""

    def test_simple_peak_to_trough(self, synthetic_equity_curve):
        """Equity 1000→1500→1200 → MDD = (1200-1500)/1500 = -20%."""
        mdd, _ = get_max_drawdown(synthetic_equity_curve, 1000.0)
        assert mdd == pytest.approx(-0.20, abs=1e-9), (
            f"Expected -20%, got {mdd:.4f}. Peak-based formula broken."
        )

    def test_mdd_is_bounded_below_neg_one(self):
        """Standard MDD is in [-1, 0]. >100% MDD = formula bug (old behavior)."""
        rng = np.random.default_rng(42)
        # Pathological: equity dives below starting cap. Old formula would give
        # >100% MDD here; new formula must stay bounded.
        eq = pd.Series([100, 50, 30, 20, 15], index=pd.date_range("2024-01-01", periods=5))
        mdd, _ = get_max_drawdown(eq, starting_capital=100.0)
        assert -1.0 <= mdd <= 0.0, (
            f"MDD {mdd} outside [-1, 0]. Regression: drop/starting_cap bug."
        )

    def test_mdd_zero_when_equity_only_rises(self):
        """Monotone-rising equity has no drawdown."""
        eq = pd.Series([100, 110, 120, 150, 200], index=pd.date_range("2024-01-01", periods=5))
        mdd, dd_series = get_max_drawdown(eq, 100.0)
        assert mdd == pytest.approx(0.0, abs=1e-9)
        assert (dd_series == 0).all()

    def test_mdd_cap_invariance_when_equity_dominates(self, synthetic_equity_curve):
        """Once peak > starting_cap, the MDD should be cap-INVARIANT.

        Peak-based MDD = drop/peak — denominator is observed peak, not seed.
        Changing seed (as long as it stays below peak) must not shift MDD.
        """
        mdd_low_cap, _ = get_max_drawdown(synthetic_equity_curve, 100.0)
        mdd_high_cap, _ = get_max_drawdown(synthetic_equity_curve, 999.0)
        assert mdd_low_cap == pytest.approx(mdd_high_cap, abs=1e-9), (
            f"MDD shifted with cap: {mdd_low_cap} vs {mdd_high_cap}. "
            "Should be peak-based and cap-invariant."
        )

    def test_underwater_from_start_uses_cap_floor(self, synthetic_underwater_equity):
        """If equity starts ABOVE then dips BELOW seed, peak is the high point."""
        # synthetic_underwater_equity: 1000 → 1200 (peak) → 800 → 700 → 600 (trough)
        # Drop = 1200 - 600 = 600. MDD = 600 / 1200 = -50%.
        mdd, _ = get_max_drawdown(synthetic_underwater_equity, 1000.0)
        assert mdd == pytest.approx(-0.50, abs=1e-9)

    def test_empty_series_returns_zero(self):
        """Empty input → 0 MDD, no exception."""
        mdd, dd = get_max_drawdown(pd.Series([], dtype=float), 1000.0)
        assert mdd == 0.0
        assert len(dd) == 0

    def test_dd_series_never_positive(self, synthetic_bull_pnl):
        """Drawdown series should be ≤ 0 at every point."""
        equity = synthetic_bull_pnl.cumsum() + 1000
        _, dd = get_max_drawdown(equity, 1000.0)
        assert (dd <= 1e-9).all(), "Drawdown series has positive values"
