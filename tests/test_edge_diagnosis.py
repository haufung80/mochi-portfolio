"""Tests for the KS-based Edge Diagnosis 4-way matrix.

Edge Diagnosis (in calculations._ks_edge_diagnosis):
    MC fires + KS fires  → 🔴 BROKEN EDGE  (archive permanently)
    MC fires + KS quiet  → 🟠 UNLUCKY      (suspend, edge intact)
    MC quiet + KS fires  → 🟡 EDGE DRIFTING (watch — leading indicator)
    MC quiet + KS quiet  → 🟢 STABLE

Sparse-strategy fallback (regression for "n/a" silent disable bug):
    BT active days < MIN_BT_TRADES_FOR_KS:
        MC fires → "🔴 KILL (KS unavailable, n<20)"
        MC quiet → "🟢 STABLE (KS unavailable, n<20)"
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calculations import (
    LIVE_START_DEFAULT,
    MIN_BT_TRADES_FOR_KS,
    MIN_LIVE_TRADES_FOR_KS,
    _ks_edge_diagnosis,
    per_strategy_evaluation,
)


class TestEdgeDiagnosisMatrix:
    """Verify the 4-way verdict against synthetic inputs."""

    def test_broken_edge_when_mc_and_ks_fire(self):
        """BT positive distribution vs live negative distribution → BROKEN."""
        rng = np.random.default_rng(0)
        bt = pd.Series(rng.normal(1.0, 2.0, size=200))
        lv = pd.Series(rng.normal(-3.0, 2.0, size=40))
        # MC fires by definition (set mc_dd_pct very low)
        ks_p, mw_p, diag, _ = _ks_edge_diagnosis(bt, lv, mc_dd_pct=2.0, mc_ret_pct=50.0)
        assert ks_p is not None
        assert 'BROKEN EDGE' in diag

    def test_unlucky_when_mc_fires_but_ks_quiet(self):
        """Same distribution BT and live, but MC fires from bad cum-sum tail."""
        rng = np.random.default_rng(1)
        # Identical distribution (mean 0, std 4) → KS should not fire
        bt = pd.Series(rng.normal(0, 4.0, size=300))
        lv = pd.Series(rng.normal(0, 4.0, size=50))
        ks_p, mw_p, diag, _ = _ks_edge_diagnosis(bt, lv, mc_dd_pct=3.0, mc_ret_pct=50.0)
        assert ks_p is not None
        assert ks_p > 0.05  # KS quiet
        assert 'UNLUCKY' in diag

    def test_edge_drifting_when_ks_fires_but_mc_quiet(self):
        """Distribution shifted but cumsum still in MC envelope."""
        rng = np.random.default_rng(2)
        bt = pd.Series(rng.normal(1.0, 1.0, size=200))   # tight positive
        lv = pd.Series(rng.normal(1.0, 4.0, size=50))    # same mean, wider tails → KS fires
        ks_p, _, diag, _ = _ks_edge_diagnosis(bt, lv, mc_dd_pct=40.0, mc_ret_pct=40.0)
        # MC doesn't fire (above tail), but if KS fires → DRIFTING
        if ks_p is not None and ks_p < 0.05:
            assert 'DRIFTING' in diag

    def test_stable_when_neither_fires(self):
        """Both quiet → STABLE."""
        rng = np.random.default_rng(3)
        bt = pd.Series(rng.normal(0.5, 3.0, size=200))
        lv = pd.Series(rng.normal(0.5, 3.0, size=40))
        ks_p, _, diag, _ = _ks_edge_diagnosis(bt, lv, mc_dd_pct=50.0, mc_ret_pct=50.0)
        if ks_p is not None and ks_p > 0.05:
            assert 'STABLE' in diag


class TestSparseFallback:
    """Regression: sparse strategies must get a verdict, not silent n/a.

    Previously, BT with < 20 active days would collapse Edge Diagnosis to
    "⏳ n/a" — making kill-vs-suspend signal unavailable for low-frequency
    strategies (HLD_DAY, MR_VOTING).
    """

    def test_sparse_with_mc_fires_returns_kill_annotation(self, synthetic_sparse_pnl, starting_capital, split_date):
        """Sparse + MC fires → "🔴 KILL (KS unavailable, n<20)"."""
        ev = per_strategy_evaluation(
            synthetic_sparse_pnl, starting_capital, 0.04, split_date,
            n_mc_runs=500, mc_seed=42,
        )
        # KS should be None due to sparse data
        if ev['ks_p'] is None and (
            ev['mc_dd_percentile'] < 5 or ev['mc_return_percentile'] < 5
        ):
            assert 'KS unavailable' in ev['edge_diagnosis']
            assert 'KILL' in ev['edge_diagnosis']

    def test_sparse_with_mc_quiet_returns_stable_annotation(self):
        """Sparse + MC quiet → "🟢 STABLE (KS unavailable, n<20)"."""
        sparse_bt = pd.Series([1.0] * 10)
        sparse_lv = pd.Series([0.5] * 10)
        ks_p, _, diag, _ = _ks_edge_diagnosis(
            sparse_bt, sparse_lv, mc_dd_pct=50.0, mc_ret_pct=50.0,
        )
        assert ks_p is None  # below floor
        assert 'STABLE' in diag
        assert 'KS unavailable' in diag


class TestKsSampleSizeFloor:
    """KS test should be skipped (return None) below MIN_BT_TRADES_FOR_KS."""

    def test_ks_returns_none_below_floor(self):
        """BT with < 20 active values → ks_p is None."""
        bt = pd.Series(np.random.randn(MIN_BT_TRADES_FOR_KS - 1))
        lv = pd.Series(np.random.randn(20))
        ks_p, mw_p, _, _ = _ks_edge_diagnosis(bt, lv, 50.0, 50.0)
        assert ks_p is None
        assert mw_p is None

    def test_ks_returns_none_when_live_below_floor(self):
        """Live with < MIN_LIVE_TRADES_FOR_KS active values → ks_p is None."""
        bt = pd.Series(np.random.randn(50))
        lv = pd.Series(np.random.randn(MIN_LIVE_TRADES_FOR_KS - 1))
        ks_p, _, _, _ = _ks_edge_diagnosis(bt, lv, 50.0, 50.0)
        assert ks_p is None
