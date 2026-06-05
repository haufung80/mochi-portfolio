"""Property-based fuzzing with Hypothesis.

Hypothesis generates thousands of inputs — including the degenerate ones a
human never writes (all-zeros, single element, all-negative, huge/tiny values,
negative equity) — and shrinks any failure to a minimal counterexample.

This is aimed squarely at the bug class that bit this engine: functions that
silently produced garbage or RAISED on inputs the example suite never tried
(rolling_calmar's complex-number crash on negative equity; the impossible
MaxDD; net_of_fees on mismatched/empty arrays). The contract here is mostly
"never raises on any real-valued input, and bounded outputs stay bounded."

Run just these:  pytest tests/test_property_fuzz.py
If Hypothesis isn't installed the whole module skips (it's a dev-only dep).

MAINTENANCE CONTRACT (see tests/README.md): any calculation that ingests a
user/CSV-derived array and could meet a degenerate input gets a fuzz test that
asserts (1) it never raises and (2) its documented bounds hold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

import calculations as C

# Keep fuzz runs snappy in CI; raise locally with --hypothesis-seed for depth.
settings.register_profile("fast", max_examples=150, deadline=None)
settings.load_profile("fast")

# Finite float arrays of varied length/scale — the raw material of P&L curves.
finite_floats = st.floats(min_value=-1e6, max_value=1e6,
                          allow_nan=False, allow_infinity=False)
pnl_arrays = arrays(dtype=np.float64,
                    shape=st.integers(min_value=0, max_value=400),
                    elements=finite_floats)
nonempty_pnl = arrays(dtype=np.float64,
                      shape=st.integers(min_value=1, max_value=400),
                      elements=finite_floats)


def _series(arr: np.ndarray) -> pd.Series:
    return pd.Series(arr, index=pd.date_range("2024-01-01", periods=len(arr), freq="D"))


# ===========================================================================
# "Never raises" — robustness on arbitrary real-valued input
# ===========================================================================

class TestNeverRaises:
    @given(pnl=nonempty_pnl, cap=st.floats(1.0, 1e6))
    def test_get_max_drawdown_total(self, pnl, cap):
        mdd, dd = C.get_max_drawdown(_series(np.cumsum(pnl) + cap), cap)
        assert np.isfinite(mdd)
        assert mdd <= 1e-9  # drawdown is non-positive

    @given(pnl=nonempty_pnl, cap=st.floats(1.0, 1e6))
    def test_rolling_calmar_never_complex_or_raises(self, pnl, cap):
        """REGRESSION: rolling_calmar threw 'float() argument ... complex' when
        a trailing window's equity went negative (negative base ^ fractional
        power). The regression guard is that this RETURNS (does not raise) for
        ANY P&L incl. deeply-negative curves — reaching the asserts below means
        no TypeError fired — and yields a real float Series with finite values."""
        s = _series(pnl)
        out = C.rolling_calmar(s, starting_capital=cap, window=30)
        assert isinstance(out, pd.Series)
        assert out.dtype.kind == "f"  # real float dtype, never complex (the bug widened to complex)
        assert np.isfinite(out.dropna().values).all()

    @given(pnl=pnl_arrays)
    def test_max_drawdown_path_total(self, pnl):
        path = np.cumsum(pnl) + 1000.0
        mdd_d, mdd_p = C.max_drawdown_path(path)
        assert np.isfinite(mdd_d) and np.isfinite(mdd_p)

    @given(rets=nonempty_pnl, rfr=st.floats(0.0, 0.2))
    def test_get_risk_ratios_total(self, rets, rfr):
        s, so = C.get_risk_ratios(_series(rets) / 1e4, rfr)
        assert np.isfinite(s) and np.isfinite(so)

    @given(pnl=pnl_arrays)
    def test_profit_factor_total(self, pnl):
        pf = C.profit_factor(pnl)
        assert pf >= 0.0  # +inf allowed (no losses); never negative, never nan
        assert not np.isnan(pf)

    @given(rets=nonempty_pnl)
    def test_kelly_criterion_never_nan(self, rets):
        """REGRESSION: kelly returned NaN for a single observation — pandas
        .var() uses ddof=1, so one point yields NaN that slipped past the
        `== 0` guard. Must be finite for any real-valued input."""
        k = C.kelly_criterion(_series(rets) / 1e3)
        assert np.isfinite(k)


# ===========================================================================
# Bounds hold for ALL generated inputs (not just hand-picked examples)
# ===========================================================================

class TestBoundsHold:
    @given(pnl=nonempty_pnl)
    def test_positive_path_mdd_in_unit_interval(self, pnl):
        """For any strictly-positive equity path, MaxDD% ∈ [0, 1]."""
        path = np.abs(np.cumsum(pnl)) + 1000.0  # guaranteed > 0
        _, mdd_p = C.max_drawdown_path(path)
        assert 0.0 <= mdd_p <= 1.0 + 1e-9

    @given(
        gross=nonempty_pnl,
        cost=st.floats(0.0, 100.0),
        slip=st.floats(0.0, 50.0),
        notional_scale=st.floats(0.0, 1e4),
    )
    def test_net_of_fees_never_exceeds_gross(self, gross, cost, slip, notional_scale):
        """net ≤ gross and fee ≥ 0 for any non-negative cost/notional."""
        notional = np.full_like(gross, notional_scale)
        net, fee = C.net_of_fees(gross, notional, cost, slip)
        assert (net <= gross + 1e-9).all()
        assert fee >= 0.0

    @given(cost=st.floats(-100.0, -0.01), notional_scale=st.floats(0.0, 1e4))
    def test_net_of_fees_rejects_negative_cost(self, cost, notional_scale):
        """Negative cost is nonsensical — must raise, not silently add money."""
        gross = np.array([1.0, -2.0, 3.0])
        notional = np.full_like(gross, notional_scale)
        with pytest.raises(ValueError):
            C.net_of_fees(gross, notional, cost, 0.0)

    def test_net_of_fees_rejects_negative_cost_even_when_empty(self):
        """REGRESSION: negative-cost validation must run BEFORE the empty-array
        early return, so a negative cost raises regardless of input size (the
        empty case previously returned (array([]), 0.0) and bypassed the guard)."""
        for g, n in [(np.array([1.0, -2.0, 3.0]), np.full(3, 100.0)),
                     (np.array([]), np.array([]))]:
            with pytest.raises(ValueError):
                C.net_of_fees(g, n, cost_bps_rt=-5.0)
            with pytest.raises(ValueError):
                C.net_of_fees(g, n, slippage_bps=-1.0)

    @given(
        sr=st.floats(-2.0, 6.0),
        n_trials=st.integers(1, 500),
        n_obs=st.integers(10, 5000),
    )
    def test_deflated_sharpe_psr_is_probability(self, sr, n_trials, n_obs):
        out = C.deflated_sharpe(sr, n_trials, n_obs)
        assert 0.0 <= out["psr"] <= 1.0


# ===========================================================================
# Degenerate / empty inputs are handled gracefully (no crash, sane defaults)
# ===========================================================================

class TestDegenerate:
    def test_empty_inputs_dont_crash(self):
        empty = pd.Series(dtype=float)
        assert C.get_max_drawdown(empty, 1000.0)[0] == 0.0
        assert C.profit_factor(np.array([])) >= 0.0
        net, fee = C.net_of_fees(np.array([]), np.array([]))
        assert len(net) == 0 and fee == 0.0
        assert C.max_drawdown_path(np.array([])) == (0.0, 0.0)

    @given(val=finite_floats)
    def test_single_observation(self, val):
        s = _series(np.array([val]))
        mdd, _ = C.get_max_drawdown(s, 1000.0)
        assert np.isfinite(mdd)

    def test_all_zeros(self):
        z = np.zeros(100)
        assert C.profit_factor(z) >= 0.0
        s, so = C.get_risk_ratios(_series(z), 0.0)
        assert s == 0.0 and so == 0.0

    def test_single_and_empty_obs_return_zero_not_nan(self):
        """The <2-observation guards: Sharpe/Sortino and Kelly must degrade to
        0.0, never NaN (pandas std()/var() use ddof=1 → NaN on one point)."""
        assert C.get_risk_ratios(pd.Series([0.01]), 0.0) == (0.0, 0.0)
        assert C.kelly_criterion(pd.Series([0.01])) == 0.0
        assert C.kelly_criterion(pd.Series(dtype=float)) == 0.0

    def test_rolling_calmar_negative_equity_does_not_raise(self):
        """The exact crash shape: a curve whose trailing-window equity dives below
        zero must NOT raise (was 'TypeError: float() argument ... complex')."""
        idx = pd.date_range("2024-01-01", periods=60, freq="D")
        pnl = pd.Series([-50.0] * 60, index=idx)  # equity goes deeply negative
        out = C.rolling_calmar(pnl, starting_capital=100.0, window=30)  # must not raise
        assert isinstance(out, pd.Series)
        assert np.isfinite(out.dropna().values).all()  # all-NaN gap is fine; never inf/complex
