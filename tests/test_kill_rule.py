"""Tests for the dual-tail kill rule + sample-size floor.

The kill rule:
    KILL  if  (MC_DD_%ile ≤ MC_TAIL_PCT  OR  MC_Ret_%ile ≤ MC_TAIL_PCT)
              AND live_trades ≥ MIN_LIVE_TRADES
    WARN  if  (MC_DD_%ile ≤ MC_WARN_PCT  OR  MC_Ret_%ile ≤ MC_WARN_PCT)
              AND live_trades ≥ MIN_LIVE_TRADES
    INCUBATE  if live_trades < MIN_LIVE_TRADES
    KEEP  otherwise
"""
from __future__ import annotations

import pytest

# _kill_verdict is private but tested directly because it's the decision core
from calculations import (
    MC_TAIL_PCT,
    MC_WARN_PCT,
    MIN_LIVE_TRADES,
    _kill_verdict,
)


class TestKillRuleMatrix:
    """Exhaustive matrix of (mc_dd, mc_ret, live_trades) → verdict."""

    def test_dd_below_tail_fires_kill(self):
        """MC DD %ile ≤ MC_TAIL_PCT (5) + enough trades → KILL."""
        verdict, _ = _kill_verdict(mc_dd_pct=3.0, mc_ret_pct=50.0, live_trades=25)
        assert 'KILL' in verdict
        assert 'DD' in verdict

    def test_return_below_tail_fires_kill_slow_bleed(self):
        """MC Return %ile ≤ MC_TAIL_PCT + enough trades → KILL (slow bleed)."""
        verdict, _ = _kill_verdict(mc_dd_pct=50.0, mc_ret_pct=3.0, live_trades=25)
        assert 'KILL' in verdict
        assert 'slow bleed' in verdict or 'Return' in verdict

    def test_both_below_tail_fires_kill_combined(self):
        """Both %iles below tail → KILL combined."""
        verdict, _ = _kill_verdict(mc_dd_pct=2.0, mc_ret_pct=1.0, live_trades=25)
        assert 'KILL' in verdict

    def test_warn_when_in_warn_band(self):
        """%ile between MC_TAIL_PCT and MC_WARN_PCT → WARN."""
        verdict, _ = _kill_verdict(mc_dd_pct=10.0, mc_ret_pct=50.0, live_trades=25)
        assert 'WARN' in verdict

    def test_keep_when_above_warn(self):
        """%iles above warn → KEEP."""
        verdict, _ = _kill_verdict(mc_dd_pct=50.0, mc_ret_pct=50.0, live_trades=25)
        assert 'KEEP' in verdict

    def test_incubating_below_min_trades(self):
        """live_trades < MIN_LIVE_TRADES → INCUBATING regardless of %iles."""
        verdict, _ = _kill_verdict(mc_dd_pct=1.0, mc_ret_pct=1.0, live_trades=10)
        assert 'Incubating' in verdict
        # And the verdict mentions the floor count
        assert str(MIN_LIVE_TRADES) in verdict

    def test_insufficient_data_when_pctiles_none(self):
        """If MC %iles missing (no live data) → Insufficient data."""
        verdict, _ = _kill_verdict(mc_dd_pct=None, mc_ret_pct=None, live_trades=25)
        assert 'Insufficient' in verdict


class TestKillRuleConsistency:
    """Verify the rule is monotonic and well-ordered."""

    def test_monotonic_severity_in_dd_pctile(self):
        """As DD %ile decreases from KEEP region → WARN → KILL, severity rises."""

        def severity(verdict: str) -> int:
            if 'KILL' in verdict:
                return 3
            if 'WARN' in verdict:
                return 2
            if 'KEEP' in verdict:
                return 1
            return 0

        keep_v, _ = _kill_verdict(50.0, 50.0, 25)
        warn_v, _ = _kill_verdict(10.0, 50.0, 25)
        kill_v, _ = _kill_verdict(2.0, 50.0, 25)
        assert severity(keep_v) < severity(warn_v) < severity(kill_v)

    def test_min_trades_floor_takes_precedence_over_severe_pctile(self):
        """Even at MC %ile = 0%, < MIN_LIVE_TRADES → INCUBATING, NOT KILL.

        The sample-size guard must override the percentile signal — otherwise
        a strategy with 3 noisy live trades would get killed.
        """
        verdict, _ = _kill_verdict(mc_dd_pct=0.0, mc_ret_pct=0.0, live_trades=5)
        assert 'Incubating' in verdict, (
            f"With only 5 live active days, expected INCUBATING but got {verdict}. "
            "Min-trades floor bypass would let small samples kill strategies."
        )
