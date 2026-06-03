"""Tests for net_of_fees + regime_phase_split.

These two functions back the corrected portfolio review:
  - net_of_fees: TradingView 'Net P&L USDT' is GROSS (no commission column),
    so kill/keep decisions must subtract realistic round-trip costs first.
  - regime_phase_split: the "did it recover when its regime returned?" test —
    the most discriminating regime-vs-breakage signal — generalized to work
    with the traded ticker's OWN regime, not just BTC's.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from calculations import (
    DEFAULT_COST_BPS_RT,
    DEFAULT_SLIPPAGE_BPS,
    net_of_fees,
    regime_phase_split,
)


class TestNetOfFees:
    def test_basic_fee_subtraction(self):
        """fee = |notional| * (11+2)/10000 = 0.0013 * notional, subtracted from gross."""
        gross = [10.0, -5.0]
        notional = [1000.0, 2000.0]
        net, total_fee = net_of_fees(gross, notional, cost_bps_rt=11, slippage_bps=2)
        # fee_0 = 1000 * 0.0013 = 1.30 ; fee_1 = 2000 * 0.0013 = 2.60
        assert net[0] == pytest.approx(10.0 - 1.30, abs=1e-9)
        assert net[1] == pytest.approx(-5.0 - 2.60, abs=1e-9)
        assert total_fee == pytest.approx(3.90, abs=1e-9)

    def test_cost_charged_on_absolute_notional(self):
        """Short positions (negative notional) still incur positive fees."""
        net, fee = net_of_fees([0.0], [-5000.0], cost_bps_rt=10, slippage_bps=0)
        # fee = 5000 * 0.0010 = 5.0
        assert fee == pytest.approx(5.0, abs=1e-9)
        assert net[0] == pytest.approx(-5.0, abs=1e-9)

    def test_gross_positive_can_flip_net_negative(self):
        """A marginally-profitable trade can go net-negative after fees.

        This is THE reason the audit mattered — gross-positive strategies like
        DOUBLE_RSI SOL (+$2 gross) can be net-losers once fees are applied.
        """
        net, fee = net_of_fees([2.0], [5000.0], cost_bps_rt=11, slippage_bps=2)
        # fee = 5000 * 0.0013 = 6.50 → net = 2.0 - 6.50 = -4.50
        assert net[0] < 0, "gross-positive trade should flip net-negative under fees"
        assert net[0] == pytest.approx(-4.50, abs=1e-9)

    def test_empty_input(self):
        net, fee = net_of_fees([], [])
        assert len(net) == 0
        assert fee == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            net_of_fees([1.0, 2.0], [1000.0])

    def test_negative_bps_raises(self):
        with pytest.raises(ValueError):
            net_of_fees([1.0], [1000.0], cost_bps_rt=-1)

    def test_default_constants_applied(self):
        """When bps not given, uses DEFAULT_COST_BPS_RT + DEFAULT_SLIPPAGE_BPS."""
        net, fee = net_of_fees([0.0], [10000.0])
        expected = 10000.0 * (DEFAULT_COST_BPS_RT + DEFAULT_SLIPPAGE_BPS) / 10000.0
        assert fee == pytest.approx(expected, abs=1e-9)

    def test_total_fee_scales_with_trade_count(self):
        """More trades on same notional → proportionally more fees (the high-freq tax)."""
        _, fee_10 = net_of_fees([0.0] * 10, [1000.0] * 10)
        _, fee_50 = net_of_fees([0.0] * 50, [1000.0] * 50)
        assert fee_50 == pytest.approx(5 * fee_10, abs=1e-6)


class TestRegimePhaseSplit:
    def _regime(self, labels, start="2025-12-03"):
        idx = pd.date_range(start, periods=len(labels), freq="D")
        return pd.Series(labels, index=idx)

    def test_buckets_pnl_by_regime(self):
        """P&L is summed within each regime label."""
        idx = pd.date_range("2025-12-03", periods=4, freq="D")
        pnl = pd.Series([10.0, -5.0, 3.0, -2.0], index=idx)
        reg = pd.Series(["Bull", "Bull", "Bear", "Bear"], index=idx)
        out = regime_phase_split(pnl, reg)
        assert out["Bull"]["pnl"] == pytest.approx(5.0)   # 10 - 5
        assert out["Bear"]["pnl"] == pytest.approx(1.0)   # 3 - 2
        assert out["Bull"]["days"] == 2
        assert out["Bear"]["days"] == 2

    def test_per_day_average(self):
        idx = pd.date_range("2025-12-03", periods=2, freq="D")
        pnl = pd.Series([10.0, 20.0], index=idx)
        reg = pd.Series(["Bull", "Bull"], index=idx)
        out = regime_phase_split(pnl, reg)
        assert out["Bull"]["per_day"] == pytest.approx(15.0)

    def test_recovery_signal_bear_then_bull(self):
        """The actual use case: strategy loses in bear, check if it recovers in bull.

        A regime VICTIM makes money back in bull; a BROKEN strategy keeps bleeding.
        """
        idx = pd.date_range("2025-12-03", periods=6, freq="D")
        # Loses in bear (first 3 days), recovers in bull (last 3) → regime victim
        pnl_victim = pd.Series([-5.0, -5.0, -5.0, 6.0, 6.0, 6.0], index=idx)
        reg = pd.Series(["Bear", "Bear", "Bear", "Bull", "Bull", "Bull"], index=idx)
        out = regime_phase_split(pnl_victim, reg)
        assert out["Bear"]["pnl"] < 0
        assert out["Bull"]["pnl"] > 0, "regime victim should recover in bull"

        # Keeps bleeding in bull → broken, not regime
        pnl_broken = pd.Series([-5.0, -5.0, -5.0, -4.0, -4.0, -4.0], index=idx)
        out_b = regime_phase_split(pnl_broken, reg)
        assert out_b["Bull"]["pnl"] < 0, "broken strategy keeps losing even in bull"

    def test_reindex_ffill_misaligned_regime(self):
        """Regime series on different/sparser dates is forward-filled onto pnl index."""
        pnl_idx = pd.date_range("2025-12-03", periods=5, freq="D")
        pnl = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=pnl_idx)
        # Regime only labeled on day 0 and day 3
        reg = pd.Series(["Bull", "Bear"],
                        index=[pnl_idx[0], pnl_idx[3]])
        out = regime_phase_split(pnl, reg)
        # Days 0-2 ffill Bull (3 days), days 3-4 Bear (2 days)
        assert out["Bull"]["days"] == 3
        assert out["Bear"]["days"] == 2

    def test_empty_inputs(self):
        assert regime_phase_split(pd.Series([], dtype=float), pd.Series([], dtype=float)) == {}
        idx = pd.date_range("2025-12-03", periods=2, freq="D")
        assert regime_phase_split(pd.Series([1.0, 2.0], index=idx), None) == {}
