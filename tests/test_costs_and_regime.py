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
    net_live_pnl_from_csv,
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

    def test_cost_constants_are_single_source_of_truth(self):
        """Guard: the canonical cost constants must stay the documented Binance
        values. If these drift, the sidebar (which reads them) and every net
        figure drift together — but a silent change here would desync intent.
        """
        assert DEFAULT_COST_BPS_RT == 10.0, "Binance taker 5bps×2 = 10bps RT"
        assert DEFAULT_SLIPPAGE_BPS == 2.0
        from calculations import DEFAULT_FUNDING_BPS_PER_DAY
        assert DEFAULT_FUNDING_BPS_PER_DAY == 0.5

    def test_amortized_and_pertrade_models_reconcile(self):
        """The vol-targeting amortized model (tpy*cost*pos/365 summed over a year)
        and the per-trade model (Σ cost*|notional|) must agree at the same scale.

        This is the cross-dashboard consistency guarantee: Portfolio-tab metrics
        (amortized) and Live-table net (per-trade) use the same effective cost.
        """
        cost_pct = (DEFAULT_COST_BPS_RT + DEFAULT_SLIPPAGE_BPS) / 10000.0
        # A strategy that trades 100x/year, $1000 notional each, over exactly 1 year
        tpy, notional, n_trades = 100, 1000.0, 100
        amortized_annual = tpy * cost_pct * notional
        _, pertrade_total = net_of_fees([0.0] * n_trades, [notional] * n_trades)
        # Same scale, same year → must match exactly
        assert amortized_annual == pytest.approx(pertrade_total, rel=1e-9)

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


class TestNetLivePnlFromCsv:
    """End-to-end: read a real-format CSV → net-of-fees live summary.

    Uses the tz_aware_tv_csv fixture (from conftest) which writes a TradingView
    export with notional and tz-aware timestamps — covers both the cost calc
    and the tz handling in one path.
    """

    def test_reads_and_nets_live_segment(self, tz_aware_tv_csv):
        # Fixture has 30 trades every 10 days from 2024-01-01, notional 500.
        # Split mid-way so only later exits count as "live".
        out = net_live_pnl_from_csv(tz_aware_tv_csv, pd.Timestamp("2024-06-01"))
        assert out['n_trades'] > 0, "should find live exits after split"
        # fee = n_trades * 500 * (DEFAULT_COST_BPS_RT + DEFAULT_SLIPPAGE_BPS)/10000
        # — derive from constants so this test tracks the single source of truth.
        bps = (DEFAULT_COST_BPS_RT + DEFAULT_SLIPPAGE_BPS) / 10000.0
        assert out['fees'] == pytest.approx(out['n_trades'] * 500 * bps, abs=1e-6)
        # net = gross - fees, always
        assert out['net'] == pytest.approx(out['gross'] - out['fees'], abs=1e-6)

    def test_missing_file_returns_zeros(self, tmp_path):
        out = net_live_pnl_from_csv(tmp_path / "does_not_exist.csv", pd.Timestamp("2025-12-03"))
        assert out == {'gross': 0.0, 'fees': 0.0, 'net': 0.0, 'n_trades': 0}

    def test_split_after_all_trades_returns_zeros(self, tz_aware_tv_csv):
        """Split date after every trade → no live segment → zeros, no crash."""
        out = net_live_pnl_from_csv(tz_aware_tv_csv, pd.Timestamp("2030-01-01"))
        assert out['n_trades'] == 0
        assert out['net'] == 0.0
