"""Behavioural INVARIANT tests for the calculation engine.

These assert mathematical PROPERTIES that must hold for *every* input — not
specific example outputs. They exist because every numerical bug found while
hardening this engine (impossible 92% MaxDD, rolling_calmar complex-number
crash, gross-vs-net cost drift, MC seed not propagating) was an invariant
violation that passed the example-based suite. Properties catch the class of
bug examples can't.

Four families:
  • BOUNDS        — outputs stay in their mathematically valid range
  • SCALE         — metamorphic: scaling inputs scales outputs predictably
  • MONOTONICITY  — directional: more cost → less net, more vol → more size
  • CONSERVATION  — partitions/aggregations preserve totals

MAINTENANCE CONTRACT (see tests/README.md):
  Every NEW function in calculations.py that returns a ratio, a bounded
  quantity, an aggregate, or something with a known scaling law gets an
  invariant here. Every bug FIX gets the invariant that would have caught it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import calculations as C


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _equity_from_pnl(pnl: np.ndarray, cap: float) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(pnl), freq="D")
    return pd.Series(np.cumsum(pnl) + cap, index=idx)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _synthetic_portfolio():
    """Minimal (plot_data, metrics_df) accepted by mc_vol_targeted_allocation."""
    idx = pd.date_range("2024-01-01", periods=300, freq="D")
    cols = {}
    for i, name in enumerate(["STRAT_A_BINANCE_BTCUSDT", "STRAT_B_BINANCE_ETHUSDT"]):
        cols[name] = pd.Series(_rng(20 + i).normal(0.3, 5.0, 300), index=idx)
    plot_data = pd.DataFrame(cols)
    metrics_df = pd.DataFrame(
        {"Avg Position $": [100.0, 120.0], "Trades/Yr": [50, 60]},
        index=list(cols.keys()))
    return plot_data, metrics_df


def _write_tv_csv(path, seed: int, n: int = 60):
    """Write a minimal TradingView-style export: alternating Entry/Exit rows."""
    rng = _rng(seed)
    lines = ["Trade #,Type,Date/Time,Net P&L USDT,Signal,Cumulative P&L,Position size (value)"]
    start = pd.Timestamp("2024-02-01")
    for i in range(n):
        dt = (start + pd.Timedelta(days=i * 7)).strftime("%Y-%m-%d %H:%M:%S")
        pnl = float(rng.normal(1.0, 8.0))
        lines.append(f"{i+1},Entry long,{dt},,buy,,500")
        lines.append(f"{i+1},Exit long,{dt},{pnl:.2f},sell,,500")
    path.write_text("\n".join(lines))


@pytest.fixture
def positive_equity() -> pd.Series:
    """A non-negative equity curve with a real peak and drawdown."""
    pnl = _rng(10).normal(0.4, 6.0, size=300)
    return _equity_from_pnl(pnl, 1000.0)


# ===========================================================================
# BOUNDS — outputs must stay inside their valid mathematical range
# ===========================================================================

class TestBounds:
    def test_max_drawdown_path_pct_in_unit_interval(self):
        """mdd_pct ∈ [0, 1] and mdd_dollars ≥ 0 for any non-negative path."""
        for seed in range(25):
            path = np.cumsum(_rng(seed).normal(0.3, 5.0, size=200)) + 1000.0
            path = np.maximum(path, 1.0)  # keep strictly positive
            mdd_d, mdd_p = C.max_drawdown_path(path)
            assert 0.0 <= mdd_p <= 1.0, f"seed {seed}: mdd_pct={mdd_p}"
            assert mdd_d >= 0.0

    def test_get_max_drawdown_pct_bounded(self, positive_equity):
        """get_max_drawdown returns a fraction in [-1, 0] for positive equity."""
        mdd, dd_series = C.get_max_drawdown(positive_equity, 1000.0)
        assert -1.0 <= mdd <= 0.0
        assert (dd_series <= 1e-12).all()  # drawdown is never positive

    def test_get_max_drawdown_impossible_value_regression(self):
        """REGRESSION: the 92%+ MaxDD bug. A $50 drop from a $1500 peak is a
        3.3% drawdown, NOT a fraction of starting capital. Peak-relative only."""
        eq = pd.Series(
            [1000, 1500, 1450, 1400, 1450],
            index=pd.date_range("2024-01-01", periods=5, freq="D"),
        )
        mdd, _ = C.get_max_drawdown(eq, 1000.0)
        assert mdd == pytest.approx(-100 / 1500, abs=1e-9)  # 6.67%, not 10%

    def test_profit_factor_nonnegative(self):
        for seed in range(20):
            pnl = _rng(seed).normal(0.0, 5.0, size=100)
            pf = C.profit_factor(pnl)
            assert pf >= 0.0  # may be +inf (no losses), never negative

    def test_monte_carlo_outputs_bounded(self):
        pnl = _rng(0).normal(0.5, 5.0, size=200)
        mc = C.monte_carlo(pnl, start_equity=1000.0, ruin_equity=600.0,
                           trades_per_year=100, n_runs=200, seed=1, block_len=5)
        assert ((mc['mdd_pct'] >= 0.0) & (mc['mdd_pct'] <= 1.0)).all()
        assert set(np.unique(mc['ruined'])).issubset({True, False})
        assert np.isfinite(mc['return']).all()
        assert len(mc['return']) == 200

    def test_find_max_safe_leverage_within_search_bounds(self):
        rets = _rng(2).normal(0.01, 0.05, size=200)  # per-trade pct returns
        lev, ror = C.find_max_safe_leverage(
            rets, trades_per_year=100, target_ror=0.10,
            n_runs=200, leverage_min=0.01, leverage_max=5.0, seed=3)
        assert 0.01 <= lev <= 5.0
        assert 0.0 <= ror <= 1.0

    def test_deflated_sharpe_psr_is_probability(self):
        for sr in (0.0, 0.5, 1.5, 3.0):
            out = C.deflated_sharpe(sr, n_trials=38, n_obs=500)
            assert 0.0 <= out['psr'] <= 1.0

    def test_classify_regimes_labels_and_alignment(self):
        idx = pd.date_range("2024-01-01", periods=300, freq="D")
        price = pd.Series(np.linspace(100, 200, 300) + _rng(7).normal(0, 3, 300),
                          index=idx)
        reg = C.classify_regimes(price)
        assert set(reg.unique()).issubset({"Bull", "Bear", "Chop"})
        assert len(reg) == len(price)
        assert (reg.index == price.index).all()


# ===========================================================================
# SCALE INVARIANCE (metamorphic) — scaling inputs scales outputs predictably
# ===========================================================================

class TestScaleInvariance:
    @pytest.mark.parametrize("k", [0.1, 2.0, 37.0, 1000.0])
    def test_mdd_pct_invariant_under_path_scaling(self, k):
        """A drawdown's PERCENTAGE doesn't care about the units of the path."""
        path = np.maximum(np.cumsum(_rng(4).normal(0.3, 5.0, 200)) + 1000.0, 1.0)
        _, base_pct = C.max_drawdown_path(path)
        d_k, pct_k = C.max_drawdown_path(path * k)
        assert pct_k == pytest.approx(base_pct, rel=1e-9)

    @pytest.mark.parametrize("k", [0.5, 3.0, 100.0])
    def test_get_max_drawdown_invariant_when_equity_and_cap_coscale(self, k):
        """REGRESSION-class: scaling equity AND starting capital together must
        leave MaxDD% unchanged. The old (÷ starting_capital) bug failed this."""
        pnl = _rng(8).normal(0.4, 6.0, 250)
        eq = _equity_from_pnl(pnl, 1000.0)
        base, _ = C.get_max_drawdown(eq, 1000.0)
        scaled, _ = C.get_max_drawdown(eq * k, 1000.0 * k)
        assert scaled == pytest.approx(base, rel=1e-9)

    @pytest.mark.parametrize("k", [0.25, 4.0, 50.0])
    def test_sharpe_sortino_leverage_invariant(self, k):
        """With rfr=0, Sharpe & Sortino are invariant to position scaling —
        the textbook 'leverage doesn't change risk-adjusted return' property."""
        rets = pd.Series(_rng(11).normal(0.001, 0.02, 300))
        s0, so0 = C.get_risk_ratios(rets, 0.0)
        sk, sok = C.get_risk_ratios(rets * k, 0.0)
        assert sk == pytest.approx(s0, rel=1e-9)
        assert sok == pytest.approx(so0, rel=1e-9)

    @pytest.mark.parametrize("k", [0.3, 5.0])
    def test_profit_factor_scale_invariant(self, k):
        pnl = _rng(9).normal(0.2, 5.0, 200)
        assert C.profit_factor(pnl * k) == pytest.approx(C.profit_factor(pnl), rel=1e-9)

    @pytest.mark.parametrize("k", [0.1, 10.0])
    def test_cagr_invariant_when_start_and_end_coscale(self, k):
        assert C.get_cagr(1000 * k, 1500 * k, 365) == pytest.approx(
            C.get_cagr(1000, 1500, 365), rel=1e-12)


# ===========================================================================
# MONOTONICITY — outputs move in the correct direction with their drivers
# ===========================================================================

class TestMonotonicity:
    def test_net_of_fees_never_exceeds_gross(self):
        gross = _rng(1).normal(0.0, 10.0, 100)
        notional = np.abs(_rng(2).normal(500, 50, 100))
        net, fee = C.net_of_fees(gross, notional, 10.0, 2.0)
        assert (net <= gross + 1e-12).all()
        assert fee >= 0.0

    def test_net_of_fees_monotonic_in_cost(self):
        """Higher cost → lower (or equal) net. Never higher."""
        gross = _rng(1).normal(0.5, 10.0, 100)
        notional = np.abs(_rng(2).normal(500, 50, 100))
        net_lo, _ = C.net_of_fees(gross, notional, 5.0, 1.0)
        net_hi, _ = C.net_of_fees(gross, notional, 25.0, 5.0)
        assert net_hi.sum() <= net_lo.sum() + 1e-9

    def test_net_of_fees_zero_cost_is_identity(self):
        gross = _rng(1).normal(0.5, 10.0, 50)
        notional = np.abs(_rng(2).normal(500, 50, 50))
        net, fee = C.net_of_fees(gross, notional, 0.0, 0.0)
        assert net == pytest.approx(gross)
        assert fee == 0.0

    def test_safe_leverage_monotonic_in_target_ror(self):
        """More ruin tolerance → at least as much safe leverage (same seed)."""
        rets = _rng(5).normal(0.005, 0.04, 200)
        lev_lo, _ = C.find_max_safe_leverage(rets, 100, target_ror=0.02,
                                             n_runs=300, leverage_max=5.0, seed=42)
        lev_hi, _ = C.find_max_safe_leverage(rets, 100, target_ror=0.40,
                                             n_runs=300, leverage_max=5.0, seed=42)
        assert lev_hi >= lev_lo - 1e-9

    def test_vt_positions_scale_with_target_vol(self):
        """mc_vol_targeted_allocation: total leveraged notional is proportional
        to target_portfolio_vol (portfolio_scale = target / realized_pre)."""
        plot_data, metrics_df = _synthetic_portfolio()
        common = dict(plot_data=plot_data, metrics_df=metrics_df, total_cap=1000.0,
                      n_runs=200, block_len=5, seed=42, max_leverage_cap=3.0)
        vt_lo = C.mc_vol_targeted_allocation(target_portfolio_vol=0.10, **common)
        vt_hi = C.mc_vol_targeted_allocation(target_portfolio_vol=0.30, **common)
        tot_lo = sum(vt_lo['position_sizes'].values())
        tot_hi = sum(vt_hi['position_sizes'].values())
        assert tot_hi > tot_lo
        # 3× the vol target → ~3× the notional (exact up to per-strat leverage caps)
        assert tot_hi / tot_lo == pytest.approx(3.0, rel=0.05)


# ===========================================================================
# CONSERVATION — partitions and aggregations preserve totals
# ===========================================================================

class TestConservation:
    def test_regime_phase_split_conserves_pnl_and_days(self):
        """Σ per-regime P&L == total P&L; Σ per-regime days == total days."""
        idx = pd.date_range("2024-01-01", periods=300, freq="D")
        pnl = pd.Series(_rng(6).normal(0.3, 5.0, 300), index=idx)
        regimes = pd.Series(
            np.where(np.arange(300) % 3 == 0, "Bull",
                     np.where(np.arange(300) % 3 == 1, "Bear", "Chop")),
            index=idx)
        split = C.regime_phase_split(pnl, regimes)
        assert sum(v["pnl"] for v in split.values()) == pytest.approx(float(pnl.sum()))
        assert sum(v["days"] for v in split.values()) == len(pnl)

    def test_portfolio_equity_equals_sum_of_strategies(self, tmp_path, monkeypatch):
        """process_portfolio: Portfolio Equity == total_cap + cumulative row-sum
        of every strategy column. Real aggregation path, network-free (BTC fetch
        monkeypatched to empty so the test never touches the internet)."""
        monkeypatch.setattr(C, "fetch_btc_daily", lambda *a, **k: pd.DataFrame())
        _write_tv_csv(tmp_path / "AAA_BINANCE_BTCUSDT_2026-05-23.csv", seed=31)
        _write_tv_csv(tmp_path / "BBB_BINANCE_ETHUSDT_2026-05-23.csv", seed=32)

        _, _, plot_data, _ = C.process_portfolio(
            str(tmp_path), total_cap=2000.0, risk_free_rate=0.04,
            oos_start="2024-01-01", oos_end="2026-05-22")

        reserved_cols = {"Portfolio Equity", "Portfolio DD", "Portfolio Daily P&L",
                         "B&H BTC Equity", "Portfolio Load"}
        strat_cols = [c for c in plot_data.columns if c not in reserved_cols]
        assert strat_cols, "expected synthetic strategies to load"
        reconstructed = plot_data[strat_cols].sum(axis=1).cumsum() + 2000.0
        assert np.allclose(reconstructed.values,
                           plot_data["Portfolio Equity"].values, atol=1e-6)
