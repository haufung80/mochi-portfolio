"""Guard tests: every default is single-sourced from calculations constants.

This locks the class of bug we hunted: a value defined in calculations.py that
silently disagrees with the app.py sidebar default (cost was 11 vs 10; regime
was 30/±5% in classify_regimes vs 60/±10% everywhere else; MC was 1000/5 in
functions vs 5000/30 in the sidebar). If any function default drifts from the
canonical constant, these fail.
"""
from __future__ import annotations

import inspect

import calculations as C


class TestConstantsExist:
    """All canonical constants are defined and have sane types."""

    def test_mc_defaults_are_data_driven_optima(self):
        """Lock the measured-optimal MC params so they don't drift back.

        RUNS≥2000: below this the kill-%ile sampling noise (±0.63 at 1000 runs)
        is large enough to flip borderline strategies across the 5% threshold;
        at 2000 it collapses to ±0.22. BLOCK_LEN≈10: Politis-White n^(1/3)
        optimum for ~1300 active daily obs (near-IID P&L, lag-1 ACF≈+0.13).
        """
        assert C.MC_DEFAULT_RUNS >= 2000, "kill-%ile unstable below 2000 runs"
        assert 8 <= C.MC_DEFAULT_BLOCK_LEN <= 12, "block should be ~n^(1/3)≈10"

    def test_portfolio_mc_uses_long_blocks(self):
        """Portfolio Forward-Risk MC must use LONG blocks (≈60d), separate from
        the per-strategy block-10. Short blocks dice multi-month drawdown
        sequences and understate the portfolio ruin/MaxDD tail (history contains
        a fold-level −38.7% window that block-10 resampling barely reproduces).
        """
        assert C.MC_PORTFOLIO_BLOCK_LEN >= 40, "portfolio MC needs regime-length blocks"
        assert C.MC_PORTFOLIO_BLOCK_LEN != C.MC_DEFAULT_BLOCK_LEN, \
            "portfolio block is intentionally distinct from per-strategy block"

    def test_all_constants_present(self):
        for name in [
            'DEFAULT_COST_BPS_RT', 'DEFAULT_SLIPPAGE_BPS', 'DEFAULT_FUNDING_BPS_PER_DAY',
            'MC_TAIL_PCT', 'MC_WARN_PCT', 'MIN_LIVE_TRADES', 'MIN_LIVE_TRADES_SOFT',
            'KS_ALPHA', 'MIN_BT_TRADES_FOR_KS', 'MIN_LIVE_TRADES_FOR_KS',
            'ROLLING_WINDOW_DAYS', 'TRADING_DAYS_PER_YEAR', 'ANNUALIZATION_FACTOR',
            'CALENDAR_DAYS_PER_YEAR', 'MC_DEFAULT_RUNS', 'MC_DEFAULT_BLOCK_LEN',
            'MC_DEFAULT_SEED', 'DEFAULT_RFR', 'DEFAULT_CAPITAL',
            'REGIME_DEFAULT_LOOKBACK', 'REGIME_DEFAULT_BULL_THR', 'REGIME_DEFAULT_BEAR_THR',
            'VT_DEFAULT_TARGET_ROR', 'VT_DEFAULT_RUIN_FRAC', 'VT_DEFAULT_MAX_LEV',
            'VT_DEFAULT_PORT_VOL', 'VT_DEFAULT_N_RUNS', 'LIVE_START_DEFAULT',
        ]:
            assert hasattr(C, name), f"missing canonical constant {name}"


def _default(fn, param):
    """Return the default value of `param` in function `fn`."""
    return inspect.signature(fn).parameters[param].default


class TestFunctionDefaultsMatchConstants:
    """Function signature defaults must equal the canonical constants — not
    re-typed literals that can drift.
    """

    def test_classify_regimes_uses_regime_constants(self):
        assert _default(C.classify_regimes, 'lookback') == C.REGIME_DEFAULT_LOOKBACK
        assert _default(C.classify_regimes, 'bull_threshold') == C.REGIME_DEFAULT_BULL_THR
        assert _default(C.classify_regimes, 'bear_threshold') == C.REGIME_DEFAULT_BEAR_THR

    def test_fetch_ticker_regime_uses_regime_constants(self):
        assert _default(C._fetch_ticker_regime, 'lookback') == C.REGIME_DEFAULT_LOOKBACK
        assert _default(C._fetch_ticker_regime, 'bull_threshold') == C.REGIME_DEFAULT_BULL_THR

    def test_live_monitoring_uses_regime_constants(self):
        assert _default(C.live_monitoring_analysis, 'regime_lookback') == C.REGIME_DEFAULT_LOOKBACK
        assert _default(C.live_monitoring_analysis, 'regime_bull_thr') == C.REGIME_DEFAULT_BULL_THR
        assert _default(C.live_monitoring_analysis, 'regime_bear_thr') == C.REGIME_DEFAULT_BEAR_THR

    def test_vol_targeting_uses_vt_constants(self):
        assert _default(C.mc_vol_targeted_allocation, 'target_ror') == C.VT_DEFAULT_TARGET_ROR
        assert _default(C.mc_vol_targeted_allocation, 'ruin_fraction') == C.VT_DEFAULT_RUIN_FRAC
        assert _default(C.mc_vol_targeted_allocation, 'max_leverage_cap') == C.VT_DEFAULT_MAX_LEV
        assert _default(C.mc_vol_targeted_allocation, 'target_portfolio_vol') == C.VT_DEFAULT_PORT_VOL

    def test_mc_functions_use_mc_constants(self):
        assert _default(C.strategy_monte_carlo, 'n_runs') == C.MC_DEFAULT_RUNS
        assert _default(C.strategy_monte_carlo, 'block_len') == C.MC_DEFAULT_BLOCK_LEN
        assert _default(C.strategy_monte_carlo, 'seed') == C.MC_DEFAULT_SEED
        assert _default(C.per_strategy_evaluation, 'n_mc_runs') == C.MC_DEFAULT_RUNS
        assert _default(C.bootstrap_equity_envelope, 'seed') == C.MC_DEFAULT_SEED
        assert _default(C.bootstrap_equity_envelope, 'block_len') == C.MC_DEFAULT_BLOCK_LEN

    def test_rfr_defaults_use_constant(self):
        assert _default(C.segment_metrics, 'rfr') == C.DEFAULT_RFR
        assert _default(C.rolling_sharpe, 'risk_free_annual') == C.DEFAULT_RFR


class TestSidebarReadsConstants:
    """The app.py sidebar must READ the constants, not hardcode literals.

    We can't import app.py (Streamlit side effects), so we assert the source
    text references `calculations.<CONST>` for each sidebar default rather than
    a bare number. This catches re-introduction of a hardcoded default.
    """

    def _app_src(self):
        from pathlib import Path
        return (Path(__file__).resolve().parent.parent / 'app.py').read_text()

    def test_vt_cache_fingerprint_includes_mc_params(self):
        """auto_compute_vt passes mc_block_len/mc_seed to the computation, so its
        cache fingerprint MUST include them — else changing MC params in the
        sidebar serves stale Portfolio-tab metrics (no recompute).
        """
        src = self._app_src()
        # isolate the auto_compute_vt function body, then the data_fp tuple within it
        start = src.index('def auto_compute_vt(')
        end = src.index('\nauto_compute_vt()', start)
        body = src[start:end]
        fp_start = body.index('data_fp = (')
        fp = body[fp_start:body.index('if st.session_state', fp_start)]
        assert 'mc_block_len' in fp, "vt data_fp must include mc_block_len"
        assert 'mc_seed' in fp, "vt data_fp must include mc_seed"

    def test_portfolio_mc_wired_to_portfolio_block(self):
        """auto_compute_mc must pass MC_PORTFOLIO_BLOCK_LEN (not the sidebar
        mc_block_len) to the portfolio monte_carlo call AND include it in its
        cache fingerprint — otherwise the long-block intent silently reverts."""
        src = self._app_src()
        start = src.index('def auto_compute_mc(')
        body = src[start:src.index('\nauto_compute_mc()', start)]
        assert 'MC_PORTFOLIO_BLOCK_LEN' in body, \
            "auto_compute_mc must use calculations.MC_PORTFOLIO_BLOCK_LEN"
        assert 'block_len=mc_block_len' not in body, \
            "portfolio MC must NOT use the sidebar (per-strategy) block length"

    def test_sidebar_uses_constants_not_literals(self):
        src = self._app_src()
        for ref in [
            'calculations.DEFAULT_COST_BPS_RT',
            'calculations.DEFAULT_SLIPPAGE_BPS',
            'calculations.DEFAULT_FUNDING_BPS_PER_DAY',
            'calculations.DEFAULT_RFR',
            'calculations.DEFAULT_CAPITAL',
            'calculations.MC_DEFAULT_RUNS',
            'calculations.MC_DEFAULT_BLOCK_LEN',
            'calculations.MC_DEFAULT_SEED',
            'calculations.REGIME_DEFAULT_LOOKBACK',
            'calculations.REGIME_DEFAULT_BULL_THR',
            'calculations.REGIME_DEFAULT_BEAR_THR',
            'calculations.LIVE_START_DEFAULT',
            'calculations.VT_DEFAULT_TARGET_ROR',
        ]:
            assert ref in src, f"sidebar should read {ref} (single source of truth)"
