"""Golden-master (characterization) test of the REAL pipeline.

Locks the live portfolio's computed numbers — every per-strategy Sharpe / MaxDD
/ Calmar / Trades, plus the vol-targeted position sizes and portfolio scale — to
a committed JSON snapshot. Any silent numeric drift after a refactor lights up
RED. This is the regression armor that lets the engine be refactored fearlessly.

Determinism: the BTC benchmark fetch is monkeypatched off (it's network + time
dependent and feeds only the B&H overlay, never the strategy metrics), and the
MC sizing uses the dashboard's fixed seed. Same code + same CSVs → same numbers.

PORTABILITY: skips cleanly if the data folder isn't present, so the suite still
runs green on a machine without the (parent-repo) Portfolio data.

──────────────────────────────────────────────────────────────────────────────
REGENERATING (the maintenance step — do this DELIBERATELY, never reflexively):

    GOLDEN_REGEN=1 pytest tests/test_golden_master.py        # rewrite snapshot
    # or:
    python tests/test_golden_master.py --regen

Only regenerate when you INTENDED to change the numbers (new strategy added,
cost model changed, sizing logic updated). If a number moved and you did NOT
intend it, that's a regression — investigate, don't regenerate. Review the
JSON diff in git before committing a regenerated snapshot.
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import calculations as C

# Data folder: env override → known parent-repo location. Skip if absent.
DATA_DIR = Path(os.environ.get(
    "MOCHI_PORTFOLIO_DATA",
    "/Users/tanghaufung/Desktop/Algo Trading/algo-trade-backtesting/Portfolio",
))
SNAPSHOT = Path(__file__).resolve().parent / "golden" / "pipeline_snapshot.json"

# Dashboard defaults (mirror app.py auto_compute_vt; see calculations constants).
_OOS = ("2021-12-03", "2026-05-22")
_METRIC_KEYS = ["Sharpe", "MaxDD", "Calmar", "CAGR", "Trades", "Net Profit"]
_TOL_METRICS = 1e-6   # pure deterministic float math
_TOL_VT = 1e-4        # MC-derived (fixed seed) — allow tiny cross-lib drift


def _no_btc(*_a, **_k):
    """Stand-in for fetch_btc_daily so the snapshot never depends on the network."""
    return pd.DataFrame()


# Strategy filenames carry a trailing TradingView export date (…_YYYY-MM-DD) that
# changes on every re-export. Key the snapshot by the date-STRIPPED identity so the
# golden master survives a routine data refresh AND still diffs each strategy's
# metrics across it. With date-suffixed keys, a refresh made every strategy look
# brand-new (empty old∩new intersection) and the per-strategy comparison was
# silently skipped — the exact hole a refresh should NOT open in the regression armor.
_DATE_SUFFIX_RE = re.compile(r"_\d{4}-\d{2}-\d{2}$")


def _key_by_strategy(pairs) -> dict:
    """Build a {date-stripped strategy name → value} dict, failing loudly on a
    collision (e.g. an old and new export of the same strategy present at once,
    which would otherwise silently collapse to one key and hide a strategy)."""
    out: dict = {}
    for name, value in pairs:
        key = _DATE_SUFFIX_RE.sub("", str(name))
        if key in out:
            raise RuntimeError(
                f"duplicate strategy key after date-strip: {key!r} — two exports of "
                f"the same strategy in the data folder? Remove the stale one.")
        out[key] = value
    return out


def build_snapshot() -> dict:
    """Run the real pipeline and distill it to a stable, comparable dict."""
    orig = C.fetch_btc_daily
    C.fetch_btc_daily = _no_btc  # disable network for determinism
    try:
        metrics_df, port_stats, plot_data, exposure_df = C.process_portfolio(
            str(DATA_DIR), C.DEFAULT_CAPITAL, C.DEFAULT_RFR, *_OOS)
        vt = C.mc_vol_targeted_allocation(
            plot_data=plot_data, metrics_df=metrics_df, total_cap=C.DEFAULT_CAPITAL,
            target_ror=C.VT_DEFAULT_TARGET_ROR, ruin_fraction=C.VT_DEFAULT_RUIN_FRAC,
            max_leverage_cap=C.VT_DEFAULT_MAX_LEV, target_portfolio_vol=C.VT_DEFAULT_PORT_VOL,
            n_runs=C.VT_DEFAULT_N_RUNS, block_len=C.MC_DEFAULT_BLOCK_LEN,
            seed=C.MC_DEFAULT_SEED, cost_bps_per_round_trip=C.DEFAULT_COST_BPS_RT,
            slippage_bps=C.DEFAULT_SLIPPAGE_BPS, funding_bps_per_day=C.DEFAULT_FUNDING_BPS_PER_DAY,
            normalize_backtest_pos=False)
    finally:
        C.fetch_btc_daily = orig

    # Fail loudly if the pipeline loaded nothing: process_portfolio returns an
    # error dict (no 'Final Equity' key) when every CSV fails, and snapshotting
    # that would silently bake 0.0 / 0 strategies as the baseline.
    if "Final Equity" not in port_stats or len(metrics_df) == 0:
        raise RuntimeError(
            f"process_portfolio loaded no strategies from {DATA_DIR} "
            f"(failed_files={port_stats.get('failed_files')}) — refusing to snapshot.")

    metrics = _key_by_strategy(
        (strat, {k: float(metrics_df.loc[strat, k]) for k in _METRIC_KEYS})
        for strat in metrics_df.index
    )
    vt_positions = _key_by_strategy(
        (k, float(v)) for k, v in vt["position_sizes"].items()
    )
    return {
        "n_strategies": int(len(metrics_df)),
        "metrics": metrics,
        "vt_positions": vt_positions,
        "vt_portfolio_scale": float(vt["portfolio_scale"]),
        "vt_portfolio_vol": float(vt["portfolio_vol"]),
        "port_final_equity": float(port_stats["Final Equity"]),  # guarded above
    }


def _compare(expected: dict, actual: dict) -> list[str]:
    """Return a list of human-readable drift messages (empty == match)."""
    diffs: list[str] = []
    if expected["n_strategies"] != actual["n_strategies"]:
        diffs.append(f"n_strategies {expected['n_strategies']} → {actual['n_strategies']}")
    # Strategy set
    exp_s, act_s = set(expected["metrics"]), set(actual["metrics"])
    if exp_s != act_s:
        diffs.append(f"strategy set changed: +{act_s - exp_s} -{exp_s - act_s}")
    # Per-strategy metrics
    for strat in exp_s & act_s:
        for k in _METRIC_KEYS:
            e, a = expected["metrics"][strat][k], actual["metrics"][strat].get(k)
            if a is None or not np.isclose(e, a, rtol=_TOL_METRICS, atol=1e-9):
                diffs.append(f"{strat}.{k}: {e:.6g} → {a}")
    # VT position sizes — assert membership first; a key present on only one side
    # would otherwise be silently skipped by the intersection loop below.
    exp_vt, act_vt = set(expected["vt_positions"]), set(actual["vt_positions"])
    if exp_vt != act_vt:
        diffs.append(f"vt_positions set changed: +{act_vt - exp_vt} -{exp_vt - act_vt}")
    for strat in exp_vt & act_vt:
        e = expected["vt_positions"][strat]
        a = actual["vt_positions"][strat]
        if not np.isclose(e, a, rtol=_TOL_VT, atol=1e-6):
            diffs.append(f"vt_position[{strat}]: {e:.4f} → {a:.4f}")
    for key, tol in [("vt_portfolio_scale", _TOL_VT), ("vt_portfolio_vol", _TOL_VT),
                     ("port_final_equity", _TOL_METRICS)]:
        if not np.isclose(expected[key], actual[key], rtol=tol, atol=1e-6):
            diffs.append(f"{key}: {expected[key]:.6g} → {actual[key]:.6g}")
    return diffs


def _maybe_regen() -> bool:
    return os.environ.get("GOLDEN_REGEN") == "1" or "--regen" in sys.argv


def _write_snapshot(snap: dict) -> None:
    """Single writer for both the pytest-regen path and the __main__ CLI path."""
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(json.dumps(snap, indent=2, sort_keys=True))


@pytest.mark.skipif(not DATA_DIR.exists(),
                    reason=f"portfolio data folder not present: {DATA_DIR}")
def test_golden_master_pipeline():
    """The live pipeline's numbers match the committed snapshot.

    A MISSING snapshot FAILS (it is the committed regression baseline) — only an
    EXPLICIT regen (GOLDEN_REGEN=1 / --regen) writes it. This avoids the
    anti-pattern where a missing reference silently regenerates + skips green,
    anointing possibly-regressed output as the new baseline.
    """
    actual = build_snapshot()
    if _maybe_regen():
        existed = SNAPSHOT.exists()
        _write_snapshot(actual)
        pytest.skip(f"snapshot {'regenerated' if existed else 'created'} "
                    f"at {SNAPSHOT} ({actual['n_strategies']} strategies)")
    if not SNAPSHOT.exists():
        pytest.fail(
            f"golden snapshot missing at {SNAPSHOT} — it is the committed regression "
            f"baseline. Generate it intentionally with `GOLDEN_REGEN=1 pytest "
            f"{Path(__file__).name}` and commit the result.")
    expected = json.loads(SNAPSHOT.read_text())
    diffs = _compare(expected, actual)
    assert not diffs, (
        "Pipeline numbers drifted from golden master. If INTENDED, regenerate "
        "with GOLDEN_REGEN=1 and review the git diff. Drift:\n  " + "\n  ".join(diffs))


@pytest.mark.skipif(not DATA_DIR.exists(),
                    reason=f"portfolio data folder not present: {DATA_DIR}")
def test_portfolio_conservation_real_data():
    """Σ per-strategy daily P&L (cumulative) + capital == Portfolio Equity, on
    the REAL data. Conservation must hold for the production aggregation path."""
    orig = C.fetch_btc_daily
    C.fetch_btc_daily = _no_btc
    try:
        _, _, plot_data, _ = C.process_portfolio(
            str(DATA_DIR), C.DEFAULT_CAPITAL, C.DEFAULT_RFR, *_OOS)
    finally:
        C.fetch_btc_daily = orig
    strat_cols = [c for c in plot_data.columns if c not in C.PORTFOLIO_RESERVED_COLS]
    reconstructed = plot_data[strat_cols].sum(axis=1).cumsum() + C.DEFAULT_CAPITAL
    assert np.allclose(reconstructed.values, plot_data["Portfolio Equity"].values, atol=1e-6)


if __name__ == "__main__":
    # `python tests/test_golden_master.py --regen` to (re)write the snapshot.
    if not DATA_DIR.exists():
        sys.exit(f"data folder not found: {DATA_DIR}")
    snap = build_snapshot()
    _write_snapshot(snap)
    print(f"wrote {SNAPSHOT} — {snap['n_strategies']} strategies, "
          f"scale {snap['vt_portfolio_scale']:.3f}x")
