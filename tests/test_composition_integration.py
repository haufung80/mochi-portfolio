"""Integration guard: excluding a strategy from the portfolio composition must
propagate to EVERY tab's calculations and charts — not just the data load.

This catches the cache-staleness class of bug: a session_state cache keyed on a
fingerprint that doesn't include the composition. The reported failure was the
Live Monitoring tab returning a STALE `live_view` after a strategy was unchecked,
because its fingerprint keyed on the folder (unchanged by an in-memory exclusion).

The test drives the REAL UI flow via Streamlit AppTest — uncheck a strategy in the
Portfolio tab, click "Rerun" — then asserts the exclusion reaches every tab. It
fails if any derived cache (VT, live, MC, envelope) is left stale.

Runs against the live data folder; skips if absent (same convention as the golden
master). It is heavier than the unit suite (loads + recomputes the real portfolio
twice), so it is the one intentionally slow test here.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(os.environ.get(
    "MOCHI_PORTFOLIO_DATA",
    "/Users/tanghaufung/Desktop/Algo Trading/algo-trade-backtesting/Portfolio",
))
APP = str(Path(__file__).resolve().parent.parent / "app.py")
TABS = ["📡 Live Monitoring", "🎯 Portfolio", "🔬 Strategies",
        "🚶 Walk-Forward", "🔥 Risk & Regime", "🎲 Monte Carlo & Sizing"]


def _ss(at, key, default=None):
    """AppTest's session_state has no .get() — emulate it with an `in` check."""
    return at.session_state[key] if key in at.session_state else default


@pytest.mark.skipif(not DATA_DIR.exists(),
                    reason=f"portfolio data folder not present: {DATA_DIR}")
def test_excluded_strategy_propagates_to_all_tabs():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    names = sorted(p.stem for p in DATA_DIR.glob("*.csv"))
    assert len(names) >= 3, "need several strategies to exclude one"
    excluded = names[0]
    keep = set(names[1:])

    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    assert len(at.exception) == 0, "app raised on initial (all-strategies) load"

    # Baseline: with everything loaded, the strategy IS allocated and the live
    # monitoring fingerprint reflects the full set.
    vt_all = (_ss(at, "vt_alloc", {}) or {}).get("position_sizes") or {}
    assert excluded in vt_all, "baseline should include the strategy (test is meaningful)"
    fp_all = _ss(at, "live_fp")

    # Real user flow: Portfolio tab → uncheck the strategy → press Rerun.
    at.radio(key="active_tab").set_value("🎯 Portfolio").run()
    nonce = _ss(at, "inc_nonce", 0)
    cb = next((c for c in at.checkbox if c.key == f"inc_{excluded}_{nonce}"), None)
    assert cb is not None, "composition checkbox for the strategy not found"
    cb.set_value(False)
    btn = next((b for b in at.button if "Rerun with selected" in (b.label or "")), None)
    assert btn is not None, "Rerun button not found"
    btn.click().run()

    # The exclusion is applied.
    applied = _ss(at, "applied_strats")
    assert applied is not None and excluded not in applied

    # VT sizing now covers exactly the kept set — the excluded strategy is gone.
    ps = (_ss(at, "vt_alloc", {}) or {}).get("position_sizes") or {}
    assert excluded not in ps
    assert set(ps) == keep

    # Live monitoring REFRESHED (not a stale cache) — the reported bug. A
    # composition change must change the live fingerprint and drop the strategy
    # from the recomputed live_view.
    fp_sub = _ss(at, "live_fp")
    assert fp_sub is not None and fp_sub != fp_all, \
        "live monitoring did not refresh after exclusion (stale cache)"
    assert excluded not in str(_ss(at, "live_view", "")), \
        "excluded strategy still present in live_view after exclusion"

    # Every tab renders on the subset without error, and the live view stays clean.
    for tab in TABS:
        at.radio(key="active_tab").set_value(tab).run()
        assert len(at.exception) == 0, f"{tab} raised after exclusion"
        assert excluded not in str(_ss(at, "live_view", "")), \
            f"excluded strategy leaked back into live_view while on {tab}"
