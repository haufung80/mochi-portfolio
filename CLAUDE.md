# mochi-portfolio — project context

Streamlit dashboard for analyzing a crypto algo-trading portfolio: backtest
metrics, honest live-incubation monitoring (kill-switch), walk-forward,
vol-targeted sizing, regime attribution, and Monte-Carlo risk.

## Layout
- `app.py` — Streamlit UI. Tabs: Live Monitoring, Portfolio, Strategies,
  Walk-Forward, Risk & Regime, Monte Carlo & Sizing. Session-state radio nav,
  `@st.cache_data` data load, eager `auto_compute_*()` blocks cached by fingerprint.
- `calculations.py` — pure pandas/numpy analytics engine, **no Streamlit imports**.
  Every tunable lives in the MODULE CONSTANTS block at the top (single source of
  truth — the sidebar reads these; `tests/test_config_consistency.py` guards drift).
- `tests/` — pytest suite (~130 tests). Read `tests/README.md` first.

## Run
```bash
streamlit run app.py --server.port 8501 --server.headless true
```
A browser refresh replays the last run — **restart the server to load edited code**.

## Test
```bash
pip install -r requirements-dev.txt   # pytest, hypothesis
python3 -m pytest -q
```

## ⚠️ Data lives in a SIBLING repo
Strategy CSVs are NOT in this repo. `process_portfolio()` globs:
`/Users/tanghaufung/Desktop/Algo Trading/algo-trade-backtesting/Portfolio/*.csv`
(TradingView exports; `archieve*/` subfolders hold killed/incubated strategies).
The golden-master + integration tests reference that absolute path (override via
env `MOCHI_PORTFOLIO_DATA`) and `skipif` it's absent.

## Methodology invariants — don't break these
- **Net of fees everywhere** (10bps round-trip + 2 slippage + 0.5/day funding),
  from the cost constants. Keep gross/net consistent across every tab.
- **Peak-based MaxDD** (drop ÷ running peak, never ÷ starting capital).
- **365-day annualization** (crypto trades 24/7); Sharpe factor = `sqrt(365)`.
  Sharpe AND Sortino both use ddof=1.
- **Honest live test**: vol-targeting is solved on BACKTEST-ONLY data, then that
  sizing is applied to the live segment (no look-ahead). Never fit on live data.
- **MC blocks differ on purpose**: per-strategy MCs use block=10 (kill envelopes,
  VT sizing — Politis-White optimum); the portfolio Forward-Risk MC uses block=60
  (`MC_PORTFOLIO_BLOCK_LEN`) to preserve multi-month drawdown sequences.
- Portfolio is **equal-weight (1/N) by design** — it beats inverse-vol/ERC/
  max-Sharpe out-of-sample on this book (verified). Don't add an optimizer
  without an OOS walk-forward showing it survives.

## Behavioural test layer — MAINTAIN IT (the contract)
`test_invariants.py` (bounds/scale/monotonicity/conservation), `test_property_fuzz.py`
(Hypothesis fuzzing), `test_golden_master.py` (real-pipeline snapshot),
`test_composition_integration.py` (excluding a strategy must reach EVERY tab).
When you change `calculations.py`: add the matching invariant; **source-fix bugs,
never weaken a test**; regenerate the golden snapshot ONLY on intended numeric
changes (`GOLDEN_REGEN=1 pytest tests/test_golden_master.py`, review the diff).

## Gotchas
- **Plotly hovertemplate**: a `+` in the number format (`%{y:+.1f}`) silently
  breaks and prints the raw float — use `%{y:.1f}`.
- **Cache staleness**: any `session_state` cache derived from the loaded data MUST
  be cleared in `clear_derived_caches()` (app.py) on a composition change, or that
  tab serves stale results. The integration test guards this.
- **`PORTFOLIO_RESERVED_COLS`** (calculations.py) is the single source for which
  `plot_data` columns are portfolio-level vs per-strategy — never hardcode the set.
- Commits go to THIS repo (`github.com/haufung80/mochi-portfolio`). The data's
  parent repo (`algo-trade-backtesting`) is separate — don't commit there.
