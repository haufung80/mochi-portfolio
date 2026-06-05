# Mochi Portfolio — Test Suite

Pytest-based regression suite covering the statistical kill-switch engine and
data pipeline. Every test in this directory exists for one of two reasons:

1. **Regression for a real bug** caught during audit (e.g., MC seed wasn't
   propagated, tz-aware CSVs silently dropped, MDD formula used wrong base).
2. **Invariant check** that a mathematical property holds for all inputs
   (e.g., MC %iles are cap-invariant; MDD is bounded in [-1, 0]).

This is NOT a UI test suite — Streamlit rendering, plotly chart construction,
and column formatting are not covered (too brittle, low signal).

## Running

```bash
# Install pytest (one-time)
pip install -r requirements-dev.txt

# Run everything
pytest tests/ -v

# Run one file
pytest tests/test_mdd.py -v

# Run with coverage
pytest tests/ --cov=calculations --cov-report=term-missing
```

Each test file targets one component:

| File | Component | Bug regressions covered |
|---|---|---|
| `test_mdd.py` | `get_max_drawdown` | Peak-based formula, cap-invariance, bounds |
| `test_mc_envelope.py` | `strategy_monte_carlo`, `per_strategy_evaluation` | Seed propagation, determinism, cap-invariance, direction |
| `test_kill_rule.py` | `_kill_verdict` | Dual-tail rule, MIN_LIVE_TRADES floor precedence |
| `test_edge_diagnosis.py` | `_ks_edge_diagnosis` | 4-way matrix, sparse-strategy fallback, KS floor |
| `test_segment_metrics.py` | `segment_metrics` | Sharpe annualization (365 not 252), MDD fraction, win rate |
| `test_data_pipeline.py` | `process_portfolio`, `_normalize_tv_columns` | tz-aware CSV handling, column rename |
| `test_config_consistency.py` | constants vs sidebar/signatures | Single-source-of-truth drift (cost/MC/regime defaults) |
| `test_costs_and_regime.py` | `net_of_fees`, `regime_phase_split`, `net_live_pnl_from_csv` | Gross→net cost model, regime bucketing |
| `test_invariants.py` | engine-wide | **Behavioural invariants** — bounds, scale-invariance, monotonicity, conservation |
| `test_property_fuzz.py` | engine-wide | **Hypothesis fuzzing** — never-raises + bounds on generated/degenerate inputs |
| `test_golden_master.py` | full pipeline | **Characterization** — locks real per-strategy metrics + VT position sizes |

## Behavioural / property / golden-master layer (read this before refactoring)

Three files form a layer that the example-based tests structurally cannot cover.
Every numerical bug found while hardening this engine was an *invariant
violation that passed the example suite* — these catch that class.

- **`test_invariants.py`** — mathematical PROPERTIES that hold for every input:
  `BOUNDS` (MaxDD ∈ [-1,0], RoR ∈ [0,1]), `SCALE` (Sharpe is leverage-invariant;
  MaxDD% is unit-invariant), `MONOTONICITY` (more cost → less net; more target
  vol → more size), `CONSERVATION` (Σ per-regime P&L == total; Σ strategy P&L ==
  portfolio equity). Fast, no network.
- **`test_property_fuzz.py`** — Hypothesis generates thousands of inputs
  (all-zeros, single point, all-negative, huge/tiny, negative equity) and shrinks
  failures to a minimal counterexample. Contract: *never raises on real-valued
  input; documented bounds always hold.* (It found the single-observation NaN in
  `get_risk_ratios`.) Skips if `hypothesis` isn't installed.
- **`test_golden_master.py`** — snapshots the REAL pipeline's numbers (every
  per-strategy Sharpe/MaxDD/Calmar/Trades + the vol-targeted position sizes and
  scale) to `golden/pipeline_snapshot.json`. Any silent drift fails the build.
  Network-free (BTC fetch monkeypatched), deterministic (fixed MC seed), and
  skips if the parent-repo `Portfolio/` folder is absent.

  **Regenerate ONLY when you intended the numbers to change:**
  ```bash
  GOLDEN_REGEN=1 pytest tests/test_golden_master.py   # rewrite snapshot
  python tests/test_golden_master.py --regen          # same, standalone
  ```
  If a number moved and you did *not* intend it → that's a regression, don't
  regenerate. Always review the JSON git diff before committing a new snapshot.

## ⚠️ Maintenance contract (keep this layer alive)

This layer rots silently if not tended. When you touch `calculations.py`:

1. **New function** returning a ratio, a bounded quantity, an aggregate, or
   something with a known scaling law → add its invariant to `test_invariants.py`,
   and a never-raises fuzz test to `test_property_fuzz.py` if it ingests a
   CSV/user-derived array.
2. **Bug fix** → add the invariant or fuzz case that would have caught it
   (source-fix the function; never weaken a test to make it pass).
3. **Intended numeric change** (new strategy, cost/sizing change) → regenerate the
   golden snapshot and review the diff. **Unintended** drift → investigate.
4. **New dependency for tests** → add it to `requirements-dev.txt`.
5. Run the full suite (`pytest -q`) before committing; it must be green.

## Adding tests

Pattern: every time you find a bug, write a test that fails before your fix and
passes after. Drop it in the relevant `test_*.py` file. Future refactors get
free regression coverage.

```python
def test_my_bug_does_not_recur():
    """Brief description of the bug.

    Regression for: <link to commit / issue / session note>
    """
    # arrange — synthetic input that triggered the bug
    # act — call the function under test
    # assert — the corrected behavior
```

## Why no Streamlit / UI tests?

`st.dataframe`, `column_config`, plotly figures, and tab layouts are coupled
to display. Testing them either:
- Locks in pixel-perfect output that breaks on every visual tweak, or
- Requires browser automation (Selenium) which is slow and flaky.

The high-value invariants live in `calculations.py`, which is pure functions
over pandas/numpy data. Test those. Smoke-test the app boot. Use the eye for
the UI.
