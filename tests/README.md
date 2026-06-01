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
