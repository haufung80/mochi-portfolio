"""Tests for the Monte Carlo envelope — strategy_monte_carlo + per_strategy_evaluation.

Regression for two critical bugs:
1. MC seed not propagated through per_strategy_evaluation → _eval_mc_envelope.
   Verdicts were silently locked to seed=42 regardless of sidebar setting.
2. MC %ile direction/correctness — must be dollar-space, cap-invariant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calculations import (
    MC_DEFAULT_BLOCK_LEN,
    MC_DEFAULT_RUNS,
    MC_DEFAULT_SEED,
    per_strategy_evaluation,
    strategy_monte_carlo,
)


class TestSeedPropagation:
    """Regression for the silent seed-locked bug."""

    def test_same_seed_is_deterministic(self, synthetic_bull_pnl, starting_capital, split_date):
        """Same seed twice → identical MC %iles (no hidden randomness)."""
        ev1 = per_strategy_evaluation(
            synthetic_bull_pnl, starting_capital, 0.04, split_date,
            n_mc_runs=500, mc_seed=42,
        )
        ev2 = per_strategy_evaluation(
            synthetic_bull_pnl, starting_capital, 0.04, split_date,
            n_mc_runs=500, mc_seed=42,
        )
        assert ev1['mc_dd_percentile'] == ev2['mc_dd_percentile']
        assert ev1['mc_return_percentile'] == ev2['mc_return_percentile']

    def test_different_seeds_produce_variation(self, synthetic_bull_pnl, starting_capital, split_date):
        """Different seeds → different MC %iles. If they're identical, seed isn't plumbed."""
        seeds = [42, 99, 12345]
        dd_vals = []
        for seed in seeds:
            ev = per_strategy_evaluation(
                synthetic_bull_pnl, starting_capital, 0.04, split_date,
                n_mc_runs=500, mc_seed=seed,
            )
            dd_vals.append(ev['mc_dd_percentile'])
        # At least two should differ — otherwise seed is being ignored.
        assert len(set(dd_vals)) > 1, (
            f"MC %iles identical across seeds {seeds}: {dd_vals}. "
            "Seed not propagated to strategy_monte_carlo."
        )


class TestCapInvariance:
    """MC %iles compare dollar amounts → must be invariant to per-strategy cap."""

    def test_mc_percentiles_cap_invariant(self, synthetic_bull_pnl, split_date):
        """Changing starting_capital must NOT shift MC %iles."""
        pcts = []
        for cap in [100.0, 500.0, 1000.0, 5000.0]:
            ev = per_strategy_evaluation(
                synthetic_bull_pnl, cap, 0.04, split_date,
                n_mc_runs=500, mc_seed=42,
            )
            pcts.append((ev['mc_dd_percentile'], ev['mc_return_percentile']))
        unique = set(pcts)
        assert len(unique) == 1, (
            f"MC %iles drifted with starting_capital: {pcts}. "
            "Should compare dollar values only — cap-invariant."
        )


class TestPercentileDirection:
    """%ile semantics: lower DD %ile = worse drawdown."""

    def test_dd_percentile_lower_means_worse_dd(self, synthetic_broken_pnl, starting_capital, split_date):
        """Strategy with deep live DD → low MC DD %ile (bad tail)."""
        ev = per_strategy_evaluation(
            synthetic_broken_pnl, starting_capital, 0.04, split_date,
            n_mc_runs=500, mc_seed=42,
        )
        # Broken strategy has live DD much worse than MC bootstrap of BT — should land in low %ile
        assert ev['mc_dd_percentile'] < 25.0, (
            f"Broken strategy DD %ile is {ev['mc_dd_percentile']}, expected < 25. "
            "Sign of percentile may be inverted."
        )

    def test_return_percentile_lower_means_worse_return(self, synthetic_broken_pnl, starting_capital, split_date):
        """Strategy with bad live return → low MC return %ile."""
        ev = per_strategy_evaluation(
            synthetic_broken_pnl, starting_capital, 0.04, split_date,
            n_mc_runs=500, mc_seed=42,
        )
        assert ev['mc_return_percentile'] < 25.0


class TestBootstrapShape:
    """Verify strategy_monte_carlo returns the expected structure."""

    def test_mc_path_count_matches_n_runs(self, synthetic_bull_pnl):
        """If we ask for N paths, we get N max_dds + N final_pnls."""
        bt = synthetic_bull_pnl.iloc[:250]
        mc = strategy_monte_carlo(bt, n_horizon_days=100, n_runs=300, seed=42)
        assert mc['n_runs'] == 300
        assert len(mc['max_dds']) == 300
        assert len(mc['final_pnls']) == 300

    def test_mc_max_dd_always_negative_or_zero(self, synthetic_bull_pnl):
        """max_dd is peak-to-trough drop ($), must be ≤ 0."""
        bt = synthetic_bull_pnl.iloc[:250]
        mc = strategy_monte_carlo(bt, n_horizon_days=100, n_runs=200, seed=42)
        assert (np.asarray(mc['max_dds']) <= 0).all(), "Some MC max_dds are positive"
