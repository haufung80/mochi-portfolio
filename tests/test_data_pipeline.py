"""Tests for the data ingest pipeline — process_portfolio + CSV normalization.

Regression for the bug where tz-aware CSV timestamps silently dropped strategies
from the portfolio (comparison with tz-naive `oos_start` string raised TypeError
caught by outer except → strategy excluded with no warning).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from calculations import _normalize_tv_columns, process_portfolio


class TestColumnNormalization:
    """TradingView changed column names — both old + new format must work."""

    def test_old_format_passthrough(self):
        df = pd.DataFrame({
            'Trade #': [1], 'Type': ['Exit'], 'Date/Time': ['2024-01-01'],
            'Net P&L USDT': [5.0], 'Signal': ['exit'], 'Position size (value)': [500],
        })
        out = _normalize_tv_columns(df)
        assert 'Date/Time' in out.columns
        # Position size (value) maps to Size (value) or stays
        assert ('Size (value)' in out.columns) or ('Position size (value)' in out.columns)

    def test_new_format_renamed(self):
        """New TV exports use 'Date and time' + 'Size (value)' → must map back."""
        df = pd.DataFrame({
            'Trade #': [1], 'Type': ['Exit'], 'Date and time': ['2024-01-01'],
            'Net P&L USDT': [5.0], 'Signal': ['exit'], 'Size (value)': [500],
        })
        out = _normalize_tv_columns(df)
        assert 'Date/Time' in out.columns, "New 'Date and time' column not renamed"


class TestTzAwareCsvHandling:
    """Regression: tz-aware timestamps must not silently drop the strategy."""

    def test_process_portfolio_loads_tz_aware_csv(self, tmp_path, tz_aware_tv_csv):
        """tz-aware CSV must load into process_portfolio without being dropped."""
        # tz_aware_tv_csv fixture wrote the file already; folder = tmp_path
        metrics, port_stats, plot, exposure = process_portfolio(
            folder=str(tz_aware_tv_csv.parent),
            total_cap=1000.0, risk_free_rate=0.04,
            oos_start='2024-01-01', oos_end='2026-12-31',
        )
        # If tz handling is broken, len(metrics) == 0 (strategy silently dropped).
        assert len(metrics) >= 1, (
            "tz-aware CSV was silently dropped from portfolio. "
            "process_portfolio must call tz_localize(None) on Date/Time."
        )


class TestProcessPortfolioOutput:
    """Verify the basic shape of process_portfolio's return tuple."""

    def test_empty_folder_returns_empty_outputs(self, tmp_path):
        """No CSVs in folder → empty DataFrames, no exception."""
        metrics, port_stats, plot, exposure = process_portfolio(
            folder=str(tmp_path),
            total_cap=1000.0, risk_free_rate=0.04,
            oos_start='2024-01-01', oos_end='2026-12-31',
        )
        assert len(metrics) == 0
        assert port_stats == {} or len(port_stats) == 0
        assert len(plot) == 0
