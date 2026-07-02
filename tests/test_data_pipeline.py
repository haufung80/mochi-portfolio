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
        """2026 TV exports use 'Date and time' + 'Size (value)' + 'Net PnL USDT'
        (note 'PnL', not 'P&L') → all must map back to the canonical schema."""
        df = pd.DataFrame({
            'Trade number': [1], 'Type': ['Exit long'], 'Date and time': ['2024-01-01'],
            'Net PnL USDT': [5.0], 'Signal': ['exit'], 'Size (value)': [500],
        })
        out = _normalize_tv_columns(df)
        assert 'Date/Time' in out.columns, "New 'Date and time' column not renamed"
        assert 'Position size (value)' in out.columns, "New 'Size (value)' column not renamed"
        assert 'Net P&L USDT' in out.columns, "New 'Net PnL USDT' column not renamed"

    def test_pnl_spelling_maps_to_canonical(self):
        """Regression: mid-2026 TV exports renamed 'Net P&L USDT' → 'Net PnL USDT'
        (ampersand dropped). The whole pipeline reads the canonical 'Net P&L USDT'
        (process_portfolio, net_live_pnl_from_csv, and the app's live trade log),
        so without this alias every strategy KeyErrors on load.

        Regression for: 'Net P&L USDT' KeyError on the 2026-07-02 re-exports.
        """
        df = pd.DataFrame({'Net PnL USDT': [1.0, 2.0]})
        out = _normalize_tv_columns(df)
        assert 'Net P&L USDT' in out.columns
        assert 'Net PnL USDT' not in out.columns
        assert list(out['Net P&L USDT']) == [1.0, 2.0]


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


class TestNewFormatEndToEnd:
    """End-to-end regression: a 2026-07-02-style export must load, not drop."""

    def test_process_portfolio_loads_pnl_renamed_csv(self, tmp_path):
        """A new-format CSV ('Net PnL USDT' + 'Date and time' + UTF-8 BOM) must
        load into process_portfolio and carry its P&L — reproducing the exact
        2026-07-02 export that raised KeyError 'Net P&L USDT' and (because the
        outer try/except swallows per-file errors) silently dropped every
        strategy from the portfolio.
        """
        stem = "NEWFMT_TEST_BOTH_1H_BINANCE_BTCUSDT_2026-07-02"
        csv_path = tmp_path / f"{stem}.csv"
        header = ("Trade number,Type,Date and time,Signal,Price USDT,Size (qty),"
                  "Size (value),Net PnL USDT,Return %,Cumulative PnL USDT")
        rows = [
            header,
            "1,Entry long,2024-01-01 00:00,long,100,1,100,,,",
            "2,Exit long,2024-01-02 00:00,exit,110,1,110,10.00,10.00,10.00",
            "3,Entry long,2024-02-01 00:00,long,110,1,110,,,",
            "4,Exit long,2024-02-02 00:00,exit,105,1,105,-5.00,-4.55,5.00",
        ]
        # Real 2026 exports carry a UTF-8 BOM on the first column — include it so
        # the test faithfully mirrors the file that broke.
        csv_path.write_text("﻿" + "\n".join(rows), encoding="utf-8")

        metrics, port_stats, plot, exposure = process_portfolio(
            folder=str(tmp_path),
            total_cap=1000.0, risk_free_rate=0.04,
            oos_start='2024-01-01', oos_end='2026-12-31',
        )
        assert len(metrics) >= 1, "New-format CSV was silently dropped from the portfolio"
        # Two exits net to 10 + (-5) = 5.0; the per-strategy daily-P&L column is
        # raw gross (costs/VT applied downstream), so it must sum to exactly 5.0.
        assert stem in plot.columns, "Strategy column missing from plot_data"
        assert plot[stem].sum() == pytest.approx(5.0, abs=1e-6)


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
