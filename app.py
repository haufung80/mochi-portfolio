"""
Mochi Portfolio Analysis Dashboard (Streamlit)
Comprehensive backtest analytics + walk-forward + costs + regimes + sizing.
"""

import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import calculations
from calculations import (
    process_portfolio, monte_carlo,
    get_cagr, get_max_drawdown, get_max_duration, get_mdd_info,
    get_risk_ratios, get_var_cvar,
    get_monthly_returns, get_yearly_returns,
    walk_forward_analysis, robustness_score,
    deflated_sharpe,
    classify_regimes, regime_performance, per_strategy_regime_pnl,
    regime_segments, stress_correlation,
    rolling_sharpe,
    mc_vol_targeted_allocation, vt_max_load,
    live_monitoring_analysis,
    per_strategy_capital, live_pct_bt_end_based,
    VERDICT_COLORS,
    extract_family, extract_ticker, extract_direction, fetch_ticker_prices,
    bootstrap_equity_envelope,
    # Quantstats-style extended metrics
    skew_kurtosis, tail_ratio, omega_ratio, common_sense_ratio,
    ulcer_index, ulcer_performance_index, recovery_factor,
    smart_sharpe, kelly_criterion,
    beta_alpha_correlation, information_ratio, treynor_ratio,
    worst_drawdowns,
)


# ============================================================================
# UI HELPERS (module-level — hoisted out of Streamlit `with` blocks for testability)
# ============================================================================

def display_direction(name: str) -> str:
    """Convert raw direction tag (LONG/SHORT/BOTH) into display label for tables."""
    d = extract_direction(name)
    return {'LONG': 'Long only', 'SHORT': 'Short only', 'BOTH': 'Long+Short'}.get(d, '—')


def display_ticker(name: str) -> str:
    """Display-friendly ticker — falls back to '—' (vs 'UNKNOWN' from extract_ticker)."""
    t = extract_ticker(name)
    return t if t != 'UNKNOWN' else '—'


def verdict_category(verdict_string: str) -> str:
    """Extract the category prefix from a verdict string for chip grouping.

    Verdict strings carry per-strategy detail in parentheses
    (e.g. "🔴 KILL (DD crash < P5)", "⏳ Incubating (13/20 trades)").
    Stripping the suffix lets value_counts() group all KILL subtypes / all
    Incubating sub-counts into a single chip.
    """
    if not isinstance(verdict_string, str):
        return str(verdict_string)
    return verdict_string.split(' (', 1)[0].strip()


@st.cache_data(show_spinner=False)
def net_live_summary_cached(folder: str, strat_name: str, split_iso: str,
                            cost_bps_rt: float, slippage_bps: float, _fp) -> dict:
    """Cached net-of-fees live summary for one strategy (reads its CSV).

    Keyed on the folder fingerprint `_fp` so it refreshes when CSVs change.
    Returns {gross, fees, net, n_trades} — all zeros if the CSV is missing.
    """
    return calculations.net_live_pnl_from_csv(
        Path(folder) / f"{strat_name}.csv",
        pd.Timestamp(split_iso), cost_bps_rt, slippage_bps,
    )


def render_live_drilldown(
    sum_df: pd.DataFrame,
    per_strat_evals: dict,
    plot_data: pd.DataFrame,
    total_cap: float,
    portfolio_folder: str,
    split_date: pd.Timestamp,
) -> None:
    """Render the per-strategy drill-down section of the Live Monitoring tab.

    Replaces the former ~480-LOC inline block inside `with tab_live:`. Renders
    a strategy selector, the kill-rule + edge-diagnosis banner, equity curve
    with bootstrap envelope + MDD/Return annotations, rolling 60d Sharpe,
    drawdown depth chart, strategy-vs-ticker price chart, two MC histograms
    (Return + DD) with math-check captions, and the live trade log.

    Args:
        sum_df: per-strategy summary DataFrame (must include '_full_name', 'MC DD %ile')
        per_strat_evals: {col: StrategyEvaluation} for every loaded strategy
        plot_data: portfolio plot_data DataFrame (used for ticker price chart date range)
        total_cap: total portfolio capital (for per-strat-cap derivation)
        portfolio_folder: path to CSV folder (for trade-log lookup)
        split_date: live-segment start Timestamp
    """
    st.divider()
    st.markdown("##### 🔍 Strategy Drill-Down")
    st.caption(
        "Select a strategy-ticker to see equity vs bootstrap envelope, "
        "rolling 60d Sharpe & Calmar, and strategy equity vs traded ticker price."
    )

    # Default to the worst-performing (lowest MC DD %ile = most disqualified)
    default_strat = sum_df.sort_values(
        'MC DD %ile', ascending=True, na_position='last'
    ).iloc[0]['_full_name']
    options = sum_df.sort_values(
        'MC DD %ile', ascending=True, na_position='last',
    )['_full_name'].tolist()
    try:
        default_idx = options.index(default_strat)
    except ValueError:
        default_idx = 0

    sel_strat = st.selectbox(
        "Strategy-ticker",
        options=options,
        index=default_idx,
        key='live_drill_strat',
        format_func=lambda c: f"{extract_family(c)} · {extract_ticker(c)} · {per_strat_evals[c]['combined_verdict']}",
    )

    ev = per_strat_evals[sel_strat]
    ticker = extract_ticker(sel_strat)
    direction = extract_direction(sel_strat)
    per_strat_cap = per_strategy_capital(total_cap, len(options))
    live_pct = live_pct_bt_end_based(ev, per_strat_cap)

    # ── Header banner: kill-rule inputs + edge diagnosis ───────────────────
    verdict_color = ev['combined_color']
    mc_dd_s = f"{ev['mc_dd_percentile']:.1f}%" if ev['mc_dd_percentile'] is not None else "n/a"
    mc_ret_s = f"{ev['mc_return_percentile']:.1f}%" if ev['mc_return_percentile'] is not None else "n/a"
    live_trades_n = ev.get('live_trades', ev['live_metrics']['n_active_days'])
    min_trades_n = ev.get('min_live_trades', calculations.MIN_LIVE_TRADES)
    trades_color = VERDICT_COLORS['keep'] if live_trades_n >= min_trades_n else VERDICT_COLORS['incubating']
    edge_dx = ev.get('edge_diagnosis', '⏳ n/a')
    edge_col = ev.get('edge_diagnosis_color', VERDICT_COLORS['incubating'])
    ks_p = ev.get('ks_p')
    ks_s = f"{ks_p:.3f}" if ks_p is not None else "n/a"
    st.markdown(
        f"<div style='padding:12px 14px;border-left:6px solid {verdict_color};"
        f"background:rgba(255,255,255,0.04);border-radius:4px;margin-bottom:14px;'>"
        f"<b style='font-size:1.05rem;'>{ev['combined_verdict']}</b> · "
        f"<b style='color:{edge_col};'>{edge_dx}</b> · "
        f"<code>{extract_family(sel_strat)}</code> · "
        f"<b>{ticker}</b> · <b>{direction}</b><br>"
        f"<small><b>MC DD %ile <span style='color:{ev['dd_verdict_color']};'>{mc_dd_s}</span></b> "
        f"({ev['dd_verdict']}) · "
        f"<b>MC Return %ile <span style='color:{ev['return_verdict_color']};'>{mc_ret_s}</span></b> "
        f"({ev['return_verdict']}) · "
        f"<b>KS p {ks_s}</b> · "
        f"<b>Trades <span style='color:{trades_color};'>{live_trades_n}/{min_trades_n}</span></b> · "
        f"Live <b>{live_pct:+.1f}%</b> · Live MDD <b>{ev['live_metrics']['mdd']*100:.2f}%</b>"
        f"</small></div>",
        unsafe_allow_html=True,
    )

    # ── MASTER CHART: Equity + MC envelope (left axis) vs ticker price (right) ──
    # One comprehensive view: strategy equity (BT + live) against the MC-predicted
    # envelope, AND the traded ticker's price on a secondary axis — so you can see
    # at a glance whether live under/over-performance lines up with the ticker's
    # own move (regime). Replaces the former separate "Equity & MC envelope" and
    # "Strategy P&L vs ticker price" charts, which both re-drew the equity curve.
    st.markdown("###### 📈 Equity + MC Envelope vs Traded Ticker Price")
    st.caption(
        "Left axis: strategy equity ($) — backtest, the ±2σ MC envelope it was "
        "*expected* to follow, and the live actual. Right axis: the traded "
        f"**{ticker}** price (dotted) — to see if live moves track the underlying."
    )
    mc = ev['mc']
    bt_pnl = ev['bt_pnl']; live_pnl = ev['live_pnl']
    combined_pnl = pd.concat([bt_pnl, live_pnl]).sort_index()
    combined_eq = combined_pnl.cumsum() + per_strat_cap
    bt_eq = combined_eq[combined_eq.index < split_date]
    live_eq = combined_eq[combined_eq.index >= split_date]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # --- Right axis FIRST (drawn underneath): traded ticker price ---
    ticker_prices = fetch_ticker_prices(
        ticker, str(combined_eq.index.min().date()), str(combined_eq.index.max().date()),
    )
    if not ticker_prices.empty:
        fig.add_trace(go.Scatter(
            x=ticker_prices.index, y=ticker_prices.values,
            line=dict(color=VERDICT_COLORS['warn'], width=1.4, dash='dot'),
            name=f'{ticker} price (${ticker_prices.iloc[-1]:,.0f})', opacity=0.6,
            hovertemplate=f'{ticker} $%{{y:,.0f}}<extra></extra>',
        ), secondary_y=True)

    # --- Left axis: backtest equity ---
    fig.add_trace(go.Scatter(
        x=bt_eq.index, y=bt_eq.values,
        line=dict(color=VERDICT_COLORS['neutral'], width=1.8),
        name=f'Backtest (final ${bt_eq.iloc[-1]:,.0f})' if not bt_eq.empty else 'Backtest',
    ), secondary_y=False)

    # --- Left axis: MC envelope (P5–P95, IQR, P50) projected from BT-end ---
    if mc['n_runs'] > 0 and not bt_eq.empty and not live_eq.empty:
        bt_end_val = float(bt_eq.iloc[-1])
        n_env = min(len(mc['p50_path']), len(live_eq))
        env_x = live_eq.index[:n_env]
        p5 = bt_end_val + mc['p5_path'][:n_env]; p25 = bt_end_val + mc['p25_path'][:n_env]
        p50 = bt_end_val + mc['p50_path'][:n_env]; p75 = bt_end_val + mc['p75_path'][:n_env]
        p95 = bt_end_val + mc['p95_path'][:n_env]
        fig.add_trace(go.Scatter(x=env_x, y=p95, line=dict(width=0), showlegend=False, hoverinfo='skip'), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=env_x, y=p5, line=dict(width=0), fill='tonexty', fillcolor='rgba(149,165,166,0.18)',
            name='MC P5–P95 (±2σ)', hovertemplate='P5 $%{y:,.0f}<extra></extra>',
        ), secondary_y=False)
        fig.add_trace(go.Scatter(x=env_x, y=p75, line=dict(width=0), showlegend=False, hoverinfo='skip'), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=env_x, y=p25, line=dict(width=0), fill='tonexty', fillcolor='rgba(149,165,166,0.32)',
            name='MC P25–P75 (IQR)', hovertemplate='P25 $%{y:,.0f}<extra></extra>',
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=env_x, y=p50, mode='lines',
            line=dict(color=VERDICT_COLORS['neutral'], width=1.5, dash='dot'),
            name=f'MC P50 (expected ${p50[-1]:,.0f})',
        ), secondary_y=False)

    # --- Left axis: live actual + Live% / MDD annotations + MDD markers ---
    if not live_eq.empty:
        stitch_x = [bt_eq.index[-1]] + list(live_eq.index) if not bt_eq.empty else list(live_eq.index)
        stitch_y = ([float(bt_eq.iloc[-1])] if not bt_eq.empty else []) + list(live_eq.values)
        fig.add_trace(go.Scatter(
            x=stitch_x, y=stitch_y, mode='lines',
            line=dict(color=verdict_color, width=2.8),
            name=f'Live actual (final ${live_eq.iloc[-1]:,.0f})',
        ), secondary_y=False)
        fig.add_annotation(
            x=live_eq.index[-1], y=float(live_eq.iloc[-1]),
            text=f"<b>Live: {live_pct:+.2f}%</b>",
            showarrow=True, arrowhead=2, arrowsize=1, arrowcolor=verdict_color, arrowwidth=1.5,
            ax=40, ay=-30, bgcolor='rgba(255,255,255,0.95)',
            bordercolor=verdict_color, borderwidth=1, borderpad=4,
            font=dict(color=verdict_color, size=11),
        )
        le_vals = live_eq.values.astype(float)
        if len(le_vals) >= 2:
            running_max = np.maximum.accumulate(le_vals)
            trough_pos = int(np.argmin((le_vals - running_max) / running_max))
            if trough_pos > 0:
                peak_pos = int(np.argmax(le_vals[:trough_pos + 1]))
                peak_date = live_eq.index[peak_pos]; trough_date = live_eq.index[trough_pos]
                peak_val = float(le_vals[peak_pos]); trough_val = float(le_vals[trough_pos])
                live_mdd_pct = (trough_val - peak_val) / peak_val * 100
                fig.add_trace(go.Scatter(
                    x=[peak_date], y=[peak_val], mode='markers',
                    marker=dict(symbol='triangle-down', size=12, color='#2c3e50', line=dict(color='white', width=1)),
                    name='MDD peak', showlegend=False, hovertemplate=f'Peak ${peak_val:,.2f}<extra></extra>',
                ), secondary_y=False)
                fig.add_trace(go.Scatter(
                    x=[trough_date], y=[trough_val], mode='markers',
                    marker=dict(symbol='triangle-up', size=12, color=VERDICT_COLORS['kill'], line=dict(color='white', width=1)),
                    name='MDD trough', showlegend=False, hovertemplate=f'Trough ${trough_val:,.2f}<extra></extra>',
                ), secondary_y=False)
                fig.add_shape(type='line', x0=peak_date, x1=trough_date, y0=peak_val, y1=trough_val,
                              line=dict(color=VERDICT_COLORS['kill'], width=1.4, dash='dash'))
                fig.add_annotation(
                    x=trough_date, y=trough_val,
                    text=f"<b>Live MDD: {live_mdd_pct:.2f}%</b>",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowcolor=VERDICT_COLORS['kill'], arrowwidth=1.5,
                    ax=-40, ay=35, bgcolor='rgba(255,255,255,0.95)',
                    bordercolor=VERDICT_COLORS['kill'], borderwidth=1, borderpad=4,
                    font=dict(color=VERDICT_COLORS['kill'], size=11),
                )
    mark_live_start(fig, split_date)
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Strategy Equity ($)", secondary_y=False)
    fig.update_yaxes(title_text=f"{ticker} Price ($)", secondary_y=True, showgrid=False)
    fig.update_layout(
        title=f"{extract_family(sel_strat)} {ticker} — equity vs MC envelope vs price",
        hovermode='x', height=480,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── CHARTS 2 & 3: Rolling 60d Sharpe + Drawdown Depth ($) ──────────────
    cs1, cs2 = st.columns(2)
    roll_sh = ev['rolling_sharpe'].dropna()

    with cs1:
        st.markdown("###### ⚡ Rolling 60d Sharpe")
        fig = go.Figure()
        bt_sh = roll_sh[roll_sh.index < split_date]
        live_sh = roll_sh[roll_sh.index >= split_date]
        if not bt_sh.empty:
            fig.add_trace(go.Scatter(
                x=bt_sh.index, y=bt_sh.values,
                line=dict(color=VERDICT_COLORS['neutral'], width=1.5),
                name='BT rolling',
            ))
        if not live_sh.empty:
            fig.add_trace(go.Scatter(
                x=live_sh.index, y=live_sh.values,
                line=dict(color=verdict_color, width=2.5),
                name='Live rolling',
            ))
        bt_mean_sh = ev['bt_metrics']['sharpe']
        fig.add_hline(y=bt_mean_sh, line_dash='dot', line_color=VERDICT_COLORS['neutral'], opacity=0.6)
        fig.add_hline(y=0, line_dash='dash', line_color=VERDICT_COLORS['axis'], opacity=0.4)
        mark_live_start(fig, split_date)
        fig.update_layout(
            title=f"BT mean {bt_mean_sh:+.2f} · Live mean {ev['live_metrics']['sharpe']:+.2f}",
            xaxis_title="Date", yaxis_title="Rolling Sharpe (60d)",
            height=360, hovermode='x', showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with cs2:
        st.markdown("###### 📉 Drawdown Depth (underwater curve, $)")
        full_eq = pd.concat([ev['bt_pnl'], ev['live_pnl']]).sort_index().cumsum() + per_strat_cap
        full_peaks = full_eq.cummax()
        dd_series_dollars = (full_eq - full_peaks).fillna(0)
        bt_dd = dd_series_dollars[dd_series_dollars.index < split_date]
        live_dd = dd_series_dollars[dd_series_dollars.index >= split_date]

        fig = go.Figure()
        if not bt_dd.empty:
            fig.add_trace(go.Scatter(
                x=bt_dd.index, y=bt_dd.values,
                line=dict(color=VERDICT_COLORS['neutral'], width=1),
                fill='tozeroy', fillcolor='rgba(127,140,141,0.3)',
                name='BT drawdown',
                hovertemplate="%{x|%Y-%m-%d}<br>DD: $%{y:,.2f}<extra></extra>",
            ))
        if not live_dd.empty:
            fig.add_trace(go.Scatter(
                x=live_dd.index, y=live_dd.values,
                line=dict(color=verdict_color, width=2),
                fill='tozeroy',
                fillcolor=f'rgba({int(verdict_color[1:3],16)},'
                          f'{int(verdict_color[3:5],16)},'
                          f'{int(verdict_color[5:7],16)},0.35)',
                name='Live drawdown',
                hovertemplate="%{x|%Y-%m-%d}<br>DD: $%{y:,.2f}<extra></extra>",
            ))
        bt_mdd_dollars = ev['bt_metrics']['mdd'] * per_strat_cap
        fig.add_hline(y=bt_mdd_dollars, line_dash='dot', line_color=VERDICT_COLORS['kill'], opacity=0.6,
                      annotation_text=f"BT MDD ${bt_mdd_dollars:,.2f}",
                      annotation_position='bottom right',
                      annotation_font_color=VERDICT_COLORS['kill'])
        if ev['mc']['n_runs'] > 0:
            mc_p5_dd_dollars = float(ev['mc']['p5_dd'])
            fig.add_hline(y=mc_p5_dd_dollars, line_dash='dash', line_color=VERDICT_COLORS['edge_drift'], opacity=0.5,
                          annotation_text=f"MC P5 DD ${mc_p5_dd_dollars:,.2f}",
                          annotation_position='top right',
                          annotation_font_color=VERDICT_COLORS['edge_drift'])
        fig.add_hline(y=0, line_color=VERDICT_COLORS['axis'], line_width=0.5, opacity=0.4)
        mark_live_start(fig, split_date)
        live_mdd_dollars = ev['live_metrics']['mdd'] * per_strat_cap
        fig.update_layout(
            title=f"Live MDD ${live_mdd_dollars:,.2f} · BT MDD ${bt_mdd_dollars:,.2f}",
            xaxis_title="Date", yaxis_title="Drawdown ($)",
            height=360, hovermode='x', showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── MC DISTRIBUTION HISTOGRAMS (audit MC %ile correctness) ──────────────
    st.divider()
    st.markdown("###### 🎲 Monte Carlo Distributions — Audit MC %ile")
    st.caption(
        f"Each histogram is the distribution of **{mc['n_runs']:,}** bootstrap paths "
        f"(block-bootstrap from BT, horizon = live length). The red dashed line marks "
        f"the live observed value. The percentile = (paths ≤ live) / n_runs. "
        f"This is the ground-truth visual for the MC %ile columns in the table."
    )
    mc_c1, mc_c2 = st.columns(2)

    for col_ctx, dist_key, live_val_key, pct_key, title, xlabel, dist_label in [
        (mc_c1, 'final_pnls', 'live_final_pnl', 'mc_return_percentile',
         '**MC Final P&L Distribution**', 'Final P&L ($)', 'final P&L'),
        (mc_c2, 'max_dds', 'live_dd_dollars', 'mc_dd_percentile',
         '**MC Max DD Distribution**', 'Max Drawdown ($)', 'max DD'),
    ]:
        with col_ctx:
            st.markdown(f"{title} — live in **{ev[pct_key]:.2f}%** tail")
            fig_h = go.Figure()
            fig_h.add_trace(go.Histogram(
                x=mc[dist_key], nbinsx=50,
                marker_color='#bdc3c7', name='MC paths',
                hovertemplate='Range $%{x:,.0f}<br>Count: %{y}<extra></extra>',
            ))
            fig_h.add_vline(x=ev[live_val_key], line_dash='dash',
                            line_color=VERDICT_COLORS['kill'], line_width=2.5)
            fig_h.add_annotation(
                x=ev[live_val_key], y=1, yref='paper', yanchor='bottom',
                text=f"<b>Live: ${ev[live_val_key]:+,.2f}</b><br>(P{ev[pct_key]:.1f})",
                showarrow=False, bgcolor='rgba(255,255,255,0.95)',
                bordercolor=VERDICT_COLORS['kill'], borderwidth=1, borderpad=4,
                font=dict(color=VERDICT_COLORS['kill'], size=10),
            )
            p5_val = float(np.percentile(mc[dist_key], 5))
            p50_val = float(np.percentile(mc[dist_key], 50))
            fig_h.add_vline(x=p5_val, line_dash='dot',
                            line_color=VERDICT_COLORS['neutral'], opacity=0.7)
            fig_h.add_vline(x=p50_val, line_dash='dot',
                            line_color=VERDICT_COLORS['axis'], opacity=0.5)
            fig_h.add_annotation(
                x=p5_val, y=0.92, yref='paper', text=f'P5 ${p5_val:,.0f}',
                showarrow=False, font=dict(color=VERDICT_COLORS['neutral'], size=9), xanchor='right',
            )
            fig_h.add_annotation(
                x=p50_val, y=0.92, yref='paper', text=f'P50 ${p50_val:,.0f}',
                showarrow=False, font=dict(color=VERDICT_COLORS['axis'], size=9), xanchor='left',
            )
            fig_h.update_layout(
                xaxis_title=xlabel, yaxis_title="MC path count",
                height=320, showlegend=False, bargap=0.02,
            )
            st.plotly_chart(fig_h, use_container_width=True)
            paths_le = int((np.asarray(mc[dist_key]) <= ev[live_val_key]).sum())
            st.caption(
                f"Math check: {paths_le:,} of {mc['n_runs']:,} MC paths had a {dist_label} ≤ "
                f"live ${ev[live_val_key]:+,.2f} → {paths_le}/{mc['n_runs']} = "
                f"**{paths_le/max(mc['n_runs'],1)*100:.2f}%** "
                f"(table shows {ev[pct_key]:.2f}%)."
            )

    # ── LIVE TRADE LOG (verify trade count + inspect exits) ────────────────
    st.divider()
    st.markdown(f"###### 📋 Live Trade Log — verify trade count ({ev.get('live_trades', 0)}/{ev.get('min_live_trades', calculations.MIN_LIVE_TRADES)})")
    try:
        csv_path = Path(portfolio_folder) / f"{sel_strat}.csv"
        raw_df = pd.read_csv(csv_path)
        raw_df = calculations._normalize_tv_columns(raw_df)
        raw_df['Date/Time'] = pd.to_datetime(raw_df['Date/Time'], errors='coerce')
        # Strip tz so comparison with naive split_date doesn't raise (same
        # fix as process_portfolio — tz-aware CSVs would silently fail here).
        if isinstance(raw_df['Date/Time'].dtype, pd.DatetimeTZDtype):
            raw_df['Date/Time'] = raw_df['Date/Time'].dt.tz_localize(None)
        raw_df = raw_df.dropna(subset=['Date/Time'])
        live_exits = raw_df[
            (raw_df['Date/Time'] >= split_date) &
            (raw_df['Type'].astype(str).str.startswith('Exit', na=False))
        ].copy()
        live_exits['Net P&L USDT'] = pd.to_numeric(live_exits['Net P&L USDT'], errors='coerce').fillna(0)
        live_exits = live_exits.sort_values('Date/Time').reset_index(drop=True)
        live_exits['Cumulative P&L'] = live_exits['Net P&L USDT'].cumsum()
        live_exits['Trade #'] = range(1, len(live_exits) + 1)

        n_trades = len(live_exits)
        n_wins = int((live_exits['Net P&L USDT'] > 0).sum())
        n_losses = int((live_exits['Net P&L USDT'] < 0).sum())
        win_rate = (n_wins / n_trades * 100) if n_trades else 0
        gate_status = (f'✅ Meets ≥{calculations.MIN_LIVE_TRADES} floor' if n_trades >= ev.get('min_live_trades', calculations.MIN_LIVE_TRADES)
                       else f'⏳ Below {ev.get("min_live_trades", 20)} — incubating')
        st.caption(
            f"**{n_trades} live exits** since {split_date.date()} · "
            f"Wins **{n_wins}** / Losses **{n_losses}** ({win_rate:.1f}% win rate) · "
            f"Sum P&L **${live_exits['Net P&L USDT'].sum():+,.2f}** · {gate_status}"
        )

        # Build display frame with explicit Win/Loss column + rename P&L headers
        # for unambiguous gain/loss reading. The 🟢/🔴 prefix shows at a glance
        # which trades won and which lost — no need to read sign of P&L number.
        base_cols = ['Trade #', 'Date/Time', 'Type', 'Signal',
                     'Net P&L USDT', 'Cumulative P&L']
        base_cols = [c for c in base_cols if c in live_exits.columns]
        show_df = live_exits[base_cols].copy()
        show_df['Date/Time'] = show_df['Date/Time'].dt.strftime('%Y-%m-%d %H:%M')

        # Insert W/L emoji column right after Trade # for visual scan
        def _win_loss_icon(pnl: float) -> str:
            if pnl > 0:
                return '🟢 WIN'
            if pnl < 0:
                return '🔴 LOSS'
            return '⚪ FLAT'
        show_df.insert(1, 'Result', show_df['Net P&L USDT'].apply(_win_loss_icon))

        # Rename P&L columns to make the gain/loss meaning explicit
        show_df = show_df.rename(columns={
            'Net P&L USDT':   'Trade Gain/Loss $',
            'Cumulative P&L': 'Cumulative Gain/Loss $',
        })

        # Streamlit NumberColumn for formatting (printf style — no `,` separator).
        # Sign shown via `+` in format string AND reinforced by the Result column.
        st.dataframe(
            show_df, hide_index=True, use_container_width=True,
            height=min(420, 36 * len(show_df) + 38),
            column_config={
                'Trade #': st.column_config.NumberColumn(format="%d", width='small'),
                'Result': st.column_config.TextColumn(
                    width='small',
                    help="🟢 WIN = positive P&L · 🔴 LOSS = negative P&L · ⚪ FLAT = breakeven",
                ),
                'Trade Gain/Loss $': st.column_config.NumberColumn(
                    format="$%+.2f",
                    help="Realized $ gain (+) or loss (-) on this individual trade after fees.",
                ),
                'Cumulative Gain/Loss $': st.column_config.NumberColumn(
                    format="$%+.2f",
                    help="Running total of gains/losses since live start. Final value = total live P&L.",
                ),
            },
        )
    except FileNotFoundError:
        st.warning(f"Could not load trade log: {sel_strat}.csv not found in {portfolio_folder}")
    except Exception as e:
        st.warning(f"Could not load trade log: {e}")


@st.fragment
def render_vt_recompute_section(
    plot_data: pd.DataFrame,
    metrics_df: pd.DataFrame,
    total_cap: float,
    risk_free_rate: float,
    exposure_df,
    applies_cost: bool,
    cost_bps_rt: float,
    slippage_bps: float,
    funding_bps: float,
    mc_block_len: int,
    mc_seed: int,
    port_stats: dict,
) -> None:
    """Render the MC + Vol Targeting section as a Streamlit fragment.

    Wrapping this entire section in @st.fragment scopes button-click and
    slider-change reruns to just THIS function — the rest of the app (tabs,
    sidebar) doesn't re-render. Critical UX fix: without this, clicking
    "Recompute" triggers a full script rerun and Streamlit's st.tabs widget
    loses focus, bouncing the user back to the default (Live Monitoring) tab.

    The fragment scope means OTHER tabs (Portfolio, Risk) won't see the new
    vt_alloc until a non-fragment widget triggers a full app rerun (e.g.,
    touching a sidebar slider). The cache invalidation below ensures their
    next render uses the fresh vt regardless.
    """
    vt1, vt2, vt3, vt4, vt5 = st.columns([1, 1, 1, 1, 1.3])
    # Slider defaults read from calculations.VT_DEFAULT_* (single source).
    with vt1:
        vt_target_ror = st.slider(
            "Per-strategy target RoR (%)", 1, 30,
            int(calculations.VT_DEFAULT_TARGET_ROR * 100), step=1,
            help="MC finds the max leverage where simulated Risk of Ruin ≤ this.",
        ) / 100.0
    with vt2:
        vt_max_loss = st.slider(
            "Ruin: max loss (%)", 10, 80,
            int((1 - calculations.VT_DEFAULT_RUIN_FRAC) * 100), step=5,
            help="A strategy 'ruins' when equity drops by this from start. e.g. 40% = ruin at -40% of capital.",
        )
        vt_ruin_frac = (100.0 - vt_max_loss) / 100.0
    with vt3:
        vt_max_lev_strat = st.slider(
            "Max strategy leverage", 0.5, 10.0,
            float(calculations.VT_DEFAULT_MAX_LEV), step=0.5,
            key="vt_max_lev_strat",
            help="Cap on per-strategy leverage from MC binary search. Default 1x (no margin per strategy) — keeps post-scale RoR more contained after portfolio leveraging.",
        )
    with vt4:
        vt_port_target = st.slider(
            "Portfolio vol target (%)", 5, 40,
            int(calculations.VT_DEFAULT_PORT_VOL * 100), step=1,
            help="Whole portfolio leveraged uniformly to hit this vol.",
        ) / 100.0
    with vt5:
        vt_run_btn = st.button(
            "▶️ Compute MC + Vol-Targeted Sizing",
            type="primary", use_container_width=True,
        )

    vt_apply_costs = st.checkbox(
        "Apply costs (sidebar values: " + (
            f"{cost_bps_rt:.0f}bps fee + {slippage_bps:.0f}bps slip + {funding_bps:.2f}bps/day funding"
            if applies_cost else "no costs"
        ) + ")",
        value=applies_cost,
        key="vt_apply_costs",
    )

    if vt_run_btn:
        # Persist the user's sizing choice, clear dependent caches, then force a
        # FULL app rerun. auto_compute_vt (top of the rerun) does the single
        # recompute from these params → vt_view + EVERY tab reflect it instantly.
        # The radio-based nav preserves the active tab across the rerun, so the
        # user stays right here (no bounce) — the reason this is finally safe.
        st.session_state['vt_user_params'] = dict(
            target_ror=vt_target_ror, ruin_fraction=vt_ruin_frac,
            max_leverage_cap=vt_max_lev_strat, target_portfolio_vol=vt_port_target,
            apply_costs=bool(vt_apply_costs),
        )
        for stale_key in (
            'vt_data_fp',                                  # force vt_view re-derive
            'mc_fp', 'mc_results', 'mc_start_used', 'mc_ruin_used',
            'port_env_fp', 'port_env_df',
        ):
            st.session_state.pop(stale_key, None)
        st.toast(f"✅ Vol-targeted to {vt_port_target:.0%} — all tabs updated", icon="✅")
        st.rerun(scope="app")

    if 'vt_alloc' not in st.session_state:
        return
    vt = st.session_state['vt_alloc']

    # Headline metrics
    port_vol_actual = vt['portfolio_vol']
    port_vol_target = vt['target_portfolio_vol']
    total_pos = sum(vt['position_sizes'].values())

    vt_max_load_dollars, vt_load_curve = vt_max_load(
        exposure_df, vt.get('position_sizes', {}), vt.get('backtest_positions', {})
    ) if exposure_df is not None and not exposure_df.empty else (0.0, pd.Series(dtype=float))

    sm1, sm2, sm3, sm4, sm5 = st.columns(5)
    sm1.metric(
        "Portfolio vol (achieved)",
        f"{port_vol_actual:.1%}",
        delta=f"target {port_vol_target:.0%}",
        delta_color="normal" if abs(port_vol_actual - port_vol_target) < 0.005 else "inverse",
    )
    sm2.metric(
        "🎯 Portfolio Leverage (uniform)",
        f"{vt['portfolio_scale']:.2f}x",
        delta="applied to ALL positions",
        delta_color="off",
        help="Single multiplier applied to every strategy's per-strategy-vol position to scale the entire portfolio to the portfolio vol target. Same number for all strategies.",
    )
    sm3.metric(
        "Diversification ratio",
        f"{vt['diversification_ratio']:.2f}x",
        delta=f"sum vols {vt['sum_strat_vol_contrib']:.1%} → port {port_vol_actual:.1%}",
        delta_color="off",
        help="Higher = more covariance crushing. 1.0 = no benefit (perfect correlation). 2.0+ = strong diversification.",
    )
    sm4.metric(
        "Total notional (max possible)",
        f"${total_pos:,.0f}",
        delta=f"{total_pos / total_cap:.2f}x of ${total_cap:,.0f} capital",
        delta_color="off",
        help="Sum of all leveraged positions — UPPER BOUND if every strategy fires simultaneously.",
    )
    sm5.metric(
        "⚠️ Peak Gross Exposure",
        f"${vt_max_load_dollars:,.0f}",
        delta=f"{(vt_max_load_dollars / total_cap):.2f}x of capital (realised)" if vt_max_load_dollars > 0 else "no exposure data",
        delta_color="off",
        help="Max gross exposure ever held simultaneously at the LEVERAGED scale, netted by ticker. "
             "Set Bybit margin per symbol so this peak can be carried without liquidation.",
    )

    scale_pct = vt['portfolio_scale']
    if scale_pct > 1.05:
        st.info(
            f"📈 Pre-scaling portfolio vol was **{vt['portfolio_vol_pre']:.1%}** "
            f"(below target {port_vol_target:.0%}). All positions **leveraged UP by "
            f"{scale_pct:.2f}x** to hit portfolio vol target. Final portfolio vol: "
            f"**{port_vol_actual:.1%}**."
        )
    elif scale_pct < 0.95:
        st.warning(
            f"📉 Pre-scaling portfolio vol was **{vt['portfolio_vol_pre']:.1%}** "
            f"(above target {port_vol_target:.0%}). All positions **scaled DOWN by "
            f"{scale_pct:.1%}** to hit portfolio target. Final portfolio vol: "
            f"**{port_vol_actual:.1%}**."
        )
    else:
        st.success(
            f"✅ Per-strategy vol targeting alone achieves portfolio vol "
            f"~{port_vol_actual:.1%} (close to target {port_vol_target:.0%}). "
            f"Diversification is doing the work — minimal scaling needed ({scale_pct:.2f}x)."
        )

    # Per-strategy table
    vt_rows = []
    for col in vt['allocation']:
        tpy = int(metrics_df.loc[col, 'Trades/Yr']) if col in metrics_df.index else 0
        pos_pre = vt['position_sizes_pre'].get(col, 0)
        pos_leveraged = vt['position_sizes'].get(col, 0)
        vt_rows.append({
            'Strategy': col,
            'TPY': tpy,
            'Strategy Vol (annual)': vt['strategy_vols'].get(col, 0) * 100,
            'MC Safe Leverage': vt['safe_leverage'].get(col, 0),
            'Pre-scale RoR': vt['achieved_ror'].get(col, 0) * 100,
            'Post-scale RoR': vt['achieved_ror_post'].get(col, 0) * 100,
            'MC Position $': pos_pre,
            'Leveraged Position $': pos_leveraged,
        })
    vt_df = pd.DataFrame(vt_rows).sort_values('Leveraged Position $', ascending=False)

    target_pct = vt['target_ror'] * 100
    violators = vt_df[vt_df['Post-scale RoR'] > target_pct + 0.5]
    if len(violators) > 0:
        st.warning(
            f"⚠️ Portfolio leveraging ({vt['portfolio_scale']:.2f}x) pushed **{len(violators)} strategies** above the {target_pct:.0f}% RoR target. "
            f"Worst: `{violators.iloc[0]['Strategy'][:40]}` at **{violators.iloc[0]['Post-scale RoR']:.1f}%** post-scale RoR. "
            f"This is the trade-off for hitting 20% portfolio vol — accept it, OR reduce portfolio vol target / max leverage cap."
        )
    else:
        st.success(
            f"✅ All strategies remain under {target_pct:.0f}% RoR even after portfolio leverage of {vt['portfolio_scale']:.2f}x. "
            f"Both constraints (per-strategy RoR + portfolio vol) are satisfied simultaneously."
        )

    st.dataframe(
        vt_df,
        hide_index=True,
        use_container_width=True,
        height=420,
        column_config={
            'MC Position $': st.column_config.NumberColumn(format="$%.0f"),
            'Leveraged Position $': st.column_config.NumberColumn(format="$%.0f"),
            'Strategy Vol (annual)': st.column_config.NumberColumn(format="%.1f%%"),
            'MC Safe Leverage': st.column_config.NumberColumn(format="%.2fx"),
            'Pre-scale RoR': st.column_config.NumberColumn(
                format="%.1f%%",
                help="RoR from Step 1 MC binary search at L_safe per strategy. Should equal target.",
            ),
            'Post-scale RoR': st.column_config.NumberColumn(
                format="%.1f%%",
                help="RoR verified at the FINAL leverage (L_safe × portfolio_scale). This is your actual tail risk after portfolio leveraging. May exceed target if portfolio_scale > 1.",
            ),
            'TPY': st.column_config.NumberColumn(format="%d"),
        },
    )

    pp1, pp2 = st.columns([1, 1])
    with pp1:
        fig = go.Figure(data=[go.Pie(
            labels=[s[:35] for s in vt_df['Strategy']],
            values=vt_df['Leveraged Position $'],
            hole=0.4,
            textinfo='label+percent',
            textposition='outside',
        )])
        fig.update_layout(
            title="Leveraged Position $ distribution",
            height=420, showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    with pp2:
        if not vt_load_curve.empty and vt_load_curve.max() > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=vt_load_curve.index, y=vt_load_curve.values,
                line=dict(color='#e74c3c', width=1.5),
                fill='tozeroy', fillcolor='rgba(231,76,60,0.15)',
                name='Gross Exposure',
            ))
            fig.add_hline(y=total_cap, line_dash="dash", line_color="grey",
                          annotation_text=f"1x capital (${total_cap:,.0f})",
                          annotation_position="top left")
            fig.add_hline(y=vt_max_load_dollars, line_dash="dot", line_color="#c0392b",
                          annotation_text=f"Peak ${vt_max_load_dollars:,.0f} ({vt_max_load_dollars / total_cap:.2f}x)",
                          annotation_position="top right")
            fig.update_layout(
                title="Realised Gross Exposure Over Time (leveraged, netted by ticker)",
                xaxis_title="Date", yaxis_title="Gross Exposure ($)",
                height=420, hovermode="x",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No exposure timeline available (exposure_df empty).")

    # Headline summary
    port_returns = vt['portfolio_returns']
    port_daily_pnl_dollars = port_returns * total_cap
    port_equity = port_daily_pnl_dollars.cumsum() + total_cap
    equity_returns = port_equity.pct_change().fillna(0)
    sharpe, sortino = get_risk_ratios(equity_returns, risk_free_rate)
    cagr = get_cagr(total_cap, float(port_equity.iloc[-1]),
                    max((port_equity.index[-1] - port_equity.index[0]).days, 1))
    mdd, _ = get_max_drawdown(port_equity, total_cap)

    calmar = cagr / abs(mdd) if mdd != 0 else 0.0
    cost_label = "net of costs" if vt_apply_costs else "gross (no costs)"
    st.caption(
        "These update live each time you Compute. **CAGR / MaxDD / Final Eq scale "
        "with the vol target** (leverage); **Sharpe / Calmar are ~invariant** "
        "(leverage cancels in a ratio — cost drag nudges them slightly)."
    )
    sf1, sf2, sf3, sf4, sf5 = st.columns(5)
    sf1.metric("Vol-targeted CAGR", f"{cagr:.1%}", delta=cost_label, delta_color="off")
    sf2.metric("Vol-targeted Sharpe", f"{sharpe:.2f}",
               delta=f"Sortino {sortino:.2f}", delta_color="off")
    sf3.metric("Vol-targeted Calmar", f"{calmar:.2f}",
               delta="CAGR / |MaxDD|", delta_color="off")
    sf4.metric("Vol-targeted MaxDD", f"{mdd:.1%}", delta_color="inverse")
    sf5.metric("Vol-targeted Final Eq",
               f"${float(port_equity.iloc[-1]):,.0f}",
               delta=f"vs gross backtest ${port_stats['Final Equity']:,.0f}",
               delta_color="off")

    # Return Distribution Heatmap
    with st.expander("📊 Return Distribution Heatmap — tail shape per strategy"):
        st.caption(
            "Each row = a strategy's 1-year MC return distribution at its FINAL leveraged size. "
            "RED left tail = downside risk; GREEN right tail = upside. Sorted by P5 (worst-case 5%) — "
            "most fragile strategies at top. Vol and RoR are point estimates; this shows the whole distribution."
        )
        ret_dists = vt.get('return_distributions', {})
        pct_rows, pct_index = [], []
        for col in vt['allocation']:
            arr = ret_dists.get(col)
            if arr is None or len(arr) == 0:
                continue
            pct_rows.append({
                'P1': float(np.percentile(arr, 1)) * 100,
                'P5': float(np.percentile(arr, 5)) * 100,
                'P25': float(np.percentile(arr, 25)) * 100,
                'P50': float(np.percentile(arr, 50)) * 100,
                'P75': float(np.percentile(arr, 75)) * 100,
                'P95': float(np.percentile(arr, 95)) * 100,
                'P99': float(np.percentile(arr, 99)) * 100,
            })
            pct_index.append(col)
        if pct_rows:
            pct_grid = pd.DataFrame(pct_rows, index=pct_index).sort_values('P5')
            fig = go.Figure(data=go.Heatmap(
                z=pct_grid.values,
                x=pct_grid.columns, y=pct_grid.index,
                colorscale='RdYlGn', zmid=0,
                text=np.around(pct_grid.values, 0),
                texttemplate="%{text}%",
                textfont={"size": 10},
                colorbar=dict(title="Return %"),
                hovertemplate="%{y}<br>%{x}: %{z:.1f}%<extra></extra>",
            ))
            fig.update_layout(
                height=max(400, 28 * len(pct_grid)),
                xaxis_title="Percentile of 1-yr MC outcomes (most fragile at top)",
                yaxis_title="Strategy",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No return distribution data available.")

    # TradingView / Bybit Deployment Config
    st.divider()
    st.subheader("🚀 TradingView / Bybit Deployment Config")
    st.caption("""
    Copy these numbers into your TradingView strategy `Order size` field (set "Currency" mode in TV settings)
    and your Bybit `Leverage` setting (per symbol, isolated margin recommended). The **Position $** value is
    what to actually trade for each strategy. **Min Leverage** is the minimum Bybit setting needed to open the position.
    """)

    deploy_rows = []
    for col in vt['allocation']:
        pos = vt['position_sizes'].get(col, 0)
        alloc = vt['allocation'][col]
        lev_required = (pos / alloc) if alloc > 0 else 0
        bybit_lev = max(1, math.ceil(lev_required + 0.5))
        deploy_rows.append({
            'Strategy': col,
            'Ticker': display_ticker(col),
            'Direction': display_direction(col),
            'Margin / Capital $': alloc,
            'Position $': pos,
            'Required Leverage': lev_required,
            'Set Bybit Leverage ≥': bybit_lev,
        })
    deploy_df = pd.DataFrame(deploy_rows).sort_values('Position $', ascending=False)

    st.dataframe(
        deploy_df,
        hide_index=True,
        use_container_width=True,
        height=min(420, 38 * len(deploy_df) + 50),
        column_config={
            'Margin / Capital $': st.column_config.NumberColumn(format="$%.0f"),
            'Position $': st.column_config.NumberColumn(
                format="$%.0f",
                help="The notional size to enter into TradingView's 'Order size' field.",
            ),
            'Required Leverage': st.column_config.NumberColumn(format="%.2fx"),
            'Set Bybit Leverage ≥': st.column_config.NumberColumn(
                format="%dx",
                help="Minimum Bybit leverage setting (per symbol, isolated margin) to open the position. Higher is fine; this is just the minimum.",
            ),
        },
    )

    total_margin = sum(vt['allocation'].values())
    total_notional = sum(vt['position_sizes'].values())
    avg_lev = total_notional / total_margin if total_margin > 0 else 0
    dep_c1, dep_c2, dep_c3 = st.columns(3)
    dep_c1.metric("Total Margin Used", f"${total_margin:,.0f}",
                  delta="= total portfolio capital")
    dep_c2.metric("Total Notional Traded", f"${total_notional:,.0f}",
                  delta=f"{avg_lev:.2f}x avg portfolio leverage")
    dep_c3.metric("Target Portfolio Vol", f"{vt['portfolio_vol']:.1%}",
                  delta=f"Diversification {vt['diversification_ratio']:.2f}x")

    st.download_button(
        "⬇️ Download Deployment Config (CSV)",
        data=deploy_df.to_csv(index=False).encode(),
        file_name="tradingview_bybit_deployment.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("ℹ️ How to use these numbers"):
        st.markdown("""
        1. **In TradingView** (per strategy): open Strategy Settings → Properties → set **Order size = Position $** (USD mode).
        2. **In Bybit** (per symbol): set leverage **≥ "Set Bybit Leverage ≥"** column. Use **Isolated Margin** so a single strategy blow-up cannot cascade across the portfolio.
        3. **Margin per strategy** is shown as "Margin / Capital $" — set aside this amount per symbol's isolated margin.
        4. **Total Margin Used = $%s** (your full portfolio capital) → **Total Notional Traded = $%s** at avg leverage **%.2fx**.
        5. The portfolio leverage is uniform across all strategies (single multiplier from MC + Vol Targeting). The per-strategy leverage differs because vol-targeting sized them to different starting points.
        """ % (f"{total_margin:,.0f}", f"{total_notional:,.0f}", avg_lev))

    st.divider()
    dl_vt = vt_df.copy()
    st.download_button(
        "⬇️ Download Full Vol-Targeted Sizing (CSV, all columns)",
        data=dl_vt.to_csv(index=False).encode(),
        file_name="vol_targeted_sizing_full.csv",
        mime="text/csv",
    )


def mark_live_start(
    fig,
    split_date,
    label: str = "📡 Live starts",
    short: bool = False,
    bounds=None,
) -> None:
    """Draw vertical dashed line at split_date with a label that dodges the legend.

    Single helper replacing the former duplicates `_draw_live_marker` (live tab)
    and `_mark_live_start` (portfolio tab). The label is positioned INSIDE the
    plot area (y=0.97, top-anchored, left-anchored, xshift=4) so it never
    overlaps the legend that lives above at y=1.02.

    Args:
        fig: plotly Figure to modify in place
        split_date: timestamp where live segment begins
        label: text shown next to the line. Pass short=True to drop "Live starts"
        short: replace "Live starts" with bare 📡 — for narrow side charts
        bounds: optional (min, max) tuple — skip drawing if split outside data range
    """
    try:
        ls = pd.Timestamp(split_date)
    except Exception:
        return
    if bounds is not None and not (bounds[0] <= ls <= bounds[1]):
        return
    fig.add_vline(x=ls, line_dash="dash", line_color=VERDICT_COLORS['axis'], opacity=0.7)
    text = label.replace('📡 Live starts', '📡') if short else label
    fig.add_annotation(
        x=ls, y=0.97, yref='paper', yanchor='top', xanchor='left',
        text=text, showarrow=False,
        font=dict(color=VERDICT_COLORS['axis'], size=10),
        bgcolor='rgba(255,255,255,0.85)', borderpad=3,
        xshift=4,
    )

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Mochi Portfolio Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stMetric { background-color: rgba(255,255,255,0.05); padding: 8px 12px; border-radius: 6px; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; padding: 8px 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Mochi Protocol — Portfolio Analytics")
st.caption("Backtest analytics · Walk-forward robustness · Cost-net performance · Regime conditional")

# Columns in plot_data that are NOT individual strategies — used to extract
# strategy columns by exclusion. SINGLE SOURCE: sourced from calculations so the
# app, the engine's internal filters, and the tests can never desync.
PORTFOLIO_RESERVED_COLS = calculations.PORTFOLIO_RESERVED_COLS

# ============================================================================
# SIDEBAR CONFIG
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    base_path = Path("/Users/tanghaufung/Desktop/Algo Trading/algo-trade-backtesting")

    with st.expander("📁 Data Sources", expanded=False):
        portfolio_folder = st.text_input(
            "Portfolio folder",
            value=str(base_path / "Portfolio"),
            help="Folder containing strategy CSVs (TradingView export format)",
        )
        st.caption("📡 BTC benchmark auto-fetched from Binance public API for the selected date range.")

    with st.expander("📅 Date Range", expanded=False):
        oos_start = st.text_input("Start (YYYY-MM-DD)", value="2021-12-03")
        oos_end = st.text_input("End (YYYY-MM-DD)", value="2026-05-22",
                                help="Set to today's date (or the most-recent data day) to include live incubation in the analysis.")
        live_start = st.text_input(
            "Live incubation start (YYYY-MM-DD)",
            value=calculations.LIVE_START_DEFAULT,
            help="Split date between in-sample backtest and out-of-sample live trading. Used by the Live Monitoring tab.",
        )

    with st.expander("💰 Risk Parameters", expanded=False):
        risk_free_rate = st.number_input(
            "Risk-free rate (annual %)",
            value=float(calculations.DEFAULT_RFR * 100), min_value=0.0, max_value=20.0,
            help="Used in Sharpe / Sortino calculations",
        ) / 100.0
        total_cap = st.number_input(
            "Total portfolio capital ($)",
            value=float(calculations.DEFAULT_CAPITAL), min_value=100.0,
            help="Capital is split equally across all strategies (default)",
        )

    with st.expander("💸 Cost Model", expanded=False):
        # Defaults read from calculations constants = SINGLE SOURCE OF TRUTH.
        # This guarantees the sidebar, the vol-targeting amortized model, and
        # the per-trade net_of_fees model all start from the same bps.
        cost_bps_rt = st.number_input(
            "Exchange fee (bps, round trip)",
            value=float(calculations.DEFAULT_COST_BPS_RT), min_value=0.0, step=1.0,
            help="Binance perp taker = 5bps × 2 entries/exits = ~10bps round trip",
        )
        slippage_bps = st.number_input(
            "Slippage (bps, round trip)",
            value=float(calculations.DEFAULT_SLIPPAGE_BPS), min_value=0.0, step=0.5,
            help="Estimated market impact — typically 1-5bps for liquid pairs",
        )
        funding_bps = st.number_input(
            "Daily funding cost (bps)",
            value=float(calculations.DEFAULT_FUNDING_BPS_PER_DAY), min_value=-10.0, step=0.1,
            help="Average daily funding paid (negative = received). Crypto perps ~ 0.5-1bps/day on average",
        )

    with st.expander("🎲 Monte Carlo", expanded=False):
        mc_trades_per_year = st.number_input(
            "Trades per year (portfolio MC)", value=365, min_value=10,
            help="Portfolio-level MC samples daily P&L = 365/yr. Per-strategy MC in MC+Vol Targeting uses each strategy's actual rate.",
        )
        # All MC defaults read from calculations constants → the sidebar, the
        # per-strategy kill-rule MC, and the portfolio envelope bootstrap
        # IDENTICALLY (one source). Previously the sidebar defaulted to 5000
        # runs / block 30 while the functions defaulted to 1000 / 5.
        mc_n_runs = st.number_input(
            "Number of runs", value=int(calculations.MC_DEFAULT_RUNS), min_value=100, step=500)
        mc_block_len = st.number_input(
            "Block length (days)", value=int(calculations.MC_DEFAULT_BLOCK_LEN), min_value=1,
            help="Block bootstrap preserves serial dependence",
        )
        mc_seed = st.number_input(
            "RNG seed (0 = random)", value=int(calculations.MC_DEFAULT_SEED), min_value=0)
        st.caption("ℹ️ Portfolio MC start/ruin are auto-set from your total capital and the 40% ruin threshold.")

    with st.expander("🌊 Regime Classification", expanded=False):
        # Defaults read from calculations.REGIME_DEFAULT_* (single source). 60d
        # lookback + ±10% bands cut regime flips ~50% vs noisy 30d/±5% — stable
        # labels for "long-only in bear" headwind detection.
        regime_lookback = st.number_input(
            "Lookback (days)", value=int(calculations.REGIME_DEFAULT_LOOKBACK), min_value=5)
        regime_bull_thr = st.number_input(
            "Bull threshold (%)", value=float(calculations.REGIME_DEFAULT_BULL_THR * 100), step=0.5) / 100.0
        regime_bear_thr = st.number_input(
            "Bear threshold (%)", value=float(calculations.REGIME_DEFAULT_BEAR_THR * 100), step=0.5) / 100.0

    with st.expander("🔬 Selection Bias", expanded=False):
        n_strategies_tested = st.number_input(
            "# strategies/configs you tested before keeping these",
            value=50, min_value=1,
            help="Used in Deflated Sharpe Ratio. Honest answer matters — including failed pilots, parameter scans, etc.",
        )

    st.divider()
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ============================================================================
# DATA LOADING
# ============================================================================

def _folder_fingerprint(folder: str) -> tuple:
    """(file_count, max_mtime) — busts the cache whenever any CSV in the
    Portfolio folder is added, removed, or modified. Cheap to compute."""
    folder_path = Path(folder)
    if not folder_path.exists():
        return (0, 0.0)
    csvs = sorted(folder_path.glob('*.csv'))
    if not csvs:
        return (0, 0.0)
    return (len(csvs), max(f.stat().st_mtime for f in csvs))


@st.cache_data(show_spinner=False)
def load_portfolio_data(folder, start, end, rfr, cap, folder_fp):
    """Load + aggregate strategy CSVs; BTC benchmark auto-fetched inside process_portfolio.

    `folder_fp` is part of the cache key — when any CSV in the folder is
    edited/added/removed, the fingerprint changes and Streamlit reloads.
    """
    return process_portfolio(str(folder), cap, rfr, start, end)


def df_to_csv_bytes(df: pd.DataFrame, index: bool = True) -> bytes:
    return df.to_csv(index=index).encode('utf-8')


_folder_fp = _folder_fingerprint(portfolio_folder)
with st.spinner(f"Loading portfolio ({_folder_fp[0]} CSVs) + auto-fetching BTC benchmark..."):
    metrics_df, port_stats, plot_data, exposure_df = load_portfolio_data(
        portfolio_folder, oos_start, oos_end, risk_free_rate, total_cap, _folder_fp
    )

# Surface any silent CSV parse failures so we don't get masked schema breaks again
_failed = port_stats.get('failed_files', []) if port_stats else []
if _failed:
    with st.expander(
        f"⚠️ {len(_failed)} of {port_stats.get('files_scanned', 0)} CSV files failed to parse — click to see why",
        expanded=True,
    ):
        for fname, err in _failed:
            st.error(f"**{fname}** — {err}")
        st.caption("Fix the source CSV(s) or update column aliases in `calculations.TV_COLUMN_ALIASES`.")

# Cost flag (used by auto-vt + Deflated Sharpe sections)
applies_cost = (cost_bps_rt + slippage_bps + abs(funding_bps)) > 0

# ============================================================================
# AUTO-COMPUTE MC + VOL TARGETING (default sizing, used everywhere)
# ============================================================================

# Defaults for auto-compute — single-sourced from calculations constants so the
# sidebar, the auto-compute, and the calc-side signatures can never drift apart.
VT_DEFAULT_TARGET_ROR = calculations.VT_DEFAULT_TARGET_ROR
VT_DEFAULT_RUIN_FRAC = calculations.VT_DEFAULT_RUIN_FRAC
VT_DEFAULT_MAX_LEV = calculations.VT_DEFAULT_MAX_LEV
VT_DEFAULT_PORT_VOL = calculations.VT_DEFAULT_PORT_VOL
VT_DEFAULT_N_RUNS = calculations.VT_DEFAULT_N_RUNS


def auto_compute_vt():
    """Auto-run MC + Vol Targeting with default settings; cache in session_state.

    Recompute trigger: only when **data inputs** change (capital, dates, costs, folder).
    Manual user runs from the MC tab persist until those underlying inputs change.
    """
    if plot_data is None or plot_data.empty:
        return

    # The user's manual VT sliders (from the MC tab fragment) persist here so the
    # WHOLE app — including the Portfolio/Risk tabs via vt_view — stays consistent
    # with their choice. Without this, any sidebar change would silently revert
    # the vol-target to the 20% default and the manual recompute would be lost.
    up = st.session_state.get('vt_user_params') or {}
    eff_target_ror = up.get('target_ror', VT_DEFAULT_TARGET_ROR)
    eff_ruin_frac = up.get('ruin_fraction', VT_DEFAULT_RUIN_FRAC)
    eff_max_lev = up.get('max_leverage_cap', VT_DEFAULT_MAX_LEV)
    eff_port_vol = up.get('target_portfolio_vol', VT_DEFAULT_PORT_VOL)
    # Cost on/off follows the user's VT-section checkbox once they've run it,
    # else the sidebar cost setting.
    eff_apply_cost = up.get('apply_costs', applies_cost)

    # Data fingerprint — recompute when any input that AFFECTS the vol-targeting
    # changes. Includes the user's VT params + mc_block_len/mc_seed, so changing
    # any of them refreshes the Portfolio tab (was the stale-metrics bug).
    data_fp = (
        total_cap, float(risk_free_rate),
        float(cost_bps_rt) if eff_apply_cost else 0.0,
        float(slippage_bps) if eff_apply_cost else 0.0,
        float(funding_bps) if eff_apply_cost else 0.0,
        oos_start, oos_end, portfolio_folder,
        plot_data.shape, len(metrics_df),
        int(mc_block_len), int(mc_seed),
        float(eff_target_ror), float(eff_ruin_frac),
        float(eff_max_lev), float(eff_port_vol),
    )
    if st.session_state.get('vt_data_fp') == data_fp and 'vt_alloc' in st.session_state:
        return  # nothing affecting vt changed — keep current

    with st.spinner(f"Auto-computing MC + Vol Targeting (RoR≤{eff_target_ror:.0%}, "
                    f"{1-eff_ruin_frac:.0%} ruin, max {eff_max_lev:g}x lev, {eff_port_vol:.0%} port vol)..."):
        vt = mc_vol_targeted_allocation(
            plot_data=plot_data,
            metrics_df=metrics_df,
            total_cap=total_cap,
            target_ror=eff_target_ror,
            ruin_fraction=eff_ruin_frac,
            max_leverage_cap=eff_max_lev,
            target_portfolio_vol=eff_port_vol,
            n_runs=VT_DEFAULT_N_RUNS,
            block_len=mc_block_len,
            seed=mc_seed if mc_seed > 0 else None,
            cost_bps_per_round_trip=cost_bps_rt if eff_apply_cost else 0.0,
            slippage_bps=slippage_bps if eff_apply_cost else 0.0,
            funding_bps_per_day=funding_bps if eff_apply_cost else 0.0,
            normalize_backtest_pos=False,
        )
        st.session_state['vt_alloc'] = vt
        st.session_state['vt_data_fp'] = data_fp


auto_compute_vt()


# Helper: when MC + Vol Targeting has been run, derive its portfolio stats
def get_vt_portfolio_view(total_cap_local, rfr_local):
    """If user has run MC + Vol Targeting, return its leveraged portfolio data.
    Returns dict with keys: equity (pd.Series), daily_pnl (pd.Series), stats (dict),
    or None if MC + Vol Targeting has not been computed yet.
    """
    if 'vt_alloc' not in st.session_state:
        return None
    vt = st.session_state['vt_alloc']
    if not vt or 'portfolio_returns' not in vt:
        return None
    port_returns = vt['portfolio_returns']
    daily_pnl = port_returns * total_cap_local
    equity = daily_pnl.cumsum() + total_cap_local
    days = max((equity.index[-1] - equity.index[0]).days, 1)
    final = float(equity.iloc[-1])
    cagr = get_cagr(total_cap_local, final, days)
    mdd, dd_series = get_max_drawdown(equity, total_cap_local)
    rets = equity.pct_change().fillna(0)
    sharpe, sortino = get_risk_ratios(rets, rfr_local)
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    # Recompute peak gross exposure at the leveraged scale (netted by ticker).
    # Falls back to raw backtest Max Load × avg scale if exposure_df missing.
    max_load_vt = 0.0
    load_curve_vt = pd.Series(dtype=float)
    if exposure_df is not None and not exposure_df.empty:
        max_load_vt, load_curve_vt = vt_max_load(
            exposure_df, vt.get('position_sizes', {}), vt.get('backtest_positions', {})
        )

    return {
        'equity': equity,
        'daily_pnl': daily_pnl,
        'dd_series': dd_series,
        'load_curve': load_curve_vt,
        'stats': {
            'Final Equity': final,
            'CAGR': cagr,
            'Sharpe': sharpe,
            'Sortino': sortino,
            'MaxDD': mdd,
            'Calmar': calmar,
            'portfolio_vol': vt.get('portfolio_vol', 0),
            'portfolio_scale': vt.get('portfolio_scale', 1.0),
            'diversification_ratio': vt.get('diversification_ratio', 1.0),
            'max_load': max_load_vt,
        },
        'vt': vt,
    }


vt_view = get_vt_portfolio_view(total_cap, risk_free_rate)


def auto_compute_mc():
    """Auto-run portfolio-level Forward-Risk MC on the vol-targeted daily P&L.

    Recompute trigger: vt portfolio P&L identity changed OR sidebar MC params changed.
    Result cached in session_state so navigating tabs doesn't re-simulate.
    """
    if vt_view is None:
        return
    daily_pnl = vt_view['daily_pnl'].values
    if len(daily_pnl) == 0:
        return
    # Fingerprint: include vt portfolio identity (sum + final equity uniquely identifies
    # the vol-targeted P&L stream) and all MC sidebar params + the ruin fraction.
    start_eq = float(total_cap)
    ruin_eq = float(total_cap * VT_DEFAULT_RUIN_FRAC)
    # Fingerprint: sum + std of daily P&L identifies the distribution shape
    # (not just total). std catches vol-target changes that re-scale positions.
    mc_fp = (
        float(np.nansum(daily_pnl)),
        float(np.nanstd(daily_pnl)),
        float(vt_view['stats']['Final Equity']),
        len(daily_pnl),
        start_eq, ruin_eq,
        int(mc_trades_per_year), int(mc_n_runs), int(mc_block_len), int(mc_seed),
    )
    if st.session_state.get('mc_fp') == mc_fp and 'mc_results' in st.session_state:
        return  # nothing changed
    with st.spinner(f"Auto-computing Forward-Risk MC ({mc_n_runs:,} runs)..."):
        st.session_state['mc_results'] = monte_carlo(
            trade_pnls=daily_pnl,
            start_equity=start_eq,
            ruin_equity=ruin_eq,
            trades_per_year=mc_trades_per_year,
            n_runs=mc_n_runs,
            seed=mc_seed if mc_seed > 0 else None,
            block_len=mc_block_len,
        )
        st.session_state['mc_start_used'] = start_eq
        st.session_state['mc_ruin_used'] = ruin_eq
        st.session_state['mc_fp'] = mc_fp


auto_compute_mc()


def auto_compute_live():
    """Auto-run honest live monitoring (backtest-only sizing applied to live segment).

    Caches in session_state['live_view'] under fingerprint of inputs that
    actually affect the split — folder fingerprint, dates, capital, costs,
    live_start. Skips when live segment is empty.
    """
    if plot_data is None or plot_data.empty:
        return
    try:
        split = pd.Timestamp(live_start)
    except Exception:
        st.session_state.pop('live_view', None)
        return
    if not ((plot_data.index < split).any() and (plot_data.index >= split).any()):
        st.session_state['live_view'] = {
            'error': (
                f"Live split at {live_start} not within data range "
                f"{plot_data.index.min().date()} → {plot_data.index.max().date()}. "
                f"Adjust the 'Live incubation start' or extend the date range."
            )
        }
        return
    # Fingerprint MUST include every sidebar parameter that affects the output —
    # otherwise stale cached verdicts are silently returned when the user adjusts
    # MC seed/block/runs. Bug found in audit: changing mc_seed didn't refresh.
    live_fp = (
        _folder_fp,
        oos_start, oos_end, live_start,
        float(total_cap), float(risk_free_rate),
        float(cost_bps_rt) if applies_cost else 0.0,
        float(slippage_bps) if applies_cost else 0.0,
        float(funding_bps) if applies_cost else 0.0,
        int(regime_lookback), float(regime_bull_thr), float(regime_bear_thr),
        int(mc_n_runs), int(mc_block_len), int(mc_seed),
    )
    if st.session_state.get('live_fp') == live_fp and 'live_view' in st.session_state:
        return
    with st.spinner("Auto-computing live monitoring (backtest-only sizing applied to live segment)..."):
        st.session_state['live_view'] = live_monitoring_analysis(
            plot_data=plot_data, metrics_df=metrics_df,
            total_cap=total_cap, rfr=risk_free_rate,
            live_start=live_start,
            vt_kwargs={
                'target_ror': VT_DEFAULT_TARGET_ROR,
                'ruin_fraction': VT_DEFAULT_RUIN_FRAC,
                'max_leverage_cap': VT_DEFAULT_MAX_LEV,
                'target_portfolio_vol': VT_DEFAULT_PORT_VOL,
                'n_runs': VT_DEFAULT_N_RUNS,
                'block_len': mc_block_len,
                'seed': mc_seed if mc_seed > 0 else None,
                'cost_bps_per_round_trip': cost_bps_rt if applies_cost else 0.0,
                'slippage_bps': slippage_bps if applies_cost else 0.0,
                'funding_bps_per_day': funding_bps if applies_cost else 0.0,
                'normalize_backtest_pos': False,
            },
            regime_lookback=regime_lookback,
            regime_bull_thr=regime_bull_thr,
            regime_bear_thr=regime_bear_thr,
            mc_n_runs=mc_n_runs,
            mc_block_len=mc_block_len,
            mc_seed=mc_seed if mc_seed > 0 else None,
        )
        st.session_state['live_fp'] = live_fp


auto_compute_live()


# Canonical vol-targeted series — used by every tab/chart that needs portfolio
# equity, P&L, or drawdown. Falls back to raw backtest ONLY if vt failed to
# compute (rare: empty portfolio data). Centralized here so we never accidentally
# mix raw + vol-targeted in different places.
if vt_view is not None:
    vt_equity = vt_view['equity']
    vt_pnl = vt_view['daily_pnl']
    vt_dd = vt_view['dd_series']
elif plot_data is not None and not plot_data.empty:
    vt_equity = plot_data['Portfolio Equity']
    vt_pnl = plot_data['Portfolio Daily P&L']
    vt_dd = plot_data['Portfolio DD']
else:
    vt_equity = pd.Series(dtype=float)
    vt_pnl = pd.Series(dtype=float)
    vt_dd = pd.Series(dtype=float)


# Top status bar — always uses vol-targeted portfolio (auto-computed)
if port_stats and vt_view is not None:
    vs = vt_view['stats']
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("📊 Strategies", len(metrics_df) if not metrics_df.empty else 0)
    c2.metric("💵 Vol-Targeted Equity",
              f"${vs['Final Equity']:,.0f}",
              f"{vs['CAGR']:.1%} CAGR")
    c3.metric("⚡ Vol-Targeted Sharpe",
              f"{vs['Sharpe']:.2f}",
              f"Sortino {vs['Sortino']:.2f}")
    c4.metric("📉 MaxDD",
              f"{vs['MaxDD']:.1%}",
              f"@ {vs['portfolio_vol']:.1%} portfolio vol",
              delta_color="inverse")
    c5.metric("🎯 Calmar",
              f"{vs['Calmar']:.2f}",
              f"Lev {vs['portfolio_scale']:.2f}x uniform")
    c6.metric("🛡️ Diversification",
              f"{vs['diversification_ratio']:.2f}x",
              f"Peak Load ${vs.get('max_load', 0):,.0f}",
              help="Diversification ratio (sum of strategy vols / portfolio vol) · "
                   "Peak Load = max gross exposure at the LEVERAGED scale, netted by ticker. "
                   "This is the realised peak margin/notional you'd carry on Bybit.",
              )
elif port_stats:
    st.info("⏳ Auto-computing vol-targeted portfolio…")
else:
    st.warning("⚠️ No portfolio data loaded. Check the folder path in the sidebar.")

st.divider()

# ============================================================================
# MAIN TABS
# ============================================================================

# Session-state-backed tab navigation (replaces st.tabs).
#
# st.tabs resets to the first tab on every full rerun, which forced us into
# @st.fragment workarounds and still left other tabs stale after a manual VT
# recompute. A radio keyed in session_state PERSISTS the selection across
# st.rerun(), so the manual recompute can force a full app rerun (updating ALL
# tabs instantly) WITHOUT bouncing the user off their current tab. Bonus: only
# the active tab's body executes per rerun (st.tabs ran all six), so each
# interaction is faster.
TAB_LIVE = "📡 Live Monitoring"
TAB_PORT = "🎯 Portfolio"
TAB_STRAT = "🔬 Strategies"
TAB_WF = "🚶 Walk-Forward"
TAB_RISK = "🔥 Risk & Regime"
TAB_MC = "🎲 Monte Carlo & Sizing"
active_tab = st.radio(
    "Navigation",
    [TAB_LIVE, TAB_PORT, TAB_STRAT, TAB_WF, TAB_RISK, TAB_MC],
    horizontal=True, label_visibility="collapsed", key="active_tab",
)
st.divider()

# ============================================================================
# TAB: LIVE MONITORING
# ============================================================================
# Truth comes from out-of-sample live performance. This tab applies the
# *backtest-only* vol-targeted sizing to the live segment (no look-ahead),
# compares per-strategy backtest vs live, runs statistical drift tests,
# and classifies each strategy's health.

if active_tab == TAB_LIVE:
    st.subheader("📡 Live Incubation Monitoring — Per-Strategy")
    st.caption(
        "Each strategy-ticker instance evaluated against its own backtest-derived "
        "Monte Carlo distribution. **Return Eff %** and **DD Eff %** quantify pace; "
        "**MC percentiles** test statistical significance; rolling Sharpe/Calmar show "
        "the trajectory; bootstrap envelope shows the ±2σ confidence band."
    )

    live_view = st.session_state.get('live_view')
    if plot_data is None or plot_data.empty:
        st.warning("⚠️ No portfolio data loaded.")
    elif live_view is None:
        st.info("⏳ Auto-computing live monitoring…")
    elif 'error' in live_view:
        st.error(f"🚫 {live_view['error']}")
    else:
        split_date = live_view['split_date']
        bt_m = live_view['bt_metrics']
        live_m = live_view['live_metrics']
        per_strat_evals = live_view.get('per_strategy_evals', {}) or {}

        # ── COMPACT PORTFOLIO HEADER (single line context) ─────────────────────
        pnl_pct = (live_m['total_pnl'] / total_cap) * 100 if total_cap else 0
        pnl_color = '#27ae60' if live_m['total_pnl'] >= 0 else '#c0392b'
        st.markdown(
            f"<div style='padding:8px 14px;background:rgba(255,255,255,0.03);"
            f"border-radius:6px;margin-bottom:10px;font-size:0.92rem;'>"
            f"<b>Split {split_date.strftime('%Y-%m-%d')}</b> · "
            f"BT {live_view['backtest_days']}d / Live {live_view['live_days']}d · "
            f"Portfolio (vol-targeted): "
            f"<b style='color:{pnl_color};'>${live_m['total_pnl']:+,.0f} "
            f"({pnl_pct:+.1f}%)</b> · "
            f"Sharpe {live_m['sharpe']:+.2f} (BT {bt_m['sharpe']:+.2f}) · "
            f"PF {live_m['pf']:.2f} (BT {bt_m['pf']:.2f})"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── STEP 1: PER-STRATEGY SUMMARY TABLE ─────────────────────────────────
        st.markdown("##### 📊 Per-Strategy Kill-Switch Evaluation")
        st.caption(
            "**KILL RULE (Verdict column) — MC-based only:** "
            "`(MC DD %ile ≤ 5  OR  MC Return %ile ≤ 5)  AND  Live Trades ≥ 20`. "
            "KS p is **NOT** part of the kill rule — it powers Edge Diagnosis as a secondary signal. "
            "Verdicts: **🔴 KILL** · **🟡 WARN** · **🟢 KEEP** · **⏳ INCUBATING**.  \n\n"
            "**EDGE DIAGNOSIS (Edge Diagnosis column) — KS-test diagnostic:** "
            "**🔴 BROKEN EDGE** (kill fires AND KS p<0.05 → archive permanently) · "
            "**🟠 UNLUCKY** (kill fires BUT KS p≥0.05 → suspend, edge intact) · "
            "**🟡 EDGE DRIFTING** (kill quiet BUT KS p<0.05 → watch — leading indicator) · "
            "**🟢 STABLE**. "
            "Use Edge Diagnosis to decide *archive vs suspend* when Verdict fires KILL."
        )

        if not per_strat_evals:
            st.info("No per-strategy evaluations available yet.")
        else:
            per_strat_cap = per_strategy_capital(total_cap, len(per_strat_evals))
            # Net-of-fees: TradingView P&L is GROSS. Apply round-trip cost so the
            # displayed Live % reflects what the strategy actually nets. Uses the
            # sidebar cost assumptions (defaults to the 11+2 bps model).
            _net_cost_rt = cost_bps_rt if cost_bps_rt else calculations.DEFAULT_COST_BPS_RT
            _net_slip = slippage_bps if slippage_bps else calculations.DEFAULT_SLIPPAGE_BPS
            summary_rows = []
            for col, ev in per_strat_evals.items():
                gross_pct = live_pct_bt_end_based(ev, per_strat_cap)
                bt_end_eq = per_strat_cap + ev['bt_metrics']['total_pnl']
                nl = net_live_summary_cached(
                    portfolio_folder, col, str(split_date.date()),
                    _net_cost_rt, _net_slip, _folder_fp,
                )
                net_pct = (nl['net'] / bt_end_eq * 100.0) if bt_end_eq > 0 else 0.0
                summary_rows.append({
                    'Strategy': extract_family(col),
                    'Ticker': extract_ticker(col),
                    'Direction': extract_direction(col),
                    'Verdict': ev['combined_verdict'],
                    'Edge Diagnosis': ev.get('edge_diagnosis', '⏳ n/a'),
                    'MC DD %ile': ev['mc_dd_percentile'],
                    'MC Return %ile': ev['mc_return_percentile'],
                    'KS p': ev.get('ks_p'),
                    'Live Trades': ev.get('live_trades', ev['live_metrics']['n_active_days']),
                    'Live % (net)': net_pct,
                    'Live % (gross)': gross_pct,
                    'Fees $': nl['fees'],
                    'Live MDD %': ev['live_metrics']['mdd'] * 100,
                    'Days Since Last Trade': ev['live_metrics']['days_since_last_trade'] or 0,
                    '_full_name': col,
                })
            sum_df = pd.DataFrame(summary_rows)

            # Verdict-count chips (quick scan) — group by category, ignoring
            # the parenthetical detail (e.g. "13/20 trades", "DD crash < P5")
            # so all 🔴 KILL subtypes count as one chip, all ⏳ Incubating sub-
            # counts collapse into one chip, etc.
            verdict_categories = sum_df['Verdict'].map(verdict_category)
            # Preserve a stable display order: KILL → WARN → KEEP → INCUBATING/Insufficient
            _order = {'🔴 KILL': 0, '🟡 WARN': 1, '🟢 KEEP': 2,
                      '⏳ Incubating': 3, '⏳ Insufficient data': 4}
            verdict_counts = (verdict_categories.value_counts()
                              .reindex(sorted(verdict_categories.unique(),
                                              key=lambda v: _order.get(v, 99)))
                              .dropna().astype(int))
            chip_cols = st.columns(min(len(verdict_counts), 5) or 1)
            for i, (verdict, n) in enumerate(verdict_counts.items()):
                if i < len(chip_cols):
                    chip_cols[i].metric(verdict, int(n))

            # Hard-sort by MC DD %ile ascending — worst (lowest percentile) at the top
            sum_df_display = sum_df.drop(columns=['_full_name']).sort_values(
                'MC DD %ile', ascending=True, na_position='last',
            )

            st.dataframe(
                sum_df_display,
                hide_index=True,
                use_container_width=True,
                height=min(620, 42 * len(sum_df_display) + 50),
                column_config={
                    'Direction': st.column_config.TextColumn(width='small'),
                    'Verdict': st.column_config.TextColumn(width='medium'),
                    'Edge Diagnosis': st.column_config.TextColumn(
                        width='medium',
                        help="Second-layer diagnostic from KS test on per-trade PnL distribution.\n\n"
                             "🔴 BROKEN EDGE: MC fires AND distribution shifted (p<0.05) — kill permanently.\n"
                             "🟠 UNLUCKY: MC fires BUT per-trade distribution unchanged — suspend, edge intact.\n"
                             "🟡 EDGE DRIFTING: MC quiet BUT distribution shifted — watch closely.\n"
                             "🟢 STABLE: both quiet — no concern.",
                    ),
                    'MC DD %ile': st.column_config.NumberColumn(
                        format="%.1f%%",
                        help="Where live MDD falls in 1000-path bootstrap MC distribution. "
                             "≤5% fires KILL (sharp crash). ≤15% triggers WARN. "
                             "Kill-rule input #1.",
                    ),
                    'MC Return %ile': st.column_config.NumberColumn(
                        format="%.1f%%",
                        help="Where live cumulative P&L falls in MC distribution. "
                             "≤5% fires KILL (slow-bleed decay — return < P5 even if DD didn't crash). "
                             "Kill-rule input #2.",
                    ),
                    'KS p': st.column_config.NumberColumn(
                        format="%.3f",
                        help="Kolmogorov-Smirnov test on per-trade PnL: backtest vs live distribution. "
                             "p<0.05 = per-trade distribution has changed → suggests edge drift.\n\n"
                             "⚠️ KS p is NOT a kill-rule input. The Verdict column uses ONLY MC %iles. "
                             "KS p feeds the Edge Diagnosis column to distinguish a broken edge from bad luck.",
                    ),
                    'Live Trades': st.column_config.NumberColumn(
                        format="%d",
                        help="Active live trading days. Kill rule requires ≥20 to avoid "
                             "small-sample false positives. Below 20 → ⏳ INCUBATING.",
                    ),
                    'Live % (net)': st.column_config.NumberColumn(
                        format="%+.1f%%",
                        help="Live P&L NET of round-trip trading costs, as % of BT-end equity. "
                             "TradingView P&L is gross — this subtracts the modeled fee+slippage. "
                             "THIS is the real number to judge profitability on.",
                    ),
                    'Live % (gross)': st.column_config.NumberColumn(
                        format="%+.1f%%",
                        help="Live P&L BEFORE costs (raw TradingView 'Net P&L USDT'). "
                             "Compare to 'Live % (net)' to see the fee drag.",
                    ),
                    'Fees $': st.column_config.NumberColumn(
                        format="$%.0f",
                        help="Total round-trip trading cost over the live segment "
                             "(|notional| × (fee+slip) bps per trade). High-frequency "
                             "strategies pay the most — this is what flips marginal edges negative.",
                    ),
                    'Live MDD %': st.column_config.NumberColumn(
                        format="%.2f%%",
                        help="Live segment maximum drawdown (gross), as % of peak equity.",
                    ),
                    'Days Since Last Trade': st.column_config.NumberColumn(format="%d"),
                },
            )

            # ── STEP 2: PER-STRATEGY DRILL-DOWN (moved to render_live_drilldown helper)
            render_live_drilldown(
                sum_df=sum_df,
                per_strat_evals=per_strat_evals,
                plot_data=plot_data,
                total_cap=total_cap,
                portfolio_folder=portfolio_folder,
                split_date=split_date,
            )


            # ── CSV download (full summary) ──────────────────────────────────
            st.divider()
            st.download_button(
                "⬇️ Download Per-Strategy Live Evaluation (CSV)",
                data=sum_df.drop(columns=['_full_name']).to_csv(index=False).encode(),
                file_name=f"live_per_strategy_{split_date.strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

# TAB: PORTFOLIO OVERVIEW
# ============================================================================

if active_tab == TAB_PORT:
    if plot_data is None or plot_data.empty:
        st.error("No portfolio data. Check sidebar configuration.")
    elif vt_view is None:
        st.info("⏳ Vol-targeted portfolio is being auto-computed. Refresh in a moment.")
    else:
        vs = vt_view['stats']
        st.success(
            f"📈 **Vol-Targeted Portfolio** at {vs['portfolio_scale']:.2f}x uniform leverage · "
            f"target σ = {vs['portfolio_vol']:.1%} · "
            f"final = ${vs['Final Equity']:,.0f} ({vs['CAGR']:.1%} CAGR) · "
            f"Sharpe {vs['Sharpe']:.2f} · MaxDD {vs['MaxDD']:.1%}"
        )
        port_equity_active = vt_view['equity']
        port_dd_active = vt_view['dd_series']
        port_pnl_active = vt_view['daily_pnl']
        view_label = "Vol-Targeted Portfolio"
        view_color = "#9b59b6"

        st.subheader(f"Equity Curve — {view_label}")

        strategy_cols = [c for c in plot_data.columns if c not in PORTFOLIO_RESERVED_COLS]

        c_show1, c_show2, c_show3 = st.columns(3)
        show_individual = c_show1.checkbox("Show individual strategies (raw)", value=False)
        show_benchmark = c_show2.checkbox("Show B&H benchmark", value=True)
        show_envelope = c_show3.checkbox(
            "Show MC envelope (±2σ)", value=True,
            help="OOS hypothesis test: block-bootstrap (1000 paths, B=5) of the "
                 "BACKTEST-only daily P&L, projected forward from the equity at "
                 "live-incubation start. P5–P95 = ±2σ band of where live equity COULD "
                 "be under H0 'no decay'. Envelope only appears AFTER live_start "
                 "(backtest is the in-sample reference, not testable against itself). "
                 "Live line sustained below P5 = statistically rare → regime change.",
        )

        per_strat_cap = per_strategy_capital(total_cap, len(strategy_cols))
        fig = go.Figure()

        # ── MC Envelope (proper OOS hypothesis test) ──
        # The envelope is the bootstrap of BACKTEST-only daily P&L, projected
        # FORWARD from the equity at live_start. This is the correct statistical
        # framing: "given what backtest history showed, what range of equity
        # paths could live have produced?" Inside band = consistent with H0
        # 'no decay'. Below P5 = statistically rare, real decay signal.
        # Backtest segment has no envelope (it IS the in-sample reference).
        if show_envelope and not port_equity_active.empty:
            try:
                _split_ts = pd.Timestamp(live_start)
            except Exception:
                _split_ts = None
            if _split_ts is not None and (port_equity_active.index >= _split_ts).any() \
                    and (port_equity_active.index < _split_ts).any():
                # Split portfolio equity at live_start
                port_bt = port_equity_active[port_equity_active.index < _split_ts]
                port_live = port_equity_active[port_equity_active.index >= _split_ts]
                bt_pnl_series = port_bt.diff().fillna(0)
                live_anchor_eq = float(port_bt.iloc[-1])
                n_live_horizon = len(port_live)

                # Fingerprint includes BOTH sum AND std of bt_pnl_series so two
                # vt configurations with the same total P&L but different daily
                # volatility (which CAN happen when target_portfolio_vol changes
                # the position scale) are detected as distinct.
                env_fp = (
                    int(len(bt_pnl_series)), int(n_live_horizon),
                    float(total_cap), float(live_anchor_eq),
                    float(bt_pnl_series.sum()),
                    float(bt_pnl_series.std()),
                )
                if (st.session_state.get('port_env_fp') != env_fp
                        or 'port_env_df' not in st.session_state):
                    with st.spinner("Computing MC envelope (1000 bootstrap paths from backtest)..."):
                        st.session_state['port_env_df'] = bootstrap_equity_envelope(
                            bt_daily_pnl=bt_pnl_series,
                            n_live_days=max(n_live_horizon, 1),
                            starting_equity=live_anchor_eq,
                            n_sims=1000, seed=42, block_len=5,
                        )
                        st.session_state['port_env_fp'] = env_fp
                env_df = st.session_state.get('port_env_df')
                if env_df is not None and not env_df.empty:
                    n_env = min(len(env_df), n_live_horizon)
                    # Envelope x = anchor (split date with bt end equity) + live dates
                    env_x = pd.DatetimeIndex([port_bt.index[-1]]).append(port_live.index[:n_env - 1])

                    def _to_pct(arr):
                        return (np.asarray(arr[:n_env]) / total_cap - 1) * 100

                    p5 = _to_pct(env_df['P5'].values)
                    p25 = _to_pct(env_df['P25'].values)
                    p50 = _to_pct(env_df['P50'].values)
                    p75 = _to_pct(env_df['P75'].values)
                    p95 = _to_pct(env_df['P95'].values)
                    # P5–P95 outer band
                    fig.add_trace(go.Scatter(
                        x=env_x, y=p95, line=dict(width=0),
                        showlegend=False, hoverinfo='skip',
                    ))
                    fig.add_trace(go.Scatter(
                        x=env_x, y=p5, line=dict(width=0),
                        fill='tonexty', fillcolor='rgba(149,165,166,0.16)',
                        name='MC P5–P95 (±2σ, BT-bootstrap → OOS)',
                        hovertemplate='P5: %{y:.1f}%<extra>±2σ lower</extra>',
                    ))
                    # P25–P75 IQR band
                    fig.add_trace(go.Scatter(
                        x=env_x, y=p75, line=dict(width=0),
                        showlegend=False, hoverinfo='skip',
                    ))
                    fig.add_trace(go.Scatter(
                        x=env_x, y=p25, line=dict(width=0),
                        fill='tonexty', fillcolor='rgba(149,165,166,0.30)',
                        name='MC P25–P75 (IQR)',
                        hovertemplate='P25: %{y:.1f}%<extra>IQR lower</extra>',
                    ))
                    # P50 expected median
                    fig.add_trace(go.Scatter(
                        x=env_x, y=p50, mode='lines',
                        line=dict(color='#95a5a6', width=1.3, dash='dot'),
                        name='MC P50 (expected median from BT)',
                    ))

        if show_individual:
            for col in strategy_cols:
                strat_eq = plot_data[col].cumsum() + per_strat_cap
                y = (strat_eq / per_strat_cap - 1) * 100
                fig.add_trace(go.Scatter(
                    x=plot_data.index, y=y, name=col,
                    line=dict(color='rgba(128,128,128,0.25)', width=1),
                    hoverinfo="skip", showlegend=False,
                ))
        # Portfolio trace: Return % on left axis, with Equity $ shown in hover.
        # customdata carries the dollar value so the unified-hover popup shows
        # BOTH "+178% · $2,780" on one line per data point.
        port_y = (port_equity_active / total_cap - 1) * 100
        port_dollars = port_equity_active.values
        fig.add_trace(go.Scatter(
            x=port_equity_active.index, y=port_y, name=view_label,
            line=dict(color=view_color, width=3),
            customdata=port_dollars,
            # NOTE: no '+' in the % format — plotly's hovertemplate parser fails on
            # %{y:+.1f} and falls back to the raw float; %{y:.1f} renders correctly.
            hovertemplate='<b>%{y:.1f}%</b> · <b>$%{customdata:,.0f}</b><extra>' + view_label + '</extra>',
        ))
        if show_benchmark and 'B&H BTC Equity' in plot_data.columns:
            bh_eq = plot_data['B&H BTC Equity']
            bh_y = (bh_eq / total_cap - 1) * 100
            fig.add_trace(go.Scatter(
                x=plot_data.index, y=bh_y, name="B&H BTC",
                line=dict(color="#f39c12", width=2, dash="dash"),
                customdata=bh_eq.values,
                hovertemplate='<b>%{y:.1f}%</b> · <b>$%{customdata:,.0f}</b><extra>B&H BTC</extra>',
            ))

        # Invisible anchor trace on yaxis2 — an overlaying axis only RENDERS when
        # at least one trace is assigned to it (an explicit range alone won't show
        # the axis). Its values don't matter; yaxis2.range below sets the scale so
        # the $ axis stays an exact linear image of the Return % axis.
        if not port_equity_active.empty:
            fig.add_trace(go.Scatter(
                x=port_equity_active.index, y=port_dollars,
                yaxis='y2', showlegend=False, hoverinfo='skip',
                line=dict(color='rgba(0,0,0,0)'),
            ))

        # Mark live-incubation start date — use module-level `mark_live_start`.
        # Bounds tuple ensures the marker is suppressed if split falls outside data.
        _port_bounds = (
            (port_equity_active.index.min(), port_equity_active.index.max())
            if not port_equity_active.empty else None
        )
        mark_live_start(
            fig, live_start,
            label=f"📡 Live starts {pd.Timestamp(live_start).strftime('%Y-%m-%d')}",
            bounds=_port_bounds,
        )
        # Link the two Y-axes: Equity $ must be an EXACT linear image of Return %
        # (equity_$ = total_cap × (1 + %/100)). Otherwise plotly auto-ranges each
        # axis independently — driven by different traces (the % axis spans the
        # envelope/B&H/strategy lines, the $ axis only the portfolio) — and the $
        # scale stops lining up with the curve. Compute the % range over every
        # left-axis trace, then derive the matching $ range from the same endpoints.
        _pct = [np.asarray(tr.y, dtype=float) for tr in fig.data
                if getattr(tr, 'yaxis', None) in (None, 'y') and tr.y is not None]
        _pct = np.concatenate([a[np.isfinite(a)] for a in _pct]) if _pct else np.array([0.0])
        if not _pct.size:
            _pct = np.array([0.0])
        _lo, _hi = float(_pct.min()), float(_pct.max())
        _pad = max((_hi - _lo) * 0.05, 1.0)
        _lo, _hi = _lo - _pad, _hi + _pad
        fig.update_layout(
            title=f"{view_label} vs Benchmark (Return % + Equity $)",
            xaxis_title="Date",
            yaxis=dict(title="Return %", range=[_lo, _hi],
                       ticksuffix="%", tickformat=",.0f", hoverformat=".1f"),
            yaxis2=dict(
                title="Equity ($)",
                overlaying='y', side='right', showgrid=False,
                range=[total_cap * (1 + _lo / 100), total_cap * (1 + _hi / 100)],
                tickformat="$,.0f",
            ),
            hovermode="x unified", height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Drawdown + Rolling Sharpe
        st.subheader("Drawdown & Rolling Sharpe")
        cdd, csh = st.columns(2)
        with cdd:
            dd_nominal = port_dd_active * total_cap
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=port_dd_active.index, y=dd_nominal,
                fill='tozeroy', fillcolor='rgba(231,76,60,0.3)',
                line=dict(color='rgb(192,57,43)'), name='Drawdown',
                hovertemplate="%{x|%Y-%m-%d}<br>DD: $%{y:.2f}<extra></extra>",
            ))
            mark_live_start(fig, live_start, label="📡 Live", bounds=_port_bounds)
            fig.update_layout(
                title=f"Drawdown Depth (USDT) — {view_label}",
                xaxis_title="Date", yaxis_title="Drawdown ($)",
                hovermode="x", height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        with csh:
            window = st.slider("Rolling Sharpe window (days)", 14, 180, 60, key="roll_sh_win")
            port_rets = port_equity_active.pct_change().fillna(0)
            roll_sh = rolling_sharpe(port_rets, window, risk_free_rate)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=roll_sh.index, y=roll_sh.values,
                line=dict(color='#3498db', width=2),
                fill='tozeroy', fillcolor='rgba(52,152,219,0.15)',
                name='Rolling Sharpe',
            ))
            fig.add_hline(y=1.0, line_dash="dot", line_color="orange",
                          annotation_text="Sh=1")
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            mark_live_start(fig, live_start, label="📡 Live", bounds=_port_bounds)
            fig.update_layout(
                title=f"Rolling {window}-Day Sharpe — {view_label}",
                xaxis_title="Date", yaxis_title="Sharpe (annualized)",
                hovermode="x", height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Monthly returns heatmap
        st.subheader(f"Monthly Returns Heatmap — {view_label}")
        monthly = get_monthly_returns(port_equity_active)
        if not monthly.empty:
            fig = go.Figure(data=go.Heatmap(
                z=monthly.values, x=monthly.columns, y=monthly.index.astype(str),
                colorscale='RdYlGn', zmid=0,
                text=np.around(monthly.values, 1),
                texttemplate="%{text}%",
                textfont={"size": 11},
                hovertemplate="%{y} %{x}: %{z:.2f}%<extra></extra>",
                colorbar=dict(title="Return %"),
            ))
            fig.update_layout(
                title="Monthly Returns (%)",
                height=max(250, 50 * len(monthly)),
                xaxis_title="", yaxis_title="Year",
            )
            st.plotly_chart(fig, use_container_width=True)

            yearly = get_yearly_returns(port_equity_active)
            if not yearly.empty:
                fig = px.bar(
                    x=yearly.index.astype(str), y=yearly.values,
                    color=yearly.values, color_continuous_scale='RdYlGn',
                    color_continuous_midpoint=0,
                    title="Yearly Returns (%)",
                    labels={'x': 'Year', 'y': 'Return %'},
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # ============================================================
        # EXTENDED PORTFOLIO METRICS (quantstats-style)
        # ============================================================
        st.divider()
        st.subheader("📐 Risk-Adjusted & Distribution Metrics")
        st.caption(
            "Beyond Sharpe: distribution shape, drawdown texture, autocorrelation-deflated "
            "Sharpe, and benchmark-relative stats — all on the vol-targeted portfolio."
        )

        port_rets = port_equity_active.pct_change().fillna(0)

        # --- Row 1: Distribution shape ---
        skew_v, kurt_v = skew_kurtosis(port_rets)
        tail_v = tail_ratio(port_rets)
        cs_v = common_sense_ratio(port_rets)
        omega_v = omega_ratio(port_rets)
        rm1, rm2, rm3, rm4 = st.columns(4)
        rm1.metric("Skew", f"{skew_v:+.2f}",
                   delta="right tail" if skew_v > 0.3 else ("left tail" if skew_v < -0.3 else "symmetric"),
                   delta_color="normal" if skew_v > 0 else ("inverse" if skew_v < 0 else "off"),
                   help="Right-skew (+) = occasional big wins; left-skew (−) = occasional big losses. "
                        "Algo-trade target: positive skew.")
        rm2.metric("Kurtosis", f"{kurt_v:.2f}",
                   delta="fat tails" if kurt_v > 4 else "normal-ish",
                   delta_color="inverse" if kurt_v > 4 else "off",
                   help="Kurtosis = 3 for normal distribution. >4 = fatter tails than normal "
                        "(more extreme moves than Sharpe assumes).")
        rm3.metric("Tail Ratio", f"{tail_v:.2f}",
                   delta="upside dominant" if tail_v > 1.2 else ("downside dominant" if tail_v < 0.8 else "balanced"),
                   delta_color="normal" if tail_v > 1.2 else ("inverse" if tail_v < 0.8 else "off"),
                   help="P95 return / |P5 return|. >1 = right tail bigger than left (asymmetric upside).")
        rm4.metric("Common Sense Ratio", f"{cs_v:.2f}",
                   delta="robust edge" if cs_v > 1.5 else "weak",
                   delta_color="normal" if cs_v > 1.5 else "inverse",
                   help="Profit Factor × Tail Ratio. >1.5 = robust edge per quantstats heuristic.")

        # --- Row 2: Drawdown texture + autocorrelation-deflated Sharpe ---
        ulcer_v = ulcer_index(port_equity_active)
        upi_v = ulcer_performance_index(port_equity_active, risk_free_rate)
        recov_v = recovery_factor(
            (float(port_equity_active.iloc[-1]) / total_cap - 1),
            vs['MaxDD'],
        )
        smart_sh = smart_sharpe(port_rets, risk_free_rate)
        rm5, rm6, rm7, rm8 = st.columns(4)
        rm5.metric("Ulcer Index", f"{ulcer_v:.2f}%",
                   help="RMS drawdown — penalizes both depth AND duration of drawdowns. "
                        "Lower = smoother equity curve. Better than MaxDD alone for grinding pain.")
        rm6.metric("Ulcer Perf Index", f"{upi_v:.2f}",
                   delta="strong" if upi_v > 3 else ("moderate" if upi_v > 1 else "weak"),
                   delta_color="normal" if upi_v > 1 else "inverse",
                   help="(CAGR − rf) / Ulcer. Sharpe-like ratio using Ulcer as risk measure.")
        rm7.metric("Recovery Factor", f"{recov_v:.2f}",
                   delta="excellent" if recov_v > 10 else ("good" if recov_v > 3 else "weak"),
                   delta_color="normal" if recov_v > 3 else "inverse",
                   help="Total Return / |MaxDD|. >5 = strong recovery; <2 = drawdowns dominate gains.")
        rm8.metric("Smart Sharpe", f"{smart_sh:.2f}",
                   delta=f"vs raw {vs['Sharpe']:.2f}",
                   delta_color="off",
                   help="Sharpe deflated by 1-lag autocorrelation. Backtest curves with smoothed P&L "
                        "(e.g. averaging artifacts) get downweighted. Big gap = smoothing suspected.")

        # --- Row 3: Omega + Kelly + Benchmark-relative ---
        # BTC equity is normalized to start at total_cap, so check .std() > 0
        # (a flat curve at total_cap means fetch failed).
        bench_rets = None
        if 'B&H BTC Equity' in plot_data.columns and plot_data['B&H BTC Equity'].std() > 0:
            bench_rets = plot_data['B&H BTC Equity'].pct_change().fillna(0)
        # Kelly = μ/σ² → optimal LEVERAGE (not a bet fraction). For trader use, also
        # show "Quarter Kelly" (practical cap) and current portfolio leverage as context.
        kelly_lev = kelly_criterion(port_rets)
        cur_lev = vs.get('portfolio_scale', 1.0)
        quarter_kelly = kelly_lev * 0.25
        kelly_help = (
            f"Full Kelly = μ/σ² = theoretical max-growth leverage ({kelly_lev:.1f}×). "
            f"Practitioners use **¼-Kelly** ({quarter_kelly:.1f}×) for survival under estimation error. "
            f"Your current portfolio leverage is **{cur_lev:.2f}×** "
            f"({(cur_lev / kelly_lev * 100) if kelly_lev > 0 else 0:.1f}% of Kelly). "
            f"<½-Kelly = conservative, >Kelly = over-leveraged."
        )
        if bench_rets is not None and bench_rets.std() > 0:
            beta_v, alpha_v, corr_v = beta_alpha_correlation(port_rets, bench_rets, risk_free_rate)
            info_v = information_ratio(port_rets, bench_rets)
            treyn_v = treynor_ratio(port_rets, bench_rets, risk_free_rate)
            rm9, rm10, rm11, rm12 = st.columns(4)
            rm9.metric("Omega Ratio", f"{omega_v:.2f}",
                       delta="positive bias" if omega_v > 1.2 else "weak",
                       delta_color="normal" if omega_v > 1.2 else "inverse",
                       help="Σ(returns > 0) / |Σ(returns < 0)|. >1.5 = strongly positive bias even with skew.")
            rm10.metric("Kelly Leverage (Full / ¼)",
                        f"{kelly_lev:.1f}× / {quarter_kelly:.1f}×",
                        delta=f"you're at {cur_lev:.2f}×",
                        delta_color="off",
                        help=kelly_help)
            rm11.metric("β vs BTC", f"{beta_v:+.2f}",
                        delta=f"corr {corr_v:+.2f}", delta_color="off",
                        help="Beta = covariance with BTC / variance of BTC. β=0 → market-neutral, "
                             "β=1 → moves 1:1 with BTC. Crypto strategies should aim for low |β|.")
            rm12.metric("α vs BTC (annualized)", f"{alpha_v:+.1%}",
                        delta=f"Info Ratio {info_v:+.2f} · Treynor {treyn_v:+.1%}",
                        delta_color="off",
                        help="α = excess return after stripping out BTC exposure. Positive α = "
                             "you're adding value beyond just owning BTC.")
        else:
            rm9, rm10 = st.columns(2)
            rm9.metric("Omega Ratio", f"{omega_v:.2f}",
                       help="Σ(returns > 0) / |Σ(returns < 0)|. >1.5 = strongly positive bias.")
            rm10.metric("Kelly Leverage (Full / ¼)",
                        f"{kelly_lev:.1f}× / {quarter_kelly:.1f}×",
                        delta=f"you're at {cur_lev:.2f}×",
                        delta_color="off",
                        help=kelly_help)
            st.caption("⚠️ Beta/Alpha/Correlation vs BTC unavailable — Binance API fetch failed for this date range.")

        # Inline export — uses the active view
        st.divider()
        export_df = pd.DataFrame({
            'Date': port_equity_active.index,
            'Portfolio_Equity': port_equity_active.values,
            'Daily_PnL': port_pnl_active.reindex(port_equity_active.index).values,
            'Type': 'Entry',
        })
        export_label = "Vol-Targeted" if vt_view is not None else "Raw_Backtest"
        st.download_button(
            label=f"⬇️ Download {view_label} Equity Curve (CSV)",
            data=export_df.to_csv(index=False).encode(),
            file_name=f"Portfolio_{export_label}_Equity.csv",
            mime="text/csv",
        )

# ============================================================================
# TAB: STRATEGY BREAKDOWN
# ============================================================================

if active_tab == TAB_STRAT:
    st.subheader("Per-Strategy Performance")

    if metrics_df is None or metrics_df.empty:
        st.warning("No strategy metrics available.")
    else:
        sortable_cols = ['Sharpe', 'CAGR', 'Sortino', 'MaxDD', 'Calmar', 'PF',
                         'Win Rate', 'Expectancy', 'Trades', 'Net Profit']
        sort_col = st.selectbox("Sort by", sortable_cols, index=0)
        ascending = st.checkbox("Ascending", value=False)
        sorted_df = metrics_df.sort_values(sort_col, ascending=ascending)

        # Format display
        display_df = sorted_df.copy()
        for col in ['CAGR', 'MaxDD', 'Win Rate']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")
        for col in ['Sharpe', 'Sortino', 'Calmar', 'PF']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.2f}" if x < 100 else "∞"
                )
        for col in ['Net Profit', 'Avg Win', 'Avg Loss', 'Expectancy', 'Max Load']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}")
        if 'Trades/Yr' in display_df.columns:
            display_df['Trades/Yr'] = display_df['Trades/Yr'].apply(lambda x: f"{x:.0f}")

        st.dataframe(display_df, use_container_width=True, height=420)

        st.subheader("Performance Comparison")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                x=sorted_df.index, y=sorted_df['Sharpe'],
                color=sorted_df['Sharpe'],
                color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
                title="Sharpe Ratio by Strategy",
                labels={'x': 'Strategy', 'y': 'Sharpe'},
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(
                sorted_df.reset_index(),
                x='MaxDD', y='CAGR',
                size=np.abs(sorted_df['Net Profit']) + 1,
                color='Sharpe',
                color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
                hover_name='Strategy',
                title="CAGR vs Drawdown (bubble = |profit|)",
            )
            fig.update_layout(height=400)
            fig.update_xaxes(tickformat='.1%')
            fig.update_yaxes(tickformat='.1%')
            st.plotly_chart(fig, use_container_width=True)

        # Trade-level micro-stats
        st.subheader("Trade-Level Stats")
        tcol1, tcol2, tcol3 = st.columns(3)
        with tcol1:
            fig = px.histogram(
                sorted_df, x='Win Rate', nbins=20,
                title="Win Rate Distribution", labels={'x': 'Win Rate'},
            )
            fig.update_layout(height=300, showlegend=False)
            fig.update_xaxes(tickformat='.0%')
            st.plotly_chart(fig, use_container_width=True)
        with tcol2:
            payoff = (sorted_df['Avg Win'] / sorted_df['Avg Loss'].abs()).fillna(0)
            payoff_df = payoff.reset_index()
            payoff_df.columns = ['Strategy', 'Payoff']
            fig = px.bar(
                payoff_df, x='Strategy', y='Payoff',
                color='Payoff', color_continuous_scale='RdYlGn', color_continuous_midpoint=1,
                title="Payoff Ratio (avg win / |avg loss|)",
            )
            fig.update_layout(height=300, xaxis_tickangle=-45, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with tcol3:
            tpy = sorted_df['Trades/Yr']
            tpy_df = tpy.reset_index()
            tpy_df.columns = ['Strategy', 'Trades/Yr']
            fig = px.bar(
                tpy_df, x='Strategy', y='Trades/Yr',
                title="Activity (trades per year)",
            )
            fig.update_layout(height=300, xaxis_tickangle=-45, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ---- Deflated Sharpe Ratio (vol-targeted portfolio) ----
        st.subheader("🔬 Deflated Sharpe Ratio (selection-bias adjusted)")
        st.caption("""
        Bailey & López de Prado (2014). Adjusts the **vol-targeted portfolio's** Sharpe for the number of
        strategies you tested before keeping these. Higher PSR = more confident the Sharpe is real,
        not a multiple-testing artifact. Set **# strategies tested** in the sidebar.
        """)

        if vt_view is None:
            st.info("Vol-targeted portfolio not yet computed.")
        else:
            vt_eq = vt_view['equity']
            vt_rets = vt_eq.pct_change().fillna(0)
            vt_sharpe = vt_view['stats']['Sharpe']
            skew = float(vt_rets.skew())
            kurt = float(vt_rets.kurt() + 3)
            dsr = deflated_sharpe(vt_sharpe, n_strategies_tested, len(vt_rets), skew, kurt)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Vol-Targeted Sharpe", f"{vt_sharpe:.2f}")
            c2.metric(f"E[Max Sharpe] @ {n_strategies_tested} trials",
                      f"{dsr['expected_max_sharpe']:.2f}",
                      help="Expected max Sharpe from N random strategies with no edge")
            c3.metric("PSR (Probabilistic Sharpe)", f"{dsr['psr']:.1%}",
                      "✅ Significant" if dsr['is_significant'] else "⚠️ Not significant",
                      delta_color="off")
            c4.metric("Distribution", f"skew {skew:.2f}",
                      delta=f"kurt {kurt:.2f}",
                      delta_color="off",
                      help="Normal distribution has skew=0, kurt=3. Deviations affect Sharpe credibility.")

            with st.expander("📖 How to read the Deflated Sharpe Ratio"):
                st.markdown(f"""
                - **PSR**: probability that your **true** vol-targeted Sharpe exceeds the expected max under the null (no edge).
                - **Significant** = PSR ≥ 95%.
                - Your vol-targeted Sharpe of **{vt_sharpe:.2f}** vs expected max of **{dsr['expected_max_sharpe']:.2f}** at **{n_strategies_tested}** trials → **PSR = {dsr['psr']:.1%}**.
                - **Skew** = {skew:.2f}, **Kurtosis** = {kurt:.2f} (normal = 3).
                - Bump `# strategies tested` in sidebar to test sensitivity to selection bias.
                """)

            with st.expander("📊 PSR Sensitivity to # Strategies Tested"):
                trial_range = [10, 25, 50, 100, 200, 500, 1000, 2500, 5000]
                sens_data = []
                for nt in trial_range:
                    d = deflated_sharpe(vt_sharpe, nt, len(vt_rets), skew, kurt)
                    sens_data.append({'# Trials': nt, 'PSR': d['psr']})
                sens_df = pd.DataFrame(sens_data)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=sens_df['# Trials'], y=sens_df['PSR'] * 100,
                    mode='lines+markers', line=dict(color='#3498db', width=3),
                    name='PSR %',
                ))
                fig.add_hline(y=95, line_dash="dash", line_color="orange",
                              annotation_text="95% Significance")
                fig.update_layout(
                    title="PSR vs Number of Strategies Tested (Vol-Targeted Sharpe)",
                    xaxis_title="# Strategies Tested (log)",
                    yaxis_title="PSR %",
                    xaxis_type="log", height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.download_button(
            label="⬇️ Download Strategy Metrics (CSV)",
            data=df_to_csv_bytes(metrics_df),
            file_name="strategy_metrics.csv",
            mime="text/csv",
        )

# ============================================================================
# TAB: WALK-FORWARD
# ============================================================================

if active_tab == TAB_WF:
    st.subheader("Walk-Forward Robustness (K-Fold OOS)")
    st.caption("""
    Splits each strategy's history into N equal-time folds and scores each independently.
    A robust strategy is positive across **most** folds; one that crushes a single fold
    and bleeds the rest is curve-fit. Use the **KEEP / CUT** column to prune.
    """)

    if plot_data is None or plot_data.empty:
        st.error("No portfolio data loaded.")
    else:
        cfg1, cfg2, cfg3, cfg4 = st.columns(4)
        with cfg1:
            n_folds = st.slider("Number of folds", 3, 12, 5)
        with cfg2:
            keep_threshold = st.slider("KEEP threshold (% positive folds)", 0, 100, 70) / 100
        with cfg3:
            sharpe_threshold = st.number_input("Min Sharpe to count as 'positive'",
                                               value=0.0, step=0.1)
        with cfg4:
            # Calmar default — CAGR / |MaxDD| is the most drawdown-aware
            # single-number metric for walk-forward robustness. A strategy
            # with high Sharpe but a brutal tail drawdown looks much worse
            # in Calmar, which is what you want to see in a robustness check.
            heatmap_metrics = ["Calmar", "Sharpe", "CAGR", "MaxDD", "PF"]
            metric_view = st.selectbox(
                "Heatmap metric", heatmap_metrics, index=0,
                help="Calmar = CAGR / |MaxDD|. Higher = better risk-adjusted return per unit of worst-case pain.",
            )

        with st.spinner(f"Running {n_folds}-fold walk-forward on vol-targeted portfolio..."):
            wf = walk_forward_analysis(
                plot_data, total_cap, risk_free_rate, n_folds,
                portfolio_pnl=vt_pnl,  # vol-targeted P&L for portfolio fold metrics
            )

        if not wf['folds']:
            st.warning("No fold data available.")
        else:
            st.markdown("#### 📅 Fold Periods")
            fold_dates_df = pd.DataFrame([
                {
                    'Fold': f"Fold {i+1}",
                    'Start': s.strftime('%Y-%m-%d'),
                    'End': e.strftime('%Y-%m-%d'),
                    'Days': (e - s).days + 1,
                    'Portfolio Sharpe': f"{wf['portfolio'][i]['sharpe']:.2f}",
                    'Portfolio CAGR': f"{wf['portfolio'][i]['cagr']:.1%}",
                    'Portfolio MaxDD': f"{wf['portfolio'][i]['mdd']:.2%}",
                }
                for i, (s, e) in enumerate(wf['folds'])
            ])
            st.dataframe(fold_dates_df, hide_index=True, use_container_width=True)

            st.markdown(f"#### 🔥 Strategy × Fold {metric_view} Heatmap")
            metric_key = {
                'Calmar': 'calmar', 'Sharpe': 'sharpe', 'CAGR': 'cagr',
                'MaxDD': 'mdd', 'PF': 'pf',
            }[metric_view]
            matrix = pd.DataFrame({
                f"Fold {i+1}": [wf['strategies'][s][i][metric_key] for s in wf['strategies']]
                for i in range(n_folds)
            }, index=list(wf['strategies'].keys()))

            # Formatting + colormap midpoint per metric. zmid sets the green/red
            # threshold: 0 for Sharpe/CAGR/Calmar (positive=green), 1.0 for PF
            # (>1.0 is profitable), mean of matrix for MaxDD (relative ranking).
            if metric_view == 'CAGR':
                text = np.array([[f"{v:.1%}" for v in row] for row in matrix.values])
                zmid = 0
            elif metric_view == 'MaxDD':
                text = np.array([[f"{v:.1%}" for v in row] for row in matrix.values])
                zmid = matrix.values.mean()
            elif metric_view == 'PF':
                text = np.array([[f"{v:.2f}" for v in row] for row in matrix.values])
                zmid = 1.0
            elif metric_view == 'Calmar':
                text = np.array([[f"{v:.2f}" for v in row] for row in matrix.values])
                zmid = 0
            else:  # Sharpe
                text = np.array([[f"{v:.2f}" for v in row] for row in matrix.values])
                zmid = 0
            colorscale = 'RdYlGn_r' if metric_view == 'MaxDD' else 'RdYlGn'

            fig = go.Figure(data=go.Heatmap(
                z=matrix.values, x=matrix.columns, y=matrix.index,
                colorscale=colorscale, zmid=zmid,
                text=text, texttemplate="%{text}", textfont={"size": 11},
                colorbar=dict(title=metric_view),
                hovertemplate="%{y}<br>%{x}<br>" + metric_view + ": %{text}<extra></extra>",
            ))
            fig.update_layout(
                height=max(400, 32 * len(matrix)),
                xaxis_title="Time-fold (oldest → newest)", yaxis_title="Strategy",
                margin=dict(l=10, r=10, t=30, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Robustness scorecard
            st.markdown("#### ✅ Robustness Scorecard")
            scorecard_rows = []
            for strat, folds in wf['strategies'].items():
                rs = robustness_score(folds, sharpe_threshold)
                decision = "✅ KEEP" if rs['pct_positive_sharpe'] >= keep_threshold else "❌ CUT"
                scorecard_rows.append({
                    'Strategy': strat,
                    'Mean Sharpe': round(rs['mean_sharpe'], 2),
                    'Min Sharpe': round(rs['min_sharpe'], 2),
                    'Std Sharpe': round(rs['std_sharpe'], 2),
                    'Consistency': round(rs['consistency'], 2),
                    '% Positive': f"{rs['pct_positive_sharpe']:.0%}",
                    '% PF>1': f"{rs['pct_pf_above_1']:.0%}",
                    'Total P&L': round(rs['total_pnl'], 2),
                    'Decision': decision,
                })
            scorecard_df = pd.DataFrame(scorecard_rows)
            scorecard_df['_pct'] = scorecard_df['% Positive'].str.rstrip('%').astype(float)
            scorecard_df = scorecard_df.sort_values(
                ['_pct', 'Mean Sharpe'], ascending=[False, False]
            ).drop(columns=['_pct'])
            st.dataframe(scorecard_df, hide_index=True, use_container_width=True, height=420)

            keep_strats = [r['Strategy'] for r in scorecard_rows if 'KEEP' in r['Decision']]
            cut_strats = [r['Strategy'] for r in scorecard_rows if 'CUT' in r['Decision']]

            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("✅ Keep", len(keep_strats),
                       delta=f"{len(keep_strats)/len(scorecard_rows):.0%} of {len(scorecard_rows)}")
            sm2.metric("❌ Cut", len(cut_strats),
                       delta=f"{len(cut_strats)/len(scorecard_rows):.0%} of {len(scorecard_rows)}",
                       delta_color="inverse")

            all_strat_cols = [c for c in plot_data.columns if c not in PORTFOLIO_RESERVED_COLS]
            # Baseline = vol-targeted portfolio (consistent with rest of app)
            current_sharpe = vt_view['stats']['Sharpe'] if vt_view is not None else port_stats.get('Sharpe', 0)
            current_cagr = vt_view['stats']['CAGR'] if vt_view is not None else port_stats.get('CAGR', 0)
            if cut_strats and keep_strats:
                # Pruned portfolio at equal-weight raw sizing (proxy — for an exact
                # vol-targeted re-sizing you'd re-run MC+Vol Targeting on the kept set)
                kept_pnl_per_strat = total_cap / len(all_strat_cols)
                new_per_strat = total_cap / len(keep_strats)
                scale = new_per_strat / kept_pnl_per_strat
                kept_daily_pnl = plot_data[keep_strats].sum(axis=1) * scale
                kept_equity = kept_daily_pnl.cumsum() + total_cap
                kept_rets = kept_equity.pct_change().fillna(0)
                kept_sharpe, _ = get_risk_ratios(kept_rets, risk_free_rate)
                kept_days = max((kept_equity.index[-1] - kept_equity.index[0]).days, 1)
                kept_cagr = get_cagr(total_cap, float(kept_equity.iloc[-1]), kept_days)
                sm3.metric("Pruned Sharpe (raw equal-weight)", f"{kept_sharpe:.2f}",
                           f"{kept_sharpe - current_sharpe:+.2f} vs vol-targeted current")
                sm4.metric("Pruned CAGR (raw equal-weight)", f"{kept_cagr:.1%}",
                           f"{kept_cagr - current_cagr:+.1%} vs vol-targeted current")
            else:
                sm3.metric("Pruned Sharpe", f"{current_sharpe:.2f}", "no change")
                sm4.metric("Pruned CAGR", f"{current_cagr:.1%}", "no change")

            # Portfolio aggregate per fold
            st.markdown("#### 📊 Portfolio Aggregate by Fold")
            port_df = pd.DataFrame(wf['portfolio'])
            port_df['Fold'] = [f"Fold {i+1}" for i in range(len(port_df))]
            cc1, cc2 = st.columns(2)
            with cc1:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=port_df['Fold'], y=port_df['sharpe'],
                    marker_color=['#27ae60' if s > 0 else '#e74c3c' for s in port_df['sharpe']],
                    text=[f"{s:.2f}" for s in port_df['sharpe']], textposition='outside',
                ))
                fig.add_hline(y=0, line_color='black', line_width=1)
                fig.update_layout(title="Portfolio Sharpe by Fold", height=350,
                                  yaxis_title="Sharpe", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with cc2:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=port_df['Fold'], y=port_df['cagr'] * 100,
                    marker_color=['#27ae60' if c > 0 else '#e74c3c' for c in port_df['cagr']],
                    text=[f"{c:.1%}" for c in port_df['cagr']], textposition='outside',
                ))
                fig.add_hline(y=0, line_color='black', line_width=1)
                fig.update_layout(title="Portfolio Return by Fold", height=350,
                                  yaxis_title="Return %", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("📈 Per-Strategy Sharpe Trajectory Across Folds"):
                fig = go.Figure()
                for strat in wf['strategies']:
                    sharpes = [wf['strategies'][strat][i]['sharpe'] for i in range(n_folds)]
                    fig.add_trace(go.Scatter(
                        x=[f"Fold {i+1}" for i in range(n_folds)], y=sharpes,
                        mode='lines+markers', name=strat[:40], line=dict(width=1),
                    ))
                fig.add_hline(y=0, line_dash="dash", line_color="black")
                fig.update_layout(
                    title="Sharpe per fold (deteriorating slopes = decay)",
                    xaxis_title="Fold (oldest → newest)", yaxis_title="Sharpe",
                    height=500, hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 KEEP / CUT lists (copy-paste)"):
                col_keep, col_cut = st.columns(2)
                with col_keep:
                    st.write(f"**✅ KEEP ({len(keep_strats)})**")
                    st.code("\n".join(keep_strats) or "(none)", language=None)
                with col_cut:
                    st.write(f"**❌ CUT ({len(cut_strats)})**")
                    st.code("\n".join(cut_strats) or "(none)", language=None)

# ============================================================================
# TAB: RISK & REGIME
# ============================================================================

if active_tab == TAB_RISK:
    if plot_data is None or plot_data.empty:
        st.error("No portfolio data.")
    else:
        st.subheader("Drawdown Anatomy")
        # Always uses vol-targeted portfolio (canonical series from top of script)
        risk_equity = vt_equity
        risk_dd = vt_dd
        mdd_v = (vt_view['stats']['MaxDD'] if vt_view is not None
                 else port_stats.get('MaxDD', 0))
        mdd_label = "Vol-Targeted Portfolio"

        # All anatomy stats now come from the SAME (vol-targeted) curve AND the
        # SAME dd_series, so headline MaxDD / Longest DD / Peak-Trough / Worst-5
        # all agree numerically. Previously each used a slightly different
        # convention (peak-relative vs starting-cap-relative).
        max_dur_days = get_max_duration(
            risk_equity.index.to_series().reset_index(drop=True),
            risk_dd.reset_index(drop=True),
        )
        # Pass dd_series so MDD Peak/Trough align with headline MaxDD's episode
        _, peak_date_str, trough_date_str = get_mdd_info(risk_equity, total_cap, dd_series=risk_dd)

        # Use the SAME dd_series as the headline MaxDD so depths reconcile
        wd_full = worst_drawdowns(risk_equity, n=10, dd_series=risk_dd)
        deepest_dur = int(wd_full.iloc[0]['Days']) if not wd_full.empty else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Max DD ({mdd_label})", f"{mdd_v:.2%}", delta_color="inverse",
                  help=f"Deepest drawdown on the {mdd_label}.")
        c2.metric("Longest DD Days", f"{max_dur_days} days",
                  delta=f"Deepest-DD duration: {deepest_dur}d",
                  delta_color="off",
                  help="Longest TIME spent underwater (any episode). The deepest drawdown "
                       "may be a different (shorter) episode — shown in delta.")
        c3.metric("MDD Peak", peak_date_str,
                  help=f"Date equity peaked just before the deepest drawdown ({mdd_label}).")
        c4.metric("MDD Trough", trough_date_str,
                  help=f"Date equity bottomed at the deepest drawdown ({mdd_label}).")

        # ---- Worst 5 Drawdowns table ----
        st.markdown("##### ⚠️ Worst 5 Drawdowns")
        st.caption(
            f"Top 5 drawdown episodes on the **{mdd_label}** sorted by depth. "
            "Days = first-underwater to recovery (or to last date if not recovered). "
            "Note: the **longest** episode may not be #1 by depth — see 'Longest DD Days' above."
        )

        # ---- DD consistency sanity check (catches stale-cache mismatches) ----
        # Headline MaxDD ↑ MUST equal Worst-5 row #1 because they both come from
        # the SAME vt_view['dd_series']. If they diverge, the cache went stale —
        # surface it loudly so the user can refresh instead of trusting bad data.
        if not wd_full.empty:
            top_wd_pct = float(wd_full.iloc[0]['MaxDD %'])  # negative, in % units
            headline_pct = float(mdd_v) * 100  # convert fraction → %
            diff_pp = abs(top_wd_pct - headline_pct)
            if diff_pp > 0.5:
                st.error(
                    f"⚠️ DD MISMATCH: headline shows **{headline_pct:.2f}%** but worst-5 "
                    f"row #1 is **{top_wd_pct:.2f}%** (diff = {diff_pp:.2f}pp). "
                    f"This indicates a stale cache between vt_view and dd_series. "
                    f"Try refreshing the browser (Cmd+Shift+R) or restart the app."
                )

        wd = wd_full.head(5)
        if not wd.empty:
            wd_disp = wd.copy()
            wd_disp['Start'] = pd.to_datetime(wd_disp['Start']).dt.strftime('%Y-%m-%d')
            wd_disp['Valley'] = pd.to_datetime(wd_disp['Valley']).dt.strftime('%Y-%m-%d')
            wd_disp['End'] = pd.to_datetime(wd_disp['End']).dt.strftime('%Y-%m-%d')
            wd_disp.insert(0, '#', range(1, len(wd_disp) + 1))
            st.dataframe(
                wd_disp, hide_index=True, use_container_width=True,
                column_config={
                    'MaxDD %': st.column_config.NumberColumn(format="%.2f%%"),
                    'Days': st.column_config.NumberColumn(format="%d"),
                },
            )
        else:
            st.info("No drawdown episodes detected in this window.")

        st.divider()

        # Regime analysis
        st.subheader("🌊 Regime Conditional Performance")
        st.caption("BTC trend regime classification — exposes whether the edge is concentrated in one market type.")

        if 'B&H BTC Equity' in plot_data.columns:
            regime = classify_regimes(
                plot_data['B&H BTC Equity'], regime_lookback,
                regime_bull_thr, regime_bear_thr,
            )

            # ---- VISUAL REGIME CHECK: BTC + Portfolio with regime bands ----
            st.markdown("##### 🗺️ Regime Map — verify your settings are sane")
            st.caption(f"Bands show classification under current settings ({regime_lookback}d lookback, +{regime_bull_thr:.1%} / {regime_bear_thr:.1%} thresholds). If the 2022 crash isn't all red or Q4-2023 isn't mostly green, tweak the thresholds in the sidebar.")

            segments = regime_segments(regime)
            regime_colors_solid = {
                'Bull': '#27ae60',
                'Bear': '#e74c3c',
                'Chop': '#f1c40f',
            }
            regime_colors_band = {
                'Bull': 'rgba(46, 204, 113, 0.20)',
                'Bear': 'rgba(231, 76, 60, 0.20)',
                'Chop': 'rgba(241, 196, 15, 0.15)',
            }

            btc_norm = plot_data['B&H BTC Equity']
            port_eq = vt_equity  # vol-targeted portfolio (canonical)
            btc_roc = (btc_norm.pct_change(regime_lookback) * 100).fillna(0)

            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True, vertical_spacing=0.04,
                row_heights=[0.06, 0.62, 0.32],
                subplot_titles=(
                    "Regime Ribbon",
                    "BTC Price (orange) + Mochi Portfolio (green) — same colored bands behind",
                    f"BTC {regime_lookback}-Day Rolling Return % (the classifier signal)",
                ),
                specs=[[{"secondary_y": False}],
                       [{"secondary_y": True}],
                       [{"secondary_y": False}]],
            )

            # ---- Row 1: Regime ribbon (solid colored bars) ----
            for regime_type, seg_start, seg_end, n_days in segments:
                fig.add_trace(
                    go.Scatter(
                        x=[seg_start, seg_end, seg_end, seg_start, seg_start],
                        y=[0, 0, 1, 1, 0],
                        fill='toself',
                        fillcolor=regime_colors_solid.get(regime_type, '#888'),
                        line=dict(width=0),
                        mode='lines',
                        showlegend=False,
                        hovertemplate=f"{regime_type}<br>{seg_start.strftime('%Y-%m-%d')} → {seg_end.strftime('%Y-%m-%d')}<br>{n_days} days<extra></extra>",
                    ),
                    row=1, col=1,
                )

            # ---- Rows 2 & 3: subtle background bands for context ----
            for regime_type, seg_start, seg_end, _ in segments:
                fillcolor = regime_colors_band.get(regime_type, 'rgba(128,128,128,0.05)')
                fig.add_vrect(
                    x0=seg_start, x1=seg_end,
                    fillcolor=fillcolor, line_width=0, layer='below',
                    row=2, col=1,
                )
                fig.add_vrect(
                    x0=seg_start, x1=seg_end,
                    fillcolor=fillcolor, line_width=0, layer='below',
                    row=3, col=1,
                )

            # ---- Row 2: BTC + Portfolio ----
            fig.add_trace(
                go.Scatter(
                    x=btc_norm.index, y=btc_norm.values,
                    name='BTC (B&H equity)',
                    line=dict(color='#f39c12', width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>BTC: $%{y:,.0f}<extra></extra>",
                ),
                row=2, col=1, secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=port_eq.index, y=port_eq.values,
                    name='Mochi Portfolio',
                    line=dict(color='#27ae60', width=2.5),
                    hovertemplate="%{x|%Y-%m-%d}<br>Portfolio: $%{y:,.0f}<extra></extra>",
                ),
                row=2, col=1, secondary_y=True,
            )

            # ---- Row 3: BTC ROC + thresholds ----
            fig.add_trace(
                go.Scatter(
                    x=btc_roc.index, y=btc_roc.values,
                    name=f'BTC {regime_lookback}d ROC',
                    line=dict(color='#3498db', width=1.5),
                    fill='tozeroy', fillcolor='rgba(52,152,219,0.10)',
                    hovertemplate="%{x|%Y-%m-%d}<br>ROC: %{y:.1f}%<extra></extra>",
                ),
                row=3, col=1,
            )
            fig.add_hline(y=regime_bull_thr * 100, line_dash="dash", line_color="#27ae60",
                          annotation_text=f"Bull (+{regime_bull_thr:.0%})",
                          annotation_position="right", row=3, col=1)
            fig.add_hline(y=regime_bear_thr * 100, line_dash="dash", line_color="#e74c3c",
                          annotation_text=f"Bear ({regime_bear_thr:.0%})",
                          annotation_position="right", row=3, col=1)
            fig.add_hline(y=0, line_dash="dot", line_color="grey", row=3, col=1)

            # Hide regime ribbon y-axis ticks
            fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)
            fig.update_xaxes(showticklabels=False, row=1, col=1)
            fig.update_yaxes(title_text="BTC ($)", secondary_y=False, row=2, col=1)
            fig.update_yaxes(title_text="Portfolio ($)", secondary_y=True, row=2, col=1)
            fig.update_yaxes(title_text="Rolling Return %", row=3, col=1)
            fig.update_xaxes(title_text="Date", row=3, col=1)
            fig.update_layout(
                height=700, hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=80, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Legend chips for the bands
            chip1, chip2, chip3, chip4 = st.columns([1, 1, 1, 4])
            chip1.markdown("🟩 **Bull** (BTC ↑ ≥ threshold)")
            chip2.markdown("🟥 **Bear** (BTC ↓ ≥ threshold)")
            chip3.markdown("🟨 **Chop** (in between)")
            n_bull = int((regime == 'Bull').sum())
            n_bear = int((regime == 'Bear').sum())
            n_chop = int((regime == 'Chop').sum())
            chip4.markdown(
                f"<small>Days: 🟩 {n_bull} ({n_bull/len(regime):.0%}) · "
                f"🟥 {n_bear} ({n_bear/len(regime):.0%}) · "
                f"🟨 {n_chop} ({n_chop/len(regime):.0%})</small>",
                unsafe_allow_html=True,
            )

            # Regime segment timeline
            with st.expander("📅 Regime Segments (chronological list)"):
                seg_df = pd.DataFrame([
                    {
                        'Regime': r,
                        'Start': s.strftime('%Y-%m-%d'),
                        'End': e.strftime('%Y-%m-%d'),
                        'Days': d,
                    }
                    for r, s, e, d in segments
                ])
                st.caption(f"{len(seg_df)} regime switches over {len(regime)} days. Many switches = thresholds too tight or lookback too short.")
                st.dataframe(seg_df, hide_index=True, use_container_width=True, height=300)

            st.divider()

            reg_perf = regime_performance(plot_data, regime, total_cap, risk_free_rate)

            if not reg_perf.empty:
                # Format display
                disp = reg_perf.copy()
                disp['Pct of Time'] = disp['Pct of Time'].apply(lambda x: f"{x:.1%}")
                disp['Total P&L ($)'] = disp['Total P&L ($)'].apply(lambda x: f"${x:,.0f}")
                disp['Avg Daily P&L ($)'] = disp['Avg Daily P&L ($)'].apply(lambda x: f"${x:,.2f}")
                disp['Daily Win Rate'] = disp['Daily Win Rate'].apply(lambda x: f"{x:.1%}")
                disp['Sharpe (annualized)'] = disp['Sharpe (annualized)'].apply(lambda x: f"{x:.2f}")
                disp['Best Day ($)'] = disp['Best Day ($)'].apply(lambda x: f"${x:,.2f}")
                disp['Worst Day ($)'] = disp['Worst Day ($)'].apply(lambda x: f"${x:,.2f}")
                st.dataframe(disp, hide_index=True, use_container_width=True)

                # P&L concentration callout
                total_pnl_all = reg_perf['Total P&L ($)'].sum()
                if total_pnl_all > 0:
                    bull_share = float(reg_perf[reg_perf['Regime'] == 'Bull']['Total P&L ($)'].sum()) / total_pnl_all
                    bear_share = float(reg_perf[reg_perf['Regime'] == 'Bear']['Total P&L ($)'].sum()) / total_pnl_all
                    chop_share = float(reg_perf[reg_perf['Regime'] == 'Chop']['Total P&L ($)'].sum()) / total_pnl_all
                    cc1, cc2, cc3 = st.columns(3)
                    cc1.metric("🐂 Bull P&L share", f"{bull_share:.0%}")
                    cc2.metric("🐻 Bear P&L share", f"{bear_share:.0%}")
                    cc3.metric("🦘 Chop P&L share", f"{chop_share:.0%}")
                    if bull_share > 0.7:
                        st.warning(f"⚠️ {bull_share:.0%} of profit is from Bull regimes — your edge is bull-biased. Will struggle in extended bears/chops.")
                    elif bull_share < 0.4 and bear_share < 0.4:
                        st.success("✅ P&L is well distributed across regimes.")

                # Strategy × Regime heatmap
                strat_reg = per_strategy_regime_pnl(plot_data, regime)
                if not strat_reg.empty:
                    st.markdown("##### Strategy P&L by Regime")
                    fig = go.Figure(data=go.Heatmap(
                        z=strat_reg.values,
                        x=strat_reg.columns, y=strat_reg.index,
                        colorscale='RdYlGn', zmid=0,
                        text=np.around(strat_reg.values, 0),
                        texttemplate="$%{text}", textfont={"size": 10},
                        colorbar=dict(title="P&L ($)"),
                    ))
                    fig.update_layout(
                        height=max(400, 30 * len(strat_reg)),
                        xaxis_title="Regime", yaxis_title="Strategy",
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("BTC benchmark not available — regime classification disabled.")

        st.divider()

        # Rolling correlation
        st.subheader("Rolling 30-Day Correlation (Strategies vs Vol-Targeted Portfolio)")
        ignore_cols = PORTFOLIO_RESERVED_COLS
        strategy_cols = [c for c in plot_data.columns if c not in ignore_cols]

        if len(strategy_cols) > 1:
            port_pnl_diff = vt_pnl  # vol-targeted P&L (canonical)
            avg_corr = pd.Series(0.0, index=plot_data.index)
            valid = 0
            for col in strategy_cols:
                if plot_data[col].std() > 0:
                    rolling = plot_data[col].rolling(window=30).corr(port_pnl_diff).fillna(0)
                    avg_corr += rolling
                    valid += 1
            if valid > 0:
                avg_corr /= valid

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=plot_data.index, y=avg_corr,
                line=dict(color='#8e44ad', width=2),
                fill='tozeroy', fillcolor='rgba(142,68,173,0.1)',
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            fig.add_hline(y=0.5, line_dash="dot", line_color="orange",
                          annotation_text="0.5 (warning)")
            fig.update_layout(
                title="Average 30-Day Rolling Correlation",
                xaxis_title="Date", yaxis_title="Correlation",
                hovermode="x", height=400, yaxis=dict(range=[-1, 1]),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Two-panel correlation: full vs stress
        st.subheader("Exposure Correlation: Calm vs Stress")
        st.caption("Side-by-side: average correlation (left) vs correlation on the worst N% portfolio days (right). The right matrix is the one that matters during deleveraging events.")

        stress_pct = st.slider("Stress percentile (worst N% of days)", 5, 25, 10, key="stress_pct")

        col_calm, col_stress = st.columns(2)
        with col_calm:
            st.markdown("##### Full-period (calm)")
            if exposure_df is not None and not exposure_df.empty and exposure_df.shape[1] > 1:
                df_clean = exposure_df.fillna(0).copy()
                df_clean['TOTAL'] = df_clean.sum(axis=1)
                cm = df_clean.corr()
                cols = ['TOTAL'] + [c for c in cm.columns if c != 'TOTAL']
                cm = cm.loc[cols, cols]
                fig = go.Figure(data=go.Heatmap(
                    z=cm.values, x=cm.columns, y=cm.index,
                    colorscale='RdBu_r', zmid=0, zmin=-1, zmax=1,
                    text=np.around(cm.values, 2), texttemplate="%{text}",
                    textfont={"size": 8}, showscale=False,
                ))
                fig.update_layout(height=max(400, 25 * len(cm)),
                                  xaxis_tickangle=-45, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need >1 strategy.")

        with col_stress:
            st.markdown(f"##### Worst {stress_pct}% of days (stress)")
            stress_corr = stress_correlation(plot_data, stress_pct)
            if not stress_corr.empty and stress_corr.shape[1] > 1:
                fig = go.Figure(data=go.Heatmap(
                    z=stress_corr.values, x=stress_corr.columns, y=stress_corr.index,
                    colorscale='RdBu_r', zmid=0, zmin=-1, zmax=1,
                    text=np.around(stress_corr.values, 2), texttemplate="%{text}",
                    textfont={"size": 8}, showscale=False,
                ))
                fig.update_layout(height=max(400, 25 * len(stress_corr)),
                                  xaxis_tickangle=-45, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

                # Quick diagnostic
                n = len(stress_corr)
                stress_off = stress_corr.values[~np.eye(n, dtype=bool)].mean()
                if exposure_df is not None and not exposure_df.empty:
                    cm_calm = exposure_df.fillna(0).corr()
                    n2 = len(cm_calm)
                    calm_off = cm_calm.values[~np.eye(n2, dtype=bool)].mean()
                    delta = stress_off - calm_off
                    st.metric("Mean off-diagonal correlation", f"{stress_off:+.3f}",
                              f"{delta:+.3f} vs calm",
                              delta_color="inverse" if delta > 0 else "normal")
            else:
                st.info("Need stress days.")

        # Portfolio load
        if 'Portfolio Load' in plot_data.columns and plot_data['Portfolio Load'].max() > 0:
            st.divider()
            st.subheader("Portfolio Capital Load Over Time")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=plot_data.index, y=plot_data['Portfolio Load'],
                fill='tozeroy', fillcolor='rgba(52,152,219,0.3)',
                line=dict(color='#2980b9'), name='Portfolio Load',
            ))
            fig.add_hline(y=total_cap, line_dash="dash", line_color="red",
                          annotation_text=f"Total Capital ${total_cap:,.0f}")
            fig.update_layout(
                title="Total Portfolio Exposure ($)",
                xaxis_title="Date", yaxis_title="Load ($)",
                hovermode="x", height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB: MONTE CARLO
# ============================================================================

if active_tab == TAB_MC:
    st.subheader("Portfolio Forward-Risk Simulation (Vol-Targeted, Net of Costs)")
    st.caption("""
    Block-bootstrap MC on the **vol-targeted portfolio's daily P&L (net of costs when "Apply costs" is on)**
    to estimate 1-year-forward **risk of ruin**, **VaR/CVaR**, **drawdown exceedance**, and equity-path
    percentiles. This is the **portfolio-level** complement to the per-strategy Pre/Post RoR in the sizing
    table below — diversification typically brings the portfolio RoR well below any single strategy's RoR.
    """)

    if vt_view is None:
        st.warning("⏳ Vol-targeted portfolio not yet computed. The auto-compute should finish shortly.")
    elif 'mc_results' not in st.session_state:
        st.info("⏳ Forward-Risk MC auto-computing... refresh in a moment.")
    else:
        daily_pnl_for_mc = vt_view['daily_pnl'].values
        mc_start_equity_eff = st.session_state.get('mc_start_used', float(total_cap))
        mc_ruin_equity_eff = st.session_state.get('mc_ruin_used', float(total_cap * VT_DEFAULT_RUIN_FRAC))

        info_col, btn_col = st.columns([4, 1])
        with info_col:
            st.info(
                f"**{len(daily_pnl_for_mc)} vol-targeted daily P&L observations** (net of costs) · "
                f"Start ${mc_start_equity_eff:,.0f} · Ruin ${mc_ruin_equity_eff:,.0f} "
                f"(loss of {(1 - VT_DEFAULT_RUIN_FRAC) * 100:.0f}%) · "
                f"{mc_n_runs:,} runs · {mc_trades_per_year}/yr trade frequency · block={mc_block_len}"
            )
        with btn_col:
            if st.button("🔄 Recompute MC", use_container_width=True,
                         help="Force a fresh MC run (e.g. with a new RNG seed). Auto runs on data changes."):
                # Bust the fingerprint so auto_compute_mc reruns on next reload
                st.session_state.pop('mc_fp', None)
                st.rerun()

        if 'mc_results' in st.session_state:
            mc_results = st.session_state['mc_results']
            rets = mc_results['return']
            mdd_pcts = mc_results['mdd_pct']
            ruined = mc_results['ruined']

            st.markdown("#### 🎯 Headline Risk Metrics")
            c1, c2, c3, c4 = st.columns(4)
            mc_start_used = st.session_state.get('mc_start_used', total_cap)
            mc_ruin_used = st.session_state.get('mc_ruin_used', total_cap * 0.6)
            c1.metric("Risk of Ruin", f"{(ruined.mean() * 100):.2f}%",
                      help=f"% of paths reaching ${mc_ruin_used:,.0f}")
            c2.metric("Prob(Return > 0)", f"{((rets > 0).mean() * 100):.1f}%")
            c3.metric("Median Return", f"{np.median(rets):.1%}")
            c4.metric("Median MDD", f"{np.median(mdd_pcts):.1%}")

            st.markdown("#### 📉 VaR & CVaR (1-Year Returns)")
            var_5, cvar_5 = get_var_cvar(rets, 0.05)
            var_1, cvar_1 = get_var_cvar(rets, 0.01)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("VaR 95%", f"{var_5:.1%}")
            c2.metric("CVaR 95%", f"{cvar_5:.1%}")
            c3.metric("VaR 99%", f"{var_1:.1%}")
            c4.metric("CVaR 99%", f"{cvar_1:.1%}")

            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=rets * 100, nbinsx=60, marker_color='steelblue',
                ))
                fig.add_vline(x=0, line_dash="dash", line_color="black")
                fig.add_vline(x=np.median(rets) * 100, line_dash="dot",
                              line_color="green", annotation_text="Median")
                fig.add_vline(x=var_5 * 100, line_dash="dot",
                              line_color="red", annotation_text="VaR 95%")
                fig.update_layout(
                    title="Annual Return Distribution",
                    xaxis_title="Return %", yaxis_title="Frequency",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=mdd_pcts * 100, nbinsx=60, marker_color='#e74c3c',
                ))
                fig.add_vline(x=np.median(mdd_pcts) * 100, line_dash="dot",
                              line_color="orange", annotation_text="Median")
                fig.update_layout(
                    title="Max Drawdown Distribution",
                    xaxis_title="MDD %", yaxis_title="Frequency",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### 📈 Sample Equity Paths")
            col1, col2 = st.columns([3, 1])
            with col2:
                n_paths = st.slider("Sample paths to show", 10, 500, 100, step=10)
                show_pct = st.checkbox("Show percentile bands", value=True)
            with col1:
                fig = go.Figure()
                eqp = mc_results["equity_paths"]
                sample_count = min(n_paths, len(eqp))
                rng = np.random.default_rng(42)
                sample_idx = rng.choice(len(eqp), sample_count, replace=False)
                for i in sample_idx:
                    path = eqp[i, :]
                    finite = np.isfinite(path)
                    fig.add_trace(go.Scatter(
                        x=np.arange(finite.sum()), y=path[finite], mode='lines',
                        line=dict(color='rgba(100,100,200,0.2)', width=1),
                        hoverinfo='skip', showlegend=False,
                    ))
                if show_pct:
                    finite_paths = np.where(np.isfinite(eqp), eqp, np.nan)
                    p5 = np.nanpercentile(finite_paths, 5, axis=0)
                    p50 = np.nanpercentile(finite_paths, 50, axis=0)
                    p95 = np.nanpercentile(finite_paths, 95, axis=0)
                    x_axis = np.arange(len(p50))
                    fig.add_trace(go.Scatter(
                        x=x_axis, y=p95, line=dict(color='#27ae60', width=2),
                        name='95th percentile',
                    ))
                    fig.add_trace(go.Scatter(
                        x=x_axis, y=p50, line=dict(color='#f39c12', width=3),
                        name='Median',
                    ))
                    fig.add_trace(go.Scatter(
                        x=x_axis, y=p5, line=dict(color='#e74c3c', width=2),
                        name='5th percentile',
                    ))
                fig.add_hline(y=mc_ruin_used, line_dash="dash", line_color="red",
                              annotation_text=f"Ruin: ${mc_ruin_used:,.0f}")
                fig.add_hline(y=mc_start_used, line_dash="dot", line_color="grey",
                              annotation_text=f"Start: ${mc_start_used:,.0f}")
                fig.update_layout(
                    title="Monte Carlo Equity Paths (1 year ahead)",
                    xaxis_title="Trade #", yaxis_title="Equity ($)",
                    height=500, hovermode="x",
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### 📊 Tail Risk & Percentiles")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**MDD Exceedance Probabilities**")
                exc_data = []
                for thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
                    prob = (mdd_pcts >= thresh).mean()
                    exc_data.append({
                        'Threshold': f"≥ {thresh:.0%}",
                        'Probability': f"{prob:.1%}",
                        'Count': int((mdd_pcts >= thresh).sum()),
                    })
                st.dataframe(pd.DataFrame(exc_data), hide_index=True, use_container_width=True)

            with col2:
                st.write("**Return Percentiles**")
                pct_data = []
                for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                    pct_data.append({
                        'Percentile': f"P{p}",
                        'Return': f"{np.percentile(rets, p):.1%}",
                        'Final Equity': f"${mc_start_used * (1 + np.percentile(rets, p)):,.0f}",
                    })
                st.dataframe(pd.DataFrame(pct_data), hide_index=True, use_container_width=True)

    # ----------------------------------------------------------------------
    # MC + VOL TARGETING (primary workflow)
    # ----------------------------------------------------------------------
    st.divider()
    st.subheader("📊 MC + Vol Targeting (Combined Workflow)")
    st.caption("""
    **Two-step sizing:**
    1. **Per-strategy MC**: binary-search max leverage such that simulated RoR ≤ target, capped at max leverage (default 2x). Equal capital allocation ($total/N each).
    2. **Portfolio vol scaling**: apply a **single uniform multiplier** to every strategy's position so total portfolio vol = target (default 20%).
    Result: each strategy is safely sized (RoR-controlled), then the whole book is leveraged to your portfolio vol target.
    """)

    if plot_data is None or plot_data.empty:
        st.warning("Load portfolio first.")
    else:
        # Entire VT section is wrapped in @st.fragment so the recompute button
        # doesn't reset tab focus or trigger a full app rerun.
        render_vt_recompute_section(
            plot_data=plot_data,
            metrics_df=metrics_df,
            total_cap=total_cap,
            risk_free_rate=risk_free_rate,
            exposure_df=exposure_df,
            applies_cost=applies_cost,
            cost_bps_rt=cost_bps_rt,
            slippage_bps=slippage_bps,
            funding_bps=funding_bps,
            mc_block_len=mc_block_len,
            mc_seed=mc_seed,
            port_stats=port_stats,
        )


# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.caption(
    f"📊 Mochi Portfolio Analytics · {oos_start} → {oos_end} · "
    f"Capital ${total_cap:,.0f} · RFR {risk_free_rate:.1%} · "
    f"Costs {cost_bps_rt + slippage_bps:.1f}bps/trade + {funding_bps:.2f}bps/day"
)
