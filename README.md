# Mochi Portfolio Analytics

A Streamlit dashboard for **systematic crypto algo-trading portfolio analysis** — combining backtest validation, walk-forward testing, vol-targeted position sizing, Monte Carlo risk simulation, and a statistically-grounded **live monitoring & kill-rule engine**.

Built for traders running multiple strategies on Bybit/Binance perpetuals who want institutional-quality risk management without the institutional toolchain.

---

## Table of contents

- [Why this exists](#why-this-exists)
- [Key features](#key-features)
- [Quick start](#quick-start)
- [Data format](#data-format)
- [App walkthrough](#app-walkthrough)
- [Statistical methodology](#statistical-methodology)
- [Kill rule design](#kill-rule-design)
- [Architecture](#architecture)
- [Configuration reference](#configuration-reference)
- [Roadmap and known limitations](#roadmap-and-known-limitations)

---

## Why this exists

Most retail algo-trading platforms (TradingView signal pages, MetaTrader signals, public Discord groups) give you a backtest and an equity curve. That's it.

The hard part of algo trading isn't designing strategies — it's **deciding when to keep, suspend, or kill** them once they're live. Most retail traders make this call on **vibes**: "this looks bad, let me archive it." That's survivorship-bias overfitting in disguise.

This dashboard provides the **honest answer** to that question, derived from rigorous statistics:

- Is the live drawdown beyond what backtest variation predicts? *(Monte Carlo envelope test)*
- Has the underlying edge actually changed, or is the strategy just unlucky? *(KS distribution drift)*
- Should I kill it, suspend it, or keep it? *(4-way decision matrix)*

All without leaking future information back into the historical sizing.

---

## Key features

### Statistical kill-switch engine
- **Dual-tail Monte Carlo envelope test**: `KILL if (MC DD %ile ≤ 5 OR MC Return %ile ≤ 5) AND live_trades ≥ 20`
- Catches **both** sharp crashes (MDD %ile) and slow-bleed decay (Return %ile)
- Sample-size floor prevents small-sample false positives
- WARN tier at ≤15% for early signals

### Edge diagnosis (kill vs suspend)
Kolmogorov-Smirnov test on per-trade P&L distribution distinguishes:
- **🔴 BROKEN EDGE** — kill fires AND distribution drifted → archive permanently
- **🟠 UNLUCKY** — kill fires but distribution intact → suspend, edge intact
- **🟡 EDGE DRIFTING** — kill quiet but distribution shifting → leading indicator, watch closely
- **🟢 STABLE** — both quiet → keep

### Honest out-of-sample design
- Vol-targeting derived **only** from backtest data
- Applied forward to live segment with no look-ahead leakage
- Live monitoring uses backtest-only sizing → genuine OOS validation

### Comprehensive analytics
- **Per-strategy & portfolio Monte Carlo** with block bootstrap (preserves autocorrelation)
- **Walk-forward K-fold OOS analysis** with Deflated Sharpe (Bailey & López de Prado 2014)
- **Regime classification** (60d / ±10% defaults) with regime-conditional health stats
- **Strategy family roll-up** for cross-ticker pair analysis (Fisher z-test for correlation divergence)
- **Per-strategy bootstrap equity envelope** with ±2σ confidence bands
- **Live trade log** with cumulative reconciliation for full auditability

### Production-ready UX
- Auto-refresh on data folder fingerprint change (no manual reload)
- Auto-fetched BTC benchmark from Binance public API (no CSV needed)
- Streamlit fragment-scoped reruns to preserve tab focus on widget interactions
- Cap-invariant percentile statistics — kill decisions don't shift with starting capital

---

## Quick start

### Requirements
- Python 3.10+
- ~500MB disk for dependencies

### Installation

```bash
git clone <this-repo-url> mochi-portfolio
cd mochi-portfolio
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### First-run setup

1. **Export trades from TradingView** — open your strategy, go to *Strategy Tester → List of Trades → ⋮ → Export*. Save as CSV.
2. **Place CSVs in a folder** anywhere on your machine, e.g. `~/trading/Portfolio/`. One CSV per strategy-ticker combination.
3. **Open the app sidebar → Data Sources** and paste the folder path.
4. The app auto-loads all `.csv` files, parses TradingView's format, and computes the full analytics suite.

That's it. The sidebar lets you tune capital, costs, regime params, and Monte Carlo settings; everything auto-recomputes when inputs change.

---

## Data format

The app expects **TradingView-exported "List of Trades" CSVs**. Both the legacy and the post-2024 column formats are auto-detected:

| Required column | Aliases supported |
|---|---|
| `Date/Time` | `Date and time` (TV 2024+) |
| `Type` | (must contain `Entry long`, `Entry short`, `Exit long`, etc.) |
| `Net P&L USDT` | (or other quote currency) |
| `Size` | `Position size (value)`, `Size (value)` |

Filename convention drives auto-extraction of strategy family, ticker, and direction:

```
{STRATEGY_FAMILY}_{DIRECTION}_{TIMEFRAME}_{EXCHANGE}_{TICKER}_{EXPORT_DATE}.csv

Example:
MR_VOTING_SYSTEM_BOTH_6H_BINANCE_BTCUSDT_2026-05-23.csv
                                                   │
        DIRECTION ─┴─── TIMEFRAME            EXPORT_DATE
        (LONG/SHORT/BOTH)
```

If your filenames don't follow this convention, the strategy will load but family/pair analytics will be limited.

---

## App walkthrough

The dashboard has six tabs, ordered by deployment workflow:

### 1. 📡 Live Monitoring (default landing)
**Per-strategy kill-switch evaluation.** Sortable table of all strategies with:
- Verdict (KILL / WARN / KEEP / INCUBATING)
- Edge Diagnosis (BROKEN / UNLUCKY / DRIFTING / STABLE)
- MC DD %ile + MC Return %ile (kill-rule inputs)
- KS p-value (distribution shift test)
- Live Trades (sample-size gate)
- Live % and Live MDD %

**Drill-down per strategy** opens charts for: equity vs MC envelope, rolling 60d Sharpe, drawdown depth, equity vs traded ticker price, MC distribution histograms with math-check captions, full live trade log.

### 2. 📊 Portfolio Overview
Vol-targeted portfolio equity curve vs BTC benchmark on dual axes (Return % left, Equity $ right). Performance metrics (CAGR / Sharpe / Sortino / Calmar), monthly/yearly heatmaps, quantstats-style extended stats (skew/kurtosis, omega, ulcer index, beta/alpha vs BTC).

### 3. 🎯 Strategy Breakdown
Per-strategy metrics, correlation heatmap, deflated Sharpe ratio (selection-bias corrected).

### 4. 🔄 Walk-Forward
K-fold OOS analysis with per-fold Sharpe trajectory and KEEP/CUT recommendation lists.

### 5. ⚠️ Risk Analytics
Drawdown anatomy (longest underwater, deepest DD, worst-5 episodes), VaR/CVaR, regime-conditional returns, stress correlation under various market segments.

### 6. 🎲 Monte Carlo + Vol Targeting
Two-step sizing workflow:
1. Per-strategy MC binary-search for max leverage given target RoR
2. Uniform portfolio scaling to hit target portfolio vol

Output: TradingView/Bybit deployment config (position $, required leverage per symbol).

---

## Statistical methodology

### Monte Carlo (block bootstrap)
- **Block length 5** to preserve return autocorrelation
- **1000 paths** at fixed seed 42 (reproducible)
- Per-strategy MC bootstraps backtest daily P&L; live observed value compared against the resulting distribution

### Kill-rule percentile interpretation
- `MC_DD_%ile = P(MC_path_DD ≤ live_DD)` — lower = worse (live drawdown is in the bad tail)
- `MC_Return_%ile = P(MC_path_return ≤ live_return)` — lower = worse (live return is in the bad tail)
- A 5% percentile means live is in the worst 1-in-20 outcome under the null hypothesis "no decay"

### Edge diagnosis logic

|  | KS p < 0.05 (distribution drifted) | KS p ≥ 0.05 (distribution intact) |
|---|---|---|
| **MC fires (kill rule triggers)** | 🔴 **BROKEN EDGE** — archive | 🟠 **UNLUCKY** — suspend, edge intact |
| **MC quiet** | 🟡 **EDGE DRIFTING** — watch | 🟢 **STABLE** — keep |

### Drawdown definition
Peak-based MDD: `MDD = max((peak − trough) / peak)`. This is the industry-standard finance definition. A $50 drop from a $200 peak is a 25% drawdown — independent of starting capital. *(Bug fix history: an earlier version divided by starting capital, which produced impossibly large %s when per-strategy capital was small relative to equity excursion.)*

### Deflated Sharpe Ratio
Walk-forward analysis includes Bailey & López de Prado's Deflated Sharpe Ratio, which corrects backtest Sharpe for the multiple-testing implied by trying many strategies. Reported on the Strategy Breakdown tab.

### Cost model
Default: 11bps round-trip fee + 2bps slippage + 0.5bps/day funding (matches Bybit perp typical retail tier). Adjustable in sidebar; applied to all P&L calculations when "Apply costs" is enabled.

---

## Kill rule design

```
KILL  if  (MC_DD_%ile ≤ 5 OR MC_Return_%ile ≤ 5)  AND  live_trades ≥ 20
WARN  if  (MC_DD_%ile ≤ 15 OR MC_Return_%ile ≤ 15) AND live_trades ≥ 20
INCUBATING  if  live_trades < 20
KEEP  otherwise
```

**Why dual-tail (OR semantics on DD and Return %ile):**
- MC DD %ile catches sharp crashes (e.g., a strategy that hits an unexpected -20% drawdown)
- MC Return %ile catches **slow-death bleeders** — strategies whose drawdowns never deepen but whose cumulative return drifts negative over time
- Single-criterion rules miss one or the other failure mode

**Why min 20 trades floor:**
- With 5-10 live trades, a single losing trade can push the percentile to ≤5% by pure noise
- 20 trades gives the per-trade distribution enough support to make tail tests meaningful
- This is a Type I error guard

**Why the WARN tier:**
- 15% threshold is roughly 1-in-7 outcomes — flags strategies before they cross the firing line
- Acts as an early-warning system without triggering kill action

**Why edge diagnosis is *separate* from the kill rule:**
- KS test is direction-blind (says "different" not "worse")
- Used to **interpret** a kill firing (broken edge vs unlucky), not to **trigger** one
- This keeps the kill rule MC-based and statistically clean

---

## Architecture

```
mochi-portfolio/
├── app.py              # Streamlit UI (~2950 LOC) — six tabs, sidebar config
├── calculations.py     # Analytics engine (~3430 LOC) — pure functions, no Streamlit
├── requirements.txt    # streamlit, pandas, numpy, plotly, scipy, requests
├── README.md
└── .gitignore          # Excludes user CSVs, pycache, OS files
```

**Design principles:**
1. **Separation of concerns**: `calculations.py` is pure (testable without Streamlit). `app.py` handles UI/state.
2. **Single source of truth**: kill thresholds, colors, defaults centralized as module constants in `calculations.py`.
3. **Fingerprint-based caching**: heavy computations cached in `st.session_state` keyed by data fingerprint (folder mtime + file count + config tuple). Auto-invalidates when inputs change.
4. **Fragment-scoped reruns**: Vol-targeting section wrapped in `@st.fragment` so widget interactions don't reset tab focus.
5. **Cap-invariant statistics**: MC percentiles compare dollar amounts directly (independent of per-strategy capital choice).

---

## Configuration reference

All settings live in the sidebar (collapsible expanders):

| Expander | Setting | Default | Notes |
|---|---|---:|---|
| 📁 Data Sources | Portfolio folder | (user path) | Folder of TradingView CSVs |
| 📅 Date Range | OOS Start | 2021-12-03 | Earliest date to include |
| 📅 Date Range | OOS End | 2026-05-22 | Latest date to include |
| 📅 Date Range | Live incubation start | 2025-12-03 | Split point for live monitoring |
| 💰 Risk Parameters | Total Capital | $2000 | Spreads equally across strategies for unleveraged baseline |
| 💰 Risk Parameters | Risk-free rate | 4% | For Sharpe/Sortino |
| 💸 Cost Model | Fee per round-trip | 11 bps | Bybit perp default |
| 💸 Cost Model | Slippage | 2 bps | Per round-trip |
| 💸 Cost Model | Funding | 0.5 bps/day | Bybit perp typical |
| 🎲 Monte Carlo | Runs | 5000 | Portfolio-level MC paths |
| 🎲 Monte Carlo | Trades per year | 365 | Portfolio MC = daily resolution |
| 🌊 Regime Classification | Lookback | 60 days | Longer = more stable regime labels |
| 🌊 Regime Classification | Bull threshold | +10% | Rolling return above this → BULL |
| 🌊 Regime Classification | Bear threshold | −10% | Rolling return below this → BEAR |

Kill-rule thresholds live in `calculations.py` as module constants:

```python
MC_TAIL_PCT = 5         # KILL fires at this percentile or below
MC_WARN_PCT = 15        # WARN fires at this percentile or below
MIN_LIVE_TRADES = 20    # Sample-size floor for kill rule to apply
KS_ALPHA = 0.05         # Edge diagnosis distribution-shift threshold
ROLLING_WINDOW_DAYS = 60 # Rolling Sharpe / Calmar window
```

---

## Roadmap and known limitations

### Honest limitations
1. **MC envelope is static** — uses backtest distribution only, never updates with accumulated live data. A Bayesian posterior update would be more rigorous but requires a heavier statistical framework.
2. **No multiple-hypothesis correction on kills** — at α=5% across ~16 strategies, any single review has a meaningful chance of containing one noise-driven kill. In practice this is softened two ways: reviews run **~quarterly** (the realized cadence — two post-launch cull dates in 6+ months — not monthly), and a noise kill surfaces as a *recoverable* UNLUCKY **suspend** unless the KS test independently fires (permanent archive requires both). A Benjamini-Hochberg threshold would still harden permanent-archive decisions, but the architecture already absorbs most of the cost — see the roadmap for why this is downgraded.
3. **KS test has weak power at n=20-30** — non-firing KS isn't strong evidence of stability, just "not enough data to conclude". Diagnosis is honest about this in the tooltip.
4. **Kill rule is regime-agnostic (by design)** — the kill verdict is a pure MC-envelope tail test on each strategy's *own* P&L; it consumes **no** regime label, BTC or per-ticker. Regime classification feeds the attribution, family, and pair-divergence views for *human* interpretation only. This is deliberate — regime-conditioning the envelope would split the already-thin ≥20-trade live sample across regimes and undermine the `MIN_LIVE_TRADES` guard. The practical consequence: a LONG-only alt strategy that drew down during a BTC-only bear must be cross-checked against its *own* ticker's regime by eye (the per-ticker regime is computed and shown for exactly this) before you archive it.
5. **Cost model is generic** — real Bybit fee tier varies; users can override sidebar values.

### Where the next phase should go

The engine is **feature-complete and test-hardened** — the six tabs cover the full
deploy → monitor → size workflow, and ~130 tests guard `calculations.py`. So the
honest question for the next phase is *not* "what statistics can we add" but **"have
the statistics we already shipped actually worked?"** Six-plus months of live
incubation (since 2025-12-03) and ~20 archived strategies across three cull
generations now make that empirically answerable. The list below is reprioritized
accordingly — and most of the old "future additions" are deliberately demoted.

**P1 — close the loop (highest value, only now possible):**
- **Verdict-history persistence.** Every verdict is recomputed *stateless* each run;
  nothing records when a strategy first crossed WARN → KILL. Persist a per-strategy
  verdict log. This is the prerequisite for everything below.
- **Kill-rule back-test.** With ~20 realized kills, test the engine against its own
  history: did killed strategies stay dead? did any UNLUCKY-suspended ones recover?
  This validates (or recalibrates) the 5 % / 15 % / 20-trade thresholds on *real
  outcomes* — worth more than any new statistic bolted onto an unvalidated rule.

**P1 — operational:**
- **Data-staleness guard.** Verdicts are only valid on fresh exports, but the loaded
  CSVs can silently age and the tool has no concept of "last refreshed." Surface a
  staleness banner when the newest export is older than the review cadence. A stale
  kill verdict is worse than no verdict.

**P2 — only if the back-test shows real noise:**
- **Temporal kill hysteresis** *(supersedes the old "2-tier kill" item — suspend-vs-
  archive already exists via the edge diagnosis).* The genuine gap is debouncing:
  require **N consecutive** KILL verdicts before a *permanent* archive. For this
  architecture that is a better false-positive defense than a multiple-testing
  correction, and it falls straight out of verdict-history persistence.

**Downgraded (marginal given the current design):**
- **FDR / Bonferroni on kills.** The realized review cadence is ~quarterly, not
  monthly, and a false kill already lands as a *recoverable* UNLUCKY suspend unless
  KS independently fires. The architecture absorbs most of the cost a correction
  would buy back. Revisit only if the back-test shows *good* strategies being
  *permanently* archived by noise.

**Reconsidered / likely drop:**
- **Online MC envelope update.** As originally framed (re-bootstrap with live data
  weighted in) it **contaminates the out-of-sample test** — the core invariant is
  "never fit on live data." The only OOS-preserving version is a Bayesian posterior
  on Sharpe (backtest prior + live likelihood) shown as a *separate* evidence-
  accumulation panel that never feeds the kill envelope.
- **Per-ticker regime *in the kill rule*.** See limitation #4: the kill rule is
  regime-agnostic by design, and conditioning it would fight the ≥20-trade floor.
  Per-ticker regime already lives where it belongs — attribution and pair views.

**Still genuinely open (unchanged scope):**
- Time-varying regime model (HMM or regime-switching VAR) for the attribution tab.

### Out of scope
This is not an execution engine. It analyzes TradingView CSV exports and outputs sizing recommendations; trade execution remains manual (TradingView alerts → Bybit) or via a separate broker integration.

---

## Acknowledgements

Methodology informed by:
- **Bailey & López de Prado** — *The Deflated Sharpe Ratio* (2014)
- **Lo (2002)** — *The Statistics of Sharpe Ratios* (Sharpe ratio standard errors)
- **Bailey, Borwein, López de Prado, Zhu** — *The Probability of Backtest Overfitting* (2017)
- **QuantStats** library — inspiration for the extended performance metrics block

---

## License

MIT — see [LICENSE](LICENSE).

---

## Contributing

This is primarily a personal toolkit. Issues and PRs are welcome, especially for:
- Additional broker CSV formats (besides TradingView)
- Statistical improvements (multiple-testing correction, Bayesian updates)
- UI polish

For methodological discussion (kill-rule design, edge-diagnosis logic, regime classification), please open an Issue first.
