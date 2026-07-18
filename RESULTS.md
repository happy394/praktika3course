# Results — Statistical Arbitrage on MOEX (Russian re-run of the Polish study)

> Companion to `REPLICATION_SPEC.md` (method + Polish targets) and `RU_ADAPTATION.md`
> (Polish→Russian mapping). All numbers produced by `run_experiments.py` /
> `run_lstm.py` from MOEX data built by `moex_data.py`; raw run records in
> `results/summary.json`, equity curves in `results/equity_*.csv`.

## TL;DR — does it work in Russia?

**No — not in the paper's main test period, and the paper's own ranking of methods does
not survive the transfer.** Over 2017–2019 the gross (pre-fee) trading P&L of the best
Polish method (PCA eigenportfolios) on the Russian top-60 universe is **−0.1 RUB per
100 RUB of capital — i.e. exactly zero edge** — and after 0.1% transaction costs every
PCA configuration loses 16–27 RUB relative to the cash baseline. The ETF-factor methods
trade little and end within ±3 RUB of cash. In the 2020 stress every method ends ≈ flat
vs cash — Russia produced **neither** the Polish PCA blow-up **nor** the Polish ETF
profit.

The LSTM method — the paper's novel contribution, ≈+10% in Poland — is the *worst*
performer in Russia (−12.9%, α −37 RUB): long overshoot-exit holds bleed fees and the
money-market spread on a signal that isn't there.

The one genuinely strong result is **2015–2016**: RU-optimized PCA thresholds earn
**+123% over two years** (gross +96.6 RUB vs a 23-RUB cash baseline), ~85% of it from
**short** trades against single-name squeezes in the wild post-2014-crash market. That
edge is regime-local: the same thresholds applied out-of-sample to 2017–2019 lose money.
Mean-reversion of factor-model residuals on MOEX was a crisis phenomenon, not a stable
property.

## 1. Setup (one paragraph)

Identical pipeline to the paper (factor model → cumulated residual → OU/AR(1) → s-score
→ threshold rules → 60 independent per-stock traders, W=120, κ>4, 2:1 leverage, Λ=E/30,
c=0.1% per side, E₀=100). Universe: 60 most liquid MOEX names 2014–2020 (top MOEXBMI
weights, ≥95% coverage; commons+prefs, survivorship caveat §7). Prices: MOEX candles,
self-adjusted for dividends incl. share-basis rescaling for post-2020 splits (GMKN,
TRNFP, VTBR, PLZL). Factors: PCA eigenportfolios (r=15 or var≥55%); "existing ETFs" =
MCFTR+MESMTR total-return indices (Russia has 2 cap tiers, not 3; +RGBITR variant);
"artificial ETFs" = 8 MOEX sector TR indices. Risk-free = 1Y OFZ zero-coupon average:
**10.39%** (2015–16), **7.34%** (2017–19), **4.86%** (2020) — the single biggest
difference vs Poland's 1.5%/0.1%. Periods match the paper exactly.

**Reading the tables.** The paper's equity `E_t = 100·e^{r_f t} + Σ realized P&L`
compounds cash automatically, so with RU rates "did nothing" = **+24.5%** over
2017–2019 and **+4.9%** over 2020. The honest metric is **α = E_final − cash
baseline** (RUB per 100 initial).

## 2. Transfer test — the paper's Polish thresholds on Russian data

2017–2019 (cash baseline 124.5 RUB):

| Method | thresholds | RU total ret | **RU α (RUB)** | RU Sharpe 17/18/19 | trades | Poland (paper) |
|---|---|---|---|---|---|---|
| PCA r=15 | 1.10/−0.50 | −2.1% | **−26.7** | −1.08 / +0.88 / −2.66 | 677 | **+20%**, 2.63/1.01/−1.16 |
| PCA var-r | 1.10/−0.50 | −1.7% | **−26.3** | −1.08 / +0.88 / −2.28 | 675 | +20%, 2.51/0.44/−0.91 |
| ETF sparse | 2.10/+0.75 | +22.9% | −1.8 | −0.86 / +0.23 / +0.28 | 188 | +5%, −0.25/−0.46/+1.43 |
| ETF sparse+bonds | 2.10/+0.75 | +25.0% | +0.4 | −0.86 / +0.60 / +0.96 | 194 | — |
| ETF sector | 1.95/+0.40 | +18.6% | −6.1 | −0.88 / −0.38 / −0.41 | 276 | ≈+5% |

2020 (cash baseline 104.9 RUB):

| Method | RU total ret | **RU α (RUB)** | RU Sharpe | Poland (paper) |
|---|---|---|---|---|
| PCA r=15 | +4.7% | −0.3 | −0.01 | **big loss**, Sharpe −1.39 |
| PCA var-r | +4.5% | −0.4 | −0.03 | loss, Sharpe +0.59 |
| ETF sparse | +5.0% | +0.1 | +0.05 | **+3%**, Sharpe 0.56 |
| ETF sector | +4.1% | −0.9 | −0.22 | **+5%**, Sharpe 0.68 |

**Poland's story does not reproduce.** There, PCA earned Sharpe up to 2.63 and ~20%
while ETFs were weakest; in Russia PCA is the *worst* (α −27 RUB) and nothing beats
cash. In 2020 the paper's headline claim — "only the ETF methods survive the crash,
PCA blows up" — also fails to transfer: in Russia *nothing* blew up and *nothing*
made money; all methods ended within ±1 RUB of cash.

## 3. RU-optimized thresholds (grid on 2015–2016, as the paper did on its own market)

Grid search (g_open ∈ [0.8, 2.4], g_close ∈ [−0.8, 1.0], objective = final profit):

| Method | best (g_open, g_close) | 2015–16 in-sample | gross of fees | OOS 2017–19 α | OOS 2020 α |
|---|---|---|---|---|---|
| PCA r=15 | (1.30, **+0.90**) | **+123.3%** (α +100.3) | +96.6 RUB | **−16.4** | −0.7 |
| PCA var-r | (1.20, +0.50) | +121.2% | — | **−23.5** | −0.8 |
| ETF sparse | (2.40, +1.00) | +18.6% (α −4.2, below cash!) | — | −2.4 | −0.6 |
| ETF sparse+bonds | (2.40, +1.00) | +17.9% | — | −1.3 | +0.3 |
| ETF sector | (1.70, +0.80) | +47.5% (α +24.7) | — | −3.3 | **+1.4** |

Three observations:

1. **2015–2016 Russia had a real, large residual-reversion edge** — PCA gross P&L
   +96.6 RUB on a 100-RUB book (674 trades, median hold 8 days). This is not a 2–3
   stock artifact: the top-3 names contribute only 19% of gross P&L. But it is
   **short-driven**: shorts made +115.7 RUB (65% win rate), longs *lost* −38.1. The
   strategy was systematically fading single-name spikes in a thin, volatile,
   post-crisis market. (Borrow availability for mid-caps in 2015 Russia makes the
   realizable fraction of this smaller — §7.) Sharper still: the equity curve shows
   **nearly the entire two-year profit accrued in Q1 2015** — the direct aftermath of
   the December-2014 ruble crisis — followed by a ~40 RUB give-back within weeks and a
   plateau. "It worked in 2015–16" really means "it monetized one quarter of
   post-crisis chaos".
2. **The edge died on schedule.** The same thresholds out-of-sample on 2017–2019: gross
   P&L −0.1 RUB (yes, zero), net −15.3 after fees. Loss is diffuse (top-3 names = 6% of
   gross) — no single name to blame. **But the mechanism is not "reversion stopped"**
   (see §4a): the *typical* trade behaves identically in both regimes; what disappeared
   is the crisis tail.
3. Even *in-sample*, the sparse-ETF method cannot beat RU cash (+18.6% vs +23.0%
   baseline) — with a 10.4% risk-free rate, a low-turnover hedged strategy has
   nothing left after the money-market leg.

## 4a. What actually differs between the regimes — it is a tail, not an edge

Trade-level statistics (PCA, RU-tuned thresholds, **gross** of fees, P&L per RUB of
position) settle what "the signal died" really means:

| | 2015–2016 | 2017–2019 |
|---|---|---|
| median trade | +1.30% | +1.05% |
| trimmed mean (5% each tail) | **+0.36%** | **+0.39%** |
| win rate | 64% | 65% |
| **untrimmed mean** | **+4.17%** | **+0.02%** |
| n trades | 674 | 1,092 |

The *typical* trade is **statistically indistinguishable across the two regimes**, and
in both it earns ≈ +0.4% gross — almost exactly the ~0.4% round-trip cost. The strategy
never had a steady edge in Russia: its median trade merely pays the toll booth.

The entire difference sits in one fat right tail:

- **Q1 2015 alone**: 111 trades, mean **+25.6%** per trade, **+101 RUB** — more than the
  whole two-year P&L (+77.6 net).
- **The other 563 trades of 2015–16 lost −4.6 RUB gross** (−20.7 net).
- That quarter is *not* a leverage artifact (gross exposure Σ|β| median 2.75 — same as
  the rest of the period) and *not* a single-name artifact (56 stocks contributed, top-3
  = 28%). It is the post-ruble-crisis unwind: market volatility **42%** in Q1 2015 vs
  **26%** in 2017–2019, and it is ~80% short-side (+128 vs −30 RUB).

So on MOEX this is a **bet on crisis dispersion, not a market-neutral income stream** —
and it paid exactly once in six years. Figure: `figures/03_a_tail_bet_not_an_edge.png`.

## 4. Why it fails in 2017–2019 — diagnostics

- **Not the factor structure.** Prior-year correlation matrices need r = 13–16 PCs for
  55% variance (λ₁ = 13–24%) — essentially the same dimensionality the paper reports
  for Poland (r = 16–18) and uses for the US. The Russian market is *not* a degenerate
  one-factor market at daily frequency; the APT decomposition itself is fine.
- **The residual signal is flat.** Gross-of-fee P&L 2017–2019: PCA −0.1 RUB on 1,092
  trades; sector-ETF +4.3 RUB on 502 trades. s-score reversion after |s|>1.1–1.3
  entries simply nets to ~zero per trade before costs.
- **Costs finish it off.** With ~0.4% round-trip cost (both legs, paper's formula) and
  8–14-day holds, 500–1,100 trades cost 7–19 RUB per period. Dead signal − fees =
  the observed α.
- **Win/loss shape.** Win rates are fine (58–62%) but avg win ≈ 0.5–0.6× avg loss —
  quick same-side exits cap winners while losers ride to the opposite threshold.
  Rebalancing thresholds can't fix a zero-mean signal.

## 5. 2020 — a different kind of stress answer

Russia's COVID+oil-war crash (March 2020) was as violent as Poland's COVID crash, yet
no method blew up: the κ>4 gate plus wide entries kept books small, and the V-shaped
MOEX recovery meant the residuals that did dislocate mostly round-tripped. The paper's
"PCA loses big / ETFs profit" asymmetry is absent: all ten 2020 configurations end
between −1.1 and +1.4 RUB of cash. The only (weakly) positive 2020 config is
RU-tuned sector-ETF (+1.4 RUB, Sharpe 0.50).

## 6. LSTM method — the paper's novel contribution fares worst of all

240 stock-year models (2-layer LSTM(64), MSE + 1e-5·L1, Adam, batch 16, yearly
retraining on 3-year windows — spec §3B exactly), paper thresholds (1.10, −0.15):

| Period | RU total ret | **RU α (RUB)** | RU Sharpe by year | trades | win | hold | Poland (paper) |
|---|---|---|---|---|---|---|---|
| 2017–2019 | **−12.9%** | **−37.4** | −0.20 / −0.60 / −2.27 | 711 | 61% | 20d | **≈ +10%**, 0.60/2.09/−1.53 |
| 2020 | +3.9% | −1.0 | +0.03 | 261 | 49% | 12d | small loss, −0.34 |

The LSTM replica itself works as designed (residual std collapses to the idiosyncratic
level, exactly like on synthetic data) — but its Polish threshold profile exits on
*overshoot* (close long only once s > +0.15), so positions sit for ~20 days on a
zero-mean signal, bleeding the money-market spread and fees on both legs. A structural
instability also shows up that PCA/ETF don't have: the LSTM's **net hedge notional
q^M = Σβ is unconstrained** (the paper's L1 = 1e-5 shrinks individual weights, not the
sum), so occasional trades carry |q^M| ≫ 1 and produce ±8–12 RUB single-trade P&L
swings (visible as the early-2017 equity spike/crash). The paper flags its LSTM
hyperparameters as unoptimized; this is a concrete instance of what that costs. In Russia the
method ranking becomes simply **"whoever trades least loses least"**: ETF (α −2…−6) >
PCA (−27) > LSTM (−37) — a full inversion of Poland's PCA > LSTM > ETF.

## 7. Caveats

1. **Survivorship**: fixed 60-name universe with full 2014–2020 history (paper does the
   same); delistings (URKA, MFON, DIXY…) excluded — this *flatters* the results, so the
   negative 2017–2019 verdict is conservative.
2. **Shorting realism**: 2015–16 P&L is 85%+ short-side; RU borrow in mid-caps was
   scarce and expensive then; no borrow fees modeled (paper models none either).
2b. **The paper's cost model is optimistic (measured).** Fees are charged on the *net*
   replica notional `|1+q^M|`, where `q^M = Σβ ≈ 1`. But the replica must actually be
   traded leg by leg, and its **gross** book `Σ|β|` is far larger: median **3.0** and
   p95 **5.5** for PCA (max 32), i.e. real costs are ~2–3× what the formula charges;
   sector-ETF betas, being collinear, occasionally explode (max Σ|β| = 668 → 2,365 RUB
   of gross hedge on a 100 RUB book). The paper's stated 2:1 leverage refers to the net
   book only. Charging costs on gross exposure would make every α in this report
   *worse*, so the negative verdict is conservative. (The paper acknowledges the sector
   collinearity but leaves it unaddressed.)
3. **Sector/SMID indices were not directly investable** pre-2018 (the paper splices ETF
   with index history the same way; FXRL existed from 03.2016 for MCFTR).
4. **Dividend data**: merged ISS+dohod per-payment records; QIWI pre-2016 payments
   missing; USD dividends converted at cum-date FX; share-basis rescaling applied for
   post-window splits (GMKN 1/100, TRNFP 1/100, PLZL 1/10, VTBR ×5000).
5. **Sharpe convention**: paper's realized-only equity (open positions unmarked) —
  computed identically here; a marked-to-market variant is also stored in the CSVs and
  tells the same story.
6. r_f uses period-average 1Y OFZ zero-coupon yields; the paper used constants.

## 8. Conclusions (for the practicum write-up)

1. **The pipeline replicates; the alpha does not.** Every mechanical element of the
   Polish study (factor construction, OU fit, s-scores, threshold logic, accounting)
   ports to MOEX data without modification, and behaves sanely (validated on synthetic
   OU data). What's missing in Russia 2017–2019 is the raw material: residuals whose
   mean reversion pays more than ~0.4% per round trip.
2. **The paper's cross-method ranking is market-specific.** PCA-best (Poland) inverts
   to PCA-worst (Russia); the "ETFs are robust in crashes" conclusion degenerates to
   "everything is cash" in Russia's 2020.
3. **It is a crisis-dispersion bet, not an income stream** (§4a — the sharpest finding
   here). The typical Russian trade earns ≈+0.4% gross in *both* regimes — i.e. exactly
   its own transaction cost — so the strategy's expected value is zero plus a tail. That
   tail arrived once, in Q1 2015 (+101 RUB from 111 trades at 42% market vol), and never
   again: strip that single quarter and even the "successful" 2015–16 period loses
   money. Any production use would need a volatility/regime filter, borrow-aware costs,
   and the honest expectation of dormancy most of the time. It also reframes the
   paper's Polish result: 2017–19 Warsaw may simply have been a dispersion-rich regime,
   which is testable by re-running their code on later GPW data.
4. **High RUB rates change the game**: with cash at 7–10%, a hedged RUB strategy needs
   ~8%+ net just to tie the baseline the paper's accounting hands out for free. The
   same strategy would look far better in the paper's 1.5% world — worth stating
   explicitly when comparing Sharpe tables across the two markets.
