# RU Adaptation Spec — Statistical Arbitrage on MOEX (Russian re-implementation)

> Companion to `REPLICATION_SPEC.md`. **The method is unchanged** (factor model → residual →
> OU/AR(1) → s-score → thresholds → backtest, all parameters W=120, n=252, κ>4, 2:1 leverage,
> Λ=E/30, c=0.1%). This file records only what is *replaced* when moving GPW → MOEX, and why.
> Research question: does the Adamczyk–Dąbrowski (2024) / Avellaneda–Lee (2008) strategy
> transfer to the Russian equities market?

## 1. Element mapping

| Element | Poland (paper) | Russia (this repo) |
|---|---|---|
| Exchange | GPW (Warsaw) | MOEX (Moscow), board TQBR |
| Currency / E₀ | 100 PLN | 100 RUB |
| Universe (d) | WIG20 + mWIG40 = 60 | Top-60 MOEXBMI (Broad Market) names by index weight ≈ IMOEX + liquid mid-caps (§2) |
| PCA factors | eigenportfolios of the 60, r=15 / var ≥55% | identical, on the RU universe |
| "Existing ETFs" (sparse) | WIG20TR, mWIG40TR, sWIG80TR | **MCFTR** (MOEX TR) + **MESMTR** (SMID TR) — Russia has 2 cap tiers, not 3 (§3) |
| "Artificial ETFs" (dense) | 14 WIG sector TR indices | **8 MOEX sector TR indices**: MEOGTR, MEMMTR, MEFNTR, MECNTR, MEEUTR, METLTR, METNTR, MECHTR (§3) |
| LSTM | per-stock 2×LSTM(64), input = other 59 stocks | identical, input = other d−1 stocks |
| Prices | adjusted close (vendor) | MOEX close, **self-adjusted for dividends** (§4) |
| Risk-free | 1.5% (2017–19), 0.1% (2020) | 1Y OFZ zero-coupon yield (MOEX G-curve), period averages (§5) — ~7% / ~5% |
| Transaction cost | 0.1% per side | 0.1% per side (kept; realistic for RU broker + slippage) |
| Periods | opt 2015–16, main 2017–19, stress 2020 | **identical calendar periods** → results directly comparable |

Everything not listed here follows `REPLICATION_SPEC.md` exactly.

## 2. Universe construction

Rule: take MOEXBMI (Broad Market Index, ~100 names) compositions from the MOEX ISS
analytics endpoint at the start of each year 2015–2020, score every ticker by average index
weight across snapshots, keep tickers with ≥95% trading-day price coverage over
2014-03-01…2020-12-30 on TQBR, take the **top 60** (or as many as pass — final list in
`data/universe.json`).

- Mirrors WIG20+mWIG40 = "large + mid caps, fixed 60-name list over the period" (the paper
  also used one fixed list of constituents + reserves).
- **Survivorship caveat**: requiring full-period history drops names delisted mid-period
  (URKA 2019, MFON 2018, DIXY 2018…) and late listings (POLY on MOEX 2017+, FIVE 2018+,
  UPRO post-rename 2016+). The paper has the same issue with its fixed list. Documented, not fixed.
- Both common and preferred shares of one issuer may enter (SBER/SBERP, SNGS/SNGSP…), as on
  GPW nothing forbids near-duplicates; PCA handles them as highly correlated columns.

## 3. Factor analogs — what changed and why

**Sparse ("existing ETFs").** Poland had 3 investable size-tier TR ETFs (splicing in the TR
indices before the 2018 ETF launches). Russia's real ETF lineup in 2017–2019 was thin
(FXRL since 03.2016, SBMX since 09.2018 — both large-cap only), so we use the same splice
logic the paper used, but for the whole sample: factor returns = TR **indices** MCFTR
(large, tracked by FXRL/SBMX) and MESMTR (mid/small, tracked by no ETF then — investability
caveat). Russia has no third tier → **2 factors, not 3**. Variant `etf_real3` adds RGBITR
(state-bond TR, tradable via OFZ) as a rates factor.

**Dense ("artificial ETFs").** MOEX publishes 8 sector total-return indices with history from
2015: oil&gas (MEOGTR), metals&mining (MEMMTR), financials (MEFNTR), consumer (MECNTR),
electric utilities (MEEUTR), telecom (METLTR), transport (METNTR), chemicals (MECHTR).
Poland had 14 sector indices; 8 is what exists — fewer, coarser sectors (no IT/RE indices
until 2020). Factor count r=8 vs paper's 14.

## 4. Dividend-adjusted prices

MOEX candle closes are **not** adjusted. Russian dividend yields are large (5–10%/yr for many
top names), and an ex-date drop looks exactly like the idiosyncratic dislocation the strategy
buys — unadjusted prices would generate systematically false long signals. Adjustment:

1. Per-payment records (value + **registry close date**) from dohod.ru
   (`/ik/analytics/dividend/<ticker>`, table «Дата закрытия реестра»), cross-merged with the
   MOEX ISS `/securities/<secid>/dividends` endpoint (ISS is incomplete before ~2018 for many
   names — SBER starts 2019, MTSS/NVTK 2018).
2. Ex-date under T+2 (MOEX regime since 09.2013): last cum-dividend day = last trading day
   `X` with `X+2` trading days ≤ registry-close date; ex-date = next trading day after `X`.
3. Back-adjustment: on each ex-date multiply all prior prices by `(close_cum − div)/close_cum`
   (multiplicative chain, like any adjusted-close feed). USD-denominated dividends (AGRO)
   converted at the USDRUB rate on the cum-date.
4. Sanity check: >40% overnight moves in the *adjusted* series flagged (splits/renames).

## 5. Risk-free rate

1Y point of the MOEX zero-coupon government yield curve (ISS `zcyc` endpoint, `yearyields`,
period=1.0), averaged over each backtest period (monthly sampling). Approximate levels:
**~9.5%** (2015–16), **~7.2%** (2017–19), **~5.0%** (2020) — exact values computed by
`moex_data.py` and stored in `data/risk_free.json`.

This is the single biggest structural difference vs Poland (1.5% / 0.1%): the money-market leg
of the P&L formula earns/costs far more, and the Sharpe hurdle is much higher. A strategy
"working" in Russia means beating ~7% RUB cash, not ~1.5%.

## 6. Experiment plan

1. **Transfer test** — run PCA (r=15, var-r), ETF sparse (2 factors), ETF dense (8 sectors) on
   2017–2019 and 2020 with the **paper's Polish-optimized thresholds** (spec §5). Question:
   does the strategy transfer as-is?
2. **Re-optimization** — grid-search (g_open, g_close) on RU 2015–2016 (objective: final
   profit, symmetric cutoffs, same grid spirit as paper), rerun 2017–2019/2020. Question:
   does it work with honest out-of-sample threshold selection?
3. **LSTM** (optional, after 1–2) — identical architecture per spec §3B.
4. Compare against paper's targets (REPLICATION_SPEC.md §6): Poland PCA ≈ +20% / Sharpe 2.63
   (2017), LSTM ≈ +10%, ETF ≈ +5% but only-ETF-profits-in-2020.

## 7. Hypotheses (to check against results)

- RU market is dominated by 2–3 macro factors (oil, FX, sanctions news) → first eigenvalue
  heavier than Poland's; fewer meaningful PCs; idiosyncratic residuals smaller share of variance.
- High r_f (7%+) eats a large part of gross P&L → lower net Sharpe than Poland even if the
  mechanics work.
- 2020 stress in Russia = COVID **plus** the March oil-price war → stock-based factor methods
  (PCA/LSTM) should break at least as badly as in Poland; sector/index factors more robust.
- Fewer sectors (8 vs 14) → dense-ETF residuals dirtier than Poland's.

## 8. Files

- `moex_data.py` — data layer: ISS fetchers (candles, index history, compositions, zcyc,
  dividends), dohod.ru dividend parser, adjusted-close builder, caching to `data/*.parquet`.
- `statarb_pipeline.py` — method engine (shared with the Polish spec; two-stage: s-score
  precompute → fast signal/P&L backtest).
- `run_experiments.py` — experiment driver producing `results/`.
