"""
make_figures.py — the visual story of the strategy on the Russian market.
Writes annotated PNGs to figures/ (+ figures/README.md with descriptions).

Every figure is generated from the cached data/ + results/; no network, no
LSTM retraining. Run: python3 make_figures.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch

from run_experiments import PERIODS, build_panel, prepare_data, traded_mask
from statarb_pipeline import (KAPPA_MIN, N_COV, THRESHOLDS, W_RESID, backtest,
                              ols_residuals_multi, trade_stats)

# PRINT mode (FIG_PRINT=1): for the PDF — drop the in-image caption (the document
# typesets it properly instead) and scale every text object up, because the figure
# gets shrunk to page width and would otherwise be unreadable.
PRINT = os.environ.get("FIG_PRINT") == "1"
FONT_SCALE = 1.30
FIG = Path(__file__).parent / ("figures/print" if PRINT else "figures")
FIG.mkdir(parents=True, exist_ok=True)

# one colour per entity, consistent across every figure (dataviz: identity, not rank)
C = {"pca_const": "#2a78d6", "pca_var": "#1baf7a", "etf_real": "#eda100",
     "etf_real3": "#008300", "etf_sector": "#4a3aa7", "lstm": "#e34948",
     "cash": "#8d97a7", "ink": "#1b2430", "ink2": "#5b6675", "rule": "#dde2e9",
     "pos": "#0f7a3d", "neg": "#b3352f", "stock": "#2a78d6", "repl": "#eb6834"}
NAME = {"pca_const": "PCA r=15", "pca_var": "PCA var-r", "etf_real": "ETF sparse",
        "etf_real3": "ETF sparse+bonds", "etf_sector": "ETF sector", "lstm": "LSTM"}

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": C["rule"], "axes.labelcolor": C["ink2"],
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": C["ink"],
    "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "xtick.color": C["ink2"], "ytick.color": C["ink2"],
    "axes.grid": True, "grid.color": C["rule"], "grid.alpha": .9,
    "grid.linewidth": .6, "axes.axisbelow": True, "legend.frameon": False,
    "legend.fontsize": 8.5, "font.size": 9.5,
})


def finish(ax, note: str | None = None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if note and not PRINT:
        ax.text(0, -0.20, note, transform=ax.transAxes, fontsize=8,
                color=C["ink2"], va="top", wrap=True)


def save(fig, name: str):
    if PRINT:
        for t in fig.findobj(mpl.text.Text):
            t.set_fontsize(t.get_fontsize() * FONT_SCALE)
    p = FIG / name
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {p.name}")


print("loading cached data + rebuilding signal panels…")
data = prepare_data()
R = data["returns"]
RF = data["rf"]
panels = {(m, p): build_panel(m, data, p)
          for p in ("2015-2016", "2017-2019", "2020")
          for m in ("pca_const", "pca_var", "etf_real", "etf_real3", "etf_sector")}


# =====================================================================
# 1 · Anatomy of one real trade — the whole strategy in one picture
# =====================================================================
def fig_trade():
    panel = panels[("pca_const", "2015-2016")]
    res = backtest(panel, 1.30, 0.90, r_f=RF["2015-2016"],
                   traded=traded_mask(data, panel))
    # a clear, profitable short: stock ran ahead of its replica, then converged
    cands = [t for t in res["trades"]
             if t.side == -1 and t.pnl > 0 and 8 <= t.t_close - t.t_open <= 30
             and t.t_open > 45 and abs(t.s_open) > 1.5]
    tr = max(cands, key=lambda t: t.pnl)
    i = panel.stocks.index(tr.stock)
    a, b = tr.t_open - 30, min(tr.t_close + 15, len(panel.dates) - 1)
    dts = panel.dates[a:b + 1]

    # cumulative net returns from the left edge, stock vs its frozen replica
    stock = np.expm1(panel.stock_cumlog[a:b + 1, i] - panel.stock_cumlog[a, i]) * 100
    gross = np.expm1(panel.instr_cumlog[a:b + 1] - panel.instr_cumlog[a])
    repl = (gross @ tr.w_vec) / tr.qM * 100      # per 1 RUB of replica

    o, c = tr.t_open - a, tr.t_close - a
    gap = stock - repl
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.6, 6.8), sharex=True,
                                   gridspec_kw={"height_ratios": [1.35, 1]})
    ax1.plot(dts, stock, color=C["stock"], lw=1.8, label=f"{tr.stock} (the stock)")
    ax1.plot(dts, repl, color=C["repl"], lw=1.8,
             label="its replica — 15 PCA eigenportfolios, weights fit on trailing 120d")
    ax1.fill_between(dts, stock, repl, where=stock >= repl, color=C["stock"],
                     alpha=.10, lw=0)
    ax1.fill_between(dts, stock, repl, where=stock < repl, color=C["repl"],
                     alpha=.10, lw=0)
    ax1.axvline(dts[o], color=C["ink2"], lw=1, ls=":")
    ax1.axvline(dts[c], color=C["ink2"], lw=1, ls=":")
    ax1.annotate("", xy=(dts[o], stock[o]), xytext=(dts[o], repl[o]),
                 arrowprops=dict(arrowstyle="<->", lw=1.6, color=C["neg"]))
    ax1.annotate("", xy=(dts[c], stock[c]), xytext=(dts[c], repl[c]),
                 arrowprops=dict(arrowstyle="<->", lw=1.6, color=C["pos"]))
    ax1.annotate(f"1. gap = {gap[o]:.1f} pp → SHORT {tr.stock},\n     buy the replica",
                 xy=(dts[o], (stock[o] + repl[o]) / 2),
                 xytext=(dts[max(o - 28, 0)], min(repl) - 1.5),
                 fontsize=8.5, color=C["neg"], fontweight="bold",
                 arrowprops=dict(arrowstyle="->", lw=1, color=C["neg"]))
    ax1.annotate(f"2. gap = {gap[c]:.1f} pp → close\n     +{tr.pnl:.2f} RUB on a "
                 f"{tr.scale:.1f} RUB position ({tr.pnl / tr.scale:+.0%})",
                 xy=(dts[c], (stock[c] + repl[c]) / 2),
                 xytext=(dts[min(c + 1, len(dts) - 1)], min(repl) - 1.5),
                 fontsize=8.5, color=C["pos"], fontweight="bold",
                 arrowprops=dict(arrowstyle="->", lw=1, color=C["pos"]))
    ax1.set_ylabel("cumulative return, %")
    ax1.set_ylim(min(repl) - 4.5, max(stock) + 2)
    ax1.set_title(f"Anatomy of one trade — {tr.stock}, {dts[o].date()} → {dts[c].date()} "
                  f"(PCA method, 2015–16)", pad=10)
    ax1.legend(loc="upper left", ncol=1)
    finish(ax1)

    s = panel.s[a:b + 1, i]
    ax2.plot(dts, s, color=C["ink"], lw=1.6)
    ax2.axhline(0, color=C["cash"], lw=1)
    ax2.axhline(1.30, color=C["neg"], lw=1.1, ls="--")
    ax2.axhline(0.90, color=C["pos"], lw=1.1, ls="--")
    ax2.text(dts[-1], 1.32, "  open short above +1.30", fontsize=8, color=C["neg"],
             va="bottom", ha="right")
    ax2.text(dts[-1], 0.86, "  close short below +0.90", fontsize=8, color=C["pos"],
             va="top", ha="right")
    ax2.scatter([dts[o]], [s[o]], s=60, color=C["neg"], zorder=5)
    ax2.scatter([dts[c]], [s[c]], s=60, color=C["pos"], zorder=5)
    ax2.set_ylabel("s-score")
    ax2.set_ylim(-0.6, 2.15)
    ax2.set_title("The signal: how far the gap sits from its own equilibrium (σ units)")
    finish(ax2, "The strategy never bets on the stock's direction — only on the GAP between "
                "the stock and its replica closing. Here NVTK had run ~10pp ahead of its "
                "replica; the book shorts NVTK, buys the replica, and closes 10 days later "
                "once the gap is ~3pp. Both legs are held in equal RUB size, so the market's "
                "direction cancels out.")
    fig.subplots_adjust(hspace=.30)
    save(fig, "01_anatomy_of_a_trade.png")


# =====================================================================
# 2 · Where the tradable signal comes from: return decomposition
# =====================================================================
def fig_decomposition():
    day = pd.Timestamp("2017-06-01")
    ti = R.index.searchsorted(day)
    Rwin = R.values[ti - W_RESID + 1:ti + 1]
    end = R.index.searchsorted(pd.Timestamp("2017-01-01"))
    win = R.values[end - N_COV:end]
    std = win.std(0, ddof=1); std[std == 0] = 1
    Y = (win - win.mean(0)) / std
    ev, vec = np.linalg.eigh((Y.T @ Y) / (len(win) - 1))
    Q = vec[:, np.argsort(ev)[::-1][:15]] / std[:, None]
    F = Rwin @ Q
    resid, beta = ols_residuals_multi(F, Rwin)
    i = R.columns.get_loc("SBER")
    dts = R.index[ti - W_RESID + 1:ti + 1]

    total = np.cumsum(Rwin[:, i]) * 100
    syst = np.cumsum(Rwin[:, i] - resid[:, i]) * 100
    idio = np.cumsum(resid[:, i]) * 100

    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.plot(dts, total, color=C["stock"], lw=1.9, label="SBER total return")
    ax.plot(dts, syst, color=C["repl"], lw=1.6,
            label="systematic part (explained by the 15 factors)")
    ax.plot(dts, idio, color=C["pca_var"], lw=1.9,
            label="idiosyncratic residual  $I_t$  ← the only thing traded")
    ax.axhline(0, color=C["cash"], lw=1)
    ax.set_ylabel("cumulative, %")
    ax.set_title("Splitting a stock into market + noise — SBER, 120-day window ending 2017-06-01")
    ax.legend(loc="upper left")
    r2 = 1 - resid[:, i].var() / Rwin[:, i].var()
    ax.text(.985, .06, f"factors explain {r2:.0%} of SBER's daily variance\n"
            f"residual std: {resid[:, i].std():.2%}/day",
            transform=ax.transAxes, ha="right", fontsize=8.5, color=C["ink2"],
            bbox=dict(fc="white", ec=C["rule"], boxstyle="round,pad=0.4"))
    finish(ax, "The residual is a bridge: it must return to 0 at the window's end by "
               "construction of the regression. The bet is that when it wanders far from "
               "its equilibrium mid-window, it comes back — fast enough to beat costs.")
    save(fig, "02_return_decomposition.png")


# =====================================================================
# 3 · THE diagnostic: the strategy is a tail bet on crisis dispersion
# =====================================================================
def gross_trades(period: str, g=(1.30, 0.90)) -> pd.DataFrame:
    """Every trade the strategy actually took, priced GROSS of fees."""
    p = panels[("pca_const", period)]
    r = backtest(p, *g, r_f=RF[period], traded=traded_mask(data, p), cost=0.0,
                 compute_mtm=False)
    return pd.DataFrame([{"date": p.dates[t.t_open], "ret": t.pnl / t.scale * 100,
                          "pnl": t.pnl} for t in r["trades"]])


def fig_tail_bet():
    t15, t17 = gross_trades("2015-2016"), gross_trades("2017-2019")
    t20 = gross_trades("2020")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.8, 7.4),
                                   gridspec_kw={"height_ratios": [1, 1.05]})

    # -- panel A: the typical trade is IDENTICAL in both regimes ------------
    lim, bins = 12, np.linspace(-12, 12, 61)
    for t, col, lab in [(t17, C["neg"], "2017–2019"), (t15, C["pos"], "2015–2016")]:
        ax1.hist(t.ret.clip(-lim, lim), bins=bins, color=col, alpha=.45,
                 label=f"{lab} — {len(t):,} trades", lw=0)
    for t, col in [(t17, C["neg"]), (t15, C["pos"])]:
        trim = t.ret.clip(t.ret.quantile(.05), t.ret.quantile(.95)).mean()
        ax1.axvline(trim, color=col, lw=1.8, ls="--")
    ax1.axvline(0, color=C["ink2"], lw=1)
    ax1.set_xlim(-lim - .6, lim + .6)
    n_hi = (t15.ret > lim).sum() + (t17.ret > lim).sum()
    n_lo = (t15.ret < -lim).sum() + (t17.ret < -lim).sum()
    ax1.text(lim, ax1.get_ylim()[1] * .40, f"tails\nclipped\nhere →\n({n_hi} trades)",
             fontsize=7.5, color=C["ink2"], ha="right", va="center")
    ax1.text(-lim, ax1.get_ylim()[1] * .40, f"← tails\nclipped\nhere\n({n_lo} trades)",
             fontsize=7.5, color=C["ink2"], ha="left", va="center")
    ax1.set_ylabel("number of trades")
    ax1.set_xlabel("gross return per trade, % of position (clipped at ±12%)")
    ax1.set_title("The typical trade is the SAME in both regimes — and it only just pays the fee")
    ax1.legend(loc="upper left")
    ax1.text(.985, .93, "trimmed mean (dashed):\n2015–16  +0.36%   |   2017–19  +0.39%\n"
             "median:   +1.30%   |   +1.05%\nwin rate:   64%   |   65%\n"
             "→ round-trip cost ≈ 0.4% eats exactly this",
             transform=ax1.transAxes, ha="right", va="top", fontsize=8.5,
             color=C["ink"], bbox=dict(fc="white", ec=C["rule"],
                                       boxstyle="round,pad=0.45"))
    finish(ax1)

    # -- panel B: the entire edge is one quarter --------------------------
    allt = pd.concat([t15, t17, t20])
    q = allt.groupby(allt.date.dt.to_period("Q")).ret.mean()
    x = np.arange(len(q))
    cols = [C["pos"] if v > 5 else (C["cash"] if abs(v) < 1 else
            (C["pca_const"] if v > 0 else C["neg"])) for v in q.values]
    ax2.bar(x, q.values, color=cols, width=.72)
    ax2.axhline(0, color=C["ink"], lw=1.1)
    ax2.set_xticks(x, [str(p) for p in q.index], rotation=90, fontsize=7)
    ax2.set_ylabel("mean gross return per trade, %")
    ax2.set_title("…and the entire six-year edge is one quarter: the 2014 ruble crisis unwinding")
    ax2.annotate(f"Q1 2015: {q.iloc[0]:+.1f}% per trade\n111 trades → +101 RUB\n"
                 "(market vol 42% vs 26% later)",
                 xy=(0, q.iloc[0]), xytext=(2.4, q.iloc[0] * .78), fontsize=9,
                 fontweight="bold", color=C["pos"],
                 arrowprops=dict(arrowstyle="->", lw=1.2, color=C["pos"]))
    ax2.annotate("every other quarter: ≈ 0 before fees, negative after",
                 xy=(11, 0.9), xytext=(6.5, q.iloc[0] * .40), fontsize=9,
                 color=C["ink2"],
                 arrowprops=dict(arrowstyle="->", lw=1, color=C["ink2"]))
    finish(ax2, "PCA method, RU-tuned thresholds, fees excluded. The strategy is not a "
                "steady edge that 'died' — its typical trade behaves identically in both "
                "regimes and merely covers costs. What made 2015–16 was a fat right tail "
                "during the post-crisis unwind (56 different stocks, mostly shorts, at 42% "
                "market volatility). Take that quarter away and the remaining 563 trades of "
                "2015–16 lose 4.6 RUB. In Russia, this is a bet on crisis dispersion — and "
                "it paid exactly once in six years.")
    fig.subplots_adjust(hspace=.42)
    save(fig, "03_a_tail_bet_not_an_edge.png")


# =====================================================================
# 4 · Outcome: equity curves, all methods vs cash
# =====================================================================
def fig_equity():
    # portrait pages shrink a wide figure until its labels die → stack for print
    fig, axes = (plt.subplots(2, 1, figsize=(9.6, 8.2),
                              gridspec_kw={"height_ratios": [1.25, 1], "hspace": .34})
                 if PRINT else
                 plt.subplots(1, 2, figsize=(13.2, 4.8),
                              gridspec_kw={"width_ratios": [2.1, 1], "wspace": .22}))
    for ax, period in zip(axes, ("2017-2019", "2020")):
        rf = RF[period]
        for m in ("pca_const", "pca_var", "etf_real", "etf_real3", "etf_sector"):
            p = panels[(m, period)]
            r = backtest(p, *THRESHOLDS[m], r_f=rf, traded=traded_mask(data, p))
            ax.plot(p.dates, r["equity"], color=C[m], lw=1.4, label=NAME[m])
        e = pd.read_csv(f"results/equity_lstm_{period}_paper_thr.csv",
                        index_col=0, parse_dates=True)["E"]
        ax.plot(e.index, e, color=C["lstm"], lw=1.4, label="LSTM")
        dts = panels[("pca_const", period)].dates
        cash = 100 * np.exp(rf * np.arange(len(dts)) / 252)
        ax.plot(dts, cash, color=C["cash"], lw=1.8, ls="--",
                label=f"cash at {rf:.1%}  ← the benchmark")
        ax.set_title(f"{period}" + (" — the paper's main test" if period == "2017-2019"
                                    else " — the stress test"))
        ax.set_ylabel("RUB per 100 invested")
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylim(82, 140)
    axes[0].text(pd.Timestamp("2017-09-20"), 133,
                 "PCA r=15 (blue): the paper's best method — +20% and Sharpe 2.63 in\n"
                 "Poland. Here it ends at 98, i.e. 26.7 RUB behind simply holding cash.",
                 fontsize=8.5, color=C["pca_const"], fontweight="bold", va="top",
                 bbox=dict(fc="white", ec=C["pca_const"], boxstyle="round,pad=0.35"))
    axes[0].annotate("LSTM: unbounded hedge notional →\n±8–12 RUB single-trade swings",
                     xy=(pd.Timestamp("2017-05-25"), 86),
                     xytext=(pd.Timestamp("2017-08-20"), 92), fontsize=8,
                     color=C["lstm"],
                     arrowprops=dict(arrowstyle="->", lw=1, color=C["lstm"]))
    for ax in axes:
        finish(ax)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4 if PRINT else 7, fontsize=8.5,
               bbox_to_anchor=(.5, -.06 if PRINT else -.10))
    fig.suptitle("Every method, the paper's own thresholds, on Russian data",
                 fontsize=13, fontweight="bold", y=1.01 if PRINT else 1.02)
    if not PRINT:
        fig.text(0, -.20, "The dashed line is the honest benchmark: the paper's accounting "
                 "compounds idle cash at the risk-free rate, and RUB cash paid 7.3% "
                 "(2017–19). Everything that looks like a gain here is cash, not alpha — "
                 "the ETF curves simply track the baseline, while PCA and LSTM fall below "
                 "it.", fontsize=8.5, color=C["ink2"])
    save(fig, "04_equity_curves.png")


# =====================================================================
# 5 · The scoreboard: alpha net of cash
# =====================================================================
def fig_alpha():
    methods = ["pca_const", "pca_var", "etf_real", "etf_real3", "etf_sector", "lstm"]
    rows = {}
    for period in ("2017-2019", "2020"):
        rf, vals = RF[period], []
        for m in methods:
            if m == "lstm":
                e = pd.read_csv(f"results/equity_lstm_{period}_paper_thr.csv",
                                index_col=0)["E"]
                fin, n = float(e.iloc[-1]), len(e)
            else:
                p = panels[(m, period)]
                r = backtest(p, *THRESHOLDS[m], r_f=rf, traded=traded_mask(data, p))
                fin, n = float(r["equity"].iloc[-1]), len(p.dates)
            vals.append(fin - 100 * np.exp(rf * (n - 1) / 252))
        rows[period] = vals

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    y = np.arange(len(methods))
    h = .38
    ax.barh(y + h / 2, rows["2017-2019"], height=h, color=[C[m] for m in methods],
            label="2017–2019")
    ax.barh(y - h / 2, rows["2020"], height=h, color=[C[m] for m in methods],
            alpha=.45, label="2020")
    for yy, v in zip(y + h / 2, rows["2017-2019"]):
        ax.text(v - .8 if v < 0 else v + .8, yy, f"{v:+.1f}", va="center",
                ha="right" if v < 0 else "left", fontsize=8.5, fontweight="bold",
                color=C["neg"] if v < 0 else C["pos"])
    for yy, v in zip(y - h / 2, rows["2020"]):
        ax.text(v - .8 if v < 0 else v + .8, yy, f"{v:+.1f}", va="center",
                ha="right" if v < 0 else "left", fontsize=8, color=C["ink2"])
    ax.axvline(0, color=C["ink"], lw=1.2)
    ax.set_yticks(y, [NAME[m] for m in methods])
    ax.set_xlabel("RUB per 100 invested, net of the cash baseline")
    ax.set_title("The scoreboard: nothing beats simply holding cash")
    ax.set_xlim(-42, 21)
    ax.legend(loc="lower right")
    ax.text(3.0, 4.15, "Poland: PCA best (+20%),\nLSTM ≈+10%, ETF ≈+5%\n\n"
            "Russia: the ranking inverts —\nwhoever trades least,\nloses least",
            fontsize=8.5, color=C["ink"], va="center",
            bbox=dict(fc="#fbf7e8", ec=C["etf_real"], boxstyle="round,pad=0.45"))
    finish(ax, "Solid = 2017–2019 (paper's main period), translucent = 2020 (stress). "
               "Positive means the strategy added value over doing nothing; every method "
               "except a rounding-error +0.4 is negative.")
    save(fig, "05_alpha_scoreboard.png")


# =====================================================================
# 6 · The one regime that worked — and how narrow it was
# =====================================================================
def fig_regime():
    panel = panels[("pca_const", "2015-2016")]
    rf = RF["2015-2016"]
    res = backtest(panel, 1.30, 0.90, r_f=rf, traded=traded_mask(data, panel))
    eq = res["equity"]
    tr = pd.DataFrame([{"side": t.side, "pnl": t.pnl} for t in res["trades"]])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.6),
                                   gridspec_kw={"width_ratios": [1.7, 1]})
    cash = 100 * np.exp(rf * np.arange(len(panel.dates)) / 252)
    ax1.plot(eq.index, eq, color=C["pca_const"], lw=1.7, label="PCA, RU-tuned thresholds")
    ax1.plot(panel.dates, cash, color=C["cash"], lw=1.5, ls="--",
             label=f"cash at {rf:.1%}")
    q1 = eq.index <= pd.Timestamp("2015-04-01")
    ax1.fill_between(eq.index[q1], 100, eq[q1], color=C["pos"], alpha=.16, lw=0)
    ax1.annotate("Q1 2015: almost the entire\ntwo-year profit is made here",
                 xy=(pd.Timestamp("2015-02-20"), 195),
                 xytext=(pd.Timestamp("2015-06-01"), 150), fontsize=9,
                 fontweight="bold", color=C["pos"],
                 arrowprops=dict(arrowstyle="->", lw=1.2, color=C["pos"]))
    ax1.annotate("…then 21 months of nothing", xy=(pd.Timestamp("2016-06-01"), 228),
                 xytext=(pd.Timestamp("2015-09-15"), 245), fontsize=9,
                 color=C["ink2"], arrowprops=dict(arrowstyle="->", lw=1, color=C["ink2"]))
    ax1.set_ylabel("RUB per 100 invested")
    ax1.set_title("2015–2016: +123% — but it is one quarter of post-crisis chaos")
    ax1.legend(loc="lower right")
    ax1.tick_params(axis="x", rotation=30)
    finish(ax1)

    by = tr.groupby(tr.side.map({1: "long", -1: "short"}))["pnl"].sum()
    bars = ["shorts\n(366 trades)", "longs\n(308 trades)", "net"]
    vals = [by["short"], by["long"], tr.pnl.sum()]
    cols = [C["pos"], C["neg"], C["pca_const"]]
    ax2.bar(bars, vals, color=cols, width=.6)
    for x, v in zip(bars, vals):
        ax2.text(x, v + (4 if v > 0 else -8), f"{v:+.0f}", ha="center",
                 fontsize=10, fontweight="bold", color=C["ink"])
    ax2.axhline(0, color=C["ink"], lw=1.2)
    ax2.set_ylabel("realized P&L, RUB")
    ax2.set_title("…and it is a short book")
    finish(ax2, "Shorting mid-caps in 2015 Russia was scarce and expensive,\n"
                "and no borrow fee is modelled — so a good part of\nthe short leg was "
                "not realizable in practice.")
    fig.subplots_adjust(wspace=.25)
    save(fig, "06_the_regime_that_worked.png")


# =====================================================================
# 7 · Why it fails: the signal is dead BEFORE fees
# =====================================================================
def fig_gross_net():
    """Uses the RU-TUNED thresholds — i.e. the configuration most favourable to the
    strategy, chosen by grid search on Russian 2015-16 data. Even there: no edge."""
    from matplotlib.patches import Patch
    opt = json.loads(Path("results/optimized_thresholds.json").read_text())
    methods = ["pca_const", "pca_var", "etf_real", "etf_real3", "etf_sector"]
    gross, net, ntr = [], [], []
    for m in methods:
        p = panels[(m, "2017-2019")]
        tm, rf = traded_mask(data, p), RF["2017-2019"]
        g = (opt[m]["g_open"], opt[m]["g_close"])
        rn = backtest(p, *g, r_f=rf, traded=tm, compute_mtm=False)
        rg = backtest(p, *g, r_f=rf, traded=tm, compute_mtm=False, cost=0.0)
        net.append(sum(t.pnl for t in rn["trades"]))
        gross.append(sum(t.pnl for t in rg["trades"]))
        ntr.append(rn["n_trades"])

    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    x = np.arange(len(methods))
    ax.bar(x - .2, gross, width=.4, color=[C[m] for m in methods])
    ax.bar(x + .2, net, width=.4, color=[C[m] for m in methods], alpha=.40,
           hatch="///", edgecolor="white")
    for xx, v in zip(x - .2, gross):
        ax.text(xx, v + (.35 if v >= 0 else -.9), f"{v:+.1f}", ha="center",
                fontsize=9, fontweight="bold", color=C["ink"])
    for xx, v in zip(x + .2, net):
        ax.text(xx, v + (.35 if v >= 0 else -.9), f"{v:+.1f}", ha="center",
                fontsize=9, color=C["neg"] if v < 0 else C["pos"])
    ax.axhline(0, color=C["ink"], lw=1.2)
    ax.set_xticks(x, [f"{NAME[m]}\n({n:,} trades)" for m, n in zip(methods, ntr)])
    ax.set_ylabel("total realized trading P&L, RUB per 100")
    ax.set_ylim(min(net) - 3.5, max(gross) + 3.5)
    ax.set_title("Why it fails: there is no edge to tax — the signal is flat BEFORE fees")
    ax.legend(handles=[Patch(fc=C["ink2"], label="gross: what the signal earns before costs"),
                       Patch(fc=C["ink2"], alpha=.40, hatch="///", ec="white",
                             label="net: after 0.1% per side")],
              loc="lower right", ncol=2)
    ax.annotate(f"{ntr[0]:,} PCA trades → {gross[0]:+.1f} RUB gross: pure noise.\n"
                f"Fees then make it {net[0]:+.1f}.",
                xy=(-.2, gross[0]), xytext=(0.30, 3.2), fontsize=9,
                fontweight="bold", color=C["pca_const"],
                arrowprops=dict(arrowstyle="->", lw=1.1, color=C["pca_const"]))
    finish(ax, "2017–2019, using each method's RU-TUNED thresholds (grid-searched on Russian "
               "2015–16 data) — the configuration most favourable to the strategy. Solid = "
               "gross of costs, hatched = net. If Russia merely had expensive trading, the "
               "solid bars would be clearly positive; they hover at zero. The fees then "
               "convert ≈nothing into a loss. (With the paper's own Polish thresholds the "
               "gross bars are worse still — its overshoot exits lose money before fees.)")
    save(fig, "07_gross_vs_net.png")


# =====================================================================
# 8 · Poland vs Russia — the ranking inversion
# =====================================================================
def fig_poland_russia():
    rows = [("PCA r=15", +20.0, -26.7, "2.63 / 1.01 / −1.16", "−1.08 / +0.88 / −2.66"),
            ("PCA var-r", +20.0, -26.3, "2.51 / 0.44 / −0.91", "−1.08 / +0.88 / −2.28"),
            ("LSTM", +10.0, -37.4, "0.60 / 2.09 / −1.53", "−0.20 / −0.60 / −2.27"),
            ("ETF sparse", +5.0, -1.8, "−0.25 / −0.46 / +1.43", "−0.86 / +0.23 / +0.28"),
            ("ETF sector", +5.0, -6.1, "—", "−0.88 / −0.38 / −0.41")]
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    y = np.arange(len(rows))[::-1]
    for yy, (nm, pl, ru, _, _) in zip(y, rows):
        ax.plot([ru, pl], [yy, yy], color=C["rule"], lw=2.5, zorder=1)
        ax.scatter([pl], [yy], s=110, color=C["pos"], zorder=3)
        ax.scatter([ru], [yy], s=110, color=C["neg"], zorder=3)
        ax.text(pl + 1.2, yy, f"+{pl:.0f}", va="center", fontsize=9,
                fontweight="bold", color=C["pos"])
        ax.text(ru - 1.2, yy, f"{ru:+.1f}", va="center", ha="right", fontsize=9,
                fontweight="bold", color=C["neg"])
    ax.axvline(0, color=C["ink"], lw=1.2)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(-46, 30)
    ax.set_xlabel("2017–2019 outcome, RUB/PLN per 100 invested "
                  "(Poland: reported return · Russia: net of cash)")
    ax.set_title("Same machine, two markets: Poland's best method is Russia's worst")
    ax.scatter([], [], s=110, color=C["pos"], label="Poland (paper)")
    ax.scatter([], [], s=110, color=C["neg"], label="Russia (this replication)")
    ax.legend(loc="lower left")
    finish(ax, "Poland's ranking runs PCA > LSTM > ETF; Russia's runs exactly backwards — "
               "the methods that trade the least (few index factors, few signals) simply "
               "lose the least. A ranking that inverts across markets is a sign the "
               "original ordering reflected the Warsaw regime, not a property of the method.")
    save(fig, "08_poland_vs_russia.png")


for f in (fig_trade, fig_decomposition, fig_tail_bet, fig_equity, fig_alpha,
          fig_regime, fig_gross_net, fig_poland_russia):
    print(f"building {f.__name__}…")
    f()

(FIG / "README.md").write_text("""# Figures — how the strategy worked on the Russian market

Generated by `make_figures.py` from cached `data/` + `results/`. Order = the argument.

| # | file | what it shows |
|---|---|---|
| 1 | `01_anatomy_of_a_trade.png` | **How the strategy works**, one real trade end-to-end: a stock runs ahead of its 15-eigenportfolio replica, the s-score crosses +1.30, the book shorts the stock and buys the replica, the gap closes, +profit. The strategy never bets on direction — only on the gap. |
| 2 | `02_return_decomposition.png` | **Where the tradable signal comes from**: SBER's return split into the systematic part (explained by the factors) and the idiosyncratic residual — the small wiggle that is the only thing traded. |
| 3 | `03_a_tail_bet_not_an_edge.png` | **The key diagnostic.** The typical trade is *identical* in both regimes (trimmed mean +0.36% vs +0.39%, median +1.3% vs +1.1%, win rate 64% vs 65%) and merely covers the ~0.4% round-trip cost. The whole six-year edge is **one quarter** — Q1 2015's ruble-crisis unwind (+25.6%/trade, 111 trades, +101 RUB, at 42% market vol). Strip that quarter and even 2015–16 loses money. |
| 4 | `04_equity_curves.png` | **The outcome**: every method vs the cash baseline, 2017–19 and 2020. What looks like a gain is cash compounding at 7.3%, not alpha. |
| 5 | `05_alpha_scoreboard.png` | **The scoreboard**: final equity minus cash, per method and period. Nothing beats doing nothing; the LSTM is worst at −37 RUB. |
| 6 | `06_the_regime_that_worked.png` | **The exception, in perspective**: 2015–16 made +123%, but nearly all of it lands in Q1 2015 (post-ruble-crisis chaos) and it is a short book — with no borrow fees modelled. |
| 7 | `07_gross_vs_net.png` | **Why it fails**: gross vs net trading P&L. The signal is flat *before* fees — 1,092 PCA trades produce −0.1 RUB gross. It is not a cost problem, it is an alpha problem. |
| 8 | `08_poland_vs_russia.png` | **The ranking inversion**: Poland's PCA > LSTM > ETF becomes Russia's exact reverse. A ranking that flips across markets reflects the regime, not the method. |

Full write-up: `RESULTS.md`. Step-by-step recreation: `replication_steps.ipynb`.
""")
print(f"\ndone — 8 figures + README in {FIG}/")
