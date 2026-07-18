"""
run_experiments.py
===================================================================
Driver for the MOEX adaptation (RU_ADAPTATION.md §6):

  1. transfer  — run all methods on 2017-2019 + 2020 with the PAPER's
                 Polish-optimized thresholds (pure transfer test)
  2. optimize  — grid-search (g_open, g_close) per method on 2015-2016
                 RU data (objective: final profit, like the paper),
                 then rerun 2017-2019 + 2020 with RU thresholds
  3. all       — both (default)

Results -> results/summary.json, results/equity_*.csv, stdout tables.
===================================================================
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import moex_data
from statarb_pipeline import (THRESHOLDS, SignalPanel, backtest,
                              build_panel_etf, build_panel_pca, trade_stats)

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

PERIODS = {
    "2015-2016": ("2015-01-01", "2016-12-31"),
    "2017-2019": ("2017-01-01", "2019-12-31"),
    "2020":      ("2020-01-01", "2020-12-31"),
}

GRID_OPEN = np.round(np.arange(0.80, 2.45, 0.10), 2)     # 17 values
GRID_CLOSE = np.round(np.arange(-0.80, 1.05, 0.10), 2)   # 19 values


# ------------------------------------------------------------------
def prepare_data() -> dict:
    d = moex_data.load_all()
    adj = d["adj"].ffill()
    returns = adj.pct_change().iloc[1:].fillna(0.0)
    traded = d["close"].notna().iloc[1:]                 # aligned to returns
    # align index prices to the stock calendar BEFORE differencing
    idx_ret = (d["indices"].reindex(d["adj"].index).ffill()
               .pct_change().iloc[1:].fillna(0.0))
    factor_sets = {
        "etf_real":   idx_ret[moex_data.SPARSE_ETF],
        "etf_real3":  idx_ret[moex_data.SPARSE_ETF3],
        "etf_sector": idx_ret[moex_data.SECTOR_ETF],
    }
    return {"returns": returns, "traded": traded, "factor_sets": factor_sets,
            "rf": d["rf"], "universe": d["universe"]}


def build_panel(method: str, data: dict, period_key: str) -> SignalPanel:
    period = PERIODS[period_key]
    R = data["returns"]
    if method == "pca_const":
        return build_panel_pca(R, period, r_factors=15, method_name=method)
    if method == "pca_var":
        return build_panel_pca(R, period, r_factors=None, var_threshold=0.55,
                               method_name=method)
    if method in data["factor_sets"]:
        return build_panel_etf(R, data["factor_sets"][method], period, method)
    raise ValueError(method)


def traded_mask(data: dict, panel: SignalPanel) -> np.ndarray:
    return data["traded"].reindex(panel.dates).fillna(False).values


def run_one(method: str, data: dict, period_key: str, g_open: float,
            g_close: float, tag: str) -> dict:
    t0 = time.time()
    panel = build_panel(method, data, period_key)
    res = backtest(panel, g_open, g_close, r_f=data["rf"][period_key],
                   traded=traded_mask(data, panel))
    ts = trade_stats(res["trades"])
    out = {
        "method": method, "period": period_key, "tag": tag,
        "g_open": g_open, "g_close": g_close,
        "final_return": res["final_return"],
        "final_return_mtm": float(res["mtm"].iloc[-1] / 100.0 - 1.0),
        "sharpe_by_year": res["sharpe_by_year"],
        "sharpe_by_year_mtm": res["sharpe_by_year_mtm"],
        "trades": ts,
        "runtime_s": round(time.time() - t0, 1),
    }
    eq = pd.DataFrame({"E": res["equity"], "M": res["mtm"]})
    eq.to_csv(RESULTS / f"equity_{method}_{period_key}_{tag}.csv")
    print(f"  {method:11s} {period_key:9s} [{tag}] g=({g_open:+.2f},{g_close:+.2f})"
          f"  ret={res['final_return']:+7.2%}  mtm={out['final_return_mtm']:+7.2%}"
          f"  sharpe={ {y: round(s,2) for y,s in res['sharpe_by_year'].items()} }"
          f"  trades={ts.get('n',0)} win={ts.get('win_rate',float('nan')):.0%}"
          f"  ({out['runtime_s']}s)")
    return out


def grid_search(method: str, data: dict, period_key: str = "2015-2016") -> dict:
    """Paper's objective: final accumulated profit on the optimization period.
    Symmetric cutoffs; panel built once, backtest reran over the grid."""
    panel = build_panel(method, data, period_key)
    tm = traded_mask(data, panel)
    rf = data["rf"][period_key]
    best = None
    t0 = time.time()
    for go in GRID_OPEN:
        for gc in GRID_CLOSE:
            if gc >= go:                       # close bound must sit inside open bound
                continue
            res = backtest(panel, float(go), float(gc), r_f=rf,
                           compute_mtm=False, traded=tm)
            profit = res["final_return"]
            if best is None or profit > best["profit"]:
                best = {"g_open": float(go), "g_close": float(gc),
                        "profit": profit, "n_trades": res["n_trades"]}
    best["runtime_s"] = round(time.time() - t0, 1)
    print(f"  {method:11s} grid 2015-16: best g=({best['g_open']:+.2f},"
          f"{best['g_close']:+.2f}) profit={best['profit']:+.2%} "
          f"trades={best['n_trades']} ({best['runtime_s']}s)")
    return best


# ------------------------------------------------------------------
def main(mode: str = "all") -> None:
    data = prepare_data()
    print(f"universe: {len(data['universe'])} stocks | rf: "
          f"{ {k: f'{v:.2%}' for k, v in data['rf'].items()} }")
    methods = ["pca_const", "pca_var", "etf_real", "etf_real3", "etf_sector"]
    summary: list[dict] = []

    if mode in ("transfer", "all"):
        print("\n== TRANSFER TEST: paper's Polish thresholds on RU data ==")
        for period in ("2017-2019", "2020"):
            for m in methods:
                go, gc = THRESHOLDS[m]
                summary.append(run_one(m, data, period, go, gc, tag="paper_thr"))

    if mode in ("optimize", "all"):
        print("\n== GRID SEARCH on 2015-2016 (RU-optimized thresholds) ==")
        opt = {}
        for m in methods:
            opt[m] = grid_search(m, data)
        (RESULTS / "optimized_thresholds.json").write_text(json.dumps(opt, indent=2))
        print("\n== RERUN with RU-optimized thresholds ==")
        for period in ("2017-2019", "2020"):
            for m in methods:
                summary.append(run_one(m, data, period, opt[m]["g_open"],
                                       opt[m]["g_close"], tag="ru_thr"))

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {len(summary)} runs -> {RESULTS}/summary.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
