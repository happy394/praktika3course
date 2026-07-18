"""
run_lstm.py — LSTM method (spec §3B) on MOEX data: 2017-2019 + 2020 with the
paper's thresholds (transfer test). Models cached; safe to rerun after a crash.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

import lstm_factors
from lstm_factors import build_panel_lstm, train_stock
from run_experiments import PERIODS, prepare_data, traded_mask
from statarb_pipeline import THRESHOLDS, backtest, trade_stats

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)


def main() -> None:
    data = prepare_data()
    R = data["returns"]
    go, gc = THRESHOLDS["lstm"]

    # timing probe: one model
    t0 = time.time()
    train_stock(R, R.columns[0], 2017)
    print(f"[probe] one model trained in {time.time() - t0:.0f}s "
          f"-> est. total {(time.time() - t0) * 240 / 60:.0f} min "
          f"(240 stock-years, cached ones are free)", flush=True)

    out = []
    for period in ("2017-2019", "2020"):
        print(f"== LSTM panel {period}", flush=True)
        t0 = time.time()
        panel = build_panel_lstm(R, PERIODS[period], verbose=True)
        print(f"   panel built in {(time.time() - t0) / 60:.1f} min", flush=True)
        res = backtest(panel, go, gc, r_f=data["rf"][period],
                       traded=traded_mask(data, panel))
        ts = trade_stats(res["trades"])
        row = {
            "method": "lstm", "period": period, "tag": "paper_thr",
            "g_open": go, "g_close": gc,
            "final_return": res["final_return"],
            "sharpe_by_year": res["sharpe_by_year"],
            "trades": ts,
        }
        out.append(row)
        pd.DataFrame({"E": res["equity"], "M": res["mtm"]}).to_csv(
            RESULTS / f"equity_lstm_{period}_paper_thr.csv")
        print(f"  lstm {period}: ret={res['final_return']:+.2%} "
              f"sharpe={res['sharpe_by_year']} trades={ts.get('n', 0)} "
              f"win={ts.get('win_rate', float('nan')):.0%}", flush=True)
    (RESULTS / "summary_lstm.json").write_text(json.dumps(out, indent=2))
    print("LSTM DONE", flush=True)


if __name__ == "__main__":
    main()
