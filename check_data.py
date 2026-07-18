"""
check_data.py — post-fetch data validation. Run after moex_data.py.
Verifies: panel shapes/coverage, dividend adjustment magnitude for known
big payers, index completeness, risk-free levels, and prints the universe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import moex_data

d = moex_data.load_all()
close, adj, idx = d["close"], d["adj"], d["indices"]
uni = d["universe"]

print(f"universe ({len(uni)}): {', '.join(uni)}\n")
print(f"price panel: {close.shape[0]} days x {close.shape[1]} stocks, "
      f"{close.index.min().date()} .. {close.index.max().date()}")
cov = close.loc['2014-03-01':].notna().mean().sort_values()
print(f"coverage: min {cov.iloc[0]:.1%} ({cov.index[0]}), "
      f"median {cov.median():.1%}")

# dividend adjustment: total adjustment factor per stock = adj[0]/close[0] vs 1
both = close.notna() & adj.notna()
first = {t: close.index[both[t]][0] for t in uni if both[t].any()}
fac = pd.Series({t: float(adj.loc[first[t], t] / close.loc[first[t], t])
                 for t in first})
print("\nlargest total dividend adjustments (adj/raw at series start):")
print((1 - fac).sort_values(ascending=False).head(12).map("{:.1%}".format).to_string())
weak = (1 - fac)
for t in ("SBER", "MTSS", "GAZP", "LKOH"):
    if t in weak.index and weak[t] < 0.15:
        print(f"  [warn] {t}: only {weak[t]:.1%} total adjustment over 7 years — "
              f"dividends likely missing!")

print("\nindices coverage:")
for c in idx.columns:
    s = idx[c].dropna()
    print(f"  {c:8s}: {s.index.min().date()} .. {s.index.max().date()} ({len(s)})")

print("\nrisk-free period averages:", {k: f"{v:.2%}" for k, v in d["rf"].items()})

# quick eyeball: 2017-2019 total returns of key indices
for c in ("MCFTR", "MESMTR"):
    s = idx[c].dropna()
    r = s.asof(pd.Timestamp("2019-12-30")) / s.asof(pd.Timestamp("2017-01-03")) - 1
    print(f"  {c} 2017-2019 total return: {r:+.1%}")

n_div = pd.read_csv(moex_data.DATA_DIR / "dividends.csv")
print(f"\ndividend records in window: {len(n_div)} across "
      f"{n_div['ticker'].nunique()} tickers "
      f"(sources: {dict(n_div['src'].value_counts())})")
print("\nCHECK DONE")
