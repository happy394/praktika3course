"""
Sanity harness for statarb_pipeline on synthetic data where the model is
TRUE by construction: returns = beta @ factor_returns + dI, with dI an OU
process (kappa ~ 15-40). On such data the strategy must:
  * produce s-scores that are ~N(0,1)-ish and kappa estimates near truth
  * make money with sensible thresholds (gross of the tiny cost)
Run: python3 test_synthetic.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from statarb_pipeline import (backtest, build_panel_etf, build_panel_pca,
                              fit_ou_panel, trade_stats)

rng = np.random.default_rng(7)

N_DAYS, N_STOCKS, N_FAC = 1300, 40, 3
dates = pd.bdate_range("2015-01-01", periods=N_DAYS)

# factors: persistent market-ish returns
F = rng.normal(0.0003, 0.01, size=(N_DAYS, N_FAC))
beta_true = rng.normal(0.8, 0.4, size=(N_FAC, N_STOCKS))

# OU idiosyncratic LEVEL process I_t: dI = kappa*(0 - I)dt + sigma dW
# realistic calibration: idio vol 15-35%/yr, kappa 8-30 => equilibrium
# residual std sigma/sqrt(2 kappa) ~ 2-6% >> round-trip cost 0.35%
kappa_true = rng.uniform(8, 30, size=N_STOCKS)
sig = rng.uniform(0.15, 0.35, size=N_STOCKS)
I = np.zeros((N_DAYS + 1, N_STOCKS))
dt = 1 / 252
for t in range(1, N_DAYS + 1):
    I[t] = I[t - 1] + kappa_true * (0 - I[t - 1]) * dt \
           + sig[None, :] * np.sqrt(dt) * rng.normal(size=N_STOCKS)
dI = np.diff(I, axis=0)

R = F @ beta_true + dI
returns = pd.DataFrame(R, index=dates, columns=[f"S{i:02d}" for i in range(N_STOCKS)])
fac_ret = pd.DataFrame(F, index=dates, columns=["F1", "F2", "F3"])

# --- 1. OU fit sanity on a raw window --------------------------------
win = dI[-120:]
kap, s, valid = fit_ou_panel(np.cumsum(win, axis=0))
print(f"OU fit: {valid.sum()}/{N_STOCKS} valid | kappa est median "
      f"{np.nanmedian(kap):.1f} (true median {np.median(kappa_true):.1f})")

# --- 2. ETF-style panel with the TRUE factors ------------------------
period = ("2016-01-04", str(dates[-1].date()))
panel = build_panel_etf(returns, fac_ret, period, "synthetic_true_fac")
inside = np.isfinite(panel.s)
print(f"panel: {inside.mean():.0%} of day/stock cells have s-scores; "
      f"s std={np.nanstd(panel.s):.2f} (want ~1); "
      f"kappa>4 share={np.nanmean(panel.kappa > 4):.0%}")

res = backtest(panel, g_open=1.10, g_close=0.50, r_f=0.02)
ts = trade_stats(res["trades"])
print(f"backtest (g=1.10/0.50, cost 0.1%): final={res['final_return']:+.2%} "
      f"trades={ts['n']} win={ts['win_rate']:.0%} "
      f"median_hold={ts['median_hold_days']:.0f}d")
assert res["final_return"] > 0.02, "must profit when the model is true!"

# --- 3. PCA panel runs and is shaped right ---------------------------
panel_pca = build_panel_pca(returns, period, r_factors=3, method_name="pca_synth")
res2 = backtest(panel_pca, g_open=1.10, g_close=0.50, r_f=0.02)
ts2 = trade_stats(res2["trades"])
print(f"PCA r=3 synthetic: final={res2['final_return']:+.2%} trades={ts2['n']} "
      f"win={ts2.get('win_rate', float('nan')):.0%}")
assert np.isfinite(res2["final_return"])

# --- 4. no-signal degenerate: absurd thresholds => ~rf growth only ----
res3 = backtest(panel, g_open=9.0, g_close=8.0, r_f=0.02)
drift = res3["equity"].iloc[-1] / 100.0 - 1.0
print(f"no-trade run: {res3['n_trades']} trades, equity drift {drift:+.2%} "
      f"(pure rf compounding — sanity)")
assert res3["n_trades"] == 0

print("\nALL SYNTHETIC CHECKS PASSED")
