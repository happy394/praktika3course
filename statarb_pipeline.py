"""
statarb_pipeline.py
===================================================================
Engine for the stat-arb replication (Avellaneda-Lee 2008 / Adamczyk-
Dabrowski Polish paper), adapted to run on any market panel — here MOEX
(see RU_ADAPTATION.md; method parameters follow REPLICATION_SPEC.md).

Two-stage design:

  Stage 1  SignalPanel precompute (per factor method, per period)
           for each day t, stock i: OLS beta over trailing W=120 on the
           method's factors -> residual increments -> cumulate -> AR(1)
           -> OU params -> kappa + s-score; plus the replica weight
           vector in *instrument space* (stocks for PCA/LSTM, indices
           for ETF methods) used to price the hedge leg of a trade.

  Stage 2  backtest(): day-by-day state machine over 60 independent
           per-stock traders. Frozen weights at open, paper's P&L
           formula, E_t accumulation, per-year Sharpe. Cheap ->
           threshold grid search reuses stage-1 panels.
===================================================================
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

# ---- constants from REPLICATION_SPEC.md §4/§5 --------------------
DT = 1.0 / 252.0
N_COV = 252                 # PCA covariance window (prior calendar year)
W_RESID = 120               # residual + OU estimation window (all methods)
KAPPA_MIN = 4.0             # tradeability filter
E0 = 100.0
LEVERAGE = 2.0
COST = 0.001                # 0.1% per side
STOP_OPEN_BEFORE_END = 60   # = W/2

# paper's Polish-optimized thresholds (transfer test)
THRESHOLDS = {
    "pca_const":  (1.10, -0.50),
    "pca_var":    (1.10, -0.50),
    "lstm":       (1.10, -0.15),
    "etf_real":   (2.10,  0.75),
    "etf_real3":  (2.10,  0.75),
    "etf_sector": (1.95,  0.40),
}


# ==================================================================
# Stage 1a: vectorized OU fit across all stocks for one day
# ==================================================================
def fit_ou_panel(I: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """I: (W x N) cumulated residual paths. Returns (kappa, s, valid) each (N,).

    AR(1): I_k = phi0 + phi1 I_{k-1} + zeta. OU: kappa=-ln(phi1)*252,
    mu=phi0/(1-phi1), equilibrium std = sqrt(var_zeta/(1-phi1^2)),
    s = (I_last - mu)/eq_std."""
    x, y = I[:-1], I[1:]
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x - mx, y - my
    sxx = (xc * xc).sum(0)
    sxy = (xc * yc).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi1 = np.where(sxx > 0, sxy / sxx, np.nan)
        phi0 = my - phi1 * mx
        resid = y - (phi0 + phi1 * x)
        var_z = resid.var(0, ddof=1)
        valid = (phi1 > 1e-8) & (phi1 < 1 - 1e-8) & np.isfinite(phi1) & (var_z > 0)
        kappa = np.where(valid, -np.log(phi1) * 252.0, np.nan)
        mu = np.where(valid, phi0 / (1.0 - phi1), np.nan)
        eq_std = np.where(valid, np.sqrt(var_z / (1.0 - phi1 ** 2)), np.nan)
        s = np.where(valid & (eq_std > 0), (I[-1] - mu) / eq_std, np.nan)
    return kappa, s, valid & np.isfinite(s)


def ols_residuals_multi(F: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS of every stock column of Y (W x N) on factors F (W x r) + intercept.
    Returns (residuals W x N, betas r x N) — intercept row dropped from betas."""
    X = np.column_stack([np.ones(len(F)), F])
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ coef
    return resid, coef[1:]


# ==================================================================
# Stage 1b: signal panels per factor method
# ==================================================================
@dataclass
class SignalPanel:
    """Everything stage 2 needs. T backtest days, N stocks, K instruments."""
    dates: pd.DatetimeIndex          # (T,)
    stocks: list[str]                # (N,)
    s: np.ndarray                    # (T, N) s-scores
    kappa: np.ndarray                # (T, N)
    w: np.ndarray                    # (T, N, K) replica weights, instrument space
    instr_logret: np.ndarray         # (T_full_hist rows aligned to dates: see cum)
    instr_cumlog: np.ndarray         # (T, K) cumulative log(1+r) of instruments
    stock_cumlog: np.ndarray         # (T, N) cumulative log(1+r) of the stocks
    method: str = ""


def _cumlog(returns: np.ndarray) -> np.ndarray:
    return np.cumsum(np.log1p(np.nan_to_num(returns, nan=0.0)), axis=0)


def build_panel_pca(returns: pd.DataFrame, period: tuple[str, str],
                    r_factors: int | None = 15, var_threshold: float = 0.55,
                    method_name: str = "pca_const") -> SignalPanel:
    """PCA eigenportfolio factors. Eigenvectors refit each Jan 1 on the prior
    calendar year's 252 returns (normalized); Q_k = eigvec_k / sigma_k.
    Instruments = the stocks themselves: w = Q @ beta."""
    stocks = list(returns.columns)
    R = returns.values
    dates_all = returns.index
    t0, t1 = (dates_all.searchsorted(pd.Timestamp(period[0])),
              dates_all.searchsorted(pd.Timestamp(period[1]), side="right"))
    span = range(t0, t1)
    T, N = len(span), len(stocks)
    s_p = np.full((T, N), np.nan)
    k_p = np.full((T, N), np.nan)
    w_p = np.full((T, N, N), np.nan)
    chosen_r: dict[int, int] = {}

    Q_by_year: dict[int, np.ndarray] = {}

    def eig_Q(year: int) -> np.ndarray:
        if year in Q_by_year:
            return Q_by_year[year]
        # prior calendar year's data, last N_COV rows
        end = dates_all.searchsorted(pd.Timestamp(f"{year}-01-01"))
        win = R[max(0, end - N_COV):end]
        mean, std = win.mean(0), win.std(0, ddof=1)
        std = np.where(std > 0, std, 1.0)
        Ynorm = (win - mean) / std
        corr = (Ynorm.T @ Ynorm) / (len(win) - 1)
        eigval, eigvec = np.linalg.eigh(corr)
        order = np.argsort(eigval)[::-1]
        eigval, eigvec = eigval[order], eigvec[:, order]
        if r_factors is None:
            cum = np.cumsum(eigval) / eigval.sum()
            r = int(np.searchsorted(cum, var_threshold) + 1)
        else:
            r = r_factors
        chosen_r[year] = r
        # sign convention: market portfolio positive
        for k in range(r):
            if eigvec[:, k].sum() < 0:
                eigvec[:, k] = -eigvec[:, k]
        Q_by_year[year] = eigvec[:, :r] / std[:, None]
        return Q_by_year[year]

    for row, ti in enumerate(span):
        if ti < W_RESID - 1:
            continue
        year = dates_all[ti].year
        Q = eig_Q(year)                       # (N x r)
        Rwin = R[ti - W_RESID + 1:ti + 1]     # trailing W days incl. today's close
        F = Rwin @ Q                          # (W x r) eigenportfolio returns
        resid, beta = ols_residuals_multi(F, Rwin)
        I = np.cumsum(resid, axis=0)
        kappa, s, _ = fit_ou_panel(I)
        k_p[row], s_p[row] = kappa, s
        w_p[row] = (Q @ beta).T               # (N stocks_target x N instruments)
    if chosen_r:
        print(f"   [{method_name}] r by year: {chosen_r}")

    dates = dates_all[t0:t1]
    ret_win = returns.iloc[t0:t1].values
    cum = _cumlog(ret_win)
    return SignalPanel(dates=dates, stocks=stocks, s=s_p, kappa=k_p, w=w_p,
                       instr_logret=ret_win, instr_cumlog=cum, stock_cumlog=cum,
                       method=method_name)


def build_panel_etf(returns: pd.DataFrame, factor_returns: pd.DataFrame,
                    period: tuple[str, str], method_name: str) -> SignalPanel:
    """ETF/index factors: F = index TR returns, instruments = the indices."""
    stocks = list(returns.columns)
    fnames = list(factor_returns.columns)
    # align factor panel to the stock calendar
    fac = factor_returns.reindex(returns.index).ffill().fillna(0.0)
    R, Fall = returns.values, fac.values
    dates_all = returns.index
    t0, t1 = (dates_all.searchsorted(pd.Timestamp(period[0])),
              dates_all.searchsorted(pd.Timestamp(period[1]), side="right"))
    span = range(t0, t1)
    T, N, K = len(span), len(stocks), len(fnames)
    s_p = np.full((T, N), np.nan)
    k_p = np.full((T, N), np.nan)
    w_p = np.full((T, N, K), np.nan)

    for row, ti in enumerate(span):
        if ti < W_RESID - 1:
            continue
        Rwin = R[ti - W_RESID + 1:ti + 1]
        Fwin = Fall[ti - W_RESID + 1:ti + 1]
        resid, beta = ols_residuals_multi(Fwin, Rwin)
        I = np.cumsum(resid, axis=0)
        kappa, s, _ = fit_ou_panel(I)
        k_p[row], s_p[row] = kappa, s
        w_p[row] = beta.T
    dates = dates_all[t0:t1]
    ret_win = returns.iloc[t0:t1].values
    return SignalPanel(dates=dates, stocks=stocks, s=s_p, kappa=k_p, w=w_p,
                       instr_logret=Fall[t0:t1], instr_cumlog=_cumlog(Fall[t0:t1]),
                       stock_cumlog=_cumlog(ret_win), method=method_name)


# ==================================================================
# Stage 2: backtest state machine (paper's accounting)
# ==================================================================
@dataclass
class Trade:
    stock: str
    side: int            # +1 long stock / -1 short stock
    t_open: int          # row index into panel.dates
    t_close: int = -1
    scale: float = 0.0   # Lambda at open (frozen)
    qM: float = 0.0      # algebraic RUB in replica per 1 RUB stock (frozen)
    pnl: float = np.nan
    s_open: float = np.nan
    s_close: float = np.nan
    w_vec: np.ndarray | None = None   # frozen replica weights (instrument space)


def backtest(panel: SignalPanel, g_open: float, g_close: float, r_f: float,
             e0: float = E0, cost: float = COST, leverage: float = LEVERAGE,
             kappa_min: float = KAPPA_MIN, compute_mtm: bool = True,
             traded: np.ndarray | None = None) -> dict:
    """Paper's accounting: 60 independent traders, frozen weights, realized-
    only equity E_t = e0*exp(rf t) + sum pnl_i*exp(rf (t-t1)). Also returns a
    marked-to-market curve for honesty (not in the paper)."""
    T, N = panel.s.shape
    n_traders = N
    lam_frac = leverage / n_traders          # Lambda_t = lev/N * E_t
    stop_open = T - STOP_OPEN_BEFORE_END

    pos = np.zeros(N, dtype=int)
    open_tr: dict[int, Trade] = {}
    closed: list[Trade] = []
    E = np.empty(T)
    M = np.empty(T)                          # marked-to-market variant
    E_prev = e0

    def hedge_ret(tr: Trade, t: int, i: int) -> tuple[float, float]:
        """(stock simple return, hedge-leg PLN return H) over [t_open, t]."""
        r_stock = np.expm1(panel.stock_cumlog[t, i] - panel.stock_cumlog[tr.t_open, i])
        gross = np.expm1(panel.instr_cumlog[t] - panel.instr_cumlog[tr.t_open])  # (K,)
        H = float(np.dot(tr.w_vec, gross))
        return float(r_stock), H

    for t in range(T):
        # --- close decisions, then open decisions (one position per stock) --
        for i in range(N):
            s = panel.s[t, i]
            if pos[i] != 0:
                if not np.isfinite(s):
                    force = (t == T - 1)
                    if not force:
                        continue
                close_sig = (t == T - 1)
                if np.isfinite(s):
                    if pos[i] == +1 and s > -g_close:
                        close_sig = True
                    elif pos[i] == -1 and s < +g_close:
                        close_sig = True
                if close_sig:
                    tr = open_tr.pop(i)
                    r_stock, H = hedge_ret(tr, t, i)
                    dt_y = (t - tr.t_open) * DT
                    hold = np.exp(r_f * dt_y)
                    sgn = 1.0 if tr.side == +1 else -1.0
                    # long: +1 stock, -qM replica, cash leg (qM-1) accrues rf
                    core = sgn * ((r_stock - H) - (hold - 1.0) * (1.0 - tr.qM))
                    # paper's cost term: notional traded at open and at close
                    fees = cost * (hold * abs(1.0 + tr.qM)
                                   + abs((1.0 + r_stock) + (tr.qM + H)))
                    tr.pnl = tr.scale * (core - fees)
                    tr.t_close = t
                    tr.s_close = s if np.isfinite(s) else np.nan
                    closed.append(tr)
                    pos[i] = 0
            elif t < stop_open and np.isfinite(s):
                if traded is not None and not traded[t, i]:
                    continue                       # suspended: no new position
                k = panel.kappa[t, i]
                if not np.isfinite(k) or k <= kappa_min:
                    continue
                side = +1 if s < -g_open else (-1 if s > +g_open else 0)
                if side != 0:
                    w_vec = panel.w[t, i].copy()
                    if not np.all(np.isfinite(w_vec)):
                        continue
                    tr = Trade(stock=panel.stocks[i], side=side, t_open=t,
                               scale=lam_frac * E_prev, qM=float(w_vec.sum()),
                               s_open=float(s))
                    tr.w_vec = w_vec
                    open_tr[i] = tr
                    pos[i] = side

        # --- equity curves ------------------------------------------------
        t_y = t * DT
        acc = e0 * np.exp(r_f * t_y)
        acc += sum(tr.pnl * np.exp(r_f * (t - tr.t_close) * DT) for tr in closed)
        E[t] = acc
        # MTM: value open trades as-if closed today (incl. exit costs)
        if compute_mtm:
            mtm = 0.0
            for i, tr in open_tr.items():
                r_stock, H = hedge_ret(tr, t, i)
                dt_y = (t - tr.t_open) * DT
                hold = np.exp(r_f * dt_y)
                sgn = 1.0 if tr.side == +1 else -1.0
                core = sgn * ((r_stock - H) - (hold - 1.0) * (1.0 - tr.qM))
                fees = cost * (hold * abs(1.0 + tr.qM)
                               + abs((1.0 + r_stock) + (tr.qM + H)))
                mtm += tr.scale * (core - fees)
            M[t] = acc + mtm
        else:
            M[t] = acc
        E_prev = E[t]

    eq = pd.Series(E, index=panel.dates, name="E")
    mtm_eq = pd.Series(M, index=panel.dates, name="M")
    return {
        "equity": eq,
        "mtm": mtm_eq,
        "final_return": float(eq.iloc[-1] / e0 - 1.0),
        "sharpe_by_year": sharpe_by_year(eq, r_f),
        "sharpe_by_year_mtm": sharpe_by_year(mtm_eq, r_f),
        "trades": closed,
        "n_trades": len(closed),
    }


def sharpe_by_year(equity: pd.Series, r_f: float) -> dict[int, float]:
    """S_y = (252*mean_daily - r_f) / (sqrt(252)*std_daily), per calendar year."""
    out = {}
    daily = equity.pct_change().dropna()
    for year, grp in daily.groupby(daily.index.year):
        sd = grp.std()
        out[int(year)] = float((252 * grp.mean() - r_f) / (np.sqrt(252) * sd)) \
            if sd > 0 else np.nan
    return out


def trade_stats(trades: list[Trade]) -> dict:
    if not trades:
        return {"n": 0}
    pnls = np.array([tr.pnl for tr in trades])
    hold = np.array([tr.t_close - tr.t_open for tr in trades])
    return {
        "n": len(trades),
        "win_rate": float((pnls > 0).mean()),
        "avg_pnl": float(pnls.mean()),
        "median_hold_days": float(np.median(hold)),
        "long_share": float(np.mean([tr.side == 1 for tr in trades])),
    }
