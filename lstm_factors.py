"""
lstm_factors.py
===================================================================
Method 3B: learned replicating portfolio. One stacked 2-layer LSTM
(hidden 64) PER stock: input = the other N-1 stocks' returns over a
W=120 window, output = beta_t (N-1 weights) at every step; replica
return X_t^T beta_t. Loss = MSE over the window + p*L1(beta), p=1e-5,
Adam, batch 16, windows sampled from a 3-year training span, retrained
yearly (spec §3B). Residual dI_t = R_t - X_t^T beta_t feeds the same
OU pipeline as every other method (NO intercept, so I_W != 0 here).

Models cached to data/lstm/{stock}_{year}.pt.
===================================================================
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from statarb_pipeline import SignalPanel, W_RESID, _cumlog, fit_ou_panel

HIDDEN = 64
L1_P = 1e-5
BATCH = 16
EPOCHS = 30
LR = 1e-3
TRAIN_YEARS = 3
CACHE = Path(__file__).parent / "data" / "lstm"
CACHE.mkdir(parents=True, exist_ok=True)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class StockLSTM(nn.Module):
    def __init__(self, n_others: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_others, hidden_size=HIDDEN,
                            num_layers=2, batch_first=True)
        self.head = nn.Linear(HIDDEN, n_others)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, W, n_others)
        h, _ = self.lstm(x)
        return self.head(h)                                # (B, W, n_others) betas


def _windows(X: np.ndarray, y: np.ndarray, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """All sliding windows of length w. X: (T, n_others), y: (T,)."""
    T = len(y)
    idx = np.arange(w)[None, :] + np.arange(T - w + 1)[:, None]
    return torch.from_numpy(X[idx]).float(), torch.from_numpy(y[idx]).float()


def train_stock(returns: pd.DataFrame, target: str, year: int,
                epochs: int = EPOCHS, seed: int = 0,
                verbose: bool = False) -> StockLSTM:
    """Train (or load cached) model for `target`, usable during `year`.
    Training span: the TRAIN_YEARS calendar years before `year`."""
    path = CACHE / f"{target}_{year}.pt"
    others = [c for c in returns.columns if c != target]
    net = StockLSTM(len(others))
    if path.exists():
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return net
    torch.manual_seed(seed + hash(target) % 10_000)
    span = returns.loc[f"{year - TRAIN_YEARS}-01-01":f"{year - 1}-12-31"]
    X, y = _windows(span[others].values, span[target].values, W_RESID)
    dev = _device()
    net.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for b0 in range(0, n, BATCH):
            sel = perm[b0:b0 + BATCH]
            xb, yb = X[sel].to(dev), y[sel].to(dev)
            beta = net(xb)                                  # (B, W, n_others)
            repl = (beta * xb).sum(-1)                      # (B, W)
            loss = ((yb - repl) ** 2).mean() + L1_P * beta.abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(sel)
        if verbose and (ep % 10 == 9 or ep == 0):
            print(f"      {target} {year} ep{ep + 1}: loss {tot / n:.3e}")
    net.to("cpu")
    torch.save(net.state_dict(), path)
    return net


@torch.no_grad()
def predict_betas(net: StockLSTM, returns: pd.DataFrame, target: str,
                  dates: pd.DatetimeIndex) -> pd.DataFrame:
    """beta_t (last-step output) for each date in `dates`; window = trailing
    W_RESID days of the other stocks' returns ending at t."""
    others = [c for c in returns.columns if c != target]
    dev = _device()
    net.to(dev).eval()
    Xall = returns[others].values
    pos = returns.index.get_indexer(dates)
    assert (pos >= W_RESID - 1).all(), "need W_RESID history before first date"
    wins = np.stack([Xall[p - W_RESID + 1:p + 1] for p in pos])   # (T, W, n_others)
    out = []
    for b0 in range(0, len(wins), 128):
        xb = torch.from_numpy(wins[b0:b0 + 128]).float().to(dev)
        out.append(net(xb)[:, -1, :].cpu().numpy())               # last-step beta
    net.to("cpu")
    return pd.DataFrame(np.vstack(out), index=dates, columns=others)


def build_panel_lstm(returns: pd.DataFrame, period: tuple[str, str],
                     epochs: int = EPOCHS, verbose: bool = True) -> SignalPanel:
    """SignalPanel for the LSTM method over `period`.

    Paper cadence (spec §3B): ONE model per stock per backtest YEAR, trained on
    the 3 prior calendar years, then run from W_RESID days before Jan 1 through
    Dec 31 (the warmup residuals deliberately use the target year's model, so
    they overlap its training span — as in the paper). Every day's OU fit runs
    on the trailing W_RESID residuals produced by that year's model."""
    stocks = list(returns.columns)
    dates_all = returns.index
    t0 = dates_all.searchsorted(pd.Timestamp(period[0]))
    t1 = dates_all.searchsorted(pd.Timestamp(period[1]), side="right")
    dates = dates_all[t0:t1]
    N = len(stocks)
    T = len(dates)
    s_p = np.full((T, N), np.nan)
    k_p = np.full((T, N), np.nan)
    w_p = np.full((T, N, N), np.nan)

    for Y in sorted(set(dates.year)):
        loc = np.where(dates.year == Y)[0]
        y_start, y_end = t0 + loc[0], t0 + loc[-1] + 1          # global idx
        r0 = y_start - (W_RESID - 1)                            # warmup start
        assert r0 >= 0, "not enough history for LSTM warmup"
        ydates = dates_all[r0:y_end]
        residY = np.full((len(ydates), N), np.nan)
        betaY: list[np.ndarray] = [None] * N
        for si, tkr in enumerate(stocks):
            net = train_stock(returns, tkr, Y, epochs=epochs)
            B = predict_betas(net, returns, tkr, ydates)        # (len(ydates), N-1)
            others = list(B.columns)
            X = returns.loc[ydates, others].values
            y_t = returns.loc[ydates, tkr].values
            residY[:, si] = y_t - (B.values * X).sum(1)
            betaY[si] = B.values
            if verbose:
                print(f"   [lstm] {Y} {tkr} ({si + 1}/{N}) resid std "
                      f"{np.nanstd(residY[:, si]):.4f}", flush=True)
        for gi in range(y_start, y_end):
            row, li = gi - t0, gi - r0
            I = np.cumsum(residY[li - W_RESID + 1:li + 1], axis=0)
            kappa, s, _ = fit_ou_panel(I)
            k_p[row], s_p[row] = kappa, s
            for si in range(N):
                w_p[row, si] = np.insert(betaY[si][li], si, 0.0)

    ret_win = returns.iloc[t0:t1].values
    cum = _cumlog(ret_win)
    return SignalPanel(dates=dates, stocks=stocks, s=s_p, kappa=k_p, w=w_p,
                       instr_logret=ret_win, instr_cumlog=cum, stock_cumlog=cum,
                       method="lstm")


# ------------------------------------------------------------------
if __name__ == "__main__":
    # tiny smoke test: 6 fake stocks, fast settings
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2014-01-01", periods=1100)
    F = rng.normal(0, 0.01, size=(1100, 2))
    B = rng.normal(0.7, 0.3, size=(2, 6))
    R = F @ B + rng.normal(0, 0.01, size=(1100, 6))
    rets = pd.DataFrame(R, index=dates, columns=[f"S{i}" for i in range(6)])
    import lstm_factors as lf
    lf.EPOCHS = 2
    panel = build_panel_lstm(rets, ("2017-01-02", str(dates[-1].date())),
                             epochs=2, verbose=True)
    print("panel s finite share:", np.isfinite(panel.s).mean())
    print("smoke OK")
