"""
moex_data.py
===================================================================
Data layer for the Russian (MOEX) adaptation of the Polish stat-arb
replication. See RU_ADAPTATION.md for every design decision.

Provides:
  * MOEX ISS fetchers: stock candles, index history, historical index
    compositions, zero-coupon 1Y yields, dividends
  * dohod.ru dividend-history parser (ISS dividends are incomplete pre-2018)
  * dividend-adjusted close builder (T+2 ex-date logic)
  * universe construction: top-60 MOEXBMI names by average index weight
  * caching of everything under data/

Run:  python3 moex_data.py          # fetch + build everything
===================================================================
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

ISS = "https://iss.moex.com/iss"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"}

FETCH_FROM = "2013-06-01"           # history start (LSTM pre-training buffer)
FETCH_TILL = "2020-12-31"
COVERAGE_FROM = "2014-03-01"        # universe filter: need data from here
COVERAGE_MIN = 0.95                 # >=95% of trading days present

# Index factor sets (RU_ADAPTATION.md §3)
SPARSE_ETF = ["MCFTR", "MESMTR"]                       # large TR + SMID TR
SPARSE_ETF3 = ["MCFTR", "MESMTR", "RGBITR"]            # + state-bond TR variant
SECTOR_ETF = ["MEOGTR", "MEMMTR", "MEFNTR", "MECNTR",
              "MEEUTR", "METLTR", "METNTR", "MECHTR"]  # 8 sector TR indices
ALL_INDICES = sorted(set(SPARSE_ETF3 + SECTOR_ETF + ["IMOEX", "MCXSM"]))


# ------------------------------------------------------------------
# low-level fetch
# ------------------------------------------------------------------
def _get(url: str, retries: int = 6, pause: float = 0.2) -> str:
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                time.sleep(pause)
                return r.read().decode("utf-8")
        except Exception as e:      # SSL resets, truncation, timeouts, throttling
            last = e
            time.sleep(2.0 * (i + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url}: {last}")


def iss_json(path: str, **params) -> dict:
    q = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{ISS}/{path}.json" + (f"?{q}" if q else "")
    for attempt in range(3):
        try:
            return json.loads(_get(url))
        except json.JSONDecodeError:        # truncated body — refetch
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"unparseable JSON after retries: {url}")


def _block(d: dict, name: str) -> pd.DataFrame:
    b = d[name]
    return pd.DataFrame(b["data"], columns=b["columns"])


# ------------------------------------------------------------------
# stock candles (daily), paginated by advancing `from`
# ------------------------------------------------------------------
def fetch_stock_candles(secid: str, frm: str = FETCH_FROM, till: str = FETCH_TILL,
                        board: str = "TQBR") -> pd.DataFrame:
    """Daily close + RUB turnover for one stock. Empty DF if nothing trades."""
    rows, cur = [], frm
    while True:
        d = iss_json(f"engines/stock/markets/shares/boards/{board}/"
                     f"securities/{secid}/candles",
                     **{"from": cur, "till": till, "interval": 24})
        df = _block(d, "candles")
        if df.empty:
            break
        df["date"] = pd.to_datetime(df["begin"]).dt.normalize()
        rows.append(df[["date", "close", "value"]])
        last = df["date"].iloc[-1]
        nxt = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if nxt > till or len(df) < 2:
            break
        cur = nxt
    if not rows:
        return pd.DataFrame(columns=["date", "close", "value"])
    out = pd.concat(rows).drop_duplicates("date").set_index("date").sort_index()
    return out


# ------------------------------------------------------------------
# index history (close), paginated via `start`
# ------------------------------------------------------------------
def fetch_index_close(secid: str, frm: str = FETCH_FROM, till: str = FETCH_TILL) -> pd.Series:
    rows, start = [], 0
    while True:
        d = iss_json(f"history/engines/stock/markets/index/securities/{secid}",
                     **{"from": frm, "till": till, "start": start,
                        "history.columns": "TRADEDATE,CLOSE"})
        df = _block(d, "history")
        if df.empty:
            break
        rows.append(df)
        start += len(df)
        if len(df) < 100:
            break
    if not rows:
        return pd.Series(dtype=float, name=secid)
    df = pd.concat(rows)
    s = pd.Series(df["CLOSE"].values, index=pd.to_datetime(df["TRADEDATE"]),
                  name=secid).dropna()
    return s[~s.index.duplicated()].sort_index()


# ------------------------------------------------------------------
# historical index composition (analytics endpoint)
# ------------------------------------------------------------------
def fetch_index_composition(index: str, date: str) -> pd.DataFrame:
    """Constituents + weights at (or near) `date`. Tries date, date+1, ... +7."""
    d0 = pd.Timestamp(date)
    for shift in range(8):
        dt = (d0 + pd.Timedelta(days=shift)).strftime("%Y-%m-%d")
        rows, start = [], 0
        while True:
            d = iss_json(f"statistics/engines/stock/markets/index/analytics/{index}",
                         date=dt, limit=100, start=start)
            df = _block(d, "analytics")
            if df.empty:
                break
            rows.append(df)
            start += len(df)
            if len(df) < 100:
                break
        if rows:
            out = pd.concat(rows)[["ticker", "weight", "tradedate"]]
            return out.reset_index(drop=True)
    return pd.DataFrame(columns=["ticker", "weight", "tradedate"])


# ------------------------------------------------------------------
# risk-free: 1Y point of the zero-coupon G-curve, monthly samples
# ------------------------------------------------------------------
def fetch_zcyc_1y(frm: str = "2014-06-01", till: str = FETCH_TILL) -> pd.Series:
    dates = pd.date_range(frm, till, freq="MS") + pd.Timedelta(days=14)
    vals = {}
    for d0 in dates:
        for shift in range(6):
            dt = (d0 + pd.Timedelta(days=shift)).strftime("%Y-%m-%d")
            try:
                d = iss_json("engines/stock/zcyc", date=dt)
            except RuntimeError:
                continue
            yy = _block(d, "yearyields")
            if not yy.empty:
                row = yy[yy["period"] == 1.0]
                if not row.empty:
                    vals[pd.Timestamp(dt)] = float(row["value"].iloc[0]) / 100.0
                    break
    return pd.Series(vals, name="ofz_1y").sort_index()


# ------------------------------------------------------------------
# dividends: ISS + dohod.ru merge
# ------------------------------------------------------------------
def fetch_dividends_iss(secid: str) -> pd.DataFrame:
    d = iss_json(f"securities/{secid}/dividends")
    df = _block(d, "dividends")
    if df.empty:
        return pd.DataFrame(columns=["close_date", "value", "currency", "src"])
    out = pd.DataFrame({
        "close_date": pd.to_datetime(df["registryclosedate"]),
        "value": df["value"].astype(float),
        "currency": df["currencyid"].fillna("RUB"),
        "src": "iss",
    })
    return out


_DOHOD_TABLE_RE = re.compile(r"<table[^>]*>.*?</table>", re.S)
_TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_TD_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def fetch_dividends_dohod(ticker: str) -> pd.DataFrame:
    """Parse dohod.ru per-payment table: (announce, registry close, year, value).
    Skips forecast rows. Returns empty DF on any failure (fall back to ISS)."""
    url = f"https://www.dohod.ru/ik/analytics/dividend/{ticker.lower()}"
    try:
        html = _get(url, retries=2, pause=0.3)
    except RuntimeError:
        return pd.DataFrame(columns=["close_date", "value", "currency", "src"])
    recs = []
    for t in _DOHOD_TABLE_RE.findall(html):
        if "Дата закрытия реестра" not in t:
            continue
        for tr in _TR_RE.findall(t):
            cells = [_TAG_RE.sub("", c).replace("&nbsp;", " ").strip()
                     for c in _TD_RE.findall(tr)]
            if len(cells) < 4 or "прогноз" in cells[1]:
                continue
            m = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", cells[1])
            if not m:
                continue
            try:
                val = float(cells[3].replace(",", ".").replace(" ", ""))
            except ValueError:
                continue
            if val <= 0:
                continue
            recs.append({"close_date": pd.Timestamp(f"{m.group(3)}-{m.group(2)}-{m.group(1)}"),
                         "value": val, "currency": "RUB?", "src": "dohod"})
        break
    return pd.DataFrame(recs, columns=["close_date", "value", "currency", "src"])


def merge_dividends(iss: pd.DataFrame, dohod: pd.DataFrame,
                    tol_days: int = 12) -> pd.DataFrame:
    """Union of the two per-payment lists, deduped by close-date proximity.
    Where both have a record, the ISS one wins (official values + currency).
    dohod-only records: currency assumed RUB unless the ISS overlap shows the
    issuer pays USD (then flagged for conversion)."""
    iss = iss.copy()
    out = [iss]
    usd_payer = (iss["currency"] == "USD").any() if len(iss) else False
    for _, r in dohod.iterrows():
        if len(iss) and (abs((iss["close_date"] - r["close_date"]).dt.days) <= tol_days).any():
            continue                      # ISS already covers this payment
        r = r.copy()
        r["currency"] = "USD" if usd_payer else "RUB"
        out.append(r.to_frame().T)
    df = pd.concat(out, ignore_index=True)
    df["value"] = df["value"].astype(float)
    df["close_date"] = pd.to_datetime(df["close_date"])
    return df.sort_values("close_date").reset_index(drop=True)


# ------------------------------------------------------------------
# share-basis factors: ISS *candles* are retroactively split-adjusted
# (GMKN/TRNFP 100:1 2024, VTBR 1:5000 2024, PLZL 10:1 2025 ...), while
# dividends are as-paid per OLD share. factor = candle/as-traded price;
# dividends get multiplied by it. Two reference dates detect (unhandled)
# in-window splits.
# ------------------------------------------------------------------
def _as_traded_close(secid: str, around: str) -> float | None:
    d0 = pd.Timestamp(around)
    d = iss_json(f"history/engines/stock/markets/shares/boards/TQBR/"
                 f"securities/{secid}",
                 **{"from": around,
                    "till": (d0 + pd.Timedelta(days=8)).strftime("%Y-%m-%d"),
                    "history.columns": "TRADEDATE,CLOSE"})
    df = _block(d, "history")
    df = df.dropna(subset=["CLOSE"])
    return float(df["CLOSE"].iloc[0]) if len(df) else None


def fetch_basis_factors(close: pd.DataFrame,
                        ref_dates: tuple[str, str] = ("2014-06-02", "2020-06-01")
                        ) -> dict[str, float]:
    """Per-stock candle/as-traded price ratio (constant across the window if
    no in-window split; verified on two dates)."""
    cache = DATA_DIR / "basis_factors.json"
    if cache.exists():
        return json.loads(cache.read_text())
    out = {}
    for tkr in close.columns:
        facs = []
        for rd in ref_dates:
            asof = close[tkr].loc[rd:].dropna()
            if asof.empty:
                continue
            traded = _as_traded_close(tkr, rd)
            if traded and traded > 0:
                facs.append(float(asof.iloc[0]) / traded)
        if not facs:
            out[tkr] = 1.0
            continue
        if len(facs) == 2 and abs(facs[0] / facs[1] - 1.0) > 0.01:
            print(f"  [warn] {tkr}: basis factor differs across window "
                  f"({facs[0]:.4g} vs {facs[1]:.4g}) — IN-WINDOW SPLIT, "
                  f"using per-date handling needed!")
        f = facs[-1]
        out[tkr] = 1.0 if abs(f - 1.0) < 0.01 else f
        if out[tkr] != 1.0:
            print(f"   basis factor {tkr}: {out[tkr]:.6g}")
    cache.write_text(json.dumps(out, indent=2))
    return out


# ------------------------------------------------------------------
# adjusted close
# ------------------------------------------------------------------
def last_cum_day(close_date: pd.Timestamp, calendar: pd.DatetimeIndex) -> pd.Timestamp | None:
    """Last trading day X whose T+2 settlement lands on/before registry close.
    Settlement of a buy on trading day X = 2nd trading day after X."""
    pos = calendar.searchsorted(close_date, side="right") - 1
    # walk back until settle(X) <= close_date
    while pos >= 0:
        settle_idx = pos + 2
        if settle_idx < len(calendar) and calendar[settle_idx] > close_date:
            pos -= 1
            continue
        return calendar[pos]
    return None


def build_adjusted_close(close: pd.DataFrame, dividends: dict[str, pd.DataFrame],
                         usdrub: pd.Series,
                         basis: dict[str, float] | None = None) -> pd.DataFrame:
    """Back-adjust closes: on each ex-date multiply all prices <= cum-day by
    (P_cum - div)/P_cum. USD dividends converted at the cum-day USDRUB rate.
    `basis` rescales as-paid dividend values to the candles' share basis."""
    adj = close.copy().astype(float)
    cal = close.index
    basis = basis or {}
    warns = []
    for tkr, divs in dividends.items():
        if tkr not in adj.columns or divs.empty:
            continue
        px = close[tkr]
        for _, r in divs.iterrows():
            if r["close_date"] < cal[0] or r["close_date"] > cal[-1]:
                continue
            cum = last_cum_day(r["close_date"], cal)
            if cum is None:
                continue
            p_cum = px.loc[:cum].dropna()
            if p_cum.empty:
                continue
            p = float(p_cum.iloc[-1])
            val = float(r["value"]) * float(basis.get(tkr, 1.0))
            if r.get("currency") == "USD":
                fx = usdrub.reindex([cum], method="ffill").iloc[0]
                val *= float(fx)
            if not np.isfinite(p) or val >= p:
                warns.append(f"{tkr}: div {val:.2f} >= price {p:.2f} on {cum.date()}, skipped")
                continue
            factor = (p - val) / p
            mask = adj.index <= p_cum.index[-1]
            adj.loc[mask, tkr] = adj.loc[mask, tkr] * factor
    for w in warns:
        print("  [warn]", w)
    return adj


# ------------------------------------------------------------------
# USDRUB (for USD dividends / optional FX factor)
# ------------------------------------------------------------------
def fetch_usdrub(frm: str = FETCH_FROM, till: str = FETCH_TILL) -> pd.Series:
    rows, cur = [], frm
    while True:
        d = iss_json("engines/currency/markets/selt/boards/CETS/"
                     "securities/USD000UTSTOM/candles",
                     **{"from": cur, "till": till, "interval": 24})
        df = _block(d, "candles")
        if df.empty:
            break
        df["date"] = pd.to_datetime(df["begin"]).dt.normalize()
        rows.append(df[["date", "close"]])
        nxt = (df["date"].iloc[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if nxt > till or len(df) < 2:
            break
        cur = nxt
    df = pd.concat(rows).drop_duplicates("date").set_index("date").sort_index()
    return df["close"].rename("USDRUB")


# ------------------------------------------------------------------
# universe
# ------------------------------------------------------------------
def build_universe_candidates() -> pd.DataFrame:
    """Average MOEXBMI weight per ticker across yearly snapshots 2015-2020."""
    snaps = {}
    for y in range(2015, 2021):
        comp = fetch_index_composition("MOEXBMI", f"{y}-01-10")
        if comp.empty:
            print(f"  [warn] no MOEXBMI composition for {y}")
            continue
        snaps[y] = comp.set_index("ticker")["weight"]
        print(f"  MOEXBMI {y}: {len(comp)} names (as of {comp['tradedate'].iloc[0]})")
    W = pd.DataFrame(snaps)                       # tickers x years
    score = W.fillna(0.0).mean(axis=1)
    out = pd.DataFrame({"avg_weight": score, "years_present": W.notna().sum(axis=1)})
    return out.sort_values("avg_weight", ascending=False)


# ------------------------------------------------------------------
# orchestrator
# ------------------------------------------------------------------
def fetch_all(n_universe: int = 60) -> None:
    print("== 1. universe candidates (MOEXBMI yearly snapshots)")
    cand_file = DATA_DIR / "universe_candidates.csv"
    if cand_file.exists():
        cand = pd.read_csv(cand_file, index_col=0)
        print("   (cached)")
    else:
        cand = build_universe_candidates()
        cand.to_csv(cand_file)
    print(f"   {len(cand)} candidate tickers")

    print("== 2. stock candles (top candidates, until %d pass coverage)" % n_universe)
    cache_close = DATA_DIR / "_cache_close.parquet"
    cache_turn = DATA_DIR / "_cache_turn.parquet"
    if cache_close.exists():
        close = pd.read_parquet(cache_close)
        turn = pd.read_parquet(cache_turn)
        universe = list(close.columns)
        dropped = []
        print(f"   (cached: {len(universe)} tickers)")
    else:
        closes, turnover, kept, dropped = {}, {}, [], []
        for tkr in cand.index:
            if len(kept) >= n_universe:
                break
            try:
                df = fetch_stock_candles(tkr)
            except RuntimeError as e:
                print(f"  [drop] {tkr}: fetch failed ({e})")
                dropped.append((tkr, "fetch_failed"))
                continue
            if df.empty:
                dropped.append((tkr, "no_data"))
                continue
            closes[tkr] = df["close"]
            turnover[tkr] = df["value"]
            kept.append(tkr)
        close = pd.DataFrame(closes).sort_index()
        # coverage vs the union calendar over the filter window
        win = close.loc[COVERAGE_FROM:]
        cov = win.notna().mean()
        bad = cov[cov < COVERAGE_MIN].index.tolist()
        for t in bad:
            print(f"  [drop] {t}: coverage {cov[t]:.1%} < {COVERAGE_MIN:.0%}")
            dropped.append((t, f"coverage {cov[t]:.2f}"))
        close = close.drop(columns=bad)
        turn = pd.DataFrame(turnover).sort_index().drop(columns=bad)
        universe = list(close.columns)
        print(f"   kept {len(universe)} tickers")

        # top-up: if drops pushed us below target, walk further down the list
        if len(universe) < n_universe:
            for tkr in cand.index:
                if len(universe) >= n_universe or tkr in universe:
                    continue
                if tkr in dict(dropped):
                    continue
                try:
                    df = fetch_stock_candles(tkr)
                except RuntimeError:
                    continue
                if df.empty:
                    continue
                s = df["close"]
                covr = s.reindex(close.index).loc[COVERAGE_FROM:].notna().mean()
                if covr >= COVERAGE_MIN:
                    close[tkr] = s
                    turn[tkr] = df["value"]
                    universe.append(tkr)
                    print(f"   [top-up] {tkr} (coverage {covr:.1%})")
        universe = sorted(universe)
        close = close[universe]
        turn = turn[universe]
        close.to_parquet(cache_close)
        turn.to_parquet(cache_turn)

    print("== 3. indices")
    idx_file = DATA_DIR / "indices.parquet"
    if idx_file.exists():
        indices = pd.read_parquet(idx_file)
        print(f"   (cached: {indices.shape})")
    else:
        idx = {}
        for secid in ALL_INDICES:
            s = fetch_index_close(secid)
            idx[secid] = s
            print(f"   {secid}: {len(s)} rows [{s.index.min().date() if len(s) else '-'} .. "
                  f"{s.index.max().date() if len(s) else '-'}]")
        indices = pd.DataFrame(idx).sort_index()
        indices.to_parquet(idx_file)

    print("== 4. USDRUB")
    fx_file = DATA_DIR / "usdrub.parquet"
    if fx_file.exists():
        usdrub = pd.read_parquet(fx_file)["USDRUB"]
        print("   (cached)")
    else:
        usdrub = fetch_usdrub()
        usdrub.to_frame().to_parquet(fx_file)
        print(f"   {len(usdrub)} rows")

    print("== 5. dividends (ISS + dohod)")
    div_file = DATA_DIR / "dividends.csv"
    if div_file.exists():
        div_df = pd.read_csv(div_file, parse_dates=["close_date"])
        all_divs = {t: g.drop(columns="ticker")
                    for t, g in div_df.groupby("ticker")}
        for t in universe:
            all_divs.setdefault(t, pd.DataFrame(
                columns=["close_date", "value", "currency", "src"]))
        print(f"   (cached: {len(div_df)} records)")
    else:
        all_divs, div_rows = {}, []
        for tkr in universe:
            iss_d = fetch_dividends_iss(tkr)
            doh_d = fetch_dividends_dohod(tkr)
            merged = merge_dividends(iss_d, doh_d)
            merged = merged[(merged["close_date"] >= pd.Timestamp(FETCH_FROM)) &
                            (merged["close_date"] <= pd.Timestamp(FETCH_TILL))]
            all_divs[tkr] = merged
            for _, r in merged.iterrows():
                div_rows.append({"ticker": tkr, **r.to_dict()})
            print(f"   {tkr}: iss={len(iss_d)} dohod={len(doh_d)} -> "
                  f"{len(merged)} in window")
        div_df = pd.DataFrame(div_rows)
        div_df.to_csv(div_file, index=False)

    print("== 6. adjusted close (with share-basis rescaling of dividends)")
    basis = fetch_basis_factors(close)
    adj = build_adjusted_close(close, all_divs, usdrub, basis=basis)
    # sanity: big overnight moves in adjusted series
    r = adj.pct_change()
    big = (r.abs() > 0.40)
    for tkr in universe:
        for dt in adj.index[big[tkr].values]:
            print(f"  [check] {tkr} {dt.date()}: adjusted overnight move "
                  f"{r.loc[dt, tkr]:+.1%} (split/rename/data issue?)")

    print("== 7. risk-free (zcyc 1Y, monthly)")
    rf_file = DATA_DIR / "ofz_1y.parquet"
    if rf_file.exists():
        rf = pd.read_parquet(rf_file)["ofz_1y"]
        print("   (cached)")
    else:
        rf = fetch_zcyc_1y()
        rf.to_frame().to_parquet(rf_file)
    rf_periods = {
        "2015-2016": float(rf.loc["2015-01-01":"2016-12-31"].mean()),
        "2017-2019": float(rf.loc["2017-01-01":"2019-12-31"].mean()),
        "2020": float(rf.loc["2020-01-01":"2020-12-31"].mean()),
    }
    print("   period averages:", {k: f"{v:.2%}" for k, v in rf_periods.items()})

    print("== 8. save")
    close.to_parquet(DATA_DIR / "prices_close.parquet")
    adj.to_parquet(DATA_DIR / "prices_adj.parquet")
    turn.to_parquet(DATA_DIR / "turnover.parquet")
    (DATA_DIR / "risk_free.json").write_text(json.dumps(rf_periods, indent=2))
    (DATA_DIR / "universe.json").write_text(json.dumps({
        "tickers": universe, "n": len(universe),
        "dropped": dropped, "rule": "top MOEXBMI avg weight 2015-2020, "
        f"coverage>={COVERAGE_MIN:.0%} from {COVERAGE_FROM}",
    }, indent=2, ensure_ascii=False))
    print(f"   done: {len(universe)} stocks, {close.index.min().date()} .. "
          f"{close.index.max().date()}, files in {DATA_DIR}/")


def load_all() -> dict:
    """Load everything the backtest needs from data/."""
    out = {
        "close": pd.read_parquet(DATA_DIR / "prices_close.parquet"),
        "adj": pd.read_parquet(DATA_DIR / "prices_adj.parquet"),
        "turnover": pd.read_parquet(DATA_DIR / "turnover.parquet"),
        "indices": pd.read_parquet(DATA_DIR / "indices.parquet"),
        "usdrub": pd.read_parquet(DATA_DIR / "usdrub.parquet")["USDRUB"],
        "rf": json.loads((DATA_DIR / "risk_free.json").read_text()),
        "universe": json.loads((DATA_DIR / "universe.json").read_text())["tickers"],
    }
    return out


if __name__ == "__main__":
    fetch_all()
