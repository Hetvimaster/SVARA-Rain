"""Verification metrics by model, lead day, regime and IMD threshold (RMSE, bias, POD, FAR, CSI, ETS, FSS).

python -m svara.metrics              # score the raw GEFS ensemble mean
python -m svara.metrics --selftest   # check the metric code on tiny known cases
"""
import argparse

import numpy as np
import pandas as pd
import xarray as xr

from . import config as C
from .align import pair, valid_offset

THRESHOLDS = (2.5, 15.6, 64.5, 115.6)  # mm/day: rain, moderate, heavy, very heavy (IMD categories)
FSS_WINDOWS = (1, 5, 9)                # square neighbourhood in grid cells (0.25 deg each)
REGIMES = ("active", "break", "normal")
LABELS = C.DATA / "labels" / "day_regime.csv"
PLAN = C.DATA / "labels" / "init_plan.txt"
OUT = C.DATA / "reports" / "metrics.csv"


def _div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else np.nan


def _box_sum(a: np.ndarray, k: int) -> np.ndarray:
    """Sum of a over a k x k window centred on each cell (zeros outside the grid). a is (T, H, W)."""
    r = k // 2
    c = np.pad(a, ((0, 0), (r + 1, r), (r + 1, r))).cumsum(1).cumsum(2)
    return c[:, k:, k:] - c[:, :-k, k:] - c[:, k:, :-k] + c[:, :-k, :-k]


def score_group(f: np.ndarray, o: np.ndarray, w: np.ndarray,
                thresholds=THRESHOLDS, windows=FSS_WINDOWS) -> list[dict]:
    """Scores for one set of days. f, o: (T, H, W) mm/day, NaN = missing. w: (T,) day weights."""
    m = np.isfinite(f) & np.isfinite(o)
    f, o = np.where(m, f, 0.0), np.where(m, o, 0.0)
    W = w[:, None, None] * m
    n = W.sum()
    if n == 0:
        return []
    base = {"n_days": int(m.any((1, 2)).sum()),
            "bias_mm": float((W * (f - o)).sum() / n),
            "rmse": float(np.sqrt((W * (f - o) ** 2).sum() / n))}
    mf = m.astype(np.float64)
    box_m = {k: _box_sum(mf, k) for k in windows}
    rows = []
    for thr in thresholds:
        fe, oe = (f >= thr) & m, (o >= thr) & m
        h = (W * (fe & oe)).sum()
        fa = (W * (fe & ~oe)).sum()
        mi = (W * (~fe & oe)).sum()
        hr = (h + mi) * (h + fa) / n  # hits expected by chance
        row = dict(base, threshold=thr,
                   obs_event_cells=int(oe.sum()), obs_event_days=int(oe.any((1, 2)).sum()),
                   pod=_div(h, h + mi), far=_div(fa, h + fa), csi=_div(h, h + mi + fa),
                   freq_bias=_div(h + fa, h + mi), ets=_div(h - hr, h + mi + fa - hr))
        for k in windows:
            pf = np.divide(_box_sum(fe * 1.0, k), box_m[k], out=np.zeros_like(f), where=box_m[k] > 0)
            po = np.divide(_box_sum(oe * 1.0, k), box_m[k], out=np.zeros_like(f), where=box_m[k] > 0)
            row[f"fss_{k}"] = 1 - _div((W * (pf - po) ** 2).sum(), (W * (pf ** 2 + po ** 2)).sum())
        rows.append(row)
    return rows


def evaluate(forecasts: dict[str, xr.DataArray], obs: xr.DataArray, labels: pd.Series,
             weights: pd.Series | None = None, leads=range(1, C.MAX_LEAD + 1)) -> pd.DataFrame:
    """forecasts: name -> (init_time, lead_day, lat, lon) mm/day. labels: regime by valid date.
    weights: day weight by init_time (default 1). Regime = regime of the day being forecast."""
    rows = []
    for name, fc in forecasts.items():
        fc = fc.load()
        init = pd.DatetimeIndex(fc.init_time.values)
        w = np.ones(len(init)) if weights is None else weights.reindex(init).fillna(1.0).to_numpy()
        for lead in leads:
            f, o = pair(fc, obs, lead)
            f, o = f.values.astype(np.float64), o.values.astype(np.float64)
            reg = labels.reindex(init + valid_offset(lead)).to_numpy()
            groups = {"all": np.ones(len(init), bool)} | {r: reg == r for r in REGIMES}
            for g, sel in groups.items():
                if sel.any():
                    for row in score_group(f[sel], o[sel], w[sel]):
                        rows.append({"model": name, "lead": lead, "regime": g, **row})
    return pd.DataFrame(rows)


def selection_weights(init: pd.DatetimeIndex, labels: pd.Series, normal_every: int) -> pd.Series:
    """Inverse sampling weight. plan_dates keeps every active/break day but 1 in N normal days,
    chosen by the regime of the lead-1 valid day, so a kept 'normal' init stands for N days."""
    reg = labels.reindex(init + valid_offset(1)).to_numpy()
    return pd.Series(np.where(reg == "normal", float(normal_every), 1.0), index=init)


def selftest() -> None:
    rng = np.random.default_rng(0)
    o = rng.gamma(0.5, 12.0, size=(6, 30, 30))
    r = score_group(o, o, np.ones(6), thresholds=(5.0,))[0]
    assert r["rmse"] == 0 and abs(r["ets"] - 1) < 1e-12 and abs(r["csi"] - 1) < 1e-12
    assert all(abs(r[f"fss_{k}"] - 1) < 1e-12 for k in FSS_WINDOWS), r
    # 2x2 case: hits=1, false alarms=1, misses=1, correct negatives=1
    f = np.array([[[10.0, 10.0, 0.0, 0.0]]])
    ob = np.array([[[10.0, 0.0, 10.0, 0.0]]])
    r = score_group(f, ob, np.ones(1), thresholds=(5.0,), windows=(1,))[0]
    exp = dict(pod=0.5, far=0.5, csi=1 / 3, freq_bias=1.0, ets=0.0, rmse=np.sqrt(50.0), bias_mm=0.0)
    for k, v in exp.items():
        assert abs(r[k] - v) < 1e-9, (k, r[k], v)
    assert abs(r["fss_1"] - 0.5) < 1e-9, r["fss_1"]  # sum sq diff 2 / sum sq 4
    # displaced event: 3 cells apart, 9x9 window -> neighbourhoods overlap in 54 of 81 cells -> FSS = 2/3
    f = np.zeros((1, 40, 40)); ob = np.zeros((1, 40, 40)); f[0, 20, 20] = 50; ob[0, 20, 23] = 50
    r = score_group(f, ob, np.ones(1), thresholds=(5.0,), windows=(1, 9))[0]
    assert r["fss_1"] == 0 and abs(r["fss_9"] - 2 / 3) < 1e-9, (r["fss_1"], r["fss_9"])
    # day weights: weight 3 on a day equals repeating the day 3 times
    a, b = rng.gamma(0.5, 12, (1, 8, 8)), rng.gamma(0.5, 12, (1, 8, 8))
    fa, fb = rng.gamma(0.5, 12, (1, 8, 8)), rng.gamma(0.5, 12, (1, 8, 8))
    w3 = score_group(np.r_[fa, fb], np.r_[a, b], np.array([3.0, 1.0]), (5.0,), (3,))[0]
    rep = score_group(np.r_[fa, fa, fa, fb], np.r_[a, a, a, b], np.ones(4), (5.0,), (3,))[0]
    for k in ("rmse", "bias_mm", "pod", "far", "csi", "ets", "fss_3"):
        assert abs(w3[k] - rep[k]) < 1e-9, k
    print("selftest passed")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--normal-every", type=int, default=4, help="must match plan_dates")
    p.add_argument("--no-plan", action="store_true", help="score every init in the zarr, unweighted")
    a = p.parse_args()
    if a.selftest:
        return selftest()

    labels = pd.read_csv(LABELS, parse_dates=["date"]).set_index("date").regime
    obs = xr.open_zarr(C.IMD_ZARR).obs
    raw = xr.open_zarr(C.GEFS_ZARR).apcp
    weights = None
    if not a.no_plan:
        plan = pd.DatetimeIndex(pd.read_csv(PLAN, header=None, parse_dates=[0])[0])
        raw = raw.sel(init_time=raw.init_time.to_index().intersection(plan))
        weights = selection_weights(pd.DatetimeIndex(raw.init_time.values), labels, a.normal_every)
        print(f"{raw.sizes['init_time']} planned inits scored; normal-day weight {a.normal_every}")
    else:
        print(f"{raw.sizes['init_time']} inits scored, unweighted (includes any non-planned dates)")

    res = evaluate({"raw_ens_mean": raw.mean("member")}, obs, labels, weights)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    show = res[res.threshold.isin((15.6, 64.5))]
    print(show[["lead", "regime", "threshold", "n_days", "obs_event_days", "bias_mm", "rmse",
                "pod", "far", "csi", "ets", "freq_bias", "fss_5"]].round(2).to_string(index=False))
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()