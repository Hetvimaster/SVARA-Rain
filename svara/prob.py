"""Calibrated heavy / very heavy rain probabilities, with and without forecast-regime features.

LGBM classifier per threshold (train years, early stop on val) -> isotonic calibration on val -> test years.
Scored by Brier skill score vs climatology (block bootstrap over days), plus reliability table.

python -m svara.prob --train-years 2015 2016 --val-years 2017 --test-years 2018 2019
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.isotonic import IsotonicRegression

from . import config as C
from .align import valid_offset
from .compare import MIN_DAYS
from .fc_regimes import OUT_LABELS as FC_LABELS
from .gate import _boot_idx
from .lgbm_correct import LEADS, features
from .metrics import LABELS, PLAN, REGIMES, selection_weights
from .qmap import paired

THRS = (64.5, 115.6)
BINS = np.array([0, .02, .05, .1, .2, .3, .5, .7, 1.0001])
OUT_SCORES = C.DATA / "reports" / "prob_scores.csv"
OUT_REL = C.DATA / "reports" / "prob_reliability.csv"
OUT_PROB = C.DATA / "products" / "prob_test.zarr"


def rows(raw, obs, ens, inits, land, zfs, w, with_regime, stride):
    sub = np.zeros_like(land)
    sub[::stride, ::stride] = True
    X, Y, W = [], [], []
    for lead in LEADS:
        F, names = features(raw, inits, lead, zfs[lead], with_regime)
        o = paired(ens.sel(init_time=inits), obs, lead)[1]
        ww = np.broadcast_to(w.reindex(inits).to_numpy()[:, None, None], o.shape)
        ok = np.isfinite(o) & (land & sub)[None]
        X.append(F[ok]); Y.append(o[ok]); W.append(ww[ok])
    return np.concatenate(X), np.concatenate(Y), np.concatenate(W), names


def fit_prob(X, y, w, Xv, yv, wv, thr, seed):
    b, bv = y >= thr, yv >= thr
    print(f"  thr {thr}: event rows train {int(b.sum())}, val {int(bv.sum())}")
    clf = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=200,
                             subsample=0.7, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                             random_state=seed, verbose=-1)
    clf.fit(X, b, sample_weight=w, eval_set=[(Xv, bv)], eval_sample_weight=[wv],
            callbacks=[lgb.early_stopping(30, verbose=False)])
    iso = None
    if bv.sum() >= 100:
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(
            clf.predict_proba(Xv)[:, 1], bv.astype(float), sample_weight=wv)
    else:
        print(f"  WARNING: only {int(bv.sum())} val events at {thr}, skipping isotonic calibration")
    return clf, iso, float(np.average(b, weights=w))


def predict_prob(clf, iso, F, land):
    T, K = F.shape[0], F.shape[-1]
    p = np.full(F.shape[:3], np.nan, np.float32)
    q = clf.predict_proba(F[:, land].reshape(-1, K))[:, 1]
    p[:, land] = (iso.predict(q) if iso is not None else q).reshape(T, -1)
    return p


def day_stats(p, o, land, thr, base):
    y = (o >= thr).astype(np.float32)
    ok = np.isfinite(o) & np.isfinite(p) & land[None]
    B = np.where(ok, (p - y) ** 2, 0).sum((1, 2))
    R = np.where(ok, (base - y) ** 2, 0).sum((1, 2))
    return B, R, int((ok & (y > 0)).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-years", type=int, nargs="+", default=[2015, 2016])
    ap.add_argument("--val-years", type=int, nargs="+", default=[2017])
    ap.add_argument("--test-years", type=int, nargs="+", default=[2018, 2019])
    ap.add_argument("--normal-every", type=int, default=4)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--block", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    global OUT_SCORES, OUT_REL, OUT_PROB
    sfx = f"_{a.tag}" if a.tag else ""
    OUT_SCORES = C.DATA / "reports" / f"prob_scores{sfx}.csv"
    OUT_REL = C.DATA / "reports" / f"prob_reliability{sfx}.csv"
    OUT_PROB = C.DATA / "products" / f"prob_test{sfx}.zarr"
    obs_lab = pd.read_csv(LABELS, parse_dates=["date"]).set_index("date").regime
    fcl = pd.read_csv(FC_LABELS, parse_dates=["init_time", "date"])
    zfs = {lead: g.set_index("init_time")[["z_f", "fc_regime"]] for lead, g in fcl.groupby("lead")}
    obs = xr.open_zarr(C.IMD_ZARR).obs
    land = obs.notnull().any("time").values
    raw = xr.open_zarr(C.GEFS_ZARR).apcp
    plan = pd.DatetimeIndex(pd.read_csv(PLAN, header=None, parse_dates=[0])[0])
    inits = raw.init_time.to_index().intersection(plan)
    w = selection_weights(inits, obs_lab, a.normal_every)
    tr, va, te = (inits[inits.year.isin(y)] for y in (a.train_years, a.val_years, a.test_years))
    ens = raw.mean("member").sel(init_time=inits).load()
    wte = w.reindex(te).to_numpy()
    obs_te = {lead: paired(ens.sel(init_time=te), obs, lead)[1] for lead in LEADS}

    P, base = {}, {}
    for name, with_regime in (("lgbm", False), ("lgbm_regime", True)):
        print(f"\n--- {name} ---")
        X, y, wt, _ = rows(raw, obs, ens, tr, land, zfs, w, with_regime, a.stride)
        Xv, yv, wv, _ = rows(raw, obs, ens, va, land, zfs, w, with_regime, a.stride)
        for thr in THRS:
            clf, iso, base[thr] = fit_prob(X, y, wt, Xv, yv, wv, thr, a.seed)
            for lead in LEADS:
                F, _ = features(raw, te, lead, zfs[lead], with_regime)
                P.setdefault((name, thr), {})[lead] = predict_prob(clf, iso, F, land)
        for lead in LEADS:  # a higher threshold must never be more likely
            P[(name, THRS[1])][lead] = np.minimum(P[(name, THRS[1])][lead], P[(name, THRS[0])][lead])

    # reference: raw ensemble exceedance fraction (64.5 only; 5 members -> coarse)
    for lead in LEADS:
        F, names = features(raw, te, lead, zfs[lead], False)
        fr = F[..., names.index("frac_64.5")].astype(np.float32)
        fr[:, ~land] = np.nan
        P.setdefault(("raw_frac", 64.5), {})[lead] = fr

    rng = np.random.default_rng(a.seed)
    out, rel, dayrows = [], {}, []
    for thr in THRS:
        models = [m for m in ("raw_frac", "lgbm", "lgbm_regime") if (m, thr) in P]
        for lead in LEADS:
            o = obs_te[lead]
            st = {m: day_stats(P[(m, thr)][lead], o, land, thr, base[thr]) for m in models}
            reg = obs_lab.reindex(te + valid_offset(lead)).to_numpy()
            for m in models:
                B_, R_, _ = st[m]
                for i, t in enumerate(te):
                    dayrows.append((t, lead, thr, m, wte[i], B_[i], R_[i], reg[i]))
            groups = {"all": np.ones(len(te), bool)} | {r: reg == r for r in REGIMES}
            for g, sel in groups.items():
                if sel.sum() < MIN_DAYS:
                    continue
                ws = wte[sel]
                if (ws * st[models[0]][1][sel]).sum() <= 0:
                    continue
                idx = _boot_idx(int(sel.sum()), a.block, a.n_boot, rng)
                bss = {}
                for m in models:
                    B, R, ne = st[m]
                    num, den = ws * B[sel], ws * R[sel]
                    bss[m] = (1 - num.sum() / den.sum(), 1 - num[idx].sum(1) / den[idx].sum(1), ne)
                    lo, hi = np.nanpercentile(bss[m][1], [2.5, 97.5])
                    out.append(dict(lead=lead, regime=g, threshold=thr, model=m, n_days=int(sel.sum()),
                                    n_event_cells=ne, bss=bss[m][0], lo=lo, hi=hi))
                if "lgbm" in bss:
                    d = bss["lgbm_regime"][1] - bss["lgbm"][1]
                    lo, hi = np.nanpercentile(d, [2.5, 97.5])
                    out.append(dict(lead=lead, regime=g, threshold=thr, model="regime_minus_noregime",
                                    n_days=int(sel.sum()), bss=bss["lgbm_regime"][0] - bss["lgbm"][0],
                                    lo=lo, hi=hi, sig=bool(lo > 0 or hi < 0)))
            # reliability, pooled over days, regime = all
            ww = np.broadcast_to(wte[:, None, None], o.shape)
            y = (o >= thr).astype(float)
            for m in models:
                p = P[(m, thr)][lead]
                ok = np.isfinite(p) & np.isfinite(o) & land[None]
                k = np.digitize(p[ok], BINS) - 1
                acc = rel.setdefault((m, thr), np.zeros((3, len(BINS) - 1)))
                for j, v in enumerate((ww[ok] * p[ok], ww[ok] * y[ok], ww[ok])):
                    acc[j] += np.bincount(k, weights=v, minlength=len(BINS) - 1)[:len(BINS) - 1]

    res = pd.DataFrame(out)
    OUT_SCORES.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_SCORES, index=False)
    rl = pd.concat([pd.DataFrame({"model": m, "threshold": t, "bin_lo": BINS[:-1],
                                  "mean_fc": acc[0] / np.maximum(acc[2], 1e-12),
                                  "obs_freq": acc[1] / np.maximum(acc[2], 1e-12), "weight": acc[2]})
                    for (m, t), acc in rel.items()])
    rl.to_csv(OUT_REL, index=False)

    pd.set_option("display.width", 200)
    a_ = res[res.regime == "all"]
    print("\n=== Brier skill score vs climatology, regime=all ===")
    print(a_[a_.model != "regime_minus_noregime"].pivot_table(index=["threshold", "lead"], columns="model",
                                                              values="bss").round(3).to_string())
    print("\n=== lgbm_regime minus lgbm (BSS), with 95% CI ===")
    print(a_[a_.model == "regime_minus_noregime"][["threshold", "lead", "bss", "lo", "hi", "sig"]].round(4).to_string(index=False))
    ev = a_[a_.model == "lgbm"].groupby("threshold").n_event_cells.sum()
    print(f"\nevent cells in test (summed over leads): {ev.to_dict()}")
    print("\nreliability, lgbm_regime:\n", rl[rl.model == "lgbm_regime"].round(3).to_string(index=False))
    pd.DataFrame(dayrows, columns=["init", "lead", "threshold", "model", "w", "B", "R", "regime"]).to_csv(
        C.DATA / "reports" / f"prob_days{sfx}.csv", index=False)
    ds = xr.Dataset({f"p{int(t)}": (("init_time", "lead_day", "lat", "lon"),
                                    np.stack([P[("lgbm_regime", t)][l] for l in LEADS], 1))
                     for t in THRS},
                    coords={"init_time": te, "lead_day": list(LEADS), "lat": raw.lat.values, "lon": raw.lon.values})
    OUT_PROB.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(OUT_PROB, mode="w")
    print(f"\nwritten: {OUT_SCORES}\nwritten: {OUT_REL}\nwritten: {OUT_PROB}")


if __name__ == "__main__":
    main()