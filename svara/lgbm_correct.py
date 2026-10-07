"""LightGBM post-processing of the GEFS ensemble, with and without forecast-regime features.

Rungs scored (same metrics and bootstrap as the quantile-mapping stage):
  raw_ens_mean   reference
  qm_global      global quantile mapping, fitted on train+val inits (same data the LGBM pipeline sees)
  lgbm           LightGBM, no regime information
  lgbm_regime    LightGBM + forecast-only regime features (CMZ forecast anomaly z_f, active/break code)

Pipeline per variant:  fit LGBM on TRAIN years (early stopping on VAL)  ->  fit a quantile map on the VAL
predictions (undoes the regression-to-the-mean that shrinks heavy rain)  ->  apply to TEST years.
Splits are by whole years. Heavy-rain cells get extra training weight.

python -m svara.lgbm_correct --train-years 2015 2016 --val-years 2017 --test-years 2018 2019
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import maximum_filter, uniform_filter

from . import config as C
from .align import valid_offset
from .compare import MIN_DAYS, THRESHOLDS, counts, ets_fb
from .fc_regimes import OUT_LABELS as FC_LABELS
from .gate import _boot_idx
from .metrics import LABELS, PLAN, REGIMES, evaluate, selection_weights
from .qmap import QMap, apply_models, fit_models, paired

OUT_METRICS = C.DATA / "reports" / "metrics_lgbm.csv"
OUT_COMPARE = C.DATA / "reports" / "compare_lgbm.csv"
LEADS = range(1, C.MAX_LEAD + 1)
PAIRS = (("lgbm_regime", "lgbm"), ("lgbm", "qm_global"), ("lgbm_regime", "qm_global"))
REG_CODE = {"active": 1.0, "break": -1.0, "normal": 0.0}


def features(raw, inits, lead, zf: pd.DataFrame, with_regime: bool):
    """Feature cube (T, H, W, K) for one lead. Forecast fields only, so it works at prediction time."""
    m = raw.sel(init_time=inits, lead_day=lead).transpose("init_time", "member", "lat", "lon")
    m = m.values.astype(np.float32)
    mean = m.mean(1)
    T, H, W = mean.shape
    cols = {"ens_mean": mean, "ens_std": m.std(1), "ens_max": m.max(1),
            "frac_2.5": (m >= 2.5).mean(1), "frac_15.6": (m >= 15.6).mean(1), "frac_64.5": (m >= 64.5).mean(1),
            "nbr_mean5": uniform_filter(mean, size=(1, 5, 5), mode="nearest"),
            "nbr_max5": maximum_filter(mean, size=(1, 5, 5), mode="nearest"),
            "nbr_mean11": uniform_filter(mean, size=(1, 11, 11), mode="nearest")}
    lat = np.broadcast_to(raw.lat.values.astype(np.float32)[None, :, None], (T, H, W))
    lon = np.broadcast_to(raw.lon.values.astype(np.float32)[None, None, :], (T, H, W))
    doy = (pd.DatetimeIndex(inits) + valid_offset(lead)).dayofyear.to_numpy()
    ang = 2 * np.pi * doy / 365.25
    bc = lambda v: np.broadcast_to(np.asarray(v, np.float32)[:, None, None], (T, H, W))
    cols |= {"lat": lat, "lon": lon, "lead": np.full((T, H, W), lead, np.float32),
             "doy_sin": bc(np.sin(ang)), "doy_cos": bc(np.cos(ang))}
    if with_regime:
        z = zf.z_f.reindex(inits).to_numpy()
        code = zf.fc_regime.map(REG_CODE).reindex(inits).to_numpy()
        cols |= {"z_f": bc(z), "reg_code": bc(code)}
    return np.stack(list(cols.values()), -1), list(cols)


def design(raw, obs, ens, inits, land, zfs, w, with_regime, stride, boost):
    """Training rows (X, y, weight) pooled over leads, on a strided subset of land cells."""
    sub = np.zeros_like(land)
    sub[::stride, ::stride] = True
    X, Y, WT = [], [], []
    for lead in LEADS:
        F, names = features(raw, inits, lead, zfs[lead], with_regime)
        o = paired(ens.sel(init_time=inits), obs, lead)[1]
        ww = np.broadcast_to(w.reindex(inits).to_numpy()[:, None, None], o.shape)
        ok = np.isfinite(o) & (land & sub)[None]
        X.append(F[ok]); Y.append(o[ok]); WT.append(ww[ok])
    y = np.concatenate(Y)
    wt = np.concatenate(WT) * (1 + (boost - 1) * (y >= 64.5))
    return np.concatenate(X), np.log1p(np.clip(y, 0, None)), wt, names


def predict_all(model, raw, inits, land, zfs, with_regime) -> dict:
    """lead -> (T, H, W) mm/day prediction on land cells (NaN over ocean)."""
    out = {}
    for lead in LEADS:
        F, _ = features(raw, inits, lead, zfs[lead], with_regime)
        T, K = F.shape[0], F.shape[-1]
        p = np.full(F.shape[:3], np.nan)
        p[:, land] = model.predict(F[:, land].reshape(-1, K)).reshape(T, -1)
        out[lead] = np.clip(np.expm1(p), 0.0, None)
    return out


def boot_compare(fcs, obs, obs_lab, te, wt, a, rng) -> pd.DataFrame:
    rows = []
    for lead in LEADS:
        o = paired(fcs["raw_ens_mean"], obs, lead)[1]
        f = {n: paired(fc, obs, lead)[0] for n, fc in fcs.items()}
        reg = obs_lab.reindex(te + valid_offset(lead)).to_numpy()
        groups = {"all": np.ones(len(te), bool)} | {r: reg == r for r in REGIMES}
        for thr in THRESHOLDS:
            cnt = {n: counts(f[n], o, thr) for n in f}
            for g, sel in groups.items():
                if sel.sum() < MIN_DAYS:
                    continue
                n = int(sel.sum())
                idx, pt = _boot_idx(n, a.block, a.n_boot, rng), np.arange(n)[None]
                st = {k: ets_fb(cnt[k][sel], wt[sel], idx) for k in f}
                p0 = {k: ets_fb(cnt[k][sel], wt[sel], pt) for k in f}
                for A, B in PAIRS:
                    d_e = st[A][0] - st[B][0]
                    d_b = np.abs(st[A][1] - 1) - np.abs(st[B][1] - 1)
                    le, he = np.nanpercentile(d_e, [2.5, 97.5])
                    lb, hb = np.nanpercentile(d_b, [2.5, 97.5])
                    rows.append(dict(lead=lead, regime=g, threshold=thr, A=A, B=B, n_days=n,
                                     d_ets=float(p0[A][0][0] - p0[B][0][0]), d_ets_lo=le, d_ets_hi=he,
                                     ets_sig=bool(le > 0 or he < 0),
                                     d_fbdev=float(abs(p0[A][1][0] - 1) - abs(p0[B][1][0] - 1)),
                                     d_fbdev_lo=lb, d_fbdev_hi=hb, fb_sig=bool(lb > 0 or hb < 0)))
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-years", type=int, nargs="+", default=[2015, 2016])
    p.add_argument("--val-years", type=int, nargs="+", default=[2017])
    p.add_argument("--test-years", type=int, nargs="+", default=[2018, 2019])
    p.add_argument("--normal-every", type=int, default=4, help="must match plan_dates")
    p.add_argument("--stride", type=int, default=3, help="use every Nth grid cell when training")
    p.add_argument("--heavy-boost", type=float, default=3.0, help="extra weight on cells with obs >= 64.5 mm")
    p.add_argument("--n-estimators", type=int, default=600)
    p.add_argument("--block", type=int, default=3)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    ys = [set(a.train_years), set(a.val_years), set(a.test_years)]
    assert not (ys[0] & ys[1] or ys[0] & ys[2] or ys[1] & ys[2]), "train/val/test years must not overlap"

    obs_lab = pd.read_csv(LABELS, parse_dates=["date"]).set_index("date").regime
    fcl = pd.read_csv(FC_LABELS, parse_dates=["init_time", "date"])
    zfs = {lead: g.set_index("init_time")[["z_f", "fc_regime"]] for lead, g in fcl.groupby("lead")}
    fc_lab = {lead: z.fc_regime for lead, z in zfs.items()}
    obs = xr.open_zarr(C.IMD_ZARR).obs
    land = obs.notnull().any("time").values
    raw = xr.open_zarr(C.GEFS_ZARR).apcp

    plan = pd.DatetimeIndex(pd.read_csv(PLAN, header=None, parse_dates=[0])[0])
    inits = raw.init_time.to_index().intersection(plan)
    w = selection_weights(inits, obs_lab, a.normal_every)
    tr, va, te = (inits[inits.year.isin(y)] for y in (a.train_years, a.val_years, a.test_years))
    print(f"train {len(tr)}, val {len(va)}, test {len(te)} inits")
    ens = raw.mean("member").sel(init_time=inits).load()

    # baseline: global quantile mapping on the same train+val data
    trva = tr.union(va)
    qm_models = fit_models(ens.sel(init_time=trva), obs, w, obs_lab, fc_lab)
    fc_te = ens.sel(init_time=te)
    fcs = {"raw_ens_mean": fc_te, "qm_global": apply_models(fc_te, qm_models, "global", obs_lab, fc_lab)}

    for name, with_regime in (("lgbm", False), ("lgbm_regime", True)):
        print(f"\n--- {name} ---")
        X, y, wt, names = design(raw, obs, ens, tr, land, zfs, w, with_regime, a.stride, a.heavy_boost)
        Xv, yv, wv, _ = design(raw, obs, ens, va, land, zfs, w, with_regime, a.stride, a.heavy_boost)
        print(f"{len(y):,} train rows, {len(yv):,} val rows, {len(names)} features")
        model = lgb.LGBMRegressor(n_estimators=a.n_estimators, learning_rate=0.05, num_leaves=63,
                                  min_child_samples=200, subsample=0.7, subsample_freq=1,
                                  colsample_bytree=0.8, reg_lambda=5.0, random_state=a.seed, verbose=-1)
        model.fit(X, y, sample_weight=wt, eval_set=[(Xv, yv)], eval_sample_weight=[wv],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
        imp = pd.Series(model.booster_.feature_importance("gain"), index=names)
        print(f"best iteration {model.best_iteration_}; gain share:")
        print((imp / imp.sum()).sort_values(ascending=False).round(3).head(8).to_string())

        pv = predict_all(model, raw, va, land, zfs, with_regime)   # calibration set (never trained on)
        pt = predict_all(model, raw, te, land, zfs, with_regime)
        out = fc_te.copy(deep=True)
        for lead in LEADS:
            o_va = paired(ens.sel(init_time=va), obs, lead)[1]
            qm = QMap(pv[lead], o_va, w.reindex(va).to_numpy())
            out.loc[{"lead_day": lead}] = qm(pt[lead])
        fcs[name] = out

    wte = w.reindex(te)
    res = evaluate(fcs, obs, obs_lab, wte)
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_METRICS, index=False)
    for thr, g in res[res.threshold.isin((15.6, 64.5))].groupby("threshold"):
        print(f"\n=== threshold {thr} mm/day, test years {a.test_years} ===")
        print(g.pivot_table(index=["lead", "regime"], columns="model",
                            values=["ets", "freq_bias", "fss_5"]).round(2).to_string())

    cmp_ = boot_compare(fcs, obs, obs_lab, te, wte.to_numpy(), a, np.random.default_rng(a.seed))
    cmp_.to_csv(OUT_COMPARE, index=False)
    cols = ["lead", "regime", "threshold", "n_days", "d_ets", "d_ets_lo", "d_ets_hi", "ets_sig",
            "d_fbdev", "d_fbdev_lo", "d_fbdev_hi", "fb_sig"]
    for A, B in PAIRS:
        d = cmp_[(cmp_.A == A) & (cmp_.B == B)]
        print(f"\n=== {A} minus {B} (d_ets>0 better; d_fbdev<0 better), regime=all ===")
        print(d[d.regime == "all"][cols].round(3).to_string(index=False))
        print(f"significant over all {len(d)} rows: ETS {int(d.ets_sig.sum())}, freq-bias {int(d.fb_sig.sum())}"
              f" (expect ~{0.05 * len(d):.1f} each by chance)")
    print(f"\nwritten: {OUT_METRICS}\nwritten: {OUT_COMPARE}")


if __name__ == "__main__":
    main()