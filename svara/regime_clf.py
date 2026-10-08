"""Regime classifier: forecast fields only -> P(active / break / normal) on the valid day.

Features: core-monsoon-zone (CMZ) forecast rainfall statistics, ensemble spread, west-coast and all-India
means, forecast anomaly z_f, season, lead. Multinomial logistic regression, weighted by the inverse selection
probability. Probabilities for every start date are out-of-fold (leave-one-year-out), so no date is scored by
a model that saw its own year. A final model on all years is saved for deployment.

python -m svara.regime_clf
"""
import argparse

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .align import valid_offset
from .fc_regimes import OUT_LABELS as FC_LABELS
from .metrics import LABELS, PLAN, selection_weights
from .regimes import CMZ_LAT, CMZ_LON, Z_THR
ORDER = ["active", "break", "normal"]          # sklearn sorts classes alphabetically
OUT_PROBA = C.DATA / "labels" / "regime_proba.csv"
OUT_MODEL = C.DATA / "models" / "regime_clf.joblib"
WC_LAT, WC_LON = (8, 20), (72, 77)             # west-coast box


def box(da, lat, lon):
    return da.sel(lat=slice(*lat), lon=slice(*lon))


def build_features(inits) -> pd.DataFrame:
    raw = xr.open_zarr(C.GEFS_ZARR).apcp.sel(init_time=inits)
    land = xr.open_zarr(C.IMD_ZARR).obs.notnull().any("time").load()
    zf = pd.read_csv(FC_LABELS, parse_dates=["init_time", "date"])
    out = []
    for lead in range(1, C.MAX_LEAD + 1):
        r = raw.sel(lead_day=lead).load().where(land)
        mean, std = r.mean("member"), r.std("member")
        cmz = box(mean, CMZ_LAT, CMZ_LON)
        wts = np.cos(np.deg2rad(cmz.lat))
        ll = ("lat", "lon")
        f = {"cmz_mean": cmz.weighted(wts).mean(ll).values,
             "cmz_p90": cmz.quantile(0.9, ll).values,
             "cmz_std": cmz.std(ll).values,
             "cmz_spread": box(std, CMZ_LAT, CMZ_LON).mean(ll).values,
             "wc_mean": box(mean, WC_LAT, WC_LON).mean(ll).values,
             "india_mean": mean.mean(ll).values}
        for thr in (2.5, 15.6, 64.5):
            f[f"cmz_frac_{thr}"] = (cmz >= thr).where(cmz.notnull()).mean(ll).values
        d = pd.DataFrame(f)
        d["init_time"] = pd.DatetimeIndex(r.init_time.values)
        d["lead"] = lead
        d["z_f"] = d.init_time.map(zf[zf.lead == lead].set_index("init_time").z_f)
        d["z_hi"] = (d.z_f >= Z_THR).astype(float)
        d["z_lo"] = (d.z_f <= -Z_THR).astype(float)
        doy = (d.init_time + valid_offset(lead)).dt.dayofyear
        d["doy_sin"], d["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
        out.append(d)
    return pd.concat(out, ignore_index=True)


def make_model(c):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))


def recalls(true, pred, w):
    return {c: round(float(np.average(pred[true == c] == c, weights=w[true == c])), 2)
            for c in ORDER if (true == c).any()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal-every", type=int, default=4, help="must match plan_dates")
    ap.add_argument("--C", type=float, default=0.3, help="inverse L2 strength")
    a = ap.parse_args()

    obs_lab = pd.read_csv(LABELS, parse_dates=["date"]).set_index("date").regime
    plan = pd.DatetimeIndex(pd.read_csv(PLAN, header=None, parse_dates=[0])[0])
    inits = xr.open_zarr(C.GEFS_ZARR).init_time.to_index().intersection(plan)
    w = selection_weights(inits, obs_lab, a.normal_every)

    df = build_features(inits)
    feats = [c for c in df.columns if c != "init_time"]
    df[feats] = df[feats].fillna(df[feats].median())
    df["date"] = df.init_time + df.lead.map(valid_offset)
    df["y"] = obs_lab.reindex(df.date).to_numpy()
    df["w"] = w.reindex(pd.DatetimeIndex(df.init_time)).to_numpy()
    lab = (df.y.notna() & df.w.notna()).to_numpy()
    year = df.init_time.dt.year.to_numpy()
    print(f"{len(df)} rows ({lab.sum()} labelled), {len(feats)} features, years {sorted(set(year))}")

    P = np.full((len(df), 3), np.nan)
    for yr in sorted(set(year)):
        tr, te = lab & (year != yr), year == yr
        m = make_model(a.C).fit(df.loc[tr, feats], df.y[tr], logisticregression__sample_weight=df.w[tr])
        P[te] = m.predict_proba(df.loc[te, feats])
    res = df[["init_time", "lead"]].copy()
    res[["p_active", "p_break", "p_normal"]] = P
    OUT_PROBA.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_PROBA, index=False)

    # evaluation against the threshold rule used so far (fc_regime)
    ev = df[lab].copy()
    ev[["p_active", "p_break", "p_normal"]] = P[lab]
    rule = pd.read_csv(FC_LABELS, parse_dates=["init_time"])[["init_time", "lead", "fc_regime"]]
    ev = ev.merge(rule, on=["init_time", "lead"], how="left")
    yi = pd.Categorical(ev.y, categories=ORDER).codes
    pr = np.clip(ev[["p_active", "p_break", "p_normal"]].to_numpy()[np.arange(len(ev)), yi], 1e-6, 1)
    clim = np.array([np.average(ev.y == c, weights=ev.w) for c in ORDER])
    ll_m = np.average(-np.log(pr), weights=ev.w)
    ll_c = np.average(-np.log(clim[yi]), weights=ev.w)
    print(f"\nweighted log-loss: classifier {ll_m:.3f}, climatology {ll_c:.3f}, skill {1 - ll_m / ll_c:+.3f}")
    hard = np.array(ORDER)[ev[["p_active", "p_break", "p_normal"]].to_numpy().argmax(1)]
    true, wv = ev.y.to_numpy(), ev.w.to_numpy()
    print("recall by observed regime, argmax of classifier:", recalls(true, hard, wv))
    print("recall by observed regime, threshold rule      :", recalls(true, ev.fc_regime.to_numpy(), wv))
    prior_adj = np.array(ORDER)[(ev[["p_active", "p_break", "p_normal"]].to_numpy() / clim).argmax(1)]
    print("recall by observed regime, prior-corrected     :", recalls(true, prior_adj, wv))
    prec = lambda pred: {c: round(float(np.average(true[pred == c] == c, weights=wv[pred == c])), 2)
        for c in ORDER if (pred == c).any()}
    print("precision, threshold rule:", prec(ev.fc_regime.to_numpy()), "| prior-corrected:", prec(prior_adj))
    print("mean predicted probability by observed regime:")
    print(ev.groupby("y")[["p_active", "p_break", "p_normal"]].mean().round(3).to_string())

    final = make_model(a.C).fit(df.loc[lab, feats], df.y[lab], logisticregression__sample_weight=df.w[lab])
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "features": feats}, OUT_MODEL)
    print(f"\nwritten: {OUT_PROBA}\nwritten: {OUT_MODEL}")


if __name__ == "__main__":
    main()