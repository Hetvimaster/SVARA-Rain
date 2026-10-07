"""Forecast-implied monsoon regimes, and a gate test that groups days by them.

The original gate groups days by the regime of the OBSERVED rainfall, which is also what the forecast is
verified against; that alone makes break days look "too wet" and active days "too dry". Here the regime
is read from the FORECAST (CMZ-mean rainfall anomaly of the ensemble mean), which is what is available in
real time. Days are weighted by inverse selection probability (plan_dates keeps 1 in N normal days).

python -m svara.fc_regimes
"""
import argparse

import numpy as np
import pandas as pd
import xarray as xr

from . import config as C
from .align import valid_offset
from .gate import _boot_idx, daily_bias
from .metrics import PLAN, REGIMES, selection_weights
from .regimes import CMZ_LAT, CMZ_LON, OUT as OBS_LABELS, Z_THR, cmz_series

OUT_LABELS = C.DATA / "labels" / "fc_regime.csv"
OUT_GATE = C.DATA / "reports" / "gate_fc_regimes.csv"


def obs_climatology():
    """Same climatology and sigma as regimes.standardised_anomaly, so z is on the same scale."""
    s = cmz_series()
    key = pd.Series(s.index.strftime("%m-%d"), index=s.index)
    clim = s.groupby(key).mean().sort_index().rolling(15, center=True, min_periods=1).mean()
    sigma = float((s - clim.reindex(key).to_numpy()).std())
    return s, clim, sigma


def fc_cmz(fc: xr.DataArray, lead: int, land: xr.DataArray) -> pd.Series:
    """cos(lat)-weighted CMZ mean of the ensemble-mean forecast over IMD land cells, by init_time."""
    f = fc.sel(lead_day=lead).sel(lat=slice(*CMZ_LAT), lon=slice(*CMZ_LON)).load()
    f = f.where(land.sel(lat=f.lat, lon=f.lon))
    m = f.weighted(np.cos(np.deg2rad(f.lat))).mean(("lat", "lon"))
    return m.to_series()


def forecast_regimes(inits, weights, debias=True) -> pd.DataFrame:
    """One row per (init, lead): forecast CMZ anomaly z_f and regime (single-day |z| >= Z_THR)."""
    fc = xr.open_zarr(C.GEFS_ZARR).apcp.mean("member").sel(init_time=inits)
    obs = xr.open_zarr(C.IMD_ZARR).obs
    land = obs.notnull().any("time").load()
    s, clim, sigma = obs_climatology()
    rows = []
    for lead in range(1, C.MAX_LEAD + 1):
        f = fc_cmz(fc, lead, land)
        vd = pd.DatetimeIndex(f.index + valid_offset(lead))
        o = s.reindex(vd).to_numpy()
        bias = 0.0
        if debias:  # mean CMZ forecast-minus-obs error at this lead, weighted for the sampling
            ok = np.isfinite(o)
            bias = float(np.average(f.to_numpy()[ok] - o[ok], weights=weights.reindex(f.index).to_numpy()[ok]))
        z = (f.to_numpy() - bias - clim.reindex(vd.strftime("%m-%d")).to_numpy()) / sigma
        reg = np.select([z >= Z_THR, z <= -Z_THR], ["active", "break"], "normal").astype(object)
        reg[~np.isfinite(z)] = np.nan
        print(f"lead {lead}: CMZ forecast bias {bias:+.2f} mm/day ({bias / sigma:+.2f} sigma)")
        rows.append(pd.DataFrame({"init_time": f.index, "lead": lead, "date": vd, "z_f": z, "fc_regime": reg}))
    return pd.concat(rows, ignore_index=True)


def _stats(f, o, w, idx):
    fb, ob, wb = f[idx], o[idx], w[idx]
    return ((fb - ob) * wb).sum(1) / wb.sum(1), (fb * wb).sum(1) / (ob * wb).sum(1) - 1


def gate_rows(d: pd.DataFrame, regcol: str, name: str, lead: int, block: int, n_boot: int, rng) -> list[dict]:
    grp, boot = {}, {}
    for r in REGIMES:
        g = d[d[regcol] == r].sort_index()
        grp[r] = (g.fc.to_numpy(), g.obs.to_numpy(), g.w.to_numpy())
        if len(g):
            boot[r] = _stats(*grp[r], _boot_idx(len(g), block, n_boot, rng))
    rows = []
    for r in REGIMES:
        f, o, w = grp[r]
        row = {"labels": name, "lead": lead, "regime": r, "n_days": len(f)}
        if len(f):
            bm, br = boot[r]
            rel = 100 * ((w * f).sum() / (w * o).sum() - 1)
            row.update(bias_mm=(w * (f - o)).sum() / w.sum(),
                       bias_lo=np.percentile(bm, 2.5), bias_hi=np.percentile(bm, 97.5),
                       rel_pct=rel, rel_lo=100 * np.percentile(br, 2.5), rel_hi=100 * np.percentile(br, 97.5))
            if r != "normal" and "normal" in boot:
                nf, no, nw = grp["normal"]
                lo, hi = 100 * np.percentile(br - boot["normal"][1], [2.5, 97.5])
                row.update(rel_diff_vs_normal=rel - 100 * ((nw * nf).sum() / (nw * no).sum() - 1),
                           rel_diff_lo=lo, rel_diff_hi=hi, differs=bool(lo > 0 or hi < 0))
        rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--normal-every", type=int, default=4, help="must match plan_dates")
    p.add_argument("--no-debias", action="store_true", help="do not remove the mean CMZ forecast bias")
    p.add_argument("--block", type=int, default=3)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    lab = pd.read_csv(OBS_LABELS, parse_dates=["date"]).set_index("date").regime
    plan = pd.DatetimeIndex(pd.read_csv(PLAN, header=None, parse_dates=[0])[0])
    zinit = pd.DatetimeIndex(xr.open_zarr(C.GEFS_ZARR).init_time.values)
    inits = zinit.intersection(plan)
    weights = selection_weights(inits, lab, a.normal_every)
    print(f"{len(inits)} planned inits; normal-day weight {a.normal_every}")

    fr = forecast_regimes(inits, weights, debias=not a.no_debias)
    OUT_LABELS.parent.mkdir(parents=True, exist_ok=True)
    fr.to_csv(OUT_LABELS, index=False)

    rng = np.random.default_rng(a.seed)
    rows = []
    for lead in range(1, C.MAX_LEAD + 1):
        d = daily_bias(lead)
        meta = fr[fr.lead == lead].set_index("date")[["fc_regime"]].join(lab.rename("obs_regime"))
        meta["w"] = weights.reindex(pd.DatetimeIndex(fr[fr.lead == lead].init_time)).to_numpy()
        d = d.join(meta, how="inner")
        ct = pd.crosstab(d.obs_regime, d.fc_regime, values=d.w, aggfunc="sum", normalize="index")
        print(f"\nlead {lead}: P(forecast regime | observed regime), weighted")
        print(ct.round(2).to_string())
        for name, col in (("observed", "obs_regime"), ("forecast", "fc_regime")):
            rows += gate_rows(d, col, name, lead, a.block, a.n_boot, rng)

    res = pd.DataFrame(rows)
    OUT_GATE.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_GATE, index=False)
    cols = ["labels", "lead", "regime", "n_days", "bias_mm", "rel_pct", "rel_diff_vs_normal",
            "rel_diff_lo", "rel_diff_hi", "differs"]
    print("\n" + res[cols].round(2).to_string(index=False))
    print(f"\nwritten: {OUT_LABELS}\nwritten: {OUT_GATE}")


if __name__ == "__main__":
    main()