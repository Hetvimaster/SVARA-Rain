"""Quantile-mapping baselines: global vs regime-wise, scored with the same metrics as the raw forecast.

Ladder rungs produced here:
  raw_ens_mean       (reference)
  qm_global          one mapping per lead
  qm_regime_fc       one mapping per (lead, FORECAST regime)   <- deployable, uses only forecast fields
  qm_regime_obs      one mapping per (lead, OBSERVED regime)   <- oracle upper bound, NOT deployable

Training and testing are split by whole years (never random days). Training pairs are weighted by the
inverse selection probability from plan_dates, so the sampled normal days are not under-represented.

python -m svara.qmap --train-years 2015 2016 2017 --test-years 2018 2019
"""
import argparse

import numpy as np
import pandas as pd
import xarray as xr

from . import config as C
from .align import pair, valid_offset
from .fc_regimes import OUT_LABELS as FC_LABELS
from .metrics import LABELS, PLAN, REGIMES, evaluate, selection_weights

OUT = C.DATA / "reports" / "metrics_qmap.csv"
MIN_DAYS = 15      # fewer training days than this in a (lead, regime) group -> fall back to the global map
CELL_STRIDE = 2    # subsample grid cells when fitting (neighbouring cells are highly correlated)
Q = np.unique(np.r_[np.linspace(0, 0.9, 91), np.linspace(0.9, 0.99, 19),
                    0.995, 0.998, 0.999, 0.9995, 0.9999, 1.0])


def wquantile(x: np.ndarray, w: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Weighted quantiles of x at levels q."""
    order = np.argsort(x, kind="stable")
    xs, cw = x[order], np.cumsum(w[order])
    return np.interp(q, (cw - 0.5 * w[order]) / cw[-1], xs)


class QMap:
    """Monotone transfer function forecast quantiles -> observed quantiles. Above the top quantile the
    last offset is carried forward (additive), so heavy tails are not clipped."""

    def __init__(self, f: np.ndarray, o: np.ndarray, w: np.ndarray):
        """f, o: (T, H, W) paired mm/day; w: (T,) day weights."""
        f, o = f[:, ::CELL_STRIDE, ::CELL_STRIDE], o[:, ::CELL_STRIDE, ::CELL_STRIDE]
        m = np.isfinite(f) & np.isfinite(o)
        wm = np.broadcast_to(w[:, None, None], f.shape)[m]
        self.n_days = int(m.any((1, 2)).sum())
        self.fq = np.maximum.accumulate(wquantile(f[m], wm, Q))
        self.oq = np.maximum.accumulate(wquantile(o[m], wm, Q))

    def __call__(self, f: np.ndarray) -> np.ndarray:
        out = np.interp(f, self.fq, self.oq)  # NaN stays NaN
        hi = f > self.fq[-1]
        out[hi] = self.oq[-1] + (f[hi] - self.fq[-1])
        return np.clip(out, 0.0, None)


def paired(fc, obs, lead):
    f, o = pair(fc, obs, lead)
    return f.values.astype(np.float64), o.values.astype(np.float64)


def fit_models(fc, obs, w, obs_lab, fc_lab) -> dict:
    """models[(kind, lead, regime)] with kind in global / fc / obs. fc, w cover training inits only."""
    init = pd.DatetimeIndex(fc.init_time.values)
    models = {}
    for lead in range(1, C.MAX_LEAD + 1):
        f, o = paired(fc, obs, lead)
        wv = w.reindex(init).to_numpy()
        models[("global", lead, "all")] = glob = QMap(f, o, wv)
        regs = {"obs": obs_lab.reindex(init + valid_offset(lead)).to_numpy(),
                "fc": fc_lab[lead].reindex(init).to_numpy()}
        for kind, reg in regs.items():
            for r in REGIMES:
                sel = reg == r
                if sel.sum() >= MIN_DAYS:
                    models[(kind, lead, r)] = QMap(f[sel], o[sel], wv[sel])
                else:
                    models[(kind, lead, r)] = glob
                    print(f"  lead {lead} {kind}/{r}: only {int(sel.sum())} training days, using global map")
    return models


def apply_models(fc, models, kind, obs_lab, fc_lab) -> xr.DataArray:
    """Correct every (init, lead) of fc with the map for its regime (kind) or the global map."""
    init = pd.DatetimeIndex(fc.init_time.values)
    out = fc.copy(deep=True)
    for lead in range(1, C.MAX_LEAD + 1):
        f = fc.sel(lead_day=lead).transpose("init_time", "lat", "lon").values.astype(np.float64)
        if kind == "global":
            reg = np.array(["all"] * len(init), dtype=object)
        elif kind == "obs":
            reg = obs_lab.reindex(init + valid_offset(lead)).to_numpy()
        else:
            reg = fc_lab[lead].reindex(init).to_numpy()
        new = np.empty_like(f)
        for i, r in enumerate(reg):
            key = (kind, lead, r) if r in REGIMES or kind == "global" else ("global", lead, "all")
            new[i] = models.get(key, models[("global", lead, "all")])(f[i])
        out.loc[{"lead_day": lead}] = new
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-years", type=int, nargs="+", default=[2015, 2016, 2017])
    p.add_argument("--test-years", type=int, nargs="+", default=[2018, 2019])
    p.add_argument("--normal-every", type=int, default=4, help="must match plan_dates")
    a = p.parse_args()
    assert not set(a.train_years) & set(a.test_years), "train and test years must not overlap"

    obs_lab = pd.read_csv(LABELS, parse_dates=["date"]).set_index("date").regime
    fcl = pd.read_csv(FC_LABELS, parse_dates=["init_time", "date"])
    fc_lab = {lead: g.set_index("init_time").fc_regime for lead, g in fcl.groupby("lead")}
    obs = xr.open_zarr(C.IMD_ZARR).obs
    raw = xr.open_zarr(C.GEFS_ZARR).apcp

    plan = pd.DatetimeIndex(pd.read_csv(PLAN, header=None, parse_dates=[0])[0])
    inits = raw.init_time.to_index().intersection(plan)
    w = selection_weights(inits, obs_lab, a.normal_every)
    tr = inits[inits.year.isin(a.train_years)]
    te = inits[inits.year.isin(a.test_years)]
    print(f"train inits {len(tr)} ({a.train_years}), test inits {len(te)} ({a.test_years})")

    ens = raw.mean("member")
    fc_tr, fc_te = ens.sel(init_time=tr).load(), ens.sel(init_time=te).load()

    print("fitting...")
    models = fit_models(fc_tr, obs, w, obs_lab, fc_lab)

    forecasts = {"raw_ens_mean": fc_te}
    for name, kind in (("qm_global", "global"), ("qm_regime_fc", "fc"), ("qm_regime_obs", "obs")):
        forecasts[name] = apply_models(fc_te, models, kind, obs_lab, fc_lab)

    res = evaluate(forecasts, obs, obs_lab, w.reindex(te))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    show = res[res.threshold.isin((15.6, 64.5))]
    for thr, g in show.groupby("threshold"):
        print(f"\n=== threshold {thr} mm/day (test years {a.test_years}) ===")
        t = g.pivot_table(index=["lead", "regime"], columns="model", values=["ets", "freq_bias", "rmse"])
        print(t.round(2).to_string())
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()