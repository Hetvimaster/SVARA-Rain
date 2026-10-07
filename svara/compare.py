"""Paired block-bootstrap comparison of the quantile-mapping rungs (ETS and frequency bias).

For each (lead, observed-regime group, threshold) it resamples test days with circular blocks, using the SAME
resampled days for both models in a pair, and reports the difference with a 95% interval.

  d_ets      = ETS(A) - ETS(B)               > 0 means A is better
  d_fbdev    = |freqbias(A)-1| - |freqbias(B)-1|   < 0 means A is closer to unbiased

python -m svara.compare --train-years 2015 2016 2017 --test-years 2018 2019
"""
import argparse

import numpy as np
import pandas as pd
import xarray as xr

from . import config as C
from .align import valid_offset
from .fc_regimes import OUT_LABELS as FC_LABELS
from .gate import _boot_idx
from .metrics import LABELS, PLAN, REGIMES, selection_weights
from .qmap import apply_models, fit_models, paired

OUT = C.DATA / "reports" / "compare_qmap.csv"
THRESHOLDS = (15.6, 64.5)
PAIRS = (("qm_regime_fc", "qm_global"),   # does regime-awareness help (deployable)?
         ("qm_regime_obs", "qm_global"),  # same, with oracle regimes
         ("qm_global", "raw_ens_mean"),   # does any quantile mapping help?
         ("qm_regime_fc", "raw_ens_mean"))
MIN_DAYS = 10


def counts(f: np.ndarray, o: np.ndarray, thr: float) -> np.ndarray:
    """Per-day contingency counts, shape (T, 4): hits, false alarms, misses, valid cells."""
    m = np.isfinite(f) & np.isfinite(o)
    fe, oe = (f >= thr) & m, (o >= thr) & m
    return np.stack([(fe & oe).sum((1, 2)), (fe & ~oe).sum((1, 2)),
                     (~fe & oe).sum((1, 2)), m.sum((1, 2))], 1).astype(np.float64)


def ets_fb(c: np.ndarray, w: np.ndarray, idx: np.ndarray):
    """ETS and frequency bias for each resample. c: (T,4), w: (T,), idx: (B,T') day indices."""
    h, fa, mi, n = (c[idx] * w[idx][..., None]).sum(1).T
    with np.errstate(divide="ignore", invalid="ignore"):
        hr = (h + mi) * (h + fa) / n
        return (h - hr) / (h + mi + fa - hr), (h + fa) / (h + mi)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-years", type=int, nargs="+", default=[2015, 2016, 2017])
    p.add_argument("--test-years", type=int, nargs="+", default=[2018, 2019])
    p.add_argument("--normal-every", type=int, default=4, help="must match plan_dates")
    p.add_argument("--block", type=int, default=3)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
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
    tr, te = inits[inits.year.isin(a.train_years)], inits[inits.year.isin(a.test_years)]
    print(f"train inits {len(tr)}, test inits {len(te)}")

    ens = raw.mean("member")
    fc_tr, fc_te = ens.sel(init_time=tr).load(), ens.sel(init_time=te).load()
    models = fit_models(fc_tr, obs, w, obs_lab, fc_lab)
    fcs = {"raw_ens_mean": fc_te}
    for name, kind in (("qm_global", "global"), ("qm_regime_fc", "fc"), ("qm_regime_obs", "obs")):
        fcs[name] = apply_models(fc_te, models, kind, obs_lab, fc_lab)

    wt = w.reindex(te).to_numpy()
    rng = np.random.default_rng(a.seed)
    rows = []
    for lead in range(1, C.MAX_LEAD + 1):
        o = paired(fcs["raw_ens_mean"], obs, lead)[1]
        f = {name: paired(fc, obs, lead)[0] for name, fc in fcs.items()}
        reg = obs_lab.reindex(te + valid_offset(lead)).to_numpy()
        groups = {"all": np.ones(len(te), bool)} | {r: reg == r for r in REGIMES}
        for thr in THRESHOLDS:
            cnt = {name: counts(f[name], o, thr) for name in f}
            for g, sel in groups.items():
                if sel.sum() < MIN_DAYS:
                    continue
                n = int(sel.sum())
                idx = _boot_idx(n, a.block, a.n_boot, rng)       # shared by every model in this group
                pt = np.arange(n)[None]
                st = {}
                for name in f:
                    c, ww = cnt[name][sel], wt[sel]
                    st[name] = (ets_fb(c, ww, idx), ets_fb(c, ww, pt))
                for A, B in PAIRS:
                    (eA, bA), (pA_e, pA_b) = st[A]
                    (eB, bB), (pB_e, pB_b) = st[B]
                    d_e = eA - eB
                    d_b = np.abs(bA - 1) - np.abs(bB - 1)
                    lo_e, hi_e = np.nanpercentile(d_e, [2.5, 97.5])
                    lo_b, hi_b = np.nanpercentile(d_b, [2.5, 97.5])
                    rows.append(dict(
                        lead=lead, regime=g, threshold=thr, A=A, B=B, n_days=n,
                        obs_event_days=int((cnt["raw_ens_mean"][sel][:, 0] + cnt["raw_ens_mean"][sel][:, 2] > 0).sum()),
                        ets_A=float(pA_e[0]), ets_B=float(pB_e[0]),
                        d_ets=float(pA_e[0] - pB_e[0]), d_ets_lo=lo_e, d_ets_hi=hi_e,
                        ets_sig=bool(lo_e > 0 or hi_e < 0),
                        d_fbdev=float(abs(pA_b[0] - 1) - abs(pB_b[0] - 1)), d_fbdev_lo=lo_b, d_fbdev_hi=hi_b,
                        fb_sig=bool(lo_b > 0 or hi_b < 0)))

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    cols = ["lead", "regime", "threshold", "n_days", "d_ets", "d_ets_lo", "d_ets_hi", "ets_sig",
            "d_fbdev", "d_fbdev_lo", "d_fbdev_hi", "fb_sig"]
    for A, B in PAIRS:
        d = res[(res.A == A) & (res.B == B)]
        print(f"\n=== {A} minus {B}  (d_ets>0 better; d_fbdev<0 better) ===")
        print(d[cols].round(3).to_string(index=False))
        print(f"significant: ETS {int(d.ets_sig.sum())}/{len(d)}, freq-bias {int(d.fb_sig.sum())}/{len(d)}"
              f"  (expect ~{0.05 * len(d):.1f} each by chance alone)")
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()