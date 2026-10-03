"""Gate test: does raw GEFS bias differ by monsoon regime? Circular block bootstrap on daily India-mean values."""
import argparse

import numpy as np
import pandas as pd
import xarray as xr

from . import config as C
from .align import pair, valid_offset

REGIMES = ("active", "break", "normal")
LABELS = C.DATA / "labels" / "day_regime.csv"
OUT = C.DATA / "reports" / "gate_test.csv"


def daily_bias(lead: int) -> pd.DataFrame:
    """Land-mean ensemble-mean forecast and IMD obs per valid date (mm/day)."""
    fc = xr.open_zarr(C.GEFS_ZARR).apcp.mean("member")
    obs = xr.open_zarr(C.IMD_ZARR).obs
    f, o = (x.load() for x in pair(fc, obs, lead))
    land = o.notnull().any("init_time")
    w = np.cos(np.deg2rad(f.lat))
    fm = f.where(land).weighted(w).mean(("lat", "lon"))
    om = o.where(land).weighted(w).mean(("lat", "lon"))
    vd = pd.DatetimeIndex(fm.init_time.values + valid_offset(lead).to_timedelta64()).rename("date")
    return pd.DataFrame({"fc": fm.values, "obs": om.values}, index=vd).dropna()


def _boot_idx(m: int, block: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """(n, m) day indices from a circular block bootstrap (blocks wrap around the series end)."""
    b = min(block, m)
    starts = rng.integers(0, m, size=(n, int(np.ceil(m / b))))
    return ((starts[..., None] + np.arange(b)) % m).reshape(n, -1)[:, :m]


def _boot_stats(f: np.ndarray, o: np.ndarray, idx: np.ndarray):
    """Bootstrap distributions of mean bias (mm/day) and relative bias (ratio of totals - 1)."""
    fb, ob = f[idx], o[idx]
    return (fb - ob).mean(1), fb.sum(1) / ob.sum(1) - 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--block", type=int, default=3)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    lab = pd.read_csv(LABELS, parse_dates=["date"]).set_index("date").regime
    rng = np.random.default_rng(a.seed)
    rows = []
    for lead in range(1, C.MAX_LEAD + 1):
        d = daily_bias(lead).join(lab, how="inner")
        grp, boot = {}, {}
        for r in REGIMES:
            g = d[d.regime == r].sort_index()
            grp[r] = (g.fc.to_numpy(), g.obs.to_numpy())
            if len(g):
                boot[r] = _boot_stats(*grp[r], _boot_idx(len(g), a.block, a.n_boot, rng))
        for r in REGIMES:
            f, o = grp[r]
            row = {"lead": lead, "regime": r, "n_days": len(f)}
            if len(f):
                bm, br = boot[r]
                rel = 100 * (f.sum() / o.sum() - 1)
                row.update(bias_mm=(f - o).mean(),
                           bias_lo=np.percentile(bm, 2.5), bias_hi=np.percentile(bm, 97.5),
                           rel_pct=rel,
                           rel_lo=100 * np.percentile(br, 2.5), rel_hi=100 * np.percentile(br, 97.5))
                if r != "normal" and "normal" in boot:
                    nf, no = grp["normal"]
                    lo, hi = 100 * np.percentile(br - boot["normal"][1], [2.5, 97.5])
                    row.update(rel_diff_vs_normal=rel - 100 * (nf.sum() / no.sum() - 1),
                               rel_diff_lo=lo, rel_diff_hi=hi, differs=bool(lo > 0 or hi < 0))
            rows.append(row)

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    print(res.round(2).to_string(index=False))
    print(f"\nwritten: {OUT}")
    if res.n_days.min() < 20:
        print("NOTE: some groups have < 20 days; intervals are indicative only.")


if __name__ == "__main__":
    main()