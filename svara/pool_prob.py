"""Pool per-day Brier stats from the leave-one-year-out folds; bootstrap BSS and the regime effect."""
import glob

import numpy as np
import pandas as pd

from . import config as C
from .compare import MIN_DAYS
from .gate import _boot_idx


def main() -> None:
    files = sorted(glob.glob(str(C.DATA / "reports" / "prob_days_*.csv")))
    d = pd.concat([pd.read_csv(f, parse_dates=["init"]) for f in files])
    print(f"{len(files)} folds, {d.init.nunique()} test inits")
    rng, rows = np.random.default_rng(0), []
    for (thr, lead), g in d.groupby(["threshold", "lead"]):
        piv = lambda c: g.pivot(index="init", columns="model", values=c).sort_index()
        B, R, W = piv("B"), piv("R"), piv("w")["lgbm"].to_numpy()
        reg = g[g.model == "lgbm"].set_index("init").regime.reindex(B.index).to_numpy()
        for grp in ("all", "active", "break", "normal"):
            sel = np.ones(len(B), bool) if grp == "all" else reg == grp
            n = int(sel.sum())
            if n < MIN_DAYS:
                continue
            idx, ws, bs = _boot_idx(n, 3, 2000, rng), W[sel], {}
            for m in B.columns:
                num, den = ws * B[m].to_numpy()[sel], ws * R[m].to_numpy()[sel]
                bs[m] = (1 - num.sum() / den.sum(), 1 - num[idx].sum(1) / den[idx].sum(1))
                lo, hi = np.percentile(bs[m][1], [2.5, 97.5])
                rows.append(dict(threshold=thr, lead=lead, regime=grp, model=m, n_days=n, bss=bs[m][0], lo=lo, hi=hi))
            dd = bs["lgbm_regime"][1] - bs["lgbm"][1]
            lo, hi = np.percentile(dd, [2.5, 97.5])
            rows.append(dict(threshold=thr, lead=lead, regime=grp, model="regime_minus_noregime", n_days=n,
                             bss=bs["lgbm_regime"][0] - bs["lgbm"][0], lo=lo, hi=hi, sig=bool(lo > 0 or hi < 0)))
    r = pd.DataFrame(rows)
    out = C.DATA / "reports" / "prob_pooled.csv"
    r.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    x = r[r.model == "regime_minus_noregime"]
    print(x[["threshold", "lead", "regime", "n_days", "bss", "lo", "hi", "sig"]].round(4).to_string(index=False))
    print(f"significant {int(x.sig.sum())} of {len(x)} (expect ~{0.05 * len(x):.1f} by chance)\nwritten: {out}")


if __name__ == "__main__":
    main()