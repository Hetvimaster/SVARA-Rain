import numpy as np
import xarray as xr

from . import config as C


def main() -> None:
    fc = xr.open_zarr(C.GEFS_ZARR).apcp.sel(lead_day=1).mean("member").load()
    obs = xr.open_zarr(C.IMD_ZARR).obs
    print(f"{fc.sizes['init_time']} init dates; configured IMD_LABEL={C.IMD_LABEL!r}")
    print("lag(days)  mean pixel anomaly-correlation  (lead-1 ens-mean vs IMD at init+lag)")
    best = (None, -9.0)
    for lag in range(-1, 4):
        vt = fc.init_time.values + np.timedelta64(lag, "D")
        o = obs.reindex(time=vt).load().values
        a = fc.values
        keep = np.isfinite(o).all(0)
        a, b = a[:, keep], o[:, keep]
        a, b = a - a.mean(0), b - b.mean(0)
        r = ((a * b).sum(0) / np.sqrt((a**2).sum(0) * (b**2).sum(0) + 1e-12)).mean()
        print(f"{lag:>8}  {r:.3f}")
        best = (lag, r) if r > best[1] else best
    print(f"\nbest lag = {best[0]}  ->  IMD_LABEL should be {'end' if best[0] == 1 else 'start' if best[0] == 0 else '??? (investigate)'}")


if __name__ == "__main__":
    main()