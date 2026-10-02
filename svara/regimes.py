"""Day-level active / break / normal labels for the Indian summer monsoon (Rajeevan et al. 2010 style)."""
import argparse

import numpy as np
import pandas as pd
import xarray as xr

from . import config as C

CMZ_LAT, CMZ_LON = (18.0, 28.0), (65.0, 88.0)  # core monsoon zone
Z_THR, MIN_RUN = 1.0, 3                          # |z| >= 1 for >= 3 consecutive days
OUT = C.DATA / "labels" / "day_regime.csv"


def cmz_series() -> pd.Series:
    """cos(lat)-weighted CMZ-mean IMD rainfall (mm/day), JJAS only."""
    obs = xr.open_zarr(C.IMD_ZARR).obs
    box = obs.sel(lat=slice(*CMZ_LAT), lon=slice(*CMZ_LON))
    s = box.weighted(np.cos(np.deg2rad(box.lat))).mean(("lat", "lon")).to_series()
    return s[s.index.month.isin((6, 7, 8, 9))].dropna()


def standardised_anomaly(s: pd.Series) -> pd.Series:
    key = pd.Series(s.index.strftime("%m-%d"), index=s.index)
    clim = s.groupby(key).mean().sort_index().rolling(15, center=True, min_periods=1).mean()
    anom = s - clim.reindex(key).to_numpy()
    return anom / anom.std()


def _runs(flag: pd.Series, min_len: int) -> pd.Series:
    """True only inside runs of >= min_len consecutive True days, never spanning two seasons."""
    year = pd.Series(flag.index.year, index=flag.index)
    grp = ((flag != flag.shift()) | (year != year.shift())).cumsum()
    return flag & (flag.groupby(grp).transform("size") >= min_len)


def label_days(min_years: int = 10) -> pd.DataFrame:
    s = cmz_series()
    n_years = s.index.year.nunique()
    if n_years < min_years:
        raise SystemExit(
            f"Only {n_years} season(s) of IMD data; the climatology needs >= {min_years}. "
            "Download more years (python -m svara.build_data --years 2000 2019 --steps imd) "
            "or pass --min-years 1 to test the code only."
        )
    z = standardised_anomaly(s)
    regime = pd.Series("normal", index=s.index)
    regime[_runs(z >= Z_THR, MIN_RUN)] = "active"
    regime[_runs(z <= -Z_THR, MIN_RUN)] = "break"
    return pd.DataFrame({"cmz_mean": s, "z": z, "regime": regime}).rename_axis("date")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--min-years", type=int, default=10)
    df = label_days(p.parse_args().min_years)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT)
    print(pd.crosstab(df.index.year, df.regime))
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()