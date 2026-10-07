"""District product for one (init, lead): mean/max corrected rain, heavy-rain probabilities, area fraction,
dominant forecast regime, provisional colour category.

python -m svara.district --districts data/geo/districts.geojson --name-col DISTRICT --init 2018-07-15 --lead 1
"""
import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
import regionmask
import xarray as xr

from . import config as C
from .fc_regimes import OUT_LABELS as FC_LABELS

PROD = C.DATA / "products"


def colour(mx: float) -> str:  # provisional, on district max corrected rain (IMD 24 h categories)
    return "green" if mx < 15.6 else "yellow" if mx < 64.5 else "orange" if mx < 115.6 else "red"


def district_table(gdf, name_col, init, lead) -> pd.DataFrame:
    rain = xr.open_zarr(PROD / "corrected_test.zarr").rain.sel(init_time=init, lead_day=lead).load()
    pr = xr.open_zarr(PROD / "prob_test.zarr").sel(init_time=init, lead_day=lead).load()
    fcl = pd.read_csv(FC_LABELS, parse_dates=["init_time"])
    reg = fcl[(fcl.init_time == pd.Timestamp(init)) & (fcl.lead == lead)].fc_regime.iloc[0]

    gdf = gdf.reset_index(drop=True)
    mask = regionmask.mask_geopandas(gdf, rain.lon, rain.lat).values
    r, p64, p115 = rain.values, pr.p64.values, pr.p115.values
    out = []
    for i, name in enumerate(gdf[name_col]):
        m = (mask == i) & np.isfinite(r)
        if not m.any():  # district smaller than one 0.25 deg cell centre
            continue
        out.append({"district": name, "cells": int(m.sum()), "rain_mean": r[m].mean(), "rain_max": r[m].max(),
                    "area_frac_ge64.5": (r[m] >= 64.5).mean(),
                    "p_heavy_mean": p64[m].mean(), "p_heavy_max": p64[m].max(),
                    "p_vheavy_max": p115[m].max(), "regime": reg, "colour": colour(r[m].max())})
    return pd.DataFrame(out).round(3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--districts", required=True)
    ap.add_argument("--name-col", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--lead", type=int, default=1)
    a = ap.parse_args()
    gdf = gpd.read_file(a.districts).to_crs(4326)
    t = district_table(gdf, a.name_col, a.init, a.lead)
    out = PROD / f"district_{a.init}_L{a.lead}.csv"
    t.to_csv(out, index=False)
    print(t.sort_values("rain_max", ascending=False).head(15).to_string(index=False))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()