"""pip install fastapi uvicorn ;  uvicorn api:app --reload   then open http://127.0.0.1:8000/docs"""
import glob

import pandas as pd
import xarray as xr
from fastapi import FastAPI, HTTPException

from svara import config as C
from svara.fc_regimes import OUT_LABELS as FC_LABELS

app = FastAPI(title="SVARA-Rain", description="AI-assisted post-processing guidance, not an official IMD forecast.")
PROD = C.DATA / "products"


def _open(year: int) -> xr.Dataset:
    f = PROD / f"prob_test_{year}.zarr"
    if not f.exists():
        raise HTTPException(404, f"no probabilities for {year}")
    return xr.open_zarr(f)


@app.get("/dates")
def dates():
    out = []
    for f in sorted(glob.glob(str(PROD / "prob_test_*.zarr"))):
        out += [str(pd.Timestamp(t).date()) for t in xr.open_zarr(f).init_time.values]
    return {"init_dates": out}


@app.get("/probability")
def probability(init: str, lead: int, lat: float, lon: float):
    t = pd.Timestamp(init)
    ds = _open(t.year)
    try:
        pt = ds.sel(init_time=t, lead_day=lead).sel(lat=lat, lon=lon, method="nearest")
    except KeyError:
        raise HTTPException(404, "init date or lead not available")
    fcl = pd.read_csv(FC_LABELS, parse_dates=["init_time"])
    r = fcl[(fcl.init_time == t) & (fcl.lead == lead)]
    return {"init": init, "lead": lead,
            "grid_lat": float(pt.lat), "grid_lon": float(pt.lon),
            "p_heavy_ge_64.5": float(pt.p64), "p_very_heavy_ge_115.6": float(pt.p115),
            "forecast_regime": None if r.empty else r.iloc[0].fc_regime,
            "regime_z": None if r.empty else float(r.iloc[0].z_f)}