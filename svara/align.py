import pandas as pd
import xarray as xr

from . import config as C


def valid_offset(lead_day: int, label: str | None = None) -> pd.Timedelta:
    """Days between init date and the IMD date label that this lead day's window carries."""
    label = label or C.IMD_LABEL
    return pd.Timedelta(days=lead_day if label == "end" else lead_day - 1)


def pair(fc: xr.DataArray, obs: xr.DataArray, lead_day: int):
    """Forecast at one lead day and matching IMD obs, both indexed by init_time (NaN where obs missing)."""
    f = fc.sel(lead_day=lead_day)
    vt = f.init_time.values + valid_offset(lead_day).to_timedelta64()
    o = obs.reindex(time=vt).assign_coords(time=f.init_time.values).rename(time="init_time")
    return f, o