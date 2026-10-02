from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
IMD_RAW = DATA / "raw" / "imd"
GEFS_TMP = DATA / "interim" / "gefs"
ZARR = DATA / "zarr"
GEFS_ZARR = ZARR / "gefs_apcp.zarr"
IMD_ZARR = ZARR / "imd_rain.zarr"

# Canonical IMD 0.25 deg grid (129 lat x 135 lon). GEFS 0.25 deg points coincide with it.
LAT = 6.5 + 0.25 * np.arange(129)
LON = 66.5 + 0.25 * np.arange(135)

MEMBERS = ("c00", "p01", "p02", "p03", "p04")  # the 5 members present on every reforecast day
MAX_LEAD = 5

# One 00 UTC init per day, June 1 - Sept 30.
SEASON = ("06-01", "09-30")

BUCKET = "noaa-gefs-retrospective"
KEY = "GEFSv12/reforecast/{y}/{ymd}00/{mem}/Days:1-10/apcp_sfc_{ymd}00_{mem}.grib2"

# "end":   IMD value labelled D = 03 UTC (D-1) -> 03 UTC D  (lead d is labelled init + d days)
# "start": IMD value labelled D = 03 UTC D -> 03 UTC (D+1)  (lead d is labelled init + d - 1 days)
# Confirm with `python -m svara.check_alignment`.
IMD_LABEL = "end"