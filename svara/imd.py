import time
from pathlib import Path

import imdlib as imd
import requests

from . import config as C

URL = "https://imdpune.gov.in/cmpg/Griddata/rainfall.php"  # same endpoint imdlib uses
BYTES_PER_DAY = 129 * 135 * 4  # float32 grid


def _is_leap(y: int) -> bool:
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


def _download_year(year: int, dest: Path, tries: int = 5) -> None:
    expected = (366 if _is_leap(year) else 365) * BYTES_PER_DAY
    if dest.exists() and dest.stat().st_size == expected:
        return
    tmp = dest.with_suffix(".part")
    for i in range(tries):
        try:
            with requests.post(URL, data={"rain": year}, stream=True, timeout=(30, 180)) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if tmp.stat().st_size != expected:
                raise IOError(f"got {tmp.stat().st_size} bytes, expected {expected}")
            tmp.rename(dest)  # a finished file is never partial
            return
        except (requests.RequestException, IOError) as e:
            print(f"IMD {year}: attempt {i + 1}/{tries} failed: {e}")
            tmp.unlink(missing_ok=True)
            if i == tries - 1:
                raise
            time.sleep(5 * 2**i)


def build_truth(y0: int, y1: int) -> None:
    """Download IMD 0.25 deg daily rain for y0..y1 and store on the canonical grid as Zarr."""
    rain_dir = C.IMD_RAW / "rain"
    rain_dir.mkdir(parents=True, exist_ok=True)
    for y in range(y0, y1 + 1):
        _download_year(y, rain_dir / f"{y}.grd")

    ds = imd.open_data("rain", y0, y1, "yearwise", file_dir=str(C.IMD_RAW)).get_xarray()
    ds = ds.rename({k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items() if k in ds.dims})
    da = ds["rain"].sortby("lat").where(lambda x: x >= 0)  # -999 / NaN -> NaN (off-land)
    da = da.reindex(lat=C.LAT, lon=C.LON, method="nearest", tolerance=1e-2)
    if da.isnull().all("time").all():
        raise RuntimeError("IMD grid did not match the canonical grid; inspect imdlib output")
    out = da.astype("float32").rename("obs").to_dataset().chunk({"time": 366})
    C.ZARR.mkdir(parents=True, exist_ok=True)
    out.to_zarr(C.IMD_ZARR, mode="w")