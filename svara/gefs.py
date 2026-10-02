from __future__ import annotations
import logging
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from eccodes import codes_get, codes_get_array, codes_get_values, codes_new_from_message, codes_release

from . import config as C

log = logging.getLogger(__name__)
_FS = s3fs.S3FileSystem(anon=True)
_WIN = re.compile(r"(\d+)-(\d+)\s*hour\s*acc")
_HORIZON = 24 * C.MAX_LEAD + 3  # last hour needed: end of lead-day 5 IMD window


def season_inits(year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-{C.SEASON[0]}", f"{year}-{C.SEASON[1]}")


def _retry(fn, tries: int = 4):
    for i in range(tries):
        try:
            return fn()
        except FileNotFoundError:
            raise
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2**i)


def _byte_ranges(idx_text: str) -> list[tuple[int, int | None]]:
    """(start, end_exclusive) of every APCP message whose window ends within the horizon."""
    offs, wins = [], []
    for ln in idx_text.splitlines():
        parts = ln.split(":")
        if len(parts) < 6:
            continue
        offs.append(int(parts[1]))
        m = _WIN.search(ln)
        wins.append((int(m[1]), int(m[2])) if (m and parts[3] == "APCP") else None)
    ends = offs[1:] + [None]
    return [(o, e) for o, e, w in zip(offs, ends, wins) if w and w[0] < _HORIZON]

def _to_lead_days(msgs: list[tuple[int, int, np.ndarray]]) -> np.ndarray:
    """Turn accumulation messages into IMD-day (03-03 UTC) totals for lead days 1..MAX_LEAD."""
    msgs.sort(key=lambda m: m[1])
    if len(msgs) > 1 and all(s == 0 for s, _, _ in msgs):  # cumulative from 0 -> de-accumulate
        buckets, prev, prev_t = [], None, 0
        for _, t, f in msgs:
            buckets.append((prev_t, t, f if prev is None else np.maximum(f - prev, 0)))
            prev, prev_t = f, t
    else:
        # Each 6h block has a 3h window (first half) and a 6h window (whole block).
        # Recover the second half as 6h minus 3h so we get clean 3-hourly buckets.
        by = {(s, t): b for s, t, b in msgs}
        buckets = []
        for (s, t), b in sorted(by.items()):
            if t - s == 3 and s % 6 == 0:
                buckets.append((s, t, b))
                six = by.get((s, s + 6))
                if six is not None:
                    buckets.append((s + 3, s + 6, np.maximum(six - b, 0)))
            elif t - s == 6 and (s, s + 3) not in by:
                buckets.append((s, t, b))  # no 3h partner: keep as-is
        buckets.sort(key=lambda m: m[1])

    out = np.zeros((C.MAX_LEAD, *buckets[0][2].shape), np.float32)
    hours = np.zeros(C.MAX_LEAD, int)
    for s, t, b in buckets:
        if t <= 3 or t <= s:  # entirely before the first 03 UTC boundary
            continue
        dur = t - s
        cur = max(s, 3)
        while cur < t:
            d = (cur - 3) // 24
            if d >= C.MAX_LEAD:
                break
            end = min(t, 3 + 24 * (d + 1))
            out[d] += b * ((end - cur) / dur)  # split straddling windows proportionally
            hours[d] += end - cur
            cur = end
    if not (hours == 24).all():
        raise ValueError(f"incomplete IMD-day coverage (hours per lead day: {hours.tolist()})")
    return out


def _member_daily(init: pd.Timestamp, mem: str) -> np.ndarray:
    path = f"{C.BUCKET}/" + C.KEY.format(y=init.year, ymd=init.strftime("%Y%m%d"), mem=mem)
    idx = _retry(lambda: _FS.cat_file(path + ".idx")).decode()
    rngs = _byte_ranges(idx)
    bufs = _retry(lambda: _FS.cat_ranges(
        [path] * len(rngs), [s for s, _ in rngs], [e for _, e in rngs], on_error="raise"))
    msgs, ilat, ilon = [], None, None
    for buf in bufs:
        gid = codes_new_from_message(buf)
        try:
            ni, nj = codes_get(gid, "Ni"), codes_get(gid, "Nj")
            if ilat is None:  # grid lookup once per file; lat/lon arrays match `values` order
                lats = codes_get_array(gid, "latitudes").reshape(nj, ni)[:, 0]
                lons = codes_get_array(gid, "longitudes").reshape(nj, ni)[0, :] % 360
                ilat = np.abs(lats[:, None] - C.LAT[None, :]).argmin(0)
                ilon = np.abs(lons[:, None] - C.LON[None, :]).argmin(0)
                if np.abs(lats[ilat] - C.LAT).max() > 1e-3 or np.abs(lons[ilon] - C.LON).max() > 1e-3:
                    raise ValueError("GEFS grid does not coincide with the IMD grid")
            vals = codes_get_values(gid).reshape(nj, ni)[np.ix_(ilat, ilon)].astype(np.float32)
            msgs.append((int(codes_get(gid, "startStep")), int(codes_get(gid, "endStep")), vals))
        finally:
            codes_release(gid)
    return _to_lead_days(msgs)


def _fetch_init(init: pd.Timestamp) -> np.ndarray:
    out = np.full((len(C.MEMBERS), C.MAX_LEAD, len(C.LAT), len(C.LON)), np.nan, np.float32)
    for i, mem in enumerate(C.MEMBERS):
        try:
            t0 = time.time()
            out[i] = _member_daily(init, mem)
            log.info("%s %s done in %.0fs", init.date(), mem, time.time() - t0)
        except FileNotFoundError:
            log.warning("missing %s %s", init.date(), mem)
    return out


def _job(init: pd.Timestamp) -> None:
    arr = _fetch_init(init)
    tmp = C.GEFS_TMP / f"{init:%Y%m%d}.tmp.npy"
    np.save(tmp, arr)
    tmp.rename(C.GEFS_TMP / f"{init:%Y%m%d}.npy")  # atomic: a finished file is never partial


def download(years, workers: int = 12, limit: int | None = None) -> None:
    C.GEFS_TMP.mkdir(parents=True, exist_ok=True)
    todo = [d for y in years for d in season_inits(y) if not (C.GEFS_TMP / f"{d:%Y%m%d}.npy").exists()]
    todo = todo[:limit] if limit else todo
    log.info("%d init dates to download", len(todo))
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(_job, d): d for d in todo}
        for n, f in enumerate(as_completed(futs), 1):
            try:
                f.result()
            except Exception:
                log.exception("failed %s", futs[f].date())
            if n % 5 == 0 or n == len(todo):
                log.info("%d/%d", n, len(todo))


def build_zarr(years) -> None:
    """Consolidate per-init .npy files into one Zarr (init_time, member, lead_day, lat, lon)."""
    if C.GEFS_ZARR.exists():
        shutil.rmtree(C.GEFS_ZARR)
    C.ZARR.mkdir(parents=True, exist_ok=True)
    first = True
    for y in years:
        inits = [d for d in season_inits(y) if (C.GEFS_TMP / f"{d:%Y%m%d}.npy").exists()]
        if not inits:
            continue
        arr = np.stack([np.load(C.GEFS_TMP / f"{d:%Y%m%d}.npy") for d in inits])
        ds = xr.Dataset(
            {"apcp": (("init_time", "member", "lead_day", "lat", "lon"), arr, {"units": "mm"})},
            coords={"init_time": inits, "member": list(C.MEMBERS),
                    "lead_day": np.arange(1, C.MAX_LEAD + 1), "lat": C.LAT, "lon": C.LON},
        ).chunk({"init_time": 16})
        if first:
            ds.to_zarr(C.GEFS_ZARR, mode="w")
            first = False
        else:
            ds.to_zarr(C.GEFS_ZARR, mode="a", append_dim="init_time")
