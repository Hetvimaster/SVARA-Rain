import argparse
import logging

import pandas as pd

from . import gefs, imd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=int, nargs="+", required=True)
    p.add_argument("--steps", nargs="+", default=["imd", "gefs", "zarr"], choices=["imd", "gefs", "zarr"])
    p.add_argument("--dates", nargs=2, metavar=("START", "END"), help="GEFS init dates to download, inclusive")
    p.add_argument("--dates-file", help="text file with one GEFS init date (YYYY-MM-DD) per line")
    p.add_argument("--limit", type=int, help="download only N init dates (smoke test)")
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if "imd" in a.steps:
        imd.build_truth(min(a.years), max(a.years))
    if "gefs" in a.steps:
        dates = None
        if a.dates_file:
            dates = pd.to_datetime(open(a.dates_file).read().split())
        elif a.dates:
            dates = pd.date_range(*a.dates)
        gefs.download(a.years, a.workers, a.limit, dates)
    if "zarr" in a.steps:
        gefs.build_zarr(a.years)


if __name__ == "__main__":
    main()