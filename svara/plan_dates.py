"""Pick GEFS init dates: every day whose lead-1 valid date is active/break, plus every Nth normal day."""
import argparse

import pandas as pd

from . import config as C
from .align import valid_offset

LABELS = C.DATA / "labels" / "day_regime.csv"
OUT = C.DATA / "labels" / "init_plan.txt"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=int, nargs=2, required=True, metavar=("FIRST", "LAST"))
    p.add_argument("--normal-every", type=int, default=4)
    a = p.parse_args()

    lab = pd.read_csv(LABELS, parse_dates=["date"]).set_index("date").regime
    lab = lab[(lab.index.year >= a.years[0]) & (lab.index.year <= a.years[1])]
    sel = pd.concat([lab[lab != "normal"], lab[lab == "normal"].iloc[:: a.normal_every]]).sort_index()
    inits = sel.index - valid_offset(1)
    keep = (inits.month >= 6) & (inits.month <= 9)  # inits must fall inside the season
    inits, sel = inits[keep], sel[keep]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(d.strftime("%Y-%m-%d") for d in inits) + "\n")
    print(sel.value_counts().to_string())
    print(f"\n{len(inits)} init dates (~{len(inits) / 72:.1f} h at 72/h) -> {OUT}")


if __name__ == "__main__":
    main()