"""Leave-one-year-out prob runs over all available years, then pool.
python -m svara.run_folds
"""
import subprocess
import sys

YEARS = list(range(2015, 2020))
for i, test in enumerate(YEARS):
    val = YEARS[(i + 1) % len(YEARS)]
    train = [y for y in YEARS if y not in (test, val)]
    cmd = [sys.executable, "-m", "svara.prob", "--train-years", *map(str, train),
           "--val-years", str(val), "--test-years", str(test), "--tag", str(test)]
    print("\n>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

subprocess.run([sys.executable, "-m", "svara.pool_prob"], check=True)