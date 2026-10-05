# SVARA-Rain: Regime-Aware AI Post-Processing of Monsoon Rainfall Forecasts

**Smart India Hackathon 2026** | Problem Statement **SIH26080** | Theme: Smart Automation | Category: Software | Team **SVARA** 

> Raw ensemble rainfall forecasts are biased, and the bias is **not the same in every weather situation**. SVARA first identifies the monsoon regime (active, break, ...), then corrects rainfall **separately for each regime**, instead of applying one global correction.

---

## 1. The idea in one minute

| Step | What happens |
|---|---|
| 1. Identify the regime | Classify each forecast day as *active*, *break* or *normal* monsoon (later also monsoon low, orographic, coastal) |
| 2. Test before claiming | A **gate test** checks, with confidence intervals, whether raw forecast bias really differs by regime. We only build regime-specific correction if the evidence supports it |
| 3. Correct per regime | Quantile mapping per regime, then LightGBM conditioned on regime probabilities |
| 4. Calibrated probabilities | Heavy (>= 64.5 mm) and very heavy (>= 115.6 mm) rain, using IMD's own categories |
| 5. Deliver | District-level table and map, plus a verification report (RMSE, ETS, CSI, POD, FAR, FSS), split by regime, lead day and threshold |

**Our commitment is honest reporting.** Every score is shown against the raw forecast and a global correction. If regime-awareness does not help, we say so.

---

## 2. Project status (what is built vs. planned)

This repository is a **working prototype of the data and evidence layers**. The modelling and dashboard layers are **not built yet**.

| Stage | Status | Notes |
|---|---|---|
| IMD 0.25 deg truth data | Done | Daily rain 2010-2019, stored as Zarr on the 129 x 135 grid |
| GEFSv12 reforecast download | Done | Lead days 1-5, 5 members, summed to IMD rain days (03 to 03 UTC) |
| Time-alignment check | Done | Verified with data, see Section 4 |
| Day-level regime labels (active / break / normal) | Done | Rajeevan-style, from core-monsoon-zone rainfall anomaly |
| Gate test (does bias differ by regime?) | Done (code) | Result on current sample: **inconclusive**, larger run in progress |
| Regime-targeted larger download (2015-2019) | In progress | ~261 init dates chosen by regime |
| Monsoon low / depression labels (ERA5) | Planned | Needs Copernicus CDS access |
| Orographic and coastal grid-level regimes | Planned | Needs terrain data |
| Regime classifier (forecast fields only) | Planned | |
| Per-regime quantile mapping, LightGBM | Planned | |
| Calibrated heavy / very heavy probabilities | Planned | |
| District product, verification report | Planned | |
| FastAPI and Streamlit dashboard | Planned | |

---

## 3. Repository layout

```
svara/
  config.py          paths, grid, ensemble members, season, IMD day-label convention
  imd.py             download IMD 0.25 deg daily rainfall -> data/zarr/imd_rain.zarr
  gefs.py            download GEFSv12 reforecast rainfall (AWS), decode GRIB2, build IMD-day totals
  align.py           pairing forecasts with observations on the right valid day
  build_data.py      command-line entry point for the data steps
  check_alignment.py tests which day shift makes forecast and observation agree best
  regimes.py         day-level active / break / normal labels
  gate.py            gate test: raw bias by regime, block-bootstrap confidence intervals
  plan_dates.py      chooses which forecast start dates to download (regime-targeted)
requirements.txt
```

---

## 4. Methods and findings so far

### Data (all public, no licence cost)
- **Forecasts:** NOAA GEFSv12 **reforecast** (2000-2019), 5 ensemble members, lead days 1-5, via the AWS open-data bucket.
- **Truth:** IMD 0.25 deg gridded daily rainfall, via `imdlib`.
- **Regime labels (planned extension):** ERA5.

### Aligning forecasts to the IMD rain day
GEFS rainfall comes as overlapping 3-hourly and 6-hourly accumulation windows. We recover clean 3-hourly amounts (6 h window minus the 3 h window for the second half of each block) and sum them into **03 UTC to 03 UTC** days. A coverage check requires exactly 24 hours in every lead day, so a misalignment fails loudly instead of silently.

### Verifying the day convention
`check_alignment.py` correlates forecast anomalies with IMD at several day shifts. On 15 start dates the correlation peaks at lag 1 (0.29, versus 0.13 at lag 0 and lag 2), confirming `IMD_LABEL = "end"`.

### Regime labels
Days are labelled from the standardised, 15-day-smoothed rainfall anomaly over the core monsoon zone (18-28 N, 65-88 E): **active** if the anomaly is >= +1 standard deviation for 3 or more consecutive days, **break** if <= -1 for 3 or more. Over 2010-2019 this gives about 8% active, 11% break and 80% normal days.

### Gate test
`gate.py` computes the raw forecast bias for each regime and lead day, with circular block-bootstrap 95% intervals, and checks whether each regime differs from normal days. On the current small sample the result is **inconclusive**: a "break days lean wet" signal rests on essentially one 12-day spell, and active days show no signal. We are not drawing conclusions from this yet.

### One observation so far
Lead-day 1 forecasts are about 8-12% wetter per day than later leads, with no spin-up spike. This looks like a real model property, so correction models will take lead day as an input.

---

## 5. Quick start

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. IMD truth
python -m svara.build_data --years 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 --steps imd

# 2. Regime labels (needs >= 10 seasons of IMD data)
python -m svara.regimes

# 3. A small GEFS sample, then verify the time alignment
python -m svara.build_data --years 2019 --steps gefs zarr --limit 12 --workers 6
python -m svara.check_alignment

# 4. Regime-targeted download of the dates that matter, then the gate test
python -m svara.plan_dates --years 2015 2019
python -m svara.build_data --years 2015 2016 2017 2018 2019 --steps gefs zarr \
       --dates-file data/labels/init_plan.txt --workers 6
python -m svara.gate
```

Downloads resume safely: finished start dates are skipped. The regime-targeted run takes a few hours.

---

## 6. Design choices and limitations (stated openly)

- **Regime labels are proxies, not official IMD records.** The thresholds above are documented and can be changed.
- **Pooled standard deviation** over all June-September days is used for the anomaly, which probably under-detects June spells. A per-month alternative is available if needed.
- **Targeted sampling:** the download keeps every active and break day but only every 4th normal day. Each regime's bias is measured only on its own days, so this does not bias the regime means; it only widens the interval for the normal group slightly.
- **Rare events:** heavy and very heavy rainfall are rare, so evaluation is noisy. We will report event counts and avoid strong claims on the rarest categories.
- **Reforecast ends in 2019.** Training and validation use the frozen-model reforecast, split by whole years (never random days, to avoid leakage). The 2020 monsoon season is excluded (model transition). Testing on 2021 onward needs the operational GEFS archive, which has a different file layout; an input adapter for it is planned.
- **Western disturbances** fall outside June-September and are handled in the plan by an interaction flag plus a separate winter module (not built yet).
- **Not an official forecast.** Outputs are an AI-assisted post-processing guidance product and not an IMD warning.

---

## 7. Data sources and references

- GEFSv12 reforecast (AWS open data): https://registry.opendata.aws/noaa-gefs-reforecast/
- GEFS operational archive (AWS): https://registry.opendata.aws/noaa-gefs/
- IMD gridded rainfall access: https://github.com/iamsashwata/imdlib
- ERA5 (Copernicus CDS): https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
- Rajeevan, Gadgil & Bhate (2010), *Active and break spells of the Indian summer monsoon*, J. Earth System Science
- Rajeevan, Bhate, Kale & Lal (2006), *High resolution daily gridded rainfall data for the Indian region*, Current Science
- Roberts & Lean (2008), *Scale-selective verification of rainfall accumulations*, Monthly Weather Review (basis for FSS)

Data are used under their open-data / research-use terms, with attribution.

---

**Team SVARA** | SIH26080 