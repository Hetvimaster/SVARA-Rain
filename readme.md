# SVARA-Rain: Regime-Aware AI Post-Processing of Monsoon Rainfall Forecasts

**Smart India Hackathon 2026** | Problem Statement **SIH26080** | Theme: Smart Automation | Category: Software | Team **SVARA** (ID 134437)

> Raw ensemble rainfall forecasts are biased, and the bias is not the same in every weather situation. SVARA first identifies the monsoon regime, then corrects rainfall and estimates heavy-rain probabilities with regime information, instead of one global correction. **Every score is reported with confidence intervals, against the raw forecast and a global correction.**

---

## 1. Proposed architecture

| Stage | Description | Status |
|---|---|---|
| 1. Data | GEFSv12 reforecast (5 members, lead days 1-5) as raw NWP; IMD 0.25 deg rainfall as truth; ERA5 only for regime labels | Forecast and IMD data built; ERA5 planned |
| 2. Alignment and features | IMD-day (03-03 UTC) totals on the IMD grid; ensemble mean, spread, exceedance fractions, neighbourhood statistics, location, season | Built. Moisture, CAPE, 850 hPa wind and vorticity, terrain, distance to coast: next |
| 3. Regime layer | Day level: active, break, monsoon low/depression. Grid level: orographic, coastal, plain. Western-disturbance flag | Active / break / normal built (forecast-implied, from core-monsoon-zone anomaly). Remaining regimes and a trained classifier: next |
| 4. Correction and probabilities | Quantile mapping per regime; LightGBM conditioned on regime information; calibrated probabilities for heavy (>= 64.5 mm) and very heavy (>= 115.6 mm) rain | Built with the first regime layer; to be extended with the full regime layer |
| 5. Outputs | District table and map, verification report (RMSE, ETS, CSI, POD, FAR, FSS), dashboard | Gridded probabilities, verification tables, replay demo built; district product and PDF report planned |

The model ladder (raw, global quantile mapping, regime quantile mapping, LightGBM, LightGBM + regime) is scored identically at every rung, so the contribution of each added piece of information is measurable.

---

## 2. What makes the approach rigorous

1. **Gate test before modelling.** Block-bootstrap confidence intervals check whether raw bias differs by regime.
2. **Forecast-derived regimes.** Grouping by the regime of the *observed* rainfall makes break days look too wet and active days too dry by construction. Deployable regimes here come from the forecast, which is also what is available in real time. An observed-regime oracle is kept only as a non-deployable upper bound.
3. **Weighted sampling.** The download keeps every active and break day but every 4th normal day. Training and scoring use inverse-selection-probability weights.
4. **No leakage.** Splits are by whole years. Calibration and early stopping use a validation year never used for fitting.
5. **Multiple-comparison awareness.** Comparison tables report how many results were significant against how many are expected by chance.
6. **Verified time alignment.** A coverage check fails loudly on any gap, and a data-based test confirms the day convention.
7. **Frozen-model training data.** The GEFSv12 reforecast keeps the model version fixed across years.
8. **IMD's own heavy-rain categories**, with probabilities kept consistent across thresholds.

---

## 3. Interim results (first prototype stage)

These come from the first stage, where regime information is limited to two day-level features (forecast core-zone anomaly and an active/break code). Moisture, CAPE, wind, terrain, ERA5-based regimes and a trained regime classifier are not yet included. Numbers will be refreshed after the 2010-2014 extension.

**Spatial-context ML versus global quantile mapping.** At 15.6 mm/day, LightGBM improves ETS by 0.02 to 0.03 over global quantile mapping at all five lead days, with every confidence interval excluding zero (test years 2018-2019). Neighbourhood rainfall features carry most of the model's gain.

**Heavy rain (>= 64.5 mm/day).** Both quantile mapping and LightGBM are far better than the raw ensemble mean, which under-forecasts heavy-rain frequency (ETS at lead 3: raw 0.05, global QM 0.12, LightGBM 0.13, LightGBM + regime 0.14).

**Calibrated probabilities.** Brier skill score against climatology is 0.07 to 0.13 at >= 64.5 mm and 0.02 to 0.06 at >= 115.6 mm. The raw ensemble exceedance fraction has about zero or negative skill at 64.5 mm. Probabilities are well calibrated up to about 0.2 and overconfident above 0.3.

**Regime information at this stage.** Over 261 test start dates in five leave-one-year-out folds, regime-versus-no-regime Brier differences are small and mostly within their confidence intervals (2 of 40 comparisons significant, about 2 expected by chance). Very heavy rain on active-monsoon days at leads 2-3 is the one place a positive signal appears, and it needs more data. We treat the contribution of regime information as **still to be established with the full regime layer and the larger 2010-2019 sample**.

---

## 4. Repository layout

```
svara/
  config.py          paths, grid, ensemble members, season, IMD day-label convention
  imd.py             IMD 0.25 deg daily rainfall -> Zarr
  gefs.py            GEFSv12 reforecast download, GRIB2 decode, IMD-day totals
  align.py           pairing forecasts with observations on the right valid day
  build_data.py      command-line entry point for the data steps
  check_alignment.py tests which day shift makes forecast and observation agree best
  regimes.py         observed day-level active / break / normal labels
  fc_regimes.py      forecast-implied regimes, gate test on them
  gate.py            gate test with block-bootstrap intervals
  plan_dates.py      regime-targeted choice of forecast start dates
  metrics.py         verification metrics, selection weights
  compare.py         bootstrap comparison between ladder rungs
  qmap.py            quantile-mapping baselines (global, forecast-regime, oracle)
  lgbm_correct.py    LightGBM correction with / without regime features
  prob.py            calibrated heavy / very heavy probabilities
  pool_prob.py       pooled leave-one-year-out comparison
  run_folds.py       all leave-one-year-out folds, then pooling
app.py               Streamlit replay demo
api.py               FastAPI service
```

---

## 5. Data

- **Forecasts:** NOAA GEFSv12 reforecast, 5 members, lead days 1-5 (AWS open data).
- **Truth:** IMD 0.25 deg gridded daily rainfall via `imdlib`, 2010-2019.
- **Current sample:** 261 regime-targeted start dates (2015-2019). Extension to 2010-2014 is on the roadmap.
- **Observed regime shares (2010-2019):** about 8% active, 11% break, 80% normal.

---

## 6. Quick start

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m svara.build_data --years 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 --steps imd
python -m svara.regimes
python -m svara.plan_dates --years 2010 2019
python -m svara.build_data --years 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 \
       --steps gefs zarr --dates-file data/labels/init_plan.txt --workers 6

python -m svara.gate
python -m svara.fc_regimes
python -m svara.qmap --train-years 2015 2016 2017 --test-years 2018 2019
python -m svara.lgbm_correct --train-years 2015 2016 --val-years 2017 --test-years 2018 2019
python -m svara.run_folds

python -m streamlit run app.py
uvicorn api:app --reload
```

Downloads resume safely: finished start dates are skipped.

---

## 7. Roadmap

1. Extend the sample to 2010-2019 and refresh all results.
2. Add moisture, CAPE, 850 hPa wind and vorticity, terrain and distance-to-coast features.
3. Add ERA5 monsoon-low / depression labels and grid-level orographic and coastal regimes.
4. Train the regime classifier on forecast fields only and use its probabilities in the correction models.
5. Frequency-bias adjustment of the final correction, and recalibration of probabilities with more validation data.
6. District aggregation and PDF verification report.
7. Operational GEFS input adapter and out-of-period test.

---

## 8. Limitations (stated openly)

- Regime labels are proxies, not official IMD records. Thresholds are documented and adjustable.
- The current prototype implements only the first regime layer (see Section 3), so its results describe that layer and not the full proposed architecture.
- Heavy and very heavy rainfall are rare and strongly correlated in space and time. Intervals are bootstrapped over days in blocks, and rare categories should not be over-interpreted.
- At lead 1 the LightGBM correction over-forecasts 15.6 mm and 64.5 mm events (frequency bias about 1.3-1.5). A frequency-bias adjustment is on the roadmap.
- Probabilities above about 0.3 are overconfident; calibration used a single validation year.
- The reforecast ends in 2019. Operational-era testing needs an input adapter for the GEFS operational archive.
- Western disturbances (outside June-September) are handled by a planned flag and winter module, not yet built.
- **Not an official forecast.** Outputs are AI-assisted post-processing guidance, not an IMD warning.

---

## 9. Data sources and references

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