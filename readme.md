# SVARA-Rain: Regime-Aware AI Post-Processing of Monsoon Rainfall Forecasts

**Smart India Hackathon 2026** | Problem Statement **SIH26080** | Theme: Smart Automation | Category: Software | Team **SVARA** (ID 134437)

> Raw ensemble rainfall forecasts are biased, and the bias is not the same in every weather situation. SVARA first identifies the monsoon regime, then corrects rainfall and estimates heavy-rain probabilities using regime information, instead of one global correction. Every score is reported with confidence intervals, against the raw forecast and a global correction.

---

## 1. Summary of results

Results come from the 2015-2019 sample (282 regime-targeted start dates). Correction scores use test years 2018-2019 (126 start dates). Probability scores use five leave-one-year-out folds (282 start dates). All intervals are 95% block-bootstrap intervals over days.

- **The correction beats global quantile mapping.** LightGBM with regime probabilities improves ETS by 0.018 to 0.031 at 15.6 mm/day (all five lead days significant) and by 0.025 to 0.048 at 64.5 mm/day (four of five lead days significant).
- **Heavy-rain frequency bias is closest to 1 with regime information.** At 64.5 mm/day the bias is 1.16, 0.96, 1.08, 1.06 and 1.18 at leads 1-5, compared with 1.42, 1.03, 1.15, 1.22 and 1.39 for the same model without regime information. The change is significant at leads 1 and 5.
- **Heavy-rain ETS also rises slightly with regime information** (+0.007 at lead 4 and +0.010 at lead 5 at 64.5 mm/day, intervals exclude zero). At 15.6 mm/day there is no gain.
- **Calibrated heavy-rain probabilities have skill at every lead.** Brier skill score against climatology is 0.07 to 0.14 at >= 64.5 mm and 0.04 to 0.07 at >= 115.6 mm. The raw ensemble exceedance fraction has negative skill at 64.5 mm at four of five leads.
- **For probabilities, adding regime information is neutral at this stage.** The pooled all-days Brier skill differences are all below 0.01.
- **A forecast-only regime classifier** gives calibrated active / break / normal probabilities (log-loss skill +0.34 over climatology). Break days are identified best; active days are the hardest.

---

## 2. Proposed architecture and build status

| Stage | Description | Status |
|---|---|---|
| 1. Data | GEFSv12 reforecast (5 members, lead days 1-5) as raw NWP; IMD 0.25 deg rainfall as truth; ERA5 only for regime labels | Forecast and IMD data built (2015-2019); ERA5 planned |
| 2. Alignment and features | IMD-day (03-03 UTC) totals on the IMD grid; ensemble mean, spread and exceedance fractions; neighbourhood statistics; location and season | Built. Moisture, CAPE, 850 hPa wind and vorticity, terrain and distance to coast: planned |
| 3. Regime layer | Day level: active, break, monsoon low/depression. Grid level: orographic, coastal, plain. Western-disturbance flag | Active / break / normal built (rule-based labels and a trained forecast-only classifier). Remaining regimes: planned |
| 4. Correction and probabilities | Quantile mapping per regime; LightGBM conditioned on regime probabilities; calibrated probabilities for heavy (>= 64.5 mm) and very heavy (>= 115.6 mm) rain | Built with the first regime layer |
| 5. Outputs | District table and map; verification report; dashboard | Gridded probabilities, verification tables and a replay demo (Streamlit and FastAPI) are built. District product and PDF report: planned |

The model ladder is: raw ensemble mean, global quantile mapping, regime quantile mapping (forecast regime, plus an observed-regime oracle kept only as a non-deployable upper bound), LightGBM, and LightGBM with regime probabilities. Every rung is scored with the same metrics and the same bootstrap.

---

## 3. What makes the evaluation rigorous

1. **Gate test before modelling.** Block-bootstrap confidence intervals check whether raw forecast bias differs by regime (`gate.py`, `fc_regimes.py`).
2. **Forecast-derived regimes.** Grouping by the regime of the observed rainfall makes break days look too wet and active days too dry by construction. Deployable regimes here come from the forecast, which is also what is available in real time.
3. **Weighted sampling.** The download keeps every active and break day but only every 4th normal day. Training and scoring use inverse-selection-probability weights, so the sampling does not bias the results.
4. **No leakage between years.** Splits are by whole years, never random days. Calibration and early stopping use a validation year that is never used for model fitting.
5. **Multiple-comparison awareness.** Every comparison table reports how many results were significant against how many are expected by chance.
6. **Verified time alignment.** A coverage check requires exactly 24 hours in every lead day, so a misalignment fails loudly. `check_alignment.py` confirms the day convention from data: on 15 start dates the forecast-observation correlation peaks at lag 1 (0.29, versus 0.13 at lags 0 and 2).
7. **Frozen-model training data.** The GEFSv12 reforecast keeps the forecast model version fixed across years.
8. **IMD's own heavy-rain categories**, with probabilities kept consistent across thresholds (p(>= 115.6 mm) never exceeds p(>= 64.5 mm)).
9. **Heavy-tail handling.** Heavy-rain cells get extra training weight (3x), and a quantile map fitted on validation predictions restores the heavy tail that regression shrinks.

---

## 4. Results

### 4.1 Regime classifier (forecast fields only)

A logistic-regression classifier predicts the probability of active, break and normal for each forecast day from core-monsoon-zone forecast rainfall statistics, ensemble spread, west-coast and all-India means, forecast anomaly, season and lead. Probabilities are out-of-fold (leave-one-year-out), over 1,380 labelled forecast days.

| | Active | Break | Normal |
|---|---|---|---|
| Recall, classifier (prior-corrected) | 0.68 | 0.84 | 0.71 |
| Recall, threshold rule | 0.66 | 0.60 | 0.85 |
| Precision, classifier (prior-corrected) | 0.31 | 0.51 | 0.93 |
| Precision, threshold rule | 0.43 | 0.62 | 0.89 |
| Mean predicted probability of the true class | 0.30 | 0.57 | 0.84 |

Weighted log-loss is 0.439 for the classifier against 0.664 for climatology (skill +0.339). The classifier and the threshold rule are different operating points: the classifier finds more break days, the rule is more precise on active days. Active days are the hardest regime to identify from forecast fields alone.

### 4.2 Correction skill (test years 2018-2019, 126 start dates, all days)

**Heavy rain, threshold 64.5 mm/day:**

| Lead | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| ETS, raw ensemble mean | 0.11 | 0.07 | 0.05 | 0.04 | 0.03 |
| ETS, global quantile mapping | 0.13 | 0.14 | 0.12 | 0.12 | 0.11 |
| ETS, LightGBM | 0.18 | 0.17 | 0.14 | 0.14 | 0.13 |
| ETS, LightGBM + regime | 0.18 | 0.17 | 0.15 | 0.14 | 0.14 |
| Frequency bias, raw ensemble mean | 0.66 | 0.26 | 0.18 | 0.17 | 0.16 |
| Frequency bias, global quantile mapping | 1.22 | 1.19 | 1.27 | 1.28 | 1.16 |
| Frequency bias, LightGBM | 1.42 | 1.03 | 1.15 | 1.22 | 1.39 |
| Frequency bias, LightGBM + regime | 1.16 | 0.96 | 1.08 | 1.06 | 1.18 |
| FSS (5-cell), raw ensemble mean | 0.39 | 0.28 | 0.20 | 0.16 | 0.12 |
| FSS (5-cell), global quantile mapping | 0.42 | 0.45 | 0.40 | 0.38 | 0.36 |
| FSS (5-cell), LightGBM | 0.48 | 0.47 | 0.39 | 0.38 | 0.35 |
| FSS (5-cell), LightGBM + regime | 0.48 | 0.48 | 0.41 | 0.41 | 0.37 |

**Moderate rain, threshold 15.6 mm/day (ETS):**

| Lead | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Raw ensemble mean | 0.23 | 0.22 | 0.20 | 0.19 | 0.17 |
| Global quantile mapping | 0.24 | 0.22 | 0.20 | 0.19 | 0.17 |
| LightGBM | 0.27 | 0.24 | 0.22 | 0.21 | 0.19 |
| LightGBM + regime | 0.27 | 0.24 | 0.22 | 0.21 | 0.19 |

**LightGBM + regime minus global quantile mapping (ETS difference):**

| Lead | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 15.6 mm/day | +0.031 | +0.021 | +0.024 | +0.025 | +0.018 |
| 64.5 mm/day | +0.048 | +0.026 | +0.025 | +0.025 | +0.027 |

All 15.6 mm/day differences are significant. At 64.5 mm/day every difference is significant except lead 2 (+0.026, interval -0.004 to 0.051). Neighbourhood rainfall features carry most of the model's gain (the 5x5 and 11x11 cell means account for about 68% of gain share).

### 4.3 Effect of regime information on the correction

Comparison of LightGBM + regime against LightGBM (all days):

- **64.5 mm/day, frequency bias:** closer to 1 at every lead except lead 2, where the two are about equally close. The change is significant at lead 1 (|bias - 1| smaller by 0.26) and lead 5 (smaller by 0.21).
- **64.5 mm/day, ETS:** +0.007 at lead 4 and +0.010 at lead 5 (intervals exclude zero); +0.008 at lead 3 (interval -0.001 to 0.016). Lead 1 is -0.007 and lead 2 is +0.003 (not significant).
- **15.6 mm/day:** ETS differences are within +/-0.004; lead 2 is slightly lower (-0.004, interval -0.008 to 0.000).

### 4.4 Calibrated probabilities (five leave-one-year-out folds, 282 start dates, all days)

Brier skill score against climatology, with 95% intervals:

| Threshold | Lead | Raw ensemble fraction | LightGBM | LightGBM + regime |
|---|---|---|---|---|
| >= 64.5 mm | 1 | -0.110 (-0.157, -0.066) | 0.140 (0.113, 0.162) | 0.141 (0.117, 0.163) |
| | 2 | -0.012 (-0.038, 0.012) | 0.112 (0.087, 0.138) | 0.109 (0.083, 0.136) |
| | 3 | -0.026 (-0.048, -0.004) | 0.087 (0.063, 0.109) | 0.086 (0.062, 0.109) |
| | 4 | -0.042 (-0.067, -0.018) | 0.080 (0.055, 0.107) | 0.073 (0.045, 0.100) |
| | 5 | -0.053 (-0.082, -0.026) | 0.075 (0.046, 0.104) | 0.072 (0.041, 0.103) |
| >= 115.6 mm | 1 | not computed | 0.071 (0.033, 0.107) | 0.070 (0.033, 0.104) |
| | 2 | not computed | 0.060 (0.035, 0.084) | 0.066 (0.041, 0.089) |
| | 3 | not computed | 0.042 (0.015, 0.068) | 0.041 (0.014, 0.069) |
| | 4 | not computed | 0.046 (0.017, 0.075) | 0.045 (0.013, 0.073) |
| | 5 | not computed | 0.047 (0.012, 0.082) | 0.047 (0.015, 0.078) |

Both LightGBM variants have skill above climatology at every lead and threshold, with every interval excluding zero. The raw ensemble exceedance fraction (5 members) is worse than climatology at 64.5 mm at four of five leads and indistinguishable from climatology at lead 2.

Adding regime information changed the pooled all-days Brier skill by less than 0.01 at every lead and threshold. Across the regime-split comparisons, 4 of 40 were significant (about 2 expected by chance), so for probabilities the regime layer built so far is neutral. Very heavy rain on active-monsoon days at lead 2 is the one place a positive effect appears (+0.017, interval 0.001 to 0.030); this is a hypothesis that needs more data.

### 4.5 Reliability

Forecast and observed frequency agree closely up to a forecast probability of about 0.2 for both thresholds. Above about 0.3 the probabilities are overconfident (for example, in the 2015 held-out year, forecasts near 0.60 for >= 64.5 mm verified about 0.38). Per-fold reliability tables are written to `data/reports/prob_reliability_<year>.csv`. Treat high probabilities as an upper bound.

---

## 5. Repository layout

```
svara/
  config.py          paths, grid, ensemble members, season, IMD day-label convention
  imd.py             download IMD 0.25 deg daily rainfall -> data/zarr/imd_rain.zarr
  gefs.py            GEFSv12 reforecast download, GRIB2 decode, IMD-day totals
  align.py           pairing forecasts with observations on the right valid day
  build_data.py      command-line entry point for the data steps
  check_alignment.py tests which day shift makes forecast and observation agree best
  regimes.py         observed day-level active / break / normal labels
  fc_regimes.py      forecast-implied regimes and gate test on them
  gate.py            gate test: raw bias by regime, block-bootstrap intervals
  plan_dates.py      regime-targeted choice of forecast start dates
  metrics.py         verification metrics and selection weights
  compare.py         bootstrap comparison between ladder rungs
  qmap.py            quantile-mapping baselines (global, forecast regime, oracle observed regime)
  regime_clf.py      forecast-only regime classifier (logistic regression), out-of-fold probabilities
  lgbm_correct.py    LightGBM correction with and without regime features
  prob.py            calibrated heavy and very heavy probabilities
  pool_prob.py       pooled leave-one-year-out comparison
  run_folds.py       runs all leave-one-year-out folds, then pools
app.py               Streamlit replay demo
api.py               FastAPI service
```

---

## 6. Data

- **Forecasts:** NOAA GEFSv12 reforecast, 5 ensemble members, lead days 1-5 (AWS open data), summed to IMD rain days (03 UTC to 03 UTC).
- **Truth:** IMD 0.25 deg gridded daily rainfall via `imdlib`, stored as Zarr on the 129 x 135 grid (2010-2019).
- **Sample used for results:** 282 regime-targeted start dates in 2015-2019. All scripts use only start dates present in both `data/labels/init_plan.txt` and the forecast Zarr.
- **Observed regime shares (2010-2019):** about 8% active, 11% break, 80% normal days.
- **Regime labels:** days are labelled from the standardised, 15-day-smoothed rainfall anomaly over the core monsoon zone (18-28 N, 65-88 E): active if the anomaly is >= +1 standard deviation for 3 or more consecutive days, break if <= -1 for 3 or more. These are proxies, not official IMD records.

---

## 7. Quick start

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Data
python -m svara.build_data --years 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 --steps imd
python -m svara.regimes
python -m svara.plan_dates --years 2015 2019
python -m svara.build_data --years 2015 2016 2017 2018 2019 --steps gefs zarr \
       --dates-file data/labels/init_plan.txt --workers 6
python -m svara.check_alignment

# Evidence and models
python -m svara.gate
python -m svara.fc_regimes
python -m svara.regime_clf --C 1.0
python -m svara.qmap --train-years 2015 2016 2017 --test-years 2018 2019
python -m svara.lgbm_correct --train-years 2015 2016 --val-years 2017 --test-years 2018 2019
python -m svara.run_folds

# Demo
python -m streamlit run app.py
uvicorn api:app --reload        # then open http://127.0.0.1:8000/docs
```

Downloads resume safely: finished start dates are skipped. Reports are written to `data/reports/`, gridded probabilities to `data/products/`.

---

## 8. Limitations (stated openly)

- **Regime labels are proxies, not official IMD records.** Thresholds are documented and adjustable.
- **The regime layer is the first stage of the proposed architecture.** It covers active / break / normal days, with a classifier built from rainfall-based forecast features. Moisture, CAPE, 850 hPa wind and vorticity, terrain, ERA5 monsoon-low labels, orographic and coastal regimes, and the western-disturbance flag are not yet included, so the results above describe this first stage.
- **Small sample.** Five seasons and 282 start dates. The correction comparison uses a 2-year test (126 start dates). Heavy and very heavy rainfall are rare and strongly correlated in space and time, so intervals are bootstrapped over days in blocks and rare categories should not be over-interpreted.
- **Regime classifier is weak on active days** (recall 0.68 at precision 0.31).
- **Possible small leakage.** Classifier probabilities are out-of-fold across the five years, while the correction and probability models use different splits of the same years.
- **Probabilities above about 0.3 are overconfident.** Isotonic calibration is fitted on a single validation year per fold.
- **Raw ensemble exceedance fraction at 115.6 mm was not computed**, so that reference is missing in the probability table.
- **Reforecast ends in 2019.** All testing is on reforecast years. Testing on 2021 onward needs the operational GEFS archive, which has a different file layout; an input adapter is planned.
- **Western disturbances** (outside June-September) are not handled yet.
- **District product and PDF report are not built.** Gridded probabilities are produced; district aggregation is planned.
- **Not an official forecast.** Outputs are AI-assisted post-processing guidance, not an IMD warning.

---

## 9. Roadmap

1. Extend the sample to 2010-2014 and refresh all results.
2. Add moisture, CAPE, 850 hPa wind and vorticity, terrain and distance-to-coast features.
3. Add ERA5 monsoon-low / depression labels and grid-level orographic and coastal regimes.
4. Add the western-disturbance interaction flag.
5. Per-regime quantile mapping driven by the classifier's probabilities, and a final frequency-bias adjustment of the correction.
6. District aggregation and a PDF verification report with reliability plots and CSI, POD and FAR.
7. Operational GEFS input adapter and out-of-period test.

---

## 10. Data sources and references

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