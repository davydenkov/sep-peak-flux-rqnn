# sep-peak-flux-rqnn

Data and code for the paper

> I. Davydenkov, *Peak Proton Flux of Solar Energetic Particle Events from Soft X-Ray
> Flares: The Garcia–Balch Relations Tested against a Recurrent Quantum Neural Network*,
> submitted to **Solar Physics** (Springer).

Every number, table and figure of the paper is regenerated from the files in this
repository, without network access:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh            # retrains all models, about 2 h on 8 cores
FAST=1 bash run_all.sh     # keeps the committed model predictions, about 2 min
```

`NPROC` sets the number of worker processes (default 8). All random seeds are fixed, and
the reference outputs are committed in `results/` and `figures/`, so a re-run can be
compared with them file by file. Quantum circuits are simulated exactly and without noise
on a classical computer; no quantum hardware or quantum library is needed to run the
pipeline.

## What the pipeline does

| Step | Script | Output in `results/` |
|---|---|---|
| 1 | `01_xrs_features.py` | `xrs_features.csv`: integrated flux, peak flux, duration, temperature, emission measure |
| 2 | `02_prior_flare.py` | `prior_flare.csv`: preceding flare of the same region, for the α term of Balch (2008) |
| 3 | `03_peak_models.py` | `peak_predictions.json`: RQNN and classical models, leave one year out and holdout |
| 4 | `04_same_input_models.py` | `rqnn_gb_predictions.json`: the same models on the Balch and Garcia inputs |
| 5 | `05_compare.py` | `scores_loyo.csv`, `scores_chrono.csv`, `paired.csv`, `forecasts_loyo.csv`, `quad.json` |
| 6 | `06_sensitivity.py` | `sensitivity.json`, `residual_physics.csv` |
| 7 | `07_truncated.py` | `truncated.json`: exponent with the 10 pfu truncation, scores by time of maximum |
| 8 | `08_same_inputs.py` | `same_inputs.json`, `forecasts_same_inputs.csv` |
| 9 | `09_shape.py` | printed check of low-order classical laws |
| 10 | `10_noise_models.py` | `noise_predictions.json`: forecasts under input errors |
| 11 | `11_noise_analysis.py` | `noise_scores.csv`, `noise_paired.csv` |
| 12 | `12_figures.py` | `figures/fig1–fig4.{pdf,eps}` |
| 13 | `13_macros_tables.py` | `latex/macros.tex` (every number quoted in the paper) and `latex/table_*.tex` |
| 14 | `14_esm.py` | `esm_events.csv`: Electronic Supplementary Material, one row per event |

## Main results (reference run, 111 events, one year left out)

| Model | MAE (dex) | Bias | RMSE | r |
|---|---|---|---|---|
| Climatology | 0.705 | −0.19 | 0.88 | – |
| Garcia (2004), published | 1.104 | −1.04 | 1.44 | 0.45 |
| Balch (2008), published | 0.636 | −0.38 | 0.83 | 0.49 |
| Fluence law, refitted | 0.603 | −0.01 | 0.75 | 0.46 |
| RQNN, 12 inputs | 0.580 | −0.02 | 0.74 | 0.50 |
| Random forest, 12 inputs | 0.580 | −0.00 | 0.73 | 0.51 |
| RQNN, fluence and connection (2 qubits) | 0.568 | −0.01 | 0.72 | 0.54 |
| Parabolic fluence law, refitted | 0.556 | +0.00 | 0.70 | 0.58 |

The RQNN is significantly better than climatology but not than the fluence law:
−0.056 [−0.134, +0.019] dex from the published Balch relation. It is also not more robust
to input noise, to a missing fluence, or to a wrong flux scale. The parabolic law was
chosen after seeing the residuals, so it needs confirmation on future events.

## Repository layout

```
data/input/events_final.csv      112 proton events with all inputs (cycles 23-25)
data/input/xrs_windows.csv.gz    1-min GOES XRS series around each flare used
data/input/flares_regions.csv    flare catalogue rows of the same active regions
rqnn/                            RQNN simulator and model classes
scripts/                         pipeline steps 01-14
results/                         numerical results and LaTeX tables
figures/                         Fig. 1-4 (PDF and EPS)
provenance/                      scripts that produced data/input (see below)
```

`rqnn/validate_simulator.py` checks the state-vector simulator against PennyLane. It is
not part of the pipeline and needs `pennylane` in addition.

## Flux scale

All X-ray quantities are on the operational GOES scale used before GOES-16, on which the
Balch and Garcia relations were calibrated. Reprocessed and GOES-R fluxes are multiplied
by 0.70 (1–8 Å) and 0.85 (0.5–4 Å). `xrs_windows.csv.gz` keeps the fluxes as archived;
the scaling is applied in `01_xrs_features.py`, and `flux_rescaled_from_true` in the event
table records the decision for each flare.

The HEK entry of the X2.3 flare of 6 June 2000 is a fragment 10 minutes after the true
maximum, so step 1 uses the GOES light curve (14:58, 15:25, and 15:40 UT). The flare of
17 June 2003 falls in a data gap, which leaves 111 of 112 events in the scores.

## Data provenance (requires network and local caches)

`data/input/events_final.csv` was built by `provenance/01_events.py` to `04_features.py`
for the companion study of the onset delay. `provenance/extract_xrs_windows.py` reduced a
local GOES cache (3.6 GB) to `xrs_windows.csv.gz`. These scripts depend on local caches
and are not needed to reproduce the paper.

Public sources:

* NOAA SWPC Solar Proton Events: <https://www.ngdc.noaa.gov/stp/space-weather/interplanetary-data/solar-proton-events/SEP%20page%20code.html>
* GOES XRS 1-min data, NOAA NCEI: <https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/>
* Heliophysics Event Knowledgebase: <https://www.lmsal.com/hek/>
* SOHO/LASCO CME catalogue: <https://cdaw.gsfc.nasa.gov/CME_list/>
* OMNI and ACE EPAM through CDAWeb: <https://cdaweb.gsfc.nasa.gov/>

## Main columns of `events_final.csv`

| Column | Meaning |
|---|---|
| `noaa_id`, `begin`, `tmax`, `peak_pfu` | NOAA event, begin and maximum times, peak >10 MeV flux |
| `t_start, t_peak, t_end`, `hek_class`, `score` | flare times, class, peak flux / 10⁻⁸ W m⁻² |
| `lon_w`, `lat`, `region` | flare location (west positive) and NOAA region |
| `connect` | Parker connection factor |
| `log_flux, log_flu, log_rise, log_dur` | flare inputs; `log_flu` is the triangular proxy |
| `cme_v, cme_w` | CME speed and width, only if within 60 min of the flare peak |
| `DE4_rise` | rise of 175–315 keV electrons in ACE EPAM |
| `imf_align_mean, imf_stability, imf_theta_deg` | field direction in the 6 h before the flare |
| `cycle, year` | solar cycle and calendar year used for the folds |

## Licence

Code: MIT. Derived data and results: CC BY 4.0. External data remain under the terms of
their providers (U.S. Government works).

## Citation

See `CITATION.cff`.
