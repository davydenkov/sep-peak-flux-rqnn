# sep-peak-flux-rqnn

Data and code for the paper

> I. Davydenkov, *Peak Proton Flux of Solar Energetic Particle Events from Soft X-Ray
> Flares: The Garcia–Balch Relations Compared with Quantum Regression*,
> submitted to **Solar Physics** (Springer).

Every number, table and figure of the paper is regenerated from the files in this
repository, without network access:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
FAST=1 bash run_all.sh     # keeps the committed model predictions, a few minutes
bash run_all.sh            # also retrains all models, about 3 h on 8 cores
```

`NPROC` sets the number of worker processes (default 8). All random seeds are fixed. The
reference outputs are committed in `results/` and `figures/`, so a re-run can be compared
with them file by file. `FAST=1` was checked to reproduce the numbers and tables of the
paper. The committed model predictions come from separate runs of the same training
scripts; a single uninterrupted `bash run_all.sh` from a clean copy has not been completed.

The main model is a variational quantum regression circuit (VQR, class `VQR` in
`rqnn/adv_lib.py`). The repository name and the `rqnn` folder come from a staged variant,
which is kept for comparison. Quantum circuits are simulated exactly and without noise on
a classical computer; no quantum hardware or quantum library is needed.

## What the pipeline does

| Step | Script | Output in `results/` |
|---|---|---|
| 1 | `01_xrs_features.py` | `xrs_features.csv`: integrated flux, peak flux, duration, temperature, emission measure |
| 2 | `02_prior_flare.py` | `prior_flare.csv`: preceding flare of the same region, for the α term of Balch (2008) |
| 3 | `03_peak_models.py`, `04_same_input_models.py` | classical models and the staged circuit |
| 4 | `15_regression_vqc.py` | `vqc_predictions.json`: VQR and circuit variants |
| 5 | `05_compare.py` to `09_shape.py` | scores, paired differences, sensitivity, exponent, residuals |
| 6 | `10_noise_models.py`, `11_noise_analysis.py` | forecasts and scores under input errors |
| 7 | `16_physics.py` | events by time of proton maximum and by cycle |
| 8 | `17_vqc_scores.py` | circuit variants |
| 9 | `18_qnn_claims.py`, `19_claims_analysis.py` | extrapolation, small samples, Fourier surrogates, smoothness, gradients |
| 10 | `20_extrapolation.py` | forecasts above the training maximum |
| 11 | `21_early_protons.py` | forecasts updated 1, 3, and 6 h after the proton onset |
| 12 | `12_figures.py`, `13_macros_tables.py`, `14_esm.py` | `figures/`, `latex/macros.tex`, `latex/table_*.tex`, `esm_events.csv` |

## Main results (reference run, 111 events, one year left out)

| Model | MAE (dex) | Bias | RMSE | r |
|---|---|---|---|---|
| Climatology | 0.705 | −0.19 | 0.88 | – |
| Garcia (2004), published | 1.104 | −1.04 | 1.44 | 0.45 |
| Balch (2008), published | 0.636 | −0.38 | 0.83 | 0.49 |
| Fluence law, refitted | 0.603 | −0.01 | 0.75 | 0.46 |
| Random forest, 12 inputs | 0.590 | +0.00 | 0.73 | 0.52 |
| VQR, 12 inputs | 0.584 | −0.01 | 0.75 | 0.48 |
| VQR, fluence and connection | 0.567 | −0.01 | 0.72 | 0.54 |
| Parabolic fluence law, refitted | 0.556 | +0.00 | 0.70 | 0.58 |
| Small classical network, 2 inputs | 0.550 | | | |
| Ridge on four Fourier features | 0.542 | | | 0.59 |

VQR is significantly better than climatology, but its difference from the published Balch
relation is within the uncertainty: −0.069 [−0.142, +0.003] dex with two inputs. Compact
classical models reach the same accuracy. The parabolic law was chosen after seeing the
residuals, so it needs confirmation on future events. Adding the >10 MeV flux observed
3–6 h after the onset lowers the error of events still rising from about 0.6 to 0.4 dex.

## Repository layout

```
data/input/events_final.csv      112 proton events with all inputs (cycles 23-25)
data/input/xrs_windows.csv.gz    1-min GOES XRS series around each flare used
data/input/flares_regions.csv    flare catalogue rows of the same active regions
data/input/omni_early_proton.csv >10 MeV flux 1, 3, 6 h after the NOAA begin (OMNI hourly)
data/input/goes_proton_windows.csv.gz  >10 MeV proxy series from GOES channels
data/input/ephin_early.csv       SOHO/EPHIN proton intensities around the begin time
rqnn/                            RQNN simulator and model classes
scripts/                         pipeline steps 01-21
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

`data/input/events_final.csv` was built by `provenance/01_events.py` to `04_features.py`. `provenance/extract_xrs_windows.py` reduced a
local GOES cache (3.6 GB) to `xrs_windows.csv.gz`, and `extract_protons.py` and
`extract_ephin_early.py` did the same for the proton data. These scripts depend on local caches
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
