#!/usr/bin/env bash
# Reproduce every number, table and figure of the paper from data/input (no network access).
#   bash run_all.sh          full run, retrains all models (about 2 h on 8 cores)
#   FAST=1 bash run_all.sh   keeps the committed model predictions and repeats everything else (about 2 min)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
mkdir -p results/latex figures
$PY scripts/01_xrs_features.py
$PY scripts/02_prior_flare.py
if [ -z "${FAST:-}" ]; then
  $PY scripts/03_peak_models.py
  $PY scripts/04_same_input_models.py
fi
$PY scripts/05_compare.py
$PY scripts/06_sensitivity.py
$PY scripts/07_truncated.py
$PY scripts/08_same_inputs.py
$PY scripts/09_shape.py
if [ -z "${FAST:-}" ]; then
  $PY scripts/10_noise_models.py
fi
$PY scripts/11_noise_analysis.py
if [ -z "${FAST:-}" ]; then
  $PY scripts/15_regression_vqc.py
fi
if [ -z "${FAST:-}" ]; then
  $PY scripts/18_qnn_claims.py
fi
$PY scripts/16_physics.py
$PY scripts/17_vqc_scores.py
$PY scripts/19_claims_analysis.py
$PY scripts/20_extrapolation.py
$PY scripts/21_early_protons.py
$PY scripts/12_figures.py
$PY scripts/13_macros_tables.py
$PY scripts/14_esm.py
