#!/usr/bin/env bash
# Reproduce the full POD-LSTM pipeline.
# Assumes the environment is already activated and data/ is populated.

set -e

cd src

echo "Preprocessing (skip if data/ is already populated)..."
# python 01_download_era5_data.py
# python 02_preprocess_era5_data.py
python 03_pod_decomposition.py

echo "Training..."
python 04_train_pod_lstm.py
python 05_train_baselines.py
python 06_ablation_multiseed.py

echo "Figures..."
for f in 10 11 12 13 14 15 16 17 18; do
    python ${f}_*.py
done

echo "Tables and statistics..."
for f in 20 21 22 23; do
    python ${f}_*.py
done

echo "Done. Outputs in results/ and figures/."
