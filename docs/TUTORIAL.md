# Tutorial: Reproducing Figure 6 from scratch

This tutorial walks through reproducing Figure 6 (POD coefficient predictions over the test period) using the preprocessed data in `data/`.

## Step 1: Set up the environment

    conda env create -f environment.yml
    conda activate pod-lstm

or

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Step 2: Verify the preprocessed data

    ls data/
    # X_train.npy, X_val.npy, X_test.npy, pod_A_train.npy, pod_Vt.npy, ...

If these files are missing, run:

    python src/01_download_era5_data.py
    python src/02_preprocess_era5_data.py
    python src/03_pod_decomposition.py

## Step 3: Train the POD-LSTM

    cd src
    python 04_train_pod_lstm.py

This writes the model prediction and metrics to `results/`.

## Step 4: Generate Figure 6

    python 15_make_fig6_predictions.py

The figure is written to `figures/fig6_predictions.png`.

## Step 5: Check against the paper

Open `figures/fig6_predictions.png` and compare with Figure 6 in the paper. The RMSE values in the panel titles should match Table 3.

## Reproducing the full paper

To reproduce every figure and table in one go:

    ./run_all.sh

See `docs/USER_GUIDE.md` for details on each script.
