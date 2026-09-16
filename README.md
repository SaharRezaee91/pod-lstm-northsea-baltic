# A Multivariate POD-LSTM Reduced-Order Model for Ocean Surface Dynamics in the North Sea and Baltic Sea

This repository contains the complete analysis pipeline for the paper:

> **Rezaee, S., Ardestani, M., Pourroostaei Ardakani, S.**
> *A Multivariate POD-LSTM Reduced-Order Model for Ocean Surface Dynamics in the North Sea and Baltic Sea.*
> Computers & Geosciences (submitted).

## Overview

We introduce a **multivariate Proper Orthogonal Decomposition – Long Short-Term Memory (POD-LSTM)** framework that jointly embeds five coupled surface fields from ERA5 reanalysis (1979-2024) into a single low-dimensional subspace, and predicts their one-month-ahead evolution over the North Sea and Baltic Sea. The five fields are **sea surface temperature**, **mean sea-level pressure**, **surface heat flux**, **surface wind**, and **net precipitation**.

Key results:

- 20 leading POD modes capture **79.6%** of the multivariate variance
- LSTM achieves a mean test RMSE of **0.878** (10-seed: **0.879 +/- 0.001**)
- Statistically indistinguishable from validation-tuned Ridge (RMSE **0.882**, Wilcoxon p = 0.105)
- Significantly better than ARIMA (RMSE **0.923**, Wilcoxon p = 0.041)
- Removing the four auxiliary variables costs **6.5%** RMSE, an order of magnitude larger than the LSTM vs. GRU/Transformer swap (~1%)

## Authors

- **Sahar Rezaee** — s_rezaee@mathdep.iust.ac.ir
- **Mahnaz Ardestani** — mahnaz.ardestani@outlook.com
- **Saeid Pourroostaei Ardakani** (corresponding) — s.pourroostaeiardakani@qmul.ac.uk

## Repository Structure

pod-lstm-northsea-baltic/

- README.md
- requirements.txt
- environment.yml
- .gitignore
- data/
- src/
- results/
- figures/

## Installation

### Option 1: conda (recommended)

    conda env create -f environment.yml
    conda activate pod-lstm

### Option 2: pip

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

### CDS API setup (for ERA5 download)

1. Register at https://cds.climate.copernicus.eu/
2. Set up your API key following the CDS API instructions
3. Create `~/.cdsapirc` with your credentials

## Running the Pipeline

All scripts are numbered in execution order. Run them from inside the `src/` directory.

### Data preparation

    cd src
    python 01_download_era5_data.py       # Download ERA5 (~10 GB, 1-2 h)
    python 02_preprocess_era5_data.py     # Standardize + ocean mask
    python 03_pod_decomposition.py        # SVD on training period only

### Model training

    python 04_train_pod_lstm.py           # Main POD-LSTM (~5 s on CPU)
    python 05_train_baselines.py          # Ridge + ARIMA (a few minutes)
    python 06_ablation_multiseed.py       # GRU / univariate / 10 seeds

### Figures 1-9

    python 10_make_fig1_study_area.py
    python 11_make_fig2_pod_variance.py
    python 12_make_fig3_spatial_modes.py
    python 13_make_fig4_temporal_modes.py
    python 14_make_fig5_training_curves.py
    python 15_make_fig6_predictions.py
    python 16_make_fig7_comparison.py
    python 17_make_fig8_ablation.py
    python 18_make_fig9_reconstruction.py

### Tables and statistics

    python 20_make_table3.py              # Table 3 (RMSE / MAE)
    python 21_make_table_a2.py            # Tables A1, A2 (RMSE + R2/r)
    python 22_significance_tests.py       # Wilcoxon + Diebold-Mariano
    python 23_make_fig_a1_sensitivity.py  # Figure A1

## Figure and Table Mapping

- Figure 1 (study area) - src/10_make_fig1_study_area.py
- Figure 2 (POD variance) - src/11_make_fig2_pod_variance.py
- Figure 3 (spatial modes) - src/12_make_fig3_spatial_modes.py
- Figure 4 (temporal coefficients) - src/13_make_fig4_temporal_modes.py
- Figure 5 (training curves) - src/14_make_fig5_training_curves.py
- Figure 6 (predictions) - src/15_make_fig6_predictions.py
- Figure 7 (model comparison) - src/16_make_fig7_comparison.py
- Figure 8 (ablation study) - src/17_make_fig8_ablation.py
- Figure 9 (spatial reconstruction) - src/18_make_fig9_reconstruction.py
- Table 3 (RMSE / MAE, modes 1-10) - src/20_make_table3.py
- Table A1 (RMSE / MAE, modes 11-20) - src/21_make_table_a2.py
- Table A2 (R2, Pearson r) - src/21_make_table_a2.py
- Figure A1 (sensitivity analysis) - src/23_make_fig_a1_sensitivity.py

## Data

The ERA5 reanalysis data used in this study are publicly available from the
Copernicus Climate Data Store (CDS): https://cds.climate.copernicus.eu/

- Product: ERA5 monthly averaged reanalysis (reanalysis-era5-single-levels-monthly-means)
- Period: January 1979 to December 2024 (552 monthly snapshots)
- Domain: 50N to 65N, 5W to 30E (0.25 x 0.25 deg grid, 61 x 141 points)
- Variables: sst, msl, sshf, slhf, u10, v10, tp, e

## Expected Runtime

- ERA5 download: 1-2 hours (network-dependent)
- Preprocessing: < 1 minute
- POD (SVD): < 10 seconds
- LSTM training (300 epochs, CPU): ~5 seconds
- Ridge / ARIMA baselines: 2-5 minutes
- Ablation (10 seeds x 3 models): 3-5 minutes
- All figures: < 1 minute

Inference at prediction time is under 1 ms per month on standard CPU hardware.

## Citation

If you use this code, please cite:

@article{rezaee2026podlstm,
  title   = {A Multivariate POD-LSTM Reduced-Order Model for Ocean Surface Dynamics in the North Sea and Baltic Sea},
  author  = {Rezaee, Sahar and Ardestani, Mahnaz and Pourroostaei Ardakani, Saeid},
  journal = {Computers and Geosciences},
  year    = {2026},
  note    = {Submitted}
}

## Contact

For questions or issues, please open a GitHub issue or contact:

- Sahar Rezaee — s_rezaee@mathdep.iust.ac.ir