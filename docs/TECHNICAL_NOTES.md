# Technical Notes

This file contains additional technical clarifications that complement 
the published paper.

---

## 1. Ocean Mask vs. Full-Grid POD Basis Representation

### Summary

| Representation | Grid points per variable | Variables | Total columns |
|---|---|---|---|
| Masked (active ocean) | 3,505 | 5 | **17,525** |
| Full grid (61 × 141) | 8,601 | 5 | **43,005** |
| Difference (non-ocean) | 5,096 | 5 | **25,324** |

### Details

The active ocean mask, derived exclusively from the training period 
(January 1979 – December 2010), contains **3,505 grid points**, yielding 
**N = 5 × 3,505 = 17,525** spatial degrees of freedom in the standardized 
state matrix `X_train`.

The POD basis distributed with this repository (`data/pod_Vt.npy`) retains 
the **full 61 × 141 grid** (8,601 points per variable, **43,005 columns** 
in total), with non-ocean points assigned **zero loadings**. This choice 
preserves the full grid geometry, which makes it straightforward to reshape 
the spatial modes back into 2D maps (61 × 141) for visualization and 
reconstruction without needing a separate mask file.

### Algebraic equivalence

The two representations are **algebraically equivalent** for projection 
onto the first 20 modes:

    A_test = X_test @ V20.T

Since the extra columns of `V20` corresponding to non-ocean points are 
exactly zero, they contribute nothing to the projection. Therefore the 
resulting POD coefficients — and all downstream LSTM predictions — are 
identical whether one uses the masked (17,525-column) or the full-grid 
(43,005-column) representation.

### Why this note exists

The paper reports **17,525 spatial degrees of freedom** (Section 3.3), 
while the distributed POD basis file has **43,005 columns**. A reader 
inspecting both might wonder whether these are consistent. They are — 
the difference is purely representational:

- The paper's count (17,525) reflects the **active ocean degrees of freedom** 
  actually used in the SVD.
- The distributed file (43,005 columns) preserves the **full grid geometry** 
  for easier visualization.

### Reference

See Section 3.3 of the paper for the original definition of the state 
matrix and the POD decomposition.

---

## 2. Variance explained by retained modes

For reference, the cumulative variance explained by the leading POD modes 
(reported in Figure 2 and Section 4.1 of the paper):

| Modes retained | Cumulative variance |
|---|---|
| 3 | 51.3% |
| 14 | 75.2% |
| 20 | 79.6% |
| 52 | 90.0% |

These values are reproduced exactly by `data/pod_variance_ratio.npy`.
---

## 3. Reproducibility and ERA5 Data Versioning

The figures and tables in the paper were generated from the preprocessed 
arrays in `data/`, which were produced from the ERA5 version available 
at the time of submission.

If you re-run the full pipeline from scratch (`01_download_era5_data.py` → 
`02_preprocess_era5_data.py` → `03_pod_decomposition.py`), the results may 
differ slightly, because:

1. **ERA5 is periodically updated** by ECMWF (new data added, occasional 
   revisions to existing values).
2. The downloaded data will therefore not be byte-identical to what was 
   used in the paper.

**To reproduce the paper exactly**, use the preprocessed arrays in `data/` 
and skip `01_download_era5_data.py`. The `run_all.sh` script is configured 
to do this by default.

| Scenario | Expected result |
|---|---|
| Use provided `data/` arrays | Figures and tables match the paper |
| Re-download ERA5 and re-run full pipeline | Figures may differ slightly (ERA5 versioning) |
This issue documents two clarifications that may be helpful to readers and reviewers.

---

## 1. Ocean mask vs. full-grid POD basis

The paper reports **17,525 spatial degrees of freedom** (5 variables × 3,505 ocean grid points).

The distributed POD basis (`data/pod_Vt.npy`) has **43,005 columns** (5 × 8,601 full-grid points).

These are **algebraically equivalent** for projection onto the first 20 modes, since non-ocean points have zero loadings in `V20`.

See [docs/TECHNICAL_NOTES.md](docs/TECHNICAL_NOTES.md) and the "Note on dimensionality" section in [README.md](README.md) for details.

---

## 2. Reproducibility and ERA5 data versioning

If you **re-run the full pipeline from scratch** (`01_download_era5_data.py` → `02_preprocess_era5_data.py` → `03_pod_decomposition.py`), the figures and tables you obtain may differ **slightly** from those in the paper. This is expected, for two reasons:

1. **ERA5 is periodically updated** by ECMWF. New reanalysis data are added, and (rarely) existing values are revised. Re-downloading ERA5 today will therefore not yield byte-identical data to what was used in the paper.

2. **The paper's figures and tables were generated from the preprocessed arrays in `data/`**, not from a fresh ERA5 download. Those arrays were produced from the ERA5 version available at the time of submission.

**To reproduce the paper exactly**, do **not** re-run `01_download_era5_data.py`. Instead, use the preprocessed data already provided:

    data/X_train.npy
    data/X_val.npy
    data/X_test.npy
    data/V20.npy
    data/ocean_mask.npy

Then run the downstream scripts:

    python src/03_pod_decomposition.py    # uses data/ if present
    python src/04_train_pod_lstm.py
    python src/10_make_fig1_study_area.py

The `run_all.sh` script is configured to use the preprocessed data in `data/` and skip the ERA5 download step unless explicitly requested.

---

## Summary

| Scenario | Expected result |
|---|---|
| Use provided `data/` arrays | Figures and tables match the paper |
| Re-download ERA5 and re-run full pipeline | Figures may differ slightly (ERA5 versioning) |
| Use masked (17,525) or full-grid (43,005) POD basis | Identical results (algebraically equivalent) |
