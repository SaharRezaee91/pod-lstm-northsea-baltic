"""
01_download_era5_data.py — Download ERA5 monthly means for the study domain.

Reads from the Copernicus Climate Data Store (CDS) API. Downloads eight
ERA5 variables (SST, MSLP, sensible/latent heat flux, u10, v10, total
precipitation, evaporation) over the North Sea and Baltic Sea on a
0.25-degree grid for January 1979 – December 2024.

Input : CDS API credentials (~/.cdsapirc)
Output: data/era5_raw_<variable>.nc   (one file per variable)
        data/era5_processed_1979_2024.nc  (merged)
        data/era5_sst_1979_2024.nc        (SST only)

Paper : Section 3.1 (ERA5 reanalysis data)
"""
import os
import cdsapi
import xarray as xr
from tqdm import tqdm
import config

DATASET = "reanalysis-era5-single-levels-monthly-means"
AREA = [config.LAT_MAX, config.LON_MIN, config.LAT_MIN, config.LON_MAX]
VARIABLES = [
    "sea_surface_temperature", "mean_sea_level_pressure",
    "surface_sensible_heat_flux", "surface_latent_heat_flux",
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "total_precipitation", "evaporation",
]


def download_variable(client, var, y0, y1, out_path):
    if os.path.exists(out_path):
        print(f"  [skip] {var}"); return
    years = [str(y) for y in range(y0, y1 + 1)]
    months = [f"{m:02d}" for m in range(1, 13)]
    print(f"  [download] {var}...")
    client.retrieve(DATASET, {
        "product_type": "monthly_averaged_reanalysis",
        "variable": var, "year": years, "month": months,
        "time": "00:00", "area": AREA, "format": "netcdf",
    }, out_path)


def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    client = cdsapi.Client()
    print("=" * 60)
    print(f"ERA5 download: {config.YEAR_START}-{config.YEAR_END}")
    print("=" * 60)
    for var in tqdm(VARIABLES, desc="Downloading"):
        safe = var.replace("/", "_")
        out = os.path.join(config.DATA_DIR, f"era5_raw_{safe}.nc")
        download_variable(client, var, config.YEAR_START, config.YEAR_END, out)
    print("\nMerging...")
    ds_list = [
        xr.open_dataset(os.path.join(config.DATA_DIR, f"era5_raw_{v.replace('/', '_')}.nc"))
        for v in VARIABLES
    ]
    ds = xr.merge(ds_list)
    ds.to_netcdf(config.ERA5_FULL)
    ds[["sst"]].to_netcdf(config.ERA5_SST)
    print(f"Saved: {config.ERA5_FULL}")


if __name__ == "__main__":
    main()
