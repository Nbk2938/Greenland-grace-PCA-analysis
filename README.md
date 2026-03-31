# Greenland GRACE–Temperature Co-Variability

This project analyzes how Greenland's ice-mass anomalies from GRACE relate to regional temperature anomalies. The workflow in `grace_temp_covariability.py` ingests Level-3 GRACE mascon data and temperature fields (GRIB or NetCDF), builds regional time series, and runs PCA/correlation diagnostics.

## Getting the Repository

If you download a ZIP archive everything under `Data/` is already included. When cloning from GitHub you must pull the NetCDF datasets via Git LFS because they exceed the default blob size limit. Use whichever remote style you prefer:

```bash
git lfs install                     # one-time per machine
git clone git@github.com:Nbk2938/Greenland-grace-PCA-analysis.git
cd Greenland-grace-PCA-analysis
git lfs pull                        # ensures the .nc assets are fetched
```

HTTPS alternative:

```bash
git clone https://github.com/Nbk2938/Greenland-grace-PCA-analysis.git
cd Greenland-grace-PCA-analysis
git lfs pull
```

If LFS is not installed the `.nc` files will appear as tiny pointer files and the analysis script will raise "file not found" errors.

## Requirements

- Python 3.10+ (tested with 3.12)
- Access to the required data files under `Data/`
- Packages: `numpy`, `pandas`, `netCDF4`, `xarray`, `cfgrib`, `scipy`, `scikit-learn`, `matplotlib`

## Environment Setup

Run all commands from the project root: `Greenland_grace_PCA_analysis`.

1. **Create a virtual environment**

   ```bash
   python -m venv .venv
   ```

2. **Activate the environment**

   - **Windows (PowerShell)**

     ```powershell
     .venv\Scripts\Activate.ps1
     ```

   - **macOS / Linux (bash/zsh)**

     ```bash
     source .venv/bin/activate
     ```

3. **Upgrade pip inside the venv (optional but recommended)**

   ```bash
   python -m pip install --upgrade pip
   ```

4. **Install project dependencies**

   ```bash
   python -m pip install numpy pandas netCDF4 xarray cfgrib scipy scikit-learn matplotlib
   ```

5. **(Optional) Freeze the environment**

   ```bash
   python -m pip freeze > requirements.txt
   ```

## Running the Analysis

With the environment activated, execute:

```bash
python grace_temp_covariability.py \
    --grace-file Data/GRCTellus.JPL.200204_202512.GLO.RL06.3M.MSCNv04CRI.nc \
    --temp-file Data/temp_data.nc \
    --temp-var t2m
```

Use `--help` to explore additional options such as rolling windows, climatology periods, and pre-analysis plots.

## Notes

- `cfgrib` relies on the ECMWF ecCodes library; the pip wheel bundles it, so no manual install is typically required.
- If `temp_data` ships as GRIB (`.grib`/`.grb`), keep the `--temp-var` argument aligned with the short name (e.g., `t2m`).
- Outputs (plots/tables) are written to `outputs/` by default and the folder is created automatically if it does not exist.
