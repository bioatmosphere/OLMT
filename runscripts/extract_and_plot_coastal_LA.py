#!/usr/bin/env python3
"""
Extract and plot the comparison between the baseline and TAM runs for the
coastal Louisiana regional simulation.

Adapted from extract_annual_outputs.py. The original script assumed a single
grid point (lndgrid=0). This version handles the regional (lat, lon) grid
produced by run_helm.py with runtype='latlon_bbox' (lat_bounds=[28.8, 30.8],
lon_bounds=[-94.0, -88.8]). For each year and variable we compute an
area+landfrac-weighted spatial mean over active land cells, write annual
time series to text files, and generate comparison plots of baseline vs TAM.

Each yearly ELM history file already contains one full year of daily data,
so no ncrcat concatenation step is needed.
"""

import os
import sys
import glob
from pathlib import Path
import numpy as np
from netCDF4 import Dataset
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# USER CONFIGURATION
# =============================================================================

RUN_ROOT = "/scratch/hpcl-cli185/6lw/e3sm_run"

cases = {
    "baseline": f"{RUN_ROOT}/20260422_coastal_LA_ICB20TRCNPRDCTCBC_baseline/run",
    "TAM":      f"{RUN_ROOT}/20260422_coastal_LA_ICB20TRCNPRDCTCBC_TAM/run",
}

postproc_startyear = 1850
postproc_endyear   = 2024

output_directory = "./extracted_outputs/coastal_LA_comparison"
history_num = 0

variables_to_extract = [
    "GPP", "NPP", "ER", "NEE", "NBP", "NEP", "HR",
    "FSH", "EFLX_LH_TOT", "FPSN",
    "TOTSOMC", "TOTVEGC", "TOTECOSYSC",
]

# Per-second fluxes that should be integrated to annual totals (gC/m2/yr).
FLUX_VARS = {
    "GPP", "NPP", "ER", "NEE", "NBP", "NEP", "HR", "AR",
    "AGNPP", "BGNPP", "LITHR", "SOMHR", "SOILC_HR", "LITTERC_HR",
}
ENERGY_FLUX_VARS = {"FSH", "EFLX_LH_TOT", "FSH_G", "FSH_V"}


# =============================================================================
# EXTRACTION
# =============================================================================

def get_casename(run_dir):
    nc_files = glob.glob(os.path.join(run_dir, "*.elm.h*.nc"))
    if not nc_files:
        raise ValueError(f"No ELM history files found in {run_dir}")
    return os.path.basename(nc_files[0]).split(".elm.")[0]


def find_annual_files(run_dir, casename, startyear, endyear, hnum=0):
    pairs = []
    for year in range(startyear, endyear + 1):
        matches = glob.glob(os.path.join(
            run_dir, f"{casename}.elm.h{hnum}.{year:04d}-*.nc"))
        if not matches:
            print(f"  Warning: no files for year {year} in {run_dir}")
            continue
        pairs.append((year, sorted(matches)[0]))
    return pairs


def spatial_weights(nc):
    """Area * landfrac weights for lat/lon grid, masked to land cells."""
    area = np.asarray(nc.variables["area"][:], dtype=float)
    landfrac = np.ma.asarray(nc.variables["landfrac"][:], dtype=float)
    mask = np.ma.getmaskarray(landfrac)
    w = area * landfrac.filled(0.0)
    w[mask] = 0.0
    return w


def reduce_timestep_spatial(field, weights):
    """Collapse one time step of (lat, lon) to a weighted mean scalar."""
    field = np.ma.asarray(field)
    valid = ~np.ma.getmaskarray(field)
    ww = weights * valid
    denom = ww.sum()
    if denom <= 0:
        return np.nan
    arr = field.filled(0.0) if np.ma.is_masked(field) else field
    return float((arr * ww).sum() / denom)


def weighted_spatial_mean_ts(field, weights):
    """Reduce (time, lat, lon) -> (time,) using area-weighted mean."""
    field = np.ma.asarray(field)
    if field.ndim == 2:  # (time, lndgrid)
        return field.mean(axis=1).filled(np.nan) if np.ma.is_masked(field) \
               else field.mean(axis=1)
    valid = ~np.ma.getmaskarray(field)
    w = weights[np.newaxis, :, :]
    ww = w * valid
    denom = ww.sum(axis=(1, 2))
    arr = field.filled(0.0) if np.ma.is_masked(field) else field
    numer = (arr * ww).sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, numer / denom, np.nan)


def annual_value(var, units, year_ts):
    """Convert a per-timestep spatial-mean series into one annual value."""
    if units in ("umol/m2s", "umolC/m2/s"):
        return float(np.nanmean(year_ts) * 86400.0)
    if var in FLUX_VARS:
        return float(np.nansum(year_ts) * 86400.0)
    return float(np.nanmean(year_ts))


def extract_run(run_dir, label, startyear, endyear, variables, outdir):
    casename = get_casename(run_dir)
    print(f"\n[{label}] case = {casename}")
    files = find_annual_files(run_dir, casename, startyear, endyear, history_num)
    if not files:
        raise RuntimeError(f"No files found for {label} in {run_dir}")
    print(f"  Found {len(files)} yearly files")

    years = [p[0] for p in files]
    data = {v: np.full(len(files), np.nan) for v in variables}
    meta = {}
    available = None

    for i, (year, path) in enumerate(files):
        with Dataset(path, "r") as nc:
            weights = spatial_weights(nc)
            if available is None:
                available = [v for v in variables if v in nc.variables]
                missing = [v for v in variables if v not in nc.variables]
                if missing:
                    print(f"  Missing variables: {missing}")
                for v in available:
                    var = nc.variables[v]
                    meta[v] = {
                        "units": getattr(var, "units", "unknown"),
                        "long_name": getattr(var, "long_name", v),
                    }
            for v in available:
                ts = weighted_spatial_mean_ts(nc.variables[v][:], weights)
                data[v][i] = annual_value(v, meta[v]["units"], ts)

    annual = {"Year": np.asarray(years)}
    for v in available:
        annual[v] = data[v]

    txt_path = os.path.join(
        outdir, f"{casename}_annual_{startyear:04d}-{endyear:04d}.txt")
    write_txt(txt_path, annual, meta, casename)
    return annual, meta


def write_txt(path, annual, meta, casename):
    vars_ = [k for k in annual if k != "Year"]
    with open(path, "w") as f:
        f.write("# Annual region-averaged outputs (area*landfrac weighted)\n")
        f.write(f"# Case: {casename}\n")
        f.write(f"# Years: {int(annual['Year'][0])} - {int(annual['Year'][-1])}\n")
        f.write("#\n# Variables:\n")
        for v in vars_:
            kind = ("annual sum, gC/m2/yr" if v in FLUX_VARS
                    else f"annual mean, {meta[v]['units']}")
            f.write(f"#   {v}: {meta[v]['long_name']} ({kind})\n")
        f.write("#\nYear\t" + "\t".join(vars_) + "\n")
        for i, y in enumerate(annual["Year"]):
            row = [str(int(y))] + [f"{annual[v][i]:.6f}" for v in vars_]
            f.write("\t".join(row) + "\n")
    print(f"  Wrote {path}")


# =============================================================================
# PLOTTING
# =============================================================================

def plot_comparison(results, metas, variables, outdir):
    labels = list(results.keys())
    vars_ = [v for v in variables if all(v in results[lbl] for lbl in labels)]
    if not vars_:
        print("No variables shared between runs; skipping plots.")
        return

    ncols = 3
    nrows = int(np.ceil(len(vars_) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows),
                             squeeze=False)
    colors = {"baseline": "tab:blue", "TAM": "tab:red"}

    for ax, v in zip(axes.flat, vars_):
        for lbl in labels:
            ax.plot(results[lbl]["Year"], results[lbl][v],
                    label=lbl, color=colors.get(lbl), lw=1.4)
        units = metas[labels[0]][v]["units"]
        ylabel = "gC/m2/yr" if v in FLUX_VARS else units
        ax.set_title(v)
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes.flat[len(vars_):]:
        ax.axis("off")

    fig.suptitle("Coastal LA: baseline vs TAM (area-weighted annual means)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    multi_path = os.path.join(outdir, "coastal_LA_baseline_vs_TAM.png")
    fig.savefig(multi_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved overview plot: {multi_path}")

    if set(labels) >= {"baseline", "TAM"}:
        diff_path = os.path.join(outdir, "coastal_LA_TAM_minus_baseline.txt")
        yb, yt = results["baseline"]["Year"], results["TAM"]["Year"]
        common = np.intersect1d(yb, yt)
        with open(diff_path, "w") as f:
            f.write("# Mean(TAM - baseline) over overlapping years "
                    f"{int(common[0])}-{int(common[-1])}\n")
            f.write("Variable\tMean_baseline\tMean_TAM\tMean_diff\tPct_diff(%)\n")
            for v in vars_:
                ib = np.isin(yb, common)
                it = np.isin(yt, common)
                mb = float(np.nanmean(results["baseline"][v][ib]))
                mt = float(np.nanmean(results["TAM"][v][it]))
                pct = 100.0 * (mt - mb) / mb if mb != 0 else np.nan
                f.write(f"{v}\t{mb:.6f}\t{mt:.6f}\t{mt-mb:.6f}\t{pct:.3f}\n")
        print(f"Saved diff summary: {diff_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Coastal LA: baseline vs TAM annual extraction & plotting")
    print("=" * 70)

    Path(output_directory).mkdir(parents=True, exist_ok=True)

    results, metas = {}, {}
    for label, run_dir in cases.items():
        if not os.path.isdir(run_dir):
            print(f"Skipping {label}: directory missing ({run_dir})")
            continue
        annual, meta = extract_run(
            run_dir, label, postproc_startyear, postproc_endyear,
            variables_to_extract, output_directory,
        )
        results[label] = annual
        metas[label] = meta

    if len(results) < 2:
        print("\nNeed both baseline and TAM results to plot; aborting plot step.")
        return 1

    plot_comparison(results, metas, variables_to_extract, output_directory)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
