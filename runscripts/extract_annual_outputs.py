#!/usr/bin/env python3
"""
Extract annual outputs from ELM simulation NetCDF files to a text file.

This script reads NetCDF output files for specified years from an ELM run directory,
calculates annual values for selected variables, and writes them to a text file.
No NetCDF files are copied or created in the output directory.

Usage:
    python extract_annual_outputs.py

Configuration:
    Edit the parameters below to specify:
    - run_directory: Path to the ELM run directory containing .nc files
    - postproc_startyear: First year to extract (inclusive)
    - postproc_endyear: Last year to extract (inclusive)
    - output_directory: Where to save the text file
    - variables_to_extract: List of variables to include in output
"""

import os
import sys
import glob
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from netCDF4 import Dataset

# =============================================================================
# USER CONFIGURATION - Edit these parameters
# =============================================================================

# Input directory containing ELM NetCDF output files
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20251016_CA-Oas_ICB20TRCNPRDCTCBC/run'
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20251016_BR-Sa1_ICB20TRCNPRDCTCBC/run'
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20251017_US-Var_ICB20TRCNPRDCTCBC/run'

### PFT 1
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20251016_US-Ho1_ICB20TRCNPRDCTCBC/run'
#postproc_startyear = 2008
#postproc_endyear = 2014
### PFT 2
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20251008_FI-Hyy_ICB20TRCNPRDCTCBC/run'
#postproc_startyear = 2007
#postproc_endyear = 2014
### PFT 3
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20251019_RU-SkP_ICB20TRCNPRDCTCBC_baseline/run'
#postproc_startyear = 2012
#postproc_endyear = 2014
### PFT 4
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20251016_BR-Sa1_ICB20TRCNPRDCTCBC/run'
#postproc_startyear = 2002
#postproc_endyear = 2011
### PFT 5
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20251017_AU-Tum_ICB20TRCNPRDCTCBC_baseline/run'
#postproc_startyear = 2001
#postproc_endyear = 2014
### PFT 6
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20251018_PA-SPn_ICB20TRCNPRDCTCBC/run'
#postproc_startyear = 2007
#postproc_endyear = 2009
### PFT 7
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20251017_US-MOz_ICB20TRCNPRDCTCBC/run'
#postproc_startyear = 2007
#postproc_endyear = 2014
### PFT 8
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20251016_CA-Oas_ICB20TRCNPRDCTCBC/run'
#postproc_startyear = 2001
#postproc_endyear = 2010
### PFT 9
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20251023_ES-LJu_ICB20TRCNPRDCTCBC_baseline/run'
#postproc_startyear = 2006
#postproc_endyear = 2013
### PFT 10
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20251023_US-SRC_ICB20TRCNPRDCTCBC_baseline/run'
#postproc_startyear = 2008
#postproc_endyear = 2014
### PFT 11
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20251021_RU-Cok_ICB20TRCNPRDCTCBC_baseline/run'
#postproc_startyear = 2003
#postproc_endyear = 2013
### PFT 12
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20251020_US-Atq_ICB20TRCNPRDCTCBC_baseline/run'
#postproc_startyear = 2003
#postproc_endyear = 2008
### PFT 13
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20251017_US-Var_ICB20TRCNPRDCTCBC/run'
#postproc_startyear = 2007
#postproc_endyear = 2014
### PFT 14
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20251020_AU-DaP_ICB20TRCNPRDCTCBC_baseline/run'
#postproc_startyear = 2008
#postproc_endyear = 2013

#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20251017_AU-Tum_ICB20TRCNPRDCTCBC_baseline/run'
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20251018_PA-SPn_ICB20TRCNPRDCTCBC/run'
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20251019_RU-SkP_ICB20TRCNPRDCTCBC_baseline/run'
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20251020_US-Atq_ICB20TRCNPRDCTCBC_baseline/run'
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20251020_AU-DaP_ICB20TRCNPRDCTCBC_baseline/run'
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20251021_RU-Cok_ICB20TRCNPRDCTCBC_baseline/run'
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20251021_US-SRC_ICB20TRCNPRDCTCBC_baseline/run'
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20251021_ES-LJu_ICB20TRCNPRDCTCBC_baseline/run'
#NOTE:ES-LJu assigned as PFT 9 
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20251023_ES-LJu_ICB20TRCNPRDCTCBC_baseline/run'
#NOTE: US-SRC assigned as PFT 10
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20251023_US-SRC_ICB20TRCNPRDCTCBC_baseline/run'
#PFT 1
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20251028_IT-Ren_ICB20TRCNPRDCTCBC_baseline/run'
#PFT 1
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20251029_US-Blo_ICB20TRCNPRDCTCBC_baseline/run'

##########################################
### Validation with optimized parameters
##########################################
#PFT 9+7: IT-Cpz(independent site; GSWP3)
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20260321_IT-Cpz_ICB20TRCNPRDCTCBC_baseline/run'
run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20260321_IT-Cpz_ICB20TRCNPRDCTCBC_optimized/run'
postproc_startyear = 1997
postproc_endyear = 2009


#PFT 14: US-RO4(independent site; CRUJRA)
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20260319_US-RO4_ICB20TRCNPRDCTCBC_baseline/run'
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20260319_US-RO4_ICB20TRCNPRDCTCBC_optimized/run'
#postproc_startyear = 2014
#postproc_endyear = 2023


#PFT 13: US-Var(same site but with CRUJRA)
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20260312_US-Var_ICB20TRCNPRDCTCBC_baseline/run'
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20260312_US-Var_ICB20TRCNPRDCTCBC_optimized/run'
#postproc_startyear = 2015
#postproc_endyear = 2021


#PFT 13: CN-Du2(independent site; GSWP3)
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20260319_CN-Du2_ICB20TRCNPRDCTCBC_baseline/run'
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20260319_CN-Du2_ICB20TRCNPRDCTCBC_optimized/run'
#postproc_startyear = 2006
#postproc_endyear = 2008


# PFT 12: CN-Dan(independent site; GSWP3)
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20260319_CN-Dan_ICB20TRCNPRDCTCBC_baseline/run'
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20260319_CN-Dan_ICB20TRCNPRDCTCBC_optimized/run'
#postproc_startyear = 2004
#postproc_endyear = 2005

# PFT 11: GL-ZaH (independent site; GSWP3)
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20260319_GL-ZaH_ICB20TRCNPRDCTCBC_baseline/run'
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20260319_GL-ZaH_ICB20TRCNPRDCTCBC_optimized/run'
#postproc_startyear = 2000
#postproc_endyear = 2014

#PFT 7: US-MOz(calibration with GSWP3, validation withCRUJRA)
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20260312_US-MOz_ICB20TRCNPRDCTCBC_baseline/run'
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20260311_US-MOz_ICB20TRCNPRDCTCBC_optimized/run'
#postproc_startyear = 2015
#postproc_endyear = 2021

#PFT 7: US-Ha1 (independent site; GSWP3)
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20260309_US-Ha1_ICB20TRCNPRDCTCBC_baseline/run'
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20260309_US-Ha1_ICB20TRCNPRDCTCBC_optimized/run'
#postproc_startyear = 1992
#postproc_endyear = 2014

# PFT 6: ZM-Mon(independent site; GSWP3)
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20260319_ZM-Mon_ICB20TRCNPRDCTCBC_baseline/run'
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20260319_ZM-Mon_ICB20TRCNPRDCTCBC_optimized/run'
#postproc_startyear = 2000
#postproc_endyear = 2009

#PFT 4: GH-Ank(independent site; GSWP3)
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20260319_GH-Ank_ICB20TRCNPRDCTCBC_baseline/run'
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20260319_GH-Ank_ICB20TRCNPRDCTCBC_optimized/run'
#postproc_startyear = 2011
#postproc_endyear = 2014

#PFT 2: RU-Fyo (independent site; GSWP3)
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20260319_RU-Fyo_ICB20TRCNPRDCTCBC_baseline/run'
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20260319_RU-Fyo_ICB20TRCNPRDCTCBC_optimized/run'
#postproc_startyear = 1998
#postproc_endyear = 2014


#PFT 1: US-Ho1(same site; but CRUJRA)
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20260315_US-Ho1_ICB20TRCNPRDCTCBC_baseline/run'
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20260315_US-Ho1_ICB20TRCNPRDCTCBC_optimized/run'
#postproc_startyear = 2015
#postproc_endyear = 2021


#PFT 1: US-Blo(independent site; GSWP3)
#run_directory = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_run/20260306_US-Blo_ICB20TRCNPRDCTCBC/run'
#postproc_startyear = 2000
#postproc_endyear = 2006

# Year range to extract (inclusive)
#postproc_startyear = 2000
#postproc_endyear = 2006

# Output directory for extracted files
output_directory = './extracted_outputs'

# History file number (h0 for monthly, h1 for daily, etc.)
history_num = 0

# Write annual values to text file (no NetCDF output files will be created)
write_annual_txt = True

# Variables to extract for annual output (leave empty to extract all common variables)
# Flux variables (gC/m2/s) will be converted to annual sums (gC/m2/yr)
# State variables will be annual means
variables_to_extract = ['GPP', 'NPP', 'ER', 'NEE','NBP','NEP', 'HR', 'FSH', 'EFLX_LH_TOT',
                        'FPSN', 'TOTSOMC', 'TOTVEGC','TOTECOSYSC']

# =============================================================================
# EXTRACTION FUNCTIONS
# =============================================================================

def get_casename_from_files(run_dir):
    """
    Extract the case name from NetCDF files in the run directory.

    Parameters
    ----------
    run_dir : str
        Path to the run directory

    Returns
    -------
    str
        Case name extracted from file naming pattern
    """
    nc_files = glob.glob(os.path.join(run_dir, '*.elm.h*.nc'))
    if not nc_files:
        raise ValueError(f"No ELM history files found in {run_dir}")

    # Extract casename from first file
    # Pattern: <casename>.elm.h0.YYYY-01-01-00000.nc
    basename = os.path.basename(nc_files[0])
    casename = basename.split('.elm.')[0]
    return casename


def find_annual_files(run_dir, casename, startyear, endyear, hnum=0):
    """
    Find all NetCDF files for the specified year range.

    Parameters
    ----------
    run_dir : str
        Path to the run directory
    casename : str
        ELM case name
    startyear : int
        First year to extract (inclusive)
    endyear : int
        Last year to extract (inclusive)
    hnum : int
        History file number (0 for h0, 1 for h1, etc.)

    Returns
    -------
    list
        Sorted list of NetCDF file paths for the specified years
    """
    files = []
    for year in range(startyear, endyear + 1):
        # Pattern: casename.elm.h0.YYYY-01-01-00000.nc
        pattern = f"{casename}.elm.h{hnum}.{year:04d}-*.nc"
        year_files = glob.glob(os.path.join(run_dir, pattern))

        if not year_files:
            print(f"Warning: No files found for year {year}")
        else:
            files.extend(year_files)

    return sorted(files)


def extract_annual_values_to_txt(nc_file, txt_file, variables=None):
    """
    Extract annual values from NetCDF file and write to text file.

    Parameters
    ----------
    nc_file : str
        Path to the concatenated NetCDF file
    txt_file : str
        Path to output text file
    variables : list of str, optional
        List of variable names to extract. If None, extracts common variables.

    Returns
    -------
    dict
        Dictionary with variable names as keys and annual arrays as values
    """

    print(f"\nExtracting annual values from NetCDF file...")
    print(f"  Input file: {nc_file}")

    # Open NetCDF file
    nc = Dataset(nc_file, 'r')

    # Get time information
    mcdate = nc.variables['mcdate'][:]
    years = mcdate // 10000
    unique_years = np.unique(years)
    nyears = len(unique_years)

    print(f"  Found {len(mcdate)} time steps spanning {nyears} years")
    print(f"  Year range: {unique_years[0]} - {unique_years[-1]}")

    # Determine which variables to extract
    if variables is None or len(variables) == 0:
        # Default common variables
        variables = ['GPP', 'NPP', 'ER', 'NEE', 'HR', 'FSH', 'EFLX_LH_TOT', 'FPSN']

    # Check which variables exist in the file
    available_vars = []
    missing_vars = []
    for var in variables:
        if var in nc.variables:
            available_vars.append(var)
        else:
            missing_vars.append(var)

    if missing_vars:
        print(f"  Warning: Variables not found in file: {missing_vars}")

    print(f"  Extracting {len(available_vars)} variables: {available_vars}")

    # Extract and calculate annual values
    annual_data = {}
    annual_data['Year'] = unique_years

    # Flux variables that need to be summed (gC/m2/s -> annual totals)
    # These are typically flux variables with units like gC/m2/s
    # Note: FPSN is excluded because it has umol/m2s units, handled separately
    flux_vars = ['GPP', 'NPP', 'ER', 'NEE', 'HR', 'AR', 'AGNPP', 'BGNPP',
                 'LITHR', 'SOMHR', 'SOILC_HR', 'LITTERC_HR']

    # Energy flux variables (W/m2)
    energy_flux_vars = ['FSH', 'EFLX_LH_TOT', 'FSH_G', 'FSH_V']

    for var in available_vars:
        var_data = nc.variables[var][:, 0]  # Assuming single grid point (lndgrid=0)
        var_units = nc.variables[var].units if hasattr(nc.variables[var], 'units') else 'unknown'

        annual_values = np.zeros(nyears)

        for i, year in enumerate(unique_years):
            year_mask = (years == year)
            year_data = var_data[year_mask]

            # Determine processing based on units first, then variable type
            if var_units in ['umol/m2s', 'umolC/m2/s']:
                # Photosynthesis flux: umol/m2/s -> umol/m2/day (daily average)
                # Convert to daily flux: multiply by seconds per day
                annual_values[i] = np.mean(year_data) * 86400  # Average umol/m2/s * 86400 s/day
            elif var in flux_vars:
                # Carbon flux: gC/m2/s -> gC/m2/yr
                # Sum over all time steps and convert from per-second to per-year
                # Assuming daily data: multiply by seconds per day
                annual_values[i] = np.sum(year_data) * 86400  # 86400 seconds per day
            elif var in energy_flux_vars:
                # Energy flux: W/m2 (average over year)
                annual_values[i] = np.mean(year_data)
            else:
                # State variable: take annual mean
                annual_values[i] = np.mean(year_data)

        annual_data[var] = annual_values

    nc.close()

    # Write to text file
    print(f"  Writing annual values to: {txt_file}")

    with open(txt_file, 'w') as f:
        # Write header
        f.write("# Annual output values extracted from ELM simulation\n")
        f.write(f"# Source file: {os.path.basename(nc_file)}\n")
        f.write(f"# Years: {unique_years[0]} - {unique_years[-1]}\n")
        f.write("#\n")

        # Write variable info
        f.write("# Variables:\n")
        for var in available_vars:
            var_obj = Dataset(nc_file, 'r').variables[var]
            var_units = var_obj.units if hasattr(var_obj, 'units') else 'unknown'
            var_long = var_obj.long_name if hasattr(var_obj, 'long_name') else var

            if var in flux_vars:
                f.write(f"#   {var}: {var_long} (annual sum, gC/m2/yr)\n")
            elif var in energy_flux_vars:
                f.write(f"#   {var}: {var_long} (annual mean, {var_units})\n")
            elif var_units in ['umol/m2s', 'umolC/m2/s']:
                f.write(f"#   {var}: {var_long} (annual mean daily flux, umol/m2/day)\n")
            else:
                f.write(f"#   {var}: {var_long} (annual mean, {var_units})\n")
            Dataset(nc_file, 'r').close()

        f.write("#\n")

        # Write column headers
        header = "Year"
        for var in available_vars:
            header += f"\t{var}"
        f.write(header + "\n")

        # Write data
        for i in range(nyears):
            line = f"{int(annual_data['Year'][i])}"
            for var in available_vars:
                line += f"\t{annual_data[var][i]:.6f}"
            f.write(line + "\n")

    print(f"  Successfully wrote {nyears} years of data for {len(available_vars)} variables")

    return annual_data


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""

    print("="*70)
    print("ELM Annual Output Extraction Tool")
    print("="*70)

    # Validate inputs
    if not os.path.isdir(run_directory):
        raise ValueError(f"Run directory does not exist: {run_directory}")

    if postproc_endyear < postproc_startyear:
        raise ValueError(f"End year ({postproc_endyear}) must be >= start year ({postproc_startyear})")

    print(f"\nConfiguration:")
    print(f"  Run directory: {run_directory}")
    print(f"  Year range: {postproc_startyear} - {postproc_endyear}")
    print(f"  Output directory: {output_directory}")
    print(f"  History file: h{history_num}")
    print(f"  Variables to extract: {variables_to_extract if variables_to_extract else 'default set'}")

    # Get case name
    casename = get_casename_from_files(run_directory)
    print(f"  Case name: {casename}")

    # Find files for specified years
    print(f"\nSearching for files...")
    annual_files = find_annual_files(run_directory, casename,
                                     postproc_startyear, postproc_endyear,
                                     hnum=history_num)

    if not annual_files:
        raise ValueError(f"No files found for years {postproc_startyear}-{postproc_endyear}")

    print(f"Found {len(annual_files)} files")

    # Create output directory
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Extract annual values to text file
    if write_annual_txt:
        # Create temporary concatenated NetCDF file for processing
        print(f"\nCreating temporary concatenated file for processing...")

        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp_file:
            temp_nc_path = tmp_file.name

        try:
            # Concatenate files temporarily using ncrcat
            cmd = ['ncrcat', '-O', '-o', temp_nc_path] + annual_files
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"  Successfully concatenated {len(annual_files)} files to temporary file")

            # Generate text file name
            basename = os.path.basename(annual_files[0])
            casename_for_file = basename.split('.elm.')[0]
            startyear = int(basename.split('.')[3].split('-')[0])
            endyear = int(os.path.basename(annual_files[-1]).split('.')[3].split('-')[0])
            txt_filename = f"{casename_for_file}_annual_{startyear:04d}-{endyear:04d}.txt"
            txt_filepath = os.path.join(output_directory, txt_filename)

            # Extract and write annual values
            extract_annual_values_to_txt(temp_nc_path, txt_filepath, variables_to_extract)

        except subprocess.CalledProcessError as e:
            print(f"Error during concatenation: {e}")
            print(f"STDERR: {e.stderr}")
            raise
        finally:
            # Clean up temporary NetCDF file
            if os.path.exists(temp_nc_path):
                os.remove(temp_nc_path)
                print(f"\n  Removed temporary NetCDF file")

    # Print summary
    print("\n" + "="*70)
    print("EXTRACTION SUMMARY")
    print("="*70)
    print(f"Number of input files processed: {len(annual_files)}")
    print(f"Year range: {os.path.basename(annual_files[0]).split('.')[3].split('-')[0]} - " +
          f"{os.path.basename(annual_files[-1]).split('.')[3].split('-')[0]}")
    print(f"Output text file: {txt_filepath}")
    if os.path.exists(txt_filepath):
        size_kb = os.path.getsize(txt_filepath) / 1024
        print(f"Text file size: {size_kb:.2f} KB")
    print("="*70 + "\n")

    print("Extraction complete!")
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
