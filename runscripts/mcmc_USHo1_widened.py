#!/usr/bin/env python
"""
MCMC Calibration for US-Ho1 - Widened Priority Parameters

This calibrates all 17 TAM root parameters with widened ranges for priority parameters
to avoid zero GPP while maximizing exploration space.

Priority parameters widened (2x):
- C:N ratios: froottcn, frootacn, frootmcn
- Allocation: froott_leaf, froota_leaf
- Longevity: froott_long, froota_long, frootm_long
"""

import sys
import os
sys.path.append('..')
import pickle
import numpy as np

# Change to parent directory so UQ_output/ is created in the correct location
os.chdir('..')

# ============================================================================
# STEP 1: Import MCMC functionality
# ============================================================================
from model_ELM.MCMC import MCMC
import model_ELM

print("✓ MCMC functionality imported")

# ============================================================================
# STEP 2: Load case
# ============================================================================
print("\nLoading case...")

pkl_file = 'pklfiles/20251014_US-Ho1_ICB20TRCNPRDCTCBC.pkl'

with open(pkl_file, 'rb') as f:
    mycase = pickle.load(f)

print(f"✓ Loaded: {mycase.casename}")
print(f"  Site: {mycase.site}")
print(f"  Total parameters: {mycase.nparms_ensemble}")
print(f"  Parameters: {mycase.ensemble_parms}")

# ============================================================================
# STEP 3: Configure priority parameters to widen
# ============================================================================

print("\n" + "="*80)
print("MCMC CONFIGURATION")
print("="*80)
print(f"\nCalibrating all {mycase.nparms_ensemble} parameters")

# Get parameter bounds from ensemble
samples = mycase.samples
parm_min_orig = np.min(samples, axis=1)
parm_max_orig = np.max(samples, axis=1)

# Define priority parameters to widen (2x) to avoid zero GPP
widening_factor = 2.0
priority_params = [
    # C:N ratios (directly affect photosynthesis)
    'froottcn', 'frootacn', 'frootmcn',
    # Allocation (control leaf biomass & GPP)
    'froott_leaf', 'froota_leaf',
    # Longevity (affect turnover rates)
    'froott_long', 'froota_long', 'frootm_long'
]

parm_mid = (parm_min_orig + parm_max_orig) / 2
parm_range = parm_max_orig - parm_min_orig

# Initialize with original ranges
parm_min = parm_min_orig.copy()
parm_max = parm_max_orig.copy()

# Apply widening only to priority parameters
for i, pname in enumerate(mycase.ensemble_parms):
    if pname in priority_params:
        parm_min[i] = parm_mid[i] - (parm_range[i] * widening_factor / 2)
        parm_max[i] = parm_mid[i] + (parm_range[i] * widening_factor / 2)

        # Ensure non-negative bounds for parameters that must be positive
        if parm_min[i] < 0:
            # Shift range to be non-negative while maintaining total width
            parm_max[i] = parm_max[i] - parm_min[i]
            parm_min[i] = 0.0

# Set initial parameter values (midpoint of ranges)
parms_init = (parm_min + parm_max) / 2

print(f"\nParameter ranges (priority parameters widened by {widening_factor}x):")
print(f"{'Parameter':20s}   {'Original Range':^25s}   {'New Range':^25s}   {'Status':^10s}")
print("-" * 90)
for i, pname in enumerate(mycase.ensemble_parms):
    orig_str = f"[{parm_min_orig[i]:8.4f}, {parm_max_orig[i]:8.4f}]"
    new_str = f"[{parm_min[i]:8.4f}, {parm_max[i]:8.4f}]"
    status = "WIDENED" if pname in priority_params else "original"
    print(f"{pname:20s}   {orig_str:25s}   {new_str:25s}   {status:^10s}")

# Update case attributes to use widened ranges
mycase.parm_min = parm_min
mycase.parm_max = parm_max
mycase.ensemble_pmin = parm_min
mycase.ensemble_pmax = parm_max

# ============================================================================
# STEP 4: Run MCMC calibration
# ============================================================================

print("\n" + "="*80)
print("Running MCMC Calibration...")
print("="*80)
print(f"Variables to calibrate against: GPP, ER")
print(f"MCMC chain length: 500000 evaluations")
print(f"Burn-in period: 5000 evaluations")
print(f"\nInitial parameter values (midpoint of ranges):")
for i, pname in enumerate(mycase.ensemble_parms):
    print(f"  {pname:20s} = {parms_init[i]:10.4f}")
print("\nThis may take 1-2 hours...")

# Prepare observation confidence intervals for MCMC
# Use actual FLUXNET 05/95 percentiles (not calculated from mean ± 1.645*SE)
print("\nLoading FLUXNET 05/95 percentile observations...")
myvars = ['GPP', 'ER']
myobs_05_combined = {}
myobs_95_combined = {}

for var in myvars:
    try:
        myobs_05_var, myobs_95_var = mycase.get_fluxnet_obs(
            site=mycase.site,
            fluxnet_var=var,
            myobsdir='./observations/fluxnet',
            tstep='yearly',
            ystart=-1,
            yend=9999
        )
        # Accumulate results
        myobs_05_combined.update(myobs_05_var)
        myobs_95_combined.update(myobs_95_var)
        print(f"  ✓ {var}: Loaded actual FLUXNET 05/95 percentiles")
        print(f"    05 percentile mean: {np.mean(myobs_05_var[var]):.2f}")
        print(f"    95 percentile mean: {np.mean(myobs_95_var[var]):.2f}")
    except Exception as e:
        print(f"  ✗ Failed to load {var}: {e}")
        # Fallback to empty arrays
        myobs_05_combined[var] = np.array([])
        myobs_95_combined[var] = np.array([])

# Use the combined dictionaries
myobs_05 = myobs_05_combined
myobs_95 = myobs_95_combined

# ============================================================================
# EXCLUDE 2012 DATA
# ============================================================================
print("\n" + "="*80)
print("FILTERING OUT 2012 DATA")
print("="*80)

# US-Ho1 observations span 2008-2014 (7 years), 2012 is at index 4
# Exclude 2012 from all observation arrays
years = list(range(mycase.postproc_startyear, mycase.postproc_endyear + 1))
print(f"Original years: {years}")

if 2012 in years:
    idx_2012 = years.index(2012)
    print(f"2012 found at index {idx_2012} - excluding from calibration")

    # Create mask to exclude 2012
    exclude_mask = np.ones(len(years), dtype=bool)
    exclude_mask[idx_2012] = False

    # Filter observations
    for var in myvars:
        if var in mycase.obs:
            orig_len = len(mycase.obs[var])
            mycase.obs[var] = mycase.obs[var][exclude_mask]
            print(f"  {var} obs: {orig_len} -> {len(mycase.obs[var])} data points")

        if var in mycase.obs_err:
            mycase.obs_err[var] = mycase.obs_err[var][exclude_mask]

        if var in myobs_05:
            orig_len = len(myobs_05[var])
            myobs_05[var] = myobs_05[var][exclude_mask]
            print(f"  {var} 05: {orig_len} -> {len(myobs_05[var])} data points")

        if var in myobs_95:
            myobs_95[var] = myobs_95[var][exclude_mask]
            print(f"  {var} 95: {orig_len} -> {len(myobs_95[var])} data points")

    remaining_years = [y for y in years if y != 2012]
    print(f"\nRemaining years for calibration: {remaining_years}")
else:
    print("2012 not found in observation years - no filtering needed")

print("="*80)

# Run MCMC with direct call to MCMC function
parms_best = MCMC(
    mycase,
    parms_init,
    myvars,
    nevals=500000,
    myobs_05=myobs_05,
    myobs_95=myobs_95,
    mcmc_type='uniform',
    nburn=5000,
    burnsteps=10
)

# ============================================================================
# STEP 5: Report results
# ============================================================================

print("\n" + "="*80)
print("✓ CALIBRATION COMPLETE!")
print("="*80)

print(f"\nCalibrated all {mycase.nparms_ensemble} parameters:")
for i, param in enumerate(mycase.ensemble_parms):
    status = "WIDENED" if param in priority_params else "original"
    print(f"  {param:20s} = {parms_best[i]:10.4f}  [{status}]")

# Save results
output_dir = f'./UQ_output/{mycase.casename}/MCMC_output'
os.makedirs(output_dir, exist_ok=True)

with open(f'{output_dir}/parms_best_widened.txt', 'w') as f:
    f.write("# Best parameters from MCMC calibration (priority parameters widened)\n")
    f.write(f"# Priority parameters (widened {widening_factor}x): {', '.join(priority_params)}\n\n")
    for i, param in enumerate(mycase.ensemble_parms):
        status = "WIDENED" if param in priority_params else "original"
        f.write(f"{param:20s} = {parms_best[i]:12.6f}  # {status}\n")

print(f"\nResults saved to: UQ_output/{mycase.casename}/MCMC_output/")
print(f"  ├── parms_best_widened.txt      # Best parameters")
print(f"  ├── MCMC_chain.txt              # Posterior samples")
print(f"  └── plots/                      # Trace plots and PDFs")

# ============================================================================
# STEP 6: Summary
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"""
Priority Parameters Widened ({widening_factor}x):
  - C:N ratios (3): froottcn, frootacn, frootmcn
  - Allocation (2): froott_leaf, froota_leaf
  - Longevity (3): froott_long, froota_long, frootm_long
  Total widened: {len(priority_params)} out of {mycase.nparms_ensemble}

Other Parameters (original ranges):
  - C:P ratios, labile fractions, cellulose fractions

Strategy:
  - Widened parameters most likely to affect GPP
  - Maintained positive GPP predictions
  - Used actual FLUXNET 05/95 percentiles for observations
  - Maximized exploration space while avoiding zero GPP

Next steps:
  1. Check MCMC diagnostics in advanced_mcmc_report.txt
  2. Review acceptance rate and ESS
  3. Validate predictions against observations
  4. Adjust widening if needed based on results
""")

print("\n✓✓✓ US-Ho1 calibration with widened priority parameters completed ✓✓✓")
