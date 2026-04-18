#!/usr/bin/env python
"""
MCMC Calibration for AU-Tum - All 20 Parameters

This calibrates all 20 parameters (17 TAM root + 3 leaf) using the trained surrogates.
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

pkl_file = 'pklfiles/20251031_AU-Tum_ICB20TRCNPRDCTCBC.pkl'

with open(pkl_file, 'rb') as f:
    mycase = pickle.load(f)

print(f"✓ Loaded: {mycase.casename}")
print(f"  Total parameters: {mycase.nparms_ensemble}")
print(f"  Parameters: {mycase.ensemble_parms}")

# ============================================================================
# STEP 3: Set up MCMC configuration
# ============================================================================

print("\n" + "="*80)
print("MCMC CONFIGURATION")
print("="*80)
print(f"\nCalibrating all {mycase.nparms_ensemble} parameters:")
for i, p in enumerate(mycase.ensemble_parms, 1):
    print(f"  {i:2d}. {p}")

# Get parameter bounds from ensemble
samples = mycase.samples
parm_min_orig = np.min(samples, axis=1)
parm_max_orig = np.max(samples, axis=1)

# Widen parameter ranges for leaf parameters, C/N ratios, and longevity
# Use a widening factor (e.g., 2x means the range becomes 2x wider)
widening_factor = 2.0
params_to_widen = [
    # Leaf parameters
    'leafcn', 'slatop', 'flnr',
    # C:N ratios (Transport, Absorptive, Mycorrhizal)
    'froottcn', 'frootacn', 'frootmcn',
    # Longevity (Transport, Absorptive, Mycorrhizal)
    'froott_long', 'froota_long', 'frootm_long'
]

parm_mid = (parm_min_orig + parm_max_orig) / 2
parm_range = parm_max_orig - parm_min_orig

# Initialize with original ranges
parm_min = parm_min_orig.copy()
parm_max = parm_max_orig.copy()

# Apply widening only to specified parameters
for i, pname in enumerate(mycase.ensemble_parms):
    if pname in params_to_widen:
        parm_min[i] = parm_mid[i] - (parm_range[i] * widening_factor / 2)
        parm_max[i] = parm_mid[i] + (parm_range[i] * widening_factor / 2)

        # Ensure non-negative bounds for parameters that must be positive
        if parm_min[i] < 0:
            # Shift range to be non-negative while maintaining total width
            parm_max[i] = parm_max[i] - parm_min[i]
            parm_min[i] = 0.0

# Set initial parameter values (midpoint of ranges)
parms_init = (parm_min + parm_max) / 2

print(f"\nParameter ranges (C/N, longevity, and leaf parameters widened by {widening_factor}x):")
print(f"{'Parameter':15s}   {'Original Range':^25s}   {'New Range':^25s}   {'Status':^10s}")
print("-" * 90)
for i, pname in enumerate(mycase.ensemble_parms):
    orig_str = f"[{parm_min_orig[i]:8.4f}, {parm_max_orig[i]:8.4f}]"
    new_str = f"[{parm_min[i]:8.4f}, {parm_max[i]:8.4f}]"
    status = "WIDENED" if pname in params_to_widen else "original"
    print(f"{pname:15s}   {orig_str:25s}   {new_str:25s}   {status:^10s}")

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
    print(f"  {pname:15s} = {parms_init[i]:10.4f}")
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
    print(f"  {param:15s} = {parms_best[i]:10.4f}")

# Save results
output_dir = f'./UQ_output/{mycase.casename}/MCMC_output'
os.makedirs(output_dir, exist_ok=True)

with open(f'{output_dir}/parms_best_all20.txt', 'w') as f:
    f.write("# Best parameters from MCMC calibration (all 20 parameters)\n\n")
    for i, param in enumerate(mycase.ensemble_parms):
        f.write(f"{param:15s} = {parms_best[i]:12.6f}\n")

print(f"\nResults saved to: UQ_output/{mycase.casename}/MCMC_output/")
print(f"  ├── parms_best_all20.txt        # Best parameters (all 20)")
print(f"  ├── MCMC_chain.txt              # Posterior samples")
print(f"  └── plots/                      # Trace plots and PDFs")

# ============================================================================
# STEP 6: Summary
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"""
All 20 Parameters Calibrated:
  - TAM root parameters: 17
  - Leaf trait parameters: 3
  - Total dimensions: 20

Next steps:
  1. Check MCMC diagnostics in advanced_mcmc_report.txt
  2. Review trace plots in MCMC_output/plots/
  3. Examine parameter correlations
  4. Validate predictions against observations
""")

print("\n✓✓✓ All-parameter calibration completed successfully ✓✓✓")
