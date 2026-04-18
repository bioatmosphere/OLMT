#!/usr/bin/env python
"""
MCMC Calibration: All 17 TAM Root Parameters (Fixing 3 Leaf Parameters)

This calibrates all TAM root parameters while fixing leaf parameters at reasonable values.
Uses surrogates trained on all 20 parameters (17 root + 3 leaf).

Strategy:
- Calibrate: All 17 root parameters (C:N, C:P, longevity, chemistry, allocation)
- Fix: 3 leaf parameters (leafcn, slatop, flnr) at literature values for AU-Tum

Key advantage: Train surrogates once on all 20 parameters, use for root-only calibration.
"""

import sys
import os
sys.path.append('..')
import pickle

# Change to parent directory so UQ_output/ is created in the correct location
os.chdir('..')

# ============================================================================
# STEP 1: Integrate the functionality (do once per session)
# ============================================================================
from model_ELM.MCMC_subset import run_MCMC_subset
import model_ELM
model_ELM.main.ELMcase.run_MCMC_subset = run_MCMC_subset

print("✓ MCMC subset functionality integrated")

# ============================================================================
# STEP 2: Load your case with trained surrogates
# ============================================================================
pkl_file = 'pklfiles/20251031_AU-Tum_ICB20TRCNPRDCTCBC.pkl'

print(f"\nLoading case: {pkl_file}")
print(f"Working directory: {os.getcwd()}")
with open(pkl_file, 'rb') as f:
    mycase = pickle.load(f)

print(f"✓ Loaded: {mycase.casename}")
print(f"  Total parameters: {mycase.nparms_ensemble}")
print(f"  Parameters: {mycase.ensemble_parms}")
print(f"  Available surrogates: {list(mycase.surrogate.keys())}")

# ============================================================================
# CALIBRATE ALL 17 ROOT PARAMETERS, FIX 3 LEAF PARAMETERS
# ============================================================================
print("\n" + "="*80)
print("MCMC CALIBRATION: 17 Root Parameters (Fix 3 Leaf Parameters)")
print("="*80)

# All 17 TAM root parameters to calibrate
calibrate_params = [
    # Root C:N ratios (Transport, Absorptive, Mycorrhizal)
    'froottcn',
    'frootacn',
    'frootmcn',

    # Root C:P ratios (Transport, Absorptive, Mycorrhizal)
    'froottcp',
    'frootacp',
    'frootmcp',

    # Root longevity (Transport, Absorptive, Mycorrhizal)
    'froott_long',
    'froota_long',
    'frootm_long',

    # Root chemistry - labile fractions (Transport, Absorptive, Mycorrhizal)
    'frt_flab',
    'fra_flab',
    'frm_flab',

    # Root chemistry - cellulose fractions (Transport, Absorptive, Mycorrhizal)
    'frt_fcel',
    'fra_fcel',
    'frm_fcel',

    # Root-leaf allocation (Transport, Absorptive)
    'froott_leaf',
    'froota_leaf'
]

# Fix 3 leaf parameters at literature values for AU-Tum (Eucalyptus forest)
fixed_params = {
    'leafcn': 30.0,    # 26; Leaf C:N ratio (typical for Eucalyptus: 20-35)
    'slatop': 0.012,   # Specific leaf area (m²/gC) (typical for Eucalyptus: 0.010-0.015)
    'flnr': 0.0515      # 0.085Fraction of leaf N in Rubisco (typical: 0.05-0.12)
}

print(f"\nCalibrating {len(calibrate_params)} root parameters:")
for i, param in enumerate(calibrate_params, 1):
    print(f"  {i:2d}. {param}")

print(f"\nFixing {len(fixed_params)} leaf parameters:")
for param, value in fixed_params.items():
    print(f"  {param:10s} = {value}")

# Run MCMC calibration
print("\n" + "="*80)
print("Running MCMC Calibration...")
print("="*80)
print(f"Variables to calibrate against: GPP, ER")
print(f"MCMC chain length: 10000 evaluations")
print(f"Burn-in period: 2000 evaluations")
print("\nThis may take several minutes...")

parms_best = mycase.run_MCMC_subset(
    calibrate_params=calibrate_params,
    fixed_params=fixed_params,
    myvars=['GPP', 'ER'],           # Calibrate against GPP and ER observations
    nevals=500000,                    # MCMC chain length
    nburn=5000,                      # Burn-in period
    burnsteps=10
)

print("\n" + "="*80)
print("✓ CALIBRATION COMPLETE!")
print("="*80)

print(f"\nBest calibrated root parameters:")
for i, param in enumerate(calibrate_params):
    print(f"  {param:15s} = {parms_best[i]:10.4f}")

print(f"\nFixed leaf parameters:")
for param, value in fixed_params.items():
    print(f"  {param:15s} = {value:10.4f}")

print(f"\nResults saved to: UQ_output/{mycase.casename}/MCMC_output/")
print(f"  ├── parms_best_full.txt     # Best parameters (all 20)")
print(f"  ├── MCMC_chain.txt          # Posterior samples (17 calibrated)")
print(f"  ├── subset_info.txt         # Calibration configuration")
print(f"  └── plots/                  # Trace plots and posterior PDFs")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"""
Calibration Strategy:
  - Calibrated: {len(calibrate_params)} TAM root parameters
  - Fixed: {len(fixed_params)} leaf parameters at literature values
  - Total: {len(calibrate_params) + len(fixed_params)} parameters in model

Why fix leaf parameters?
  1. Focus on TAM-specific root trait variation
  2. Reduce parameter space from 20D to 17D (faster convergence)
  3. Use well-constrained leaf traits from literature/measurements
  4. Avoid compensatory effects between root and leaf parameters

Next steps:
  1. Check convergence: Review trace plots in MCMC_output/plots/
  2. Validate predictions: Compare GPP/ER predictions to observations
  3. Analyze posterior: Check parameter correlations and uncertainties
  4. If needed: Re-run with different fixed leaf values or calibrate them too

Key advantage of subset approach:
  - Surrogates trained ONCE on all 20 parameters
  - Can run MULTIPLE calibrations with different fixed/calibrated subsets
  - No need to retrain surrogates for different calibration scenarios
""")

print("\n✓✓✓ Root parameter calibration completed successfully ✓✓✓")
