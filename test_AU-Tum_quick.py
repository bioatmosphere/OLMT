#!/usr/bin/env python
"""
Quick test of MCMC subset with AU-Tum site (short runs for testing)
"""

import sys
sys.path.append('..')
import pickle

# Integrate MCMC_subset functionality
from model_ELM.MCMC_subset import run_MCMC_subset
import model_ELM
model_ELM.main.ELMcase.run_MCMC_subset = run_MCMC_subset

print("="*80)
print("AU-TUM SITE MCMC SUBSET TEST (SHORT VERSION)")
print("="*80)

# Load case
pkl_file = 'pklfiles/20251017_AU-Tum_ICB20TRCNPRDCTCBC.pkl'
print(f"\nLoading case: {pkl_file}")

with open(pkl_file, 'rb') as f:
    mycase = pickle.load(f)

print(f"✓ Loaded: {mycase.casename}")
print(f"  Parameters: {mycase.nparms_ensemble}")
print(f"  Available surrogates: {list(mycase.surrogate.keys())}")

# Check observations
print(f"\nChecking observations...")
if hasattr(mycase, 'obs'):
    print(f"  ✓ Observations: {list(mycase.obs.keys())}")
    import numpy as np
    for var in ['GPP', 'ER']:
        if var in mycase.obs:
            obs_array = np.array(mycase.obs[var])
            valid_obs = obs_array[obs_array > -9000]
            print(f"  {var}: {len(valid_obs)} observations, range {np.min(valid_obs)*31536000:.0f}-{np.max(valid_obs)*31536000:.0f} gC/m²/year")

# ============================================================================
# Test: Calibrate Root C:N Ratios (3 parameters)
# ============================================================================
print(f"\n" + "="*80)
print("TEST: Calibrate Root C:N Ratios (3 parameters)")
print("="*80)

calibrate_params = ['froottcn', 'frootacn', 'frootmcn']

fixed_params = {
    'froottcp': 700.0,
    'frootacp': 500.0,
    'frootmcp': 750.0,
    'froott_long': 6.5,
    'froota_long': 1.0,
    'frootm_long': 0.5,
    'frt_flab': 0.23,
    'frt_fcel': 0.50,
    'fra_flab': 0.23,
    'fra_fcel': 0.50,
    'frm_flab': 0.23,
    'frm_fcel': 0.50,
    'froott_leaf': 0.22,
    'froota_leaf': 0.40
}

print(f"\nCalibrating: {calibrate_params}")
print(f"Fixing: {len(fixed_params)} parameters")
print(f"Variables: ['GPP', 'ER']")
print(f"Evaluations: 500 (quick test)")

print(f"\nRunning MCMC...")

try:
    parms_best = mycase.run_MCMC_subset(
        calibrate_params=calibrate_params,
        fixed_params=fixed_params,
        myvars=['GPP', 'ER'],
        nevals=500,      # Short for quick test
        nburn=50,        # Reduced burn-in
        burnsteps=10
    )

    print(f"\n" + "="*80)
    print("SUCCESS!")
    print("="*80)

    print(f"\nBest parameter values (calibrated):")
    for i, pname in enumerate(calibrate_params):
        print(f"  {pname:15s}: {parms_best[mycase.ensemble_parms.index(pname)]:.4f}")

    print(f"\nFull parameter vector saved to:")
    print(f"  UQ_output/{mycase.casename}/MCMC_output/parms_best_full.txt")

    print(f"\n✓✓✓ AU-TUM MCMC SUBSET TEST PASSED ✓✓✓")

except Exception as e:
    print(f"\n" + "="*80)
    print("FAILED!")
    print("="*80)
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
