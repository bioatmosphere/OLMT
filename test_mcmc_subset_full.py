#!/usr/bin/env python
"""
Quick test of run_MCMC_subset with main MCMC function
"""

import sys
sys.path.append('..')
import pickle

# Integrate MCMC_subset functionality
from model_ELM.MCMC_subset import run_MCMC_subset
import model_ELM
model_ELM.main.ELMcase.run_MCMC_subset = run_MCMC_subset

print("="*80)
print("TEST: MCMC SUBSET WITH MAIN MCMC FUNCTION")
print("="*80)

# Load case
pkl_file = 'pklfiles/20251017_AU-Tum_ICB20TRCNPRDCTCBC.pkl'
print(f"\nLoading case: {pkl_file}")

with open(pkl_file, 'rb') as f:
    mycase = pickle.load(f)

print(f"✓ Loaded: {mycase.casename}")
print(f"  Parameters: {mycase.nparms_ensemble}")

# Check observations
print(f"\nChecking observations...")
if hasattr(mycase, 'obs'):
    print(f"  ✓ Observations present: {list(mycase.obs.keys())}")
else:
    print(f"  ❌ No observations found!")
    sys.exit(1)

# Define small subset for quick test
calibrate_params = ['froottcn', 'frootacn']
fixed_params = {
    'frootmcn': 16.0,
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

print(f"\nTest configuration:")
print(f"  Calibrating: {calibrate_params}")
print(f"  Fixed: {len(fixed_params)} parameters")
print(f"  Variables: ['GPP', 'ER']")
print(f"  Evaluations: 200 (very short test)")

print(f"\n" + "="*80)
print("RUNNING MCMC SUBSET")
print("="*80)

try:
    parms_best = mycase.run_MCMC_subset(
        calibrate_params=calibrate_params,
        fixed_params=fixed_params,
        myvars=['GPP', 'ER'],
        nevals=200,      # Very short test
        nburn=20,        # Minimal burn-in
        burnsteps=5
    )

    print(f"\n" + "="*80)
    print("SUCCESS!")
    print("="*80)
    print(f"\nBest parameters: {parms_best}")
    print(f"\n✓✓✓ MCMC SUBSET WORKS WITH MAIN MCMC FUNCTION ✓✓✓")

except Exception as e:
    print(f"\n" + "="*80)
    print("FAILED!")
    print("="*80)
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
