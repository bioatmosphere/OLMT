#!/usr/bin/env python
"""
Retrain surrogates for Oct 30 US-Blo case with 20 parameters

This script loads the Oct 30 case and retrains surrogate models
if the ensemble data is available.
"""

import pickle
import os

# Load the Oct 30 case from the case directory (has 20 params, no surrogates yet)
case_pkl = '/gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_cases/20251030_US-Blo_ICB20TRCNPRDCTCBC/20251030_US-Blo_ICB20TRCNPRDCTCBC.pkl'

print("="*70)
print("RETRAINING SURROGATES FOR OCT 30 US-BLO (20 parameters)")
print("="*70)

print(f"\n1. Loading case from: {case_pkl}")
with open(case_pkl, 'rb') as f:
    mycase = pickle.load(f)

print(f"   ✓ Loaded: {mycase.casename}")
print(f"   ✓ Parameters: {mycase.nparms_ensemble}")
print(f"   ✓ Param names: {mycase.ensemble_parms}")

# Check if postprocessed output exists
if not hasattr(mycase, 'output') or not mycase.output:
    print("\n✗ ERROR: No ensemble output data found!")
    print("   The ensemble needs to be post-processed first.")
    print("   Run: ./manage_ensemble.py --case 20251030_US-Blo_ICB20TRCNPRDCTCBC --postproc_only")
    exit(1)

print(f"\n2. Found output for {len(mycase.output)} variables")
print(f"   Variables: {list(mycase.output.keys())[:5]}...")

# Check if ensemble matrix exists
if not hasattr(mycase, 'ensemble_matrix') or mycase.ensemble_matrix is None:
    print("\n✗ ERROR: No ensemble parameter matrix found!")
    print("   Cannot train surrogates without parameter samples.")
    exit(1)

print(f"\n3. Ensemble matrix shape: {mycase.ensemble_matrix.shape}")
print(f"   Expected: (n_samples, {mycase.nparms_ensemble})")

if mycase.ensemble_matrix.shape[1] != mycase.nparms_ensemble:
    print(f"\n✗ ERROR: Ensemble matrix has {mycase.ensemble_matrix.shape[1]} columns")
    print(f"   but case expects {mycase.nparms_ensemble} parameters!")
    exit(1)

# Train surrogates
print(f"\n4. Training surrogate models...")
print(f"   This may take 20-40 minutes...")

try:
    mycase.train_surrogate(mycase.postproc_vars)
    print(f"   ✓ Surrogates trained successfully!")
except Exception as e:
    print(f"   ✗ ERROR during training: {e}")
    exit(1)

# Save the updated case
print(f"\n5. Saving case with trained surrogates...")
mycase.create_pkl(outdir='/autofs/nccsopen-svm1_home/6lw/models/OLMT/pklfiles/')
print(f"   ✓ Saved to: pklfiles/{mycase.casename}.pkl")

print("\n" + "="*70)
print("SUCCESS! Surrogates trained and saved")
print("="*70)
print("\nNow run MCMC calibration:")
print("  See example_mcmc_subset_AU-Tum.py for MCMC subset usage")
print("  Or: sbatch runscripts/MCMC_only.sh")
print()
