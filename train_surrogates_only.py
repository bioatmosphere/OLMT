#!/usr/bin/env python
"""
Train surrogates only (skip GSA to avoid the 17 vs 20 parameter issue)
"""

import pickle
import sys
import os

print("="*70)
print("TRAINING SURROGATES FOR OCT 30 US-BLO (20 parameters)")
print("="*70)

# Load case
pkl_file = 'pklfiles/20251030_US-Blo_ICB20TRCNPRDCTCBC.pkl'
print(f"\n1. Loading case from: {pkl_file}")

with open(pkl_file, 'rb') as f:
    mycase = pickle.load(f)

print(f"   ✓ Case: {mycase.casename}")
print(f"   ✓ Parameters: {mycase.nparms_ensemble}")
print(f"   ✓ Param names: {mycase.ensemble_parms}")

# Verify we have ensemble data
if not hasattr(mycase, 'output') or not mycase.output:
    print("\n✗ ERROR: No ensemble output found!")
    sys.exit(1)

print(f"\n2. Found ensemble output for {len(mycase.output)} variables")

# Train surrogates
print(f"\n3. Training surrogate models...")
print(f"   Variables: {mycase.postproc_vars}")
print(f"   This will take 20-40 minutes...")

try:
    mycase.train_surrogate(mycase.postproc_vars)
    print(f"\n   ✓ Surrogate training completed!")
except Exception as e:
    print(f"\n   ✗ ERROR during training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Verify surrogates work
print(f"\n4. Verifying surrogates...")
import numpy as np

if 'GPP' in mycase.surrogate:
    test_input = np.zeros((1, mycase.nparms_ensemble))
    try:
        result = mycase.pscaler['GPP'].transform(test_input)
        pred = mycase.surrogate['GPP'].predict(result)
        print(f"   ✓ Surrogates work correctly with {mycase.nparms_ensemble} parameters")
    except Exception as e:
        print(f"   ✗ Surrogate test failed: {e}")
        sys.exit(1)
else:
    print(f"   ✗ GPP surrogate not found!")
    sys.exit(1)

# Save case
print(f"\n5. Saving case with trained surrogates...")
mycase.create_pkl(outdir='pklfiles/')
print(f"   ✓ Saved to: {pkl_file}")

print("\n" + "="*70)
print("SUCCESS!")
print("="*70)
print(f"\nSurrogates trained and saved successfully.")
print(f"Case now has 20-parameter surrogates ready for MCMC.")
print(f"\nNext steps:")
print(f"  1. Run MCMC subset calibration:")
print(f"     See example_mcmc_subset_AU-Tum.py for usage")
print(f"  2. Or run full MCMC:")
print(f"     sbatch runscripts/MCMC_only.sh")
print()
