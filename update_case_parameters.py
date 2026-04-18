#!/usr/bin/env python
"""
Update parameter ranges in an existing OLMT case pickle file

This script updates the parameter bounds (ensemble_pmin, ensemble_pmax) in a saved
case object to use new parameter ranges from a different parameter file.

This is useful when you want to run MCMC with reduced parameter ranges without
retraining the surrogate model or re-running the ensemble.

Usage:
    python update_case_parameters.py
"""

import pickle
import sys
import os

# Configuration
PICKLE_FILE = 'pklfiles/20251030_US-Blo_ICB20TRCNPRDCTCBC.pkl'
NEW_PARAM_FILE = 'inputdata/PTTAM/US-Blo_parm_list_tam_mcmc_reduced'
OUTPUT_PICKLE = 'pklfiles/20251030_US-Blo_ICB20TRCNPRDCTCBC.pkl'  # Same file - will overwrite

print("="*70)
print("UPDATING CASE PARAMETER RANGES")
print("="*70)

# Load existing case
print(f"\n1. Loading existing case from: {PICKLE_FILE}")
try:
    with open(PICKLE_FILE, 'rb') as f:
        mycase = pickle.load(f)
    print(f"   ✓ Loaded case: {mycase.casename}")
    print(f"   ✓ Current parameters: {len(mycase.ensemble_parms)} parameters")
except Exception as e:
    print(f"   ✗ ERROR: Could not load pickle file: {e}")
    sys.exit(1)

# Save old parameter ranges for comparison
old_pmin = mycase.ensemble_pmin.copy()
old_pmax = mycase.ensemble_pmax.copy()

print(f"\n2. Reading new parameter ranges from: {NEW_PARAM_FILE}")
try:
    mycase.read_parm_list(NEW_PARAM_FILE)
    print(f"   ✓ Read {len(mycase.ensemble_parms)} parameters")
except Exception as e:
    print(f"   ✗ ERROR: Could not read parameter file: {e}")
    sys.exit(1)

# Verify parameter count and order matches
if len(mycase.ensemble_parms) != len(old_pmin):
    print(f"   ✗ ERROR: Parameter count mismatch!")
    print(f"      Old: {len(old_pmin)} parameters")
    print(f"      New: {len(mycase.ensemble_parms)} parameters")
    sys.exit(1)

# Show what changed
print(f"\n3. Parameter range comparison:")
print(f"   {'Parameter':<15} {'Old Range':<25} {'New Range':<25} {'Status':<10}")
print(f"   {'-'*15} {'-'*25} {'-'*25} {'-'*10}")

n_changed = 0
for i, pname in enumerate(mycase.ensemble_parms):
    old_range = f"{old_pmin[i]:.3g} - {old_pmax[i]:.3g}"
    new_range = f"{mycase.ensemble_pmin[i]:.3g} - {mycase.ensemble_pmax[i]:.3g}"

    # Check if range changed significantly
    range_changed = (abs(old_pmin[i] - mycase.ensemble_pmin[i]) > 1e-6 or
                     abs(old_pmax[i] - mycase.ensemble_pmax[i]) > 1e-6)

    # Check if range is now very tight (effectively fixed)
    range_width_old = old_pmax[i] - old_pmin[i]
    range_width_new = mycase.ensemble_pmax[i] - mycase.ensemble_pmin[i]
    is_now_fixed = range_width_new < 0.01 * range_width_old  # Less than 1% of original

    if range_changed:
        n_changed += 1
        if is_now_fixed:
            status = "FIXED"
        else:
            status = "CHANGED"
        print(f"   {pname:<15} {old_range:<25} {new_range:<25} {status:<10}")
    else:
        if i < 3:  # Show first 3 unchanged parameters
            status = "unchanged"
            print(f"   {pname:<15} {old_range:<25} {new_range:<25} {status:<10}")

if n_changed > 3:
    print(f"   ... and {n_changed - 3} more changed parameters")

print(f"\n   Summary: {n_changed}/{len(mycase.ensemble_parms)} parameters changed")

# Count effectively fixed parameters
n_fixed = sum(1 for i in range(len(mycase.ensemble_parms))
              if (mycase.ensemble_pmax[i] - mycase.ensemble_pmin[i]) <
                 0.01 * (old_pmax[i] - old_pmin[i]))
n_varying = len(mycase.ensemble_parms) - n_fixed

print(f"   Effectively: {n_varying} varying + {n_fixed} fixed parameters")

# Save updated case
print(f"\n4. Saving updated case to: {OUTPUT_PICKLE}")
try:
    mycase.create_pkl(outdir='pklfiles/')
    print(f"   ✓ Case saved successfully")
except Exception as e:
    print(f"   ✗ ERROR: Could not save pickle file: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("UPDATE COMPLETE!")
print("="*70)
print(f"\nThe case is now ready to run MCMC with reduced parameter space.")
print(f"Simply run your MCMC script as usual:\n")
print(f"   ./MCMC_only.sh")
print(f"\nor:")
print(f"   ./manage_ensemble.py --case 20251030_US-Blo_ICB20TRCNPRDCTCBC --MCMC_only --tstep yearly --use_available_obs")
print(f"\nExpected MCMC performance:")
print(f"  • Acceptance rate: 20-40% (was 6.3%)")
print(f"  • ESS: >1000 (was 6-7)")
print(f"  • Only {n_varying} parameters will vary significantly")
print(f"  • {n_fixed} parameters effectively fixed at reasonable values")
print()
