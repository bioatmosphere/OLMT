#!/usr/bin/env python
"""
Test script to verify widened parameter ranges don't produce zero GPP
"""

import sys
import os
sys.path.append('..')
import pickle
import numpy as np

pkl_file = '../pklfiles/20251014_US-Ho1_ICB20TRCNPRDCTCBC.pkl'

print("Loading case...")
with open(pkl_file, 'rb') as f:
    mycase = pickle.load(f)

print(f"✓ Loaded: {mycase.casename}")

# Define priority parameters to widen
widening_factor = 2.0
priority_params = [
    'froottcn', 'frootacn', 'frootmcn',  # C:N ratios
    'froott_leaf', 'froota_leaf',         # Allocation
    'froott_long', 'froota_long', 'frootm_long'  # Longevity
]

# Get widened parameter ranges
samples = mycase.samples
parm_min_orig = np.min(samples, axis=1)
parm_max_orig = np.max(samples, axis=1)

parm_mid = (parm_min_orig + parm_max_orig) / 2
parm_range = parm_max_orig - parm_min_orig

parm_min = parm_min_orig.copy()
parm_max = parm_max_orig.copy()

for i, pname in enumerate(mycase.ensemble_parms):
    if pname in priority_params:
        parm_min[i] = parm_mid[i] - (parm_range[i] * widening_factor / 2)
        parm_max[i] = parm_mid[i] + (parm_range[i] * widening_factor / 2)

        if parm_min[i] < 0:
            parm_max[i] = parm_max[i] - parm_min[i]
            parm_min[i] = 0.0

print(f"\n{'='*80}")
print(f"TESTING WIDENED PARAMETER RANGES")
print(f"{'='*80}")

# Generate test parameter combinations
np.random.seed(42)
n_test = 100

print(f"\nGenerating {n_test} random parameter combinations from widened ranges...")

zero_gpp_count = 0
negative_gpp_count = 0
valid_count = 0
gpp_values = []

for i in range(n_test):
    # Generate random parameters within widened ranges
    test_params = np.array([
        np.random.uniform(parm_min[j], parm_max[j])
        for j in range(len(mycase.ensemble_parms))
    ]).reshape(1, -1)

    try:
        output = mycase.run_surrogate(test_params, ['GPP', 'ER'])

        gpp_val = output['GPP']
        if isinstance(gpp_val, np.ndarray):
            gpp_val = gpp_val.flatten()[0]

        # Convert to annual
        gpp_annual = gpp_val * 31536000.0
        gpp_values.append(gpp_annual)

        if gpp_annual <= 0:
            zero_gpp_count += 1
            if gpp_annual < 0:
                negative_gpp_count += 1
            print(f"  Test {i:3d}: ❌ GPP = {gpp_annual:>10.2f} gC/m²/yr (ZERO/NEGATIVE)")

            # Print the parameters that caused zero GPP
            print(f"    Parameters:")
            for j, pname in enumerate(mycase.ensemble_parms):
                status = "WIDENED" if pname in priority_params else "original"
                print(f"      {pname:20s} = {test_params[0, j]:10.6f}  [{status}]")
        else:
            valid_count += 1
            if i < 10:  # Show first 10 successful tests
                print(f"  Test {i:3d}: ✓ GPP = {gpp_annual:>10.2f} gC/m²/yr")
    except Exception as e:
        print(f"  Test {i:3d}: ERROR - {e}")

print(f"\n{'='*80}")
print(f"TEST RESULTS")
print(f"{'='*80}")

if len(gpp_values) > 0:
    print(f"\nGPP Statistics from {len(gpp_values)} successful predictions:")
    print(f"  Min:    {np.min(gpp_values):>10.2f} gC/m²/yr")
    print(f"  Max:    {np.max(gpp_values):>10.2f} gC/m²/yr")
    print(f"  Mean:   {np.mean(gpp_values):>10.2f} gC/m²/yr")
    print(f"  Median: {np.median(gpp_values):>10.2f} gC/m²/yr")
    print(f"  Std:    {np.std(gpp_values):>10.2f} gC/m²/yr")

print(f"\nSummary:")
print(f"  Valid (GPP > 0):        {valid_count:3d} / {n_test} ({100*valid_count/n_test:.1f}%)")
print(f"  Zero/Negative GPP:      {zero_gpp_count:3d} / {n_test} ({100*zero_gpp_count/n_test:.1f}%)")
print(f"  Negative GPP:           {negative_gpp_count:3d} / {n_test} ({100*negative_gpp_count/n_test:.1f}%)")

if zero_gpp_count == 0:
    print(f"\n✓✓✓ SUCCESS: No zero GPP values found with widened ranges!")
    print(f"    Safe to proceed with MCMC calibration.")
elif zero_gpp_count < n_test * 0.05:  # Less than 5%
    print(f"\n⚠️  WARNING: Found {zero_gpp_count} zero GPP cases ({100*zero_gpp_count/n_test:.1f}%)")
    print(f"    This is acceptable (<5%), but MCMC may occasionally sample these regions.")
else:
    print(f"\n❌ ERROR: Too many zero GPP cases ({100*zero_gpp_count/n_test:.1f}%)")
    print(f"    Consider reducing widening factor or constraining specific parameters.")

print(f"\n{'='*80}")
print(f"WIDENED PARAMETER RANGES")
print(f"{'='*80}")
print(f"{'Parameter':20s}   {'Original Range':^25s}   {'Widened Range':^25s}   {'Status':^10s}")
print("-" * 90)
for i, pname in enumerate(mycase.ensemble_parms):
    orig_str = f"[{parm_min_orig[i]:8.4f}, {parm_max_orig[i]:8.4f}]"
    new_str = f"[{parm_min[i]:8.4f}, {parm_max[i]:8.4f}]"
    status = "WIDENED" if pname in priority_params else "original"
    print(f"{pname:20s}   {orig_str:25s}   {new_str:25s}   {status:^10s}")
