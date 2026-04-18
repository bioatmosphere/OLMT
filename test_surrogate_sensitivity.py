#!/usr/bin/env python
"""
Test surrogate sensitivity with extreme parameter values
"""
import pickle
import numpy as np

# Load the case
pkl_file = '/autofs/nccsopen-svm1_home/6lw/models/OLMT/pklfiles/20251029_AU-Tum_ICB20TRCNPRDCTCBC.pkl'

with open(pkl_file, 'rb') as f:
    mycase = pickle.load(f)

print("="*80)
print("TESTING SURROGATE WITH EXTREME PARAMETER VALUES")
print("="*80)

# Test 1: Minimum values
parms_min = np.array(mycase.ensemble_pmin)
output_min = mycase.run_surrogate(parms_min.reshape(1, -1), ['GPP', 'ER'])
print(f"\nMinimum parameters:")
print(f"  GPP: {output_min['GPP'].flatten()}")
print(f"  GPP mean: {np.mean(output_min['GPP'])*31536000:.2f} gC/m²/year")
print(f"  ER mean: {np.mean(output_min['ER'])*31536000:.2f} gC/m²/year")

# Test 2: Maximum values
parms_max = np.array(mycase.ensemble_pmax)
output_max = mycase.run_surrogate(parms_max.reshape(1, -1), ['GPP', 'ER'])
print(f"\nMaximum parameters:")
print(f"  GPP mean: {np.mean(output_max['GPP'])*31536000:.2f} gC/m²/year")
print(f"  ER mean: {np.mean(output_max['ER'])*31536000:.2f} gC/m²/year")

# Test 3: Midpoint
parms_mid = (np.array(mycase.ensemble_pmin) + np.array(mycase.ensemble_pmax)) / 2
output_mid = mycase.run_surrogate(parms_mid.reshape(1, -1), ['GPP', 'ER'])
print(f"\nMidpoint parameters:")
print(f"  GPP mean: {np.mean(output_mid['GPP'])*31536000:.2f} gC/m²/year")
print(f"  ER mean: {np.mean(output_mid['ER'])*31536000:.2f} gC/m²/year")

# Calculate range
gpp_range = np.mean(output_max['GPP']) - np.mean(output_min['GPP'])
er_range = np.mean(output_max['ER']) - np.mean(output_min['ER'])

print(f"\n" + "="*80)
print(f"SURROGATE DYNAMIC RANGE:")
print(f"="*80)
print(f"  GPP range: {gpp_range*31536000:.2f} gC/m²/year")
print(f"  ER range:  {er_range*31536000:.2f} gC/m²/year")
print(f"\n  Observed GPP range: {4232.53 - 2788.82:.2f} gC/m²/year")
print(f"  Observed ER range:  {3401.33 - 1991.41:.2f} gC/m²/year")

# Check if surrogate range covers observed range
if gpp_range*31536000 < (4232.53 - 2788.82):
    print(f"\n  ⚠️  WARNING: Surrogate GPP range ({gpp_range*31536000:.2f}) is SMALLER than observed range ({4232.53 - 2788.82:.2f})!")
    print(f"      The model cannot fit the data!")

# Test with original wider bounds (from original AU-Tum file)
print(f"\n" + "="*80)
print(f"TESTING WITH ORIGINAL WIDER BOUNDS:")
print(f"="*80)

# Original AU-Tum bounds
orig_min = [20, 11, 7, 375, 250, 10, 3, 0.5, 0.13, 0.1, 0.35, 0.1, 0.35, 0.1, 0.35, 0.05, 0.2]
orig_max = [184, 119, 25, 1125, 750, 1600, 10, 4, 1, 0.35, 0.65, 0.35, 0.65, 0.35, 0.65, 0.4, 0.6]

print(f"\nOriginal min (if within training range):")
parms_orig_min = np.array(orig_min)
try:
    output_orig_min = mycase.run_surrogate(parms_orig_min.reshape(1, -1), ['GPP', 'ER'])
    print(f"  GPP mean: {np.mean(output_orig_min['GPP'])*31536000:.2f} gC/m²/year")
    print(f"  ER mean: {np.mean(output_orig_min['ER'])*31536000:.2f} gC/m²/year")
except Exception as e:
    print(f"  Error: {e}")

print(f"\nOriginal max (if within training range):")
parms_orig_max = np.array(orig_max)
try:
    output_orig_max = mycase.run_surrogate(parms_orig_max.reshape(1, -1), ['GPP', 'ER'])
    print(f"  GPP mean: {np.mean(output_orig_max['GPP'])*31536000:.2f} gC/m²/year")
    print(f"  ER mean: {np.mean(output_orig_max['ER'])*31536000:.2f} gC/m²/year")

    gpp_range_orig = np.mean(output_orig_max['GPP']) - np.mean(output_orig_min['GPP'])
    print(f"\n  GPP range with original bounds: {gpp_range_orig*31536000:.2f} gC/m²/year")
    print(f"  vs Observed range: {4232.53 - 2788.82:.2f} gC/m²/year")
except Exception as e:
    print(f"  Error: {e}")

# Check ensemble training bounds
print(f"\n" + "="*80)
print(f"ENSEMBLE TRAINING PARAMETER BOUNDS (from scaler stats):")
print(f"="*80)

pscaler = mycase.pscaler['GPP']
# Estimate bounds as mean ± 3*std (covers ~99.7% of uniform distribution)
estimated_min = pscaler.mean_ - 3*pscaler.scale_
estimated_max = pscaler.mean_ + 3*pscaler.scale_

for i, pname in enumerate(mycase.ensemble_parms):
    print(f"  {pname:15s}: [{estimated_min[i]:7.2f}, {estimated_max[i]:7.2f}]  "
          f"vs MCMC prior [{mycase.ensemble_pmin[i]:7.2f}, {mycase.ensemble_pmax[i]:7.2f}]")
