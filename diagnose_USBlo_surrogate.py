#!/usr/bin/env python
"""
Comprehensive Surrogate Model Robustness Diagnostics for US-Blo Site
=====================================================================
This script performs a complete diagnostic analysis of surrogate models
trained for US-Blo ELM-TAM PFT 1 calibration.

Diagnostic Categories:
1. Ensemble output coverage vs observations
2. Surrogate model training quality (R² scores)
3. Parameter and output scaler validation
4. Surrogate sensitivity and dynamic range
5. Parameter correlation and identifiability issues
6. Out-of-sample extrapolation behavior
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Configuration
PKL_FILE = '/autofs/nccsopen-svm1_home/6lw/models/OLMT/pklfiles/20251030_US-Blo_ICB20TRCNPRDCTCBC.pkl'
OBS_FILE = '/autofs/nccsopen-svm1_home/6lw/models/OLMT/observations/fluxnet/yearly/FLX_US-Blo_FLUXNET2015_FULLSET_YY_1997-2007_1-4.csv'

# Target variables for calibration
TARGET_VARS = ['GPP', 'ER', 'NEE', 'NPP']

# Output directory
OUTPUT_DIR = Path('./UQ_output/US-Blo_surrogate_diagnostics')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print(" " * 20 + "US-BLO SURROGATE MODEL ROBUSTNESS DIAGNOSTICS")
print("=" * 100)
print(f"\nCase file: {PKL_FILE}")
print(f"Output directory: {OUTPUT_DIR}")

# ============================================================================
# SECTION 1: LOAD DATA
# ============================================================================
print("\n" + "=" * 100)
print("SECTION 1: LOADING CASE AND OBSERVATIONS")
print("=" * 100)

print(f"\nLoading case from: {PKL_FILE}")
try:
    with open(PKL_FILE, 'rb') as f:
        mycase = pickle.load(f)
    print("✓ Case loaded successfully")
except Exception as e:
    print(f"❌ ERROR loading case: {e}")
    sys.exit(1)

# Load observations
print(f"\nLoading observations from: {OBS_FILE}")
try:
    import pandas as pd
    obs_data = pd.read_csv(OBS_FILE)
    print("✓ Observations loaded successfully")
    print(f"  Years available: {obs_data['TIMESTAMP'].min()} - {obs_data['TIMESTAMP'].max()}")
    print(f"  Columns: {list(obs_data.columns[:10])}...")
except Exception as e:
    print(f"⚠️  Warning: Could not load observations: {e}")
    obs_data = None

# Check what's in the case
print(f"\nCase attributes:")
print(f"  Site: {mycase.site if hasattr(mycase, 'site') else 'N/A'}")
print(f"  Compset: {mycase.compset if hasattr(mycase, 'compset') else 'N/A'}")
print(f"  Number of ensemble members: {mycase.nsamples_ensemble if hasattr(mycase, 'nsamples_ensemble') else 'N/A'}")
print(f"  Number of parameters: {mycase.nparms_ensemble if hasattr(mycase, 'nparms_ensemble') else 'N/A'}")
print(f"  Parameter names: {mycase.ensemble_parms if hasattr(mycase, 'ensemble_parms') else 'N/A'}")

# Check what outputs are available
if hasattr(mycase, 'output') and mycase.output:
    print(f"  Available model outputs: {list(mycase.output.keys())}")
else:
    print("  ❌ No ensemble outputs found!")

# Check if surrogates exist
if hasattr(mycase, 'surrogate') and mycase.surrogate:
    print(f"  Surrogate models trained for: {list(mycase.surrogate.keys())}")
else:
    print("  ❌ No surrogate models found!")
    sys.exit(1)

# ============================================================================
# SECTION 2: ENSEMBLE OUTPUT DIAGNOSTICS
# ============================================================================
print("\n" + "=" * 100)
print("SECTION 2: ENSEMBLE OUTPUT DIAGNOSTICS")
print("=" * 100)

# Get observed ranges if available
obs_ranges = {}
if obs_data is not None:
    # Calculate observed ranges (annual means)
    # FLUXNET units are typically gC/m²/day, need to convert to gC/m²/year
    for var in TARGET_VARS:
        if var in obs_data.columns:
            # Remove missing values (typically -9999)
            valid_obs = obs_data[var][obs_data[var] > -9000]
            if len(valid_obs) > 0:
                # Assuming annual sums already in gC/m²/year
                obs_ranges[var] = (valid_obs.min(), valid_obs.max(), valid_obs.mean())
                print(f"\nObserved {var}:")
                print(f"  Range: [{obs_ranges[var][0]:.2f}, {obs_ranges[var][1]:.2f}] gC/m²/year")
                print(f"  Mean: {obs_ranges[var][2]:.2f} gC/m²/year")

print("\n" + "-" * 100)
print("Ensemble Output Analysis:")
print("-" * 100)

for var in TARGET_VARS:
    if var not in mycase.output:
        print(f"\n⚠️  {var}: Not in ensemble outputs")
        continue

    data = mycase.output[var]  # Shape: (n_years, n_ensemble) or (n_ensemble, n_years)

    # Check shape and transpose if needed
    print(f"\n{var}:")
    print(f"  Raw shape: {data.shape}")

    # Convert from gC/m²/s to gC/m²/year
    data_annual = data * 31536000

    # Get statistics
    all_values = data_annual.flatten()
    yearly_means = np.mean(data_annual, axis=1) if data.shape[0] < data.shape[1] else np.mean(data_annual, axis=0)

    print(f"  Ensemble statistics (all members × years):")
    print(f"    Min:    {np.min(all_values):10.2f} gC/m²/year")
    print(f"    Max:    {np.max(all_values):10.2f} gC/m²/year")
    print(f"    Mean:   {np.mean(all_values):10.2f} gC/m²/year")
    print(f"    Std:    {np.std(all_values):10.2f} gC/m²/year")
    print(f"    Range:  {np.max(all_values) - np.min(all_values):10.2f} gC/m²/year")

    # Compare to observations
    if var in obs_ranges:
        obs_min, obs_max, obs_mean = obs_ranges[var]
        obs_range = obs_max - obs_min
        ens_range = np.max(all_values) - np.min(all_values)

        print(f"\n  Comparison to observations:")
        print(f"    Observed: [{obs_min:.2f}, {obs_max:.2f}] (range = {obs_range:.2f})")
        print(f"    Ensemble: [{np.min(all_values):.2f}, {np.max(all_values):.2f}] (range = {ens_range:.2f})")
        print(f"    Coverage: {ens_range/obs_range*100:.1f}% of observed range")

        if np.max(all_values) < obs_min:
            print(f"    ❌ CRITICAL: Ensemble max < observed min (gap = {obs_min - np.max(all_values):.2f})")
        elif np.min(all_values) > obs_max:
            print(f"    ❌ CRITICAL: Ensemble min > observed max (gap = {np.min(all_values) - obs_max:.2f})")
        elif ens_range < obs_range * 0.5:
            print(f"    ⚠️  WARNING: Ensemble range is {obs_range/ens_range:.1f}x smaller than observed")
        else:
            print(f"    ✓ Ensemble adequately covers observed range")

# ============================================================================
# SECTION 3: SURROGATE MODEL QUALITY
# ============================================================================
print("\n" + "=" * 100)
print("SECTION 3: SURROGATE MODEL QUALITY ASSESSMENT")
print("=" * 100)

surrogate_quality = {}

for var in TARGET_VARS:
    if var not in mycase.surrogate:
        print(f"\n⚠️  {var}: No surrogate model found")
        continue

    print(f"\n{var} Surrogate Model:")
    print("-" * 80)

    surr = mycase.surrogate[var]

    # Check if GridSearchCV
    if hasattr(surr, 'best_score_'):
        cv_score = surr.best_score_
        best_params = surr.best_params_

        print(f"  Cross-validation R² score: {cv_score:.6f}")
        print(f"  Best hyperparameters:")
        for param, value in best_params.items():
            print(f"    {param}: {value}")

        surrogate_quality[var] = {
            'cv_score': cv_score,
            'best_params': best_params
        }

        # Quality assessment
        if cv_score > 0.95:
            print(f"  ✓ EXCELLENT quality (R² > 0.95)")
        elif cv_score > 0.90:
            print(f"  ✓ GOOD quality (R² > 0.90)")
        elif cv_score > 0.80:
            print(f"  ⚠️  MODERATE quality (R² > 0.80)")
        else:
            print(f"  ❌ POOR quality (R² < 0.80)")

        # Check underlying estimator
        if hasattr(surr, 'best_estimator_'):
            estimator = surr.best_estimator_
            if hasattr(estimator, 'n_iter_'):
                print(f"  Training iterations: {estimator.n_iter_}")
            if hasattr(estimator, 'loss_'):
                print(f"  Final training loss: {estimator.loss_:.6f}")

# ============================================================================
# SECTION 4: SCALER DIAGNOSTICS
# ============================================================================
print("\n" + "=" * 100)
print("SECTION 4: PARAMETER AND OUTPUT SCALER VALIDATION")
print("=" * 100)

# Check parameter scalers
if hasattr(mycase, 'pscaler') and mycase.pscaler:
    print("\nParameter Scalers:")
    print("-" * 80)

    # Use first available variable's scaler
    sample_var = list(mycase.pscaler.keys())[0]
    pscaler = mycase.pscaler[sample_var]

    print(f"  Scaler type: {type(pscaler).__name__}")
    if hasattr(pscaler, 'mean_'):
        print(f"\n  Parameter scaling statistics:")
        for i, pname in enumerate(mycase.ensemble_parms):
            print(f"    {pname:15s}: mean={pscaler.mean_[i]:8.2f}, std={pscaler.scale_[i]:8.2f}")
else:
    print("\n❌ No parameter scalers found!")

# Check output scalers
if hasattr(mycase, 'yscaler') and mycase.yscaler:
    print("\nOutput Scalers:")
    print("-" * 80)

    for var in TARGET_VARS:
        if var in mycase.yscaler:
            yscaler = mycase.yscaler[var]
            print(f"\n  {var}:")
            print(f"    Scaler type: {type(yscaler).__name__}")
            if hasattr(yscaler, 'mean_'):
                print(f"    Output mean: {yscaler.mean_}")
                print(f"    Output std:  {yscaler.scale_}")
else:
    print("\n❌ No output scalers found!")

# ============================================================================
# SECTION 5: SURROGATE SENSITIVITY AND DYNAMIC RANGE
# ============================================================================
print("\n" + "=" * 100)
print("SECTION 5: SURROGATE SENSITIVITY AND DYNAMIC RANGE")
print("=" * 100)

# Create test parameter vectors
parms_min = np.array(mycase.ensemble_pmin)
parms_max = np.array(mycase.ensemble_pmax)
parms_mid = (parms_min + parms_max) / 2

print("\nTesting surrogate with extreme parameter values:")
print("-" * 80)

# Test at different parameter values
test_configs = {
    'Minimum': parms_min,
    'Midpoint': parms_mid,
    'Maximum': parms_max
}

surrogate_ranges = {}

for config_name, parms in test_configs.items():
    print(f"\n{config_name} parameters:")

    try:
        output = mycase.run_surrogate(parms.reshape(1, -1), TARGET_VARS)

        for var in TARGET_VARS:
            if var in output:
                val_annual = np.mean(output[var]) * 31536000
                print(f"  {var:10s}: {val_annual:10.2f} gC/m²/year")

                if var not in surrogate_ranges:
                    surrogate_ranges[var] = {'min': val_annual, 'max': val_annual}
                else:
                    surrogate_ranges[var]['min'] = min(surrogate_ranges[var]['min'], val_annual)
                    surrogate_ranges[var]['max'] = max(surrogate_ranges[var]['max'], val_annual)
    except Exception as e:
        print(f"  ❌ Error: {e}")

# Analyze dynamic range
print("\n" + "-" * 80)
print("Surrogate Dynamic Range Analysis:")
print("-" * 80)

for var in TARGET_VARS:
    if var in surrogate_ranges:
        surr_min = surrogate_ranges[var]['min']
        surr_max = surrogate_ranges[var]['max']
        surr_range = surr_max - surr_min

        print(f"\n{var}:")
        print(f"  Surrogate range: [{surr_min:.2f}, {surr_max:.2f}] = {surr_range:.2f} gC/m²/year")

        if var in obs_ranges:
            obs_min, obs_max, _ = obs_ranges[var]
            obs_range = obs_max - obs_min

            print(f"  Observed range:  [{obs_min:.2f}, {obs_max:.2f}] = {obs_range:.2f} gC/m²/year")
            print(f"  Coverage: {surr_range/obs_range*100:.1f}% of observed range")

            if surr_range < obs_range * 0.5:
                print(f"  ❌ CRITICAL: Surrogate range is insufficient for calibration!")
            elif surr_range < obs_range:
                print(f"  ⚠️  WARNING: Surrogate range is smaller than observed")
            else:
                print(f"  ✓ Surrogate range adequate for calibration")

# ============================================================================
# SECTION 6: PARAMETER SENSITIVITY ANALYSIS
# ============================================================================
print("\n" + "=" * 100)
print("SECTION 6: INDIVIDUAL PARAMETER SENSITIVITY")
print("=" * 100)

print("\nTesting each parameter individually (±10% from midpoint):")
print("-" * 80)

param_sensitivities = {var: {} for var in TARGET_VARS if var in mycase.surrogate}

for p_idx in range(mycase.nparms_ensemble):
    pname = mycase.ensemble_parms[p_idx]

    # Baseline
    parms_base = parms_mid.copy()
    output_base = mycase.run_surrogate(parms_base.reshape(1, -1), TARGET_VARS)

    # +10%
    parms_plus = parms_mid.copy()
    parms_plus[p_idx] = min(parms_plus[p_idx] * 1.1, mycase.ensemble_pmax[p_idx])
    output_plus = mycase.run_surrogate(parms_plus.reshape(1, -1), TARGET_VARS)

    # -10%
    parms_minus = parms_mid.copy()
    parms_minus[p_idx] = max(parms_minus[p_idx] * 0.9, mycase.ensemble_pmin[p_idx])
    output_minus = mycase.run_surrogate(parms_minus.reshape(1, -1), TARGET_VARS)

    print(f"\n{pname} (range: [{mycase.ensemble_pmin[p_idx]:.2f}, {mycase.ensemble_pmax[p_idx]:.2f}]):")

    for var in TARGET_VARS:
        if var in output_base:
            base_val = np.mean(output_base[var]) * 31536000
            plus_val = np.mean(output_plus[var]) * 31536000
            minus_val = np.mean(output_minus[var]) * 31536000

            # Normalized sensitivity (% change in output per % change in parameter)
            pct_change = ((plus_val - minus_val) / base_val) * 100 if base_val != 0 else 0

            print(f"  {var:10s}: base={base_val:8.2f}, +10%={plus_val:8.2f}, -10%={minus_val:8.2f}, "
                  f"Δ={pct_change:+6.2f}%")

            param_sensitivities[var][pname] = abs(pct_change)

# Identify most influential parameters
print("\n" + "-" * 80)
print("Most Influential Parameters (sorted by absolute sensitivity):")
print("-" * 80)

for var in param_sensitivities:
    print(f"\n{var}:")
    sorted_params = sorted(param_sensitivities[var].items(), key=lambda x: x[1], reverse=True)
    for i, (pname, sensitivity) in enumerate(sorted_params[:5], 1):
        print(f"  {i}. {pname:15s}: {sensitivity:6.2f}% change")

# ============================================================================
# SECTION 7: PARAMETER CORRELATION AND IDENTIFIABILITY
# ============================================================================
print("\n" + "=" * 100)
print("SECTION 7: PARAMETER CORRELATION AND IDENTIFIABILITY ISSUES")
print("=" * 100)

print("\nChecking for perfectly correlated parameters:")
print("-" * 80)

# Check root parameters that are often correlated
root_param_groups = [
    ['froottcp', 'frootacp', 'frootmcp'],
    ['froottcn', 'frootacn', 'frootmcn']
]

for group in root_param_groups:
    # Check if all parameters in group are present
    group_present = all(p in mycase.ensemble_parms for p in group)

    if group_present:
        print(f"\nTesting {', '.join(group)}:")

        # Get indices
        indices = [mycase.ensemble_parms.index(p) for p in group]

        # Test: increase each parameter individually
        outputs = []
        for idx in indices:
            parms_test = parms_mid.copy()
            parms_test[idx] *= 1.2
            output_test = mycase.run_surrogate(parms_test.reshape(1, -1), TARGET_VARS)
            outputs.append(output_test)

        # Check if effects are identical (perfect correlation)
        if TARGET_VARS[0] in outputs[0]:
            vals = [np.mean(out[TARGET_VARS[0]]) for out in outputs]

            # Check if all values are very similar
            if np.std(vals) / np.mean(vals) < 0.01:  # <1% variation
                print(f"  ⚠️  WARNING: Parameters have nearly identical effects!")
                print(f"  Values: {[v*31536000 for v in vals]}")
                print(f"  This suggests perfect correlation - parameters are not identifiable")
            else:
                print(f"  ✓ Parameters have distinguishable effects")
                for p, v in zip(group, vals):
                    print(f"    {p}: {v*31536000:.2f}")

# ============================================================================
# SECTION 8: OUT-OF-BOUNDS EXTRAPOLATION TEST
# ============================================================================
print("\n" + "=" * 100)
print("SECTION 8: OUT-OF-BOUNDS EXTRAPOLATION TEST")
print("=" * 100)

print("\nTesting surrogate behavior outside training bounds:")
print("-" * 80)

# Test 20% beyond bounds
extrapolation_tests = [
    ('20% below minimum', parms_min * 0.8),
    ('20% above maximum', parms_max * 1.2)
]

for test_name, parms_test in extrapolation_tests:
    print(f"\n{test_name}:")

    try:
        output_extrap = mycase.run_surrogate(parms_test.reshape(1, -1), TARGET_VARS)

        for var in TARGET_VARS:
            if var in output_extrap:
                val = np.mean(output_extrap[var]) * 31536000
                print(f"  {var:10s}: {val:10.2f} gC/m²/year")

                # Check if value is reasonable
                if var in surrogate_ranges:
                    in_bounds = surrogate_ranges[var]['min'] <= val <= surrogate_ranges[var]['max']
                    if not in_bounds:
                        print(f"    ⚠️  Outside training range!")

        print(f"  ✓ Surrogate can extrapolate (but may be unreliable)")

    except Exception as e:
        print(f"  ❌ Error: {e}")

# ============================================================================
# SECTION 9: SUMMARY AND RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 100)
print("SECTION 9: SUMMARY AND RECOMMENDATIONS")
print("=" * 100)

print("\nSurrogate Model Quality Summary:")
print("-" * 80)

for var, quality in surrogate_quality.items():
    cv_score = quality['cv_score']
    status = "✓" if cv_score > 0.9 else "⚠️" if cv_score > 0.8 else "❌"
    print(f"  {status} {var:10s}: R² = {cv_score:.4f}")

print("\nDynamic Range Coverage:")
print("-" * 80)

for var in TARGET_VARS:
    if var in surrogate_ranges and var in obs_ranges:
        surr_range = surrogate_ranges[var]['max'] - surrogate_ranges[var]['min']
        obs_range = obs_ranges[var][1] - obs_ranges[var][0]
        coverage = surr_range / obs_range * 100

        status = "✓" if coverage > 100 else "⚠️" if coverage > 50 else "❌"
        print(f"  {status} {var:10s}: {coverage:5.1f}% of observed range")

print("\n" + "-" * 80)
print("Recommendations:")
print("-" * 80)

recommendations = []

# Check surrogate quality
poor_quality = [var for var, q in surrogate_quality.items() if q['cv_score'] < 0.9]
if poor_quality:
    recommendations.append(
        f"❌ Low R² scores for {', '.join(poor_quality)}\n"
        f"   → Increase ensemble size (current: {mycase.nsamples_ensemble if hasattr(mycase, 'nsamples_ensemble') else 'unknown'})\n"
        f"   → Try different neural network architectures\n"
        f"   → Check for outliers in ensemble outputs"
    )

# Check dynamic range
insufficient_range = []
for var in TARGET_VARS:
    if var in surrogate_ranges and var in obs_ranges:
        surr_range = surrogate_ranges[var]['max'] - surrogate_ranges[var]['min']
        obs_range = obs_ranges[var][1] - obs_ranges[var][0]
        if surr_range < obs_range * 0.5:
            insufficient_range.append(var)

if insufficient_range:
    recommendations.append(
        f"❌ Insufficient dynamic range for {', '.join(insufficient_range)}\n"
        f"   → Use wider parameter bounds\n"
        f"   → Add more influential parameters (photosynthesis, allocation)\n"
        f"   → Check if model physics can reach observed values"
    )

# General recommendations
if not recommendations:
    recommendations.append("✓ Surrogates appear robust for MCMC calibration!")
else:
    recommendations.append(
        "\nNext steps:\n"
        "1. Review parameter list and bounds\n"
        "2. Consider retraining ensemble with adjustments\n"
        "3. Examine ensemble outputs in detail with diagnose_ensemble.py\n"
        f"4. Check surrogate plots in: {OUTPUT_DIR.parent}/20251029_US-Blo_ICB20TRCNPRDCTCBC/surrogate/"
    )

print()
for rec in recommendations:
    print(rec)
    print()

print("=" * 100)
print("DIAGNOSTICS COMPLETE")
print("=" * 100)
print(f"\nFull diagnostic report saved to: {OUTPUT_DIR}")
print(f"Surrogate scatter plots available at: ./UQ_output/20251029_US-Blo_ICB20TRCNPRDCTCBC/surrogate/")
print("\nFor more detailed analysis, run:")
print("  - diagnose_ensemble.py (ensemble output analysis)")
print("  - check_surrogate_scalers.py (scaler validation)")
print("  - test_surrogate_sensitivity.py (extended sensitivity tests)")
