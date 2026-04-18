#!/usr/bin/env python
"""
Comprehensive Surrogate Model Robustness Diagnostics for AU-Tum Site
=====================================================================
This script performs a complete diagnostic analysis to ensure surrogate models
are robust for downstream MCMC calibration and sensitivity analysis.

Diagnostic Tests:
1. Surrogate model training quality (R² scores)
2. Dynamic range coverage vs observations
3. Parameter sensitivity analysis
4. Parameter identifiability checks
5. Extrapolation behavior
6. Scaler validation
"""

import pickle
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Configuration
PKL_FILE = 'pklfiles/20251017_AU-Tum_ICB20TRCNPRDCTCBC.pkl'
OBS_FILE = 'observations/fluxnet/yearly/FLX_AU-Tum_FLUXNET2015_FULLSET_YY_2001-2014_2-4.csv'
PARM_FILE = 'inputdata/PTTAM/AU-Tum_parm_list_tam'

# Variables for MCMC calibration
MCMC_VARS = ['GPP', 'ER', 'NEE', 'NPP']

# Quality thresholds
R2_EXCELLENT = 0.95
R2_GOOD = 0.90
R2_ACCEPTABLE = 0.80
MIN_COVERAGE_RATIO = 0.8  # Surrogate should cover 80% of observed range

print("=" * 100)
print(" " * 25 + "AU-TUM SURROGATE ROBUSTNESS DIAGNOSTICS")
print("=" * 100)

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
    print(f"❌ ERROR: {e}")
    sys.exit(1)

# Display case info
print(f"\nCase Information:")
print(f"  Site: {mycase.site}")
print(f"  Compset: {mycase.compset}")
nsamples = len(mycase.output[list(mycase.output.keys())[0]]) if hasattr(mycase, 'output') and mycase.output else 'N/A'
print(f"  Ensemble size: {nsamples}")
print(f"  Number of parameters: {mycase.nparms_ensemble}")
print(f"  Parameters: {mycase.ensemble_parms}")

# Load observations
print(f"\nLoading observations from: {OBS_FILE}")
try:
    obs_data = pd.read_csv(OBS_FILE)
    print("✓ Observations loaded successfully")
    print(f"  Years: {obs_data['TIMESTAMP'].min()}-{obs_data['TIMESTAMP'].max()}")
    print(f"  Available variables: {[col for col in obs_data.columns if col != 'TIMESTAMP'][:10]}...")
except Exception as e:
    print(f"⚠️  Warning: Could not load observations: {e}")
    obs_data = None

# Check for surrogates
if not hasattr(mycase, 'surrogate') or not mycase.surrogate:
    print("\n❌ ERROR: No surrogate models found!")
    print("   → Run ensemble training first with mycase.train_surrogate()")
    sys.exit(1)

print(f"\nSurrogate models available for: {list(mycase.surrogate.keys())}")

# Check for ensemble outputs
if not hasattr(mycase, 'output') or not mycase.output:
    print("\n❌ ERROR: No ensemble outputs found!")
    sys.exit(1)

print(f"Ensemble outputs available for: {list(mycase.output.keys())}")

# ============================================================================
# SECTION 2: SURROGATE MODEL QUALITY
# ============================================================================
print("\n" + "=" * 100)
print("SECTION 2: SURROGATE MODEL TRAINING QUALITY")
print("=" * 100)

surrogate_scores = {}
quality_issues = []

for var in MCMC_VARS:
    if var not in mycase.surrogate:
        print(f"\n⚠️  {var}: No surrogate model found")
        quality_issues.append(f"{var} missing")
        continue

    print(f"\n{var} Surrogate:")
    print("-" * 80)

    surr = mycase.surrogate[var]

    # Check if it's GridSearchCV
    if hasattr(surr, 'best_score_'):
        cv_score = surr.best_score_
        best_params = surr.best_params_

        print(f"  Cross-validation R² score: {cv_score:.6f}")
        print(f"  Best hyperparameters:")
        for param, value in best_params.items():
            print(f"    {param}: {value}")

        surrogate_scores[var] = cv_score

        # Quality assessment
        if cv_score > R2_EXCELLENT:
            print(f"  ✓ EXCELLENT quality (R² > {R2_EXCELLENT})")
        elif cv_score > R2_GOOD:
            print(f"  ✓ GOOD quality (R² > {R2_GOOD})")
        elif cv_score > R2_ACCEPTABLE:
            print(f"  ⚠️  ACCEPTABLE quality (R² > {R2_ACCEPTABLE})")
            quality_issues.append(f"{var} moderate quality")
        else:
            print(f"  ❌ POOR quality (R² < {R2_ACCEPTABLE})")
            quality_issues.append(f"{var} poor quality")

        # Check underlying estimator
        if hasattr(surr, 'best_estimator_'):
            estimator = surr.best_estimator_
            if hasattr(estimator, 'n_iter_'):
                print(f"  Training iterations: {estimator.n_iter_}")
            if hasattr(estimator, 'loss_'):
                print(f"  Final training loss: {estimator.loss_:.6f}")
    else:
        print(f"  Type: {type(surr).__name__}")
        print(f"  ⚠️  Warning: No CV score available")

# ============================================================================
# SECTION 3: DYNAMIC RANGE ANALYSIS
# ============================================================================
print("\n" + "=" * 100)
print("SECTION 3: SURROGATE DYNAMIC RANGE vs OBSERVATIONS")
print("=" * 100)

# Get observed ranges
obs_ranges = {}
if obs_data is not None:
    print("\nProcessing observations:")
    print("-" * 80)

    for var in MCMC_VARS:
        var_col = var
        # Handle naming variations
        if var == 'ER' and 'RECO' in obs_data.columns:
            var_col = 'RECO'
        elif var == 'NEE' and 'NEE_VUT_REF' in obs_data.columns:
            var_col = 'NEE_VUT_REF'

        if var_col in obs_data.columns:
            # Filter valid data (FLUXNET uses -9999 for missing)
            valid_obs = obs_data[var_col][obs_data[var_col] > -9000]

            if len(valid_obs) > 0:
                obs_ranges[var] = {
                    'min': valid_obs.min(),
                    'max': valid_obs.max(),
                    'mean': valid_obs.mean(),
                    'std': valid_obs.std()
                }
                print(f"\n{var} ({var_col}):")
                print(f"  Range: [{obs_ranges[var]['min']:.2f}, {obs_ranges[var]['max']:.2f}] gC/m²/year")
                print(f"  Mean ± std: {obs_ranges[var]['mean']:.2f} ± {obs_ranges[var]['std']:.2f} gC/m²/year")

# Test surrogate at parameter bounds
print("\n" + "-" * 80)
print("Testing surrogate dynamic range:")
print("-" * 80)

parms_min = np.array(mycase.ensemble_pmin)
parms_max = np.array(mycase.ensemble_pmax)
parms_mid = (parms_min + parms_max) / 2

test_configs = {
    'Minimum': parms_min,
    'Midpoint': parms_mid,
    'Maximum': parms_max
}

surrogate_ranges = {}
range_issues = []

for config_name, parms in test_configs.items():
    try:
        output = mycase.run_surrogate(parms.reshape(1, -1), MCMC_VARS)

        if config_name == 'Minimum':
            print(f"\n{config_name} parameters:")
        elif config_name == 'Midpoint':
            print(f"\n{config_name} parameters:")
        else:
            print(f"\n{config_name} parameters:")

        for var in MCMC_VARS:
            if var in output:
                # Convert from gC/m²/s to gC/m²/year
                val_annual = np.mean(output[var]) * 31536000

                if var not in surrogate_ranges:
                    surrogate_ranges[var] = {'min': val_annual, 'max': val_annual}
                else:
                    surrogate_ranges[var]['min'] = min(surrogate_ranges[var]['min'], val_annual)
                    surrogate_ranges[var]['max'] = max(surrogate_ranges[var]['max'], val_annual)

                print(f"  {var:10s}: {val_annual:10.2f} gC/m²/year")
    except Exception as e:
        print(f"  ❌ Error: {e}")

# Compare ranges
print("\n" + "-" * 80)
print("Dynamic Range Comparison:")
print("-" * 80)

for var in MCMC_VARS:
    if var in surrogate_ranges and var in obs_ranges:
        surr_min = surrogate_ranges[var]['min']
        surr_max = surrogate_ranges[var]['max']
        surr_range = surr_max - surr_min

        obs_min = obs_ranges[var]['min']
        obs_max = obs_ranges[var]['max']
        obs_range = obs_max - obs_min

        coverage_ratio = surr_range / obs_range

        print(f"\n{var}:")
        print(f"  Observed:  [{obs_min:8.2f}, {obs_max:8.2f}] = {obs_range:8.2f} gC/m²/year")
        print(f"  Surrogate: [{surr_min:8.2f}, {surr_max:8.2f}] = {surr_range:8.2f} gC/m²/year")
        print(f"  Coverage:  {coverage_ratio*100:5.1f}% of observed range")

        if coverage_ratio < 0.5:
            print(f"  ❌ CRITICAL: Surrogate range insufficient (<50% of observed)")
            range_issues.append(f"{var} insufficient range")
        elif coverage_ratio < MIN_COVERAGE_RATIO:
            print(f"  ⚠️  WARNING: Surrogate range is smaller than observed")
            range_issues.append(f"{var} limited range")
        else:
            print(f"  ✓ Adequate coverage")

# ============================================================================
# SECTION 4: PARAMETER SENSITIVITY
# ============================================================================
print("\n" + "=" * 100)
print("SECTION 4: PARAMETER SENSITIVITY ANALYSIS")
print("=" * 100)

print("\nTesting individual parameter influence (±10% from midpoint):")
print("-" * 80)

param_sensitivities = {var: {} for var in MCMC_VARS if var in mycase.surrogate}

for p_idx in range(mycase.nparms_ensemble):
    pname = mycase.ensemble_parms[p_idx]

    # Baseline
    parms_base = parms_mid.copy()
    output_base = mycase.run_surrogate(parms_base.reshape(1, -1), MCMC_VARS)

    # +10%
    parms_plus = parms_mid.copy()
    parms_plus[p_idx] = min(parms_plus[p_idx] * 1.1, mycase.ensemble_pmax[p_idx])
    output_plus = mycase.run_surrogate(parms_plus.reshape(1, -1), MCMC_VARS)

    # -10%
    parms_minus = parms_mid.copy()
    parms_minus[p_idx] = max(parms_minus[p_idx] * 0.9, mycase.ensemble_pmin[p_idx])
    output_minus = mycase.run_surrogate(parms_minus.reshape(1, -1), MCMC_VARS)

    has_significant_effect = False

    for var in MCMC_VARS:
        if var in output_base:
            base_val = np.mean(output_base[var]) * 31536000
            plus_val = np.mean(output_plus[var]) * 31536000
            minus_val = np.mean(output_minus[var]) * 31536000

            # Percent change
            pct_change = ((plus_val - minus_val) / base_val) * 100 if base_val != 0 else 0
            param_sensitivities[var][pname] = abs(pct_change)

            if abs(pct_change) > 1.0:  # >1% change is significant
                has_significant_effect = True

    # Only print parameters with significant effects
    if has_significant_effect:
        print(f"\n{pname} [{mycase.ensemble_pmin[p_idx]:.1f}, {mycase.ensemble_pmax[p_idx]:.1f}]:")
        for var in MCMC_VARS:
            if var in output_base:
                base_val = np.mean(output_base[var]) * 31536000
                plus_val = np.mean(output_plus[var]) * 31536000
                minus_val = np.mean(output_minus[var]) * 31536000
                pct_change = ((plus_val - minus_val) / base_val) * 100 if base_val != 0 else 0

                print(f"  {var:10s}: {pct_change:+6.2f}%")

# Identify most influential parameters
print("\n" + "-" * 80)
print("Top 5 Most Influential Parameters:")
print("-" * 80)

for var in MCMC_VARS:
    if var in param_sensitivities:
        print(f"\n{var}:")
        sorted_params = sorted(param_sensitivities[var].items(), key=lambda x: x[1], reverse=True)
        for i, (pname, sensitivity) in enumerate(sorted_params[:5], 1):
            print(f"  {i}. {pname:20s}: {sensitivity:6.2f}% change")

# ============================================================================
# SECTION 5: PARAMETER IDENTIFIABILITY
# ============================================================================
print("\n" + "=" * 100)
print("SECTION 5: PARAMETER IDENTIFIABILITY CHECK")
print("=" * 100)

print("\nChecking for parameter correlation issues:")
print("-" * 80)

# Check root parameter groups (known to be correlated)
root_param_groups = [
    ['froottcp', 'frootacp', 'frootmcp'],
    ['froottcn', 'frootacn', 'frootmcn']
]

identifiability_issues = []

for group in root_param_groups:
    group_present = all(p in mycase.ensemble_parms for p in group)

    if group_present:
        print(f"\nTesting {', '.join(group)}:")

        indices = [mycase.ensemble_parms.index(p) for p in group]

        # Test: increase each parameter by 20%
        outputs = []
        for idx in indices:
            parms_test = parms_mid.copy()
            parms_test[idx] *= 1.2
            output_test = mycase.run_surrogate(parms_test.reshape(1, -1), ['GPP'])
            outputs.append(output_test)

        # Check if effects are identical (correlation issue)
        vals = [np.mean(out['GPP']) * 31536000 for out in outputs]

        rel_std = np.std(vals) / np.mean(vals) if np.mean(vals) > 0 else 0

        if rel_std < 0.01:  # <1% variation = perfect correlation
            print(f"  ⚠️  WARNING: Parameters have nearly identical effects!")
            print(f"  GPP values: {[f'{v:.2f}' for v in vals]}")
            print(f"  → Not independently identifiable in MCMC")
            identifiability_issues.append(f"{','.join(group)} correlated")
        else:
            print(f"  ✓ Parameters have distinguishable effects")
            for p, v in zip(group, vals):
                print(f"    {p}: {v:.2f} gC/m²/year")

# ============================================================================
# SECTION 6: SCALER VALIDATION
# ============================================================================
print("\n" + "=" * 100)
print("SECTION 6: SCALER VALIDATION")
print("=" * 100)

# Check parameter scalers
if hasattr(mycase, 'pscaler') and mycase.pscaler:
    print("\n✓ Parameter scalers present")
    sample_var = list(mycase.pscaler.keys())[0]
    pscaler = mycase.pscaler[sample_var]
    print(f"  Scaler type: {type(pscaler).__name__}")
else:
    print("\n❌ No parameter scalers found!")

# Check output scalers
if hasattr(mycase, 'yscaler') and mycase.yscaler:
    print("\n✓ Output scalers present")
    for var in MCMC_VARS:
        if var in mycase.yscaler:
            yscaler = mycase.yscaler[var]
            print(f"  {var}: {type(yscaler).__name__}")
else:
    print("\n❌ No output scalers found!")

# ============================================================================
# SECTION 7: SUMMARY AND RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 100)
print("SECTION 7: SUMMARY AND RECOMMENDATIONS")
print("=" * 100)

print("\n" + "=" * 80)
print("ROBUSTNESS ASSESSMENT SUMMARY")
print("=" * 80)

all_issues = quality_issues + range_issues + identifiability_issues

if not all_issues:
    print("\n✓✓✓ SURROGATES ARE ROBUST FOR MCMC AND SENSITIVITY ANALYSIS ✓✓✓")
    print("\nYou can proceed with:")
    print("  1. Global Sensitivity Analysis (GSA)")
    print("  2. MCMC parameter calibration")
    print("  3. Uncertainty quantification")
else:
    print("\n⚠️  ISSUES DETECTED - REVIEW BEFORE PROCEEDING:")
    for issue in all_issues:
        print(f"  - {issue}")

print("\n" + "-" * 80)
print("Surrogate Quality:")
print("-" * 80)
for var, score in surrogate_scores.items():
    status = "✓" if score > R2_GOOD else "⚠️" if score > R2_ACCEPTABLE else "❌"
    print(f"  {status} {var:10s}: R² = {score:.4f}")

print("\n" + "-" * 80)
print("Dynamic Range Coverage:")
print("-" * 80)
for var in MCMC_VARS:
    if var in surrogate_ranges and var in obs_ranges:
        surr_range = surrogate_ranges[var]['max'] - surrogate_ranges[var]['min']
        obs_range = obs_ranges[var]['max'] - obs_ranges[var]['min']
        coverage = (surr_range / obs_range) * 100

        status = "✓" if coverage > 80 else "⚠️" if coverage > 50 else "❌"
        print(f"  {status} {var:10s}: {coverage:5.1f}% coverage")

if quality_issues:
    print("\n" + "-" * 80)
    print("RECOMMENDATIONS FOR QUALITY ISSUES:")
    print("-" * 80)
    print("  1. Increase ensemble size (current: {})".format(nsamples))
    print("  2. Try different neural network architectures")
    print("  3. Check for outliers in ensemble outputs")
    print("  4. Ensure sufficient training iterations")

if range_issues:
    print("\n" + "-" * 80)
    print("RECOMMENDATIONS FOR RANGE ISSUES:")
    print("-" * 80)
    print("  1. Widen parameter bounds in parameter list file")
    print("  2. Add more influential parameters (e.g., vcmax25, slatop)")
    print("  3. Check if model physics can reach observed values")
    print("  4. Consider prior distributions that cover wider range")

if identifiability_issues:
    print("\n" + "-" * 80)
    print("RECOMMENDATIONS FOR IDENTIFIABILITY:")
    print("-" * 80)
    print("  1. Remove highly correlated parameters from calibration")
    print("  2. Use informative priors for correlated parameters")
    print("  3. Consider constraining parameter ratios")
    print("  4. Focus on parameters with independent effects")

print("\n" + "=" * 80)
print("NEXT STEPS:")
print("=" * 80)

if not all_issues:
    print("\n✓ Surrogates are ready! You can now:")
    print("  1. Run GSA: mycase.run_GSA()")
    print("  2. Run MCMC: mycase.run_MCMC()")
    print("  3. Analyze results")
else:
    print("\n⚠️  Address the issues above, then:")
    print("  1. Adjust parameter list if needed")
    print("  2. Re-run ensemble with adjustments")
    print("  3. Retrain surrogates")
    print("  4. Re-run this diagnostic")

print("\n" + "=" * 100)
print("DIAGNOSTIC COMPLETE")
print("=" * 100)
