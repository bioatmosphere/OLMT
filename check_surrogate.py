#!/usr/bin/env python
"""
Check surrogate model quality for AU-Tum case
"""
import pickle
import numpy as np
import sys

# Load the case
pkl_file = '/autofs/nccsopen-svm1_home/6lw/models/OLMT/pklfiles/20251029_AU-Tum_ICB20TRCNPRDCTCBC.pkl'

print(f"Loading case from: {pkl_file}")
with open(pkl_file, 'rb') as f:
    mycase = pickle.load(f)

print("\n" + "="*80)
print("SURROGATE MODEL DIAGNOSTICS")
print("="*80)

# Check if surrogate exists
if not hasattr(mycase, 'surrogate') or not mycase.surrogate:
    print("ERROR: No surrogate models found!")
    sys.exit(1)

print(f"\nSurrogate models available for: {list(mycase.surrogate.keys())}")

# Variables used in MCMC
mcmc_vars = ['GPP', 'ER']
print(f"\nVariables used in MCMC: {mcmc_vars}")

# Check each variable
for var in mcmc_vars:
    if var not in mycase.surrogate:
        print(f"\n❌ MISSING: {var} not in surrogate models!")
        continue

    print(f"\n{'='*80}")
    print(f"Variable: {var}")
    print(f"{'='*80}")

    # Get surrogate info
    surr = mycase.surrogate[var]

    # Check if it's a list (one per year) or single model
    if isinstance(surr, list):
        print(f"  Type: Multi-year surrogate ({len(surr)} years)")

        # Check each year
        for year_idx, year_model in enumerate(surr):
            if hasattr(year_model, 'score'):
                # Try to get R² score if available
                try:
                    # Assuming we have access to test data
                    print(f"  Year {year_idx}: Model type = {type(year_model).__name__}")
                except:
                    print(f"  Year {year_idx}: Model exists (type: {type(year_model).__name__})")
            else:
                print(f"  Year {year_idx}: Model exists (type: {type(year_model).__name__})")
    else:
        print(f"  Type: Single surrogate model ({type(surr).__name__})")

# Test surrogate with initial parameters
print(f"\n{'='*80}")
print("TESTING SURROGATE WITH SAMPLE PARAMETERS")
print(f"{'='*80}")

# Get parameter info
print(f"\nNumber of parameters: {mycase.nparms_ensemble}")
print(f"Parameter names: {mycase.ensemble_parms}")

# Create test parameter vector (midpoint of bounds)
test_parms = np.array([(mycase.ensemble_pmin[i] + mycase.ensemble_pmax[i])/2
                        for i in range(mycase.nparms_ensemble)])
print(f"\nTest parameters (midpoint of bounds):")
for i, (pname, pval) in enumerate(zip(mycase.ensemble_parms, test_parms)):
    print(f"  {pname:20s}: {pval:10.4f}  (range: [{mycase.ensemble_pmin[i]:.2f}, {mycase.ensemble_pmax[i]:.2f}])")

# Run surrogate
print(f"\nRunning surrogate for {mcmc_vars}...")
try:
    output = mycase.run_surrogate(test_parms.reshape(1, -1), mcmc_vars)
    print(f"\n✓ Surrogate executed successfully!")
    print(f"\nSurrogate outputs:")
    for var in mcmc_vars:
        if var in output:
            val = output[var]
            if isinstance(val, np.ndarray):
                print(f"  {var:10s}: shape={val.shape}, values={val.flatten()[:3]}..." if len(val.flatten()) > 3 else f"  {var:10s}: {val}")
            else:
                print(f"  {var:10s}: {val}")
except Exception as e:
    print(f"\n❌ ERROR running surrogate: {e}")
    import traceback
    traceback.print_exc()

# Test with parameter variations to check sensitivity
print(f"\n{'='*80}")
print("TESTING PARAMETER SENSITIVITY")
print(f"{'='*80}")

# Test each parameter individually
print(f"\nVarying each parameter ±10% while keeping others at midpoint:")
for p_idx in range(min(5, mycase.nparms_ensemble)):  # Test first 5 parameters
    pname = mycase.ensemble_parms[p_idx]

    # Baseline
    parms_base = test_parms.copy()
    output_base = mycase.run_surrogate(parms_base.reshape(1, -1), mcmc_vars)

    # +10%
    parms_plus = test_parms.copy()
    parms_plus[p_idx] *= 1.1
    parms_plus[p_idx] = min(parms_plus[p_idx], mycase.ensemble_pmax[p_idx])
    output_plus = mycase.run_surrogate(parms_plus.reshape(1, -1), mcmc_vars)

    # -10%
    parms_minus = test_parms.copy()
    parms_minus[p_idx] *= 0.9
    parms_minus[p_idx] = max(parms_minus[p_idx], mycase.ensemble_pmin[p_idx])
    output_minus = mycase.run_surrogate(parms_minus.reshape(1, -1), mcmc_vars)

    print(f"\n  {pname:20s}:")
    for var in mcmc_vars:
        base_val = output_base[var].flatten()[0]
        plus_val = output_plus[var].flatten()[0]
        minus_val = output_minus[var].flatten()[0]

        # Calculate sensitivity
        sensitivity = (plus_val - minus_val) / (2 * 0.1 * test_parms[p_idx])
        pct_change = ((plus_val - minus_val) / base_val) * 100 if base_val != 0 else 0

        print(f"    {var:10s}: base={base_val:10.2f}, +10%={plus_val:10.2f}, -10%={minus_val:10.2f}, "
              f"sensitivity={sensitivity:10.4f}, Δ={pct_change:6.2f}%")

# Check for parameter correlations in surrogate response
print(f"\n{'='*80}")
print("CHECKING FOR PARAMETER CORRELATION ISSUES")
print(f"{'='*80}")

# Test if changing multiple parameters gives same result (perfect correlation)
print(f"\nTesting if froottcp and frootmcp are perfectly correlated in surrogate...")
if 'froottcp' in mycase.ensemble_parms and 'frootmcp' in mycase.ensemble_parms:
    idx_tcp = mycase.ensemble_parms.index('froottcp')
    idx_mcp = mycase.ensemble_parms.index('frootmcp')

    # Test 1: Increase both by 10%
    parms_both = test_parms.copy()
    parms_both[idx_tcp] *= 1.1
    parms_both[idx_mcp] *= 1.1
    output_both = mycase.run_surrogate(parms_both.reshape(1, -1), mcmc_vars)

    # Test 2: Increase tcp, decrease mcp
    parms_opposite = test_parms.copy()
    parms_opposite[idx_tcp] *= 1.1
    parms_opposite[idx_mcp] *= 0.9
    output_opposite = mycase.run_surrogate(parms_opposite.reshape(1, -1), mcmc_vars)

    # Test 3: Just tcp
    parms_tcp = test_parms.copy()
    parms_tcp[idx_tcp] *= 1.1
    output_tcp = mycase.run_surrogate(parms_tcp.reshape(1, -1), mcmc_vars)

    # Test 4: Just mcp
    parms_mcp = test_parms.copy()
    parms_mcp[idx_mcp] *= 1.1
    output_mcp = mycase.run_surrogate(parms_mcp.reshape(1, -1), mcmc_vars)

    print(f"\nResults for GPP:")
    print(f"  Baseline:              {output_base['GPP'].flatten()[0]:.2f}")
    print(f"  Both +10%:             {output_both['GPP'].flatten()[0]:.2f}")
    print(f"  tcp+10%, mcp-10%:      {output_opposite['GPP'].flatten()[0]:.2f}")
    print(f"  Only tcp +10%:         {output_tcp['GPP'].flatten()[0]:.2f}")
    print(f"  Only mcp +10%:         {output_mcp['GPP'].flatten()[0]:.2f}")

    # Check if tcp and mcp have identical effects (perfect correlation)
    if abs(output_tcp['GPP'].flatten()[0] - output_mcp['GPP'].flatten()[0]) < 1e-6:
        print(f"\n  ⚠️  WARNING: froottcp and frootmcp have IDENTICAL effects on GPP!")
        print(f"      This explains the perfect correlation (r=1.000) in MCMC!")
else:
    print(f"  froottcp or frootmcp not found in parameters")

print(f"\n{'='*80}")
print("DIAGNOSIS COMPLETE")
print(f"{'='*80}")
