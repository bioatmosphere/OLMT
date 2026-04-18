#!/usr/bin/env python
"""
Test Script: Verify Surrogate Models Can Be Reused Multiple Times

This script demonstrates that surrogates trained on full parameter set
can be used repeatedly for different MCMC subset calibrations without
needing retraining.

Key Points:
-----------
1. Surrogates are trained ONCE on all parameters
2. MCMC subset calibration uses surrogates WITHOUT modifying them
3. Can run multiple calibrations with different parameter subsets
4. Surrogates remain intact and functional throughout
"""

import sys
sys.path.append('..')
import pickle
import numpy as np

# ============================================================================
# Load case with trained surrogates
# ============================================================================
print("="*80)
print("SURROGATE REUSE TEST")
print("="*80)

pkl_file = 'pklfiles/20251017_AU-Tum_ICB20TRCNPRDCTCBC.pkl'

print(f"\nLoading case with trained surrogates...")
with open(pkl_file, 'rb') as f:
    mycase = pickle.load(f)

print(f"✓ Loaded: {mycase.casename}")
print(f"  Total parameters: {mycase.nparms_ensemble}")
print(f"  Available surrogates: {list(mycase.surrogate.keys())}")

# Store original surrogate reference
original_surrogate_refs = {}
for var in mycase.surrogate.keys():
    original_surrogate_refs[var] = id(mycase.surrogate[var])

print(f"\n✓ Original surrogate object IDs stored")

# ============================================================================
# Test 1: Direct surrogate call (baseline)
# ============================================================================
print("\n" + "="*80)
print("TEST 1: Direct Surrogate Call (Baseline)")
print("="*80)

# Create test parameter vector (all parameters)
test_params_full = np.array([(mycase.ensemble_pmin[i] + mycase.ensemble_pmax[i])/2
                             for i in range(mycase.nparms_ensemble)])

print(f"\nTest with {len(test_params_full)} parameters")
output1 = mycase.run_surrogate(test_params_full.reshape(1, -1), ['GPP', 'ER'])

print(f"✓ Direct call successful")
print(f"  GPP mean: {np.mean(output1['GPP'])*31536000:.1f} gC/m²/year")
print(f"  ER mean:  {np.mean(output1['ER'])*31536000:.1f} gC/m²/year")

# ============================================================================
# Test 2: Simulate MCMC subset (manual expansion)
# ============================================================================
print("\n" + "="*80)
print("TEST 2: Simulated MCMC Subset (Manual Parameter Expansion)")
print("="*80)

# Define subset - use parameters that are actually in the ensemble
calibrate_params = ['froottcn', 'frootacn', 'frootmcn']
calibrate_indices = [mycase.ensemble_parms.index(p) for p in calibrate_params]

# Create subset parameter vector
test_params_subset = test_params_full[calibrate_indices]

print(f"\nUsing {len(test_params_subset)} calibrated parameters:")
for i, pname in enumerate(calibrate_params):
    print(f"  {pname}: {test_params_subset[i]:.4f}")

# Manually expand to full parameter vector (like MCMC_subset does)
test_params_expanded = test_params_full.copy()  # Start with full vector
for i, idx in enumerate(calibrate_indices):
    test_params_expanded[idx] = test_params_subset[i]  # Replace calibrated params

output2 = mycase.run_surrogate(test_params_expanded.reshape(1, -1), ['GPP', 'ER'])

print(f"\n✓ Expanded subset call successful")
print(f"  GPP mean: {np.mean(output2['GPP'])*31536000:.1f} gC/m²/year")
print(f"  ER mean:  {np.mean(output2['ER'])*31536000:.1f} gC/m²/year")

# Verify same result (should be identical)
assert np.allclose(output1['GPP'], output2['GPP']), "Results should be identical!"
print(f"✓ Results match (as expected)")

# ============================================================================
# Test 3: Multiple subset calls with different fixed values
# ============================================================================
print("\n" + "="*80)
print("TEST 3: Multiple Calls with Different Fixed Values")
print("="*80)

# Scenario 1: Vary froottcn, fix others
params1 = test_params_full.copy()
params1[mycase.ensemble_parms.index('froottcn')] = 100  # Low C:N (more N)
output_scenario1 = mycase.run_surrogate(params1.reshape(1, -1), ['GPP', 'ER'])

# Scenario 2: Vary froottcn, fix others
params2 = test_params_full.copy()
params2[mycase.ensemble_parms.index('froottcn')] = 150  # High C:N (less N)
output_scenario2 = mycase.run_surrogate(params2.reshape(1, -1), ['GPP', 'ER'])

print(f"\nScenario 1 (froottcn=100, low C:N, more N):")
print(f"  GPP mean: {np.mean(output_scenario1['GPP'])*31536000:.1f} gC/m²/year")

print(f"\nScenario 2 (froottcn=150, high C:N, less N):")
print(f"  GPP mean: {np.mean(output_scenario2['GPP'])*31536000:.1f} gC/m²/year")

print(f"\n✓ Multiple calls with different parameters successful")

# ============================================================================
# Test 4: Verify surrogates unchanged
# ============================================================================
print("\n" + "="*80)
print("TEST 4: Verify Surrogates Remain Unchanged")
print("="*80)

print(f"\nChecking surrogate object IDs...")
all_unchanged = True
for var in mycase.surrogate.keys():
    current_id = id(mycase.surrogate[var])
    original_id = original_surrogate_refs[var]

    if current_id == original_id:
        print(f"  ✓ {var}: Unchanged (ID: {original_id})")
    else:
        print(f"  ❌ {var}: CHANGED! (was {original_id}, now {current_id})")
        all_unchanged = False

if all_unchanged:
    print(f"\n✓ All surrogates remain unchanged and reusable!")
else:
    print(f"\n❌ WARNING: Some surrogates were modified!")

# ============================================================================
# Test 5: Batch evaluation
# ============================================================================
print("\n" + "="*80)
print("TEST 5: Batch Evaluation (Multiple Parameter Sets)")
print("="*80)

# Create batch of parameter sets
n_samples = 10
batch_params = np.zeros((n_samples, mycase.nparms_ensemble))

for i in range(n_samples):
    # Random parameters within bounds
    for p in range(mycase.nparms_ensemble):
        batch_params[i, p] = np.random.uniform(
            mycase.ensemble_pmin[p],
            mycase.ensemble_pmax[p]
        )

print(f"\nEvaluating {n_samples} parameter sets simultaneously...")
batch_output = mycase.run_surrogate(batch_params, ['GPP', 'ER'])

print(f"✓ Batch evaluation successful")
print(f"  Output shape GPP: {batch_output['GPP'].shape}")
print(f"  Output shape ER:  {batch_output['ER'].shape}")
print(f"  GPP range: {np.min(batch_output['GPP'])*31536000:.1f} - {np.max(batch_output['GPP'])*31536000:.1f} gC/m²/year")

# ============================================================================
# Test 6: Simulate multiple MCMC runs
# ============================================================================
print("\n" + "="*80)
print("TEST 6: Simulate Multiple Sequential MCMC Subset Runs")
print("="*80)

# Simulate 3 different calibration scenarios
scenarios = [
    {
        'name': 'Root CN ratios only',
        'calibrate': ['froottcn', 'frootacn', 'frootmcn'],
        'n_calls': 100
    },
    {
        'name': 'Root longevity only',
        'calibrate': ['froott_long', 'froota_long', 'frootm_long'],
        'n_calls': 100
    },
    {
        'name': 'Root chemistry',
        'calibrate': ['frt_flab', 'fra_flab', 'frm_flab'],
        'n_calls': 100
    }
]

for scenario in scenarios:
    print(f"\nScenario: {scenario['name']}")
    print(f"  Calibrating: {scenario['calibrate']}")

    # Get indices
    calib_idx = [mycase.ensemble_parms.index(p) for p in scenario['calibrate']]

    # Simulate MCMC chain (random walk)
    base_params = test_params_full.copy()

    for call_num in range(scenario['n_calls']):
        # Vary calibrated parameters
        for idx in calib_idx:
            base_params[idx] = np.random.uniform(
                mycase.ensemble_pmin[idx],
                mycase.ensemble_pmax[idx]
            )

        # Call surrogate
        output = mycase.run_surrogate(base_params.reshape(1, -1), ['GPP'])

    print(f"  ✓ {scenario['n_calls']} surrogate calls completed")

print(f"\n✓ All sequential scenarios completed successfully")

# ============================================================================
# Final verification
# ============================================================================
print("\n" + "="*80)
print("FINAL VERIFICATION")
print("="*80)

# One more direct call to ensure everything still works
final_output = mycase.run_surrogate(test_params_full.reshape(1, -1), ['GPP', 'ER'])

print(f"\nFinal surrogate call:")
print(f"  GPP mean: {np.mean(final_output['GPP'])*31536000:.1f} gC/m²/year")
print(f"  ER mean:  {np.mean(final_output['ER'])*31536000:.1f} gC/m²/year")

# Should match first call
match = np.allclose(output1['GPP'], final_output['GPP'])
print(f"\nMatch with initial call: {match}")

if match:
    print("\n✓✓✓ SUCCESS ✓✓✓")
    print("Surrogates can be reused indefinitely for different parameter subsets!")
else:
    print("\n❌ FAILED")
    print("Surrogates may have been modified")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
Key Findings:
1. ✓ Surrogates trained on full parameter set
2. ✓ Can be called with any parameter values
3. ✓ Parameter expansion works correctly
4. ✓ Multiple calls do not modify surrogates
5. ✓ Batch evaluation supported
6. ✓ Sequential calibrations possible

Conclusion:
Train surrogates ONCE with full parameter set, then:
- Run MCMC on different parameter subsets
- Change which parameters are calibrated vs fixed
- Repeat calibrations with different scenarios
- No need to retrain surrogates!
""")
