# MCMC Subset Calibration - Quick Start

## What This Does

Allows you to **calibrate only some parameters** while keeping others fixed, using surrogates trained on **all parameters**.

**Key advantage**: Train surrogates once, use for many different calibration scenarios.

## Complete Example (5 Minutes)

```python
import pickle
import sys
sys.path.append('..')

# ============================================================================
# STEP 1: Integrate the functionality (do once)
# ============================================================================
from model_ELM.MCMC_subset import run_MCMC_subset
import model_ELM
model_ELM.main.ELMcase.run_MCMC_subset = run_MCMC_subset

# ============================================================================
# STEP 2: Load your case with trained surrogates
# ============================================================================
with open('pklfiles/20251017_AU-Tum_ICB20TRCNPRDCTCBC.pkl', 'rb') as f:
    mycase = pickle.load(f)

print(f"Loaded: {mycase.casename}")
print(f"Parameters: {mycase.nparms_ensemble}")
print(f"Surrogates: {list(mycase.surrogate.keys())}")

# ============================================================================
# STEP 3: Define what to calibrate and what to fix
# ============================================================================

# Calibrate only leaf parameters (4 out of 21)
calibrate_params = ['leafcn', 'slatop', 'flnr', 'leafmr_base']

# Fix all TAM root parameters (17 parameters)
fixed_params = {
    # Root C:N ratios
    'froottcn': 120.0,
    'frootacn': 85.0,
    'frootmcn': 17.0,

    # Root C:P ratios
    'froottcp': 700.0,
    'frootacp': 500.0,
    'frootmcp': 750.0,

    # Root longevity
    'froott_long': 6.5,
    'froota_long': 1.0,
    'frootm_long': 0.5,

    # Root chemistry
    'frt_flab': 0.23,
    'frt_fcel': 0.50,
    'fra_flab': 0.23,
    'fra_fcel': 0.50,
    'frm_flab': 0.23,
    'frm_fcel': 0.50,

    # Root-leaf allocation
    'froott_leaf': 0.22,
    'froota_leaf': 0.40
}

# ============================================================================
# STEP 4: Run MCMC calibration
# ============================================================================

parms_best = mycase.run_MCMC_subset(
    calibrate_params=calibrate_params,   # What to calibrate
    fixed_params=fixed_params,            # What to fix
    myvars=['GPP', 'ER'],                 # Calibration targets
    nevals=5000,                          # MCMC chain length
    nburn=1000,                           # Burn-in
    burnsteps=10,
    sampler='custom'
)

print("\nCalibration complete!")
print("Best parameters:", parms_best)

# ============================================================================
# DONE! Results saved to ./UQ_output/[casename]/MCMC_output/
# ============================================================================
```

## What Gets Saved

```
UQ_output/YOUR_CASE/MCMC_output/
├── parms_best_full.txt         # Best parameters (full vector, marks calibrated/fixed)
├── MCMC_chain.txt              # Posterior samples (calibrated params only)
├── subset_info.txt             # Which params were calibrated vs fixed
└── plots/
    ├── chains/                 # Trace plots for calibrated params
    └── pdfs/                   # Posterior distributions
```

## Run Multiple Calibrations

**The key point**: You can run multiple calibrations **without retraining surrogates**!

```python
# First calibration: Leaf parameters
calibrate1 = ['leafcn', 'slatop', 'flnr']
fixed1 = {... all TAM params ...}

mycase.run_MCMC_subset(calibrate1, fixed1, ['GPP', 'ER'], 5000)

# Second calibration: Root CN ratios
# NO RELOADING OR RETRAINING NEEDED!
calibrate2 = ['froottcn', 'frootacn', 'frootmcn']
fixed2 = {
    'leafcn': 26,  # Use result from first calibration
    'slatop': 0.012,
    # ... other params
}

mycase.run_MCMC_subset(calibrate2, fixed2, ['GPP'], 5000)

# Third calibration: Different scenario
# STILL using same surrogates!
calibrate3 = ['leafcn', 'froottcn']
fixed3 = {... different values ...}

mycase.run_MCMC_subset(calibrate3, fixed3, ['GPP', 'ER'], 5000)
```

## Verify Surrogates Work

Run the test script to verify everything works:

```bash
python test_surrogate_reuse.py
```

Expected output:
```
✓✓✓ SUCCESS ✓✓✓
Surrogates can be reused indefinitely for different parameter subsets!
```

## When to Use This

**Use MCMC subset when**:
- ✓ You have prior knowledge about some parameters
- ✓ You want faster MCMC convergence (fewer parameters)
- ✓ You want to stage calibration (roots first, then leaves)
- ✓ Some parameters are poorly constrained by data

**Don't use when**:
- ❌ You want to calibrate all parameters simultaneously
- ❌ You don't have good estimates for fixed parameters
- ❌ Fixed parameters are highly sensitive (should calibrate those)

## Common Scenarios

### Scenario 1: Calibrate Leaf, Fix Roots

```python
calibrate = ['leafcn', 'slatop', 'flnr', 'leafmr_base']
fixed = {... all 17 TAM root parameters ...}
```

**Use case**: You trust literature values for root parameters, want to focus on leaf physiology.

### Scenario 2: Calibrate Roots, Fix Leaves

```python
calibrate = ['froottcn', 'frootacn', 'frootmcn',
             'froott_long', 'froota_long', 'frootm_long']
fixed = {'leafcn': 26, 'slatop': 0.012, 'flnr': 0.08, ...}
```

**Use case**: You have leaf measurements from site, want to calibrate root traits.

### Scenario 3: Calibrate High-Sensitivity Only

```python
# After running GSA
sensitive = get_params_above_threshold(mycase, 'GPP', 0.05)

calibrate = sensitive  # e.g., ['leafcn', 'slatop', 'froottcn']
fixed = {... all others at mean values ...}
```

**Use case**: Focus calibration on parameters that matter most.

## Troubleshooting

### Error: "Invalid calibration parameters"
**Fix**: Check parameter names match exactly
```python
print(mycase.ensemble_parms)  # See available parameters
```

### Error: "Parameters cannot be both calibrated and fixed"
**Fix**: Remove parameter from one of the lists
```python
calibrate_set = set(calibrate_params)
fixed_set = set(fixed_params.keys())
print("Overlap:", calibrate_set & fixed_set)  # Should be empty
```

### Low acceptance rate (<10%)
**Fix**: Use adaptive sampler or reduce parameters
```python
mycase.run_MCMC_subset(..., sampler='custom_adaptive', nburn=2000)
```

## Next Steps

1. **Read full guide**: `MCMC_SUBSET_GUIDE.md`
2. **Understand reusability**: `SURROGATE_REUSE_README.md`
3. **See complete example**: `example_mcmc_subset_AU-Tum.py`
4. **Test your setup**: `python test_surrogate_reuse.py`

## Key Takeaway

```
Train surrogates ONCE with all parameters
    ↓
Use MANY TIMES for different calibration scenarios
    ↓
No retraining needed!
```

---

**Ready to go?** Just run the example above with your case file!
