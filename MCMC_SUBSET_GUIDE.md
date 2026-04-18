# MCMC with Parameter Subset - User Guide

## Overview

The MCMC subset functionality allows you to calibrate only specific parameters while keeping others fixed at predetermined values. This is useful when:

1. **You have prior knowledge** about some parameters from literature or measurements
2. **You want to reduce MCMC dimensionality** for faster convergence
3. **You want to stage calibration** (e.g., calibrate roots first, then leaves)
4. **You trained surrogates with many parameters** but only want to calibrate a subset

## Key Concept

```
┌─────────────────────────────────────────┐
│  Ensemble/Surrogate Training            │
│  Uses ALL 21 parameters                 │
│  (17 TAM + 4 leaf parameters)           │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  MCMC Calibration (Subset)              │
│  Calibrate: leafcn, slatop, flnr (3)    │
│  Fixed: All 17 TAM parameters           │
│  Total: 3 calibrated + 18 fixed = 21    │
└─────────────────────────────────────────┘
```

**Important**: The surrogate models are trained on ALL parameters, but MCMC only varies a subset while keeping others fixed.

## Quick Start Example

### Example 1: Calibrate Only Leaf Parameters

```python
import sys
sys.path.append('..')
import model_ELM
import pickle
import numpy as np

# Integrate subset functionality (run once)
from model_ELM.MCMC_subset import run_MCMC_subset
model_ELM.main.ELMcase.run_MCMC_subset = run_MCMC_subset

# Load case with trained surrogates
with open('pklfiles/20251017_AU-Tum_ICB20TRCNPRDCTCBC.pkl', 'rb') as f:
    mycase = pickle.load(f)

# Define which parameters to calibrate
calibrate_params = ['leafcn', 'slatop', 'flnr', 'leafmr_base']

# Fix all TAM root parameters at reasonable values
fixed_params = {
    # Root C:N ratios
    'froottcn': 120,
    'frootacn': 85,
    'frootmcn': 17,

    # Root C:P ratios
    'froottcp': 700,
    'frootacp': 500,
    'frootmcp': 750,

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

# Run MCMC on subset
parms_best = mycase.run_MCMC_subset(
    calibrate_params=calibrate_params,
    fixed_params=fixed_params,
    myvars=['GPP', 'ER'],
    nevals=5000,
    nburn=1000,
    burnsteps=10,
    sampler='custom'
)

print("Best calibrated parameters:")
for pname in calibrate_params:
    idx = mycase.ensemble_parms.index(pname)
    print(f"  {pname}: {parms_best[idx]:.6f}")
```

### Example 2: Calibrate Only TAM C:N Ratios

```python
# Calibrate only root C:N ratios
calibrate_params = ['froottcn', 'frootacn', 'frootmcn']

# Fix everything else (including leaf parameters)
fixed_params = {
    # Leaf parameters
    'leafcn': 26,
    'slatop': 0.012,
    'flnr': 0.08,
    'leafmr_base': 1.5e-06,

    # Root C:P ratios
    'froottcp': 700,
    'frootacp': 500,
    'frootmcp': 750,

    # Root longevity
    'froott_long': 6.5,
    'froota_long': 1.0,
    'frootm_long': 0.5,

    # ... (all other parameters)
}

parms_best = mycase.run_MCMC_subset(
    calibrate_params=calibrate_params,
    fixed_params=fixed_params,
    myvars=['GPP'],
    nevals=5000,
    nburn=1000
)
```

## Detailed Usage

### Step 1: Prepare Your Case

Ensure you have:
1. **Trained surrogates** on ALL parameters
2. **Validated surrogate quality** (R² > 0.90)
3. **Observations** for calibration variables

```python
# Load case
with open('pklfiles/YOUR_CASE.pkl', 'rb') as f:
    mycase = pickle.load(f)

# Check surrogate is trained
print("Available surrogates:", list(mycase.surrogate.keys()))
print("Number of parameters:", mycase.nparms_ensemble)
print("Parameter names:", mycase.ensemble_parms)
```

### Step 2: Choose Parameters to Calibrate

**Decision criteria**:
- **High sensitivity** to calibration variables (GPP, ER)
- **Uncertain** from literature/measurements
- **Identifiable** from observations

```python
# Example: Choose parameters with >5% impact on GPP
from analyze_gsa import get_sensitive_parameters

sensitive_params = get_sensitive_parameters(mycase, 'GPP', threshold=0.05)
print("Highly sensitive parameters:", sensitive_params)

# Choose subset
calibrate_params = ['leafcn', 'slatop', 'flnr']  # Example
```

### Step 3: Set Fixed Parameter Values

**Strategies for fixed values**:

1. **Literature values** (if available)
2. **Mean of prior range** (neutral choice)
3. **Results from previous calibration**
4. **Site measurements**

```python
# Strategy 1: Use literature values
fixed_params = {
    'froottcn': 120,  # From Iversen et al. (2017)
    'frootacn': 85,   # From root trait database
    # ...
}

# Strategy 2: Use mean of prior range
fixed_params = {}
for pname in mycase.ensemble_parms:
    if pname not in calibrate_params:
        idx = mycase.ensemble_parms.index(pname)
        mean_val = (mycase.ensemble_pmin[idx] + mycase.ensemble_pmax[idx]) / 2
        fixed_params[pname] = mean_val

# Strategy 3: Use results from full calibration
# (if you ran full MCMC first)
previous_results = np.loadtxt('UQ_output/YOUR_CASE/MCMC_output/parms_best.txt')
fixed_params = {}
for i, pname in enumerate(mycase.ensemble_parms):
    if pname not in calibrate_params:
        fixed_params[pname] = previous_results[i, 2]  # Best value
```

### Step 4: Run MCMC Subset

```python
parms_best = mycase.run_MCMC_subset(
    calibrate_params=calibrate_params,
    fixed_params=fixed_params,
    myvars=['GPP', 'ER'],          # Calibration variables
    nevals=5000,                    # MCMC chain length
    nburn=1000,                     # Burn-in iterations
    burnsteps=10,                   # Steps per burn-in
    sampler='custom',               # Sampler type
    mcmc_type='uniform'             # Prior type
)
```

**Sampler options**:
- `'custom'`: Custom Metropolis-Hastings (default, reliable)
- `'custom_adaptive'`: Custom with enhanced adaptation
- `'pymc3'`: PyMC3 with NUTS sampler (if installed)

### Step 5: Analyze Results

```python
# Load results
from model_ELM.MCMC_subset import load_MCMC_subset_results

results = load_MCMC_subset_results(mycase.casename)

# Best parameters (full vector)
parms_best_full = results['parms_best_full']
print("\nBest parameters:")
for i, pname in enumerate(results['param_names']):
    status = results['status'][i]
    print(f"  {pname:20s}: {parms_best_full[i]:.6f} [{status}]")

# Posterior samples (calibrated params only)
posterior = results['posterior_samples']
print(f"\nPosterior shape: {posterior.shape}")
print(f"  {posterior.shape[0]} samples")
print(f"  {posterior.shape[1]} calibrated parameters")

# Calculate posterior statistics
import numpy as np
for i, pname in enumerate(results['calibrated_params']):
    mean = np.mean(posterior[:, i])
    std = np.std(posterior[:, i])
    q025 = np.percentile(posterior[:, i], 2.5)
    q975 = np.percentile(posterior[:, i], 97.5)

    print(f"\n{pname}:")
    print(f"  Mean ± SD: {mean:.4f} ± {std:.4f}")
    print(f"  95% CI: [{q025:.4f}, {q975:.4f}]")
```

## Common Scenarios

### Scenario 1: Two-Stage Calibration

First calibrate structural parameters, then physiological:

```python
# Stage 1: Calibrate TAM root parameters
calibrate_stage1 = ['froottcn', 'frootacn', 'frootmcn',
                    'froott_long', 'froota_long', 'frootm_long']
fixed_stage1 = {...}  # Fix leaf and other root params

parms_stage1 = mycase.run_MCMC_subset(
    calibrate_params=calibrate_stage1,
    fixed_params=fixed_stage1,
    myvars=['GPP'],
    nevals=3000
)

# Stage 2: Fix root params at Stage 1 values, calibrate leaf params
results1 = load_MCMC_subset_results(mycase.casename)
fixed_stage2 = {}
for pname in calibrate_stage1:
    idx = mycase.ensemble_parms.index(pname)
    fixed_stage2[pname] = parms_stage1[idx]

# Add other fixed params...

calibrate_stage2 = ['leafcn', 'slatop', 'flnr', 'leafmr_base']

parms_stage2 = mycase.run_MCMC_subset(
    calibrate_params=calibrate_stage2,
    fixed_params=fixed_stage2,
    myvars=['GPP', 'ER'],
    nevals=5000
)
```

### Scenario 2: Site-Specific vs Universal Parameters

Calibrate site-specific parameters, fix universal:

```python
# Universal parameters (from meta-analysis, fix across sites)
universal_params = {
    'leafcn': 28,
    'slatop': 0.011,
    # ... (parameters that don't vary much across sites)
}

# Calibrate site-specific parameters
site_specific = ['froottcn', 'frootacn', 'froota_long']

parms_best = mycase.run_MCMC_subset(
    calibrate_params=site_specific,
    fixed_params=universal_params,
    myvars=['GPP', 'ER'],
    nevals=5000
)
```

### Scenario 3: Parameter Sensitivity Threshold

Calibrate only parameters above sensitivity threshold:

```python
# Identify sensitive parameters from GSA
mycase.run_GSA(variables=['GPP', 'ER'], method='sobol')

sensitive_threshold = 0.05  # 5% main effect
calibrate_params = []

for i, pname in enumerate(mycase.ensemble_parms):
    # Check main effect sensitivity
    main_effect = mycase.sens_main['GPP'][i].mean()
    if main_effect > sensitive_threshold:
        calibrate_params.append(pname)

print(f"Calibrating {len(calibrate_params)} sensitive parameters")

# Fix all others at mean
fixed_params = {}
for pname in mycase.ensemble_parms:
    if pname not in calibrate_params:
        idx = mycase.ensemble_parms.index(pname)
        fixed_params[pname] = (mycase.ensemble_pmin[idx] +
                               mycase.ensemble_pmax[idx]) / 2

parms_best = mycase.run_MCMC_subset(
    calibrate_params=calibrate_params,
    fixed_params=fixed_params,
    myvars=['GPP'],
    nevals=5000
)
```

## Advantages of Subset MCMC

### 1. Faster Convergence
- **Fewer parameters** → Faster mixing
- **Lower dimensional** → Better exploration
- **Example**: 4 params vs 21 params → ~5x faster convergence

### 2. Better Identifiability
- **Reduced correlation** between calibrated parameters
- **Clearer interpretation** of posterior
- **More precise estimates** for calibrated parameters

### 3. Incorporating Prior Knowledge
- **Fix well-known** parameters from literature
- **Focus calibration** on uncertain parameters
- **Scientifically defensible** constraints

### 4. Computational Efficiency
- **Shorter chains** needed for convergence
- **Lower acceptance rate** is acceptable
- **Faster surrogate evaluation** (same speed, but fewer iterations)

## Best Practices

### 1. Choice of Fixed Values

✓ **Good practices**:
- Use literature values when available
- Use site measurements if available
- Document sources for fixed values
- Test sensitivity to fixed values

❌ **Avoid**:
- Arbitrary fixed values without justification
- Fixing highly sensitive parameters
- Ignoring parameter interactions

### 2. Validation

After subset MCMC, validate:

```python
# 1. Check posterior predictions
posterior_samples = results['posterior_samples']

# Run surrogate with posterior samples
# (need to expand to full parameter vector)
full_samples = np.zeros((len(posterior_samples), mycase.nparms_ensemble))
for i, pname in enumerate(calibrate_params):
    idx_full = mycase.ensemble_parms.index(pname)
    full_samples[:, idx_full] = posterior_samples[:, i]

# Set fixed parameters
for pname, val in fixed_params.items():
    idx_full = mycase.ensemble_parms.index(pname)
    full_samples[:, idx_full] = val

# Get predictions
predictions = mycase.run_surrogate(full_samples, ['GPP', 'ER'])

# Compare to observations
# ...

# 2. Check posterior parameter ranges
for i, pname in enumerate(calibrate_params):
    idx = mycase.ensemble_parms.index(pname)
    post_samples = posterior_samples[:, i]

    # Check if parameters hit boundaries
    if np.min(post_samples) <= mycase.ensemble_pmin[idx] * 1.01:
        print(f"⚠️  {pname} hitting lower boundary")
    if np.max(post_samples) >= mycase.ensemble_pmax[idx] * 0.99:
        print(f"⚠️  {pname} hitting upper boundary")
```

### 3. Sensitivity Analysis

Before fixing parameters, check their sensitivity:

```python
# Run GSA if not already done
if not hasattr(mycase, 'sens_main'):
    mycase.run_GSA(variables=['GPP', 'ER'])

# Check main effects
for pname in fixed_params.keys():
    idx = mycase.ensemble_parms.index(pname)
    gpp_sensitivity = mycase.sens_main['GPP'][idx].mean()

    if gpp_sensitivity > 0.05:  # 5% threshold
        print(f"⚠️  Fixing {pname} with {gpp_sensitivity*100:.1f}% GPP sensitivity")
```

## Troubleshooting

### Issue 1: "Invalid calibration parameters"

**Cause**: Parameter name not in `self.ensemble_parms`

**Solution**: Check exact parameter names:
```python
print("Available parameters:", mycase.ensemble_parms)
```

### Issue 2: "Parameters cannot be both calibrated and fixed"

**Cause**: Parameter appears in both lists

**Solution**: Remove from one list:
```python
# Check overlap
calibrate_set = set(calibrate_params)
fixed_set = set(fixed_params.keys())
overlap = calibrate_set & fixed_set
print("Overlapping parameters:", overlap)
```

### Issue 3: Poor mixing / Low acceptance rate

**Cause**: Too many calibrated parameters or poor initial values

**Solution**:
1. Reduce number of calibrated parameters
2. Run longer burn-in
3. Use adaptive sampler

```python
parms_best = mycase.run_MCMC_subset(
    calibrate_params=calibrate_params,
    fixed_params=fixed_params,
    myvars=['GPP', 'ER'],
    nevals=10000,  # Longer chain
    nburn=2000,    # Longer burn-in
    sampler='custom_adaptive'  # Adaptive proposals
)
```

### Issue 4: Parameters hitting boundaries

**Cause**: Prior range too narrow or fixed values inconsistent

**Solution**:
1. Widen prior ranges
2. Relax some fixed parameters
3. Check physical consistency

## Output Files

After running MCMC subset, these files are created:

```
UQ_output/YOUR_CASE/MCMC_output/
├── MCMC_chain.txt              # Posterior samples (calibrated params only)
├── parms_best.txt              # Best parameters (calibrated params only)
├── parms_best_full.txt         # Best parameters (FULL vector with fixed)
├── subset_info.txt             # Configuration (which params calibrated/fixed)
├── parms_95pctconf.txt         # 95% confidence intervals
└── plots/
    ├── chains/                 # Trace plots (calibrated params only)
    └── pdfs/                   # Posterior distributions
```

**Key file**: `parms_best_full.txt` contains the complete parameter vector:
```
# parameter_name pft value [calibrated/fixed]
leafcn 0 2.687654e+01 calibrated
slatop 0 1.234567e-02 calibrated
flnr 0 8.765432e-02 calibrated
froottcn 0 1.200000e+02 fixed
froott_long 0 6.500000e+00 fixed
...
```

## Examples in Repository

See the `examples/mcmc_subset/` directory for complete examples:
- `example_leaf_calibration.py` - Calibrate leaf parameters only
- `example_root_calibration.py` - Calibrate root parameters only
- `example_two_stage.py` - Two-stage calibration workflow
- `example_sensitivity_based.py` - Calibrate based on GSA results

## References

- Metropolis-Hastings algorithm: Metropolis et al. (1953), Hastings (1970)
- Adaptive MCMC: Haario et al. (2001)
- Parameter subset calibration: Houska et al. (2014)

---

**Author**: OLMT Development Team
**Last updated**: 2025-10-31
**Requires**: OLMT with trained surrogate models
