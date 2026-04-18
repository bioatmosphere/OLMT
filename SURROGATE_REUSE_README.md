# Surrogate Model Reusability for MCMC Subset Calibration

## Key Concept: Train Once, Use Many Times

```
┌─────────────────────────────────────────────────────────────┐
│  TRAIN SURROGATES ONCE (Full Parameter Set)                │
│  • 21 parameters (or however many you have)                 │
│  • 1000-1500 ensemble members                               │
│  • Takes hours/days                                         │
│  • DO THIS ONLY ONCE                                        │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │  Surrogates saved in case.pkl
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  USE SURROGATES REPEATEDLY (Parameter Subsets)             │
│  • Run 1: Calibrate leafcn, slatop, flnr                   │
│  • Run 2: Calibrate froottcn, frootacn, frootmcn           │
│  • Run 3: Calibrate different subset                       │
│  • Takes minutes/hours per run                             │
│  • NO RETRAINING NEEDED                                    │
└─────────────────────────────────────────────────────────────┘
```

## How It Works

### 1. Surrogates Are Unchanged

The MCMC subset functionality **does NOT modify** your trained surrogates. It only:
- Wraps the surrogate call temporarily
- Expands subset parameters to full parameter vector
- Restores original state after MCMC completes

```python
# Surrogates are trained on ALL parameters
mycase.train_surrogate(
    variables=['GPP', 'ER', 'NEE', 'NPP'],
    hidden_layer_sizes=(100, 50),
    # ... uses ALL 21 parameters
)

# Save case with surrogates
import pickle
with open('mycase.pkl', 'wb') as f:
    pickle.dump(mycase, f)

# Later: Load and use for different calibrations
# NO RETRAINING NEEDED!
```

### 2. Parameter Expansion During MCMC

When you run MCMC subset:

```python
# You specify SUBSET to calibrate
calibrate_params = ['leafcn', 'slatop', 'flnr']  # 3 params

# And which to FIX
fixed_params = {
    'froottcn': 120,
    'frootacn': 85,
    # ... 18 more parameters
}

# MCMC internally does:
# For each MCMC iteration:
#   1. MCMC proposes new values for leafcn, slatop, flnr
#   2. Create FULL parameter vector [21 values]
#   3. Fill in: leafcn, slatop, flnr from MCMC
#   4. Fill in: froottcn=120, frootacn=85, ... (fixed)
#   5. Call surrogate with FULL vector
#   6. Get GPP, ER predictions
#   7. Calculate posterior probability
```

**Key point**: Surrogate always receives full 21-parameter vector. It doesn't know or care that some are fixed.

### 3. Multiple Calibrations Workflow

```python
import pickle

# ==== ONE-TIME SETUP ====
# Load case with trained surrogates
with open('pklfiles/AU-Tum.pkl', 'rb') as f:
    mycase = pickle.load(f)

# Verify surrogates exist
print("Surrogates:", list(mycase.surrogate.keys()))
# Output: ['GPP', 'ER', 'NEE', 'NPP', ...]

# ==== CALIBRATION 1: Leaf parameters ====
calibrate1 = ['leafcn', 'slatop', 'flnr', 'leafmr_base']
fixed1 = {... all TAM root params ...}

mycase.run_MCMC_subset(
    calibrate_params=calibrate1,
    fixed_params=fixed1,
    myvars=['GPP', 'ER'],
    nevals=5000
)
# Results saved to: UQ_output/.../MCMC_output/

# ==== CALIBRATION 2: Root CN ratios ====
# NO RELOADING OR RETRAINING NEEDED
# Surrogates are still there and unchanged!

calibrate2 = ['froottcn', 'frootacn', 'frootmcn']
fixed2 = {
    'leafcn': 26,      # Use result from Calibration 1
    'slatop': 0.012,   # Use result from Calibration 1
    'flnr': 0.08,      # Use result from Calibration 1
    # ... other params
}

mycase.run_MCMC_subset(
    calibrate_params=calibrate2,
    fixed_params=fixed2,
    myvars=['GPP'],
    nevals=5000
)

# ==== CALIBRATION 3: Different scenario ====
# STILL using same surrogates!

calibrate3 = ['leafcn', 'froottcn', 'froota_long']
fixed3 = {... different fixed values ...}

mycase.run_MCMC_subset(
    calibrate_params=calibrate3,
    fixed_params=fixed3,
    myvars=['GPP', 'ER'],
    nevals=5000
)

# All three calibrations use THE SAME surrogates
# No retraining required!
```

## Verification Test

Run this to verify surrogates are reusable:

```bash
python test_surrogate_reuse.py
```

This will:
1. Load your case with trained surrogates
2. Make multiple calls with different parameter subsets
3. Verify surrogates remain unchanged
4. Confirm results are consistent

Expected output:
```
✓ All surrogates remain unchanged and reusable!
✓✓✓ SUCCESS ✓✓✓
Surrogates can be reused indefinitely for different parameter subsets!
```

## Why This Works

### Surrogate Models Store Everything

When you train surrogates, they store:
1. **Neural network weights** - The trained model
2. **Parameter scalers** - How to normalize inputs
3. **Output scalers** - How to normalize outputs
4. **Training configuration** - Architecture, activation functions, etc.

None of these are modified by MCMC subset calibration.

### MCMC Only Changes Bookkeeping

During MCMC subset:
- `mycase.ensemble_parms` temporarily modified (just metadata)
- `mycase.nparms_ensemble` temporarily changed (just a number)
- Surrogate wrapper temporarily added (just function redirect)

**After MCMC completes**:
- All attributes restored to original values
- Wrapper removed
- Surrogates completely unchanged

### Code Implementation

Here's what happens under the hood:

```python
def run_MCMC_subset(self, calibrate_params, fixed_params, ...):
    # 1. STORE original attributes
    original_surrogate = self.run_surrogate  # Save reference
    original_parms = self.ensemble_parms      # Save list
    original_nparms = self.nparms_ensemble    # Save count

    try:
        # 2. CREATE wrapper function
        def surrogate_wrapper(params_subset, variables):
            # Expand subset to full vector
            params_full = create_full_vector(params_subset, fixed_params)
            # Call ORIGINAL surrogate (unchanged!)
            return original_surrogate(params_full, variables)

        # 3. TEMPORARILY replace surrogate
        self.run_surrogate = surrogate_wrapper

        # 4. RUN MCMC
        # MCMC sees wrapper, but wrapper calls original surrogate
        # Original surrogate models are NEVER touched

    finally:
        # 5. RESTORE everything
        self.run_surrogate = original_surrogate  # Restore
        self.ensemble_parms = original_parms      # Restore
        self.nparms_ensemble = original_nparms    # Restore
```

**Result**: Surrogates completely unaffected.

## Practical Examples

### Example 1: Stage-wise Calibration

```python
# Stage 1: Calibrate roots
mycase.run_MCMC_subset(
    calibrate_params=['froottcn', 'frootacn', 'froota_long'],
    fixed_params={...},
    myvars=['GPP'],
    nevals=3000
)

# Use Stage 1 results for Stage 2
# No retraining needed - same surrogates!
mycase.run_MCMC_subset(
    calibrate_params=['leafcn', 'slatop', 'flnr'],
    fixed_params={... including Stage 1 results ...},
    myvars=['GPP', 'ER'],
    nevals=5000
)
```

### Example 2: Sensitivity-Based Selection

```python
# Run GSA once
mycase.run_GSA(variables=['GPP', 'ER'])

# Calibrate only highly sensitive parameters
sensitive = get_sensitive_params(mycase, threshold=0.05)

mycase.run_MCMC_subset(
    calibrate_params=sensitive,
    fixed_params={... all others ...},
    myvars=['GPP', 'ER'],
    nevals=5000
)

# Later: Try different threshold
# Same surrogates!
more_sensitive = get_sensitive_params(mycase, threshold=0.10)

mycase.run_MCMC_subset(
    calibrate_params=more_sensitive,
    fixed_params={... updated ...},
    myvars=['GPP', 'ER'],
    nevals=5000
)
```

### Example 3: Compare Different Fixed Values

```python
# Scenario A: Fix TAM at literature values
mycase.run_MCMC_subset(
    calibrate_params=['leafcn', 'slatop'],
    fixed_params={'froottcn': 120, 'frootacn': 85, ...},
    myvars=['GPP'],
    nevals=5000
)

# Scenario B: Fix TAM at different values
# SAME surrogates, different constraints
mycase.run_MCMC_subset(
    calibrate_params=['leafcn', 'slatop'],
    fixed_params={'froottcn': 100, 'frootacn': 70, ...},
    myvars=['GPP'],
    nevals=5000
)

# Compare results from Scenarios A and B
```

## Benefits

### 1. Computational Efficiency
- **Train once**: Hours/days for 1000+ ensemble runs
- **Calibrate many times**: Minutes/hours per MCMC run
- **No wasted computation**: Reuse training investment

### 2. Experimental Flexibility
- Try different parameter combinations
- Test different calibration strategies
- Compare different scenarios
- Iterate quickly on calibration approach

### 3. Reproducibility
- Same surrogates → Consistent basis for comparison
- Can share trained surrogates with collaborators
- Document training once, reuse for all calibrations

### 4. Scientific Workflow
```
Week 1: Train surrogates (once)
  └─ 1000 ensemble runs × 14 years = heavy computation

Week 2-4: Calibration experiments (many times)
  ├─ Calibrate leaf parameters (5000 MCMC steps)
  ├─ Calibrate root parameters (5000 MCMC steps)
  ├─ Calibrate combined (5000 MCMC steps)
  ├─ Test sensitivity-based selection
  └─ Compare different scenarios

All using SAME trained surrogates!
```

## Common Questions

### Q: Do I need to retrain if I want to calibrate different parameters?

**A: No!** Surrogates are trained on all parameters. You can calibrate any subset without retraining.

### Q: What if I want to use different fixed values?

**A: No problem!** Fixed values are applied during MCMC, not during surrogate training. Change them as much as you want.

### Q: Can I run multiple MCMCs sequentially?

**A: Yes!** Each MCMC run is independent. Surrogates restored after each run.

### Q: What if I want to add more parameters later?

**A: Need to retrain.** If you add parameters not in original training, you must retrain surrogates with expanded ensemble. But for any subset of originally trained parameters, no retraining needed.

### Q: How do I verify surrogates are unchanged?

**A: Run the test:**
```bash
python test_surrogate_reuse.py
```

### Q: Can I share surrogates with collaborators?

**A: Yes!** Save your case with trained surrogates:
```python
with open('mycase_with_surrogates.pkl', 'wb') as f:
    pickle.dump(mycase, f)
```

Share this file. Collaborators can run MCMC with different subsets without retraining.

## Technical Details

### Surrogate Training (One Time)

```python
# This is expensive - do once
mycase.train_surrogate(
    variables=['GPP', 'ER', 'NEE', 'NPP'],
    hidden_layer_sizes=(100, 50),  # Neural network architecture
    activation='relu',
    alpha=0.001,
    max_iter=1000,
    cv_folds=5
)

# What gets stored in mycase:
# mycase.surrogate['GPP']  = GridSearchCV(MLPRegressor(...))
# mycase.surrogate['ER']   = GridSearchCV(MLPRegressor(...))
# mycase.pscaler['GPP']    = StandardScaler(...)
# mycase.yscaler['GPP']    = StandardScaler(...)
# etc.
```

### MCMC Subset Call (Many Times)

```python
# This is fast - use repeatedly
mycase.run_MCMC_subset(
    calibrate_params=['leafcn', 'slatop'],  # 2 params
    fixed_params={... 19 params ...},        # Rest fixed
    myvars=['GPP', 'ER'],
    nevals=5000                              # ~minutes
)

# Internal process:
# For each MCMC step (5000 times):
#   1. Propose: leafcn=26.3, slatop=0.0117
#   2. Expand:  [leafcn=26.3, slatop=0.0117, froottcn=120, ...]
#   3. Call:    surrogate(full_vector)  # Uses trained model
#   4. Get:     GPP=3.5e-08, ER=2.8e-08
#   5. Evaluate: posterior probability
#   6. Accept/reject
```

## Summary

✓ **Train surrogates once** with full parameter set
✓ **Use repeatedly** for different parameter subsets
✓ **No retraining** needed for subset changes
✓ **No modification** of trained models
✓ **Full flexibility** in choosing calibration strategy

**Bottom line**: Your trained surrogates are a reusable asset. Train them once, use them many times for different calibration experiments.

---

**Files**:
- `model_ELM/MCMC_subset.py` - Implementation
- `test_surrogate_reuse.py` - Verification test
- `MCMC_SUBSET_GUIDE.md` - Full usage guide
- `SURROGATE_REUSE_README.md` - This file
