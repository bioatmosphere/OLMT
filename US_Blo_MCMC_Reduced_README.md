# US-Blo MCMC with Reduced Parameter Space

## Quick Start

Use this parameter file for MCMC calibration:
```
inputdata/PTTAM/US-Blo_parm_list_tam_mcmc_reduced
```

**No code changes needed!** This file works with your existing surrogate model and MCMC setup.

## What Changed

### Previous approach (20 parameters):
- All 20 parameters varied during MCMC
- Result: Complete convergence failure
  - ESS = 6-7 (need >100)
  - Acceptance rate = 6.3% (target 20-40%)
  - All parameters perfectly correlated

### New approach (3 + 17 parameters):

**3 parameters that vary (HIGH sensitivity):**
- `slatop`: 0.018 - 0.028 (full range)
- `leafcn`: 25 - 40 (full range)
- `flnr`: 0.10 - 0.18 (full range)

**17 parameters effectively fixed (LOW sensitivity <0.1%):**
- All root parameters set to very tight ranges (±0.01%) around best values
- Example: `froottcn`: 96.68 - 96.70 (was 60 - 150)
- The MCMC will still "sample" these but they won't actually change

## Why This Works

### Scientific rationale:
1. **Sensitivity analysis showed:** 99.9% of flux variance comes from slatop, leafcn, flnr
2. **Data constraints:** Only 3 observations (GPP, ER, NEE) → can only constrain ~3 parameters
3. **Identifiability:** Cannot distinguish between froottcn vs froottcp from flux data alone

### Technical implementation:
- Uses existing 20-parameter surrogate model (no retraining needed)
- MCMC still samples 20-dimensional space, but 17 dimensions are essentially flat
- Proposal covariance will adapt to the geometry (3 dimensions with variance, 17 without)

## Expected Results

After running MCMC with this file, you should see:

### Convergence diagnostics:
- **Acceptance rate:** 20-40% ✓ (was 6.3%)
- **ESS:** >1000 ✓ (was 6-7)
- **Autocorrelation time:** <100 iterations ✓ (was 63,000)
- **Parameter correlations:** <0.7 ✓ (was >0.99)

### Parameter estimates:
- **Photosynthesis parameters:** Well-constrained posteriors with reasonable uncertainty
- **Root parameters:** Will stay near their fixed values (as expected - data can't constrain them)

## How to Use

### Option A: Modify existing run script

If you have a run script like `run_TAM1.py`:

```python
# Change this line:
# parm_list = 'inputdata/PTTAM/US-Blo_parm_list_tam_v3'
# To:
parm_list = 'inputdata/PTTAM/US-Blo_parm_list_tam_mcmc_reduced'

# Keep everything else the same:
case.read_parm_list(parm_list)
case.MCMC(parms, myvars, nevals, myobs_05, myobs_95, ...)
```

### Option B: Just point to existing pickle file

Since your surrogate is already trained and saved in the pickle file, you might just need to:

```python
import pickle
with open('pklfiles/20251030_US-Blo_ICB20TRCNPRDCTCBC.pkl', 'rb') as f:
    case = pickle.load(f)

# Update parameter list to reduced version
case.read_parm_list('inputdata/PTTAM/US-Blo_parm_list_tam_mcmc_reduced')

# Run MCMC as before
# ... (same MCMC setup as previous run)
```

## Interpreting Results

### Parameter posteriors:

**For the 3 varying parameters:**
- Look at posterior distributions, 95% credible intervals
- Check for reasonable uncertainty (not too wide, not too narrow)
- Compare to prior ranges to see how much data informed the estimate

**For the 17 fixed parameters:**
- Posteriors will be very narrow (by design)
- This is correct - the data cannot constrain these parameters
- Their values are based on:
  - Best fit from initial wide-range MCMC
  - Consistency with Wang et al. (2023) literature values
  - Physiological constraints (prevented GPP=0 failures)

### Model fit:
- Check GPP, ER, NEE predictions against observations
- Should be similar to your previous best fit
- Uncertainty bounds will be narrower (more realistic given 3 degrees of freedom)

## Advantages of This Approach

1. **Fast:** Runs tonight with existing code
2. **Scientifically sound:** Only calibrate identifiable parameters
3. **No retraining:** Uses existing surrogate model
4. **Proper uncertainty:** Credible intervals reflect true parameter uncertainty
5. **Debuggable:** If something goes wrong, easy to adjust ranges

## Next Steps After This Runs

1. **Check diagnostics:** Verify ESS >1000, acceptance rate 20-40%
2. **Examine posteriors:** Look at parameter distributions for slatop, leafcn, flnr
3. **Validate fit:** Compare model predictions to observations
4. **Sensitivity check:** Try slightly different fixed values to ensure results are robust

If this works well, you can:
- Use these calibrated parameters for predictions
- Document the approach for publication (reduced parameter space based on sensitivity)
- Apply same methodology to other sites (AU-Tum, etc.)

## Troubleshooting

### If acceptance rate is still too low (<15%):
- Might need longer burn-in to adapt proposal covariance
- Try running with more burn-in iterations

### If you still see perfect correlations:
- Check that MCMC actually read the new parameter file
- Verify the surrogate model is being used (not re-running full ELM)

### If some root parameters still vary significantly:
- This would be surprising - indicates they might actually be sensitive
- Check which ones and consider allowing them to vary with full ranges

## References

Wang, B., McCormack, M.L., Ricciuto, D.M., Yang, X., and Iversen, C.M. (2023).
"Embracing fine-root system complexity in terrestrial ecosystem modelling."
Global Change Biology, 29(10), 2871-2888. https://doi.org/10.1111/gcb.16659
