# Guide: Using Lognormal Priors for TAM Parameters

## Quick Start for IT-Ren Site

### Option 1: Use the Pre-converted File

I've created a lognormal version of the IT-Ren parameter file:

```bash
# Original file
inputdata/PTTAM/IT-Ren_parm_list_tam

# Lognormal version
inputdata/PTTAM/IT-Ren_parm_list_tam_lognormal
```

Simply use the lognormal version in your run script:

```python
mycase.setup_ensemble(
    parm_list='inputdata/PTTAM/IT-Ren_parm_list_tam_lognormal',
    sampletype='monte_carlo',
    nsamples=500
)
```

### Option 2: Use the Conversion Script

Convert any parameter file to lognormal:

```bash
python examples/convert_to_lognormal.py \
    inputdata/PTTAM/IT-Ren_parm_list_tam \
    inputdata/PTTAM/IT-Ren_parm_list_tam_lognormal \
    0.5  # default log_std
```

## Understanding Lognormal Parameters

### Format
```
parameter_name  pft  lognormal  log_mean  log_std  min  max
```

### Parameters Explained

**log_mean**: Natural logarithm of the distribution's median
- Calculated as: `log_mean = (ln(min) + ln(max)) / 2`
- This centers the distribution at the geometric mean of your bounds
- Example: For bounds [20, 184], geometric mean = √(20×184) = 60.6
  - `log_mean = ln(60.6) = 4.104`

**log_std**: Controls the spread/uncertainty
- `0.3` = **Narrow** spread
  - 68% of values within [0.74×median, 1.35×median]
  - Use when: Strong prior knowledge, narrow uncertainty

- `0.5` = **Moderate** spread (DEFAULT)
  - 68% of values within [0.61×median, 1.65×median]
  - Use when: Reasonable prior knowledge, moderate uncertainty

- `0.7` = **Wide** spread
  - 68% of values within [0.50×median, 2.01×median]
  - Use when: Weak prior knowledge, large uncertainty, wide parameter range

### Example Conversions

#### Example 1: Moderate Range Parameter
```
# Original (uniform)
froottcn 1 20 184

# Converted (lognormal)
# Geometric mean = √(20 × 184) = 60.6
# log_mean = ln(60.6) = 4.104
# Range ratio = 184/20 = 9.2 → moderate spread
froottcn 1 lognormal 4.104 0.5 20 184
```

#### Example 2: Wide Range Parameter
```
# Original (uniform)
frootmcp 1 10 1600

# Converted (lognormal)
# Geometric mean = √(10 × 1600) = 126.5
# log_mean = ln(126.5) = 4.840
# Range ratio = 1600/10 = 160 → wide spread
frootmcp 1 lognormal 4.840 0.7 10 1600
```

#### Example 3: Small Values (<1)
```
# Original (uniform)
frootm_long 1 0.13 1

# Converted (lognormal)
# Geometric mean = √(0.13 × 1) = 0.36
# log_mean = ln(0.36) = -1.022  ← negative is OK!
frootm_long 1 lognormal -1.022 0.6 0.13 1
```

**Note**: Negative `log_mean` is normal for parameters with values < 1.

## When to Use Lognormal vs Uniform

### Use Lognormal When:
✅ Parameters are **strictly positive** (no zeros)
✅ Parameters span **multiple orders of magnitude**
✅ You want to give **more weight to central values**
✅ Multiplicative effects are expected
✅ Physical processes suggest right-skewed uncertainty

### Use Uniform When:
✅ No prior preference for any value in the range
✅ Parameter range is **narrow** (ratio < 3)
✅ Parameters can be **zero or negative**
✅ You want **maximum exploration** in ensemble

### Mixed Approach (Recommended):
```
# Wide-range positive parameters: lognormal
froottcn 1 lognormal 4.104 0.5 20 184
frootmcp 1 lognormal 4.840 0.7 10 1600

# Narrow-range fraction parameters: uniform
frt_fcel 1 uniform 0.35 0.65
fra_fcel 1 uniform 0.35 0.65
```

## Impact on MCMC

### With Uniform Priors:
- MCMC samples uniformly across entire parameter space
- Equal probability at min and max values
- May waste samples in unrealistic regions

### With Lognormal Priors:
- MCMC focuses sampling near geometric mean
- Less sampling at extreme values (min/max)
- Faster convergence when prior matches true distribution
- More efficient parameter space exploration

### Example Comparison:

For `froottcn` with bounds [20, 184]:

**Uniform prior**:
- Probability(value=20) = Probability(value=184) = constant
- Equal sampling everywhere

**Lognormal prior** (log_mean=4.104, log_std=0.5):
- Probability peaks around median ≈ 60.6
- ~68% of samples in [37, 100]
- ~95% of samples in [23, 161]
- Very few samples at extremes (20 or 184)

## Choosing log_std Values

### Rule of Thumb:
1. Calculate range ratio: `R = max / min`
2. Choose log_std based on ratio and confidence:

| Range Ratio (R) | Confidence Level | Recommended log_std |
|-----------------|------------------|---------------------|
| R < 3           | High             | 0.2 - 0.3          |
| 3 ≤ R < 10      | Moderate         | 0.4 - 0.5          |
| 10 ≤ R < 50     | Low              | 0.5 - 0.6          |
| R ≥ 50          | Very Low         | 0.6 - 0.8          |

### Sensitivity to log_std:

```python
# Conservative (narrow prior, strong belief)
froottcn 1 lognormal 4.104 0.3 20 184
# → 95% of samples in [40, 92]

# Moderate (default)
froottcn 1 lognormal 4.104 0.5 20 184
# → 95% of samples in [23, 161]

# Permissive (wide prior, weak belief)
froottcn 1 lognormal 4.104 0.7 20 184
# → 95% of samples in [13, 280] (some clipped to [20,184])
```

## Advanced: Manual Specification

If you have specific domain knowledge, you can set log_mean and log_std directly:

```python
import numpy as np

# I believe median should be 50, with 95% confidence interval [20, 150]
median = 50
ci_lower = 20
ci_upper = 150

# For lognormal, 95% CI ≈ [median × exp(-1.96×σ), median × exp(+1.96×σ)]
# Solve for σ:
log_std = np.log(ci_upper / median) / 1.96  # ≈ 0.56

# Calculate log_mean
log_mean = np.log(median)  # = 3.912

# Result:
# froottcn 1 lognormal 3.912 0.56 20 184
```

## Verification

After creating your lognormal parameter file, verify it works:

```python
# Test reading the parameter file
import sys
sys.path.append('/path/to/OLMT')
from model_ELM.main import ELMcase

mycase = ELMcase(...)
mycase.setup_ensemble(parm_list='inputdata/PTTAM/IT-Ren_parm_list_tam_lognormal', nsamples=10)

# Check that dist_type is populated
print("Distribution types:", mycase.ensemble_dist_type)
print("Distribution params:", mycase.ensemble_dist_params[0])  # First parameter
```

Expected output:
```
Distribution types: ['lognormal', 'lognormal', ...]
Distribution params: {'log_mean': 4.104, 'log_std': 0.5, 'min': 20.0, 'max': 184.0}
```

## Troubleshooting

### Error: "Lognormal requires positive values"
**Cause**: Parameter bounds include zero or negative values
**Solution**: Use `uniform` or `normal` distribution for that parameter

```
# Don't use lognormal for parameters that can be zero
frt_flab 1 lognormal -1.677 0.5 0.0 0.35  # ❌ min=0 causes problems

# Use uniform instead
frt_flab 1 uniform 0.0 0.35  # ✅ OK
```

### MCMC Not Using Priors
**Cause**: Parameter file format error
**Solution**: Check file has 7 columns for lognormal:
```
parm_name  pft  lognormal  log_mean  log_std  min  max
    ↓       ↓       ↓          ↓        ↓      ↓    ↓
    1       2       3          4        5      6    7
```

### Prior-Likelihood Mismatch
**Symptom**: MCMC acceptance rate very low (<1%)
**Cause**: Prior distribution incompatible with data
**Solution**:
1. Increase log_std to widen prior
2. Check if lognormal is appropriate (try `normal` or `uniform`)
3. Verify observations and uncertainties are reasonable

## Converting All TAM Sites

Batch convert all sites:

```bash
# Convert all TAM parameter files
for site in US-Var IT-Ren FI-Hyy BR-Sa1 CA-Oas AU-Tum; do
    echo "Converting $site..."
    python examples/convert_to_lognormal.py \
        inputdata/PTTAM/${site}_parm_list_tam \
        inputdata/PTTAM/${site}_parm_list_tam_lognormal \
        0.5
done
```

## Summary

1. **Quick conversion**: Use `convert_to_lognormal.py` script
2. **Default strategy**: Geometric mean for log_mean, auto log_std based on range
3. **Customization**: Adjust log_std based on prior confidence
4. **Mixed priors**: Combine lognormal, uniform, and normal in same file
5. **No impact on ensemble**: Ensemble sampling remains uniform
6. **MCMC benefits**: Faster convergence with informative priors

For IT-Ren specifically, the converted file is ready to use at:
`inputdata/PTTAM/IT-Ren_parm_list_tam_lognormal`
