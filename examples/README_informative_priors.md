# Informative Priors in OLMT MCMC

## Overview

The OLMT MCMC implementation now supports informative priors for parameter estimation, allowing you to specify different probability distributions for each parameter based on prior knowledge or expert judgment.

## Backward Compatibility

The enhancement is fully backward compatible. Existing parameter list files using the legacy 4-column format will continue to work with uniform priors:

```
parameter_name  pft  min  max
```

## Extended Format

The new extended format allows specifying distribution types:

```
parameter_name  pft  dist_type  param1  param2  [param3  param4]
```

## Supported Distributions

### 1. Uniform Distribution
**Format:** `parameter pft uniform min max`

Flat prior within bounds. All values equally likely.

**Example:**
```
bfact  23  uniform  0.05  0.15
```

**When to use:** When you have no prior knowledge beyond feasible parameter ranges.

---

### 2. Normal (Gaussian) Distribution
**Format:** `parameter pft normal mean std [min max]`

Normal distribution, optionally truncated to bounds.

**Example:**
```
# Truncated normal: mean=40, std=10, bounded to [15, 65]
graincn  23  normal  40.0  10.0  15.0  65.0

# Normal without explicit bounds (uses mean ± 4*std)
grnfill  23  normal  0.675  0.04
```

**When to use:**
- Parameters with well-characterized central tendencies
- Physical parameters measured with normally-distributed errors
- When prior knowledge suggests symmetric distributions around a mean

---

### 3. Truncated Normal Distribution
**Format:** `parameter pft truncnorm mean std min max`

Alias for normal with mandatory bounds (same as `normal` with 6 columns).

**Example:**
```
declfact  23  truncnorm  1.1375  0.2  0.7  1.575
```

**When to use:** Same as normal, but when you want to make bounds explicit.

---

### 4. Lognormal Distribution
**Format:** `parameter pft lognormal log_mean log_std [min max]`

For strictly positive parameters with right-skewed distributions.

**Example:**
```
# log_mean=1.386 (exp(1.386)≈4), log_std=0.5
allconsl  23  lognormal  1.386  0.5  1.0  5.0
```

**When to use:**
- Strictly positive parameters (rates, concentrations)
- Parameters that span orders of magnitude
- When multiplicative rather than additive effects are expected
- Right-skewed distributions (long tail toward higher values)

**Note:** `log_mean` and `log_std` are parameters of the underlying normal distribution. The median of the lognormal is `exp(log_mean)`.

---

### 5. Beta Distribution
**Format:** `parameter pft beta alpha beta min max`

Flexible distribution on bounded intervals, naturally defined on [0,1] then scaled to [min, max].

**Example:**
```
# Alpha=2, Beta=5: left-skewed, scaled to [0.02, 0.07]
slatop  23  beta  2.0  5.0  0.02  0.07

# Alpha=3, Beta=3: symmetric, bell-shaped
arootf  23  beta  3.0  3.0  0.0  0.25
```

**When to use:**
- Parameters bounded to specific intervals (e.g., fractions, proportions)
- When shape of prior needs fine control:
  - alpha = beta: symmetric
  - alpha < beta: left-skewed (favors lower values)
  - alpha > beta: right-skewed (favors higher values)
  - alpha, beta < 1: U-shaped (favors extremes)
  - alpha, beta > 1: bell-shaped

---

### 6. Gamma Distribution
**Format:** `parameter pft gamma shape scale [min max]`

For positive parameters with flexible shapes (right-skewed to bell-shaped).

**Example:**
```
# Shape=5, Scale=20, bounded to [20, 140]
fstemcn  23  gamma  5.0  20.0  20.0  140.0
```

**When to use:**
- Strictly positive parameters
- When lognormal is too skewed
- Waiting times, sizes, concentrations
- Shape parameter controls distribution form:
  - shape < 1: decreasing, long right tail
  - shape = 1: exponential distribution
  - shape > 1: bell-shaped, peak moves right as shape increases

**Note:** Mean = shape × scale, Variance = shape × scale²

---

## Implementation Details

### Prior Calculation

Priors are calculated in log-space for numerical stability. The `calc_log_prior()` function computes:

- Hard boundary checks (returns -∞ for out-of-bounds)
- Log probability density for the specified distribution
- Proper normalization for truncated distributions

### Integration with MCMC

The informative priors are integrated into:

1. **Custom MCMC sampler** (`MCMC_custom`): Uses `calc_posterior()` which now includes informative priors
2. **PyMC3 sampler** (`MCMC_pymc3`): Defines proper PyMC3 distributions for each parameter type

### Backward Compatibility

When parameter files don't specify distribution types, the code automatically:
- Defaults to uniform priors
- Uses the legacy 4-column parsing
- Maintains identical behavior to previous versions

## Example Usage

### Creating a Parameter File

See `examples/parm_list_informative_priors_example` for a complete example with all distribution types.

### Running MCMC with Informative Priors

No code changes needed! Simply provide the extended parameter file:

```python
# In your run script (e.g., runscripts/run_TAM.py)
mycase.setup_ensemble(
    parm_list='inputdata/PTTAM/US-Var_parm_list_tam',
    sampletype='sobol',
    nsamples=500
)

# MCMC will automatically use informative priors if specified
mycase.MCMC_custom(
    parms=initial_parms,
    myvars=['GPP', 'NEE'],
    nevals=10000,
    nburn=2000
)
```

## Choosing Priors: Best Practices

### 1. Use Domain Knowledge
- Normal: Physical measurements with known uncertainty
- Lognormal: Concentrations, rates (never negative, potentially spanning orders of magnitude)
- Beta: Fractions, proportions (naturally bounded to [0,1] or similar)
- Gamma: Positive quantities with known mean and variance

### 2. Match Distribution Shape to Knowledge
- Uniform: Minimal knowledge, only feasible bounds
- Normal: Symmetric uncertainty around best estimate
- Lognormal/Gamma: Positive parameters with asymmetric uncertainty
- Beta: Bounded parameters needing shape control

### 3. Avoid Over-constraining
- Start with weakly informative priors (large variance)
- Let data dominate when available
- Use tight priors only when strong prior knowledge exists

### 4. Check Prior Sensitivity
- Run MCMC with different prior specifications
- Compare posterior distributions
- Strong data should overwhelm weak priors

## Mathematical Details

### Posterior Calculation

The posterior is calculated as:

```
log P(θ|D) = log P(D|θ) + log P(θ) - log P(D)
           = log-likelihood + log-prior + constant
```

The MCMC algorithm samples from this posterior distribution.

### Truncation

For distributions with bounds (truncnorm, bounded lognormal, etc.), the probability density is renormalized to integrate to 1 over the bounded region. This is handled automatically by scipy.stats functions.

## Troubleshooting

### Prior/Likelihood Conflict
If MCMC struggles to find valid parameter space:
- Check that prior bounds include regions where likelihood is non-zero
- Widen prior distributions if too restrictive
- Verify observation uncertainties are reasonable

### Convergence Issues
With informative priors:
- Check R-hat diagnostics (should be < 1.1)
- Increase burn-in period if needed
- Try different initial parameter values
- Consider if prior is too different from likelihood

### PyMC3 Errors
If PyMC3 sampling fails:
- Fall back to custom sampler: `sampler='custom'`
- Check parameter bounds are compatible with distribution
- Verify testval (initial value) is within valid range

## References

- Gelman et al., *Bayesian Data Analysis*, 3rd ed.
- PyMC3 documentation: https://docs.pymc.io/
- SciPy stats distributions: https://docs.scipy.org/doc/scipy/reference/stats.html
