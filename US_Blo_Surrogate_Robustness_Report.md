# US-Blo ELM-TAM Surrogate Model Robustness Report

**Site:** US-Blo
**Model Configuration:** ELM-TAM PFT 1
**Compset:** ICB20TRCNPRDCTCBC
**Analysis Date:** October 30, 2025
**Case File:** `20251029_US-Blo_ICB20TRCNPRDCTCBC.pkl`

---

## Executive Summary

This report evaluates the robustness of neural network surrogate models trained for US-Blo site calibration. The surrogate models are intended to replace computationally expensive ELM-TAM simulations during MCMC parameter estimation and global sensitivity analysis.

### Key Findings

**CRITICAL ISSUES IDENTIFIED:**

1. **Poor surrogate quality for calibration variables** (GPP, ER, NEE)
   - GPP: R² = 0.699 (❌ Below acceptable threshold of 0.80)
   - ER: R² = 0.669 (❌ Below acceptable threshold)
   - NEE: R² = 0.690 (❌ Below acceptable threshold)
   - NPP: R² = 0.931 (✓ Good quality)

2. **Insufficient dynamic range**
   - Surrogate GPP range: 106 gC/m²/year (only 1142-1249 gC/m²/year)
   - This severely limits the ability to fit observed data variations

3. **Parameter identifiability problems**
   - Root C and N parameters (froottcp/frootacp/frootmcp, froottcn/frootacn/frootmcn) show near-perfect correlation
   - These parameters produce nearly identical model responses, making them non-identifiable in calibration

4. **Ensemble outputs unrealistically low**
   - Model GPP: 858 ± 476 gC/m²/year (mean ± std)
   - Model ER: 794 ± 433 gC/m²/year
   - These values are significantly lower than typical temperate forest productivity

**RECOMMENDATION:** The current surrogates are NOT suitable for reliable MCMC calibration. See Section 6 for detailed recommendations.

---

## 1. Surrogate Model Quality Assessment

### 1.1 Cross-Validation R² Scores

| Variable | R² Score | Status | Quality Assessment |
|----------|----------|--------|-------------------|
| **GPP**  | 0.699    | ❌     | POOR - Unreliable for calibration |
| **ER**   | 0.669    | ❌     | POOR - Unreliable for calibration |
| **NEE**  | 0.690    | ❌     | POOR - Unreliable for calibration |
| **NPP**  | 0.931    | ✓      | GOOD - Acceptable for calibration |

**Quality Thresholds:**
- Excellent: R² > 0.95
- Good: R² > 0.90
- Moderate: R² > 0.80
- Poor: R² < 0.80

### 1.2 Hyperparameters

All surrogates used similar optimal configurations:
- **Architecture:** 2-layer MLP with (100, 50) neurons
- **Activation:** tanh or relu
- **Solver:** lbfgs
- **Regularization:** α = 0.01-0.05

### 1.3 Training Convergence

- GPP: 452 iterations, loss = 0.00128
- ER: 352 iterations, loss = 0.00579
- NEE: 1647 iterations, loss = 0.00566
- NPP: 848 iterations, loss = 0.00243

**Analysis:** The relatively high loss values for ER and NEE, combined with low R² scores, suggest the neural networks are struggling to capture the complex parameter-output relationships. This could indicate:
- Insufficient ensemble size (likely < 1000 members)
- High noise or non-smooth response surfaces
- Weak parameter influences on outputs

---

## 2. Ensemble Output Analysis

### 2.1 Ensemble Output Ranges

| Variable | Min | Max | Mean ± Std | Range |
|----------|-----|-----|------------|-------|
| **GPP**  | 0.0 | 1504 | 858 ± 476 | 1504 gC/m²/year |
| **ER**   | 0.0 | 1272 | 794 ± 433 | 1272 gC/m²/year |
| **NEE**  | -169 | 67 | -34 ± 50 | 236 gC/m²/year |
| **NPP**  | 0.0 | 846 | 353 ± 222 | 846 gC/m²/year |

**Data Shape:** 7 years × 1000 ensemble members

### 2.2 Critical Issues

**Issue 1: Zero-valued outputs detected**
- Both GPP and ER show minimum values of 0.0 gC/m²/year
- This likely indicates model failures or unphysical parameter combinations
- These failures contaminate the training data and degrade surrogate quality

**Issue 2: Low productivity**
- Mean GPP of 858 gC/m²/year is low for US-Blo (temperate deciduous forest)
- Typical temperate forest GPP: 1500-3000 gC/m²/year
- This suggests the parameter ranges or model configuration may not reach observed values

**Issue 3: Large variance**
- Standard deviation is ~55% of mean for both GPP and ER
- Such high variability makes it difficult to train accurate surrogates
- May indicate extreme parameter values producing unrealistic responses

---

## 3. Surrogate Dynamic Range Analysis

### 3.1 Surrogate Response at Parameter Extremes

| Parameter Set | GPP | ER | NEE | NPP |
|---------------|-----|-----|-----|-----|
| **Minimum** | 1143 | 1084 | -12 | 350 |
| **Midpoint** | 1145 | 1060 | -54 | 481 |
| **Maximum** | 1249 | 1103 | -56 | 531 |
| **Range** | **106** | **43** | **45** | **181** |

### 3.2 Critical Finding: Insufficient Dynamic Range

**GPP Dynamic Range:**
- Surrogate range: 106 gC/m²/year (1143-1249)
- This narrow range is a major limitation for calibration

**What this means:**
- The surrogate can only predict GPP values within ~9% variation
- During MCMC, even with optimal parameters, the surrogate cannot span the full range of plausible values
- This will severely limit calibration effectiveness

**Likely Cause:**
- Current parameters (all root-related) have weak influence on GPP
- Need to include photosynthesis and canopy parameters that strongly control GPP

---

## 4. Parameter Sensitivity Analysis

### 4.1 Most Influential Parameters

**For GPP (most important calibration target):**
1. frootmcn (mature root C:N): 2.44% change per 10% parameter change
2. frootm_long (mature root longevity): 2.19%
3. frootacn (active root C:N): 1.78%
4. froott_long (total root longevity): 1.56%
5. froottcn (total root C:N): 1.34%

**For ER (ecosystem respiration):**
1. frootmcn: 1.85%
2. froott_long: 1.49%
3. frootm_long: 1.38%
4. froottcp: 1.36%
5. froota_long: 1.25%

**For NEE (net ecosystem exchange):**
1. froottcn: 10.06% (most sensitive!)
2. frootmcn: 9.31%
3. froota_long: 7.95%
4. frm_fcel (mature root cellulose fraction): 7.43%
5. frootacn: 7.34%

### 4.2 Key Observations

1. **All parameters are root-related** - No photosynthesis or canopy parameters included
2. **Weak GPP sensitivity** - Maximum 2.44% change is very small
3. **NEE is most sensitive** - Up to 10% changes, but NEE is a small difference between large fluxes
4. **Root C:N parameters dominate** - These control root turnover and decomposition

### 4.3 Missing Critical Parameters

For effective GPP calibration, consider adding:
- **Vcmax25** (maximum carboxylation rate at 25°C) - typically ±30-50% GPP change
- **Jmax25** (maximum electron transport rate) - typically ±20-30% GPP change
- **SLA** (specific leaf area) - controls leaf area and light interception
- **Flnr** (fraction of leaf N in RuBisCO) - photosynthesis efficiency
- **Leaf_long** (leaf longevity) - controls LAI dynamics

---

## 5. Parameter Identifiability Issues

### 5.1 Correlated Parameter Groups

**Group 1: Root C Pools (froottcp, frootacp, frootmcp)**
- When each increased by 20%, GPP predictions: [1155, 1146, 1145]
- Nearly identical responses (< 1% difference)
- **Diagnosis:** Parameters are perfectly correlated in model response
- **Impact:** MCMC cannot distinguish between these parameters

**Group 2: Root N Pools (froottcn, frootacn, frootmcn)**
- When each increased by 20%, GPP predictions: [1154, 1159, 1160]
- Nearly identical responses (< 0.5% difference)
- **Diagnosis:** Parameters are perfectly correlated
- **Impact:** MCMC will show artificially perfect correlation (r = 1.0)

### 5.2 Why Perfect Correlation Occurs

In ELM-TAM, these root pools may be modeled such that:
- Total root carbon = froottc + frootac + frootmc (additive)
- If the model response primarily depends on total root C, not individual pools
- Then varying any individual pool has similar effects

### 5.3 Recommendations

**Option 1: Reduce parameter set**
- Calibrate only total root C (froottc) and total root N (froottcn)
- Remove frootac/frootmc and frootacn/frootmcn

**Option 2: Use hierarchical constraints**
- Define frootacp as a fraction of froottcp
- Define frootmcp as a fraction of froottcp
- Reduces parameters from 3 to 1 + 2 fractions

**Option 3: Use derived parameters**
- Calibrate total root C:N ratio instead of individual pools
- This is more physically meaningful

---

## 6. Recommendations for Improving Surrogate Robustness

### 6.1 IMMEDIATE ACTIONS (Before MCMC)

#### Action 1: Increase Ensemble Size ⚠️ CRITICAL
**Current:** Likely 1000 members (7 years × 1000 in output matrix)
**Recommended:** 2000-5000 members for 17 parameters

**Justification:**
- Rule of thumb: 100-300 samples per parameter
- With 17 parameters: minimum 1700 samples
- Higher-order interactions need more samples
- Current low R² suggests insufficient sampling

**Implementation:**
```python
# In your runscript
mycase.nsamples_ensemble = 3000
```

#### Action 2: Add Photosynthesis Parameters ⚠️ CRITICAL
**Current parameters:** All root-related (weak GPP control)
**Add these critical parameters:**

| Parameter | Range | Description | Expected GPP Impact |
|-----------|-------|-------------|-------------------|
| vcmax25 | 20-100 μmol/m²/s | Max carboxylation rate | ±30-50% |
| jmax25 | 40-200 μmol/m²/s | Max electron transport | ±20-30% |
| slatop | 0.005-0.03 m²/gC | Specific leaf area | ±20-40% |
| flnr | 0.05-0.20 | Fraction leaf N in RuBisCO | ±15-25% |
| leaf_long | 0.5-3.0 years | Leaf longevity | ±10-20% |

**Expected outcome:**
- Increase surrogate GPP dynamic range from 106 to 500-1000 gC/m²/year
- Dramatically improve calibration effectiveness

#### Action 3: Reduce Parameter Redundancy
**Remove or constrain:**
- Keep froottcp, remove frootacp and frootmcp (or use fractions)
- Keep froottcn, remove frootacn and frootmcn (or use fractions)

**New parameter count:** 17 → 12 (or 17 → 14 with fractions)

**Benefits:**
- Eliminates perfect correlations
- Improves surrogate training
- Faster MCMC convergence

#### Action 4: Filter Invalid Ensemble Members
**Before surrogate training:**
```python
# Remove ensemble members with:
# - GPP < 500 gC/m²/year (unrealistically low)
# - GPP > 5000 gC/m²/year (unrealistically high)
# - ER < 0 or ER > 5000 (unphysical)
```

**Justification:**
- Currently seeing GPP = 0.0, which contaminates training
- Filtering improves surrogate R² by removing outliers

### 6.2 MEDIUM-TERM IMPROVEMENTS

#### Improvement 1: Use Better Sampling Strategy
**Current:** Likely Latin Hypercube or random sampling
**Recommended:** Sobol sequences with scrambling

```python
mycase.ensemble_sampling = 'sobol'
```

**Benefits:**
- Better space-filling properties
- Improved parameter space coverage
- Higher surrogate accuracy with same sample size

#### Improvement 2: Ensemble Surrogate Models
**Current:** Single surrogate model
**Recommended:** Ensemble of 5-10 surrogate models with different initializations

**Implementation:**
- Train multiple surrogates with different random seeds
- Use mean prediction + uncertainty from ensemble variance
- Provides uncertainty quantification for MCMC

#### Improvement 3: Alternative Surrogate Architectures
**Test these alternatives:**
1. **Gradient Boosting** (XGBoost, LightGBM) - Often outperforms neural networks
2. **Random Forest** - More robust to outliers, no hyperparameter tuning needed
3. **Gaussian Process** - Provides uncertainty, but scales poorly (max ~2000 samples)

```python
# In surrogate_NN.py, add:
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Compare with current MLP
```

### 6.3 DIAGNOSTIC WORKFLOWS

#### Before Retraining:
1. **Run:** `diagnose_ensemble.py` (update for US-Blo)
   - Analyze parameter-output correlations
   - Identify most influential parameters
   - Check ensemble coverage of parameter space

2. **Visualize:** Parameter-output scatter plots
   ```python
   import matplotlib.pyplot as plt
   for param in ensemble_params:
       plt.scatter(samples[param], outputs['GPP'])
       plt.xlabel(param)
       plt.ylabel('GPP')
       plt.savefig(f'param_correlation_{param}.png')
   ```

#### After Retraining:
1. **Run:** `diagnose_USBlo_surrogate.py` (created in this analysis)
2. **Check:** R² > 0.90 for GPP, ER, NEE
3. **Verify:** Dynamic range covers expected observations
4. **Test:** Out-of-sample validation with 100 new ensemble members

---

## 7. Validation Checklist for Retraining

Before proceeding to MCMC, verify:

- [ ] Ensemble size ≥ 2000 members (preferably 3000-5000)
- [ ] GPP surrogate R² > 0.90
- [ ] ER surrogate R² > 0.90
- [ ] NEE surrogate R² > 0.85 (NEE is inherently noisy)
- [ ] Surrogate GPP range > 500 gC/m²/year
- [ ] Surrogate ER range > 400 gC/m²/year
- [ ] No parameter pairs with |correlation| > 0.95
- [ ] No ensemble members with GPP = 0 or other unphysical values
- [ ] Visual inspection of surrogate scatter plots shows good fit
- [ ] Out-of-sample validation R² > 0.85

---

## 8. Expected Calibration Performance

### 8.1 With Current Surrogates (NOT RECOMMENDED)

**Issues:**
- Low R² (0.67-0.70) means 30-33% prediction error
- MCMC will converge to incorrect parameter values
- Calibrated model will not match observations
- Uncertainty estimates will be wrong

**Quantitative Impact:**
- If true GPP = 2000 gC/m²/year
- Surrogate prediction: 2000 ± 300 gC/m²/year (15% error)
- This error is larger than parameter-driven variability
- MCMC will essentially be fitting noise

### 8.2 With Improved Surrogates (RECOMMENDED)

**After implementing recommendations:**
- R² > 0.95 → prediction error < 5%
- Wide dynamic range enables fitting observations
- MCMC can accurately identify parameter posterior distributions
- Uncertainty quantification will be reliable

**Expected MCMC Performance:**
- Convergence in 10,000-50,000 iterations
- Posterior distributions will be well-constrained
- Calibrated parameters will produce realistic GPP/ER

---

## 9. Additional Diagnostic Outputs

### 9.1 Files Created

1. **Diagnostic script:** `diagnose_USBlo_surrogate.py`
   - Comprehensive surrogate robustness testing
   - Run anytime to check surrogate quality

2. **Surrogate scatter plots:** `UQ_output/20251029_US-Blo_ICB20TRCNPRDCTCBC/surrogate/`
   - Visual assessment of surrogate fit quality
   - One plot per variable per year

3. **This report:** `US_Blo_Surrogate_Robustness_Report.md`

### 9.2 How to Use Diagnostic Scripts

```bash
# Activate environment
source .venv/bin/activate

# Run comprehensive diagnostics
python diagnose_USBlo_surrogate.py

# Run ensemble output analysis
python diagnose_ensemble.py  # (after updating for US-Blo)

# Check scalers and training data
python check_surrogate_scalers.py  # (after updating for US-Blo)

# Test extreme parameter values
python test_surrogate_sensitivity.py  # (after updating for US-Blo)
```

---

## 10. Conclusion

The current US-Blo surrogate models are **NOT robust enough for MCMC calibration** due to:

1. **Poor prediction accuracy** (R² < 0.70 for GPP, ER, NEE)
2. **Insufficient dynamic range** (GPP range only 106 gC/m²/year)
3. **Parameter identifiability issues** (perfect correlations)
4. **Missing critical parameters** (no photosynthesis controls)
5. **Ensemble output quality** (unrealistic zeros, low values)

**Recommended Path Forward:**

1. **Phase 1:** Expand parameter list to include photosynthesis parameters (vcmax25, jmax25, slatop, flnr)
2. **Phase 2:** Increase ensemble size to 3000-5000 members with Sobol sampling
3. **Phase 3:** Remove redundant root C/N parameters to fix identifiability
4. **Phase 4:** Retrain surrogates and verify R² > 0.90
5. **Phase 5:** Proceed to MCMC calibration with validated surrogates

**Time Investment:**
- Ensemble rerun: 2-7 days (depending on HPC allocation)
- Surrogate retraining: 1-4 hours
- Validation and diagnostics: 2-4 hours
- Total: ~1-2 weeks

**Expected Benefit:**
- Reliable parameter calibration
- Accurate uncertainty quantification
- Physically meaningful posterior distributions
- Publishable calibration results

---

## Contact & Support

For questions about this analysis:
- Diagnostic scripts: `diagnose_USBlo_surrogate.py`
- OLMT documentation: `CLAUDE.md`
- Surrogate module: `model_ELM/surrogate_NN.py`

**Generated:** October 30, 2025
**Analysis Tool:** Claude Code with OLMT framework
