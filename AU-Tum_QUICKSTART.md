# AU-Tum Calibration Quick Start Guide

## TL;DR

**Use this parameter file**: `inputdata/PTTAM/AU-Tum_parm_list_tam_v2_calibration`

**What it includes**:
- ✓ All 17 TAM root parameters (complete framework)
- ✓ 4 leaf photosynthesis parameters (enables calibration)
- ✓ Zero-GPP prevention (down from 34% failure rate)
- ✓ Can reach observed GPP range (2903-4323 gC/m²/year)

**Total**: 21 parameters

## Quick Comparison

| File | Parameters | Purpose | Zero GPP | GPP Range | Use Case |
|------|-----------|---------|----------|-----------|----------|
| **v2_calibration** ✓ | 21 | **GPP/ER calibration** | <5% | **Full** | **← Start here** |
| v2_complete | 17 | TAM structure only | <5% | Limited | TAM-focused studies |
| v2_robust | 13 | Simplified robust | <5% | Full | Alternative |
| Original ❌ | 17 | Reference only | 34% | Limited | Don't use |

## Step-by-Step Instructions

### 1. Update `runscripts/run_TAM1.py`

```python
# Site configuration
sites = ['AU-Tum']
sitegroup = 'TAM'

# USE THIS PARAMETER FILE
parm_list = 'inputdata/PTTAM/AU-Tum_parm_list_tam_v2_calibration'

# Ensemble size (CRITICAL!)
nsamples = 1500      # Recommended (21 params × 70 samples/param)
np_ensemble = 384    # Adjust for your system

# Match observation period
postproc_startyear = 2001
postproc_endyear = 2014
```

### 2. Run Ensemble

```bash
cd /autofs/nccsopen-svm1_home/6lw/models/OLMT/runscripts
python run_TAM1.py
```

**Expected time**: Several hours to 1-2 days (1500 × 14 = 21,000 runs)

### 3. Validate Surrogates

```bash
cd /autofs/nccsopen-svm1_home/6lw/models/OLMT
python check_AUTum_surrogate_robustness.py
```

**Look for**:
- ✓ GPP R² > 0.90
- ✓ ER R² > 0.90
- ✓ Zero GPP < 5%
- ✓ GPP range covers 2903-4323 gC/m²/year

### 4. Run MCMC Calibration

```python
import pickle

# Load case
with open('pklfiles/LATEST_AU-Tum_ICB20TRCNPRDCTCBC.pkl', 'rb') as f:
    mycase = pickle.load(f)

# Run MCMC
mycase.run_MCMC(variables=['GPP', 'ER'], nsamples=5000, burnin=1000)
```

## What Each Parameter Does

### TAM Root Parameters (17)

**Control nitrogen supply to leaves**:
- `froottcn, frootacn, frootmcn`: Root N content
- `froottcp, frootacp, frootmcp`: Root P content
- `froott_long, froota_long, frootm_long`: Root lifespan
- `frt_flab, frt_fcel, fra_flab, fra_fcel, frm_flab, frm_fcel`: Root chemistry
- `froott_leaf, froota_leaf`: C allocation to leaves

### Leaf Photosynthesis Parameters (4)

**Convert N supply to GPP**:

1. **`leafcn`** (20-32): Leaf C:N ratio
   - Lower = More N per leaf = Higher photosynthesis
   - **Most influential for GPP** (~5-10% impact)

2. **`slatop`** (0.008-0.016 m²/gC): Specific leaf area
   - Higher = More leaf area = More light capture
   - **Most influential for GPP** (~5-10% impact)

3. **`flnr`** (0.05-0.12): Fraction of leaf N in Rubisco
   - Higher = More efficient N use in photosynthesis
   - **Moderate influence** (~3-8% impact)

4. **`leafmr_base`** (8e-07 to 2.2e-06 gC/gC/s): Leaf respiration rate
   - Controls ER magnitude
   - **Moderate influence on ER** (~2-5% impact)

## Observations to Match

**AU-Tum FLUXNET 2001-2014**:
- GPP: 2903-4323 gC/m²/year (mean 3404)
- RECO: 2134-3355 gC/m²/year (mean 2721)
- NEE: -1306 to -359 gC/m²/year (mean -667)

## Success Criteria

**Surrogates are robust when**:
- [x] R² scores > 0.90 (GPP, ER, NPP)
- [x] R² scores > 0.85 (NEE)
- [x] Zero GPP cases < 5%
- [x] GPP range: 2500-4500 gC/m²/year
- [x] ER range: 2000-3500 gC/m²/year
- [x] No unrealistic predictions

**MCMC converges when**:
- [x] Acceptance rate: 20-40%
- [x] Trace plots show good mixing
- [x] Gelman-Rubin statistic < 1.1
- [x] Posterior predictions match observations

## Files and Documentation

**Parameter files** (in order of recommendation):
1. `AU-Tum_parm_list_tam_v2_calibration` ← **Use this**
2. `AU-Tum_parm_list_tam_v2_complete` (TAM-only alternative)
3. `AU-Tum_parm_list_tam_v2_robust` (simplified alternative)

**Diagnostic tools**:
- `check_AUTum_surrogate_robustness.py` - Comprehensive validation
- `analyze_zero_gpp.py` - Zero-GPP debugging

**Documentation**:
- `AU-Tum_QUICKSTART.md` ← You are here
- `AU-Tum_Calibration_Parameters_Guide.md` - Full details
- `AU-Tum_Surrogate_Robustness_Report.md` - Analysis report

## Common Issues

### Zero GPP still occurring?
```bash
python analyze_zero_gpp.py
```
This parameter file should prevent it (<5% expected).

### Surrogate quality poor?
- Check ensemble size (need 1500+ samples for 21 parameters)
- Check for model crashes in logs
- Increase to 2000 samples if needed

### GPP range insufficient?
Should not occur with this file. If it does:
- Check that leaf parameters (leafcn, slatop, flnr) are being used
- Verify parameter bounds in generated ensemble file

### MCMC not converging?
- Run longer chain (5000 → 10000 samples)
- Check surrogate quality first (must have R² > 0.90)
- Adjust proposal variance based on acceptance rate

## Next Steps After Calibration

1. **Posterior analysis**: Identify most important parameters
2. **Posterior predictive checks**: Validate model-data fit
3. **Sensitivity analysis**: Run GSA with `mycase.run_GSA()`
4. **Uncertainty quantification**: Propagate posterior uncertainty
5. **Scientific interpretation**: What do parameters tell us about AU-Tum?

## Support

Questions? Check the detailed guides:
- Implementation details → `AU-Tum_Calibration_Parameters_Guide.md`
- Parameter changes → `AU-Tum_v2_complete_Changes.md`
- Analysis report → `AU-Tum_Surrogate_Robustness_Report.md`

---

**Last updated**: 2025-10-31
**Status**: Production-ready
**Recommendation**: Use `AU-Tum_parm_list_tam_v2_calibration` with 1500 samples
