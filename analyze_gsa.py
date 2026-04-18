#!/usr/bin/env python
"""
Analyze GSA results to identify most influential parameters
"""
import pickle
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol

# Load the case
pkl_file = '/autofs/nccsopen-svm1_home/6lw/models/OLMT/pklfiles/20251029_AU-Tum_ICB20TRCNPRDCTCBC.pkl'

print("="*80)
print("GLOBAL SENSITIVITY ANALYSIS RESULTS")
print("="*80)

with open(pkl_file, 'rb') as f:
    mycase = pickle.load(f)

# Recompute GSA for GPP and ER
target_vars = ['GPP', 'ER']

for var in target_vars:
    if var not in mycase.output:
        print(f"\n{var}: Not in ensemble outputs")
        continue

    print(f"\n{'='*80}")
    print(f"{var} SENSITIVITY INDICES")
    print(f"{'='*80}")

    # Set up SALib problem
    pbounds = [[mycase.ensemble_pmin[i], mycase.ensemble_pmax[i]]
               for i in range(mycase.nparms_ensemble)]

    problem = {
        'num_vars': mycase.nparms_ensemble,
        'names': mycase.ensemble_parms,
        'bounds': pbounds
    }

    # Generate Saltelli samples (need specific size)
    n_saltelli = 1000 #  Adjust based on ensemble size
    try:
        psamples = saltelli.sample(problem, n_saltelli, calc_second_order=False)
    except:
        print(f"  Cannot generate Saltelli samples - using ensemble directly")
        continue

    # Get surrogate outputs
    try:
        surrogate_output = mycase.run_surrogate(psamples, [var])
        Y = np.mean(surrogate_output[var] * 31536000, axis=1)  # Average over years, convert to gC/m²/year

        # Perform Sobol analysis
        Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)

        # Extract sensitivities
        S1 = Si['S1']  # First-order (main effect)
        ST = Si['ST']  # Total-order (includes interactions)

        # Sort by total-order sensitivity
        sorted_indices = np.argsort(ST)[::-1]

        print(f"\nTop 10 Most Influential Parameters:")
        print(f"{'Rank':<6} {'Parameter':<18} {'S1 (Main)':<12} {'ST (Total)':<12} {'Interactions':<12}")
        print(f"{'-'*70}")

        for rank, idx in enumerate(sorted_indices[:10], 1):
            pname = mycase.ensemble_parms[idx]
            s1 = S1[idx]
            st = ST[idx]
            interaction = st - s1  # Interaction effect

            print(f"{rank:<6} {pname:<18} {s1:>10.4f}   {st:>10.4f}   {interaction:>10.4f}")

        # Parameters with ST < 0.05 (low sensitivity)
        low_sens_params = [mycase.ensemble_parms[i] for i in range(mycase.nparms_ensemble) if ST[i] < 0.05]

        if low_sens_params:
            print(f"\nLow Influence Parameters (ST < 0.05):")
            for pname in low_sens_params:
                idx = mycase.ensemble_parms.index(pname)
                print(f"  {pname:<18} ST = {ST[idx]:.4f}")

        # Calculate total variance explained by top parameters
        cumsum_variance = np.cumsum(np.sort(ST)[::-1])
        n_params_for_90pct = np.argmax(cumsum_variance >= 0.9 * np.sum(ST)) + 1

        print(f"\nVariance Explained:")
        print(f"  Top parameter explains: {ST[sorted_indices[0]]/np.sum(ST)*100:.1f}% of variance")
        print(f"  Top 3 parameters explain: {np.sum(ST[sorted_indices[:3]])/np.sum(ST)*100:.1f}% of variance")
        print(f"  Need {n_params_for_90pct} parameters to explain 90% of variance")

    except Exception as e:
        print(f"  Error performing Sobol analysis: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("RECOMMENDATIONS FOR PARAMETER SELECTION")
print("="*80)
print("""
Based on the sensitivity analysis:

1. PARAMETERS TO INCLUDE (ST > 0.1):
   - Focus ensemble and MCMC on these high-influence parameters
   - These explain most of the model output variance

2. PARAMETERS TO CONSIDER FIXING (ST < 0.05):
   - Set these to nominal values to reduce dimensionality
   - Won't significantly affect GPP/ER

3. IF ALL SENSITIVITIES ARE LOW:
   - Current root parameters don't control GPP/ER much
   - Need to add different parameter categories:
     * Photosynthesis: vcmax25, jmax25
     * Allocation: leafcn, fineroot allocation fraction
     * Phenology: GDD thresholds
     * Respiration: q10_mr, base_mr

4. MODEL BIAS ISSUE:
   - Remember: Model produces 38% of observed GPP
   - Parameter calibration CANNOT fix a 62% bias
   - Must first fix model configuration (PFT, forcing, etc.)
   - Or accept that model has structural limitations
""")
