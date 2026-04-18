#!/usr/bin/env python
"""
Analyze which parameter combinations cause zero GPP
"""
import pickle
import numpy as np

with open('pklfiles/20251017_AU-Tum_ICB20TRCNPRDCTCBC.pkl', 'rb') as f:
    mycase = pickle.load(f)

# Get GPP for each ensemble member
gpp = mycase.output['GPP']
print(f'GPP shape: {gpp.shape}')

# Average over years
if gpp.shape[0] < gpp.shape[1]:
    gpp_annual = np.mean(gpp, axis=0) * 31536000  # (n_samples,)
else:
    gpp_annual = np.mean(gpp, axis=1) * 31536000  # (n_samples,)

print(f'Number of ensemble members: {len(gpp_annual)}')

# Identify zero GPP cases
zero_gpp = gpp_annual < 50  # < 50 gC/m²/year = effectively zero
print(f'\nZero GPP cases: {np.sum(zero_gpp)} / {len(gpp_annual)} ({np.sum(zero_gpp)/len(gpp_annual)*100:.1f}%)')
print(f'Normal GPP cases: {np.sum(~zero_gpp)} / {len(gpp_annual)} ({np.sum(~zero_gpp)/len(gpp_annual)*100:.1f}%)')

# Get parameter samples from file
import pandas as pd
ensemble_file = mycase.ensemble_file
print(f'\nLoading parameters from: {ensemble_file}')

psamples = np.loadtxt(ensemble_file)
pnames = mycase.ensemble_parms

if psamples is not None and len(psamples) > 0:

    print('\n' + '='*100)
    print('PARAMETER COMPARISON: Zero GPP vs Normal GPP')
    print('='*100)
    print(f'{"Parameter":<20} {"Zero GPP Mean":>15} {"Normal GPP Mean":>17} {"Difference":>15}')
    print('-'*100)

    critical_params = []

    for i, pname in enumerate(pnames):
        zero_mean = np.mean(psamples[zero_gpp, i])
        normal_mean = np.mean(psamples[~zero_gpp, i])
        diff = zero_mean - normal_mean

        # Flag parameters with large differences
        prange = mycase.ensemble_pmax[i] - mycase.ensemble_pmin[i]
        rel_diff = abs(diff) / prange

        marker = '**' if rel_diff > 0.15 else ''  # >15% of range

        print(f'{pname:<20} {zero_mean:>15.2f} {normal_mean:>17.2f} {diff:>15.2f} {marker}')

        if rel_diff > 0.15:
            critical_params.append((pname, diff, rel_diff))

    print('\n' + '='*100)
    print('CRITICAL PARAMETERS (>15% range difference between zero and normal GPP):')
    print('='*100)

    for pname, diff, rel_diff in sorted(critical_params, key=lambda x: -abs(x[2])):
        direction = 'HIGHER' if diff > 0 else 'LOWER'
        print(f'{pname:20s}: {direction} in zero-GPP cases (Δ = {rel_diff*100:.1f}% of range)')

    # Percentile analysis
    print('\n' + '='*100)
    print('PARAMETER PERCENTILE DISTRIBUTIONS:')
    print('='*100)

    for pname, diff, rel_diff in critical_params:
        i = pnames.index(pname)

        print(f'\n{pname}:')
        print(f'  Zero GPP cases:')
        print(f'    Min: {np.min(psamples[zero_gpp, i]):.2f}')
        print(f'    25%: {np.percentile(psamples[zero_gpp, i], 25):.2f}')
        print(f'    50%: {np.percentile(psamples[zero_gpp, i], 50):.2f}')
        print(f'    75%: {np.percentile(psamples[zero_gpp, i], 75):.2f}')
        print(f'    Max: {np.max(psamples[zero_gpp, i]):.2f}')

        print(f'  Normal GPP cases:')
        print(f'    Min: {np.min(psamples[~zero_gpp, i]):.2f}')
        print(f'    25%: {np.percentile(psamples[~zero_gpp, i], 25):.2f}')
        print(f'    50%: {np.percentile(psamples[~zero_gpp, i], 50):.2f}')
        print(f'    75%: {np.percentile(psamples[~zero_gpp, i], 75):.2f}')
        print(f'    Max: {np.max(psamples[~zero_gpp, i]):.2f}')

        # Find safe range (avoid 90% of zero-GPP cases)
        zero_p90 = np.percentile(psamples[zero_gpp, i], 90)
        zero_p10 = np.percentile(psamples[zero_gpp, i], 10)
        normal_p10 = np.percentile(psamples[~zero_gpp, i], 10)
        normal_p90 = np.percentile(psamples[~zero_gpp, i], 90)

        if diff > 0:  # Zero GPP cases have higher values
            safe_max = zero_p10  # Stay below 10th percentile of zero-GPP
            print(f'  → Suggested max to avoid zero GPP: {safe_max:.2f} (current: {mycase.ensemble_pmax[i]:.2f})')
        else:  # Zero GPP cases have lower values
            safe_min = zero_p90  # Stay above 90th percentile of zero-GPP
            print(f'  → Suggested min to avoid zero GPP: {safe_min:.2f} (current: {mycase.ensemble_pmin[i]:.2f})')

