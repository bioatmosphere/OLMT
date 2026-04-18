#!/usr/bin/env python
"""
Check surrogate model scalers and training data
"""
import pickle
import numpy as np

# Load the case
pkl_file = '/autofs/nccsopen-svm1_home/6lw/models/OLMT/pklfiles/20251029_AU-Tum_ICB20TRCNPRDCTCBC.pkl'

print(f"Loading case from: {pkl_file}")
with open(pkl_file, 'rb') as f:
    mycase = pickle.load(f)

print("\n" + "="*80)
print("CHECKING SCALERS")
print("="*80)

# Check parameter scaler
if hasattr(mycase, 'pscaler') and mycase.pscaler:
    print("\nParameter scalers found:")
    for var, scaler in mycase.pscaler.items():
        print(f"\n  Variable: {var}")
        if hasattr(scaler, 'mean_'):
            print(f"    Mean: {scaler.mean_[:5]}...")
            print(f"    Scale: {scaler.scale_[:5]}...")
        elif hasattr(scaler, 'data_min_'):
            print(f"    Min: {scaler.data_min_[:5]}...")
            print(f"    Max: {scaler.data_max_[:5]}...")
else:
    print("\n  ⚠️  No parameter scalers found!")

# Check output scaler
if hasattr(mycase, 'yscaler') and mycase.yscaler:
    print("\nOutput scalers found:")
    for var, scaler in mycase.yscaler.items():
        print(f"\n  Variable: {var}")
        if hasattr(scaler, 'mean_'):
            print(f"    Mean: {scaler.mean_}")
            print(f"    Scale (std): {scaler.scale_}")
        elif hasattr(scaler, 'data_min_'):
            print(f"    Min: {scaler.data_min_}")
            print(f"    Max: {scaler.data_max_}")
            print(f"    Range: {scaler.data_max_ - scaler.data_min_}")
else:
    print("\n  ⚠️  No output scalers found!")

# Check surrogate model scores
print("\n" + "="*80)
print("CHECKING SURROGATE MODEL QUALITY")
print("="*80)

for var in ['GPP', 'ER']:
    if var in mycase.surrogate:
        surr = mycase.surrogate[var]
        print(f"\n{var}:")

        # Check if it's GridSearchCV
        if hasattr(surr, 'best_score_'):
            print(f"  Best CV Score (R²): {surr.best_score_:.6f}")
            print(f"  Best params: {surr.best_params_}")

        # Check the underlying estimator
        if hasattr(surr, 'best_estimator_'):
            estimator = surr.best_estimator_
            print(f"  Estimator: {type(estimator).__name__}")

            # For MLPRegressor
            if hasattr(estimator, 'loss_'):
                print(f"  Training loss: {estimator.loss_:.6f}")
            if hasattr(estimator, 'n_iter_'):
                print(f"  Iterations: {estimator.n_iter_}")

# Test with actual scaling
print("\n" + "="*80)
print("TESTING WITH PROPER SCALING")
print("="*80)

# Create test parameters
test_parms = np.array([(mycase.ensemble_pmin[i] + mycase.ensemble_pmax[i])/2
                        for i in range(mycase.nparms_ensemble)])

print(f"\nTest parameters (unscaled): {test_parms[:3]}...")

# Try to manually scale and predict
if 'GPP' in mycase.pscaler and 'GPP' in mycase.yscaler:
    print("\nManually scaling and predicting...")

    # Scale inputs
    pscaler = mycase.pscaler['GPP']
    X_scaled = pscaler.transform(test_parms.reshape(1, -1))
    print(f"  Scaled inputs: {X_scaled[0][:3]}...")

    # Get predictions (scaled)
    surr = mycase.surrogate['GPP']
    if hasattr(surr, 'predict'):
        y_scaled = surr.predict(X_scaled)
        print(f"  Scaled prediction: {y_scaled}")

        # Unscale outputs
        yscaler = mycase.yscaler['GPP']
        y_unscaled = yscaler.inverse_transform(y_scaled.reshape(-1, 1))
        print(f"  Unscaled prediction: {y_unscaled.flatten()}")
        print(f"  Expected range: ~3000-4000 gC/m²/year")

# Check if ensemble matrix exists and look at training data range
print("\n" + "="*80)
print("CHECKING TRAINING DATA")
print("="*80)

if hasattr(mycase, 'output') and mycase.output:
    print("\nTraining data (model outputs) available:")
    for var in ['GPP', 'ER']:
        if var in mycase.output:
            data = mycase.output[var]
            print(f"\n  {var}:")
            print(f"    Shape: {data.shape}")
            print(f"    Min: {np.min(data):.6f}")
            print(f"    Max: {np.max(data):.6f}")
            print(f"    Mean: {np.mean(data):.6f}")
            print(f"    Units: Likely gC/m²/s (needs *31536000 for gC/m²/year)")
else:
    print("\n  ⚠️  No training output data found!")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
