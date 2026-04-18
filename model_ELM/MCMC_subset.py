"""
MCMC with Parameter Subset Support

This module extends the MCMC functionality to calibrate only a subset of parameters
while keeping others fixed at specified values.

Usage:
------
# Calibrate only leaf parameters, fix TAM root parameters
calibrate_params = ['leafcn', 'slatop', 'flnr', 'leafmr_base']
fixed_params = {
    'froottcn': 120,
    'frootacn': 85,
    'frootmcn': 17,
    # ... all other TAM parameters at fixed values
}

mycase.run_MCMC_subset(
    calibrate_params=calibrate_params,
    fixed_params=fixed_params,
    myvars=['GPP', 'ER'],
    nevals=5000,
    nburn=1000
)
"""

import numpy as np
import os

def run_MCMC_subset(self, calibrate_params, fixed_params, myvars, nevals,
                   mcmc_type='uniform', nburn=5000, burnsteps=10,
                   **kwargs):
    """
    Run MCMC calibration on a subset of parameters while keeping others fixed.

    This function allows you to calibrate only specific parameters while holding
    others at fixed values. Useful when:
    - You want to calibrate leaf parameters while keeping root parameters fixed
    - You have prior knowledge about some parameters
    - You want to reduce MCMC dimensionality for faster convergence

    Parameters
    ----------
    calibrate_params : list of str
        List of parameter names to calibrate during MCMC.
        Must be subset of self.ensemble_parms.
    fixed_params : dict
        Dictionary mapping parameter names to fixed values.
        Keys must be in self.ensemble_parms but not in calibrate_params.
        e.g., {'froottcn': 120, 'frootacn': 85}
    myvars : list of str
        Variables to use for calibration (e.g., ['GPP', 'ER'])
    nevals : int
        Number of MCMC evaluations
    mcmc_type : str
        Prior type: 'uniform' or 'informative'
    nburn : int
        Number of burn-in iterations (default: 5000)
    burnsteps : int
        Number of steps per burn-in iteration (default: 10)
    **kwargs : dict
        Additional arguments passed to main MCMC function

    Returns
    -------
    parms_best : array
        Best parameter values found (full parameter vector)
    posterior_samples : array, optional
        Posterior samples for calibrated parameters only

    Examples
    --------
    # Example 1: Calibrate only leaf parameters
    >>> calibrate_params = ['leafcn', 'slatop', 'flnr']
    >>> fixed_params = {
    ...     'froottcn': 120, 'frootacn': 85, 'frootmcn': 17,
    ...     'froott_long': 6.5, 'froota_long': 1.0, 'frootm_long': 0.5
    ... }
    >>> mycase.run_MCMC_subset(calibrate_params, fixed_params,
    ...                        ['GPP', 'ER'], 5000)

    # Example 2: Calibrate TAM C:N ratios, fix everything else
    >>> calibrate_params = ['froottcn', 'frootacn', 'frootmcn']
    >>> fixed_params = {...}  # All other parameters
    >>> mycase.run_MCMC_subset(calibrate_params, fixed_params,
    ...                        ['GPP'], 5000)
    """

    # ========================================================================
    # VALIDATION AND SETUP
    # ========================================================================

    print("\n" + "="*80)
    print("MCMC WITH PARAMETER SUBSET")
    print("="*80)

    # Validate inputs
    all_param_names = set(self.ensemble_parms)
    calibrate_set = set(calibrate_params)
    fixed_set = set(fixed_params.keys())

    # Check that calibrate_params are valid
    invalid_calibrate = calibrate_set - all_param_names
    if invalid_calibrate:
        raise ValueError(f"Invalid calibration parameters: {invalid_calibrate}")

    # Check that fixed_params are valid
    invalid_fixed = fixed_set - all_param_names
    if invalid_fixed:
        raise ValueError(f"Invalid fixed parameters: {invalid_fixed}")

    # Check for overlap
    overlap = calibrate_set & fixed_set
    if overlap:
        raise ValueError(f"Parameters cannot be both calibrated and fixed: {overlap}")

    # Check that all parameters are accounted for
    unspecified = all_param_names - calibrate_set - fixed_set
    if unspecified:
        print(f"⚠️  WARNING: Some parameters neither calibrated nor fixed: {unspecified}")
        print(f"   These will be fixed at their current mean values")
        # Add unspecified to fixed_params at mean of prior range
        for pname in unspecified:
            idx = self.ensemble_parms.index(pname)
            mean_val = (self.ensemble_pmin[idx] + self.ensemble_pmax[idx]) / 2
            fixed_params[pname] = mean_val
            fixed_set.add(pname)

    # Create mapping from subset to full parameter space
    n_calibrate = len(calibrate_params)
    calibrate_indices = [self.ensemble_parms.index(p) for p in calibrate_params]
    fixed_indices = [self.ensemble_parms.index(p) for p in fixed_params.keys()]

    # Fixed parameter values in correct order
    fixed_values = np.array([fixed_params[self.ensemble_parms[i]]
                            for i in fixed_indices])

    print(f"\nCalibrating {n_calibrate} parameters:")
    for i, pname in enumerate(calibrate_params):
        idx = calibrate_indices[i]
        print(f"  {pname:20s}: [{self.ensemble_pmin[idx]:.2e}, {self.ensemble_pmax[idx]:.2e}]")

    print(f"\nFixing {len(fixed_params)} parameters:")
    for pname, val in sorted(fixed_params.items()):
        idx = self.ensemble_parms.index(pname)
        print(f"  {pname:20s}: {val:.4f} (range: [{self.ensemble_pmin[idx]:.2e}, {self.ensemble_pmax[idx]:.2e}])")

    # ========================================================================
    # CREATE SUBSET CASE OBJECT
    # ========================================================================

    # Store original ensemble attributes
    original_parms = self.ensemble_parms.copy()
    original_nparms = self.nparms_ensemble
    original_pmin = self.ensemble_pmin.copy()
    original_pmax = self.ensemble_pmax.copy()

    # Store fixed parameter info for later restoration
    self._mcmc_subset_info = {
        'original_parms': original_parms,
        'original_nparms': original_nparms,
        'original_pmin': original_pmin,
        'original_pmax': original_pmax,
        'calibrate_params': calibrate_params,
        'calibrate_indices': calibrate_indices,
        'fixed_params': fixed_params,
        'fixed_indices': fixed_indices,
        'fixed_values': fixed_values
    }

    # Temporarily modify ensemble attributes for MCMC
    self.ensemble_parms = calibrate_params
    self.nparms_ensemble = n_calibrate
    self.ensemble_pmin = [original_pmin[i] for i in calibrate_indices]
    self.ensemble_pmax = [original_pmax[i] for i in calibrate_indices]

    # Handle distribution types if present
    if hasattr(self, 'ensemble_dist_type'):
        self._original_dist_type = self.ensemble_dist_type.copy()
        self.ensemble_dist_type = [self._original_dist_type[i] for i in calibrate_indices]

    if hasattr(self, 'ensemble_dist_params'):
        self._original_dist_params = self.ensemble_dist_params.copy()
        self.ensemble_dist_params = [self._original_dist_params[i] for i in calibrate_indices]

    if hasattr(self, 'ensemble_pfts'):
        self._original_pfts = self.ensemble_pfts.copy()
        self.ensemble_pfts = [self._original_pfts[i] for i in calibrate_indices]

    # ========================================================================
    # WRAP SURROGATE TO HANDLE FULL PARAMETER VECTOR
    # ========================================================================

    # Store original run_surrogate method
    self._original_run_surrogate = self.run_surrogate

    def run_surrogate_subset(params_subset, variables):
        """
        Wrapper for run_surrogate that expands subset to full parameter vector.

        Parameters
        ----------
        params_subset : array (n_calibrate,) or (n_samples, n_calibrate)
            Parameter values for calibrated parameters only
        variables : list
            Variables to compute

        Returns
        -------
        output : dict
            Surrogate model outputs
        """
        # Handle both single and batch inputs
        if params_subset.ndim == 1:
            params_subset = params_subset.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False

        n_samples = params_subset.shape[0]

        # Reconstruct full parameter vectors
        params_full = np.zeros((n_samples, original_nparms))

        # Set calibrated parameters
        for i, idx in enumerate(calibrate_indices):
            params_full[:, idx] = params_subset[:, i]

        # Set fixed parameters
        for i, idx in enumerate(fixed_indices):
            params_full[:, idx] = fixed_values[i]

        # Call original surrogate
        output = self._original_run_surrogate(params_full, variables)

        # Squeeze if input was 1D
        if squeeze_output:
            for v in output:
                if isinstance(output[v], np.ndarray) and output[v].shape[0] == 1:
                    output[v] = output[v].squeeze(axis=0)

        return output

    # Replace run_surrogate temporarily
    self.run_surrogate = run_surrogate_subset

    # ========================================================================
    # RUN MCMC ON SUBSET
    # ========================================================================

    # Initial parameter values for calibrated subset (use midpoint)
    parms_initial = np.array([(self.ensemble_pmin[i] + self.ensemble_pmax[i])/2
                              for i in range(n_calibrate)])

    print(f"\nInitial parameter values for MCMC:")
    for i, pname in enumerate(calibrate_params):
        print(f"  {pname:20s}: {parms_initial[i]:.4f}")

    print(f"\nRunning MCMC with {nevals} evaluations, {nburn}×{burnsteps} burn-in...")

    try:
        # Import MCMC function
        from model_ELM.MCMC import MCMC

        # Prepare observation quantiles for MCMC (used for plotting)
        myobs_05 = {}
        myobs_95 = {}

        if hasattr(self, 'obs') and hasattr(self, 'obs_err'):
            print(f"\nPreparing observation confidence intervals...")
            for var in myvars:
                if var in self.obs and var in self.obs_err:
                    obs_array = np.array(self.obs[var])
                    err_array = np.array(self.obs_err[var])

                    # Calculate 5th and 95th percentiles (90% CI)
                    # Assuming Gaussian distribution: μ ± 1.645σ for 90% CI
                    myobs_05[var] = obs_array - 1.645 * err_array
                    myobs_95[var] = obs_array + 1.645 * err_array

                    print(f"  {var}: Confidence intervals prepared")
                else:
                    print(f"  Warning: {var} not in observations, using empty arrays")
                    myobs_05[var] = np.array([])
                    myobs_95[var] = np.array([])
        else:
            print(f"\n  Warning: No observations found (self.obs or self.obs_err missing)")
            print(f"  MCMC will proceed but may not have observation data for likelihood")
            for var in myvars:
                myobs_05[var] = np.array([])
                myobs_95[var] = np.array([])

        # Run MCMC on subset using main MCMC function
        parms_best_subset = MCMC(
            self, parms_initial, myvars, nevals,
            myobs_05=myobs_05, myobs_95=myobs_95,
            mcmc_type=mcmc_type, nburn=nburn, burnsteps=burnsteps,
            **kwargs
        )

    except Exception as e:
        print(f"\n❌ MCMC failed: {e}")
        raise
    finally:
        # ========================================================================
        # RESTORE ORIGINAL ATTRIBUTES
        # ========================================================================

        print(f"\nRestoring original ensemble attributes...")

        self.ensemble_parms = original_parms
        self.nparms_ensemble = original_nparms
        self.ensemble_pmin = original_pmin
        self.ensemble_pmax = original_pmax

        if hasattr(self, '_original_dist_type'):
            self.ensemble_dist_type = self._original_dist_type
            delattr(self, '_original_dist_type')

        if hasattr(self, '_original_dist_params'):
            self.ensemble_dist_params = self._original_dist_params
            delattr(self, '_original_dist_params')

        if hasattr(self, '_original_pfts'):
            self.ensemble_pfts = self._original_pfts
            delattr(self, '_original_pfts')

        # Restore original surrogate function
        self.run_surrogate = self._original_run_surrogate
        delattr(self, '_original_run_surrogate')

    # ========================================================================
    # RECONSTRUCT FULL PARAMETER VECTOR
    # ========================================================================

    print(f"\nReconstructing full parameter vector...")

    parms_best_full = np.zeros(original_nparms)

    # Set calibrated parameters to best values
    for i, idx in enumerate(calibrate_indices):
        parms_best_full[idx] = parms_best_subset[i]

    # Set fixed parameters
    for i, idx in enumerate(fixed_indices):
        parms_best_full[idx] = fixed_values[i]

    print(f"\n✓ MCMC completed")
    print(f"\nBest parameter values (calibrated only):")
    for i, pname in enumerate(calibrate_params):
        print(f"  {pname:20s}: {parms_best_subset[i]:.6f}")

    # ========================================================================
    # SAVE RESULTS WITH SUBSET INFORMATION
    # ========================================================================

    UQ_output = f'./UQ_output/{self.casename}'

    # Save full parameter vector
    with open(f'{UQ_output}/MCMC_output/parms_best_full.txt', 'w') as f:
        f.write("# Best parameters (full vector including fixed values)\n")
        f.write("# Format: parameter_name pft value [calibrated/fixed]\n")
        for i, pname in enumerate(original_parms):
            pft = self.ensemble_pfts[i] if hasattr(self, 'ensemble_pfts') else 0
            status = 'calibrated' if pname in calibrate_set else 'fixed'
            f.write(f"{pname} {pft} {parms_best_full[i]:.10e} {status}\n")

    # Save subset information
    with open(f'{UQ_output}/MCMC_output/subset_info.txt', 'w') as f:
        f.write("MCMC Parameter Subset Configuration\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total parameters: {original_nparms}\n")
        f.write(f"Calibrated parameters: {n_calibrate}\n")
        f.write(f"Fixed parameters: {len(fixed_params)}\n\n")

        f.write("Calibrated Parameters:\n")
        f.write("-"*60 + "\n")
        for pname in calibrate_params:
            idx = original_parms.index(pname)
            f.write(f"  {pname:20s}: [{original_pmin[idx]:.4e}, {original_pmax[idx]:.4e}]\n")

        f.write("\nFixed Parameters:\n")
        f.write("-"*60 + "\n")
        for pname, val in sorted(fixed_params.items()):
            f.write(f"  {pname:20s}: {val:.6e}\n")

    print(f"\n✓ Results saved to {UQ_output}/MCMC_output/")
    print(f"  - parms_best_full.txt : Best parameters (full vector)")
    print(f"  - subset_info.txt     : Subset configuration")
    print(f"  - MCMC_chain.txt      : Posterior samples (calibrated params only)")

    return parms_best_full


def load_MCMC_subset_results(casename):
    """
    Load results from MCMC subset calibration.

    Parameters
    ----------
    casename : str
        Name of the case

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'parms_best_full': Best parameter values (full vector)
        - 'posterior_samples': Posterior samples (calibrated params only)
        - 'calibrated_params': Names of calibrated parameters
        - 'fixed_params': Dict of fixed parameter values
        - 'subset_info': Full subset configuration
    """
    UQ_output = f'./UQ_output/{casename}'

    results = {}

    # Load best parameters (full vector)
    parms_file = f'{UQ_output}/MCMC_output/parms_best_full.txt'
    if os.path.exists(parms_file):
        data = []
        with open(parms_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 4:
                    pname, pft, pval, status = parts[0], int(parts[1]), float(parts[2]), parts[3]
                    data.append({'name': pname, 'pft': pft, 'value': pval, 'status': status})

        results['parms_best_full'] = np.array([d['value'] for d in data])
        results['param_names'] = [d['name'] for d in data]
        results['calibrated_params'] = [d['name'] for d in data if d['status'] == 'calibrated']
        results['fixed_params'] = {d['name']: d['value'] for d in data if d['status'] == 'fixed'}

    # Load posterior samples (calibrated params only)
    chain_file = f'{UQ_output}/MCMC_output/MCMC_chain.txt'
    if os.path.exists(chain_file):
        chain_data = np.loadtxt(chain_file)
        if chain_data.ndim == 1:
            chain_data = chain_data.reshape(-1, 1)
        results['posterior_samples'] = chain_data

    # Load subset info
    info_file = f'{UQ_output}/MCMC_output/subset_info.txt'
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            results['subset_info'] = f.read()

    return results
