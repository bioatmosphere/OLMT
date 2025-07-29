import numpy as np
from scipy.stats import norm
from scipy.signal import correlate
import model_surrogate as models
import os, math, random
import matplotlib
matplotlib.use('Agg')
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import warnings
from optparse import OptionParser

# Optional PyMC3 imports - only import if available
try:
    import pymc3 as pm
    import theano.tensor as tt
    import arviz as az
    PYMC3_AVAILABLE = True
except ImportError:
    PYMC3_AVAILABLE = False
    print("PyMC3 not available. Using custom MCMC implementation only.")

# ======================================================================
# Effective Sample Size (ESS) Calculation Functions
# ======================================================================

def next_pow_two(n):
    """Find the next power of two greater than or equal to n."""
    return 1 << (n - 1).bit_length()

def autocorr_func_1d(x, norm=True):
    """
    Calculate the autocorrelation function for a 1D array using FFT.
    
    Parameters
    ----------
    x : array_like
        Input 1D array (MCMC chain).
    norm : bool
        If True, normalize by the zero-lag autocorrelation.
        
    Returns
    -------
    acf : ndarray
        Autocorrelation function.
    """
    x = np.atleast_1d(x)
    if len(x) < 2:
        return np.array([1.0])
        
    # Zero-pad to avoid circular correlation
    n = next_pow_two(len(x))
    
    # Subtract mean and compute FFT
    x_centered = x - np.mean(x)
    f = np.fft.fft(x_centered, n=2 * n)
    
    # Compute power spectral density and inverse FFT
    acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
    acf /= (4 * n)
    
    if norm and acf[0] != 0:
        acf /= acf[0]
    
    return acf

def auto_window(taus, c=5.0):
    """
    Automatic windowing procedure for autocorrelation time estimation.
    
    Parameters
    ----------
    taus : array_like
        Cumulative autocorrelation times at different window sizes.
    c : float
        Window size factor. Default is 5.0.
        
    Returns
    -------
    window : int
        Optimal window size index.
    """
    # Find where the window size becomes too large relative to the estimated tau
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def autocorr_time_1d(x, c=5.0, quiet=False):
    """
    Calculate the autocorrelation time for a 1D MCMC chain.
    
    Parameters
    ----------
    x : array_like
        Input 1D MCMC chain.
    c : float
        Window size factor for automatic windowing. Default is 5.0.
    quiet : bool
        If True, suppress warnings about insufficient chain length.
        
    Returns
    -------
    tau : float
        Estimated autocorrelation time.
    """
    x = np.atleast_1d(x)
    if len(x) < 10:
        if not quiet:
            warnings.warn("Chain too short for reliable autocorrelation estimate")
        return len(x)
    
    # Compute autocorrelation function
    f = autocorr_func_1d(x)
    
    # Compute integrated autocorrelation time
    taus = 2.0 * np.cumsum(f) - 1.0
    
    # Apply automatic windowing
    window = auto_window(taus, c)
    
    # Check if the window is reasonable
    if window == 0:
        if not quiet:
            warnings.warn("Autocorrelation time could not be estimated reliably")
        return len(x)
    
    return taus[window]

def autocorr_time_multi(chains, c=5.0, quiet=False):
    """
    Calculate autocorrelation time for multiple chains (ensemble average).
    
    Parameters
    ----------
    chains : array_like
        2D array where each row is a separate chain.
    c : float
        Window size factor. Default is 5.0.
    quiet : bool
        If True, suppress warnings.
        
    Returns
    -------
    tau : float
        Ensemble-averaged autocorrelation time.
    """
    chains = np.atleast_2d(chains)
    if chains.shape[0] == 1:
        return autocorr_time_1d(chains[0], c=c, quiet=quiet)
    
    # Calculate autocorrelation function for each chain
    n_chains, n_samples = chains.shape
    f_total = np.zeros(n_samples)
    
    for chain in chains:
        if len(chain) > 1:
            f_total += autocorr_func_1d(chain)
    
    # Average across chains
    f_avg = f_total / n_chains
    
    # Compute integrated autocorrelation time
    taus = 2.0 * np.cumsum(f_avg) - 1.0
    
    # Apply automatic windowing
    window = auto_window(taus, c)
    
    if window == 0:
        if not quiet:
            warnings.warn("Autocorrelation time could not be estimated reliably")
        return n_samples
    
    return taus[window]

def effective_sample_size(chain, c=5.0, quiet=False):
    """
    Calculate the effective sample size (ESS) for MCMC chains.
    
    Parameters
    ----------
    chain : array_like
        MCMC chain(s). Can be 1D (single chain) or 2D (multiple chains/parameters).
    c : float
        Window size factor for autocorrelation time estimation. Default is 5.0.
    quiet : bool
        If True, suppress warnings.
        
    Returns
    -------
    ess : float or dict
        Effective sample size. If input is 2D, returns ESS for each parameter.
        If the input corresponds to parameter chains, returns a dictionary with
        parameter names as keys.
    """
    chain = np.atleast_1d(chain)
    
    if chain.ndim == 1:
        # Single chain
        n_samples = len(chain)
        if n_samples < 10:
            if not quiet:
                warnings.warn("Chain too short for reliable ESS calculation")
            return 1.0
        
        tau = autocorr_time_1d(chain, c=c, quiet=quiet)
        return n_samples / tau
    
    elif chain.ndim == 2:
        # Multiple parameters or multiple chains
        n_params, n_samples = chain.shape
        ess_values = {}
        
        for i in range(n_params):
            tau = autocorr_time_1d(chain[i], c=c, quiet=quiet)
            ess_values[f'param_{i}'] = n_samples / tau
        
        return ess_values
    
    else:
        raise ValueError("Input chain must be 1D or 2D array")

def ess_summary_stats(ess_dict):
    """
    Calculate summary statistics for effective sample sizes.
    
    Parameters
    ----------
    ess_dict : dict
        Dictionary of ESS values for different parameters.
        
    Returns
    -------
    stats : dict
        Dictionary containing min, max, mean, and median ESS values.
    """
    if not isinstance(ess_dict, dict):
        return {'min': ess_dict, 'max': ess_dict, 'mean': ess_dict, 'median': ess_dict}
    
    ess_values = np.array(list(ess_dict.values()))
    
    return {
        'min': np.min(ess_values),
        'max': np.max(ess_values),
        'mean': np.mean(ess_values),
        'median': np.median(ess_values),
        'n_params': len(ess_values)
    }

def plot_autocorr_functions(chains, param_names=None, output_dir='.', max_lag=None):
    """
    Plot autocorrelation functions for MCMC chains.
    
    Parameters
    ----------
    chains : array_like
        2D array where each row is a parameter chain.
    param_names : list, optional
        Names of parameters. If None, uses generic names.
    output_dir : str
        Directory to save plots.
    max_lag : int, optional
        Maximum lag to plot. If None, uses chain length // 4.
    """
    chains = np.atleast_2d(chains)
    n_params, n_samples = chains.shape
    
    if param_names is None:
        param_names = [f'param_{i}' for i in range(n_params)]
    
    if max_lag is None:
        max_lag = min(n_samples // 4, 200)
    
    # Create autocorrelation plots directory
    acf_plot_dir = os.path.join(output_dir, 'plots', 'autocorr_functions')
    os.makedirs(acf_plot_dir, exist_ok=True)
    
    # Plot individual autocorrelation functions
    for i, param_name in enumerate(param_names):
        if i >= n_params:
            break
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate autocorrelation function
        acf = autocorr_func_1d(chains[i])
        lags = np.arange(min(len(acf), max_lag))
        
        # Plot autocorrelation function
        ax.plot(lags, acf[:len(lags)], 'b-', linewidth=1.5)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.axhline(y=0.05, color='r', linestyle=':', alpha=0.7, label='5% threshold')
        ax.axhline(y=-0.05, color='r', linestyle=':', alpha=0.7)
        
        # Calculate and show integrated autocorrelation time
        tau = autocorr_time_1d(chains[i], quiet=True)
        ax.axvline(x=tau, color='orange', linestyle='--', alpha=0.8, 
                  label=f'τ = {tau:.1f}')
        
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        ax.set_title(f'Autocorrelation Function - {param_name}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(acf_plot_dir, f'autocorr_{param_name}.pdf'))
        plt.close()
    
    # Create summary plot with all autocorrelation functions
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_params, 10)))
    
    for i, param_name in enumerate(param_names[:min(n_params, 10)]):
        acf = autocorr_func_1d(chains[i])
        lags = np.arange(min(len(acf), max_lag))
        
        ax.plot(lags, acf[:len(lags)], color=colors[i], 
               linewidth=1.5, label=param_name, alpha=0.8)
    
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.axhline(y=0.05, color='r', linestyle=':', alpha=0.7, label='±5% threshold')
    ax.axhline(y=-0.05, color='r', linestyle=':', alpha=0.7)
    
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Autocorrelation Functions - All Parameters')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(acf_plot_dir, 'autocorr_all_parameters.pdf'))
    plt.close()

def create_ess_report(chain_file, param_names=None, output_dir='.', 
                     chain_format='custom', burn_in_frac=0.0):
    """
    Create a comprehensive ESS report from saved MCMC chain files.
    
    Parameters
    ----------
    chain_file : str
        Path to the MCMC chain file.
    param_names : list, optional
        Names of parameters. If None, uses generic names.
    output_dir : str
        Directory to save the report.
    chain_format : str
        Format of the chain file ('custom' or 'pymc3').
    burn_in_frac : float
        Fraction of samples to discard as burn-in (0.0 to 1.0).
        
    Returns
    -------
    report : dict
        Dictionary containing ESS analysis results.
    """
    # Load chain data
    if not os.path.exists(chain_file):
        raise FileNotFoundError(f"Chain file not found: {chain_file}")
    
    chains = np.loadtxt(chain_file)
    
    # Handle different chain formats
    if chain_format == 'custom':
        # Custom format: rows are samples, columns are parameters
        chains = chains.T  # Transpose to get parameters as rows
    elif chain_format == 'pymc3':
        # PyMC3 format: already in correct format
        pass
    
    # Apply burn-in
    if burn_in_frac > 0:
        n_samples = chains.shape[1]
        burn_samples = int(burn_in_frac * n_samples)
        chains = chains[:, burn_samples:]
        print(f"Applied burn-in: removed {burn_samples} samples")
    
    n_params, n_samples = chains.shape
    
    if param_names is None:
        param_names = [f'param_{i}' for i in range(n_params)]
    
    # Calculate ESS for each parameter
    ess_results = {}
    tau_results = {}
    
    for i, param_name in enumerate(param_names[:n_params]):
        tau = autocorr_time_1d(chains[i], quiet=True)
        ess = n_samples / tau
        ess_results[param_name] = ess
        tau_results[param_name] = tau
    
    # Calculate summary statistics
    ess_stats = ess_summary_stats(ess_results)
    
    # Create report
    report = {
        'ess_by_param': ess_results,
        'tau_by_param': tau_results,
        'summary_stats': ess_stats,
        'n_samples': n_samples,
        'n_params': n_params,
        'mean_efficiency': ess_stats['mean'] / n_samples,
        'min_efficiency': ess_stats['min'] / n_samples
    }
    
    # Save comprehensive report
    report_file = os.path.join(output_dir, 'ess_comprehensive_report.txt')
    with open(report_file, 'w') as f:
        f.write("COMPREHENSIVE EFFECTIVE SAMPLE SIZE (ESS) REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Chain file: {chain_file}\n")
        f.write(f"Total samples: {n_samples}\n")
        f.write(f"Number of parameters: {n_params}\n")
        f.write(f"Burn-in fraction applied: {burn_in_frac:.1%}\n\n")
        
        f.write("SUMMARY STATISTICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Min ESS: {ess_stats['min']:.1f} ({ess_stats['min']/n_samples:.1%} efficiency)\n")
        f.write(f"Mean ESS: {ess_stats['mean']:.1f} ({ess_stats['mean']/n_samples:.1%} efficiency)\n")
        f.write(f"Median ESS: {ess_stats['median']:.1f} ({ess_stats['median']/n_samples:.1%} efficiency)\n")
        f.write(f"Max ESS: {ess_stats['max']:.1f} ({ess_stats['max']/n_samples:.1%} efficiency)\n\n")
        
        f.write("PARAMETER-SPECIFIC RESULTS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'Parameter':<20} {'ESS':<8} {'Tau':<8} {'Efficiency':<10} {'ESS/1000':<8}\n")
        f.write("-" * 60 + "\n")
        
        for param_name in param_names[:n_params]:
            if param_name in ess_results:
                ess_val = ess_results[param_name]
                tau_val = tau_results[param_name]
                efficiency = ess_val / n_samples
                ess_per_1000 = (ess_val / n_samples) * 1000
                f.write(f"{param_name:<20} {ess_val:<8.1f} {tau_val:<8.1f} {efficiency:<10.1%} {ess_per_1000:<8.1f}\n")
        
        f.write("\nRECOMMENDations:\n")
        f.write("-" * 30 + "\n")
        
        if ess_stats['min'] < 100:
            f.write("⚠️  WARNING: Some parameters have ESS < 100. Consider:\n")
            f.write("   - Running longer chains\n")
            f.write("   - Improving mixing with different step sizes\n")
            f.write("   - Using more efficient samplers\n\n")
        
        if ess_stats['mean'] / n_samples < 0.1:
            f.write("⚠️  WARNING: Mean efficiency < 10%. Chain may be poorly mixed.\n")
            f.write("   Consider tuning sampler parameters.\n\n")
        
        if ess_stats['mean'] / n_samples > 0.5:
            f.write("✅ Good: Mean efficiency > 50%. Chains are well-mixed.\n\n")
    
    # Create visualizations
    plot_autocorr_functions(chains, param_names, output_dir)
    
    print(f"Comprehensive ESS report saved to: {report_file}")
    return report

def calc_posterior(self,parms,myvars):
    """Calculate the posterior (prior and log likelihood)

    Calls run_suggrogate() in surrogate_NN.py for surrogate model evaluation.

    Parameters
    ----------
    parms : array-like
        Parameter values for which to calculate the posterior.
    myvars : list
        List of variable names for which to calculate the posterior.
    
    Returns
    -------
    post : float
        The posterior value.
    output : dict
        The model output for the specified variables.
    """

    #line = 0
    #Uniform priors
    prior = 1.0
    for j in range(0,self.nparms_ensemble):
        if (parms[j] < self.ensemble_pmin[j] or parms[j] > self.ensemble_pmax[j]):
            prior = 0.0
    post = prior
    if (prior > 0.0):
      # Run surrogate model to get predictions
      output = self.run_surrogate(parms.reshape(1, -1), myvars)
      
      # Apply unit conversions for carbon flux variables from gC/m²/s to gC/m²/year
      flux_vars = ['GPP', 'ER', 'NEE']
      for var in output:
          #var_base = var.split('_pft')[0]  # Remove _pft suffix for comparison
          if var in flux_vars:
              # Convert from gC/m²/s to gC/m²/year
              # Multiply by seconds per year: 365.25 * 24 * 3600 = 31,557,600 seconds/year
              output[var] = output[var] * 31557600.0
      
      # Calculate likelihood for each variable
      for v in myvars:
          model_output = output[v].flatten()
          observations = np.array(self.obs[v]).flatten()
          uncertainties = np.array(self.obs_err[v]).flatten()
          
          # Vectorized likelihood calculation for valid observations
          valid_mask = (observations > -9000) & (uncertainties > 0)
          valid_obs = observations[valid_mask]
          valid_pred = model_output[valid_mask]
          valid_err = uncertainties[valid_mask]
          
          if len(valid_obs) > 0:
              # Calculate residuals and normalized residuals
              residuals = valid_pred - valid_obs
              normalized_residuals_sq = (residuals / valid_err) ** 2
              
              # Gaussian log-likelihood: log(1/√(2π)) - log(σ) - (residual/σ)²/2
              log_likelihood = (
                  -0.5 * np.log(2.0 * np.pi) - 
                  np.log(valid_err) - 
                  0.5 * normalized_residuals_sq
              )
              
              # Add to total posterior
              post += np.sum(log_likelihood)
    else:
        post = -9999999
        output={}
    #print(post)
    return(post, output)

def MCMC_pymc3(self, parms, myvars, nevals, tune=1000, target_accept=0.9, sampler='NUTS'):
    """
    PyMC3-based MCMC implementation for parameter estimation.
    
    Parameters
    ----------
    parms : array-like
        Initial parameter values (not used in PyMC3, but kept for compatibility).
    myvars : list
        List of variable names for which to perform MCMC sampling.
    nevals : int
        Number of samples to draw after tuning.
    tune : int
        Number of tuning/burn-in samples. Default is 1000.
    target_accept : float
        Target acceptance rate for NUTS sampler. Default is 0.9.
    sampler : str
        Sampler type ('NUTS', 'Metropolis', 'ADVI'). Default is 'NUTS'.
    
    Returns
    -------
    parms_best : array-like
        Best parameter values (MAP estimate).
    trace : InferenceData
        ArviZ InferenceData object containing the MCMC trace.
    """
    
    if not PYMC3_AVAILABLE:
        raise ImportError("PyMC3 not available. Install with: pip install pymc3 theano arviz")
    
    UQ_output = './UQ_output/' + self.casename
    os.makedirs(UQ_output + '/PyMC3_output', exist_ok=True)
    
    # Define custom log-likelihood function for PyMC3
    @pm.as_op(itypes=[tt.dvector], otypes=[tt.dscalar])
    def loglike_op(params_tt):
        try:
            params_np = np.array(params_tt)
            post, _ = calc_posterior(self, params_np, myvars)
            return post if post > -9999999 else -1e10
        except:
            return -1e10
    
    with pm.Model() as model:
        # Define uniform priors for parameters
        params = pm.Uniform('params', 
                           lower=self.ensemble_pmin, 
                           upper=self.ensemble_pmax, 
                           shape=self.nparms_ensemble,
                           testval=parms)
        
        # Define likelihood using custom log-likelihood function
        likelihood = pm.DensityDist('likelihood', loglike_op, observed=params)
        
        # Choose sampler
        if sampler == 'NUTS':
            step = pm.NUTS(target_accept=target_accept)
        elif sampler == 'Metropolis':
            step = pm.Metropolis()
        elif sampler == 'ADVI':
            # Use ADVI for variational inference
            approx = pm.fit(n=nevals + tune, method='advi')
            trace = approx.sample(draws=nevals)
            # Convert to InferenceData format
            trace = az.from_pymc3(trace)
            
            # Get MAP estimate
            parms_best = approx.bij.rmap(approx.mean.eval())['params']
            
            # Save results
            self._save_pymc3_results(trace, parms_best, UQ_output, myvars)
            return parms_best, trace
        else:
            raise ValueError("Sampler must be 'NUTS', 'Metropolis', or 'ADVI'")
        
        # Sample
        print(f"Starting PyMC3 {sampler} sampling...")
        trace = pm.sample(draws=nevals, tune=tune, step=step, 
                         return_inferencedata=True, cores=1)
    
    # Extract best parameters (MAP estimate)
    posterior_samples = trace.posterior['params'].values
    log_likelihood = trace.log_likelihood['likelihood'].values
    
    # Find best parameters
    best_idx = np.unravel_index(np.argmax(log_likelihood), log_likelihood.shape)
    parms_best = posterior_samples[best_idx[0], best_idx[1], :]
    
    # Save results
    self._save_pymc3_results(trace, parms_best, UQ_output, myvars)
    
    print(f"PyMC3 sampling completed. Results saved to {UQ_output}/PyMC3_output/")
    return parms_best, trace

def _save_pymc3_results(self, trace, parms_best, UQ_output, myvars):
    """Save PyMC3 results in similar format to custom MCMC."""
    
    # Save best parameters
    with open(UQ_output + '/PyMC3_output/parms_best.txt', 'w') as f:
        for p, (pname, pft, pval) in enumerate(zip(self.ensemble_parms, self.ensemble_pfts, parms_best)):
            f.write(f"{pname} {pft} {pval}\n")
    
    # Save trace summary
    summary = az.summary(trace)
    summary.to_csv(UQ_output + '/PyMC3_output/trace_summary.csv')
    
    # Calculate and save ESS diagnostics
    samples = trace.posterior['params'].values
    n_chains, n_draws, n_params = samples.shape
    
    # Reshape for ESS calculation (combine chains)
    samples_combined = samples.reshape(-1, n_params)
    
    # Calculate ESS for each parameter
    ess_results = {}
    tau_results = {}
    
    for i in range(n_params):
        param_name = self.ensemble_parms[i] if i < len(self.ensemble_parms) else f'param_{i}'
        
        # Calculate ESS using our custom function
        param_samples = samples_combined[:, i]
        tau = autocorr_time_1d(param_samples, quiet=True)
        ess = len(param_samples) / tau
        
        ess_results[param_name] = ess
        tau_results[param_name] = tau
    
    # Calculate summary statistics
    ess_stats = ess_summary_stats(ess_results)
    
    # Save ESS results
    with open(UQ_output + '/PyMC3_output/ess_diagnostics.txt', 'w') as f:
        f.write("# Effective Sample Size (ESS) Diagnostics\n")
        f.write(f"# Total samples: {len(samples_combined)}\n")
        f.write(f"# Number of parameters: {n_params}\n")
        f.write(f"# Number of chains: {n_chains}\n")
        f.write(f"# Draws per chain: {n_draws}\n\n")
        
        f.write("# Summary Statistics:\n")
        f.write(f"Min ESS: {ess_stats['min']:.2f}\n")
        f.write(f"Max ESS: {ess_stats['max']:.2f}\n")
        f.write(f"Mean ESS: {ess_stats['mean']:.2f}\n")
        f.write(f"Median ESS: {ess_stats['median']:.2f}\n\n")
        
        f.write("# Parameter-specific results:\n")
        f.write("Parameter\tESS\tAutocorr_Time\tESS_per_1000\n")
        for param_name in ess_results:
            ess_val = ess_results[param_name]
            tau_val = tau_results[param_name]
            ess_per_1000 = (ess_val / len(samples_combined)) * 1000
            f.write(f"{param_name}\t{ess_val:.2f}\t{tau_val:.2f}\t{ess_per_1000:.1f}\n")
    
    # Create diagnostic plots
    os.makedirs(UQ_output + '/PyMC3_output/plots', exist_ok=True)
    
    # ESS bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    param_names = list(ess_results.keys())
    ess_values = list(ess_results.values())
    
    bars = ax.bar(range(len(param_names)), ess_values)
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Effective Sample Size')
    ax.set_title('Effective Sample Size by Parameter')
    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    
    # Add horizontal line for total samples
    ax.axhline(y=len(samples_combined), color='red', linestyle='--', 
               label=f'Total samples ({len(samples_combined)})')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(UQ_output + '/PyMC3_output/plots/ess_by_parameter.pdf')
    plt.close()
    
    # Trace plots
    az.plot_trace(trace, var_names=['params'])
    plt.savefig(UQ_output + '/PyMC3_output/plots/trace_plot.pdf')
    plt.close()
    
    # Posterior plots
    az.plot_posterior(trace, var_names=['params'])
    plt.savefig(UQ_output + '/PyMC3_output/plots/posterior_plot.pdf')
    plt.close()
    
    # Rank plots for diagnostics
    az.plot_rank(trace)
    plt.savefig(UQ_output + '/PyMC3_output/plots/rank_plot.pdf')
    plt.close()
    
    # Save raw samples
    np.savetxt(UQ_output + '/PyMC3_output/MCMC_chain.txt', samples_combined)
    
    print(f"PyMC3 diagnostics saved to {UQ_output}/PyMC3_output/plots/")
    print(f"ESS Summary - Min: {ess_stats['min']:.1f}, Mean: {ess_stats['mean']:.1f}, Max: {ess_stats['max']:.1f}")

def MCMC_custom(self, parms, myvars, nevals, *, 
         mcmc_type='uniform', nburn=1000, burnsteps=10, 
         default_output=None, sampler='custom', **kwargs):
    """
    Perform Markov Chain Monte Carlo (MCMC) to estimate the posterior distribution of parameters.

    Parameters
    ----------
    parms : array-like
        Initial parameter values for MCMC sampling.
    myvars : list
        List of variable names for which to perform MCMC sampling.
    nevals : int
        Number of evaluations for MCMC sampling.
    mcmc_type : str
        Type of MCMC sampling to perform. Default is 'uniform'.
    nburn : int
        Number of burn-in steps for MCMC sampling. Default is 1000.
    burnsteps : int
        Number of burn-in steps for MCMC sampling. Default is 10.
    default_output : list
        Default output values for comparison. Default is empty list.
    sampler : str
        MCMC implementation to use. Options:
        - 'custom': Use custom Metropolis-Hastings implementation (default)
        - 'pymc3': Use PyMC3 with NUTS sampler
        - 'pymc3_metropolis': Use PyMC3 with Metropolis sampler
        - 'pymc3_advi': Use PyMC3 with ADVI variational inference
    **kwargs : dict
        Additional arguments passed to PyMC3 sampler (e.g., tune, target_accept)

    Returns
    -------
    parms_best : array-like
        Best parameter values found during MCMC sampling.
    trace : optional
        For PyMC3 samplers, also returns the trace object.
    """
    
    # Route to appropriate implementation
    if sampler == 'custom':
        return self.MCMC(parms, myvars, nevals, mcmc_type, nburn, burnsteps, default_output)
    elif sampler in ['pymc3', 'pymc3_nuts']:
        tune = kwargs.get('tune', nburn * burnsteps)
        target_accept = kwargs.get('target_accept', 0.9)
        return MCMC_pymc3(self, parms, myvars, nevals, tune=tune, target_accept=target_accept, sampler='NUTS')
    elif sampler == 'pymc3_metropolis':
        tune = kwargs.get('tune', nburn * burnsteps)
        return MCMC_pymc3(self, parms, myvars, nevals, tune=tune, sampler='Metropolis')
    elif sampler == 'pymc3_advi':
        tune = kwargs.get('tune', nburn * burnsteps)
        return MCMC_pymc3(self, parms, myvars, nevals, tune=tune, sampler='ADVI')
    else:
        raise ValueError(f"Unknown sampler: {sampler}. Choose from 'custom', 'pymc3', 'pymc3_metropolis', 'pymc3_advi'")

def MCMC(self, parms, myvars, nevals, mcmc_type='uniform', nburn=1000, burnsteps=10, default_output=None):
    """
    Original custom Metropolis-Hastings MCMC implementation.
    
    (Documentation same as main MCMC function)
    """
    
    UQ_output='./UQ_output/'+self.casename
    print(os.path.abspath(UQ_output))
    #Metropolis-Hastings Markov Chain Monte Carlo with adaptive sampling
    post_best = -99999
    post_last = -99999
    accepted_step = 0
    accepted_tot  = 0
    nparms     = self.nparms_ensemble
    #parms      = np.zeros(nparms)
    parm_step  = np.zeros(nparms)
    chain      = np.zeros((nparms+1,nevals))
    chain_prop = np.zeros((nparms,nevals))
    chain_burn = np.zeros((nparms,nevals))
    output     = {}
    self.nobs  = {}
    for v in myvars:
      self.nobs[v] = len(self.output[v])
      output[v]     = np.zeros((self.nobs[v],nevals))
    mycov      = np.zeros((nparms,nparms))
    for p in range(0,nparms):
        #Starting step size - reduced for more conservative proposals
        #parm_step[p] = 2.4**2/nparms * (model.pmax[p]-model.pmin[p])
        base_step = 0.02 * (self.ensemble_pmax[p]-self.ensemble_pmin[p])  # Reduced from 5% to 2%
        
        # Parameter-specific scaling for sensitive parameters
        if hasattr(self, 'ensemble_parms') and p < len(self.ensemble_parms):
            parm_name = self.ensemble_parms[p].lower()
            if any(x in parm_name for x in ['vcmax', 'jmax', 'kmax']):
                parm_step[p] = base_step * 0.5  # Extra reduction for photosynthesis
            elif any(x in parm_name for x in ['q10', 'froz']):
                parm_step[p] = base_step * 0.3  # Extra reduction for temperature sensitivity
            else:
                parm_step[p] = base_step
        else:
            parm_step[p] = base_step
        #parms[p] = np.random.uniform(parms[p]-parm_step[p],parms[p]+parm_step[p],1)
        #parms[p] = self.pdef[p]
        #parms_sens = np.copy(parms)
        #vary this parameter by one step
        #parms_sens[p] = parms_sens[p]+parm_step[p]
        #post_sens = calc_posterior(parms_sens)
        #use 1D sensitivities to decrease the step sizes accordingly
        #print p, np.absolute(post_def - post_sens)
        #if (np.absolute(post_def - post_sens) > 1.0):
        #    parm_step[p] = parm_step[p]/(np.absolute(post_def - post_sens))
    for i in range(0,nparms):
        mycov[i,i] = parm_step[i]**2

    parm_last = parms
    scalefac = 1.0

    # Debug initial state
    print(f"DEBUG: Starting MCMC with {nevals} evaluations")
    print(f"DEBUG: Initial parameters: {parms}")
    print(f"DEBUG: Parameter bounds - min: {self.ensemble_pmin}, max: {self.ensemble_pmax}")
    
    # Check initial posterior
    initial_post, initial_output = calc_posterior(self, parms, myvars)
    print(f"DEBUG: Initial posterior: {initial_post}")
    if hasattr(self, 'obs'):
        print(f"DEBUG: Available observations: {list(self.obs.keys())}")
        for var in self.obs.keys():
            obs_array = np.array(self.obs[var])
            valid_mask = obs_array != -9999
            valid_obs = np.sum(valid_mask)
            print(f"DEBUG: {var} has {valid_obs} valid observations out of {len(self.obs[var])}")
            
            # Skip detailed debug output if no valid observations
            if valid_obs == 0:
                continue
            
            # Print observation values and uncertainties
            if hasattr(self, 'obs_err') and var in self.obs_err:
                err_array = np.array(self.obs_err[var])
                valid_obs_vals = obs_array[valid_mask]
                valid_err_vals = err_array[valid_mask]
                
                print(f"DEBUG: {var} observation values (valid only):")
                print(f"  Min: {np.min(valid_obs_vals):.4f}, Max: {np.max(valid_obs_vals):.4f}, Mean: {np.mean(valid_obs_vals):.4f}")
                print(f"DEBUG: {var} uncertainty values (valid only):")
                print(f"  Min: {np.min(valid_err_vals):.4f}, Max: {np.max(valid_err_vals):.4f}, Mean: {np.mean(valid_err_vals):.4f}")
                
                # Print first few values for detailed inspection
                n_show = min(5, len(valid_obs_vals))
                print(f"DEBUG: {var} first {n_show} valid obs/uncertainty pairs:")
                for i in range(n_show):
                    print(f"  [{i}] obs: {valid_obs_vals[i]:.4f} ± {valid_err_vals[i]:.4f}")
            else:
                print(f"DEBUG: No uncertainty data found for {var}")
                valid_obs_vals = obs_array[valid_mask]
                print(f"DEBUG: {var} observation values (valid only):")
                print(f"  Min: {np.min(valid_obs_vals):.4f}, Max: {np.max(valid_obs_vals):.4f}, Mean: {np.mean(valid_obs_vals):.4f}")
                
                # Print first few values
                n_show = min(5, len(valid_obs_vals))
                print(f"DEBUG: {var} first {n_show} valid observations:")
                for i in range(n_show):
                    print(f"  [{i}] obs: {valid_obs_vals[i]:.4f}")
    else:
        print("DEBUG: No observations found (self.obs not defined)")

    for i in range(0,nevals):
        #update proposal step size
        if (i > 0 and (i % nburn) == 0 and i < burnsteps*nburn):
            acc_ratio = float(accepted_step) / nburn
            mycov_step = np.cov(chain_prop[0:nparms,accepted_tot- \
                                              accepted_step:accepted_tot])
            mycov_chain = np.cov(chain_burn[0:nparms,int(accepted_tot/4):accepted_tot])
            thisscalefac = 1.0
            #Compute scaling factors for step sizes based on acceptance ratio (conservative)
            if (acc_ratio <= 0.25):  # Target higher acceptance rate
                thisscalefac = max(acc_ratio/0.35, 0.3)  # Less aggressive reduction
            elif (acc_ratio > 0.55):  # Higher upper bound
                thisscalefac = min(acc_ratio/0.35, 1.8)  # Less aggressive increase
            scalefac = scalefac * thisscalefac
            #Calculate covariance matrix of recent samples
            for j in range(0,nparms):
                for k in range(0,nparms):
                    if (acc_ratio > 0.05):
                        mycov[j,k] = mycov_chain[j,k] * scalefac
                            #if (j == k):
                            #mycov[j,k] =
                                #scalefac* max(mycov_chain[j,j] / \
                                       #  mycov_step[j,j], 1) * mycov_step[j,j]
                    else:
                        #if (j == k):
                        mycov[j,k] = thisscalefac * mycov[j,k]
                    #if (j == k):
                    #    print(j, scalefac,mycov[j,j]/(parm_step[j]**2))


            #print('BURNSTEP', i/nburn, acc_ratio, thisscalefac, scalefac)
            mycov_step = np.cov(chain_prop[0:nparms,accepted_tot- \
                                                  accepted_step:accepted_tot])
            #print(np.corrcoef(chain[0:4,i-nburn:i]))
            accepted_step = 0
    
    
        if (i == burnsteps*nburn):
            #Parameter chain plots
            for p in range(0,nparms):
                fig = plt.figure()
                xchain = np.cumsum(np.ones(int(nburn*burnsteps)))
                plt.plot(xchain, chain[p,0:int(nburn*burnsteps)])
                plt.xlabel('Evaluations')
                plt.ylabel(self.ensemble_parms[p])
                if not os.path.exists(UQ_output+'/MCMC_output/plots/chains'):
                    os.makedirs(UQ_output+'/MCMC_output/plots/chains')
                plt.savefig(UQ_output+'/MCMC_output/plots/chains/burnin_chain_'+self.ensemble_parms[p]+'.pdf')
                plt.close(fig) 
    
        #get proposal step
        parms = np.random.multivariate_normal(parm_last, mycov)
   
        #------- run the model and calculate log likelihood -------------------
        thisoutput={}
        post, thisoutput = calc_posterior(self, parms, myvars)
        
        # Debug every 1000 iterations
        if i % 1000 == 0:
            print(f"DEBUG: Iteration {i}, current posterior: {post}, best so far: {post_best}")
            
        #determine whether proposal step is accepted
        if ( (post - post_last < np.log(random.uniform(0,1))) ):
            #if not accepted, go back to previous step
            for j in range(0,nparms):
                parms[j] = parm_last[j]
        else:
            #proposal step is accepted
            post_last = post
            accepted_tot = accepted_tot+1
            accepted_step = accepted_step+1
            chain_prop[0:nparms,accepted_tot] = parms-parm_last
            chain_burn[0:nparms,accepted_tot] = parms
            parm_last = parms
            thisoutput_last = thisoutput.copy()
            #keep track of best solution so far
            if (post > post_best):
                post_best = post
                parms_best = parms.copy()  # Use copy to avoid reference issues
                print(f"DEBUG: New best posterior found at iteration {i}: {post_best}")
                #print(post_best)
                output_best = thisoutput

        #populate the chain matrix
        for j in range(0,nparms):
            chain[j][i] = parms[j]
        chain[nparms][i] = post_last
        for v in myvars:
            if (post > -9000000):
              output[v][:,i] = thisoutput[v][:]
            else:
              output[v][:,i] = thisoutput_last[v][:]
        #if (i % 1000 == 0):
        #    print(' -- '+str(i)+' --\n')

    #print("Computing statistics")
    chain_afterburn = chain[0:nparms,int(nburn*burnsteps):]
    chain_sorted = chain_afterburn
    output_sorted={}
    for v in myvars:
      output_sorted[v] = output[v][0:self.nobs[v],int(nburn*burnsteps):]
      output_sorted[v].sort()

    np.savetxt(UQ_output+'/MCMC_output/MCMC_chain.txt', np.transpose(chain_afterburn))
    #Print out some statistics
    
    # Debug: Check if parms_best is defined
    try:
        print(f"DEBUG: parms_best exists with length {len(parms_best)}")
        print(f"DEBUG: parms_best = {parms_best}")
    except NameError:
        print("ERROR: parms_best is not defined!")
        print("This suggests no MCMC iterations improved upon the initial posterior")
        print("Initializing parms_best with starting parameters...")
        parms_best = np.copy(parms)
        print(f"DEBUG: Initialized parms_best = {parms_best}")
    
    parm_best=open(UQ_output+'/MCMC_output/parms_best.txt','w')
    for p in range(0,len(parms_best)):
      parm_best.write(self.ensemble_parms[p]+' '+str(self.ensemble_pfts[p])+' '+str(parms_best[p])+'\n')
    parm_best.close()
    #np.savetxt(UQ_output+'/MCMC_output/correlation_matrix.txt',np.corrcoef(chain_afterburn))

    #parameter correlation plots (threshold correlations)
    #corr_thresh = 0.8
    #for p1 in range(0,nparms-1):
    #  for p2 in range(p1+1,nparms):
    #    if (abs(parmcorr[p1,p2]) > corr_thresh):
    #      fig = plt.figure()
    #      plt.hexbin(chain_afterburn[p1,:],chain_afterburn[p2,:])
    #      cbar = plt.colorbar()
    #      cbar.set_label('bin count')
    #      plt.xlabel(self.ensemble_parms[p1])
    #      plt.ylabel(self.ensemble_parms[p2])
    #
    #      plt.suptitle('r = '+str(parmcorr[p1,p2]))
    #      if not os.path.exists(UQ_output+'/MCMC_output/plots/corr'):
    #           os.makedirs(UQ_output+'/MCMC_output/plots/corr')
    #      plt.savefig(UQ_output+'/MCMC_output/plots/corr/corr_'+self.ensemble_parms[p1]+'_'+model.parm_names[p2]+'.pdf')
    #      plt.close(fig)
    #Parameter chain plots
    for p in range(0,nparms):
        fig = plt.figure()
        xchain = np.cumsum(np.ones(nevals-int(nburn*burnsteps)))
        plt.plot(xchain, chain_afterburn[p,:])
        plt.xlabel('Evaluations')
        plt.ylabel(self.ensemble_parms[p])
        if not os.path.exists(UQ_output+'/MCMC_output/plots/chains'):
            os.makedirs(UQ_output+'/MCMC_output/plots/chains')
        plt.savefig(UQ_output+'/MCMC_output/plots/chains/chain_'+self.ensemble_parms[p]+'.pdf')
        plt.close(fig)

    chain_sorted.sort()
    parm95=open(UQ_output+'/MCMC_output/parms_95pctconf.txt','w')
    for p in range(0,nparms):
        parm95.write(str(self.ensemble_parms[p])+' '+ \
        str(chain_sorted[p,int(0.025*(nevals-nburn*burnsteps))])+' '+ \
        str(chain_sorted[p,int(0.975*(nevals-nburn*burnsteps))])+'\n')
    parm95.close()
    print("Ratio of accepted steps to total steps:")
    print(float(accepted_tot)/nevals)
    out95=open(UQ_output+'/MCMC_output/outputs_95pctconf.txt','w')
    for v in myvars:
      for p in range(0,self.nobs[v]):
        out95.write(v+' '+str(output_sorted[v][p,int(0.025*(nevals-nburn*burnsteps))])+' '+ \
        str(output_sorted[v][p,int(0.975*(nevals-nburn*burnsteps))])+'\n')
    out95.close()
    #make parameter histogram plots
    for p in range(0,nparms):
        fig = plt.figure()
        n, bins, patches = plt.hist(chain_afterburn[p,:],25)
        plt.xlabel(self.ensemble_parms[p])
        plt.ylabel('Probability Density')
        if not os.path.exists(UQ_output+'/MCMC_output/plots/pdfs'):
            os.makedirs(UQ_output+'/MCMC_output/plots/pdfs')
        plt.savefig(UQ_output+'/MCMC_output/plots/pdfs/'+self.ensemble_parms[p]+'.pdf')
        plt.close(fig)

    #make prediction plots
    for v in myvars:
      fig = plt.figure()
      ax=fig.add_subplot(111)
      x = np.cumsum(np.ones([self.nobs[v]],float))
      obs_plot = np.array(self.obs[v].copy())
      obs_plot[obs_plot < -9000] = np.NaN
      obs_err_plot = np.array(self.obs_err[v].copy())
      obs_err_plot[obs_err_plot < -9000] = np.NaN
      ax.errorbar(x,obs_plot, yerr=obs_err_plot, label='Observations')
      ax.plot(x,output_best[v].flatten(),'r', label = 'Model best')
      ax.plot(x,output_sorted[v][:,int(0.025*(nevals-nburn*burnsteps))].flatten(), \
                 'k--', label='Model 95% CI')
      ax.plot(x,output_sorted[v][:,int(0.975*(nevals-nburn*burnsteps))].flatten(),'k--')
      #if (options.parm_default != ''):
      #  ax.plot(x,default_output[thisob], 'g', label='Default')
      #  #plt.xlabel(model.xlabel)
      #  #plt.ylabel(model.ylabel)
      box = ax.get_position()
      ax.set_position([box.x0,box.y0,box.width*0.8,box.height])
      ax.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize='small')
      if not os.path.exists(UQ_output+'/MCMC_output/plots/predictions'):
        os.makedirs(UQ_output+'/MCMC_output/plots/predictions')
      plt.savefig(UQ_output+'/MCMC_output/plots/predictions/Predictions_'+v+'.pdf')
      plt.close(fig)
    
    # ======================================================================
    # Calculate and save Effective Sample Size (ESS) diagnostics
    # ======================================================================
    print("Calculating Effective Sample Size (ESS) diagnostics...")
    
    # Calculate ESS for each parameter
    ess_results = {}
    tau_results = {}
    n_samples_afterburn = nevals - int(nburn * burnsteps)
    
    for p in range(nparms):
        param_name = self.ensemble_parms[p] if p < len(self.ensemble_parms) else f'param_{p}'
        
        # Get parameter chain after burn-in
        param_chain = chain_afterburn[p, :]
        
        # Calculate autocorrelation time and ESS
        tau = autocorr_time_1d(param_chain, quiet=True)
        ess = len(param_chain) / tau
        
        ess_results[param_name] = ess
        tau_results[param_name] = tau
    
    # Calculate summary statistics
    ess_stats = ess_summary_stats(ess_results)
    
    # Save ESS results to file
    with open(UQ_output + '/MCMC_output/ess_diagnostics.txt', 'w') as f:
        f.write("# Effective Sample Size (ESS) Diagnostics - Custom MCMC\n")
        f.write(f"# Total samples (after burn-in): {n_samples_afterburn}\n")
        f.write(f"# Number of parameters: {nparms}\n")
        f.write(f"# Burn-in samples removed: {int(nburn * burnsteps)}\n")
        f.write(f"# Acceptance rate: {float(accepted_tot)/nevals:.3f}\n\n")
        
        f.write("# Summary Statistics:\n")
        f.write(f"Min ESS: {ess_stats['min']:.2f}\n")
        f.write(f"Max ESS: {ess_stats['max']:.2f}\n")
        f.write(f"Mean ESS: {ess_stats['mean']:.2f}\n")
        f.write(f"Median ESS: {ess_stats['median']:.2f}\n\n")
        
        f.write("# Parameter-specific results:\n")
        f.write("Parameter\tESS\tAutocorr_Time\tESS_per_1000\tESS_ratio\n")
        for param_name in ess_results:
            ess_val = ess_results[param_name]
            tau_val = tau_results[param_name]
            ess_per_1000 = (ess_val / n_samples_afterburn) * 1000
            ess_ratio = ess_val / n_samples_afterburn
            f.write(f"{param_name}\t{ess_val:.2f}\t{tau_val:.2f}\t{ess_per_1000:.1f}\t{ess_ratio:.3f}\n")
    
    # Create ESS diagnostic plots
    if not os.path.exists(UQ_output + '/MCMC_output/plots/diagnostics'):
        os.makedirs(UQ_output + '/MCMC_output/plots/diagnostics')
    
    # ESS bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    param_names = list(ess_results.keys())
    ess_values = list(ess_results.values())
    
    bars = ax.bar(range(len(param_names)), ess_values)
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Effective Sample Size')
    ax.set_title('Effective Sample Size by Parameter (Custom MCMC)')
    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    
    # Add horizontal lines for reference
    ax.axhline(y=n_samples_afterburn, color='red', linestyle='--', 
               label=f'Total samples ({n_samples_afterburn})')
    ax.axhline(y=n_samples_afterburn/2, color='orange', linestyle=':', 
               label='50% efficiency')
    ax.axhline(y=100, color='green', linestyle=':', 
               label='ESS = 100 (minimum recommended)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(UQ_output + '/MCMC_output/plots/diagnostics/ess_by_parameter.pdf')
    plt.close()
    
    # Autocorrelation time plot
    fig, ax = plt.subplots(figsize=(12, 6))
    tau_values = list(tau_results.values())
    
    bars = ax.bar(range(len(param_names)), tau_values)
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Autocorrelation Time')
    ax.set_title('Autocorrelation Time by Parameter (Custom MCMC)')
    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(UQ_output + '/MCMC_output/plots/diagnostics/autocorr_time_by_parameter.pdf')
    plt.close()
    
    # Print ESS summary to console
    print("=" * 60)
    print("EFFECTIVE SAMPLE SIZE (ESS) SUMMARY")
    print("=" * 60)
    print(f"Total samples (after burn-in): {n_samples_afterburn}")
    print(f"Acceptance rate: {float(accepted_tot)/nevals:.1%}")
    print(f"Min ESS: {ess_stats['min']:.1f} ({ess_stats['min']/n_samples_afterburn:.1%} efficiency)")
    print(f"Mean ESS: {ess_stats['mean']:.1f} ({ess_stats['mean']/n_samples_afterburn:.1%} efficiency)")
    print(f"Max ESS: {ess_stats['max']:.1f} ({ess_stats['max']/n_samples_afterburn:.1%} efficiency)")
    print("\nParameter-specific ESS:")
    for param_name in ess_results:
        ess_val = ess_results[param_name]
        efficiency = ess_val / n_samples_afterburn
        print(f"  {param_name:20s}: {ess_val:6.1f} ({efficiency:5.1%})")
    print("=" * 60)
    
    return parms_best


