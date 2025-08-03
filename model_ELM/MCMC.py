import numpy as np
from scipy.stats import norm
from scipy.signal import correlate
from scipy.linalg import cholesky, LinAlgError
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

# ======================================================================
# Parameter Correlation Analysis Functions
# ======================================================================

def calculate_parameter_correlations(chains, param_names=None):
    """
    Calculate correlation matrix for MCMC parameter chains.
    
    Parameters
    ----------
    chains : array_like
        2D array where each row is a parameter chain.
    param_names : list, optional
        Names of parameters.
        
    Returns
    -------
    corr_matrix : ndarray
        Correlation matrix.
    corr_dict : dict
        Dictionary with parameter pair correlations.
    """
    chains = np.atleast_2d(chains)
    n_params = chains.shape[0]
    
    if param_names is None:
        param_names = [f'param_{i}' for i in range(n_params)]
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(chains)
    
    # Create dictionary of correlations
    corr_dict = {}
    for i in range(n_params):
        for j in range(i+1, n_params):
            pair = (param_names[i], param_names[j])
            corr_dict[pair] = corr_matrix[i, j]
    
    return corr_matrix, corr_dict

def identify_problematic_correlations(corr_matrix, param_names=None, threshold=0.7):
    """
    Identify parameter pairs with high correlations that may hurt MCMC efficiency.
    
    Parameters
    ----------
    corr_matrix : array_like
        Parameter correlation matrix.
    param_names : list, optional
        Parameter names.
    threshold : float
        Correlation threshold for flagging problematic pairs.
        
    Returns
    -------
    problematic_pairs : list
        List of tuples (param1, param2, correlation) for high correlations.
    """
    corr_matrix = np.asarray(corr_matrix)
    n_params = corr_matrix.shape[0]
    
    if param_names is None:
        param_names = [f'param_{i}' for i in range(n_params)]
    
    problematic_pairs = []
    
    for i in range(n_params):
        for j in range(i+1, n_params):
            correlation = corr_matrix[i, j]
            if abs(correlation) >= threshold:
                problematic_pairs.append((param_names[i], param_names[j], correlation))
    
    # Sort by absolute correlation (highest first)
    problematic_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    return problematic_pairs

def suggest_correlation_fixes(problematic_pairs, ess_results=None):
    """
    Suggest fixes for highly correlated parameters based on correlation analysis.
    
    Parameters
    ----------
    problematic_pairs : list
        List of (param1, param2, correlation) tuples.
    ess_results : dict, optional
        ESS results for each parameter.
        
    Returns
    -------
    suggestions : list
        List of diagnostic suggestions and potential fixes.
    """
    suggestions = []
    
    if not problematic_pairs:
        suggestions.append("✅ No highly correlated parameter pairs detected.")
        return suggestions
    
    suggestions.append(f"⚠️  Found {len(problematic_pairs)} highly correlated parameter pairs:")
    
    for param1, param2, corr in problematic_pairs:
        suggestions.append(f"   {param1} ↔ {param2}: r = {corr:.3f}")
        
        # Add ESS context if available
        if ess_results:
            ess1 = ess_results.get(param1, 'N/A')
            ess2 = ess_results.get(param2, 'N/A')
            if ess1 != 'N/A' and ess2 != 'N/A':
                suggestions.append(f"     ESS: {param1} = {ess1:.1f}, {param2} = {ess2:.1f}")
    
    suggestions.append("\nRecommended fixes:")
    suggestions.append("1. Parameter transformations:")
    suggestions.append("   - Use log transforms for positive parameters")
    suggestions.append("   - Consider orthogonal reparameterizations")
    suggestions.append("   - Use centered parameterizations")
    
    suggestions.append("2. Sampler improvements:")
    suggestions.append("   - Use blocked Gibbs sampling for correlated groups")
    suggestions.append("   - Implement Hamiltonian Monte Carlo (HMC/NUTS)")
    suggestions.append("   - Adapt proposal covariance matrix")
    
    suggestions.append("3. Prior considerations:")
    suggestions.append("   - Check if correlations arise from tight priors")
    suggestions.append("   - Consider hierarchical models to break correlations")
    
    return suggestions

def adapt_proposal_covariance(chains, scaling_factor=2.4**2, regularization=1e-6):
    """
    Calculate adaptive proposal covariance matrix from MCMC chains.
    
    Parameters
    ----------
    chains : array_like
        2D array where each row is a parameter chain.
    scaling_factor : float
        Scaling factor for optimal acceptance rate (default: 2.4² for multivariate normal).
    regularization : float
        Regularization parameter to ensure positive definiteness.
        
    Returns
    -------
    proposal_cov : ndarray
        Proposal covariance matrix.
    scaling : float
        Applied scaling factor.
    is_valid : bool
        Whether the covariance matrix is valid (positive definite).
    """
    chains = np.atleast_2d(chains)
    n_params, n_samples = chains.shape
    
    # Calculate empirical covariance
    empirical_cov = np.cov(chains)
    
    # Apply scaling for optimal acceptance rate
    scaling = scaling_factor / n_params
    proposal_cov = scaling * empirical_cov
    
    # Add regularization to diagonal for numerical stability
    proposal_cov += regularization * np.eye(n_params)
    
    # Check positive definiteness
    is_valid = True
    try:
        cholesky(proposal_cov)
    except LinAlgError:
        is_valid = False
        # Fallback: use diagonal covariance
        proposal_cov = scaling * np.diag(np.diag(empirical_cov)) + regularization * np.eye(n_params)
    
    return proposal_cov, scaling, is_valid

def calculate_condition_number(cov_matrix):
    """
    Calculate condition number of covariance matrix to assess numerical stability.
    
    Parameters
    ----------
    cov_matrix : array_like
        Covariance matrix.
        
    Returns
    -------
    condition_number : float
        Condition number (ratio of largest to smallest eigenvalue).
    eigenvalues : array
        Eigenvalues of the matrix.
    """
    eigenvalues = np.linalg.eigvals(cov_matrix)
    eigenvalues = eigenvalues[eigenvalues > 0]  # Remove negative/zero eigenvalues
    
    if len(eigenvalues) == 0:
        return np.inf, np.array([])
    
    condition_number = np.max(eigenvalues) / np.min(eigenvalues)
    return condition_number, eigenvalues

def create_correlation_report(chains, param_names=None, output_dir='.', 
                            ess_results=None, corr_threshold=0.7):
    """
    Create comprehensive parameter correlation analysis report.
    
    Parameters
    ----------
    chains : array_like
        2D array where each row is a parameter chain.
    param_names : list, optional
        Parameter names.
    output_dir : str
        Output directory for report and plots.
    ess_results : dict, optional
        ESS results for context.
    corr_threshold : float
        Threshold for flagging high correlations.
        
    Returns
    -------
    report : dict
        Dictionary containing correlation analysis results.
    """
    chains = np.atleast_2d(chains)
    n_params, n_samples = chains.shape
    
    if param_names is None:
        param_names = [f'param_{i}' for i in range(n_params)]
    
    # Calculate correlations
    corr_matrix, corr_dict = calculate_parameter_correlations(chains, param_names)
    
    # Identify problematic correlations
    problematic_pairs = identify_problematic_correlations(
        corr_matrix, param_names, corr_threshold)
    
    # Get suggestions
    suggestions = suggest_correlation_fixes(problematic_pairs, ess_results)
    
    # Calculate proposal covariance
    proposal_cov, scaling, is_valid = adapt_proposal_covariance(chains)
    
    # Calculate condition number
    condition_number, eigenvalues = calculate_condition_number(corr_matrix)
    
    # Create report
    report = {
        'correlation_matrix': corr_matrix,
        'correlation_dict': corr_dict,
        'problematic_pairs': problematic_pairs,
        'suggestions': suggestions,
        'proposal_covariance': proposal_cov,
        'scaling_factor': scaling,
        'covariance_valid': is_valid,
        'condition_number': condition_number,
        'eigenvalues': eigenvalues
    }
    
    # Save report to file
    report_file = os.path.join(output_dir, 'parameter_correlation_report.txt')
    with open(report_file, 'w') as f:
        f.write("PARAMETER CORRELATION ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Number of parameters: {n_params}\n")
        f.write(f"Number of samples: {n_samples}\n")
        f.write(f"Correlation threshold: {corr_threshold}\n\n")
        
        f.write("CORRELATION MATRIX:\n")
        f.write("-" * 30 + "\n")
        
        # Write correlation matrix with parameter names
        f.write("Parameter correlations:\n")
        f.write(f"{'':>15s}")
        for name in param_names:
            f.write(f"{name:>12s}")
        f.write("\n")
        
        for i, name in enumerate(param_names):
            f.write(f"{name:>15s}")
            for j in range(len(param_names)):
                f.write(f"{corr_matrix[i,j]:>12.3f}")
            f.write("\n")
        
        f.write(f"\nMatrix condition number: {condition_number:.2e}\n")
        if condition_number > 1e12:
            f.write("⚠️  WARNING: High condition number indicates numerical instability\n")
        
        f.write("\nEIGENVALUES:\n")
        f.write("-" * 30 + "\n")
        for i, eigval in enumerate(eigenvalues):
            f.write(f"Eigenvalue {i+1}: {eigval:.6f}\n")
        
        f.write("\nPROBLEMATIC CORRELATIONS:\n")
        f.write("-" * 30 + "\n")
        if problematic_pairs:
            for param1, param2, corr in problematic_pairs:
                f.write(f"{param1} ↔ {param2}: r = {corr:.3f}\n")
        else:
            f.write("No correlations above threshold detected.\n")
        
        f.write("\nSUGGESTIONS:\n")
        f.write("-" * 30 + "\n")
        for suggestion in suggestions:
            f.write(f"{suggestion}\n")
        
        f.write("\nPROPOSAL COVARIANCE INFO:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Scaling factor applied: {scaling:.6f}\n")
        f.write(f"Covariance matrix valid: {is_valid}\n")
        if not is_valid:
            f.write("⚠️  WARNING: Proposal covariance was not positive definite\n")
            f.write("   Fallback to diagonal covariance was used\n")
    
    print(f"Parameter correlation report saved to: {report_file}")
    return report

def plot_correlation_matrix(corr_matrix, param_names=None, output_dir='.', 
                           title='Parameter Correlation Matrix'):
    """
    Create correlation matrix heatmap visualization.
    
    Parameters
    ----------
    corr_matrix : array_like
        Correlation matrix.
    param_names : list, optional
        Parameter names for axis labels.
    output_dir : str
        Directory to save plot.
    title : str
        Plot title.
    """
    corr_matrix = np.asarray(corr_matrix)
    n_params = corr_matrix.shape[0]
    
    if param_names is None:
        param_names = [f'param_{i}' for i in range(n_params)]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use diverging colormap centered at 0
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
    
    # Set ticks and labels
    ax.set_xticks(range(n_params))
    ax.set_yticks(range(n_params))
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.set_yticklabels(param_names)
    
    # Add correlation values as text
    for i in range(n_params):
        for j in range(n_params):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                         ha="center", va="center", 
                         color="white" if abs(corr_matrix[i, j]) > 0.5 else "black",
                         fontsize=8)
    
    ax.set_title(title, pad=20)
    
    # Save plot
    os.makedirs(os.path.join(output_dir, 'plots', 'correlations'), exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'correlations', 'correlation_matrix.pdf'))
    plt.close()

def plot_pairwise_correlations(chains, param_names=None, output_dir='.', 
                              max_pairs=10, corr_threshold=0.5):
    """
    Create scatter plots for highly correlated parameter pairs.
    
    Parameters
    ----------
    chains : array_like
        2D array where each row is a parameter chain.
    param_names : list, optional
        Parameter names.
    output_dir : str
        Output directory.
    max_pairs : int
        Maximum number of pairs to plot.
    corr_threshold : float
        Minimum correlation to plot.
    """
    chains = np.atleast_2d(chains)
    n_params = chains.shape[0]
    
    if param_names is None:
        param_names = [f'param_{i}' for i in range(n_params)]
    
    # Calculate correlations and find high correlation pairs
    corr_matrix = np.corrcoef(chains)
    high_corr_pairs = []
    
    for i in range(n_params):
        for j in range(i+1, n_params):
            corr = corr_matrix[i, j]
            if abs(corr) >= corr_threshold:
                high_corr_pairs.append((i, j, corr))
    
    # Sort by absolute correlation
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    high_corr_pairs = high_corr_pairs[:max_pairs]
    
    if not high_corr_pairs:
        print(f"No parameter pairs with |correlation| >= {corr_threshold}")
        return
    
    # Create output directory
    plot_dir = os.path.join(output_dir, 'plots', 'correlations')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Create scatter plots for each high correlation pair
    for idx, (i, j, corr) in enumerate(high_corr_pairs):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x_data = chains[i]
        y_data = chains[j]
        
        # Create scatter plot with some transparency
        ax.scatter(x_data, y_data, alpha=0.5, s=1)
        
        # Add correlation coefficient to title
        ax.set_title(f'{param_names[i]} vs {param_names[j]}\nr = {corr:.3f}')
        ax.set_xlabel(param_names[i])
        ax.set_ylabel(param_names[j])
        
        # Add trend line
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(x_data.min(), x_data.max(), 100)
        ax.plot(x_trend, p(x_trend), 'r--', alpha=0.8, linewidth=2)
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'correlation_{param_names[i]}_{param_names[j]}.pdf'))
        plt.close()
    
    print(f"Created {len(high_corr_pairs)} pairwise correlation plots")

def update_mcmc_covariance(self, chains_after_burn, accept_rate_target=0.23):
    """
    Update MCMC proposal covariance matrix based on recent chains.
    
    Parameters
    ----------
    chains_after_burn : array_like
        Parameter chains after burn-in (nparms x nsamples).
    accept_rate_target : float
        Target acceptance rate for scaling adjustment.
        
    Returns
    -------
    updated_cov : ndarray
        Updated covariance matrix for MCMC proposals.
    scaling_info : dict
        Information about scaling adjustments.
    """
    # Calculate adaptive covariance
    proposal_cov, scaling, is_valid = adapt_proposal_covariance(chains_after_burn)
    
    # Calculate current acceptance rate (this would need to be tracked in MCMC)
    # For now, we'll use a placeholder - in practice this should come from MCMC state
    current_accept_rate = getattr(self, 'last_accept_rate', 0.23)
    
    # Adjust scaling based on acceptance rate
    if current_accept_rate < accept_rate_target * 0.5:
        # Very low acceptance, reduce step size more aggressively
        scale_adjustment = 0.5
    elif current_accept_rate < accept_rate_target * 0.8:
        # Low acceptance, reduce step size
        scale_adjustment = 0.8
    elif current_accept_rate > accept_rate_target * 1.5:
        # High acceptance, increase step size
        scale_adjustment = 1.2
    else:
        # Acceptable range
        scale_adjustment = 1.0
    
    # Apply adjustment
    adjusted_cov = proposal_cov * (scale_adjustment ** 2)
    
    scaling_info = {
        'original_scaling': scaling,
        'scale_adjustment': scale_adjustment,
        'target_accept_rate': accept_rate_target,
        'current_accept_rate': current_accept_rate,
        'covariance_valid': is_valid
    }
    
    return adjusted_cov, scaling_info

def adaptive_metropolis_update(current_cov, chain_samples, accept_rate, 
                              target_accept=0.234, adaptation_rate=0.05,
                              min_samples=100, regularization=1e-6):
    """
    Update proposal covariance using Adaptive Metropolis algorithm.
    
    Based on Haario et al. (2001) "An adaptive Metropolis algorithm"
    
    Parameters
    ----------
    current_cov : ndarray
        Current proposal covariance matrix.
    chain_samples : ndarray
        Recent chain samples (nparms x nsamples).
    accept_rate : float
        Current acceptance rate.
    target_accept : float
        Target acceptance rate (0.234 is optimal for multivariate normal).
    adaptation_rate : float
        Rate of adaptation (smaller = more conservative).
    min_samples : int
        Minimum samples needed before adaptation.
    regularization : float
        Regularization for numerical stability.
        
    Returns
    -------
    new_cov : ndarray
        Updated covariance matrix.
    adaptation_info : dict
        Information about the adaptation step.
    """
    chain_samples = np.atleast_2d(chain_samples)
    n_params, n_samples = chain_samples.shape
    
    adaptation_info = {
        'adapted': False,
        'reason': '',
        'old_scale': 1.0,
        'new_scale': 1.0,
        'empirical_cov_used': False
    }
    
    # Don't adapt if too few samples
    if n_samples < min_samples:
        adaptation_info['reason'] = f'Insufficient samples ({n_samples} < {min_samples})'
        return current_cov.copy(), adaptation_info
    
    # Calculate scaling adjustment based on acceptance rate
    if accept_rate > 0:
        # Robbins-Monro type scaling adjustment
        scale_factor = np.exp(adaptation_rate * (accept_rate - target_accept))
        scale_factor = np.clip(scale_factor, 0.1, 10.0)  # Prevent extreme scaling
    else:
        scale_factor = 0.5  # Reduce if no acceptances
    
    adaptation_info['old_scale'] = 1.0
    adaptation_info['new_scale'] = scale_factor
    
    # Option 1: Simple scaling (for early adaptation)
    if n_samples < 2 * min_samples:
        new_cov = (scale_factor ** 2) * current_cov
        adaptation_info['adapted'] = True
        adaptation_info['reason'] = 'Simple scaling adjustment'
        return new_cov, adaptation_info
    
    # Option 2: Full adaptive metropolis (use empirical covariance)
    try:
        empirical_cov = np.cov(chain_samples)
        
        # Haario et al. (2001) formula: 
        # C_n = s_d * Cov(X_0, ..., X_{n-1}) + s_d * ε * I_d
        # where s_d = (2.4)^2 / d (optimal scaling)
        optimal_scaling = (2.4 ** 2) / n_params
        
        new_cov = optimal_scaling * empirical_cov + regularization * np.eye(n_params)
        
        # Apply acceptance rate adjustment
        new_cov *= (scale_factor ** 2)
        
        # Check positive definiteness
        try:
            cholesky(new_cov)
            adaptation_info['adapted'] = True
            adaptation_info['empirical_cov_used'] = True
            adaptation_info['reason'] = 'Full adaptive metropolis update'
        except LinAlgError:
            # Fallback to diagonal adaptation
            diag_cov = optimal_scaling * np.diag(np.diag(empirical_cov))
            new_cov = (scale_factor ** 2) * diag_cov + regularization * np.eye(n_params)
            adaptation_info['adapted'] = True
            adaptation_info['reason'] = 'Diagonal adaptation (covariance not PD)'
    
    except Exception as e:
        # Fallback to simple scaling
        new_cov = (scale_factor ** 2) * current_cov
        adaptation_info['adapted'] = True
        adaptation_info['reason'] = f'Fallback scaling due to error: {str(e)}'
    
    return new_cov, adaptation_info

def detect_correlation_issues(chain_samples, param_names=None, 
                            correlation_threshold=0.8, condition_threshold=1e10):
    """
    Detect correlation-related issues that may hurt MCMC efficiency.
    
    Parameters
    ----------
    chain_samples : ndarray
        Chain samples (nparms x nsamples).
    param_names : list, optional
        Parameter names.
    correlation_threshold : float
        Threshold for flagging high correlations.
    condition_threshold : float
        Threshold for flagging ill-conditioned matrices.
        
    Returns
    -------
    issues : dict
        Dictionary describing detected issues.
    """
    chain_samples = np.atleast_2d(chain_samples)
    n_params, n_samples = chain_samples.shape
    
    if param_names is None:
        param_names = [f'param_{i}' for i in range(n_params)]
    
    issues = {
        'high_correlations': [],
        'condition_number': 1.0,
        'ill_conditioned': False,
        'recommendations': []
    }
    
    if n_samples < n_params + 10:
        issues['recommendations'].append("Too few samples for reliable correlation analysis")
        return issues
    
    # Calculate correlation matrix
    try:
        corr_matrix = np.corrcoef(chain_samples)
        condition_number = calculate_condition_number(corr_matrix)[0]
        
        issues['condition_number'] = condition_number
        issues['ill_conditioned'] = condition_number > condition_threshold
        
        # Find high correlations
        for i in range(n_params):
            for j in range(i+1, n_params):
                corr = corr_matrix[i, j]
                if abs(corr) >= correlation_threshold:
                    issues['high_correlations'].append({
                        'param1': param_names[i],
                        'param2': param_names[j],
                        'correlation': corr
                    })
        
        # Generate recommendations
        if issues['high_correlations']:
            issues['recommendations'].append(
                f"Found {len(issues['high_correlations'])} highly correlated pairs (|r| ≥ {correlation_threshold})")
            issues['recommendations'].append("Consider parameter transformations or reparameterization")
        
        if issues['ill_conditioned']:
            issues['recommendations'].append(
                f"Correlation matrix is ill-conditioned (κ = {condition_number:.2e})")
            issues['recommendations'].append("Add regularization or use diagonal proposals")
    
    except Exception as e:
        issues['recommendations'].append(f"Error in correlation analysis: {str(e)}")
    
    return issues

def enhanced_proposal_step(parm_last, current_cov, method='multivariate_normal'):
    """
    Generate proposal step with enhanced methods.
    
    Parameters
    ----------
    parm_last : ndarray
        Current parameter values.
    current_cov : ndarray
        Current proposal covariance matrix.
    method : str
        Proposal method ('multivariate_normal', 'cholesky', 'eigendecomp').
        
    Returns
    -------
    parms_new : ndarray
        Proposed parameter values.
    success : bool
        Whether proposal generation was successful.
    """
    parm_last = np.atleast_1d(parm_last)
    n_params = len(parm_last)
    
    try:
        if method == 'multivariate_normal':
            # Standard numpy multivariate normal
            parms_new = np.random.multivariate_normal(parm_last, current_cov)
            return parms_new, True
        
        elif method == 'cholesky':
            # Cholesky decomposition method (more stable)
            L = cholesky(current_cov, lower=True)
            z = np.random.standard_normal(n_params)
            parms_new = parm_last + L @ z
            return parms_new, True
        
        elif method == 'eigendecomp':
            # Eigendecomposition method (handles near-singular matrices better)
            eigenvals, eigenvecs = np.linalg.eigh(current_cov)
            eigenvals = np.maximum(eigenvals, 1e-12)  # Regularize small eigenvalues
            sqrt_eigenvals = np.sqrt(eigenvals)
            z = np.random.standard_normal(n_params)
            parms_new = parm_last + eigenvecs @ (sqrt_eigenvals * z)
            return parms_new, True
        
        else:
            raise ValueError(f"Unknown proposal method: {method}")
    
    except Exception:
        # Fallback to independent proposals
        diagonal_std = np.sqrt(np.diag(current_cov))
        parms_new = parm_last + diagonal_std * np.random.standard_normal(n_params)
        return parms_new, False

# ======================================================================
# Sensitivity-Informed MCMC Functions
# ======================================================================

def extract_sensitivity_info(self, myvars, aggregation_method='mean', 
                            importance_threshold=0.05):
    """
    Extract and process sensitivity analysis results for MCMC tuning.
    
    Parameters
    ----------
    myvars : list
        Variables for which sensitivity was analyzed.
    aggregation_method : str
        How to aggregate sensitivity across variables ('mean', 'max', 'weighted').
    importance_threshold : float
        Threshold below which parameters are considered low-sensitivity.
        
    Returns
    -------
    sensitivity_info : dict
        Dictionary containing processed sensitivity information.
    """
    if not hasattr(self, 'sens_main') or not hasattr(self, 'sens_tot'):
        return None
    
    sensitivity_info = {
        'param_importance': {},
        'interaction_strength': {},
        'aggregated_main': np.zeros(self.nparms_ensemble),
        'aggregated_total': np.zeros(self.nparms_ensemble),
        'high_sensitivity_params': [],
        'low_sensitivity_params': [],
        'interaction_pairs': [],
        'aggregation_method': aggregation_method
    }
    
    # Calculate aggregated sensitivity indices
    for p in range(self.nparms_ensemble):
        param_name = self.ensemble_parms[p] if p < len(self.ensemble_parms) else f'param_{p}'
        
        main_values = []
        total_values = []
        
        for v in myvars:
            if v in self.sens_main and v in self.sens_tot:
                # Average across time dimensions if multiple outputs exist
                main_sens = np.mean(self.sens_main[v][p, :])
                total_sens = np.mean(self.sens_tot[v][p, :])
                main_values.append(main_sens)
                total_values.append(total_sens)
        
        if main_values:
            if aggregation_method == 'mean':
                sensitivity_info['aggregated_main'][p] = np.mean(main_values)
                sensitivity_info['aggregated_total'][p] = np.mean(total_values)
            elif aggregation_method == 'max':
                sensitivity_info['aggregated_main'][p] = np.max(main_values)
                sensitivity_info['aggregated_total'][p] = np.max(total_values)
            elif aggregation_method == 'weighted':
                # Weight by variance of observations if available
                weights = [1.0] * len(main_values)  # Default equal weights
                if hasattr(self, 'obs'):
                    weights = []
                    for v in myvars:
                        if v in self.obs:
                            obs_var = np.var([x for x in self.obs[v] if x > -9000])
                            weights.append(obs_var if obs_var > 0 else 1.0)
                        else:
                            weights.append(1.0)
                
                weights = np.array(weights)
                weights = weights / np.sum(weights)  # Normalize
                
                sensitivity_info['aggregated_main'][p] = np.average(main_values, weights=weights)
                sensitivity_info['aggregated_total'][p] = np.average(total_values, weights=weights)
        
        # Store parameter-specific information
        sensitivity_info['param_importance'][param_name] = {
            'main_sensitivity': sensitivity_info['aggregated_main'][p],
            'total_sensitivity': sensitivity_info['aggregated_total'][p],
            'interaction_effect': sensitivity_info['aggregated_total'][p] - sensitivity_info['aggregated_main'][p]
        }
        
        # Classify parameters by importance
        if sensitivity_info['aggregated_total'][p] >= importance_threshold:
            sensitivity_info['high_sensitivity_params'].append(param_name)
        else:
            sensitivity_info['low_sensitivity_params'].append(param_name)
    
    # Identify strong parameter interactions
    if hasattr(self, 'sens_2nd'):
        for v in myvars:
            if v in self.sens_2nd:
                for i in range(self.nparms_ensemble):
                    for j in range(i+1, self.nparms_ensemble):
                        # Average 2nd order sensitivity across time dimensions
                        interaction_strength = np.mean(self.sens_2nd[v][i, j, :])
                        if interaction_strength > importance_threshold * 0.5:  # Lower threshold for interactions
                            param_i = self.ensemble_parms[i] if i < len(self.ensemble_parms) else f'param_{i}'
                            param_j = self.ensemble_parms[j] if j < len(self.ensemble_parms) else f'param_{j}'
                            sensitivity_info['interaction_pairs'].append((param_i, param_j, interaction_strength))
        
        # Sort interaction pairs by strength
        sensitivity_info['interaction_pairs'].sort(key=lambda x: x[2], reverse=True)
    
    return sensitivity_info

def create_sensitivity_based_covariance(self, sensitivity_info, base_covariance=None, 
                                       scaling_strategy='inverse_sqrt',
                                       min_scale=0.1, max_scale=10.0):
    """
    Create covariance matrix informed by sensitivity analysis.
    
    Parameters
    ----------
    sensitivity_info : dict
        Sensitivity information from extract_sensitivity_info().
    base_covariance : ndarray, optional
        Base covariance matrix. If None, uses current MCMC covariance.
    scaling_strategy : str
        How to scale based on sensitivity ('inverse', 'inverse_sqrt', 'proportional').
    min_scale : float
        Minimum scaling factor to prevent overly small step sizes.
    max_scale : float
        Maximum scaling factor to prevent overly large step sizes.
        
    Returns
    -------
    sensitivity_cov : ndarray
        Sensitivity-informed covariance matrix.
    scaling_factors : ndarray
        Applied scaling factors for each parameter.
    """
    if sensitivity_info is None:
        return base_covariance, np.ones(self.nparms_ensemble)
    
    n_params = self.nparms_ensemble
    
    # Get base covariance matrix
    if base_covariance is None:
        base_covariance = np.eye(n_params) * 0.01  # Default small diagonal matrix
    
    # Calculate scaling factors based on sensitivity
    total_sens = sensitivity_info['aggregated_total']
    max_sens = np.max(total_sens)
    
    if max_sens == 0:  # No sensitivity information
        return base_covariance, np.ones(n_params)
    
    # Normalize sensitivities
    normalized_sens = total_sens / max_sens
    
    # Apply scaling strategy
    if scaling_strategy == 'inverse':
        # High sensitivity → smaller steps (inverse relationship)
        scaling_factors = 1.0 / (normalized_sens + 0.1)  # Add small constant to avoid division by zero
    elif scaling_strategy == 'inverse_sqrt':
        # High sensitivity → smaller steps (square root dampening)
        scaling_factors = 1.0 / np.sqrt(normalized_sens + 0.1)
    elif scaling_strategy == 'proportional':
        # High sensitivity → larger steps (proportional relationship)
        scaling_factors = normalized_sens + 0.1
    else:
        raise ValueError(f"Unknown scaling strategy: {scaling_strategy}")
    
    # Apply bounds to scaling factors
    scaling_factors = np.clip(scaling_factors, min_scale, max_scale)
    
    # Create sensitivity-informed covariance matrix
    # Scale the diagonal elements based on sensitivity
    sensitivity_cov = base_covariance.copy()
    for i in range(n_params):
        for j in range(n_params):
            if i == j:
                # Scale diagonal elements
                sensitivity_cov[i, j] *= scaling_factors[i] ** 2
            else:
                # Scale off-diagonal elements by geometric mean
                sensitivity_cov[i, j] *= scaling_factors[i] * scaling_factors[j]
    
    return sensitivity_cov, scaling_factors

def get_sensitivity_based_recommendations(self, sensitivity_info):
    """
    Generate MCMC tuning recommendations based on sensitivity analysis.
    
    Parameters
    ----------
    sensitivity_info : dict
        Sensitivity information from extract_sensitivity_info().
        
    Returns
    -------
    recommendations : list
        List of recommendations for improving MCMC efficiency.
    """
    recommendations = []
    
    if sensitivity_info is None:
        recommendations.append("⚠️  No sensitivity analysis results available")
        recommendations.append("   Consider running GSA before MCMC for optimal tuning")
        return recommendations
    
    high_sens = sensitivity_info['high_sensitivity_params']
    low_sens = sensitivity_info['low_sensitivity_params']
    interactions = sensitivity_info['interaction_pairs']
    
    recommendations.append("📊 SENSITIVITY-INFORMED MCMC RECOMMENDATIONS:")
    
    # High sensitivity parameters
    if high_sens:
        recommendations.append(f"\n🎯 High-sensitivity parameters ({len(high_sens)}): {', '.join(high_sens[:5])}")
        if len(high_sens) > 5:
            recommendations.append(f"    ... and {len(high_sens) - 5} more")
        recommendations.append("   → Use smaller proposal steps for these parameters")
        recommendations.append("   → Consider log-transforms for positive parameters")
        recommendations.append("   → Monitor convergence carefully")
    
    # Low sensitivity parameters
    if low_sens:
        recommendations.append(f"\n🔽 Low-sensitivity parameters ({len(low_sens)}): {', '.join(low_sens[:5])}")
        if len(low_sens) > 5:
            recommendations.append(f"    ... and {len(low_sens) - 5} more")
        recommendations.append("   → Can use larger proposal steps")
        recommendations.append("   → Consider fixing at prior means to reduce dimensionality")
        recommendations.append("   → Focus computational effort on high-sensitivity parameters")
    
    # Parameter interactions
    if interactions:
        strong_interactions = [pair for pair in interactions if pair[2] > 0.1]
        if strong_interactions:
            recommendations.append(f"\n🔗 Strong parameter interactions ({len(strong_interactions)}):")
            for i, (p1, p2, strength) in enumerate(strong_interactions[:3]):
                recommendations.append(f"   {p1} ↔ {p2}: {strength:.3f}")
            if len(strong_interactions) > 3:
                recommendations.append(f"   ... and {len(strong_interactions) - 3} more")
            recommendations.append("   → Use correlated proposals for these parameter pairs")
            recommendations.append("   → Consider block sampling or Gibbs updates")
    
    # Overall strategy recommendations
    total_high_sens = np.sum([sensitivity_info['param_importance'][p]['total_sensitivity'] 
                            for p in high_sens])
    
    if total_high_sens > 0.8:
        recommendations.append("\n🚨 HIGH-DIMENSIONAL SENSITIVITY DETECTED:")
        recommendations.append("   → Consider dimension reduction techniques")
        recommendations.append("   → Use efficient samplers (HMC/NUTS)")
        recommendations.append("   → Implement hierarchical parameterization")
    elif len(high_sens) < 0.3 * len(sensitivity_info['param_importance']):
        recommendations.append("\n✅ LOW-DIMENSIONAL EFFECTIVE PARAMETER SPACE:")
        recommendations.append("   → Focus MCMC effort on high-sensitivity parameters")
        recommendations.append("   → Consider sensitivity-based blocking")
    
    return recommendations

def apply_sensitivity_informed_scaling(self, current_cov, sensitivity_info, 
                                     adaptation_factor=0.1):
    """
    Apply sensitivity-informed scaling to current covariance matrix.
    
    Parameters
    ----------
    current_cov : ndarray
        Current MCMC covariance matrix.
    sensitivity_info : dict
        Sensitivity information.
    adaptation_factor : float
        How much to blend sensitivity scaling with current covariance (0=none, 1=full).
        
    Returns
    -------
    updated_cov : ndarray
        Updated covariance matrix.
    scaling_info : dict
        Information about applied scaling.
    """
    if sensitivity_info is None:
        return current_cov, {'applied': False, 'reason': 'No sensitivity info'}
    
    # Create sensitivity-based covariance
    sens_cov, scaling_factors = create_sensitivity_based_covariance(
        self, sensitivity_info, current_cov)
    
    # Blend with current covariance
    updated_cov = (1 - adaptation_factor) * current_cov + adaptation_factor * sens_cov
    
    scaling_info = {
        'applied': True,
        'adaptation_factor': adaptation_factor,
        'scaling_factors': scaling_factors,
        'high_sens_params': sensitivity_info['high_sensitivity_params'],
        'low_sens_params': sensitivity_info['low_sensitivity_params'],
        'n_interactions': len(sensitivity_info['interaction_pairs'])
    }
    
    return updated_cov, scaling_info

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
         default_output=None, sampler='custom', enable_adaptive=True, **kwargs):
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
        - 'custom_adaptive': Use custom implementation with enhanced adaptive proposals
        - 'pymc3': Use PyMC3 with NUTS sampler
        - 'pymc3_metropolis': Use PyMC3 with Metropolis sampler
        - 'pymc3_advi': Use PyMC3 with ADVI variational inference
    enable_adaptive : bool
        Enable adaptive proposal updates based on correlations (default: True).
        Only applies to 'custom' sampler.
    **kwargs : dict
        Additional arguments passed to sampler (e.g., tune, target_accept)

    Returns
    -------
    parms_best : array-like
        Best parameter values found during MCMC sampling.
    trace : optional
        For PyMC3 samplers, also returns the trace object.
    """
    
    # Route to appropriate implementation
    if sampler in ['custom', 'custom_adaptive']:
        # Set adaptive flag if using custom_adaptive or enable_adaptive is True
        if sampler == 'custom_adaptive':
            enable_adaptive = True
        
        # Store adaptive setting for MCMC function to access
        self._enable_adaptive_mcmc = enable_adaptive
        
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
        raise ValueError(f"Unknown sampler: {sampler}. Choose from 'custom', 'custom_adaptive', 'pymc3', 'pymc3_metropolis', 'pymc3_advi'")

def MCMC(self, parms, myvars, nevals, mcmc_type='uniform', nburn=1000, burnsteps=10, default_output=None):
    """
    Enhanced custom Metropolis-Hastings MCMC implementation with adaptive proposals.
    
    Features:
    - Adaptive Metropolis algorithm (Haario et al. 2001)
    - Correlation-based proposal method selection
    - Enhanced numerical stability for ill-conditioned covariance matrices
    - Real-time adaptation monitoring and diagnostics
    """
    
    # Check if adaptive MCMC is enabled
    enable_adaptive = getattr(self, '_enable_adaptive_mcmc', True)
    
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

    # Initialize adaptive MCMC tracking variables (only if enabled)
    if enable_adaptive:
        print(f"MCMC: Adaptive proposals ENABLED")
        adaptation_interval = nburn  # Adapt every nburn steps during burn-in
        adaptation_history = []
        last_adaptation_step = 0
        proposal_method = 'multivariate_normal'  # Start with standard method
        
        # Check for sensitivity analysis results
        sensitivity_info = extract_sensitivity_info(self, myvars)
        if sensitivity_info is not None:
            print(f"MCMC: Using sensitivity analysis to inform proposals")
            print(f"      High-sensitivity parameters: {len(sensitivity_info['high_sensitivity_params'])}")
            print(f"      Low-sensitivity parameters: {len(sensitivity_info['low_sensitivity_params'])}")
            print(f"      Parameter interactions detected: {len(sensitivity_info['interaction_pairs'])}")
            
            # Apply initial sensitivity-based scaling to covariance matrix
            mycov, initial_scaling_info = apply_sensitivity_informed_scaling(
                self, mycov, sensitivity_info, adaptation_factor=0.3)
            
            print(f"      Applied sensitivity-based initial scaling")
        else:
            print(f"MCMC: No sensitivity analysis results found")
            print(f"      Consider running GSA first for optimal MCMC tuning")
            sensitivity_info = None
    else:
        print(f"MCMC: Using standard (non-adaptive) proposals")
        adaptation_history = []
        proposal_method = 'multivariate_normal'
        sensitivity_info = None
    
    for i in range(0,nevals):
        #update proposal step size using enhanced adaptive methods
        if enable_adaptive and (i > 0 and (i % adaptation_interval) == 0 and i < burnsteps*nburn):
            acc_ratio = float(accepted_step) / adaptation_interval
            
            # Get recent chain samples for adaptation
            recent_samples = chain_burn[0:nparms, max(0, accepted_tot-adaptation_interval):accepted_tot]
            
            # Apply enhanced adaptive metropolis update
            if recent_samples.shape[1] > nparms:  # Need enough samples
                new_cov, adaptation_info = adaptive_metropolis_update(
                    current_cov=mycov,
                    chain_samples=recent_samples, 
                    accept_rate=acc_ratio,
                    target_accept=0.234,  # Optimal for multivariate normal
                    adaptation_rate=0.05,
                    min_samples=max(50, nparms*2)
                )
                
                # Check for correlation issues
                if recent_samples.shape[1] > nparms * 3:
                    param_names = [self.ensemble_parms[p] if p < len(self.ensemble_parms) 
                                 else f'param_{p}' for p in range(nparms)]
                    issues = detect_correlation_issues(recent_samples, param_names)
                    
                    # Adjust proposal method based on detected issues
                    if issues['ill_conditioned']:
                        proposal_method = 'eigendecomp'  # More robust for ill-conditioned matrices
                        print(f"  Iteration {i}: Switching to eigendecomp method due to ill-conditioning")
                    elif len(issues['high_correlations']) > 0:
                        proposal_method = 'cholesky'  # More stable for correlated parameters
                        if i <= 2 * nburn:  # Only print during early burn-in
                            print(f"  Iteration {i}: Using Cholesky method for {len(issues['high_correlations'])} correlated pairs")
                    else:
                        proposal_method = 'multivariate_normal'
                
                # Update covariance matrix
                mycov = new_cov
                
                # Apply sensitivity-informed scaling if available
                if sensitivity_info is not None and i < burnsteps * nburn * 0.7:  # Only during early burn-in
                    mycov, sens_scaling_info = apply_sensitivity_informed_scaling(
                        self, mycov, sensitivity_info, adaptation_factor=0.1)  # Gentle blending
                    
                    if sens_scaling_info['applied'] and i <= 2 * nburn:  # Print occasionally
                        n_high = len(sens_scaling_info['high_sens_params'])
                        n_low = len(sens_scaling_info['low_sens_params'])
                        print(f"      Applied sensitivity scaling: {n_high} high-sens, {n_low} low-sens params")
                
                # Store adaptation history
                adaptation_history.append({
                    'iteration': i,
                    'accept_rate': acc_ratio,
                    'adaptation_info': adaptation_info,
                    'correlation_issues': len(issues.get('high_correlations', [])) if 'issues' in locals() else 0,
                    'condition_number': issues.get('condition_number', 1.0) if 'issues' in locals() else 1.0
                })
                
                # Print adaptation info during burn-in
                if i <= burnsteps * nburn * 0.5:  # First half of burn-in
                    print(f"  MCMC Adaptation at step {i}: accept_rate={acc_ratio:.3f}, "
                          f"method={adaptation_info['reason']}, "
                          f"scale_factor={adaptation_info['new_scale']:.3f}")
            
            else:
                # Fallback to simple scaling if not enough samples
                thisscalefac = 1.0
                if (acc_ratio <= 0.25):
                    thisscalefac = max(acc_ratio/0.35, 0.3)
                elif (acc_ratio > 0.55):
                    thisscalefac = min(acc_ratio/0.35, 1.8)
                
                mycov = (thisscalefac ** 2) * mycov
                
                adaptation_history.append({
                    'iteration': i,
                    'accept_rate': acc_ratio,
                    'adaptation_info': {'reason': 'Simple scaling (insufficient samples)', 'new_scale': thisscalefac},
                    'correlation_issues': 0,
                    'condition_number': 1.0
                })
            
            accepted_step = 0
        
        elif not enable_adaptive and (i > 0 and (i % nburn) == 0 and i < burnsteps*nburn):
            # Original non-adaptive scaling (for backward compatibility)
            acc_ratio = float(accepted_step) / nburn
            mycov_step = np.cov(chain_prop[0:nparms,accepted_tot-accepted_step:accepted_tot])
            mycov_chain = np.cov(chain_burn[0:nparms,int(accepted_tot/4):accepted_tot])
            thisscalefac = 1.0
            
            # Compute scaling factors based on acceptance ratio
            if (acc_ratio <= 0.25):
                thisscalefac = max(acc_ratio/0.35, 0.3)
            elif (acc_ratio > 0.55):
                thisscalefac = min(acc_ratio/0.35, 1.8)
            scalefac = scalefac * thisscalefac
            
            # Calculate covariance matrix of recent samples
            for j in range(0,nparms):
                for k in range(0,nparms):
                    if (acc_ratio > 0.05):
                        mycov[j,k] = mycov_chain[j,k] * scalefac
                    else:
                        mycov[j,k] = thisscalefac * mycov[j,k]
            
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
    
        #get proposal step using enhanced method (if adaptive) or standard method
        if enable_adaptive:
            parms, proposal_success = enhanced_proposal_step(parm_last, mycov, method=proposal_method)
            
            # Fallback to diagonal proposals if enhanced method fails
            if not proposal_success and proposal_method != 'multivariate_normal':
                parms, _ = enhanced_proposal_step(parm_last, mycov, method='multivariate_normal')
                if i < burnsteps * nburn and i % (nburn * 2) == 0:  # Print occasionally during burn-in
                    print(f"  Iteration {i}: Fallback to standard multivariate normal proposal")
        else:
            # Standard proposal for non-adaptive mode
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
    
    # ======================================================================
    # Adaptive MCMC Summary and Analysis
    # ======================================================================
    print("Analyzing adaptive MCMC performance...")
    
    # Save adaptation history
    if adaptation_history:
        with open(UQ_output + '/MCMC_output/adaptation_history.txt', 'w') as f:
            f.write("# Adaptive MCMC History\n")
            f.write("Iteration\tAccept_Rate\tAdaptation_Method\tScale_Factor\tCorr_Issues\tCondition_Number\n")
            
            for entry in adaptation_history:
                iteration = entry['iteration']
                accept_rate = entry['accept_rate']
                method = entry['adaptation_info']['reason']
                scale_factor = entry['adaptation_info']['new_scale']
                corr_issues = entry['correlation_issues']
                condition_num = entry['condition_number']
                
                f.write(f"{iteration}\t{accept_rate:.3f}\t{method}\t{scale_factor:.3f}\t{corr_issues}\t{condition_num:.2e}\n")
        
        # Print adaptation summary
        print(f"\nAdaptive MCMC performed {len(adaptation_history)} adaptations during burn-in")
        
        # Calculate average acceptance rate during burn-in
        burnin_adaptations = [entry for entry in adaptation_history if entry['iteration'] < burnsteps * nburn]
        if burnin_adaptations:
            avg_accept_rate = np.mean([entry['accept_rate'] for entry in burnin_adaptations])
            final_accept_rate = burnin_adaptations[-1]['accept_rate'] if burnin_adaptations else 0
            
            print(f"  Average acceptance rate during burn-in: {avg_accept_rate:.1%}")
            print(f"  Final acceptance rate before sampling: {final_accept_rate:.1%}")
            
            # Check for correlation issues during adaptation
            corr_issues_detected = any(entry['correlation_issues'] > 0 for entry in burnin_adaptations)
            if corr_issues_detected:
                max_corr_issues = max(entry['correlation_issues'] for entry in burnin_adaptations)
                print(f"  Maximum correlated parameter pairs detected: {max_corr_issues}")
            
            # Check for numerical issues
            max_condition = max(entry['condition_number'] for entry in burnin_adaptations)
            if max_condition > 1e10:
                print(f"  ⚠️  WARNING: High condition numbers detected (max: {max_condition:.2e})")
                print("    This indicates potential numerical instability in correlation structure")
        
        # Create adaptation plots
        if not os.path.exists(UQ_output + '/MCMC_output/plots/adaptation'):
            os.makedirs(UQ_output + '/MCMC_output/plots/adaptation')
        
        # Plot acceptance rate over burn-in
        iterations = [entry['iteration'] for entry in adaptation_history]
        accept_rates = [entry['accept_rate'] for entry in adaptation_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Acceptance rate plot
        ax1.plot(iterations, accept_rates, 'b-o', markersize=3)
        ax1.axhline(y=0.234, color='red', linestyle='--', label='Optimal (23.4%)')
        ax1.axhline(y=0.20, color='orange', linestyle=':', label='Acceptable range')
        ax1.axhline(y=0.50, color='orange', linestyle=':', label='')
        ax1.set_xlabel('MCMC Iteration')
        ax1.set_ylabel('Acceptance Rate')
        ax1.set_title('Acceptance Rate During Adaptive Burn-in')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # Condition number plot (if available)
        condition_numbers = [entry['condition_number'] for entry in adaptation_history]
        if any(cn > 1 for cn in condition_numbers):
            ax2.semilogy(iterations, condition_numbers, 'g-o', markersize=3)
            ax2.axhline(y=1e12, color='red', linestyle='--', label='Ill-conditioned threshold')
            ax2.set_xlabel('MCMC Iteration') 
            ax2.set_ylabel('Condition Number')
            ax2.set_title('Correlation Matrix Condition Number During Adaptation')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        else:
            # Plot scale factors instead
            scale_factors = [entry['adaptation_info']['new_scale'] for entry in adaptation_history]
            ax2.plot(iterations, scale_factors, 'purple', marker='o', markersize=3)
            ax2.axhline(y=1.0, color='red', linestyle='--', label='No scaling')
            ax2.set_xlabel('MCMC Iteration')
            ax2.set_ylabel('Scale Factor')
            ax2.set_title('Proposal Scaling During Adaptation')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(UQ_output + '/MCMC_output/plots/adaptation/adaptation_history.pdf')
        plt.close()
        
        print(f"  Adaptation history saved to: {UQ_output}/MCMC_output/adaptation_history.txt")
        print(f"  Adaptation plots saved to: {UQ_output}/MCMC_output/plots/adaptation/")
    
    # ======================================================================
    # Sensitivity-Informed MCMC Analysis
    # ======================================================================
    if sensitivity_info is not None:
        print("Analyzing sensitivity-informed MCMC performance...")
        
        # Save sensitivity analysis results
        with open(UQ_output + '/MCMC_output/sensitivity_analysis.txt', 'w') as f:
            f.write("# Sensitivity-Informed MCMC Analysis\n")
            f.write(f"# Aggregation method: {sensitivity_info['aggregation_method']}\n")
            f.write(f"# High-sensitivity parameters: {len(sensitivity_info['high_sensitivity_params'])}\n")
            f.write(f"# Low-sensitivity parameters: {len(sensitivity_info['low_sensitivity_params'])}\n")
            f.write(f"# Parameter interactions: {len(sensitivity_info['interaction_pairs'])}\n\n")
            
            f.write("Parameter\tMain_Sensitivity\tTotal_Sensitivity\tInteraction_Effect\tClassification\n")
            for param_name, param_info in sensitivity_info['param_importance'].items():
                main_sens = param_info['main_sensitivity']
                total_sens = param_info['total_sensitivity']
                interaction = param_info['interaction_effect']
                classification = 'High' if param_name in sensitivity_info['high_sensitivity_params'] else 'Low'
                f.write(f"{param_name}\t{main_sens:.4f}\t{total_sens:.4f}\t{interaction:.4f}\t{classification}\n")
            
            if sensitivity_info['interaction_pairs']:
                f.write("\n# Strong Parameter Interactions:\n")
                f.write("Param1\tParam2\tInteraction_Strength\n")
                for param1, param2, strength in sensitivity_info['interaction_pairs'][:10]:  # Top 10
                    f.write(f"{param1}\t{param2}\t{strength:.4f}\n")
        
        # Create sensitivity visualization
        if not os.path.exists(UQ_output + '/MCMC_output/plots/sensitivity'):
            os.makedirs(UQ_output + '/MCMC_output/plots/sensitivity')
        
        # Parameter sensitivity bar plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Main sensitivity plot
        param_names = list(sensitivity_info['param_importance'].keys())
        main_sens_values = [sensitivity_info['param_importance'][p]['main_sensitivity'] for p in param_names]
        total_sens_values = [sensitivity_info['param_importance'][p]['total_sensitivity'] for p in param_names]
        
        x_pos = np.arange(len(param_names))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, main_sens_values, width, label='Main Effect', alpha=0.8)
        bars2 = ax1.bar(x_pos + width/2, total_sens_values, width, label='Total Effect', alpha=0.8)
        
        ax1.set_xlabel('Parameters')
        ax1.set_ylabel('Sensitivity Index')
        ax1.set_title('Parameter Sensitivity Analysis')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(param_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add threshold line
        importance_threshold = 0.05  # Same as used in extraction
        ax1.axhline(y=importance_threshold, color='red', linestyle='--', alpha=0.7, 
                   label=f'Importance threshold ({importance_threshold})')
        
        # Interaction strength plot (if available)
        if sensitivity_info['interaction_pairs']:
            interactions = sensitivity_info['interaction_pairs'][:10]  # Top 10
            interaction_labels = [f"{p1}-{p2}" for p1, p2, _ in interactions]
            interaction_values = [strength for _, _, strength in interactions]
            
            ax2.bar(range(len(interaction_labels)), interaction_values, alpha=0.8, color='orange')
            ax2.set_xlabel('Parameter Pairs')
            ax2.set_ylabel('Interaction Strength')
            ax2.set_title('Strong Parameter Interactions')
            ax2.set_xticks(range(len(interaction_labels)))
            ax2.set_xticklabels(interaction_labels, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No significant\nparameter interactions\ndetected', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14)
            ax2.set_title('Parameter Interactions')
        
        plt.tight_layout()
        plt.savefig(UQ_output + '/MCMC_output/plots/sensitivity/sensitivity_analysis.pdf')
        plt.close()
        
        # Generate and print sensitivity-based recommendations
        recommendations = get_sensitivity_based_recommendations(self, sensitivity_info)
        
        print("\n" + "=" * 60)
        print("SENSITIVITY-INFORMED MCMC RECOMMENDATIONS")
        print("=" * 60)
        for recommendation in recommendations:
            print(recommendation)
        print("=" * 60)
        
        print(f"  Sensitivity analysis saved to: {UQ_output}/MCMC_output/sensitivity_analysis.txt")
        print(f"  Sensitivity plots saved to: {UQ_output}/MCMC_output/plots/sensitivity/")
    
    # ======================================================================
    # Parameter Correlation Analysis
    # ======================================================================
    print("Analyzing parameter correlations...")
    
    # Create correlation report
    corr_report = create_correlation_report(
        chains=chain_afterburn,
        param_names=param_names,
        output_dir=UQ_output + '/MCMC_output',
        ess_results=ess_results,
        corr_threshold=0.7
    )
    
    # Create correlation visualizations
    plot_correlation_matrix(
        corr_matrix=corr_report['correlation_matrix'],
        param_names=param_names,
        output_dir=UQ_output + '/MCMC_output',
        title='Parameter Correlation Matrix (Custom MCMC)'
    )
    
    plot_pairwise_correlations(
        chains=chain_afterburn,
        param_names=param_names,
        output_dir=UQ_output + '/MCMC_output',
        max_pairs=10,
        corr_threshold=0.5
    )
    
    # Print correlation summary to console
    print("\n" + "=" * 60)
    print("PARAMETER CORRELATION ANALYSIS")
    print("=" * 60)
    
    if corr_report['problematic_pairs']:
        print(f"Found {len(corr_report['problematic_pairs'])} highly correlated parameter pairs:")
        for param1, param2, corr in corr_report['problematic_pairs'][:5]:  # Show top 5
            print(f"  {param1} ↔ {param2}: r = {corr:.3f}")
        if len(corr_report['problematic_pairs']) > 5:
            print(f"  ... and {len(corr_report['problematic_pairs']) - 5} more")
    else:
        print("✅ No highly correlated parameter pairs detected (|r| < 0.7)")
    
    print(f"\nCorrelation matrix condition number: {corr_report['condition_number']:.2e}")
    if corr_report['condition_number'] > 1e12:
        print("⚠️  WARNING: High condition number indicates potential numerical issues")
    
    # Suggestions for improvement
    if corr_report['problematic_pairs']:
        print("\n🔧 MCMC TUNING SUGGESTIONS:")
        key_suggestions = [
            "Consider using PyMC3 NUTS sampler for better handling of correlations",
            "Use adaptive proposal covariance based on sample covariance",
            "Check for parameter transformations (e.g., log-transform positive params)",
            "Consider reparameterization to reduce correlations"
        ]
        for i, suggestion in enumerate(key_suggestions[:3], 1):
            print(f"  {i}. {suggestion}")
    
    print("=" * 60)
    
    return parms_best


