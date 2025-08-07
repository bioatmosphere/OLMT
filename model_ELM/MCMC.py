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
        
        # Enhanced Haario et al. (2001) formula with better regularization
        # C_n = s_d * Cov(X_0, ..., X_{n-1}) + s_d * ε * I_d
        # where s_d = (2.4)^2 / d (optimal scaling)
        optimal_scaling = (2.4 ** 2) / n_params
        
        # Improved regularization based on empirical covariance eigenvalues
        eigenvals = np.linalg.eigvals(empirical_cov)
        min_eigenval = np.min(eigenvals)
        adaptive_reg = max(regularization, 0.01 * min_eigenval) if min_eigenval > 0 else regularization
        
        new_cov = optimal_scaling * empirical_cov + adaptive_reg * np.eye(n_params)
        
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
    # Comprehensive validation of GSA data
    if not hasattr(self, 'sens_main') or not hasattr(self, 'sens_tot'):
        return None
    
    # Check if GSA data has any valid variables
    valid_gsa_vars = [v for v in myvars if (v in self.sens_main and v in self.sens_tot)]
    if not valid_gsa_vars:
        print("WARNING: No valid GSA data found for specified variables")
        return None
    
    # Validate GSA data structure
    for v in valid_gsa_vars:
        if (self.sens_main[v].shape[0] != self.nparms_ensemble or 
            self.sens_tot[v].shape[0] != self.nparms_ensemble):
            print(f"WARNING: GSA data dimension mismatch for variable {v}")
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
        
        for v in valid_gsa_vars:
                # Average across time dimensions if multiple outputs exist
                main_sens = np.mean(self.sens_main[v][p, :])
                total_sens = np.mean(self.sens_tot[v][p, :])
                
                # Validate sensitivity values
                if np.isfinite(main_sens) and np.isfinite(total_sens):
                    main_values.append(max(0.0, main_sens))  # Ensure non-negative
                    total_values.append(max(0.0, total_sens))  # Ensure non-negative
        
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
        # Dictionary to track strongest interaction for each parameter pair
        interaction_dict = {}
        
        for v in valid_gsa_vars:
            if v in self.sens_2nd:
                for i in range(self.nparms_ensemble):
                    for j in range(i+1, self.nparms_ensemble):
                        # Average 2nd order sensitivity across time dimensions
                        interaction_strength = np.mean(self.sens_2nd[v][i, j, :])
                        
                        # Validate interaction strength and check threshold
                        if (np.isfinite(interaction_strength) and 
                            interaction_strength > 0 and 
                            interaction_strength > importance_threshold * 0.5):  # Lower threshold for interactions
                            param_i = self.ensemble_parms[i] if i < len(self.ensemble_parms) else f'param_{i}'
                            param_j = self.ensemble_parms[j] if j < len(self.ensemble_parms) else f'param_{j}'
                            
                            # Keep the strongest interaction for each parameter pair
                            pair_key = (param_i, param_j)
                            if pair_key not in interaction_dict or interaction_strength > interaction_dict[pair_key][2]:
                                interaction_dict[pair_key] = (param_i, param_j, interaction_strength)
        
        # Convert to list and sort by strength
        sensitivity_info['interaction_pairs'] = list(interaction_dict.values())
        sensitivity_info['interaction_pairs'].sort(key=lambda x: x[2], reverse=True)
        
        # Store interactions in a different format for easier access
        sensitivity_info['interactions'] = sensitivity_info['interaction_pairs']
    
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
    
    # Validate that base_covariance is a numpy array
    if not isinstance(base_covariance, np.ndarray):
        raise TypeError(f"base_covariance must be a numpy array, got {type(base_covariance)}. "
                       f"This suggests a tuple unpacking issue in the calling function.")
    
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

# ======================================================================
# Advanced MCMC Sampling Techniques
# ======================================================================

def delayed_rejection_step(self, parm_current, mycov, myvars, post_current, 
                          max_stages=2, scale_factors=[1.0, 0.5]):
    """
    Implement delayed rejection for better acceptance rates.
    
    If first proposal is rejected, try again with smaller step size.
    This helps escape local regions and improves acceptance rates.
    
    Parameters
    ----------
    parm_current : ndarray
        Current parameter values.
    mycov : ndarray
        Current covariance matrix.
    myvars : list
        Variables for posterior calculation.
    post_current : float
        Current posterior value.
    max_stages : int
        Maximum number of rejection stages.
    scale_factors : list
        Scaling factors for each stage.
        
    Returns
    -------
    parm_new : ndarray
        New parameter values (current if all rejected).
    post_new : float
        New posterior value.
    accepted : bool
        Whether any proposal was accepted.
    stage_accepted : int
        Which stage was accepted (0 if none).
    """
    for stage in range(max_stages):
        scale = scale_factors[min(stage, len(scale_factors)-1)]
        scaled_cov = (scale ** 2) * mycov
        
        # Generate proposal
        parm_proposal, success = enhanced_proposal_step(
            parm_current, scaled_cov, method='multivariate_normal')
        
        if not success:
            continue
        
        # Calculate posterior
        post_proposal, _ = calc_posterior(self, parm_proposal, myvars)
        
        # Metropolis acceptance
        if post_proposal - post_current >= np.log(random.uniform(0, 1)):
            return parm_proposal, post_proposal, True, stage + 1
    
    # All proposals rejected
    return parm_current, post_current, False, 0

def block_sampling_step(self, parm_current, mycov, myvars, post_current, 
                       correlation_blocks=None, sensitivity_info=None):
    """
    Implement block sampling for highly correlated parameters.
    
    Parameters
    ----------
    parm_current : ndarray
        Current parameter values.
    mycov : ndarray
        Current covariance matrix.
    myvars : list
        Variables for posterior calculation.
    post_current : float
        Current posterior value.
    correlation_blocks : list of lists
        Parameter indices grouped into correlated blocks.
    sensitivity_info : dict
        Sensitivity information for identifying blocks.
        
    Returns
    -------
    parm_new : ndarray
        Updated parameter values.
    post_new : float
        New posterior value.
    n_accepted : int
        Number of blocks accepted.
    block_info : dict
        Information about block updates.
    """
    if correlation_blocks is None:
        correlation_blocks = identify_correlation_blocks(
            mycov, sensitivity_info, max_block_size=5, ensemble_parms=self.ensemble_parms)
    
    parm_new = parm_current.copy()
    post_new = post_current
    n_accepted = 0
    block_info = {'blocks_tried': len(correlation_blocks), 'blocks_accepted': 0}
    
    for block_idx, block in enumerate(correlation_blocks):
        if len(block) == 0:
            continue
        
        # Extract block covariance
        block_cov = mycov[np.ix_(block, block)]
        block_params = parm_new[block]
        
        # Generate block proposal
        try:
            block_proposal = np.random.multivariate_normal(block_params, block_cov)
        except:
            # Fallback to diagonal if covariance issues
            block_proposal = block_params + np.sqrt(np.diag(block_cov)) * np.random.randn(len(block))
        
        # Create full parameter vector with block proposal
        parm_proposal = parm_new.copy()
        parm_proposal[block] = block_proposal
        
        # Calculate posterior
        post_proposal, _ = calc_posterior(self, parm_proposal, myvars)
        
        # Block Metropolis acceptance
        if post_proposal - post_new >= np.log(random.uniform(0, 1)):
            parm_new = parm_proposal
            post_new = post_proposal
            n_accepted += 1
            block_info['blocks_accepted'] += 1
    
    return parm_new, post_new, n_accepted, block_info

def identify_correlation_blocks(mycov, sensitivity_info=None, 
                               correlation_threshold=0.6, max_block_size=5, ensemble_parms=None):
    """
    Identify correlated parameter blocks for block sampling.
    
    Parameters
    ----------
    mycov : ndarray
        Covariance matrix.
    sensitivity_info : dict, optional
        Sensitivity information.
    correlation_threshold : float
        Threshold for considering parameters correlated.
    max_block_size : int
        Maximum size of parameter blocks.
    ensemble_parms : list, optional
        List of parameter names for sensitivity-based blocking.
        
    Returns
    -------
    blocks : list of lists
        Parameter indices grouped into blocks.
    """
    n_params = mycov.shape[0]
    
    # Convert covariance matrix to correlation matrix
    if mycov.ndim == 2:
        # Compute correlation matrix from covariance matrix
        diag_sqrt = np.sqrt(np.diag(mycov))
        corr_matrix = mycov / np.outer(diag_sqrt, diag_sqrt)
        # Handle division by zero
        corr_matrix = np.where(np.isfinite(corr_matrix), corr_matrix, 0.0)
    else:
        corr_matrix = np.eye(n_params)
    
    # Start with individual parameters
    blocks = [[i] for i in range(n_params)]
    used_params = set()
    final_blocks = []
    
    # Group highly correlated parameters
    for i in range(n_params):
        if i in used_params:
            continue
        
        current_block = [i]
        used_params.add(i)
        
        # Find correlated parameters
        for j in range(i+1, n_params):
            if j in used_params:
                continue
            
            if abs(corr_matrix[i, j]) >= correlation_threshold:
                current_block.append(j)
                used_params.add(j)
                
                if len(current_block) >= max_block_size:
                    break
        
        final_blocks.append(current_block)
    
    # Add sensitivity-based blocking if available
    if sensitivity_info and 'interaction_pairs' in sensitivity_info:
        interaction_blocks = []
        for param1, param2, strength in sensitivity_info['interaction_pairs'][:3]:
            # Find parameter indices
            try:
                if ensemble_parms is not None:
                    idx1 = ensemble_parms.index(param1)
                    idx2 = ensemble_parms.index(param2)
                else:
                    continue
                if strength > 0.1:  # Strong interaction
                    interaction_blocks.append([idx1, idx2])
            except ValueError:
                continue
        
        # Merge with existing blocks if not already grouped
        for int_block in interaction_blocks:
            if not any(all(idx in block for idx in int_block) for block in final_blocks):
                final_blocks.append(int_block)
    
    return final_blocks

def parallel_tempering_step(self, parm_current, mycov, myvars, post_current,
                           n_temperatures=4, temp_schedule=None, swap_interval=50):
    """
    Simplified parallel tempering for multimodal exploration.
    
    Note: This is a simplified version. Full parallel tempering would require
    multiple chains running in parallel.
    
    Parameters
    ----------
    parm_current : ndarray
        Current parameter values.
    mycov : ndarray
        Current covariance matrix.
    myvars : list
        Variables for posterior calculation.
    post_current : float
        Current posterior value.
    n_temperatures : int
        Number of temperature levels.
    temp_schedule : list, optional
        Temperature schedule.
    swap_interval : int
        How often to attempt temperature swaps.
        
    Returns
    -------
    parm_new : ndarray
        New parameter values.
    post_new : float
        New posterior value.
    temp_info : dict
        Temperature information.
    """
    if temp_schedule is None:
        temp_schedule = [1.0 + i * 0.5 for i in range(n_temperatures)]
    
    # Simple implementation: occasionally use higher temperature
    use_high_temp = random.random() < 0.1  # 10% chance
    
    if use_high_temp:
        temperature = temp_schedule[1] if len(temp_schedule) > 1 else 2.0
        heated_cov = (temperature ** 2) * mycov
        
        # Generate proposal with higher temperature
        parm_proposal, success = enhanced_proposal_step(
            parm_current, heated_cov, method='multivariate_normal')
        
        if success:
            post_proposal, _ = calc_posterior(self, parm_proposal, myvars)
            
            # Tempered acceptance (multiply by temperature factor)
            tempered_alpha = min(1.0, np.exp((post_proposal - post_current) / temperature))
            
            if random.random() < tempered_alpha:
                return parm_proposal, post_proposal, {'temperature_used': temperature, 'accepted': True}
    
    # Standard temperature step
    parm_proposal, success = enhanced_proposal_step(
        parm_current, mycov, method='multivariate_normal')
    
    if success:
        post_proposal, _ = calc_posterior(self, parm_proposal, myvars)
        
        if post_proposal - post_current >= np.log(random.uniform(0, 1)):
            return parm_proposal, post_proposal, {'temperature_used': 1.0, 'accepted': True}
    
    return parm_current, post_current, {'temperature_used': 1.0, 'accepted': False}

def convergence_diagnostics(self, chain_samples, param_names=None, 
                           window_size=1000, r_hat_threshold=1.1):
    """
    Calculate convergence diagnostics including Gelman-Rubin R-hat.
    
    Parameters
    ----------
    chain_samples : ndarray
        Chain samples (nparms x nsamples).
    param_names : list, optional
        Parameter names.
    window_size : int
        Window size for running diagnostics.
    r_hat_threshold : float
        Threshold for convergence (R-hat < threshold).
        
    Returns
    -------
    diagnostics : dict
        Convergence diagnostic results.
    """
    chain_samples = np.atleast_2d(chain_samples)
    n_params, n_samples = chain_samples.shape
    
    if param_names is None:
        param_names = [f'param_{i}' for i in range(n_params)]
    
    diagnostics = {
        'r_hat': {},
        'ess_bulk': {},
        'ess_tail': {},
        'converged': {},
        'n_samples': n_samples,
        'window_size': window_size
    }
    
    if n_samples < window_size * 2:
        # Not enough samples for reliable diagnostics, return defaults
        diagnostics['rhat_max'] = 1.0
        diagnostics['rhat_mean'] = 1.0
        diagnostics['ess_min'] = 0.0
        diagnostics['ess_mean'] = 0.0
        diagnostics['all_converged'] = False
        diagnostics['fraction_converged'] = 0.0
        return diagnostics
    
    # Split chains in half to simulate multiple chains
    mid_point = n_samples // 2
    chain1 = chain_samples[:, :mid_point]
    chain2 = chain_samples[:, mid_point:]
    
    for i, param_name in enumerate(param_names):
        if i >= n_params:
            break
        
        # Calculate R-hat (Gelman-Rubin statistic)
        chain1_mean = np.mean(chain1[i])
        chain2_mean = np.mean(chain2[i])
        overall_mean = np.mean(chain_samples[i])
        
        # Between-chain variance
        B = mid_point * ((chain1_mean - overall_mean)**2 + (chain2_mean - overall_mean)**2)
        
        # Within-chain variance
        W = (np.var(chain1[i], ddof=1) + np.var(chain2[i], ddof=1)) / 2
        
        # Pooled variance estimate
        var_plus = ((mid_point - 1) * W + B) / mid_point
        
        # R-hat statistic
        if W > 0:
            r_hat = np.sqrt(var_plus / W)
        else:
            r_hat = 1.0
        
        diagnostics['r_hat'][param_name] = r_hat
        diagnostics['converged'][param_name] = r_hat < r_hat_threshold
        
        # Fast ESS estimates (approximate for performance)
        try:
            # Use a faster approximation for ESS to avoid expensive autocorrelation calculation
            # Simple variance-based estimate as a fast approximation
            chain_var = np.var(chain_samples[i])
            chain_mean = np.mean(chain_samples[i])
            
            # Quick autocorrelation approximation using lag-1 correlation
            if len(chain_samples[i]) > 10:
                lag1_corr = np.corrcoef(chain_samples[i][:-1], chain_samples[i][1:])[0,1]
                lag1_corr = max(-0.99, min(0.99, lag1_corr))  # Bound correlation
                # Approximate ESS using geometric series approximation
                ess = n_samples * (1 - lag1_corr) / (1 + lag1_corr) if lag1_corr < 0.99 else n_samples * 0.1
            else:
                ess = n_samples * 0.5  # Conservative default for short chains
                
            ess = max(1.0, min(ess, n_samples))  # Bound ESS between 1 and n_samples
            
        except:
            # Fallback to conservative estimate if approximation fails
            ess = n_samples * 0.3
            
        diagnostics['ess_bulk'][param_name] = ess
        diagnostics['ess_tail'][param_name] = ess * 0.8  # Conservative estimate
    
    # Compute summary statistics
    if diagnostics['r_hat']:
        rhat_values = list(diagnostics['r_hat'].values())
        diagnostics['rhat_max'] = max(rhat_values)
        diagnostics['rhat_mean'] = np.mean(rhat_values)
    else:
        diagnostics['rhat_max'] = 1.0
        diagnostics['rhat_mean'] = 1.0
    
    if diagnostics['ess_bulk']:
        ess_values = list(diagnostics['ess_bulk'].values())
        diagnostics['ess_min'] = min(ess_values)
        diagnostics['ess_mean'] = np.mean(ess_values)
    else:
        diagnostics['ess_min'] = 0.0
        diagnostics['ess_mean'] = 0.0
    
    # Overall convergence assessment
    if diagnostics['converged']:
        converged_values = list(diagnostics['converged'].values())
        diagnostics['all_converged'] = all(converged_values)
        diagnostics['fraction_converged'] = np.mean(converged_values)
    else:
        diagnostics['all_converged'] = False
        diagnostics['fraction_converged'] = 0.0
    
    return diagnostics

def auto_stopping_criterion(self, diagnostics, min_ess=100, min_samples=1000):
    """
    Determine if MCMC should stop based on convergence diagnostics.
    
    Parameters
    ----------
    diagnostics : dict
        Convergence diagnostics.
    min_ess : float
        Minimum required ESS.
    min_samples : int
        Minimum number of samples before stopping.
        
    Returns
    -------
    should_stop : bool
        Whether to stop sampling.
    stop_reason : str
        Reason for stopping decision.
    """
    if diagnostics['n_samples'] < min_samples:
        return False, "Insufficient samples"
    
    # Check individual parameter convergence
    if not diagnostics['converged']:
        return False, "No convergence data available"
    
    converged_params = [name for name, converged in diagnostics['converged'].items() if converged]
    unconverged_params = [name for name, converged in diagnostics['converged'].items() if not converged]
    
    if unconverged_params:
        rhat_info = []
        for param in unconverged_params:
            if param in diagnostics['r_hat']:
                rhat_val = diagnostics['r_hat'][param]
                rhat_info.append(f"{param}={rhat_val:.3f}")
        
        rhat_details = f" (R-hat: {', '.join(rhat_info)})" if rhat_info else ""
        return False, f"{len(unconverged_params)} parameters not converged{rhat_details}"
    
    # Check individual parameter ESS
    if not diagnostics['ess_bulk']:
        return False, "No ESS data available"
    
    insufficient_ess_params = []
    for param, ess_val in diagnostics['ess_bulk'].items():
        if ess_val < min_ess:
            insufficient_ess_params.append(f"{param}={ess_val:.1f}")
    
    if insufficient_ess_params:
        return False, f"Insufficient ESS for {len(insufficient_ess_params)} parameters: {', '.join(insufficient_ess_params)}"
    
    # All individual parameters meet convergence criteria
    min_ess_achieved = min(diagnostics['ess_bulk'].values())
    max_rhat = max(diagnostics['r_hat'].values())
    return True, f"All {len(converged_params)} parameters converged (R-hat≤{max_rhat:.3f}, ESS≥{min_ess_achieved:.1f})"

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
         default_output=None, sampler='custom', enable_adaptive=True,
         enable_delayed_rejection=True, enable_block_sampling=True,
         enable_parallel_tempering=False, enable_auto_convergence=True,
         nchains=3, **kwargs):
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
    enable_delayed_rejection : bool
        Enable delayed rejection for better acceptance rates (default: True).
    enable_block_sampling : bool
        Enable block sampling for correlated parameters (default: True).
    enable_parallel_tempering : bool
        Enable simplified parallel tempering (default: False).
    enable_auto_convergence : bool
        Enable automatic convergence detection (default: True).
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
        
        # Store settings for MCMC function to access
        self._enable_adaptive_mcmc = enable_adaptive
        self._enable_delayed_rejection = enable_delayed_rejection
        self._enable_block_sampling = enable_block_sampling
        self._enable_parallel_tempering = enable_parallel_tempering
        self._enable_auto_convergence = enable_auto_convergence
        
        # Choose between single chain and multi-chain based on nchains parameter
        if nchains > 1:
            return self.MCMC_multi_chain(parms, myvars, nevals, mcmc_type, nburn, burnsteps, default_output, nchains)
        else:
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

def MCMC_multi_chain(self, parms, myvars, nevals, mcmc_type='uniform', nburn=1000, burnsteps=10, default_output=None, nchains=3):
    """
    Multi-chain MCMC wrapper that runs multiple independent chains and combines results.
    
    This function runs multiple MCMC chains in sequence and provides enhanced convergence
    diagnostics using Gelman-Rubin R-hat statistics across chains.
    """
    print(f"🔗 Running {nchains} independent MCMC chains for robust convergence assessment")
    
    # Initialize results storage
    all_chains = []
    all_outputs = {}
    for v in myvars:
        all_outputs[v] = []
    
    chain_acceptance_rates = []
    chain_best_posteriors = []
    chain_best_params = []
    
    # Run each chain independently
    for chain_id in range(nchains):
        print(f"\n{'='*60}")
        print(f"🔗 STARTING CHAIN {chain_id+1}/{nchains}")
        print(f"{'='*60}")
        
        # Generate overdispersed starting point for this chain
        if chain_id == 0:
            chain_start = parms.copy()
        else:
            # Overdispersed random starts within parameter bounds
            parm_range = np.array(self.ensemble_pmax) - np.array(self.ensemble_pmin)
            dispersion_factor = 0.3 + 0.4 * chain_id / max(1, nchains-1)  # 0.3 to 0.7
            random_offset = (np.random.rand(len(parms)) - 0.5) * dispersion_factor * parm_range
            chain_start = np.clip(parms + random_offset, self.ensemble_pmin, self.ensemble_pmax)
        
        print(f"Chain {chain_id+1} starting parameters: {chain_start[:min(3, len(chain_start))]}{'...' if len(chain_start) > 3 else ''}")
        
        # Run single chain
        best_params_chain = self.MCMC(chain_start, myvars, nevals, mcmc_type, nburn, burnsteps, default_output)
        
        # Store chain results (read from saved files)
        chain_file = f'./UQ_output/{self.casename}/MCMC_output/MCMC_chain.txt'
        if os.path.exists(chain_file):
            chain_data = np.loadtxt(chain_file)
            all_chains.append(chain_data.T)  # Transpose to get (nparms, nsamples)
            
            # Calculate acceptance rate for this chain  
            n_accepted = len(chain_data)
            acceptance_rate = n_accepted / nevals if nevals > 0 else 0.0
            chain_acceptance_rates.append(acceptance_rate)
            
            print(f"✅ Chain {chain_id+1} completed - Acceptance rate: {acceptance_rate:.1%}")
        else:
            print(f"❌ Chain {chain_id+1} failed - no output file found")
            
        chain_best_params.append(best_params_chain)
        # Note: would need to extract best posterior from MCMC function return
    
    if len(all_chains) < 2:
        print("❌ Not enough successful chains for multi-chain analysis")
        return chain_best_params[0] if chain_best_params else parms
    
    # Combine chains and perform multi-chain diagnostics
    print(f"\n{'='*60}")
    print(f"🔍 MULTI-CHAIN CONVERGENCE ANALYSIS")
    print(f"{'='*60}")
    
    combined_chains = np.array(all_chains)  # Shape: (nchains, nparms, nsamples)
    multi_chain_diagnostics = self.multi_chain_convergence_diagnostics(combined_chains)
    
    # Report convergence results
    print(f"Multi-chain R-hat statistics:")
    param_names = [self.ensemble_parms[p] if p < len(self.ensemble_parms) else f'param_{p}' 
                   for p in range(combined_chains.shape[1])]
    
    for p, param_name in enumerate(param_names):
        rhat = multi_chain_diagnostics['r_hat'][p]
        status = "✅" if rhat < 1.1 else "⚠️" if rhat < 1.2 else "❌"
        print(f"  {param_name:20s}: R-hat = {rhat:.4f} {status}")
    
    print(f"\nOverall convergence assessment:")
    print(f"  Max R-hat: {multi_chain_diagnostics['max_rhat']:.4f}")
    print(f"  Converged parameters: {multi_chain_diagnostics['n_converged']}/{len(param_names)}")
    
    if multi_chain_diagnostics['max_rhat'] < 1.1:
        print("✅ EXCELLENT: All chains have converged (R-hat < 1.1)")
    elif multi_chain_diagnostics['max_rhat'] < 1.2:
        print("⚠️  ACCEPTABLE: Chains show reasonable convergence (R-hat < 1.2)")
    else:
        print("❌ POOR: Chains have not converged well (R-hat > 1.2)")
        print("   Consider running longer or checking for multimodality")
    
    # Return best parameters (from best chain)
    if chain_best_posteriors:
        best_chain_idx = np.argmax(chain_best_posteriors)
        return chain_best_params[best_chain_idx]
    else:
        # Return mean of best parameters across chains
        return np.mean(chain_best_params, axis=0)

def multi_chain_convergence_diagnostics(self, chains):
    """
    Calculate Gelman-Rubin R-hat convergence diagnostics across multiple chains.
    
    Parameters
    ----------
    chains : ndarray
        Chain samples with shape (nchains, nparms, nsamples).
        
    Returns
    -------
    diagnostics : dict
        Dictionary with R-hat values and convergence assessment.
    """
    nchains, nparms, nsamples = chains.shape
    
    # Calculate R-hat for each parameter
    r_hat_values = []
    
    for p in range(nparms):
        param_chains = chains[:, p, :]  # Shape: (nchains, nsamples)
        
        # Calculate within-chain variance (W)
        within_chain_vars = np.var(param_chains, axis=1, ddof=1)  # Variance for each chain
        W = np.mean(within_chain_vars)
        
        # Calculate between-chain variance (B)
        chain_means = np.mean(param_chains, axis=1)  # Mean for each chain
        overall_mean = np.mean(chain_means)
        B = nsamples * np.var(chain_means, ddof=1)  # Between-chain variance
        
        # Calculate pooled variance estimate
        var_plus = ((nsamples - 1) * W + B) / nsamples
        
        # Calculate R-hat (potential scale reduction factor)
        if W > 0:
            r_hat = np.sqrt(var_plus / W)
        else:
            r_hat = 1.0  # If no within-chain variance, assume convergence
        
        r_hat_values.append(r_hat)
    
    r_hat_values = np.array(r_hat_values)
    
    # Summary statistics
    max_rhat = np.max(r_hat_values)
    mean_rhat = np.mean(r_hat_values)
    converged = r_hat_values < 1.1  # Standard threshold
    n_converged = np.sum(converged)
    
    diagnostics = {
        'r_hat': r_hat_values,
        'max_rhat': max_rhat,
        'mean_rhat': mean_rhat,
        'converged': converged,
        'n_converged': n_converged,
        'convergence_fraction': n_converged / nparms if nparms > 0 else 0.0
    }
    
    return diagnostics

def MCMC(self, parms, myvars, nevals, mcmc_type='uniform', nburn=1000, burnsteps=10, default_output=None):
    """
    Enhanced custom Metropolis-Hastings MCMC implementation with intelligent burn-in.
    
    Features:
    - Intelligent multi-phase burn-in for faster convergence
    - Adaptive Metropolis algorithm (Haario et al. 2001)  
    - Correlation-based proposal method selection
    - Enhanced numerical stability for ill-conditioned covariance matrices
    - Real-time adaptation monitoring and diagnostics
    """
    
    # Check which MCMC enhancements are enabled
    enable_adaptive = getattr(self, '_enable_adaptive_mcmc', True)
    enable_delayed_rejection = getattr(self, '_enable_delayed_rejection', True)
    enable_block_sampling = getattr(self, '_enable_block_sampling', True)
    enable_parallel_tempering = getattr(self, '_enable_parallel_tempering', False)
    enable_auto_convergence = getattr(self, '_enable_auto_convergence', True)
    enable_auto_stopping = enable_auto_convergence  # Same setting, different variable name used in code
    
    UQ_output='./UQ_output/'+self.casename
    print(os.path.abspath(UQ_output))
    
    #Metropolis-Hastings Markov Chain Monte Carlo with adaptive sampling
    # Variables post_best, post_last, parms_best initialized after calculating initial posterior
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
        base_step = 0.05 * (self.ensemble_pmax[p]-self.ensemble_pmin[p])  # 5% for better initial mixing
        
        # Intelligent initial step size based on sensitivity if available
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

    # ======================================================================
    # Optimized Preconditioning Phase: Quick scale estimation for better initial mixing
    # ======================================================================
    print("Running optimized preconditioning phase...")
    precon_samples = min(50, burnsteps * nburn // 20)  # Reduced to 5% of burn-in for efficiency
    precon_chain = np.zeros((nparms, precon_samples))
    precon_accepted = 0
    
    current_params = parms.copy()
    current_post, _ = calc_posterior(self, current_params, myvars)
    
    # Use smaller initial steps for more conservative preconditioning
    precon_cov = mycov * 0.5  # Start with smaller steps
    
    for precon_i in range(precon_samples):
        # Simple random walk with scaled covariance
        proposal = np.random.multivariate_normal(current_params, precon_cov)
        
        # Apply bounds
        proposal = np.clip(proposal, self.ensemble_pmin, self.ensemble_pmax)
        
        prop_post, _ = calc_posterior(self, proposal, myvars)
        
        if prop_post - current_post >= np.log(random.uniform(0, 1)):
            current_params = proposal
            current_post = prop_post
            precon_accepted += 1
        
        precon_chain[:, precon_i] = current_params
        
        # Early exit if we get good acceptance rate quickly
        if precon_i > 20 and precon_accepted / (precon_i + 1) > 0.4:
            precon_samples = precon_i + 1
            precon_chain = precon_chain[:, :precon_samples]
            break
    
    # Update initial covariance based on preconditioning
    if precon_accepted > 5:  # Lower threshold for efficiency
        precon_cov_est = np.cov(precon_chain[:, :precon_samples])
        precon_accept_rate = precon_accepted / precon_samples
        
        # More nuanced scaling based on acceptance rate
        if precon_accept_rate < 0.15:
            scale_factor = 0.3  # More conservative
        elif precon_accept_rate > 0.6:
            scale_factor = 1.8  # More aggressive
        else:
            scale_factor = 1.0
            
        # Conservative blending to avoid overshooting
        mycov = 0.6 * (scale_factor * precon_cov_est) + 0.4 * mycov
        
        print(f"Preconditioning: {precon_accepted}/{precon_samples} accepted ({precon_accept_rate:.1%})")
        print(f"Applied scale factor: {scale_factor:.2f}")
    else:
        print("Preconditioning: insufficient accepted samples, using original covariance")

    parm_last = current_params  # Start from best preconditioning point
    scalefac = 1.0

    # Debug initial state
    print(f"DEBUG: Starting MCMC with {nevals} evaluations")
    print(f"DEBUG: Initial parameters: {parms}")
    print(f"DEBUG: Parameter bounds - min: {self.ensemble_pmin}, max: {self.ensemble_pmax}")
    
    # Check initial posterior and ensure it's finite
    initial_post, initial_output = calc_posterior(self, parms, myvars)
    print(f"DEBUG: Initial posterior: {initial_post}")
    
    # If initial posterior is invalid, try to find a better starting point (more efficiently)
    if initial_post <= -9999999 or not np.isfinite(initial_post):
        print("WARNING: Initial posterior is invalid, quick search for better starting point...")
        best_post = initial_post
        best_parms = parms.copy()
        
        # More efficient search: fewer attempts but smarter selection
        for attempt in range(20):  # Reduced from 50 to 20 attempts
            # Try Latin Hypercube sampling for better coverage
            if attempt < 10:
                # Random point within bounds
                trial_parms = np.random.uniform(self.ensemble_pmin, self.ensemble_pmax)
            else:
                # Try points closer to parameter bounds for better coverage
                alpha = np.random.rand(nparms)
                trial_parms = alpha * np.array(self.ensemble_pmax) + (1 - alpha) * np.array(self.ensemble_pmin)
            
            trial_post, trial_output = calc_posterior(self, trial_parms, myvars)
            
            if trial_post > best_post and np.isfinite(trial_post):
                best_post = trial_post
                best_parms = trial_parms.copy()
                # Early exit if we find a good starting point
                if trial_post > -1000:  # Much better threshold
                    print(f"Found good starting point early at attempt {attempt+1}")
                    break
        
        if best_post > initial_post:
            print(f"Found better starting point: {best_post:.3f} vs {initial_post:.3f}")
            parms = best_parms
            initial_post, initial_output = calc_posterior(self, parms, myvars)
        else:
            print("WARNING: Could not find good starting point, proceeding with original")
    
    # Initialize best parameters and posterior
    post_best = initial_post
    post_last = initial_post  # Initialize post_last with initial posterior
    parms_best = parms.copy()
    output_best = initial_output.copy() if initial_output else {}
    thisoutput_last = initial_output.copy() if initial_output else {}
    
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

    # Initialize MCMC enhancement variables
    print(f"MCMC Enhanced Sampling Configuration:")
    print(f"  Adaptive proposals: {'ENABLED' if enable_adaptive else 'DISABLED'}")
    print(f"  Delayed rejection: {'ENABLED' if enable_delayed_rejection else 'DISABLED'}")
    print(f"  Block sampling: {'ENABLED' if enable_block_sampling else 'DISABLED'}")
    print(f"  Parallel tempering: {'ENABLED' if enable_parallel_tempering else 'DISABLED'}")
    print(f"  Auto convergence: {'ENABLED' if enable_auto_convergence else 'DISABLED'}")
    
    # ======================================================================
    # Intelligent Multi-Phase Burn-in Setup
    # ======================================================================
    total_burnin = burnsteps * nburn
    
    # Phase 1: Aggressive exploration (first 30% of burn-in)
    phase1_end = int(total_burnin * 0.3)
    # Phase 2: Adaptive refinement (next 50% of burn-in)  
    phase2_end = int(total_burnin * 0.8)
    # Phase 3: Fine-tuning (final 20% of burn-in)
    phase3_end = total_burnin
    
    print(f"Multi-Phase Burn-in Structure:")
    print(f"  Phase 1 (Exploration): steps 0-{phase1_end} ({phase1_end} steps)")
    print(f"  Phase 2 (Adaptation):  steps {phase1_end+1}-{phase2_end} ({phase2_end-phase1_end} steps)")
    print(f"  Phase 3 (Fine-tuning): steps {phase2_end+1}-{phase3_end} ({phase3_end-phase2_end} steps)")
    
    # Initialize adaptive MCMC tracking variables (only if enabled)
    if enable_adaptive:
        # Phase-specific adaptation intervals
        adaptation_interval = max(15, min(nburn // 30, 50))  # Very frequent in early phases
        warmup_phase = phase2_end  # Warmup extends through adaptation phase
        adaptation_history = []
        last_adaptation_step = 0
        proposal_method = 'multivariate_normal'  # Start with standard method
        
        # Phase-specific settings
        current_burnin_phase = 1
        phase_settings = {
            1: {'adaptation_rate': 0.3, 'target_accept': 0.4, 'adaptation_interval': 15},
            2: {'adaptation_rate': 0.15, 'target_accept': 0.3, 'adaptation_interval': 25}, 
            3: {'adaptation_rate': 0.05, 'target_accept': 0.234, 'adaptation_interval': 50}
        }
        
        # Momentum-like adaptation tracking
        momentum_decay = 0.9
        momentum_vector = np.zeros(nparms)
        previous_gradient_estimate = np.zeros(nparms)
        
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
            
            # Adjust initial step sizes based on sensitivity
            for p in range(nparms):
                param_name = self.ensemble_parms[p] if p < len(self.ensemble_parms) else f'param_{p}'
                if param_name in sensitivity_info['high_sensitivity_params']:
                    # Reduce step size for highly sensitive parameters
                    mycov[p, p] *= 0.5
                elif param_name in sensitivity_info['low_sensitivity_params']:
                    # Increase step size for less sensitive parameters
                    mycov[p, p] *= 1.8
            
            print(f"      Applied sensitivity-based initial scaling and step size adjustment")
        else:
            print(f"MCMC: No sensitivity analysis results found")
            print(f"      Consider running GSA first for optimal MCMC tuning")
            sensitivity_info = None
    else:
        adaptation_history = []
        proposal_method = 'multivariate_normal'
        sensitivity_info = None
    
    # Initialize advanced sampling tracking
    if enable_delayed_rejection:
        delayed_rejection_stats = {'attempts': 0, 'stage1_accepts': 0, 'stage2_accepts': 0}
    
    if enable_block_sampling:
        correlation_blocks = None  # Will be computed dynamically
        block_sampling_stats = {'attempts': 0, 'blocks_accepted': 0, 'total_blocks': 0}
    
    if enable_parallel_tempering:
        temp_stats = {'temp_steps': 0, 'temp_accepts': 0}
        
    if enable_auto_convergence:
        convergence_check_interval = max(min(nburn // 2, 500), 250)  # More frequent initial checks
        last_convergence_check = 0
        adaptive_check_interval = convergence_check_interval  # Will adapt based on convergence progress
        poor_convergence_count = 0  # Track consecutive poor convergence checks
        convergence_history = {
            'iterations': [],
            'rhat_max': [],
            'ess_min': [],
            'converged': []
        }
    
    for i in range(0,nevals):
        # ======================================================================
        # Intelligent Phase Management during Burn-in  
        # ======================================================================
        if enable_adaptive and i < total_burnin:
            # Determine current burn-in phase
            old_phase = getattr(self, '_current_phase', 1)
            if i <= phase1_end:
                current_phase = 1
            elif i <= phase2_end:
                current_phase = 2
            else:
                current_phase = 3
            
            # Phase transition management
            if current_phase != old_phase:
                phase_names = {1: 'Exploration', 2: 'Adaptation', 3: 'Fine-tuning'}
                print(f"\n🔄 BURN-IN PHASE {current_phase}: {phase_names[current_phase]} (step {i})")
                
                # Reset adaptation statistics for new phase
                accepted_step = 0
                
                # Apply phase-specific covariance adjustments
                if current_phase == 1:
                    # Phase 1: Large steps for exploration
                    adjustment_factor = 1.5
                elif current_phase == 2:
                    # Phase 2: Moderate steps, transitioning from exploration  
                    adjustment_factor = 0.6  # Reduce from exploration
                elif current_phase == 3:
                    # Phase 3: Smaller steps for fine-tuning
                    adjustment_factor = 0.8  # Further refinement
                
                if current_phase != old_phase and old_phase > 0:
                    mycov *= adjustment_factor
                    print(f"   Applied {adjustment_factor}x covariance scaling")
                
                self._current_phase = current_phase
            
            # Get phase-specific settings
            if current_phase == 1:
                current_adapt_rate = 0.3
                current_target_accept = 0.4
                current_adapt_interval = 15
            elif current_phase == 2:
                current_adapt_rate = 0.15
                current_target_accept = 0.3
                current_adapt_interval = 25
            else:  # Phase 3
                current_adapt_rate = 0.05
                current_target_accept = 0.234
                current_adapt_interval = 50
        else:
            # Post-burn-in settings
            current_adapt_rate = 0.05
            current_target_accept = 0.234
            current_adapt_interval = 100
            
        #update proposal step size using enhanced adaptive methods
        # Dynamic adaptation frequency: more frequent during warmup
        is_warmup = i < warmup_phase if enable_adaptive else False
        
        if enable_adaptive and (i > 0 and (i % current_adapt_interval) == 0 and i < burnsteps*nburn):
            acc_ratio = float(accepted_step) / current_adapt_interval
            
            # Get recent chain samples for adaptation
            recent_samples = chain_burn[0:nparms, max(0, accepted_tot-adaptation_interval):accepted_tot]
            
            # Apply enhanced adaptive metropolis update
            if recent_samples.shape[1] > nparms:  # Need enough samples
                # Use phase-specific adaptation parameters
                phase_adapt_rate = current_adapt_rate if 'current_adapt_rate' in locals() else 0.1
                phase_target_accept = current_target_accept if 'current_target_accept' in locals() else 0.234
                
                new_cov, adaptation_info = adaptive_metropolis_update(
                    current_cov=mycov,
                    chain_samples=recent_samples, 
                    accept_rate=acc_ratio,
                    target_accept=phase_target_accept,
                    adaptation_rate=phase_adapt_rate,
                    min_samples=max(30, nparms*2)  # Reduced min samples for faster adaptation
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
                
                # Apply sensitivity-informed scaling if available (throughout burn-in)
                if sensitivity_info is not None and i < burnsteps * nburn:  # Throughout burn-in phase
                    # Dynamic adaptation factor: stronger early, gentler later
                    burnin_progress = i / (burnsteps * nburn)
                    dynamic_adapt_factor = 0.15 * (1 - burnin_progress) + 0.05 * burnin_progress
                    
                    mycov, sens_scaling_info = apply_sensitivity_informed_scaling(
                        self, mycov, sensitivity_info, adaptation_factor=dynamic_adapt_factor)
                    
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
                
                # Phase-specific adaptation reporting
                if enable_adaptive and i < total_burnin:
                    # Report more frequently in early phases, less in later phases
                    should_report = False
                    if current_phase == 1 and i % 100 == 0:  # Every 100 steps in exploration
                        should_report = True  
                    elif current_phase == 2 and i % 200 == 0:  # Every 200 steps in adaptation
                        should_report = True
                    elif current_phase == 3 and i % 500 == 0:  # Every 500 steps in fine-tuning
                        should_report = True
                        
                    if should_report:
                        phase_names = {1: 'Exploration', 2: 'Adaptation', 3: 'Fine-tuning'}
                        print(f"  Phase {current_phase} ({phase_names[current_phase]}) step {i}: "
                              f"accept_rate={acc_ratio:.3f}, "
                              f"method={adaptation_info['reason']}, "
                              f"scale_factor={adaptation_info['new_scale']:.3f}")
                        
                        # Show condition number for monitoring numerical health
                        if 'correlation_issues' in locals() and 'condition_number' in correlation_issues:
                            cond_num = correlation_issues['condition_number'] 
                            if cond_num > 1e6:
                                print(f"    ⚠️  High condition number: {cond_num:.2e}")
            
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
            # ======================================================================
            # Burn-in Completion Summary and Diagnostics
            # ======================================================================
            burnin_accept_rate = accepted_tot / (i + 1) if i > 0 else 0.0
            print(f"\n🎉 BURN-IN COMPLETED after {i+1} steps!")
            print(f"  Overall acceptance rate: {burnin_accept_rate:.1%}")
            print(f"  Total accepted steps: {accepted_tot}")
            
            # Phase-specific analysis if adaptive was enabled
            if enable_adaptive and hasattr(self, '_current_phase'):
                print(f"  Successfully completed all 3 burn-in phases:")
                print(f"    ✓ Phase 1 (Exploration): steps 1-{phase1_end}")
                print(f"    ✓ Phase 2 (Adaptation):  steps {phase1_end+1}-{phase2_end}")
                print(f"    ✓ Phase 3 (Fine-tuning): steps {phase2_end+1}-{phase3_end}")
                
                # Analyze final covariance condition
                try:
                    final_condition = np.linalg.cond(mycov)
                    if final_condition < 1e6:
                        print(f"  ✓ Final covariance condition number: {final_condition:.2e} (good)")
                    elif final_condition < 1e10:
                        print(f"  ⚠️  Final covariance condition number: {final_condition:.2e} (marginal)")
                    else:
                        print(f"  ❌ Final covariance condition number: {final_condition:.2e} (poor)")
                except:
                    print(f"  Could not compute final covariance condition number")
            
            print(f"  Starting main sampling phase with {nevals - (i+1)} iterations...")
            
            #Parameter chain plots (burn-in period)
            for p in range(0,nparms):
                fig = plt.figure()
                burnin_end = int(nburn*burnsteps)
                chain_burnin = chain[p, 0:burnin_end]
                xchain = np.arange(1, len(chain_burnin) + 1)  # Match actual chain length
                plt.plot(xchain, chain_burnin)
                plt.xlabel('Evaluations')
                plt.ylabel(self.ensemble_parms[p])
                
                # Add phase boundary markers if multi-phase burn-in was used
                if enable_adaptive:
                    plt.axvline(x=phase1_end, color='red', linestyle='--', alpha=0.7, label='Phase 1→2')
                    plt.axvline(x=phase2_end, color='orange', linestyle='--', alpha=0.7, label='Phase 2→3')
                    plt.legend()
                
                if not os.path.exists(UQ_output+'/MCMC_output/plots/chains'):
                    os.makedirs(UQ_output+'/MCMC_output/plots/chains')
                plt.savefig(UQ_output+'/MCMC_output/plots/chains/burnin_chain_'+self.ensemble_parms[p]+'.pdf')
                plt.close(fig) 
    
        # ======================================================================
        # Enhanced Proposal Generation and Sampling
        # ======================================================================
        
        # Initialize proposal variables
        parms = parm_last.copy()
        post = post_last
        step_accepted = False
        step_info = {'method': 'standard', 'details': {}}
        
        # Strategy 1: Block sampling (if enabled and blocks exist)
        if (enable_block_sampling and i > burnsteps * nburn // 4 and 
            correlation_blocks is not None and len(correlation_blocks) > 1):
            
            # Conservative block sampling probability for efficiency
            block_probability = 0.3 if i > burnsteps * nburn else 0.2  # Reduced frequency
            if random.random() < block_probability:
                parms_new, post_new, n_accepted, block_info = block_sampling_step(
                    self, parm_last, mycov, myvars, post_last, 
                    correlation_blocks, sensitivity_info)
                
                if n_accepted > 0:
                    parms = parms_new
                    post = post_new
                    step_accepted = True
                    step_info = {'method': 'block_sampling', 'details': block_info}
                    
                    if 'block_sampling_stats' in locals():
                        block_sampling_stats['attempts'] += 1
                        block_sampling_stats['blocks_accepted'] += n_accepted
                        block_sampling_stats['total_blocks'] += block_info['blocks_tried']
        
        # Strategy 2: Parallel tempering (if enabled)
        if (not step_accepted and enable_parallel_tempering and 
            i > burnsteps * nburn // 4):  # After quarter of burn-in
            
            parms_new, post_new, temp_info = parallel_tempering_step(
                self, parm_last, mycov, myvars, post_last)
            
            if temp_info['accepted']:
                parms = parms_new
                post = post_new
                step_accepted = True
                step_info = {'method': 'parallel_tempering', 'details': temp_info}
                
                if 'temp_stats' in locals():
                    temp_stats['temp_steps'] += 1
                    if temp_info['accepted']:
                        temp_stats['temp_accepts'] += 1
        
        # Strategy 3: Delayed rejection (if enabled and no previous acceptance)
        if not step_accepted and enable_delayed_rejection:
            parms_new, post_new, dr_accepted, stage = delayed_rejection_step(
                self, parm_last, mycov, myvars, post_last,
                max_stages=2, scale_factors=[1.0, 0.5])
            
            if dr_accepted:
                parms = parms_new
                post = post_new
                step_accepted = True
                step_info = {'method': 'delayed_rejection', 'details': {'stage': stage}}
                
                if 'delayed_rejection_stats' in locals():
                    delayed_rejection_stats['attempts'] += 1
                    if stage == 1:
                        delayed_rejection_stats['stage1_accepts'] += 1
                    elif stage == 2:
                        delayed_rejection_stats['stage2_accepts'] += 1
        
        # Strategy 4: Enhanced adaptive proposals (fallback)
        if not step_accepted:
            if enable_adaptive:
                parms_proposal, proposal_success = enhanced_proposal_step(
                    parm_last, mycov, method=proposal_method)
                
                # Fallback to diagonal proposals if enhanced method fails
                if not proposal_success and proposal_method != 'multivariate_normal':
                    parms_proposal, _ = enhanced_proposal_step(
                        parm_last, mycov, method='multivariate_normal')
                    if i < burnsteps * nburn and i % (nburn * 2) == 0:
                        print(f"  Iteration {i}: Fallback to standard multivariate normal proposal")
            else:
                # Standard proposal for non-adaptive mode
                parms_proposal = np.random.multivariate_normal(parm_last, mycov)
            
            # Calculate posterior for standard proposal
            post_proposal, thisoutput = calc_posterior(self, parms_proposal, myvars)
            
            # Standard Metropolis acceptance
            if post_proposal - post_last >= np.log(random.uniform(0, 1)):
                parms = parms_proposal
                post = post_proposal
                step_accepted = True
                step_info = {'method': 'standard_metropolis', 'details': {}}
            else:
                # Proposal rejected, keep current values
                parms = parm_last
                post = post_last
                step_info = {'method': 'rejected', 'details': {}}
        
        # Ensure we have output for this step
        if step_accepted:
            if 'thisoutput' not in locals():
                _, thisoutput = calc_posterior(self, parms, myvars)
        else:
            thisoutput = thisoutput_last.copy() if 'thisoutput_last' in locals() else {}
        
        # ===============================================================
        # Process acceptance and update chains
        # ===============================================================
        if step_accepted:
            # Update chain tracking
            post_last = post
            accepted_tot = accepted_tot + 1
            accepted_step = accepted_step + 1
            
            # Bounds check for chain arrays
            if accepted_tot < chain_prop.shape[1] and accepted_tot < chain_burn.shape[1]:
                chain_prop[0:nparms, accepted_tot] = parms - parm_last
                chain_burn[0:nparms, accepted_tot] = parms
            parm_last = parms
            thisoutput_last = thisoutput.copy()
            
            # Track best solution
            if post > post_best:
                post_best = post
                parms_best = parms.copy()
                output_best = thisoutput
                print(f"New best posterior at iteration {i}: {post_best:.3f}")
        
        # ===============================================================
        # Update adaptation statistics and covariance
        # ===============================================================
        if enable_adaptive and i > burnsteps // 4:  # Start adaptation after 25% of burn-in
            # Update running statistics for covariance adaptation
            if 'chain_mean' not in locals():
                chain_mean = parms.copy()
                adaptation_count = 1
            else:
                chain_mean = (chain_mean * adaptation_count + parms) / (adaptation_count + 1)
                adaptation_count += 1
                
                # Update covariance every 25 steps during burn-in (more frequent updates)
                if i < burnsteps * nburn and i % 25 == 0 and adaptation_count > 5:
                    end_idx = min(i+1, chain.shape[1])
                    chain_recent = chain[0:nparms, max(0, i-500):end_idx]
                    if chain_recent.shape[1] > nparms:
                        # Calculate current acceptance rate for adaptive update
                        current_accept_rate = accepted_tot / (i + 1) if i > 0 else 0.0
                        
                        # Apply adaptive Metropolis update with phase-specific adaptation rate
                        phase_adapt_rate = current_adapt_rate if 'current_adapt_rate' in locals() else (0.2 if i < warmup_phase else 0.1)
                        mycov, adaptation_info = adaptive_metropolis_update(
                            mycov, chain_recent, accept_rate=current_accept_rate,
                            adaptation_rate=phase_adapt_rate)
                        
                        # Apply sensitivity-informed scaling with phase-specific factors
                        if sensitivity_info is not None:
                            # Phase-specific sensitivity factors  
                            if i < total_burnin:
                                if i <= phase1_end:
                                    sensitivity_factor = 0.2  # Most aggressive in exploration
                                elif i <= phase2_end:
                                    sensitivity_factor = 0.12  # Moderate in adaptation
                                else:
                                    sensitivity_factor = 0.06  # Conservative in fine-tuning
                            else:
                                sensitivity_factor = 0.03  # Very conservative post-burn-in
                                
                            mycov, _ = apply_sensitivity_informed_scaling(
                                self, mycov, sensitivity_info, adaptation_factor=sensitivity_factor)
                        
                        # Monitor correlation issues and adjust proposal method
                        param_names = [self.ensemble_parms[p] if p < len(self.ensemble_parms) 
                                     else f'param_{p}' for p in range(nparms)]
                        correlation_issues = detect_correlation_issues(chain_recent, param_names)
                        condition_num = correlation_issues.get('condition_number', 1.0)
                        
                        if correlation_issues['ill_conditioned']:
                            if condition_num > 1e12:  # Extremely ill-conditioned
                                proposal_method = 'diagonal'
                            else:  # Moderately ill-conditioned
                                proposal_method = 'eigendecomp'
                        elif len(correlation_issues['high_correlations']) > 0:
                            proposal_method = 'cholesky'  # Good for correlated parameters
                        else:
                            proposal_method = 'cholesky'  # Default stable method
                        
                        if i % 500 == 0:  # Less frequent adaptation logging for performance
                            print(f"  Adaptation at iteration {i}: method={proposal_method}, "
                                  f"condition={np.linalg.cond(mycov):.2e}")
        
        # ===============================================================
        # Convergence monitoring and auto-stopping (if enabled)
        # ===============================================================
        if (enable_auto_stopping and i > burnsteps * nburn and 
            i % adaptive_check_interval == 0):
            
            # Check convergence every specified interval
            start_idx = int(nburn * burnsteps)
            end_idx = min(i+1, chain.shape[1])
            chain_recent = chain[0:nparms, start_idx:end_idx]
            if chain_recent.shape[1] > 4 * nparms:  # Need sufficient samples
                
                convergence_result = self.convergence_diagnostics(chain_recent)
                
                # Store convergence history
                convergence_history['iterations'].append(i)
                convergence_history['rhat_max'].append(convergence_result['rhat_max'])
                convergence_history['ess_min'].append(convergence_result['ess_min'])
                convergence_history['converged'].append(convergence_result['all_converged'])
                
                # Check auto-stopping criterion
                should_stop, stop_reason = self.auto_stopping_criterion(convergence_result, min_ess=100)
                if should_stop:
                    print(f"\n🎯 CONVERGENCE ACHIEVED at iteration {i}!")
                    print(f"   Reason: {stop_reason}")
                    print(f"   Max R-hat: {convergence_result['rhat_max']:.4f}")
                    print(f"   Min ESS: {convergence_result['ess_min']:.1f}")
                    
                    # Report individual parameter convergence status
                    converged_count = sum(convergence_result['converged'].values())
                    total_params = len(convergence_result['converged'])
                    print(f"   Parameters converged: {converged_count}/{total_params}")
                    
                    # Show any parameters that haven't converged (should be none if we reach this point)
                    unconverged_params = [name for name, converged in convergence_result['converged'].items() if not converged]
                    if unconverged_params:
                        print(f"   ⚠️  Unconverged parameters: {unconverged_params}")
                    
                    print(f"   Stopping early (requested {nevals} iterations)")
                    
                    # Truncate arrays to actual length
                    original_nevals = nevals
                    nevals = i + 1
                    
                    # Truncate chain and output arrays to actual used length
                    chain = chain[:, :nevals]
                    for v in myvars:
                        output[v] = output[v][:, :nevals]
                    
                    break
                else:
                    # Adaptive convergence check interval based on progress
                    converged_count = sum(convergence_result['converged'].values())
                    total_params = len(convergence_result['converged'])
                    convergence_fraction = converged_count / total_params if total_params > 0 else 0.0
                    
                    # Adapt check frequency based on convergence progress (optimized)
                    if convergence_fraction > 0.8:
                        # Close to convergence - check more frequently but not too often
                        adaptive_check_interval = max(convergence_check_interval // 2, 100)
                        poor_convergence_count = 0  # Reset poor convergence counter
                    elif convergence_fraction < 0.2:  # More stringent threshold
                        # Very poor convergence - check much less frequently to save time
                        adaptive_check_interval = min(convergence_check_interval * 4, 3000)
                        poor_convergence_count += 1
                        
                        # Emergency circuit breaker for very poor convergence
                        if poor_convergence_count > 5 and i > 5 * nburn * burnsteps:
                            print(f"\n⚠️  WARNING: Poor convergence detected after {poor_convergence_count} checks!")
                            print(f"   Only {converged_count}/{total_params} parameters converged after {i} iterations")
                            print(f"   Consider: longer burn-in, different priors, or model reparameterization")
                            print(f"   Continuing sampling but convergence may take much longer...")
                            poor_convergence_count = 0  # Reset to avoid repeated warnings
                    else:
                        # Normal progress - use standard interval
                        adaptive_check_interval = convergence_check_interval
                        poor_convergence_count = max(0, poor_convergence_count - 1)  # Slowly improve counter
                
                if i % (adaptive_check_interval * 10) == 0:  # Less frequent detailed reporting
                    # Simplified convergence progress reporting for performance
                    converged_count = sum(convergence_result['converged'].values())
                    total_params = len(convergence_result['converged'])
                    convergence_fraction = converged_count / total_params if total_params > 0 else 0.0
                    
                    print(f"  Convergence check at iteration {i}: "
                          f"R-hat max={convergence_result['rhat_max']:.4f}, "
                          f"ESS min={convergence_result['ess_min']:.1f}, "
                          f"converged {converged_count}/{total_params} ({convergence_fraction:.1%})")
                    
                    # Only show detailed info if close to convergence or having issues
                    if convergence_fraction > 0.8 or convergence_fraction < 0.3:
                        # Show worst R-hat parameters if any are unconverged
                        if converged_count < total_params:
                            unconverged_rhat = [(name, convergence_result['r_hat'][name]) 
                                              for name, converged in convergence_result['converged'].items() 
                                              if not converged and name in convergence_result['r_hat']]
                            if unconverged_rhat:
                                # Sort by worst R-hat and show worst 2
                                unconverged_rhat.sort(key=lambda x: x[1], reverse=True)
                                worst_params = unconverged_rhat[:2]  # Show top 2 worst
                                param_info = ", ".join([f"{name}={rhat:.3f}" for name, rhat in worst_params])
                                print(f"    Worst R-hat: {param_info}")
                        
                        # Show insufficient ESS parameters (only worst 2)
                        insufficient_ess = [(name, ess_val) 
                                          for name, ess_val in convergence_result['ess_bulk'].items() 
                                          if ess_val < 100]
                        if insufficient_ess and len(insufficient_ess) <= 5:  # Only show if not too many
                            insufficient_ess.sort(key=lambda x: x[1])  # Sort by lowest ESS
                            worst_ess = insufficient_ess[:2]  # Show top 2 worst
                            ess_info = ", ".join([f"{name}={ess:.1f}" for name, ess in worst_ess])
                            print(f"    Low ESS: {ess_info}")
        
        # Debug every 2000 iterations (less frequent for performance)
        if i % 2000 == 0:
            accept_rate = accepted_tot / (i + 1)
            print(f"Iteration {i}: posterior={post:.3f}, acceptance={accept_rate:.3f}, "
                  f"method={step_info['method']}")

        #populate the chain matrix
        for j in range(0,nparms):
            chain[j][i] = parms[j]
        chain[nparms][i] = post_last
        for v in myvars:
            if (post_last > -9000000) and v in thisoutput:
              output[v][:,i] = thisoutput[v][:]
            elif 'thisoutput_last' in locals() and v in thisoutput_last:
              output[v][:,i] = thisoutput_last[v][:]
            else:
              # Handle case where neither thisoutput nor thisoutput_last has this variable
              output[v][:,i] = np.zeros(self.nobs[v])  # or some default value
        #if (i % 1000 == 0):
        #    print(' -- '+str(i)+' --\n')

    #print("Computing statistics")
    burnin_end_idx = int(nburn*burnsteps)
    actual_chain_length = chain.shape[1]
    
    # Handle case where early stopping occurred before burn-in completed
    if burnin_end_idx >= actual_chain_length:
        print(f"Warning: Early stopping occurred at iteration {actual_chain_length}, but burn-in period was {burnin_end_idx}")
        print(f"Using last 50% of samples as 'post-burn-in' data")
        burnin_end_idx = max(0, actual_chain_length // 2)
    
    chain_afterburn = chain[0:nparms, burnin_end_idx:actual_chain_length]
    chain_sorted = chain_afterburn
    output_sorted={}
    for v in myvars:
      output_sorted[v] = output[v][0:self.nobs[v], burnin_end_idx:actual_chain_length]
      output_sorted[v].sort()

    np.savetxt(UQ_output+'/MCMC_output/MCMC_chain.txt', np.transpose(chain_afterburn))
    #Print out some statistics
    
    # Debug: parms_best is now properly initialized at start of function
    print(f"DEBUG: parms_best final values: {parms_best}")
    
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
        # Use actual chain_afterburn length to avoid dimension mismatch
        actual_length = chain_afterburn.shape[1]
        
        # Check if we have sufficient post-burn-in samples
        if actual_length <= 0:
            print(f"Error: No post-burn-in samples available for plotting parameter {p}")
            print(f"Skipping plot for {self.ensemble_parms[p]}")
            plt.close(fig)
            continue
        
        # Use actual chain data
        xchain = np.arange(1, actual_length + 1)  # 1-indexed for cleaner plots  
        chain_data = chain_afterburn[p,:]
        
        # Remove any trailing zeros that might exist from initialization
        if actual_length > 10:  # Only clean if we have sufficient data
            # Find the last non-zero value
            nonzero_indices = np.nonzero(chain_data)[0]
            if len(nonzero_indices) > 0:
                last_nonzero_idx = nonzero_indices[-1]
                # If there are many trailing zeros, trim them
                if last_nonzero_idx < actual_length - 5:
                    print(f"Trimming {actual_length - last_nonzero_idx - 1} trailing zeros from {self.ensemble_parms[p]}")
                    chain_data = chain_data[:last_nonzero_idx + 1]
                    xchain = xchain[:last_nonzero_idx + 1]
            
        plt.plot(xchain, chain_data)
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
    n_samples_afterburn = actual_chain_length - burnin_end_idx
    
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
    
    # ======================================================================
    # Generate comprehensive advanced MCMC diagnostics report
    # ======================================================================
    print("Generating comprehensive advanced MCMC diagnostics report...")
    
    import time
    
    with open(UQ_output + '/MCMC_output/advanced_mcmc_report.txt', 'w') as f:
        f.write("# Advanced MCMC Diagnostics Report\n")
        f.write("# Generated from enhanced custom MCMC implementation\n")
        f.write(f"# Simulation: {self.casename}\n")
        f.write(f"# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # ============== MCMC Configuration ================
        f.write("## MCMC Configuration\n")
        f.write(f"Total iterations: {nevals}\n")
        f.write(f"Burn-in fraction: {nburn}\n")
        f.write(f"Burn-in steps: {burnsteps}\n")
        f.write(f"Total burn-in samples: {int(nburn * burnsteps)}\n")
        f.write(f"Post-burn-in samples: {n_samples_afterburn}\n")
        f.write(f"Number of parameters: {nparms}\n")
        f.write(f"Overall acceptance rate: {float(accepted_tot)/nevals:.3f}\n\n")
        
        # ============== Enhancement Features Used ================
        f.write("## Enhancement Features\n")
        f.write(f"Adaptive proposals: {'Enabled' if enable_adaptive else 'Disabled'}\n")
        f.write(f"Sensitivity-informed scaling: {'Available' if sensitivity_info is not None else 'Not available'}\n")
        f.write(f"Block sampling: {'Enabled' if enable_block_sampling else 'Disabled'}\n")
        f.write(f"Delayed rejection: {'Enabled' if enable_delayed_rejection else 'Disabled'}\n")
        f.write(f"Parallel tempering: {'Enabled' if enable_parallel_tempering else 'Disabled'}\n")
        f.write(f"Auto-stopping: {'Enabled' if enable_auto_stopping else 'Disabled'}\n\n")
        
        # ============== Advanced Technique Statistics ================
        if 'block_sampling_stats' in locals() and block_sampling_stats['attempts'] > 0:
            f.write("## Block Sampling Statistics\n")
            f.write(f"Block sampling attempts: {block_sampling_stats['attempts']}\n")
            f.write(f"Blocks accepted: {block_sampling_stats['blocks_accepted']}\n")
            f.write(f"Total blocks tried: {block_sampling_stats['total_blocks']}\n")
            block_accept_rate = block_sampling_stats['blocks_accepted'] / block_sampling_stats['total_blocks']
            f.write(f"Block acceptance rate: {block_accept_rate:.3f}\n\n")
        
        if 'delayed_rejection_stats' in locals() and delayed_rejection_stats['attempts'] > 0:
            f.write("## Delayed Rejection Statistics\n")
            f.write(f"Delayed rejection attempts: {delayed_rejection_stats['attempts']}\n")
            f.write(f"Stage 1 acceptances: {delayed_rejection_stats['stage1_accepts']}\n")
            f.write(f"Stage 2 acceptances: {delayed_rejection_stats['stage2_accepts']}\n")
            dr_success_rate = (delayed_rejection_stats['stage1_accepts'] + 
                             delayed_rejection_stats['stage2_accepts']) / delayed_rejection_stats['attempts']
            f.write(f"Delayed rejection success rate: {dr_success_rate:.3f}\n\n")
        
        if 'temp_stats' in locals() and temp_stats['temp_steps'] > 0:
            f.write("## Parallel Tempering Statistics\n")
            f.write(f"Temperature steps attempted: {temp_stats['temp_steps']}\n")
            f.write(f"Temperature steps accepted: {temp_stats['temp_accepts']}\n")
            temp_accept_rate = temp_stats['temp_accepts'] / temp_stats['temp_steps']
            f.write(f"Temperature acceptance rate: {temp_accept_rate:.3f}\n\n")
        
        # ============== Sensitivity Analysis Integration ================
        if sensitivity_info is not None:
            f.write("## Sensitivity Analysis Integration\n")
            f.write("Sensitivity-informed scaling was applied to proposal covariance\n")
            
            # Get parameter sensitivity rankings
            param_importance = sensitivity_info.get('param_importance', {})
            if param_importance:
                high_sens_params = [p for p, info in param_importance.items() 
                                  if info['total_sensitivity'] > 0.05]
                low_sens_params = [p for p, info in param_importance.items() 
                                 if info['total_sensitivity'] < 0.01]
                
                f.write(f"High-sensitivity parameters ({len(high_sens_params)}): {', '.join(high_sens_params)}\n")
                f.write(f"Low-sensitivity parameters ({len(low_sens_params)}): {', '.join(low_sens_params)}\n")
                
                # Parameter interaction information
                interactions = sensitivity_info.get('interactions', [])
                strong_interactions = [pair for pair in interactions if len(pair) > 2 and pair[2] > 0.1]
                if strong_interactions:
                    f.write(f"Strong parameter interactions detected: {len(strong_interactions)}\n")
                    for p1, p2, strength in strong_interactions[:5]:  # Show top 5
                        f.write(f"  {p1} ↔ {p2}: {strength:.3f}\n")
            f.write("\n")
        
        # ============== Convergence Information ================
        if 'convergence_history' in locals():
            f.write("## Convergence Monitoring\n")
            f.write("Convergence was monitored using Gelman-Rubin R-hat and ESS diagnostics\n")
            f.write(f"Convergence checks performed: {len(convergence_history['iterations'])}\n")
            
            if convergence_history['rhat_max']:
                final_rhat = convergence_history['rhat_max'][-1]
                final_ess = convergence_history['ess_min'][-1]
                f.write(f"Final R-hat (max across parameters): {final_rhat:.4f}\n")
                f.write(f"Final ESS (min across parameters): {final_ess:.1f}\n")
                
                converged_checks = sum(convergence_history['converged'])
                f.write(f"Convergence achieved in {converged_checks}/{len(convergence_history['converged'])} checks\n")
            
            # Check if early stopping occurred
            if nevals < len(convergence_history.get('iterations', [])):
                f.write("Early stopping was triggered due to convergence\n")
            f.write("\n")
        
        # ============== ESS Summary ================
        f.write("## Effective Sample Size Summary\n")
        f.write(f"Min ESS: {ess_stats['min']:.2f}\n")
        f.write(f"Max ESS: {ess_stats['max']:.2f}\n")
        f.write(f"Mean ESS: {ess_stats['mean']:.2f}\n")
        f.write(f"Median ESS: {ess_stats['median']:.2f}\n")
        f.write(f"Overall sampling efficiency: {ess_stats['mean']/n_samples_afterburn:.1%}\n\n")
        
        # ============== Final Recommendations ================
        f.write("## Recommendations for Future Runs\n")
        
        # Acceptance rate recommendations
        overall_accept_rate = float(accepted_tot) / nevals
        if overall_accept_rate < 0.15:
            f.write("- Acceptance rate is low (<15%). Consider smaller proposal steps or better initial covariance\n")
        elif overall_accept_rate > 0.7:
            f.write("- Acceptance rate is high (>70%). Consider larger proposal steps for faster mixing\n")
        else:
            f.write("- Acceptance rate is in good range (15-70%)\n")
        
        # ESS recommendations
        min_ess = ess_stats['min']
        if min_ess < 50:
            f.write("- Some parameters have very low ESS (<50). Consider longer runs or better proposals\n")
        elif min_ess > 200:
            f.write("- All parameters have good ESS (>200). Current run length is sufficient\n")
        
        # Enhancement recommendations
        if not enable_adaptive:
            f.write("- Consider enabling adaptive proposals for better performance\n")
        if sensitivity_info is None:
            f.write("- Consider running Global Sensitivity Analysis first to inform MCMC proposals\n")
        if not enable_auto_stopping:
            f.write("- Consider enabling auto-stopping to avoid unnecessary long runs\n")
        
        f.write("\n# End of Advanced MCMC Diagnostics Report\n")
    
    print(f"✅ Comprehensive MCMC report saved to: {UQ_output}/MCMC_output/advanced_mcmc_report.txt")
    
    # ======================================================================
    # Save adaptation history if available
    # ======================================================================
    if 'convergence_history' in locals() and convergence_history['iterations']:
        import pickle
        
        adaptation_data = {
            'convergence_history': convergence_history,
            'ess_results': ess_results,
            'tau_results': tau_results,
            'ess_stats': ess_stats,
            'mcmc_config': {
                'nevals': nevals,
                'nburn': nburn,
                'burnsteps': burnsteps,
                'nparms': nparms,
                'acceptance_rate': float(accepted_tot) / nevals,
                'enable_adaptive': enable_adaptive,
                'enable_block_sampling': enable_block_sampling,
                'enable_delayed_rejection': enable_delayed_rejection,
                'enable_parallel_tempering': enable_parallel_tempering,
                'enable_auto_stopping': enable_auto_stopping
            }
        }
        
        # Add technique-specific stats if available
        if 'block_sampling_stats' in locals():
            adaptation_data['block_sampling_stats'] = block_sampling_stats
        if 'delayed_rejection_stats' in locals():
            adaptation_data['delayed_rejection_stats'] = delayed_rejection_stats
        if 'temp_stats' in locals():
            adaptation_data['temp_stats'] = temp_stats
        if sensitivity_info is not None:
            adaptation_data['sensitivity_info'] = sensitivity_info
        
        with open(UQ_output + '/MCMC_output/adaptation_history.pkl', 'wb') as f:
            pickle.dump(adaptation_data, f)
        
        print(f"✅ Adaptation history saved to: {UQ_output}/MCMC_output/adaptation_history.pkl")
    
    return parms_best


