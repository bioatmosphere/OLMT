"""
Script to integrate MCMC subset functionality into ELMcase class

This adds the run_MCMC_subset method to the ELMcase class dynamically.
Run this once to enable the functionality.
"""

import sys
import model_ELM
from model_ELM.MCMC_subset import run_MCMC_subset, load_MCMC_subset_results

# Add method to ELMcase class
model_ELM.main.ELMcase.run_MCMC_subset = run_MCMC_subset
model_ELM.main.ELMcase.load_MCMC_subset_results = staticmethod(load_MCMC_subset_results)

print("✓ MCMC subset functionality integrated into ELMcase class")
print("\nYou can now use:")
print("  mycase.run_MCMC_subset(calibrate_params, fixed_params, ...)")
print("  results = ELMcase.load_MCMC_subset_results(casename)")
