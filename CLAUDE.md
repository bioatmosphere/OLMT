# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Offline Land Model Testbed (OLMT) is a Python framework for running and managing offline Earth System Model (E3SM) Land Model simulations. It simplifies workflows for single-site, regional, and ensemble ELM (Energy Exascale Earth System Model Land component) simulations with uncertainty quantification capabilities.

## Common Commands

Since this is a pure Python scientific computing project, there are no traditional build/test commands. Development workflow:

```bash
# Setup conda environment (choose appropriate one for your machine)
conda env create -f conda_envs/OLMT_baseline.yml
conda activate OLMT_baseline

# Run a simulation using existing run scripts
cd runscripts/
python run_TAM.py  # or other run_*.py scripts

# Run ensemble simulations (requires SLURM environment)
python manage_ensemble.py

# Launch experimental GUI
python GUI_experimental.py

# Run post-processing/analysis
python plotcase.py
python compare_cases.py

# Modify NetCDF files
python modify_netcdf.py

# Adjust restart files
python adjust_restart.py
```

## Core Architecture

### Central Class: `ELMcase` (model_ELM/main.py)
The entire framework revolves around the `ELMcase` class (~1000+ lines) which orchestrates:
- Case setup (create_case, setup_case, build_case)
- Ensemble management with parameter sampling
- Surrogate model training and execution
- Uncertainty quantification (MCMC, Global Sensitivity Analysis)
- Post-processing and analysis

### Key Modules

**model_ELM/** - Core Python package:
- `main.py` - ELMcase class (main orchestrator)
- `ensemble.py` - Ensemble simulation management
- `surrogate_NN.py` - Neural network surrogate models using scikit-learn MLPRegressor
- `MCMC.py` - Markov Chain Monte Carlo for parameter estimation
- `postprocess.py` - Analysis and visualization tools
- `run_GSA.py` - Global Sensitivity Analysis using SALib
- `netcdf4_functions.py` - NetCDF data manipulation utilities

**runscripts/** - Entry point scripts that users customize:
- `run_TAM.py`, `run_SPRUCE.py`, `run_BGC.py` etc. - Specific experiment templates
- Users copy and modify these for their simulations

### Workflow Types

1. **Single Site**: Point-based simulations using observational data (e.g., AmeriFlux sites)
2. **Regional**: Spatial domain simulations with lat/lon bounds or point lists  
3. **Ensemble**: Parameter uncertainty quantification with:
   - Sampling methods: Monte Carlo, Latin Hypercube, Sobol sequences
   - Surrogate modeling: Neural networks for fast approximation
   - Analysis: Global sensitivity analysis, MCMC parameter estimation

### Machine Support
Auto-detects HPC environments via `get_machine_info()`:
- CADES, Chrysalis, Perlmutter
- Docker containers
- Generic Linux systems

## Development Patterns

### Parameter Files
- `examples/parm_list_*` - Define parameter names, ranges, and distributions for uncertainty quantification
- Parameter sampling uses SALib methods (sobol, latin, etc.)

### Configuration Flow
1. Create/modify run script in `runscripts/`
2. Set site/region, compset, years, resolution
3. Optionally configure ensemble parameters 
4. Execute script - OLMT handles case creation, building, submission automatically

### Data Handling
- **Input**: Meteorological forcing, surface data, parameter files (NetCDF4)
- **Output**: Model results, ensemble matrices, analysis plots (NetCDF4, pickle)
- **Integration**: Built-in support for AmeriFlux, FLUXNET observational data

### Key Environment Variables
- `$HOME/models/E3SM` - Default E3SM model source code location
- Cases written to machine-specific directories (auto-detected)
- `SLURM_JOB_NODELIST` - Used by manage_ensemble.py for parallel execution

## Conda Environment Management

Multiple conda environment files are available for different machines:
- `conda_envs/OLMT_baseline.yml` - General purpose environment
- `conda_envs/OLMT_chrysalis.yml` - Chrysalis HPC system
- `conda_envs/OLMT_pm.yml` - Perlmutter HPC system

## Important Notes

- No traditional build system - pure Python workflow
- Dependencies managed via conda environments (scientific computing stack)
- Simulation execution depends on E3SM/ELM model being available
- Supports multiple model configurations: ELM, FATES, various compsets
- Uses SLURM for HPC job submission and dependency management
- Ensemble simulations create many parallel run directories managed by MPI
- Case objects are serialized as pickle files in `pklfiles/` directory
- Modified source code can be placed in `srcmods*/` directories