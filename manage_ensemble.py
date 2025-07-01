#!/usr/bin/env python
import sys,os, time
import numpy as np
import subprocess
import pickle
import model_ELM
from optparse import OptionParser


def get_nodelist():
  """Get the list of nodes from the SLURM job node list.
  Returns
  -------
  mynodes : list
      List of nodes in the job.
  """

  mynodes=[]
  nodelist=os.environ['SLURM_JOB_NODELIST'].split('xxx')
  print(nodelist)
  for n in nodelist:
    if ('[' in n):
        node_prefix=n.split('[')[0]
        nodelist2=n.split('[')[1].split(',')
        for n2 in nodelist2:
          if ('-' in n2):
            firstnode=n2.split('-')[0]
            lastnode=n2.split('-')[1].strip(']')
            for nn in range(int(firstnode),int(lastnode)+1):
              if ('baseline' in mycase.machine):
                nstr = str(nn)
              else:
                nstr = str(10000+nn)[1:]
              mynodes.append(node_prefix+nstr)
          else:
              if ('baseline' in mycase.machine):
                nstr=str(n2).strip(']')
              else:
                nstr=str(10000+n2)[1:].strip(']')
              mynodes.append(node_prefix+nstr)
    else:
        mynodes.append(n)
  return mynodes

def get_node_submit(pactive,process_nodes,mynodes):
    node_submit=0
    for n in range(0,len(mynodes)):
         ctn=0    #Counter for active processes on each node
         for p in range(0,len(processes)):
                if pactive[p] == 1 and process_nodes[p] == n:
                    ctn=ctn+1
         if (ctn < mycase.npernode/mycase.np):
             #If this node is not full, submit
             node_submit=n
    return(node_submit)

def check_run_success(n):
    success=False
    jobst = str(100000+n)
    rundir = mycase.runroot+'/UQ/'+mycase.casename+'/g'+jobst[1:]
    yst = str(10000+mycase.startyear+mycase.run_n)[1:]
    if (os.path.isfile(rundir+'/'+mycase.casename+'.elm.r.'+yst+'-01-01-00000.nc')):
        success=True
    return success

def active_processes(processes,process_jobnum,process_hang):
    """Returns the number of processes that are still running.
    
    
    Parameters
    ----------
    processes : list
        List of processes that are currently running.
    process_jobnum : list
        List of job numbers for the processes.
    process_hang : list
        List of hang counts for the processes.

    Returns
    -------
    pactive : list
        List of active processes (1 if running, 0 if not).
    """

    pactive=[]
    n=0
    for process in processes:
        if process.poll() is None:  # None means the process is still running
            #Check if final restart file created
            pactive.append(1)
            if (check_run_success(process_jobnum[n])):
                process_hang[n] = process_hang[n]+1
            if (process_hang[n] > 30):
                process.kill()  # Force kill the process
        else:
            pactive.append(0)
            #Post-process ensemble member if it hasn't yet been done
            if (mycase.postprocessed[n] == 0):
                print(n, check_run_success(process_jobnum[n]))
                if (check_run_success(process_jobnum[n])):
                    ierr = postprocess_ensemble(process_jobnum[n])
                else:
                    print('Ensemble member '+str(process_jobnum[n])+ \
                            'Failed to complete')
                mycase.postprocessed[n] = 1
        n=n+1
    return pactive

def postprocess_ensemble(n):
  """Postprocess ensemble member outputs.

  Parameters
  ----------
  n : int
      The ensemble member number to postprocess.
  Returns
  -------
  ierr : int
      Error code (0 for success, 1 for failure).
  """

  if (mycase.postproc_vars != []):
      for v in mycase.postproc_vars:
        hnum=1
        mypfts=[0]
        if ('_pft' in v):
            #PFT level outputs requested
            hnum=2
            mypfts=mycase.postproc_pfts
        for p in mypfts:
          if (mycase.postproc_freq == 'daily' or mycase.postproc_freq == 'hourly'):  #default
            mycase.postprocess(v, ens_num=n,startyear=mycase.postproc_startyear, \
                  endyear=mycase.postproc_endyear,index=p,hnum=hnum)
          elif (mycase.postproc_freq == 'monthly'):  #monthly
            mycase.postprocess(v, ens_num=n,startyear=mycase.postproc_startyear, \
                  endyear=mycase.postproc_endyear,index=p,hnum=hnum, dailytomonthly=True)
          elif (mycase.postproc_freq == 'annual'):  #annual
            mycase.postprocess(v, ens_num=n,startyear=mycase.postproc_startyear, \
                  endyear=mycase.postproc_endyear,index=p,hnum=hnum, annualmean=True)
  return 0


parser = OptionParser()

parser.add_option("--case", dest="case", default="", \
                  help="Case name")
parser.add_option("--postproc_only", dest="postproc_only", default=False, \
                  action="store_true")
parser.add_option("--UQ_only", dest="UQ_only", default=False, \
                  action="store_true")
parser.add_option("--MCMC_only", dest="MCMC_only", default=False, \
                  action="store_true", help="Only run MCMC parameter estimation")
parser.add_option("--obs_dir", dest="obs_dir", default="observations/fluxnet", \
                  help="Directory containing FLUXNET observation files (for MCMC-only mode)")
parser.add_option("--tstep", dest="tstep", default="monthly", \
                  help="Time step for observation data ('monthly', 'daily', or 'yearly')")
(options, args) = parser.parse_args()

#Load case object
myfile=open('pklfiles/'+options.case+'.pkl','rb')
mycase=pickle.load(myfile)

#workdir = os.getcwd()

if (not options.UQ_only and not options.MCMC_only):
  processes=[]
  process_jobnum=[]
  process_hang=[]    #Keep track of how long process has been hanging
  mycase.postprocessed=np.zeros([mycase.nsamples],int)
  n_job = 1
  if (mycase.noslurm == False):
    process_nodes = []
    mynodes = get_nodelist()

  #Run the simulations 
  while (n_job <= mycase.nsamples):
    pactive = active_processes(processes,process_jobnum,process_hang)
    if (sum(pactive) < int(mycase.np_ensemble)):
      jobst = str(100000+n_job)
      rundir = mycase.runroot+'/UQ/'+mycase.casename+'/g'+jobst[1:]+'/'
      log_file_path = f"{rundir}e3sm_log.txt"
      #Copy relevant files
      if not options.postproc_only:
        mycase.ensemble_copy(n_job)
      with open(log_file_path, "w") as log_file:
        if (mycase.noslurm == False):
          node_submit=get_node_submit(pactive,process_nodes,mynodes)
          command = ['srun -n '+str(mycase.np)+' -c 1 -w '+mynodes[node_submit]+' '+mycase.exeroot+'/e3sm.exe']
          process_nodes.append(node_submit)
        else:
          command = [mycase.exeroot+'/e3sm.exe']
        if (options.postproc_only):
            command='ls'
        process = subprocess.Popen(command, shell=True, stderr=subprocess.STDOUT, cwd=rundir, stdout=log_file)
        processes.append(process)
        process_jobnum.append(n_job)
        process_hang.append(0)
      n_job=n_job+1
    else:
      time.sleep(1)

  while (sum(pactive) > 0):
    pactive = active_processes(processes,process_jobnum,process_hang)
    time.sleep(1)

  mycase.create_pkl(outdir=mycase.OLMTdir+'/pklfiles/')

#UQ part of code

if options.MCMC_only:
    print("Running MCMC-only mode")
    
    # Validate prerequisites
    if not mycase.postproc_vars:
        print("Error: No postproc_vars defined for MCMC analysis")
        sys.exit(1)
    
    # Load observations if missing
    if not mycase.obs:
        # Determine observation directory
        obs_dir = options.obs_dir or getattr(mycase, 'obs_dir', None)
        if not obs_dir:
            print("Error: No observations found and no observation directory specified")
            print("Use --obs_dir <path> or ensure mycase.obs is populated")
            sys.exit(1)
            
        if not hasattr(mycase, 'site'):
            print("Error: mycase.site not defined - required for FLUXNET observation loading")
            sys.exit(1)
            
        print("Loading FLUXNET observations...")
        fluxnet_variables = {'GPP', 'FPSN', 'NEE', 'ER', 'EFLX_LH_TOT', 'FSH'}
        vars_to_load = [var for var in mycase.postproc_vars if var in fluxnet_variables]
        
        if not vars_to_load:
            print(f"Error: No FLUXNET-compatible variables in {mycase.postproc_vars}")
            sys.exit(1)
            
        loaded_count = 0
        for var in vars_to_load:
            try:
                mycase.get_fluxnet_obs(site=mycase.site, fluxnet_var=var, myobsdir=obs_dir, 
                                      tstep=options.tstep, ystart=-1, yend=9999)
                loaded_count += 1
                print(f"✓ Loaded {var}")
            except Exception as e:
                print(f"✗ Failed to load {var}: {e}")
        
        if loaded_count == 0:
            print("Error: No observations could be loaded")
            sys.exit(1)
            
        # Check number of valid observations and filter variables
        print("\n=== Observation Quality Summary ===")
        total_valid = 0
        total_observations = 0
        valid_vars_for_mcmc = []
        
        for var in vars_to_load:
            if hasattr(mycase, 'obs') and var in mycase.obs:
                obs_data = np.array(mycase.obs[var])
                valid_count = np.sum(obs_data != -9999)
                total_count = len(obs_data)
                valid_pct = (valid_count / total_count * 100) if total_count > 0 else 0
                
                total_valid += valid_count
                total_observations += total_count
                
                print(f"{var}: {valid_count}/{total_count} valid observations ({valid_pct:.1f}%)")
                
                # Calculate expected number of observations in postproc period
                expected_postproc_obs = total_count  # default to total
                if hasattr(mycase, 'postproc_startyear') and hasattr(mycase, 'postproc_endyear'):
                    postproc_years = mycase.postproc_endyear - mycase.postproc_startyear + 1
                    if options.tstep == 'monthly':
                        expected_postproc_obs = postproc_years * 12
                    elif options.tstep == 'daily':
                        expected_postproc_obs = postproc_years * 366
                    elif options.tstep == 'yearly':
                        expected_postproc_obs = postproc_years
                
                if valid_count == 0:
                    print(f"  WARNING: No valid observations for {var} - SKIPPING from MCMC")
                elif valid_count == expected_postproc_obs:
                    valid_vars_for_mcmc.append(var)
                    print(f"  ✓ Including {var} in MCMC (complete data: {valid_count}/{expected_postproc_obs})")
                else:
                    print(f"  WARNING: Incomplete observations for {var} ({valid_count}/{expected_postproc_obs}) - SKIPPING from MCMC")
        
        overall_pct = (total_valid / total_observations * 100) if total_observations > 0 else 0
        print(f"\nOverall: {total_valid}/{total_observations} valid observations ({overall_pct:.1f}%)")
        print(f"Variables for MCMC: {valid_vars_for_mcmc}")
        
        if len(valid_vars_for_mcmc) == 0:
            print("ERROR: No variables have sufficient valid observations for MCMC!")
            print("This will cause MCMC to fail. Check observation files and time periods.")
            sys.exit(1)
        elif len(valid_vars_for_mcmc) < len(vars_to_load):
            print(f"Note: Using {len(valid_vars_for_mcmc)}/{len(vars_to_load)} variables for MCMC")
        
        print("=====================================\n")
            
        skipped = set(mycase.postproc_vars) - fluxnet_variables
        if skipped:
            print(f"Note: Skipped {skipped} (no FLUXNET equivalents)")
    
    # Check/train surrogate models
    def has_complete_surrogate(var):
        return (hasattr(mycase, 'surrogate') and var in mycase.surrogate and
                hasattr(mycase, 'pscaler') and var in mycase.pscaler and
                hasattr(mycase, 'yscaler') and var in mycase.yscaler)
    
    missing_surrogates = [var for var in mycase.postproc_vars if not has_complete_surrogate(var)]
    
    if missing_surrogates:
        print(f"Training surrogate models for: {missing_surrogates}")
        mycase.train_surrogate(mycase.postproc_vars)
    else:
        print("Using existing surrogate models")
    
    # Run MCMC
    #NOTE: different MCMC algorithms can be used here
    print("Starting MCMC parameter estimation...")
    print(f"Using variables: {valid_vars_for_mcmc}")
    parms = (np.array(mycase.ensemble_pmax) + np.array(mycase.ensemble_pmin)) / 2
    mycase.MCMC(parms, valid_vars_for_mcmc, 100000)
    
    # Save results
    mycase.create_pkl(outdir=mycase.OLMTdir+'/pklfiles/')
    print("✓ MCMC completed successfully")
    
elif (mycase.postproc_vars != []):
    #Train surrogate models
    mycase.train_surrogate(mycase.postproc_vars)

    #run GSA
    mycase.GSA(mycase.postproc_vars)
    mycase.plot_GSA(mycase.postproc_vars)

    #Save postprocessed output
    mycase.create_pkl(outdir=mycase.OLMTdir+'/pklfiles/')

    #run MCMC
    #Set intial values for parameters
    if (mycase.obs):
        parms=((np.array(mycase.ensemble_pmax)+np.array(mycase.ensemble_pmin))/2)
        #Run MCMC for the 2 varibles of interest
        #NOTE: different MCMC algorithms can be used here
        mycase.MCMC(parms, mycase.postproc_vars, 100000)

        #Save postprocessed output
        mycase.create_pkl(outdir=mycase.OLMTdir+'/pklfiles/')




