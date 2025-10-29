#!/usr/bin/env python

import os, sys, csv, time, math
import numpy as np
import datetime

def read_parm_list(self, parm_list=''):
    """Read the parameter list file

    Supports two formats:
    1. Legacy format (4 columns): parameter_name pft min max
    2. Extended format (5+ columns): parameter_name pft dist_type param1 param2 [param3 param4]

    Supported distributions:
    - uniform: min max
    - normal: mean std [min max] (truncated if bounds provided)
    - lognormal: log_mean log_std [min max]
    - beta: alpha beta min max
    - gamma: shape scale [min max]
    """

    os.chdir(self.OLMTdir)
    if (os.path.exists(parm_list)):
        myfile = open(parm_list,'r')
        self.ensemble_parms=[]
        self.ensemble_pfts=[]
        self.ensemble_pmin=[]
        self.ensemble_pmax=[]
        self.ensemble_dist_type=[]  # Distribution type for each parameter
        self.ensemble_dist_params=[]  # Additional distribution parameters

        for s in myfile:
            if (not '#' in s[0:3] and s.strip()):  # Skip comments and empty lines
              # Remove inline comments
              if '#' in s:
                  s = s[:s.index('#')]
              vals = s.split()

              if len(vals) < 4:
                  continue  # Skip malformed lines

              self.ensemble_parms.append(vals[0].strip())
              self.ensemble_pfts.append(int(vals[1].strip()))

              # Detect format: if vals[2] is a known distribution type, use extended format
              known_dists = ['uniform', 'normal', 'truncnorm', 'lognormal', 'beta', 'gamma']

              if len(vals) >= 5 and vals[2].lower() in known_dists:
                  # Extended format: parameter pft dist_type param1 param2 [param3 param4]
                  dist_type = vals[2].lower()
                  self.ensemble_dist_type.append(dist_type)

                  if dist_type == 'uniform':
                      min_val = float(vals[3])
                      max_val = float(vals[4])
                      self.ensemble_pmin.append(min_val)
                      self.ensemble_pmax.append(max_val)
                      self.ensemble_dist_params.append({'min': min_val, 'max': max_val})

                  elif dist_type in ['normal', 'truncnorm']:
                      mean = float(vals[3])
                      std = float(vals[4])
                      if len(vals) >= 7:
                          # Truncated normal with bounds
                          min_val = float(vals[5])
                          max_val = float(vals[6])
                      else:
                          # Use mean ± 4*std as default bounds
                          min_val = mean - 4 * std
                          max_val = mean + 4 * std
                      self.ensemble_pmin.append(min_val)
                      self.ensemble_pmax.append(max_val)
                      self.ensemble_dist_params.append({
                          'mean': mean, 'std': std, 'min': min_val, 'max': max_val
                      })

                  elif dist_type == 'lognormal':
                      log_mean = float(vals[3])
                      log_std = float(vals[4])
                      if len(vals) >= 7:
                          min_val = float(vals[5])
                          max_val = float(vals[6])
                      else:
                          # Use exponential of log bounds
                          min_val = np.exp(log_mean - 4 * log_std)
                          max_val = np.exp(log_mean + 4 * log_std)
                      self.ensemble_pmin.append(min_val)
                      self.ensemble_pmax.append(max_val)
                      self.ensemble_dist_params.append({
                          'log_mean': log_mean, 'log_std': log_std, 'min': min_val, 'max': max_val
                      })

                  elif dist_type == 'beta':
                      alpha = float(vals[3])
                      beta = float(vals[4])
                      min_val = float(vals[5])
                      max_val = float(vals[6])
                      self.ensemble_pmin.append(min_val)
                      self.ensemble_pmax.append(max_val)
                      self.ensemble_dist_params.append({
                          'alpha': alpha, 'beta': beta, 'min': min_val, 'max': max_val
                      })

                  elif dist_type == 'gamma':
                      shape = float(vals[3])
                      scale = float(vals[4])
                      if len(vals) >= 7:
                          min_val = float(vals[5])
                          max_val = float(vals[6])
                      else:
                          # Use gamma distribution percentiles as bounds
                          min_val = 0.0
                          max_val = shape * scale * 4  # Rough upper bound
                      self.ensemble_pmin.append(min_val)
                      self.ensemble_pmax.append(max_val)
                      self.ensemble_dist_params.append({
                          'shape': shape, 'scale': scale, 'min': min_val, 'max': max_val
                      })

              else:
                  # Legacy format: parameter pft min max (uniform distribution)
                  self.ensemble_dist_type.append('uniform')
                  min_val = float(vals[2].strip())
                  max_val = float(vals[3].strip())
                  self.ensemble_pmin.append(min_val)
                  self.ensemble_pmax.append(max_val)
                  self.ensemble_dist_params.append({'min': min_val, 'max': max_val})

        myfile.close()
    else:
        print('parm_list file '+parm_list+' does not exist.  Exiting')
        sys.exit(1)
    self.nparms_ensemble = len(self.ensemble_parms)

#def get_default_parms(self):
#    parm_file = Dataset(self.parm_file,'r')
#    parms_def=[]
#    for p in self.ensemble_parms:
#        parms_def.append(parm_file[p][
    

def create_samples(self,sampletype='monte_carlo',nsamples=100,parm_list=''):
    """Create the samples file
    
    Parameters
    ----------
    sampletype : str
        Type of sampling to do. Currently only 'monte_carlo' is supported.
    nsamples : int
        Number of samples to create.
    parm_list : str
        Path to the parameter list file. If not provided, the default parameter list will be used.
    """

    self.nsamples=nsamples
    self.samples=np.zeros((self.nparms_ensemble,self.nsamples), float)
    for i in range(0,self.nsamples):
        for j in range(0,self.nparms_ensemble):
            self.samples[j,i] = self.ensemble_pmin[j]+(self.ensemble_pmax[j]- \
                    self.ensemble_pmin[j])*np.random.rand(1)
    self.ensemble_file = 'parm_samples/mcsamples_'+self.caseid+'_'+str(self.nsamples)+'.txt'
    os.system('mkdir -p parm_samples')
    np.savetxt(self.ensemble_file,np.transpose(self.samples))

def create_ensemble_script(self, walltime=6):
    """Create the PBS script we will submit to run the ensemble

    """

    os.chdir(self.casedir)
    #Get the LD_LIBRARY_PATH from software environment
    softenv = open('software_environment.txt','r')
    for s in softenv:
        if s.split('=')[0].strip() == 'LD_LIBRARY_PATH':
            ldpath = s.split('=')[1].strip()
    softenv.close()
    self.npernode=int(self.xmlquery('MAX_TASKS_PER_NODE'))
    nnodes = int(np.ceil((self.np_ensemble*self.np)/self.npernode))

    myfile = open('case.submit_ensemble','w')
    myfile.write('#!/bin/bash -e\n\n')
    if (self.queue == 'debug'):
        walltime=2
    if ('pm-cpu' in self.machine):
        myfile.write('#SBATCH -t '+str(walltime)+'\n')
        myfile.write('#SBATCH --constraint=cpu\n')
    else:
        myfile.write('#SBATCH -t '+str(walltime)+':00:00\n')
    myfile.write('#SBATCH -J '+self.casename+'\n')
    myfile.write('#SBATCH --nodes='+str(nnodes)+'\n')  
    if (self.project != ''):
        myfile.write('#SBATCH -A '+self.project+'\n')
    myfile.write('#SBATCH -p '+self.queue+'\n')
    myfile.write('cd '+self.caseroot+'/'+self.casename+'\n')
    myfile.write('export LD_LIBRARY_PATH='+ldpath+'\n\n')
    myfile.write('./preview_namelists\n\n')
    myfile.write('ulimit -n '+str(self.nsamples+1024)+'\n')
    myfile.write('cd '+self.OLMTdir+'\n')
    myfile.write('./manage_ensemble.py --case '+self.casename+'\n')
    myfile.close()  
    os.system('chmod u+x case.submit_ensemble')
    self.rundir_UQ = self.runroot+'/UQ/'+self.casename

def create_multisite_script(self,sites,scriptdir, walltime=6):
    """Create the PBS script we will submit to run multiple sites.

    Parameters
    ----------
    sites : list
        List of sites to run.
    scriptdir : str
        Directory to write the script to.
    walltime : int
        Walltime for the job in hours.

    Returns
    -------
    str
        Path to the created script.
    """
    
    os.chdir(self.casedir)
    #Get the LD_LIBRARY_PATH from software environment
    softenv = open('software_environment.txt','r')
    for s in softenv:
        if s.split('=')[0].strip() == 'LD_LIBRARY_PATH':
            ldpath = s.split('=')[1].strip()
    softenv.close()
    self.npernode=int(self.xmlquery('MAX_TASKS_PER_NODE'))
    if sites[0] != '':
        nnodes = int(np.ceil(len(sites)/self.npernode))
    else:
        nnodes = int(np.ceil(self.np/self.npernode))
    fname = self.casename.replace('_'+self.site,'')+'.sh'
    myfile = open(fname,'w')
    myfile.write('#!/bin/bash -e\n\n')
    if (self.queue == 'debug'):
        walltime=2
    if ('pm-cpu' in self.machine):
        myfile.write('#SBATCH -t '+str(walltime)+'\n')
        myfile.write('#SBATCH --constraint=cpu\n')
    else:
        myfile.write('#SBATCH -t '+str(walltime)+':00:00\n')
    myfile.write('#SBATCH -J '+self.casename.replace('_'+self.site,'')+'\n')
    myfile.write('#SBATCH --nodes='+str(nnodes)+'\n')
    if (self.project != ''):
        myfile.write('#SBATCH -A '+self.project+'\n')
    myfile.write('#SBATCH -p '+self.queue+'\n')
    myfile.write('cd '+self.caseroot+'/'+self.casename+'\n')
    myfile.write('export LD_LIBRARY_PATH='+ldpath+'\n\n')
    for s in sites:
      myfile.write('cd '+self.caseroot+'/'+self.casename.replace(sites[0],s)+'\n')
      myfile.write('./preview_namelists\n')
      myfile.write('cd '+self.runroot+'/'+self.casename.replace(sites[0],s)+'/run\n')
      myfile.write('mkdir -p timing/checkpoints\n')
      #restart file options
      for key in self.case_options.keys():
        if ('restart_' in key):
            var   = key[8:]
            value = str(self.case_options[key])
            if ('*' in value or '+' in value):
                operator=value[0]
                value=value[1:]
                myfile.write('python '+self.OLMTdir+'/modify_netcdf.py --filename '+ \
                    self.finidat+' --var '+var+' --val '+value+ \
                    ' --operator "'+operator+'"\n')
            else:
                myfile.write('python '+self.OLMTdir+'/modify_netcdf.py --filename '+ \
                    self.finidat+' --var '+var+' --val '+value+'\n')
      if (self.noslurm):
        myfile.write(self.exeroot+'/e3sm.exe > '+self.rundir+'/e3sm_log.txt &\n\n')
      else:
        myfile.write('srun -n '+str(self.np)+' -c 1 '+self.exeroot+'/e3sm.exe > '+ \
                self.rundir+'/e3sm_log.txt &\n\n')
    myfile.write('wait\n')
    myfile.close()
    os.system('chmod u+x '+fname)
    return os.path.abspath('./'+fname)

def ensemble_copy(self, ens_num):
    """Create the ensemble run directory

    Method:
        1. This function creates a new ensemble case directory based on the original case by
        copying the original case files.
        
        2. It modifies the necessary files to set the parameters for the ensemble run.

    TODO:
        - Add support for more complex parameter modifications
        

    Parameters
    ----------
        ens_num : int
            Ensemble number to create.
    """
    
    gst=str(100000+int(ens_num))

    # create ensemble directory from original case 
    orig_dir = str(os.path.abspath(self.runroot)+'/'+self.casename+'/run')
    ens_dir  = str(os.path.abspath(self.runroot)+'/UQ/'+self.casename+'/g'+gst[1:])
            
    os.system('mkdir -p '+ens_dir+'/timing/checkpoints')
    os.system('rm -f '+ens_dir+'/*.log.* '+ens_dir+'/*.nc '+ens_dir+'/rpointer*')
    os.system('cp  '+orig_dir+'/*_in* '+ens_dir)
    os.system('cp  '+orig_dir+'/*nml '+ens_dir)
    if (not ('CB' in self.casename)):
        os.system('cp  '+orig_dir+'/*stream* '+ens_dir)
    os.system('cp  '+orig_dir+'/*.rc '+ens_dir)
    os.system('cp  '+orig_dir+'/surf*.nc '+ens_dir)
    os.system('cp  '+orig_dir+'/domain*.nc '+ens_dir)
    os.system('cp  '+orig_dir+'/*para*.nc '+ens_dir)


    # loop through all filenames, change directories in namelists, change parameter values
    for f in os.listdir(ens_dir):
        if (os.path.isfile(ens_dir+'/'+f) and (f[-2:] == 'in' or f[-3:] == 'nml' or 'streams' in f)):
            myinput=open(ens_dir+'/'+f)
            myoutput=open(ens_dir+'/'+f+'.tmp','w')
            for s in myinput:
                if ('fates_paramfile' in s):
                    paramfile_orig = ((s.split()[2]).strip("'"))
                    if (paramfile_orig[0:2] == './'):
                        paramfile_orig = orig_dir+'/'+paramfile_orig[2:]
                    paramfile_new  = ens_dir+'/fates_params_'+gst[1:]+'.nc'
                    os.system('cp '+paramfile_orig+' '+paramfile_new)
                    os.system('nccopy -3 '+paramfile_new+' '+paramfile_new+'_tmp')
                    os.system('mv '+paramfile_new+'_tmp '+paramfile_new)
                    myoutput.write(" fates_paramfile = '"+paramfile_new+"'\n")
                    fates_paramfile = ens_dir+'/fates_params_'+gst[1:]+'.nc'
                elif ('paramfile' in s):
                    paramfile_orig = ((s.split()[2]).strip("'"))
                    if (paramfile_orig[0:2] == './'):
                        paramfile_orig = orig_dir+'/'+paramfile_orig[2:]
                    paramfile_new  = ens_dir+'/clm_params_'+gst[1:]+'.nc'
                    os.system('cp '+paramfile_orig+' '+paramfile_new)
                    os.system('nccopy -3 '+paramfile_new+' '+paramfile_new+'_tmp')
                    os.system('mv '+paramfile_new+'_tmp '+paramfile_new)
                    myoutput.write(" paramfile = '"+paramfile_new+"'\n")
                    pftfile = ens_dir+'/clm_params_'+gst[1:]+'.nc'
                elif ('ppmv' in s and 'co2' in self.ensemble_parms):
                    myoutput.write(" co2_ppmv = "+str(parm_values[pnum_co2])+'\n')
                elif ('fsoilordercon' in s):
                    CNPfile_orig = ((s.split()[2]).strip("'"))
                    if (CNPfile_orig[0:2] == './'):
                        CNPfile_orig  = orig_dir+'/'+CNPfile_orig[2:]
                    CNPfile_new  = ens_dir+'/CNP_parameters_'+gst[1:]+'.nc'
                    os.system('cp '+CNPfile_orig+' '+CNPfile_new)
                    os.system('nccopy -3 '+CNPfile_new+' '+CNPfile_new+'_tmp')
                    os.system('mv '+CNPfile_new+'_tmp '+CNPfile_new)
                    myoutput.write(" fsoilordercon = '"+CNPfile_new+"'\n")
                    CNPfile = ens_dir+'/CNP_parameters_'+gst[1:]+'.nc'
                elif ('fsurdat =' in s):
                    surffile_orig = ((s.split()[2]).strip("'"))
                    if (surffile_orig[0:2] == './'):
                        surffile_orig = orig_dir+'/'+surffile_orig[2:]
                    surffile_new = ens_dir+'/surfdata_'+gst[1:]+'.nc'
                    os.system('cp '+surffile_orig+' '+surffile_new)
                    os.system('nccopy -3 '+surffile_new+' '+surffile_new+'_tmp')
                    os.system('mv '+surffile_new+'_tmp '+surffile_new)
                    myoutput.write(" fsurdat = '"+surffile_new+"'\n")
                    surffile = ens_dir+'/surfdata_'+gst[1:]+'.nc'
                elif ('finidat = ' in s and self.has_finidat):
                    finidat_file_path = os.path.abspath(self.runroot)+'/UQ/'+self.dependcase+'/g'+gst[1:]
                    finidat_file_name = self.finidat.split('/')[-1]
                    #finidat_file_orig = self.finidat
                    finidat_file_new  = finidat_file_path+'/'+finidat_file_name 
                    #if ('ad_spinup' in self.dependcase): 
                    #        os.system('python adjust_restart.py --rundir '+finidat_file_path+' --casename '+ \
                    #            self.dependcase)
                    #os.system('cp '+finidat_file_orig+' '+finidat_file_new)
                    myoutput.write(" finidat = '"+finidat_file_new+"'\n")
                    #Make any requested restart modifications
                    for key in self.case_options.keys():
                        if ('restart_' in key):
                            var   = key[8:]
                            value = self.case_options[key]
                            ncval = self.getncvar(finidat_file_new, var)
                            if ('*' in value):
                                value = value*ncval
                            if ('+' in value):
                                value = value+ncval
                            self.putncvar(finidat_file_new, var, value)
                elif ('logfile =' in s):
                    #Get the current date and time
                    now = datetime.datetime.now()
                    #Format the date and time in %y%m%d-%H%M%S format
                    date_string = now.strftime("%y%m%d-%H%M%S")
                    myoutput.write(s.replace('`date +%y%m%d-%H%M%S`',date_string))
                else:
                    myoutput.write(s.replace(orig_dir,ens_dir))
            myoutput.close()
            myinput.close()
            os.system(' mv '+ens_dir+'/'+f+'.tmp '+ens_dir+'/'+f)
    
    # loop through all parameters of interest (PoI) and set them in the files
    CNP_parms = ['ks_sorption', 'r_desorp', 'r_weather', 'r_adsorp', 'k_s1_biochem', 'smax', 'k_s3_biochem', \
             'r_occlude', 'k_s4_biochem', 'k_s2_biochem']

    fates_seed_zeroed=[False,False]
    pnum=0
    parm_values = self.samples[:,ens_num-1]
    parm_indices = self.ensemble_pfts
    for p in self.ensemble_parms:
        # ...
        if ('INI' in p):
            if ('BGC' in self.casename):
                scalevars = ['soil3c_vr','soil3n_vr','soil3p_vr']
            else:
                scalevars = ['soil4c_vr','soil4n_vr','soil4p_vr']
            sumvars = ['totsomc','totsomp','totcolc','totcoln','totcolp']
            for v in scalevars:
                myvar = self.getncvar(finidat_file_new, v)
                myvar = parm_values[pnum] * myvar
                ierr = self.putncvar(finidat_file_new, v, myvar)
        # parameters in surffile 
        elif (p == 'lai'):
            myfile = surffile
            param = self.getncvar(myfile, 'MONTHLY_LAI')
            param[:,:,:,:] = parm_values[pnum]
            ierr = self.putncvar(myfile, 'MONTHLY_LAI', param)
        # parameters in the parameter or CNP_parms file 
        elif (p != 'co2'):
            if (p in CNP_parms):
                myfile= CNPfile
            elif ('fates' in p):
                myfile = fates_paramfile
            else:
                myfile = pftfile
            # get the parameter variable from the netCDF file
            param = self.getncvar(myfile,p)
            if (('fates_prt' in p and 'stoich' in p) or ('fates_turnover' in p and 'retrans' in p)):
                #this is a 2D parameter.
                param[parm_indices[pnum] % 12 , parm_indices[pnum] / 12] = parm_values[pnum]
                param[parm_indices[pnum] % 12 , parm_indices[pnum] / 12] = parm_values[pnum]
            elif ('fates_hydr_p50_node' in p or 'fates_hydr_avuln_node' in p or 'fates_hydr_kmax_node' in p or \
                    'fates_hydr_pitlp_node' in p or 'fates_hydr_thetas_node' in p):
                param[parm_indices[pnum] / 12 , parm_indices[pnum] % 12] = parm_values[pnum]
                param[parm_indices[pnum] / 12 , parm_indices[pnum] % 12] = parm_values[pnum]
            elif ('fates_leaf_long' in p or 'fates_leaf_vcmax25top' in p):
                param[0,parm_indices[pnum]] = parm_values[pnum]
            #elif (p == 'fates_seed_alloc'):
            #    if (not fates_seed_zeroed[0]):
            #       param[:]=0.
            #       fates_seed_zeroed[0]=True
            #    param[parm_indices[pnum]] = parm_values[pnum]
            #elif (p == 'fates_seed_alloc_mature'):
            #    if (not fates_seed_zeroed[1]):
            #       param[:]=0.
            #       fates_seed_zeroed[1]=True
            #    param[parm_indices[pnum]] = parm_values[pnum]             
            elif (p == 'dayl_scaling' or p == 'vcmaxse'):
                os.system('ncap2 -O -s "'+p+' = flnr" '+myfile+' '+myfile)
                print('Creting netcdf variable for '+p)
                param = self.getncvar(myfile,'flnr')
                param[:] = parm_values[pnum]
            elif (p == 'psi50'):
                param[:,parm_indices[pnum]] = parm_values[pnum]
            elif (parm_indices[pnum] > 0):
                param[parm_indices[pnum]] = parm_values[pnum]
            elif (parm_indices[pnum] == 0):
                try:
                    param[:] = parm_values[pnum]
                except:
                    param = parm_values[pnum]
            # put the modified parameter back into the netCDF file
            ierr = self.putncvar(myfile, p, param, addvar=True)
            
            # ensure some TAM parameters sum to one
            # this assumes _flab followed by _fcel
            if (p == 'frt_fcel'):
                param=self.getncvar(myfile, 'frt_flig')
                param[parm_indices[pnum]]=1.0-parm_values[pnum]-parm_values[pnum-1]
                ierr = self.putncvar(myfile, 'frt_flig', param)
            if (p == 'fra_fcel'):
                param=self.getncvar(myfile, 'fra_flig')
                param[parm_indices[pnum]]=1.0-parm_values[pnum]-parm_values[pnum-1]
                ierr = self.putncvar(myfile, 'fra_flig', param)
            if (p == 'frm_fcel'):
                param=self.getncvar(myfile, 'frm_flig')
                param[parm_indices[pnum]]=1.0-parm_values[pnum]-parm_values[pnum-1]
                ierr = self.putncvar(myfile, 'frm_flig', param)
            # this assumes froott_leaf followed by froota_leaf
            if (p == 'froota_leaf'):
                param=self.getncvar(myfile, 'frootm_leaf')
                param[parm_indices[pnum]]=1.0-parm_values[pnum]-parm_values[pnum-1]
                ierr = self.putncvar(myfile, 'frootm_leaf', param)
        pnum = pnum+1
  
    #ensure FATES seed allocation paramters sum to one
    #if (fates_seed_zeroed[0]):
    #  param = self.getncvar(myfile,'fates_seed_alloc')
    #  param2 = self.getncvar(myfile,'fates_seed_alloc_mature')
    #  for i in range(0,12):
    #    if (param[i] + param2[i] > 1.0):
    #      sumparam= param[i]+param2[i]
    #      param[i]  = param[i]/sumparam
    #      param2[i] = param2[i]/sumparam
    #  ierr = self.putncvar(myfile, 'fates_seed_alloc', param)      
    #  ierr = self.putncvar(myfile, 'fates_seed_alloc_mature', param2)
