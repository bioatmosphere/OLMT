import numpy as np
import os

def get_fluxnet_obs(self, site='US-UMB', tstep='monthly', ystart=-1, yend=9999, fluxnet_var='GPP', myobsdir='/observations/fluxnet'):
  """
  Load and process FLUXNET observational data for model validation.
  
  This function reads FLUXNET CSV files and extracts observational data for specified
  variables, time periods, and quality control criteria. It maps ELM variable names
  to FLUXNET variable names and handles uncertainty estimates.
  
  Parameters
  ----------
  site : str, optional
      FLUXNET site code (e.g., 'US-UMB'). Default is 'US-UMB'.
  tstep : str, optional
      Time step for data ('monthly', 'daily', or 'yearly'). Default is 'monthly'.
  ystart : int, optional
      Start year for data extraction. If -1, auto-detect from file. Default is -1.
  yend : int, optional
      End year for data extraction. If 9999, auto-detect from file. Default is 9999.
  fluxnet_var : str, optional
      ELM variable name to extract (e.g., 'GPP', 'NEE', 'FPSN'). Default is 'GPP'.
  myobsdir : str, optional
      Directory path containing FLUXNET observation files. Default is '/observations/fluxnet'.
      
  Returns
  -------
  None
      Populates self.obs and self.obs_err dictionaries with observational data.
      
  Notes
  -----
  - Only data with quality control flags > 0.8 (80%) are retained
  - Missing or low-quality data are marked with -9999
  - Variable mapping handles conversion between ELM and FLUXNET naming conventions
  - Supports monthly, daily, and yearly time steps
  """
  
  # Remove unused variables and fix spacing
  # myvars = ['TBOT', 'FSDS', 'WS', 'RAIN', 'VPD', 'NEE', 'GPP', 'ER', 'EFLX_LH_TOT', 'FSH']
  # myvars = ['FPSN', 'FSH', 'EFLX_LH_TOT']

  myobsfiles = os.listdir(myobsdir + '/' + tstep + '/')
 # variable_mapping = {
 #     'NEE': ('NEE_CUT_REF', 'NEE_CUT_SE'),   # Net Ecosystem Exchange
 #     'FPSN': ('GPP_NT_CUT_REF', 'GPP_NT_CUT_SE'),      # Gross Primary Production (photosynthesis)
 #     'GPP': ('GPP_NT_CUT_REF', 'GPP_NT_CUT_SE'),       # Gross Primary Production
 #     'ER': ('RECO_NT_CUT_REF', 'RECO_NT_CUT_SE'),      # Ecosystem Respiration
 #     'EFLX_LH_TOT': ('LE_F_MDS', 'LE_RANDUNC'),        # Latent Heat Flux
 #     'FSH': ('H_F_MDS', 'H_RANDUNC'),                  # Sensible Heat Flux
 #     'TBOT': ('TA_F_MDS', 'NA'),                       # Air Temperature
 #     'FSDS': ('SW_IN_F_MDS', 'NA'),                    # Shortwave Radiation
 #     'WS': ('WS_F', 'NA'),                             # Wind Speed
 #     'RAIN': ('P_F', 'NA'),                            # Precipitation
 #     'VPD': ('VPD_F_MDS', 'NA')                        # Vapor Pressure Deficit
 # }
  variable_mapping = {
      'NEE': ('NEE_CUT_REF', 'NEE_CUT_REF_JOINTUNC', '',''),   # Net Ecosystem Exchange
      'FPSN': ('GPP_NT_CUT_REF', 'GPP_NT_CUT_SE', '',''),      # Gross Primary Production (photosynthesis)
      'GPP': ('GPP_NT_CUT_REF', 'GPP_NT_CUT_SE', 'GPP_NT_CUT_05', 'GPP_NT_CUT_95'),       # Gross Primary Production
      'ER': ('RECO_NT_CUT_REF', 'RECO_NT_CUT_SE','RECO_NT_CUT_05','RECO_NT_CUT_95'),      # Ecosystem Respiration
      #'GPP': ('GPP_NT_VUT_REF', 'NEE_VUT_REF_RANDUNC', 'GPP_NT_VUT_25', 'GPP_NT_VUT_75'),       # Gross Primary Production
      #'ER': ('RECO_NT_VUT_REF', 'NEE_VUT_REF_RANDUNC','RECO_NT_VUT_25','RECO_NT_VUT_75'),      # Ecosystem Respiration
      'EFLX_LH_TOT': ('LE_F_MDS', 'LE_RANDUNC', '',''),        # Latent Heat Flux
      'FSH': ('H_F_MDS', 'H_RANDUNC', '',''),                  # Sensible Heat Flux
      'TBOT': ('TA_F_MDS', 'NA', '',''),                       # Air Temperature
      'FSDS': ('SW_IN_F_MDS', 'NA', '',''),                    # Shortwave Radiation
      'WS': ('WS_F', 'NA', '',''),                             # Wind Speed
      'RAIN': ('P_F', 'NA', '',''),                            # Precipitation
      'VPD': ('VPD_F_MDS', 'NA', '','')                        # Vapor Pressure Deficit
  }
  vars_elm = list(variable_mapping.keys())
  vars_fluxnet = [variable_mapping[var][0] for var in vars_elm]
  vars_unc = [variable_mapping[var][1] for var in vars_elm]
  vars_05 = [variable_mapping[var][2] for var in vars_elm]
  vars_95 = [variable_mapping[var][3] for var in vars_elm]

  vars_qc = (['NEE_CUT_REF_QC'] * 4 + 
             ['LE_F_MDS_QC', 'H_F_MDS_QC', 'TA_F_MDS_QC', 'SW_IN_F_MDS_QC', 'WS_F_QC', 'P_F_QC', 'VPD_F_MDS_QC'])

  #vars_qc = (['NEE_VUT_REF_QC'] * 4 +
  #           ['LE_F_MDS_QC', 'H_F_MDS_QC', 'TA_F_MDS_QC', 'SW_IN_F_MDS_QC', 'WS_F_QC', 'P_F_QC', 'VPD_F_MDS_QC'])

  # Initialize confidence interval dictionaries
  myobs_05 = {}
  myobs_95 = {}

  # ndaysm = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # unused variable
  if tstep == 'monthly':
      nstep = 12
  elif tstep == 'daily':
      nstep = 366
  elif tstep == 'yearly':
      nstep = 1

  for v in range(0, len(vars_elm)):
      if fluxnet_var == vars_elm[v]:
          vnum = v

  for f in myobsfiles:
      if site in f and '.csv' in f and 'FULLSET' in f:
          myobsfile = myobsdir + '/' + tstep + '/' + f
          if os.path.exists(myobsfile):
              print('Observation file: ' + myobsfile)
              thisrow = 0
              myobs_input = open(myobsfile)
              if ystart <= 0 and yend >= 9000:
                  print('Getting start and end year information from observation file')
                  for j in myobs_input:
                      if thisrow == 1:
                          ystart = int(j[0:4])
                      elif thisrow > 1:
                          yend = int(j[0:4])
                      thisrow = thisrow + 1
                  myobs_input.close()
                  nrows = thisrow - 1

              nrows = (yend - ystart + 1) * nstep
              myobs = np.zeros([nrows], float)
              myobs_err = np.zeros([nrows], float)
              myobs_05_array = np.zeros([nrows], float)
              myobs_95_array = np.zeros([nrows], float)
              myobs_in = open(myobsfile)
              thisrow = 0
              thisob = 0
              for j in myobs_in:
                  if thisrow == 0:
                      header = j.split(',')
                  else:
                      myvals = j.split(',')
                      thiscol = 0
                      if int(myvals[0][0:4]) >= ystart and int(myvals[0][0:4]) <= yend:
                          isgood = False
                          for h in header:
                              if h.strip() == vars_fluxnet[vnum]:
                                  tempob = float(myvals[thiscol])
                              if h.strip() == vars_unc[vnum]:
                                  tempob_err = float(myvals[thiscol])
                              if h.strip() == vars_05[vnum] and vars_05[vnum] != '':
                                  tempob_err05 = float(myvals[thiscol])
                              if h.strip() == vars_95[vnum] and vars_95[vnum] != '':
                                  tempob_err95 = float(myvals[thiscol])
                              if h.strip() == vars_qc[vnum]:
                                  if float(myvals[thiscol]) > 0.2:
                                      isgood = True  # only advance if quality flag > 80%
                              thiscol = thiscol + 1
                          if isgood:
                              myobs[thisob] = tempob
                              #myobs_err[thisob] = tempob_err * 10
                              percentile_range = tempob_err95 - tempob_err05
                              std_approx = percentile_range / 3.29
                              #std_approx = percentile_range / 1.349
                              #valid_approx = std_approx.dropna()
                              myobs_err[thisob] = std_approx
                              myobs_05_array[thisob] = tempob_err05
                              myobs_95_array[thisob] = tempob_err95
                          else:
                              myobs[thisob] = -9999
                              myobs_err[thisob] = -9999
                              myobs_05_array[thisob] = -9999
                              myobs_95_array[thisob] = -9999
                          thisob = thisob + 1
                  thisrow = thisrow + 1
              # Check if postproc_startyear and postproc_endyear are defined for subsetting
              if hasattr(self, 'postproc_startyear') and hasattr(self, 'postproc_endyear'):
                  # Calculate subset indices based on postproc years
                  postproc_start_idx = max(0, (self.postproc_startyear - ystart) * nstep)
                  postproc_end_idx = min(len(myobs), (self.postproc_endyear - ystart + 1) * nstep)
                  
                  # Subset the arrays
                  self.obs[vars_elm[vnum]] = myobs[postproc_start_idx:postproc_end_idx]
                  self.obs_err[vars_elm[vnum]] = myobs_err[postproc_start_idx:postproc_end_idx]
                  myobs_05[vars_elm[vnum]] = myobs_05_array[postproc_start_idx:postproc_end_idx]
                  myobs_95[vars_elm[vnum]] = myobs_95_array[postproc_start_idx:postproc_end_idx]

                  print(f"Subset observations from {self.postproc_startyear} to {self.postproc_endyear}")
                  print(f"Using indices {postproc_start_idx}:{postproc_end_idx} from total length {len(myobs)}")
              else:
                  # Use full arrays if postproc years not defined
                  self.obs[vars_elm[vnum]] = myobs
                  self.obs_err[vars_elm[vnum]] = myobs_err
                  myobs_05[vars_elm[vnum]] = myobs_05_array
                  myobs_95[vars_elm[vnum]] = myobs_95_array

  return myobs_05, myobs_95
