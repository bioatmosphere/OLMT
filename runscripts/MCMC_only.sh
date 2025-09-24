#!/bin/bash -e

#SBATCH -t 6:00:00
#SBATCH -J mcmc_only
#SBATCH --nodes=3
#SBATCH -A CLI185
#SBATCH -p batch

#cd /gpfs/wolf2/cades/cli185/scratch/6lw/e3sm_cases/20250624_US-MOz_ICB20TRCNPRDCTCBC
#export LD_LIBRARY_PATH=/sw/baseline/spack-envs/base/opt/linux-rhel8-zen3/gcc-12.2.0/parallel-netcdf-1.12.3-wsupnghjnhkuibityz6k7womncpgssnb/lib:/sw/baseline/spack-envs/base/opt/linux-rhel8-zen3/gcc-12.2.0/netcdf-fortran-4.6.1-gqyszfzb4aroslnscdad4w56mxbbiqdh/lib:/sw/baseline/spack-envs/base/opt/linux-rhel8-zen3/gcc-12.2.0/netcdf-cxx-4.2-yz72q7wej6zfun22wssfggclhxe7ann4/lib:/sw/baseline/spack-envs/base/opt/linux-rhel8-zen3/gcc-12.2.0/netcdf-c-4.9.2-qp5zg6cuqzgu7knnp4qlgzszrd5a53qh/lib:/sw/baseline/spack-envs/base/opt/linux-rhel8-zen3/gcc-12.2.0/netlib-lapack-3.11.0-lpwyqsehj7wuz2i45umfhwa5ymv2dz5b/lib64:/sw/baseline/spack-envs/base/opt/linux-rhel8-zen3/gcc-12.2.0/openmpi-4.0.4-bxes2wvty3q7v55qep7hiuud6rocd4bl/lib:/sw/baseline/gcc/12.2.0/lib64:/sw/baseline/spack-envs/base/opt/linux-rhel8-zen3/gcc-12.2.0/hdf5-1.14.3-seiwvfmn4k7r5enbwfhyefa5a6nwsdwg/lib

#./preview_namelists

ulimit -n 2024
cd /autofs/nccsopen-svm1_home/6lw/models/OLMT/runscripts/..
#./manage_ensemble.py --case 20250708_US-Ho1_ICB20TRCNPRDCTCBC --MCMC_only --tstep yearly --use_available_obs
./manage_ensemble.py --case 20250910_US-MOz_ICB20TRCNPRDCTCBC --MCMC_only --tstep yearly --use_available_obs
