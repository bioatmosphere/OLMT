#!/bin/bash -e

#SBATCH -t 2:00:00
#SBATCH -J mcmc_all20
#SBATCH --nodes=1
#SBATCH -A CLI185
#SBATCH -p batch

ulimit -n 2024
cd /autofs/nccsopen-svm1_home/6lw/models/OLMT/runscripts

uv run python mcmc_all20_AU-Tum.py
