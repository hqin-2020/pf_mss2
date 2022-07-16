#!/bin/bash

#SBATCH --job-name=pf_mss
#SBATCH --output=pf_mss.%A_%a.out
#SBATCH --error=pf_mss.%A_%a.err
#SBATCH --time=0-0:30:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --mem-per-cpu=2000
#SBATCH --array=1-150
#SBATCH --constraint=edr

module load python/anaconda-2021.05

python3 multi_seeds.py 
python3 Run_Aso1_0_CPU_$SLURM_ARRAY_TASK_ID.py  