#!/bin/bash

#SBATCH --job-name=collection
#SBATCH --output=collection.out
#SBATCH --error=collection.err
#SBATCH --time=24:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=56G

module load python/anaconda-2021.05

python3 collection.py