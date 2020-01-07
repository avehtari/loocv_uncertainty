#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --time=0-01:00:00 --mem-per-cpu=5000
#SBATCH -o log/loo_unfixed_%a.txt
#SBATCH --array=0-25

#module load anaconda3

python m1_run_loo.py $SLURM_ARRAY_TASK_ID unfixed
