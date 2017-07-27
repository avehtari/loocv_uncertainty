#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --time=0-03:00:00 --mem-per-cpu=5000
#SBATCH -p batch
#SBATCH -o log_loo_sgt/%a.txt
#SBATCH --array=0-31

Rscript loo_run_sgt.R $SLURM_ARRAY_TASK_ID
