#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --time=5-00:00:00 --mem-per-cpu=5000
#SBATCH -p batch
#SBATCH -o log/%a.txt
#SBATCH --array=0-199

#module load boost
#module load R

Rscript loo_run.R $SLURM_ARRAY_TASK_ID
