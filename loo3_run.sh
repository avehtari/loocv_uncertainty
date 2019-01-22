#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --time=3-00:00:00 --mem-per-cpu=5000
#SBATCH -p batch
#SBATCH -o log3/%a.txt
#SBATCH --array=0-239

#module load boost
#module load R

Rscript loo3_run.R $SLURM_ARRAY_TASK_ID
