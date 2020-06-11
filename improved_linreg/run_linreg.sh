#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --time=3-00:00:00 --mem-per-cpu=5000
#SBATCH -o log/linreg_%a.txt
#SBATCH --array=0-251

#module load anaconda3

python run_linreg.py $SLURM_ARRAY_TASK_ID
