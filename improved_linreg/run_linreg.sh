#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00 --mem-per-cpu=5000
#SBATCH -o log/linreg_%a.txt
#SBATCH --array=0-69

#module load anaconda3

python run_linreg.py $SLURM_ARRAY_TASK_ID
