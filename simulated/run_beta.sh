#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --time=0-03:00:00 --mem-per-cpu=5000
#SBATCH -o log/beta_%a.txt
#SBATCH --array=0-71

#module load anaconda3

python run_beta.py $SLURM_ARRAY_TASK_ID
