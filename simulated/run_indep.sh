#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --time=0-03:00:00 --mem-per-cpu=5000
#SBATCH -o log/indep_%a.txt
#SBATCH --array=0-35

#module load anaconda3

python run_indep.py $SLURM_ARRAY_TASK_ID
