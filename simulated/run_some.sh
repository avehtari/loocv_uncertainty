#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00 --mem-per-cpu=5000
#SBATCH -o log/some_%a.txt
#SBATCH --array=0-55

#module load anaconda3

python run_some.py $SLURM_ARRAY_TASK_ID
