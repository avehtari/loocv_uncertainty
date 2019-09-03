#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --time=0-01:00:00 --mem-per-cpu=5000
#SBATCH -o log/%a.txt
#SBATCH --array=0-41

#module load anaconda3

python m1_run.py $SLURM_ARRAY_TASK_ID
