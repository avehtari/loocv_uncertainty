#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --time=0-03:00:00 --mem-per-cpu=5000
#SBATCH -o log/skew_b_n_%a.txt
#SBATCH --array=0-152

#module load anaconda3

python plot_skew_b_n.py $SLURM_ARRAY_TASK_ID
