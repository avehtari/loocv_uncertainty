#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --time=0-03:00:00 --mem-per-cpu=5000
#SBATCH -o log/zscore_skew_mu_b_%a.txt
#SBATCH --array=0-104

#module load anaconda3

python plot_zscore_skew_mu_b.py $SLURM_ARRAY_TASK_ID
