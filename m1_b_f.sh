#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00 --mem-per-cpu=5000
#SBATCH -o log/boot_fixed_%a.txt
#SBATCH --array=0-575

#module load anaconda3

python m1_run_boot.py $SLURM_ARRAY_TASK_ID fixed
