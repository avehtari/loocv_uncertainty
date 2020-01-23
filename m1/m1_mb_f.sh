#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --time=2-00:00:00 --mem-per-cpu=5000
#SBATCH -o log/mboot_fixed_%a.txt
#SBATCH --array=0-419

#module load anaconda3

python m1_run_mboot.py $SLURM_ARRAY_TASK_ID fixed
