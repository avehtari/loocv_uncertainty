#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --time=5-00:00:00 --mem-per-cpu=5000
#SBATCH -o log.txt

#module load anaconda3

python m2_analytic_plots_n.py
