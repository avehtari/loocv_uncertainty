"""Analyse LOOCV results

First run m1_run.py s

"""

import sys, os

import numpy as np
from scipy import linalg
from scipy import stats


# grid
n_obs_s, beta_t_s = np.meshgrid(
    (10, 20, 50, 130, 250, 400),
    (0.0, 0.2, 0.5, 1, 2, 4, 8)
)
n_obs_s = n_obs_s.ravel()
beta_t_s = beta_t_s.ravel()
n_runs = len(n_obs_s)

# load results
loo_ti_A_s = []
loo_ti_B_s = []
for run_i in range(n_runs):
    res_file = np.load('res_1/{}.npz'.format(str(run_i).zfill(4)))
    # set confs for from the first results
    if run_i == 0:
        # fetch confs
        seed = res_file['seed']
        n_obs = res_file['n_obs']
        n_trial = res_file['n_trial']
        n_coef = res_file['n_coef']
        intercept = res_file['intercept']
        beta_t = res_file['beta_t']
        beta_other = res_file['beta_other']
        beta_intercept = res_file['beta_intercept']
        t_df_data = res_file['t_df_data']
    # fetch results
    loo_ti_A_s.append(res_file['loo_ti_A'])
    loo_ti_B_s.append(res_file['loo_ti_B'])
    # close file
    res_file.close()
