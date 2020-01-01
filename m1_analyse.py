"""Analyse LOOCV results

First run results with m1_run.py

"""

import sys, os

import numpy as np
from scipy import linalg
from scipy import stats

import matplotlib.pyplot as plt


# ============================================================================
# config

# fixed or unfixed sigma
fixed_sigma2_m = False

# cut results
n_trial_cut = True
n_trial_cut_n = 400

# ============================================================================

if fixed_sigma2_m:
    folder_name = 'fixed'
else:
    folder_name = 'unfixed'

# load first result as a sample for parameters
res_file = np.load('res_1/{}/0000.npz'.format(folder_name))
# get params
seed = res_file['seed']
sigma2_m = res_file['sigma2_m']
sigma2_d = res_file['sigma2_d']
n_obs = res_file['n_obs']
prc_out = res_file['prc_out']
outlier_dev = res_file['outlier_dev']
suffle_obs = res_file['suffle_obs']
n_trial = res_file['n_trial']
n_dim = res_file['n_dim']
intercept = res_file['intercept']
beta_t = res_file['beta_t']
beta_other = res_file['beta_other']
beta_intercept = res_file['beta_intercept']
# close
res_file.close()
# set grid
n_obs_s, beta_t_s, prc_out_s, sigma2_d_s = np.meshgrid(
    n_obs,
    beta_t,
    prc_out,
    sigma2_d
)
# n_obs_s = n_obs_s.ravel()
# beta_t_s = beta_t_s.ravel()
# prc_out_s = prc_out_s.ravel()
# sigma2_d_s = sigma2_d_s.ravel()
n_runs = n_obs_s.size

if n_trial_cut:
    n_trial = n_trial_cut_n

# load results
res_A = np.empty(n_obs_s.shape, dtype=object)
res_B = np.empty(n_obs_s.shape, dtype=object)
for run_i in range(n_runs):
    run_i_idx = np.unravel_index(run_i, n_obs_s.shape)
    res_file = np.load(
        'res_1/{}/{}.npz'
        .format(folder_name, str(run_i).zfill(4))
    )
    # fetch results
    res_A[run_i_idx] = res_file['loo_ti_A']
    res_B[run_i_idx] = res_file['loo_ti_B']
    # cut results
    if n_trial_cut:
        res_A[run_i_idx] = res_A[run_i_idx][:n_trial_cut_n]
        res_B[run_i_idx] = res_B[run_i_idx][:n_trial_cut_n]
    # close file
    res_file.close()

# ============================================================================
# test some plots

# pick one
idx = (2, 3, 1, 1)
res_A_i = res_A[idx]
res_B_i = res_B[idx]

plt.hist(res_A_i.sum(axis=1)-res_B_i.sum(axis=1))


# vary beta_t
idx = (2, None, 1, 1)
# ...
