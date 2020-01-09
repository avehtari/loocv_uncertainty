"""Analyse LOOCV results

First run results with m1_run.py

"""

import sys, os

import numpy as np
from scipy import linalg
from scipy import stats

import matplotlib.pyplot as plt

from m1_problem import *


# ============================================================================
# config

# fixed or unfixed sigma
fixed_sigma2_m = False


# ============================================================================

if fixed_sigma2_m:
    folder_name = 'fixed'
else:
    folder_name = 'unfixed'


# ============================================================================
# As a function of ...

# run_i_s = n_obs_idxs
# run_i_s = sigma2_d_idxs
# run_i_s = beta_t_idxs
run_i_s = prc_out_idxs


n_probls = len(run_i_s)

# load results
res_A = np.empty(n_probls, dtype=object)
res_B = np.empty(n_probls, dtype=object)
res_test_A = np.empty(n_probls, dtype=object)
res_test_B = np.empty(n_probls, dtype=object)
for probl_i in range(n_probls):
    run_i = run_i_s[probl_i]
    res_file = np.load(
        'res_1/{}/{}.npz'
        .format(folder_name, str(run_i).zfill(4))
    )
    # fetch results
    res_A[probl_i] = res_file['loo_ti_A']
    res_B[probl_i] = res_file['loo_ti_B']
    res_test_A[probl_i] = res_file['test_t_A']
    res_test_B[probl_i] = res_file['test_t_B']
    # close file
    res_file.close()

# calc some true targets
sums_s = np.zeros((n_probls, n_trial))
mean_s = np.zeros((n_probls))
var_s = np.zeros((n_probls))
for probl_i in range(n_probls):
    sums_s[probl_i] = np.sum(res_A[probl_i] - res_B[probl_i], axis=-1)
    mean_s[probl_i] = np.mean(sums_s[probl_i])
    var_s[probl_i] = np.var(sums_s[probl_i], ddof=1)
coef_var = np.sqrt(var_s)/mean_s

# calc corr
cor_s = np.zeros((n_probls, n_trial))
for probl_i in range(n_probls):
    for trial_i in range(n_trial):
        cor_s[probl_i, trial_i] = np.corrcoef(
            res_A[probl_i][trial_i], res_B[probl_i][trial_i])[0, 1]

# plot
fig, axes = plt.subplots(n_probls, sharex=True)
for ax, probl_i in zip(axes, range(n_probls)):
    ax.hist(cor_s[probl_i], 20)

fig = plt.figure()
plt.plot(np.mean(cor_s, axis=-1), coef_var, '.')
