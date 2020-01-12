"""Analyse LOOCV results

First run results with m1_run.py

"""

import numpy as np
from scipy import linalg
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns


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

run_i_s = n_obs_idxs
var_name = 'n_obs'
var_vals = n_obs_s

# run_i_s = sigma2_d_idxs
# var_name = 'sigma2_d'
# var_vals = sigma2_d_s

# run_i_s = beta_t_idxs
# var_name = 'beta_t'
# var_vals = beta_t_s

# run_i_s = prc_out_idxs
# var_name = 'prc_out'
# var_vals = prc_out_s



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

# calc some normally obtainable values
loo_s = np.zeros((n_probls, n_trial))
var_estim_s = np.zeros((n_probls, n_trial))
cor_s = np.zeros((n_probls, n_trial))
for probl_i in range(n_probls):
    n_cur = res_A[probl_i].shape[-1]
    loo_s[probl_i] = np.sum(res_A[probl_i]-res_B[probl_i], axis=-1)
    var_estim_s[probl_i] = n_cur*np.var(
        res_A[probl_i]-res_B[probl_i], ddof=1, axis=-1)
    for trial_i in range(n_trial):
        cor_s[probl_i, trial_i] = np.corrcoef(
            res_A[probl_i][trial_i], res_B[probl_i][trial_i])[0, 1]
coef_var_s = np.sqrt(var_estim_s)/loo_s

# calc some test values
true_mean_s = np.zeros((n_probls))
true_var_s = np.zeros((n_probls))
true_plooneg = np.zeros((n_probls))
elpd_s = np.zeros((n_probls, n_trial))
for probl_i in range(n_probls):
    true_mean_s[probl_i] = np.mean(loo_s[probl_i])
    true_var_s[probl_i] = np.var(loo_s[probl_i], ddof=1)
    true_plooneg[probl_i] = np.mean(loo_s[probl_i]<0)
    elpd_s[probl_i] = res_test_A[probl_i] - res_test_B[probl_i]
true_coef_var_s = np.sqrt(true_var_s)/true_mean_s

# plots
use_sea = True

# LOO and elpd
fig, axes = plt.subplots(1, n_probls, sharey=False, figsize=(16,8))
for ax, probl_i in zip(axes, range(n_probls)):
    if use_sea:
        sns.violinplot(
            data=[loo_s[probl_i], elpd_s[probl_i]],
            orient='v',
            scale='width',
            ax=ax
        )
    else:
        ax.hist(elpd_s[probl_i], 20, color='C1')
        ax.hist(loo_s[probl_i], 20, color='C0')
    ax.set_title(var_vals[probl_i])
fig.suptitle('LOO and test elpd (y={})'.format(var_name))
fig.tight_layout()

# coef of vars
fig, axes = plt.subplots(1, n_probls, sharey=False, figsize=(16,8))
for ax, probl_i in zip(axes, range(n_probls)):
    if use_sea:
        sns.violinplot(coef_var_s[probl_i], orient='v', ax=ax)
    else:
        ax.hist(coef_var_s[probl_i], 20)
    ax.axhline(true_coef_var_s[probl_i], color='r')
    ax.set_title(var_vals[probl_i])
fig.suptitle(var_name)
fig.tight_layout()

# cors
fig, axes = plt.subplots(1, n_probls, sharey=False, figsize=(16,8))
for ax, probl_i in zip(axes, range(n_probls)):
    if use_sea:
        sns.violinplot(cor_s[probl_i], orient='v', ax=ax)
    else:
        ax.hist(cor_s[probl_i], 20)
    ax.set_title(var_vals[probl_i])
fig.suptitle(var_name)
fig.tight_layout()

fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(np.mean(cor_s, axis=-1), true_coef_var_s, '.')
axes[1].plot(np.mean(cor_s, axis=-1), true_plooneg, '.')
