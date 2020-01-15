"""Analyse LOOCV results

First run results with m1_run.py

"""

import numpy as np
from scipy import linalg, stats

import matplotlib.pyplot as plt
import seaborn as sns


from m1_problem import *


# ============================================================================
# config

# seed for randomisation
# np.random.RandomState().randint(np.iinfo(np.uint32).max)
seed_analysis = 4257151306

# fixed or unfixed sigma
fixed_sigma2_m = False

# plot using seaborn
use_sea = True

# bootstrap on LOO_i
n_booti_trial = 123


# ============================================================================

if fixed_sigma2_m:
    folder_name = 'fixed'
else:
    folder_name = 'unfixed'

rng = np.random.RandomState(seed=seed_analysis)


def remove_frame(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

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
naive_var_s = np.zeros((n_probls, n_trial))
cor_s = np.zeros((n_probls, n_trial))
skew_loo_i_s = np.zeros((n_probls, n_trial))
for probl_i in range(n_probls):
    n_cur = res_A[probl_i].shape[-1]
    loo_s[probl_i] = np.sum(res_A[probl_i]-res_B[probl_i], axis=-1)
    naive_var_s[probl_i] = n_cur*np.var(
        res_A[probl_i]-res_B[probl_i], ddof=1, axis=-1)
    for trial_i in range(n_trial):
        cor_s[probl_i, trial_i] = np.corrcoef(
            res_A[probl_i][trial_i], res_B[probl_i][trial_i])[0, 1]
    skew_loo_i_s[probl_i] = stats.skew(
        res_A[probl_i]-res_B[probl_i], axis=-1, bias=False)
coef_var_s = np.sqrt(naive_var_s)/loo_s
naive_plooneg_s = stats.norm.cdf(0, loc=loo_s, scale=np.sqrt(naive_var_s))

# calc some target values
target_mean_s = np.zeros((n_probls))
target_var_s = np.zeros((n_probls))
target_skew_s = np.zeros((n_probls))
target_plooneg_s = np.zeros((n_probls))
elpd_s = np.zeros((n_probls, n_trial))
for probl_i in range(n_probls):
    target_mean_s[probl_i] = np.mean(loo_s[probl_i])
    target_var_s[probl_i] = np.var(loo_s[probl_i], ddof=1)
    target_skew_s[probl_i] = stats.skew(loo_s[probl_i], bias=False)
    target_plooneg_s[probl_i] = np.mean(loo_s[probl_i]<0)
    elpd_s[probl_i] = res_test_A[probl_i] - res_test_B[probl_i]
target_coefvar_s = np.sqrt(target_var_s)/target_mean_s

# calc true analytic values
analytic_mean_s = np.full((n_probls), np.nan)
analytic_var_s = np.full((n_probls), np.nan)
analytic_skew_s = np.full((n_probls), np.nan)
analytic_coefvar_s = np.full((n_probls), np.nan)
if fixed_sigma2_m:
    for probl_i in range(n_probls):
        run_i = run_i_s[probl_i]
        n_obs_i, beta_t_i, prc_out_i, sigma2_d_i = run_i_to_params(run_i)
        X_mat_i, _, _, _, mu_d_i = make_data(
            n_obs_i, beta_t_i, prc_out_i, sigma2_d_i)
        if prc_out_i == 0.0:
            mu_d_i = None
        A_mat, b_vec, c_sca = get_analytic_params(X_mat_i, beta_t_i)
        analytic_mean_s[probl_i] = calc_analytic_mean(
            A_mat, b_vec, c_sca, sigma2_d_i, mu_d_i)
        analytic_var_s[probl_i] = calc_analytic_var(
            A_mat, b_vec, c_sca, sigma2_d_i, mu_d_i)
        analytic_skew_s[probl_i] = calc_analytic_skew(
            A_mat, b_vec, c_sca, sigma2_d_i, mu_d_i)
        analytic_coefvar_s[probl_i] = calc_analytic_coefvar(
            A_mat, b_vec, c_sca, sigma2_d_i, mu_d_i)

# ===========================================================================
# plots

if False:
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
        if fixed_sigma2_m:
            ax.axhline(analytic_mean_s[probl_i], color='red')
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
        ax.axhline(target_coefvar_s[probl_i], color='r')
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
    #
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(np.mean(cor_s, axis=-1), target_coefvar_s, '.')
    axes[1].plot(np.mean(cor_s, axis=-1), target_plooneg_s, '.')

    # skew
    fig, axes = plt.subplots(1, n_probls, sharey=True, figsize=(16,8))
    for ax, probl_i in zip(axes, range(n_probls)):
        if use_sea:
            sns.violinplot(skew_loo_i_s[probl_i], orient='v', ax=ax)
        else:
            ax.hist(skew_loo_i_s[probl_i], 20)
        if fixed_sigma2_m:
            ax.axhline(analytic_skew_s[probl_i], color='red')
        else:
            ax.axhline(target_skew_s[probl_i], color='red')
        ax.set_title(var_vals[probl_i])
    fig.suptitle(var_name)
    fig.tight_layout()



    if fixed_sigma2_m:

        # compare target and analytic mean
        plt.figure()
        plt.plot(target_mean_s, analytic_mean_s, '+')
        plt.plot(plt.xlim(), plt.ylim(), color='red', zorder=-1)
        plt.xlabel('target mean')
        plt.ylabel('analytic mean')

        # compare target and analytic var
        plt.figure()
        plt.plot(target_var_s, analytic_var_s, '+')
        plt.plot(plt.xlim(), plt.ylim(), color='red', zorder=-1)
        plt.xlabel('target var')
        plt.ylabel('analytic var')

        # compare target and analytic skew
        plt.figure()
        plt.plot(target_skew_s, analytic_skew_s, '+')
        plt.plot(plt.xlim(), plt.ylim(), color='red', zorder=-1)
        plt.xlabel('target skew')
        plt.ylabel('analytic skew')

        # compare target and analytic coefficient of variation
        plt.figure()
        plt.plot(target_coefvar_s, analytic_coefvar_s, '+')
        plt.plot(plt.xlim(), plt.ylim(), color='red', zorder=-1)
        plt.xlabel('target coefvar')
        plt.ylabel('analytic coefvar')


# ============================================================================
# boots


# load results
bootloo_tb = np.zeros((n_probls, n_trial, n_boot_trial))
mbootloo_tb = np.zeros((n_probls, n_trial, n_mboot_trial))
for probl_i in range(n_probls):
    run_i = run_i_s[probl_i]
    # boot
    res_file = np.load(
        'res_1/{}/boot/{}.npz'
        .format(folder_name, str(run_i).zfill(4))
    )
    bootloo_tb[probl_i] = res_file['bootloo_tb']
    res_file.close()
    # mboot
    res_file = np.load(
        'res_1/{}/mboot/{}.npz'
        .format(folder_name, str(run_i).zfill(4))
    )
    mbootloo_tb[probl_i] = res_file['mbootloo_tb']
    res_file.close()

# calc bool on LOO_i
bootlooi_tb = np.zeros((n_probls, n_trial, n_booti_trial))
for probl_i in range(n_probls):
    run_i = run_i_s[probl_i]
    n_obs_i, beta_t_i, prc_out_i, sigma2_d_i = run_i_to_params(run_i)
    for b_i in range(n_booti_trial):
        # take subsample
        idxs = rng.choice(n_obs_i, size=n_obs_i, replace=True)
        # idxs.sort()
        loo_i = res_A[probl_i]-res_B[probl_i]
        loo_i = loo_i[:, idxs]
        bootlooi_tb[probl_i,:,b_i] = np.sum(loo_i, axis=-1)

# calc probability of LOO being negative
boot_plooneg = np.mean(bootloo_tb<0.0, axis=-1)
mboot_plooneg = np.mean(mbootloo_tb<0.0, axis=-1)
booti_plooneg = np.mean(bootlooi_tb<0.0, axis=-1)


if False:
    # plot this for them all
    fig, axes = plt.subplots(1, n_probls, sharey=True, figsize=(16,8))
    for ax, probl_i in zip(axes, range(n_probls)):
        if use_sea:
            sns.violinplot(
                data=[
                    boot_plooneg[probl_i],
                    mboot_plooneg[probl_i],
                    booti_plooneg[probl_i]
                ],
                orient='v',
                scale='width',
                cut=0.0,
                ax=ax
            )
        else:
            ax.hist(booti_plooneg[probl_i], color='C2',
                orientation='horizontal',
                bins=np.linspace(0,1,30),
                alpha=0.5)
            ax.hist(mboot_plooneg[probl_i], color='C1',
                orientation='horizontal',
                bins=np.linspace(0,1,30),
                alpha=0.5)
            ax.hist(boot_plooneg[probl_i], color='C0',
                orientation='horizontal',
                bins=np.linspace(0,1,30),
                alpha=0.5)
        ax.axhline(target_plooneg_s[probl_i], color='r')
        ax.set_title(var_vals[probl_i])
    fig.suptitle(var_name)
    fig.tight_layout()

    # plot trial vice boot vs naive normal ploonegs
    fig, axes = plt.subplots(
        2, n_probls//2 + n_probls%2, figsize=(16,8))
    for probl_i in range(axes.size):
        ax = axes.flat[probl_i]
        if probl_i >= n_probls:
            ax.axis('off')
            continue
        ax.plot((0,1), (0,1), color='gray')
        ax.axhline(target_plooneg_s[probl_i], color='pink')
        ax.axvline(target_plooneg_s[probl_i], color='pink')
        ax.plot(booti_plooneg[probl_i], naive_plooneg_s[probl_i], '.')
        # ax.set_xlim((
        #     max(min(ax.get_xlim()[0], ax.get_ylim()[0]) ,0.01),
        #     min(max(ax.get_xlim()[1], ax.get_ylim()[1]), 1.01)
        # ))
        # ax.set_ylim(ax.get_xlim())
        ax.set_xlim(-0.01,1.01)
        ax.set_ylim(-0.01,1.01)
        remove_frame(ax)
    fig.suptitle(
        'p(choose B): x=bootstrap on the LOO_i, y=normal approx with naive var')
