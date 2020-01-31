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
n_booti_trial = 1000

# calibration nbins
cal_nbins = 5


# pseudo-bma+
do_bma = False


# ============================================================================
# Select problems

# grid dims:
#   sigma2_d  [0.01, 1.0, 100.0]
#   n_obs     [16, 32, 64, 128, 256, 512, 1024]
#   beta_t    [0.0, 0.5, 1.0, 4.0]
#   prc_out   [0/128, eps, 2/128, 12/128]


# # as a function of n
# idxs = (
#     [1]*grid_shape[1]*2,
#     list(range(grid_shape[1]))*2,
#     [3]*grid_shape[1]*2,
#     [0]*grid_shape[1] + [1]*grid_shape[1],
# )
# run_i_s = np.ravel_multi_index(idxs, grid_shape)
# probl_names = []
# for run_i in run_i_s:
#     n_obs_i, beta_t_i, prc_out_i, sigma2_d_i = run_i_to_params(run_i)
#     probl_names.append('n_obs={}, out_prc={:.2}'.format(n_obs_i, prc_out_i))

# as a function of n with beta_t = 0 and beta_t = 1.0
idxs = (
    [1]*grid_shape[1]*2,
    list(range(grid_shape[1]))*2,
    [0]*grid_shape[1] + [2]*grid_shape[1],
    [0]*grid_shape[1]*2,
)
run_i_s = np.ravel_multi_index(idxs, grid_shape)
probl_names = []
for run_i in run_i_s:
    n_obs_i, beta_t_i, prc_out_i, sigma2_d_i = run_i_to_params(run_i)
    probl_names.append('n_obs={}, beta_t={}'.format(n_obs_i, beta_t_i))


# # as a function of beta_t with no out and out
# idxs = (
#     [1]*grid_shape[2]*2,
#     [3]*grid_shape[2]*2,
#     list(range(grid_shape[2]))*2,
#     [0]*grid_shape[2] + [1]*grid_shape[2],
# )
#
# run_i_s = np.ravel_multi_index(idxs, grid_shape)
# probl_names = []
# for run_i in run_i_s:
#     n_obs_i, beta_t_i, prc_out_i, sigma2_d_i = run_i_to_params(run_i)
#     if prc_out_i > 0.0:
#         probl_names.append('b={}, out'.format(beta_t_i))
#     else:
#         probl_names.append('b={}'.format(beta_t_i))

# # as a function of prc_out with beta_t=[0.0, 1.0]
# idxs = (
#     [1]*grid_shape[3]*2,
#     [3]*grid_shape[3]*2,
#     [0]*grid_shape[3] + [3]*grid_shape[3],
#     list(range(grid_shape[3]))*2,
# )
# run_i_s = np.ravel_multi_index(idxs, grid_shape)
# probl_names = []
# for run_i in run_i_s:
#     n_obs_i, beta_t_i, prc_out_i, sigma2_d_i = run_i_to_params(run_i)
#     probl_names.append('pout={:.2}, b={}'.format(prc_out_i, beta_t_i))

# ============================================================================

def remove_frame(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

# ============================================================================

print('selected problems:')
print('fixed tau2: {}'.format(fixed_sigma2_m))
print('sigma2_d: {}'.format(sigma2_d_grid[idxs]))
print('n_obs: {}'.format(n_obs_grid[idxs]))
print('beta_t: {}'.format(beta_t_grid[idxs]))
print('prc_out: {}'.format(prc_out_grid[idxs]))

if fixed_sigma2_m:
    folder_name = 'fixed'
else:
    folder_name = 'unfixed'

rng = np.random.RandomState(seed=seed_analysis)

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
cor_loo_i_s = np.zeros((n_probls, n_trial))
skew_loo_i_s = np.zeros((n_probls, n_trial))
for probl_i in range(n_probls):
    n_cur = res_A[probl_i].shape[-1]
    loo_s[probl_i] = np.sum(res_A[probl_i]-res_B[probl_i], axis=-1)
    naive_var_s[probl_i] = n_cur*np.var(
        res_A[probl_i]-res_B[probl_i], ddof=1, axis=-1)
    for trial_i in range(n_trial):
        cor_loo_i_s[probl_i, trial_i] = np.corrcoef(
            res_A[probl_i][trial_i], res_B[probl_i][trial_i])[0, 1]
    skew_loo_i_s[probl_i] = stats.skew(
        res_A[probl_i]-res_B[probl_i], axis=-1, bias=False)
naive_coef_var_s = np.sqrt(naive_var_s)/loo_s
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
    # TODO calc se of this ... formulas online
    target_plooneg_s[probl_i] = np.mean(loo_s[probl_i]<0)
    elpd_s[probl_i] = res_test_A[probl_i] - res_test_B[probl_i]
pelpdneg_s = np.mean(elpd_s<0, axis=-1)
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
        X_mat_i, mu_d_i, _, _,  = make_data(
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

# misspred: TODO replace with something
naive_misspred_s = np.abs(naive_plooneg_s-target_plooneg_s[:,None])

# pseudo-bma-plus
if do_bma:
    print('calc pseudo-bma+', flush=True)
    bma_s = np.zeros((n_probls, n_trial, 2))
    for probl_i in range(n_probls):
        loo_tki = np.stack((res_A[probl_i], res_B[probl_i]), axis=1)
        bma_s[probl_i] = pseudo_bma_p(loo_tki)
    print('calc pseudo-bma+, done', flush=True)


# calibration counts
cal_limits = np.linspace(0, 1, cal_nbins+1)
cal_counts = np.zeros((n_probls, cal_nbins), dtype=int)
for probl_i in range(n_probls):
    diff_pdf = stats.norm.cdf(
        elpd_s[probl_i],
        loc=loo_s[probl_i],
        scale=np.sqrt(naive_var_s[probl_i])
    )
    cal_counts[probl_i] = np.histogram(diff_pdf, cal_limits)[0]


# ===========================================================================
# plots

if False:

    # LOO and elpd normalised
    fig, axes = plt.subplots(
        2, n_probls//2 + n_probls%2,
        # sharey='row',
        figsize=(16,12)
    )
    for probl_i in range(axes.size):
        ax = axes.flat[probl_i]
        if probl_i >= n_probls:
            ax.axis('off')
            continue
        n_obs_i = n_obs_grid.flat[run_i_s[probl_i]]
        if fixed_sigma2_m:
            ax.axhline(analytic_mean_s[probl_i], color='red')
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
        ax.set_title(probl_names[probl_i])
    fig.suptitle('LOO and test elpd (normalised)')
    # fig.tight_layout()

    # naive coef of vars
    fig, axes = plt.subplots(
        2, n_probls//2 + n_probls%2,
        # sharey='row',
        figsize=(16,12)
    )
    for probl_i in range(axes.size):
        ax = axes.flat[probl_i]
        if probl_i >= n_probls:
            ax.axis('off')
            continue
        if use_sea:
            sns.violinplot(
                naive_coef_var_s[probl_i], orient='v', ax=ax,
                label='naive coef of var')
        else:
            ax.hist(naive_coef_var_s[probl_i], 20, label='naive coef of var')
        if fixed_sigma2_m:
            ax.axhline(analytic_coefvar_s[probl_i],
                color='r', label='analytic')
        else:
            ax.axhline(target_coefvar_s[probl_i],
                color='r', label='estimated over trials')
        ax.set_title(probl_names[probl_i])
    axes.flat[-1].legend()
    fig.suptitle('naive coef of var')
    # fig.tight_layout()

    # calibration plots
    fig, axes = plt.subplots(
        2, n_probls//2 + n_probls%2,
        sharey='row',
        sharex=True,
        figsize=(16,12)
    )
    for probl_i in range(axes.size):
        ax = axes.flat[probl_i]
        if probl_i >= n_probls:
            ax.axis('off')
            continue
        ax.bar(
            cal_limits[:-1],
            cal_counts[probl_i],
            width=1/cal_nbins,
            align='edge'
        )
        ax.axhline(stats.binom.ppf(0.005, n_trial, 1/cal_nbins), color='gray')
        ax.axhline(n_trial/cal_nbins, color='gray')
        ax.axhline(stats.binom.ppf(0.995, n_trial, 1/cal_nbins), color='gray')
        ax.set_title(probl_names[probl_i])
    # axes.flat[-1].legend()
    fig.suptitle('calibration naive var with normal approx.')
    # fig.tight_layout()


    # ===========
    # cor_loo_i_s
    # ===========

    # loo_i cors
    fig, axes = plt.subplots(
        2, n_probls//2 + n_probls%2,
        sharey='row',
        figsize=(16,12)
    )
    for probl_i in range(axes.size):
        ax = axes.flat[probl_i]
        if probl_i >= n_probls:
            ax.axis('off')
            continue
        if use_sea:
            sns.violinplot(cor_loo_i_s[probl_i], orient='v', ax=ax)
        else:
            ax.hist(cor_loo_i_s[probl_i], 20)
        ax.set_title(probl_names[probl_i])
    fig.suptitle('loo_i cors')
    # fig.tight_layout()
    #
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(np.mean(cor_loo_i_s, axis=-1), target_coefvar_s, '.')
    axes[1].plot(np.mean(cor_loo_i_s, axis=-1), target_plooneg_s, '.')

    # loo_i_cors vs naive_coef_var/coef_var
    fig, axes = plt.subplots(
        2, n_probls//2 + n_probls%2,
        sharex=True,
        sharey='row',
        figsize=(16,12)
    )
    for probl_i in range(axes.size):
        ax = axes.flat[probl_i]
        if probl_i >= n_probls:
            ax.axis('off')
            continue
        if fixed_sigma2_m:
            y_i = naive_coef_var_s[probl_i]/analytic_coefvar_s[probl_i]
        else:
            y_i = naive_coef_var_s[probl_i]/target_coefvar_s[probl_i]
        ax.plot(cor_loo_i_s[probl_i], y_i,'.')
        ax.set_title(probl_names[probl_i])
    fig.suptitle('loo_i_cors vs naive_coef_var/coef_var')

    # loo_i_cors vs |naive_plooneg-plooneg|
    fig, axes = plt.subplots(
        2, n_probls//2 + n_probls%2,
        sharex=True,
        sharey='row',
        figsize=(16,12)
    )
    for probl_i in range(axes.size):
        ax = axes.flat[probl_i]
        if probl_i >= n_probls:
            ax.axis('off')
            continue
        ax.plot(cor_loo_i_s[probl_i], naive_misspred_s[probl_i], '.')
        ax.set_title(probl_names[probl_i])
    fig.suptitle('loo_i_cors vs |naive_plooneg-plooneg|')

    # loo_i_cors vs (elpd-loo)/n_obs
    fig, axes = plt.subplots(
        2, n_probls//2 + n_probls%2,
        sharex=True,
        sharey='row',
        figsize=(16,12)
    )
    for probl_i in range(axes.size):
        ax = axes.flat[probl_i]
        if probl_i >= n_probls:
            ax.axis('off')
            continue
        n_obs_i = n_obs_grid.flat[run_i_s[probl_i]]
        ax.plot(
            cor_loo_i_s[probl_i],
            (loo_s[probl_i]-elpd_s[probl_i])/n_obs_i,
            '.'
        )
        ax.set_title(probl_names[probl_i])
    fig.suptitle('loo_i_cors vs (elpd-loo)/n_obs')


    # ===========
    # skews
    # ===========

    # skew_loo_i
    fig, axes = plt.subplots(
        2, n_probls//2 + n_probls%2,
        sharey='row',
        figsize=(16,12)
    )
    for probl_i in range(axes.size):
        ax = axes.flat[probl_i]
        if probl_i >= n_probls:
            ax.axis('off')
            continue
        if use_sea:
            sns.violinplot(skew_loo_i_s[probl_i], orient='v', ax=ax)
        else:
            ax.hist(skew_loo_i_s[probl_i], 20)
        if fixed_sigma2_m:
            ax.axhline(analytic_skew_s[probl_i], color='red')
        else:
            ax.axhline(target_skew_s[probl_i], color='red')
        ax.set_title(probl_names[probl_i])
    fig.suptitle('skew_loo_i')
    # fig.tight_layout()

    # skew_loo_i vs naive_coef_var/coef_var
    fig, axes = plt.subplots(
        2, n_probls//2 + n_probls%2,
        sharex=False,
        sharey='row',
        figsize=(16,12)
    )
    for probl_i in range(axes.size):
        ax = axes.flat[probl_i]
        if probl_i >= n_probls:
            ax.axis('off')
            continue
        if fixed_sigma2_m:
            y_i = naive_coef_var_s[probl_i]/analytic_coefvar_s[probl_i]
        else:
            y_i = naive_coef_var_s[probl_i]/target_coefvar_s[probl_i]
        ax.plot(skew_loo_i_s[probl_i], y_i,'.')
        ax.set_title(probl_names[probl_i])
    fig.suptitle('skew_loo_i vs naive_coef_var/coef_var')

    # skew_loo_i vs |naive_plooneg-plooneg|
    fig, axes = plt.subplots(
        2, n_probls//2 + n_probls%2,
        sharex=False,
        sharey='row',
        figsize=(16,12)
    )
    for probl_i in range(axes.size):
        ax = axes.flat[probl_i]
        if probl_i >= n_probls:
            ax.axis('off')
            continue
        ax.plot(skew_loo_i_s[probl_i], naive_misspred_s[probl_i], '.')
        ax.set_title(probl_names[probl_i])
    fig.suptitle('skew_loo_i vs |naive_plooneg-plooneg|')

    # skew_loo_i vs (elpd-loo)/n_obs
    fig, axes = plt.subplots(
        2, n_probls//2 + n_probls%2,
        sharex=False,
        sharey='row',
        figsize=(16,12)
    )
    for probl_i in range(axes.size):
        ax = axes.flat[probl_i]
        if probl_i >= n_probls:
            ax.axis('off')
            continue
        n_obs_i = n_obs_grid.flat[run_i_s[probl_i]]
        ax.plot(
            skew_loo_i_s[probl_i],
            (loo_s[probl_i]-elpd_s[probl_i])/n_obs_i,
            '.'
        )
        ax.set_title(probl_names[probl_i])
    fig.suptitle('skew_loo_i vs (elpd-loo)/n_obs')

    # ==================
    # pseudo-bma
    # ==================

    # if do_bma:

    fig, axes = plt.subplots(
        2, n_probls//2 + n_probls%2,
        sharey=True,
        figsize=(16,12)
    )
    for probl_i in range(axes.size):
        ax = axes.flat[probl_i]
        if probl_i >= n_probls:
            ax.axis('off')
            continue
        if use_sea:
            sns.violinplot(
                bma_s[probl_i][:, 1], orient='v', ax=ax)
        else:
            ax.hist(bma_s[probl_i][:, 0], 20)
        ax.set_title(probl_names[probl_i])
    for ax in axes[:,0]:
        ax.set_ylim((0,1))
    axes.flat[-1].legend()
    fig.suptitle('pseudo-bma+ for model A')
    # fig.tight_layout()

    fig, axes = plt.subplots(
        2, n_probls//2 + n_probls%2,
        sharey=True,
        figsize=(16,12)
    )
    for probl_i in range(axes.size):
        ax = axes.flat[probl_i]
        if probl_i >= n_probls:
            ax.axis('off')
            continue
        ax.plot(loo_s[probl_i], bma_s[probl_i][:, 1], '.')
        ax.set_title(probl_names[probl_i])
        ax.axhline(pelpdneg_s[probl_i], color='r', label='$p(elpd_d<0)$')
    for ax in axes[:,0]:
        ax.set_ylim((-0.1,1.1))
    axes.flat[-1].legend()
    fig.suptitle('$\widehat{elpd}_d$ vs pseudo-bma+ for model B')
    # fig.tight_layout()

    # ===========================
    # check analytic vs estimated
    # ===========================

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


# # load results
# bootloo_tb = np.zeros((n_probls, n_trial, n_boot_trial))
# mbootloo_tb = np.zeros((n_probls, n_trial, n_mboot_trial))
# for probl_i in range(n_probls):
#     run_i = run_i_s[probl_i]
#     # boot
#     res_file = np.load(
#         'res_1/{}/boot/{}.npz'
#         .format(folder_name, str(run_i).zfill(4))
#     )
#     bootloo_tb[probl_i] = res_file['bootloo_tb']
#     res_file.close()
#     # mboot
#     res_file = np.load(
#         'res_1/{}/mboot/{}.npz'
#         .format(folder_name, str(run_i).zfill(4))
#     )
#     mbootloo_tb[probl_i] = res_file['mbootloo_tb']
#     res_file.close()

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
# boot_plooneg = np.mean(bootloo_tb<0.0, axis=-1)
# mboot_plooneg = np.mean(mbootloo_tb<0.0, axis=-1)
booti_plooneg = np.mean(bootlooi_tb<0.0, axis=-1)


# misspred: TODO replace with something
booti_misspred_s = np.abs(booti_plooneg-target_plooneg_s[:,None])

if False:
    # plot this
    fig, axes = plt.subplots(1, n_probls, sharey=True, figsize=(16,12))
    for ax, probl_i in zip(axes, range(n_probls)):
        if use_sea:
            sns.violinplot(
                data=[
                    # boot_plooneg[probl_i],
                    # mboot_plooneg[probl_i],
                    booti_plooneg[probl_i]
                ],
                orient='v',
                scale='width',
                cut=0.0,
                ax=ax
            )
        else:
            ax.hist(booti_plooneg[probl_i], color='C0',
                orientation='horizontal',
                bins=np.linspace(0,1,30),
                alpha=0.5)
            # ax.hist(mboot_plooneg[probl_i], color='C1',
            #     orientation='horizontal',
            #     bins=np.linspace(0,1,30),
            #     alpha=0.5)
            # ax.hist(boot_plooneg[probl_i], color='C0',
            #     orientation='horizontal',
            #     bins=np.linspace(0,1,30),
            #     alpha=0.5)
        ax.axhline(target_plooneg_s[probl_i], color='r')
        ax.set_title(probl_names[probl_i])
    fig.suptitle(probl_name)
    fig.tight_layout()

    # plot trial vice boot vs naive normal ploonegs
    fig, axes = plt.subplots(
        2, n_probls//2 + n_probls%2, figsize=(16,12))
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

    # plot trial vice booti misspred vs naive normal misspred
    fig, axes = plt.subplots(
        2, n_probls//2 + n_probls%2, figsize=(16,12))
    for probl_i in range(axes.size):
        ax = axes.flat[probl_i]
        if probl_i >= n_probls:
            ax.axis('off')
            continue
        if True:
            sns.violinplot(
                data=[
                    naive_misspred_s[probl_i],
                    booti_misspred_s[probl_i]
                ],
                orient='v',
                scale='width',
                cut=0.0,
                ax=ax
            )
        else:
            ax.hist(booti_misspred_s[probl_i], color='C1',
                orientation='horizontal',
                bins=np.linspace(0,1,30),
                alpha=0.5)
            ax.hist(naive_misspred_s[probl_i], color='C0',
                orientation='horizontal',
                bins=np.linspace(0,1,30),
                alpha=0.5)
    fig.suptitle('misspred')
