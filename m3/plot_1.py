"""Analyse LOOCV results

First run results with m1_run.py

"""

import numpy as np
from scipy import linalg, stats

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import seaborn as sns


from m3_problem import *


# ============================================================================
# config

# seed for randomisation
# np.random.RandomState().randint(np.iinfo(np.uint32).max)
seed_analysis = 4257151306

# fixed or unfixed sigma
fixed_sigma2_m = False

# ============================================================================

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def remove_frame(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

# ============================================================================

if fixed_sigma2_m:
    folder_name = 'fixed'
else:
    folder_name = 'unfixed'

rng = np.random.RandomState(seed=seed_analysis)

n_obs = 128

loo_s = np.zeros((grid_shape[2], grid_shape[3], n_trial))
naive_var_s = np.zeros((grid_shape[2], grid_shape[3], n_trial))
cor_loo_i_s = np.zeros((grid_shape[2], grid_shape[3], n_trial))
skew_loo_i_s = np.zeros((grid_shape[2], grid_shape[3], n_trial))

target_mean_s = np.zeros((grid_shape[2], grid_shape[3]))
target_var_s = np.zeros((grid_shape[2], grid_shape[3]))
# target_Sigma_s = np.zeros((grid_shape[2], grid_shape[3], n_obs, n_obs))
target_skew_s = np.zeros((grid_shape[2], grid_shape[3]))
target_plooneg_s = np.zeros((grid_shape[2], grid_shape[3]))
elpd_s = np.zeros((grid_shape[2], grid_shape[3], n_trial))


# load results
for o_i, prc_out in enumerate(prc_out_s):
    for b_i, beta_t in enumerate(beta_t_s):
        run_i = np.ravel_multi_index([0, 0, b_i, o_i], grid_shape)
        res_file = np.load(
            'res_3/{}/{}.npz'
            .format(folder_name, str(run_i).zfill(4))
        )
        # fetch results
        loo_ti_A = res_file['loo_ti_A']
        loo_ti_B = res_file['loo_ti_B']
        test_t_A = res_file['test_t_A']
        test_t_B = res_file['test_t_B']
        # close file
        res_file.close()

        # calc some normally obtainable values
        n_cur = loo_ti_A.shape[-1]
        loo_s[b_i, o_i] = np.sum(loo_ti_A-loo_ti_B, axis=-1)
        naive_var_s[b_i, o_i] = n_cur*np.var(
            loo_ti_A-loo_ti_B, ddof=1, axis=-1)
        for trial_i in range(n_trial):
            cor_loo_i_s[b_i, o_i, trial_i] = np.corrcoef(
                loo_ti_A[trial_i], loo_ti_B[trial_i])[0, 1]
        skew_loo_i_s[b_i, o_i] = stats.skew(
            loo_ti_A-loo_ti_B, axis=-1, bias=False)

        # calc some target values
        target_mean_s[b_i, o_i] = np.mean(loo_s[b_i, o_i])
        target_var_s[b_i, o_i] = np.var(loo_s[b_i, o_i], ddof=1)
        # target_Sigma_s[b_i, o_i] = np.cov(loo_ti_A-loo_ti_B, ddof=1, rowvar=False)
        target_skew_s[b_i, o_i] = stats.skew(loo_s[b_i, o_i], bias=False)
        # TODO calc se of this ... formulas online
        target_plooneg_s[b_i, o_i] = np.mean(loo_s[b_i, o_i]<0)
        elpd_s[b_i, o_i] = test_t_A - test_t_B

# naive var bias
naive_se_error_ratio_s = np.sqrt(naive_var_s/target_var_s[:,:,None])

# naive_coef_var_s = np.sqrt(naive_var_s)/loo_s
naive_plooneg_s = stats.norm.cdf(0, loc=loo_s, scale=np.sqrt(naive_var_s))

pelpdneg_s = np.mean(elpd_s<0, axis=-1)
# target_coefvar_s = np.sqrt(target_var_s)/target_mean_s

# misspred: TODO replace with something
naive_misspred_s = np.abs(naive_plooneg_s-pelpdneg_s[:,:,None])


# ===========================================================================
# plot 1

selected_prc_outs = [0, 1, 2]
fig, axes = plt.subplots(len(selected_prc_outs), 1, sharex=True)
for ax_i, ax in enumerate(axes):
    o_i = selected_prc_outs[ax_i]

    # cor
    median = np.percentile(cor_loo_i_s[:,o_i], 50, axis=-1)
    q025 = np.percentile(cor_loo_i_s[:,o_i], 2.5, axis=-1)
    q975 = np.percentile(cor_loo_i_s[:,o_i], 97.5, axis=-1)
    ax.fill_between(beta_t_s, q025, q975, color='C0', alpha=0.2)
    ax.plot(beta_t_s, median, 'C0')
    # ax.plot(beta_t_s, q025, 'C0', alpha=0.5)
    # ax.plot(beta_t_s, q975, 'C0', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelcolor='C0')
    if ax_i == 0:
        ax.set_ylabel('no outliers\n$\\widehat{\\mathrm{Corr}}(\\pi_{\\mathrm{A},\\,i}, \\pi_{\\mathrm{B},\\,i})$')
    elif ax_i == 1:
        ax.set_ylabel('2 outliers\n$\\widehat{\\mathrm{Corr}}(\\pi_{\\mathrm{A},\\,i}, \\pi_{\\mathrm{B},\\,i})$')
    else:
        ax.set_ylabel('4 outliers\n$\\widehat{\\mathrm{Corr}}(\\pi_{\\mathrm{A},\\,i}, \\pi_{\\mathrm{B},\\,i})$')
    # naive_misspred_s
    ax2 = ax.twinx()
    median = np.percentile(naive_misspred_s[:,o_i], 50, axis=-1)
    q025 = np.percentile(naive_misspred_s[:,o_i], 2.5, axis=-1)
    q975 = np.percentile(naive_misspred_s[:,o_i], 97.5, axis=-1)
    ax2.fill_between(beta_t_s, q025, q975, color='C1', alpha=0.2)
    ax2.plot(beta_t_s, median, 'C1')
    ax2.set_ylim((0, 1))
    # ax2.plot(beta_t_s, q025, 'C1', alpha=0.5)
    # ax2.plot(beta_t_s, q975, 'C1', alpha=0.5)
    ax2.set_ylabel(r'$|p_\mathrm{approx} - p_\mathrm{target}|$')
    ax2.tick_params(axis='y', labelcolor='C1')
axes[-1].set_xlabel(r'$\beta_t$')
# axes[-1].set_xlim(right=6.0)
fig.tight_layout()



# ===========================================================================
# plot 2

selected_prc_outs = [0, 1, 2]
fig, axes = plt.subplots(len(selected_prc_outs), 1, sharex=True)
for ax_i, ax in enumerate(axes):
    o_i = selected_prc_outs[ax_i]

    # cor
    median = np.percentile(cor_loo_i_s[:,o_i], 50, axis=-1)
    q025 = np.percentile(cor_loo_i_s[:,o_i], 2.5, axis=-1)
    q975 = np.percentile(cor_loo_i_s[:,o_i], 97.5, axis=-1)
    ax.fill_between(beta_t_s, q025, q975, color='C0', alpha=0.2)
    ax.plot(beta_t_s, median, 'C0')
    # ax.plot(beta_t_s, q025, 'C0', alpha=0.5)
    # ax.plot(beta_t_s, q975, 'C0', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelcolor='C0')
    if ax_i == 0:
        ax.set_ylabel('no outliers\n$\\widehat{\\mathrm{Corr}}(\\pi_{\\mathrm{A},\\,i}, \\pi_{\\mathrm{B},\\,i})$')
    elif ax_i == 1:
        ax.set_ylabel('2 outliers\n$\\widehat{\\mathrm{Corr}}(\\pi_{\\mathrm{A},\\,i}, \\pi_{\\mathrm{B},\\,i})$')
    else:
        ax.set_ylabel('4 outliers\n$\\widehat{\\mathrm{Corr}}(\\pi_{\\mathrm{A},\\,i}, \\pi_{\\mathrm{B},\\,i})$')
    # target_skew_s
    ax2 = ax.twinx()
    ax2.plot(beta_t_s, target_skew_s[:,o_i], 'C1')
    ax2.set_ylim(top=0)
    ax2.set_ylabel(r'skewness of $\pi_\mathrm{D}$')
    ax2.tick_params(axis='y', labelcolor='C1')
axes[-1].set_xlabel(r'$\beta_t$')
# axes[-1].set_xlim(right=6.0)
fig.tight_layout()


# ===========================================================================
# plot 3

selected_prc_outs = [0, 1, 2]
fig, axes = plt.subplots(len(selected_prc_outs), 1, sharex=True)
for ax_i, ax in enumerate(axes):
    o_i = selected_prc_outs[ax_i]

    # cor
    median = np.percentile(cor_loo_i_s[:,o_i], 50, axis=-1)
    q025 = np.percentile(cor_loo_i_s[:,o_i], 2.5, axis=-1)
    q975 = np.percentile(cor_loo_i_s[:,o_i], 97.5, axis=-1)
    ax.fill_between(beta_t_s, q025, q975, color='C0', alpha=0.2)
    ax.plot(beta_t_s, median, 'C0')
    # ax.plot(beta_t_s, q025, 'C0', alpha=0.5)
    # ax.plot(beta_t_s, q975, 'C0', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelcolor='C0')
    if ax_i == 0:
        ax.set_ylabel('no outliers\n$\\widehat{\\mathrm{Corr}}(\\pi_{\\mathrm{A},\\,i}, \\pi_{\\mathrm{B},\\,i})$')
    elif ax_i == 1:
        ax.set_ylabel('2 outliers\n$\\widehat{\\mathrm{Corr}}(\\pi_{\\mathrm{A},\\,i}, \\pi_{\\mathrm{B},\\,i})$')
    else:
        ax.set_ylabel('4 outliers\n$\\widehat{\\mathrm{Corr}}(\\pi_{\\mathrm{A},\\,i}, \\pi_{\\mathrm{B},\\,i})$')
    # SE error ratio
    ax2 = ax.twinx()
    median = np.percentile(naive_se_error_ratio_s[:,o_i], 50, axis=-1)
    q025 = np.percentile(naive_se_error_ratio_s[:,o_i], 2.5, axis=-1)
    q975 = np.percentile(naive_se_error_ratio_s[:,o_i], 97.5, axis=-1)
    ax2.fill_between(beta_t_s, q025, q975, color='C1', alpha=0.2)
    ax2.plot(beta_t_s, median, 'C1')
    # ax2.set_ylim((0, 1))
    # ax2.plot(beta_t_s, q025, 'C1', alpha=0.5)
    # ax2.plot(beta_t_s, q975, 'C1', alpha=0.5)
    ax2.set_ylabel(r'$\mathrm{SE}(\pi_\mathrm{D}) - \mathrm{naive SE}$')
    ax2.tick_params(axis='y', labelcolor='C1')
axes[-1].set_xlabel(r'$\beta_t$')
# axes[-1].set_xlim(right=6.0)
fig.tight_layout()
