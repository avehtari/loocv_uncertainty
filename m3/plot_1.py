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

# bayes boot samples
bb_n = 1000
bb_a = 1.0

# bootstrap skew
skew_boot_n = 200

selected_beta_t_s = np.arange(21)

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

beta_t_s_slice = beta_t_s[selected_beta_t_s]

loo_s = np.zeros((len(selected_beta_t_s), grid_shape[3], n_trial))
naive_var_s = np.zeros((len(selected_beta_t_s), grid_shape[3], n_trial))
cor_loo_i_s = np.zeros((len(selected_beta_t_s), grid_shape[3], n_trial))
skew_loo_i_s = np.zeros((len(selected_beta_t_s), grid_shape[3], n_trial))
bb_plooneg_s = np.zeros((len(selected_beta_t_s), grid_shape[3], n_trial))

target_mean_s = np.zeros((len(selected_beta_t_s), grid_shape[3]))
target_var_s = np.zeros((len(selected_beta_t_s), grid_shape[3]))
# target_Sigma_s = np.zeros((len(selected_beta_t_s), grid_shape[3], n_obs, n_obs))
target_skew_s = np.zeros((len(selected_beta_t_s), grid_shape[3]))
target_skew_s_boot = np.zeros((len(selected_beta_t_s), grid_shape[3], skew_boot_n))
target_plooneg_s = np.zeros((len(selected_beta_t_s), grid_shape[3]))
elpd_s = np.zeros((len(selected_beta_t_s), grid_shape[3], n_trial))

bma_s = np.zeros((len(selected_beta_t_s), grid_shape[3], n_trial, 2))
bma_elpd_s = np.zeros((len(selected_beta_t_s), grid_shape[3], n_trial, 2))

# load results
for o_i, prc_out in enumerate(prc_out_s):
    for b_i, b_ii in enumerate(selected_beta_t_s):
        beta_t = beta_t_s[b_ii]
        run_i = np.ravel_multi_index([0, 0, b_ii, o_i], grid_shape)
        res_file = np.load(
            'res_3/{}/{}.npz'
            .format(folder_name, str(run_i).zfill(4))
        )
        # fetch results
        loo_ti_A = res_file['loo_ti_A']
        loo_ti_B = res_file['loo_ti_B']
        test_elpd_t_A = res_file['test_elpd_t_A']
        test_elpd_t_B = res_file['test_elpd_t_B']
        # close file
        res_file.close()

        # calc some normally obtainable values
        n_cur = loo_ti_A.shape[-1]
        loo_i = loo_ti_A-loo_ti_B
        loo_s[b_i, o_i] = np.sum(loo_i, axis=-1)
        naive_var_s[b_i, o_i] = n_cur*np.var(
            loo_i, ddof=1, axis=-1)
        for trial_i in range(n_trial):
            cor_loo_i_s[b_i, o_i, trial_i] = np.corrcoef(
                loo_ti_A[trial_i], loo_ti_B[trial_i])[0, 1]
        skew_loo_i_s[b_i, o_i] = stats.skew(
            loo_i, axis=-1, bias=False)
        bb_plooneg_s[b_i, o_i] = bb_plooneg(loo_i)

        # calc some target values
        target_mean_s[b_i, o_i] = np.mean(loo_s[b_i, o_i])
        target_var_s[b_i, o_i] = np.var(loo_s[b_i, o_i], ddof=1)
        # target_Sigma_s[b_i, o_i] = np.cov(loo_i, ddof=1, rowvar=False)
        target_skew_s[b_i, o_i] = stats.skew(loo_s[b_i, o_i], bias=False)

        for boot_i in range(skew_boot_n):
            boot_loo = rng.choice(loo_s[b_i, o_i], size=n_obs, replace=True)
            target_skew_s_boot[b_i, o_i, boot_i] = stats.skew(
                boot_loo, bias=False)

        # TODO calc se of this ... formulas online

        target_plooneg_s[b_i, o_i] = np.mean(loo_s[b_i, o_i]<0)
        elpd_s[b_i, o_i] = test_elpd_t_A - test_elpd_t_B

        # pseudo-bma+
        loo_tki = np.stack((loo_ti_A, loo_ti_B), axis=1)
        bma_s[b_i, o_i] = pseudo_bma_p(loo_tki)

        # pseudo-bma for true elpd
        elpd_tk = np.stack((test_elpd_t_A, test_elpd_t_B), axis=1)
        bma_elpd_s[b_i, o_i] = pseudo_bma(elpd_tk)

# naive var ratio
naive_se_ratio_s_mean = np.sqrt(
    np.mean(naive_var_s, axis=-1)/target_var_s)
naive_se_ratio_s = np.sqrt(
    np.einsum('bt,ijt->ijb',
        rng.dirichlet(
            np.full(n_trial, bb_a),
            size=bb_n
        ),
        naive_var_s
    )/target_var_s[:,:,None]
)

# naive var error ratio
error_var_s = np.var(loo_s-elpd_s, ddof=1, axis=-1)
naive_error_ratio_s_mean = np.sqrt(
    np.mean(naive_var_s, axis=-1)/error_var_s)
naive_error_ratio_s = np.sqrt(
    np.einsum('bt,ijt->ijb',
        rng.dirichlet(
            np.full(n_trial, bb_a),
            size=bb_n
        ),
        naive_var_s
    )/error_var_s[:,:,None]
)


# naive_coef_var_s = np.sqrt(naive_var_s)/loo_s
naive_plooneg_s = stats.norm.cdf(0, loc=loo_s, scale=np.sqrt(naive_var_s))

pelpdneg_s = np.mean(elpd_s<0, axis=-1)
# target_coefvar_s = np.sqrt(target_var_s)/target_mean_s

# misspred: TODO replace with something
naive_misspred_s = np.abs(naive_plooneg_s-pelpdneg_s[:,:,None])


# ===========================================================================
# plot 1

selected_prc_outs = [0, 1]
for o_i in range(len(selected_prc_outs)):

    fig, axes = plt.subplots(7, 1, sharex=True, figsize=(6,10))

    # loo
    ax = axes[0]
    data_y = loo_s[:,o_i]
    median = np.percentile(data_y, 50, axis=-1)
    q025 = np.percentile(data_y, 2.5, axis=-1)
    q975 = np.percentile(data_y, 97.5, axis=-1)
    ax.fill_between(beta_t_s_slice, q025, q975, color='C0', alpha=0.2)
    ax.plot(beta_t_s_slice, median, 'C0')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.set_ylabel(r'$\mathrm{\widehat{elpd}_D}$', fontsize=16)

    # corr
    ax = axes[1]
    data_y = cor_loo_i_s[:,o_i]
    median = np.percentile(data_y, 50, axis=-1)
    q025 = np.percentile(data_y, 2.5, axis=-1)
    q975 = np.percentile(data_y, 97.5, axis=-1)
    ax.fill_between(beta_t_s_slice, q025, q975, color='C0', alpha=0.2)
    ax.plot(beta_t_s_slice, median, 'C0')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.set_ylabel('correlation', fontsize=16)
    ax.set_ylim(cor_loo_i_s.min(), 1.1)

    # skew
    ax = axes[2]
    ax.axhline(0, color='red')
    data_y = target_skew_s_boot[:,o_i]
    median = np.percentile(data_y, 50, axis=-1)
    q025 = np.percentile(data_y, 2.5, axis=-1)
    q975 = np.percentile(data_y, 97.5, axis=-1)
    ax.fill_between(beta_t_s_slice, q025, q975, color='C0', alpha=0.2)
    ax.plot(beta_t_s_slice, median, 'C0')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.set_ylabel('skewness', fontsize=16)

    # SE error ratio
    ax = axes[3]
    ax.axhline(1.0, color='red')
    data_y = naive_error_ratio_s[:,o_i]
    median = np.percentile(data_y, 50, axis=-1)
    q025 = np.percentile(data_y, 2.5, axis=-1)
    q975 = np.percentile(data_y, 97.5, axis=-1)
    ax.fill_between(beta_t_s_slice, q025, q975, color='C0', alpha=0.2)
    ax.plot(beta_t_s_slice, median, 'C0')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.set_ylabel('SE ratio', fontsize=16)

    # BMA true
    ax = axes[4]
    data_y = bma_elpd_s[:,o_i,:,0]
    median = np.percentile(data_y, 50, axis=-1)
    q025 = np.percentile(data_y, 2.5, axis=-1)
    q975 = np.percentile(data_y, 97.5, axis=-1)
    ax.fill_between(beta_t_s_slice, q025, q975, color='C0', alpha=0.2)
    ax.plot(beta_t_s_slice, median, 'C0')
    ax.set_ylim(0,1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.set_ylabel(r'true pBMA', fontsize=16)

    # pBMA+
    ax = axes[5]
    data_y = bma_s[:,o_i,:,0]
    median = np.percentile(data_y, 50, axis=-1)
    q025 = np.percentile(data_y, 2.5, axis=-1)
    q975 = np.percentile(data_y, 97.5, axis=-1)
    ax.fill_between(beta_t_s_slice, q025, q975, color='C0', alpha=0.2)
    ax.plot(beta_t_s_slice, median, 'C0')
    ax.set_ylim(0,1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.set_ylabel(r'pBMA+', fontsize=16)

    # bb_plooneg_s
    ax = axes[6]
    data_y = 1-bb_plooneg_s[:,o_i,:]
    median = np.percentile(data_y, 50, axis=-1)
    q025 = np.percentile(data_y, 2.5, axis=-1)
    q975 = np.percentile(data_y, 97.5, axis=-1)
    ax.fill_between(beta_t_s_slice, q025, q975, color='C0', alpha=0.2)
    ax.plot(beta_t_s_slice, median, 'C0')
    ax.set_ylim(0,1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.set_ylabel(r'$\mathrm{Pr}(\mathrm{\widehat{elpd}_d}>0)$', fontsize=16)

    axes[-1].set_xlabel(r'$\beta_t$', fontsize=16)
    fig.tight_layout()
