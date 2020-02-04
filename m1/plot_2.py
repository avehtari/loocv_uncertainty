"""Script for plot 1."""

import numpy as np
from scipy import linalg, stats

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import seaborn as sns


from m1_problem import *



# ============================================================================

fixed_sigma2_m = False

# [
#   [sigma_d, ...],     [0.01, 1.0, 100.0]
#   [n_obs, ...],       [16, 32, 64, 128, 256, 512, 1024]
#   [beta_t, ...],      [0.0, 0.1, 1.0, 4.0]
#   [prc_out, ...]      [0.0, np.nextafter(0,1), 0.01, 0.08]
# ]

# idxs = (
#     [1, 1, 1, 1, 1, 2, 2, 2, 2],
#     [6, 1, 6, 1, 6, 6, 1, 1, 1],
#     [2, 2, 0, 0, 2, 2, 2, 0, 0],
#     [0, 0, 0, 0, 2, 0, 0, 0, 2],
# )

idxs = (
    [1, 1, 1, 1, 1, 2, 2],
    [6, 1, 6, 6, 6, 6, 1],
    [2, 2, 0, 1, 2, 2, 0],
    [0, 0, 0, 0, 2, 0, 2],
)

run_i_s = np.ravel_multi_index(idxs, grid_shape)

# histogram bins
cal_nbins = 7


fontsize = 16


# ============================================================================

def remove_frame(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

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
    res_test_A[probl_i] = res_file['test_elpd_t_A']
    res_test_B[probl_i] = res_file['test_elpd_t_B']
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
target_coefvar_s = np.sqrt(target_var_s)/target_mean_s

pelpdneg_s = np.mean(elpd_s<0, axis=-1)

# pseudo-bma-plus
print('calc pseudo-bma+', flush=True)
bma_s = np.zeros((n_probls, n_trial, 2))
for probl_i in range(n_probls):
    loo_tki = np.stack((res_A[probl_i], res_B[probl_i]), axis=1)
    bma_s[probl_i] = pseudo_bma_p(loo_tki)
print('calc pseudo-bma+, done', flush=True)

# pseudo-bma
print('calc pseudo-bma', flush=True)
bma_elpd_s = np.zeros((n_probls, n_trial, 2))
for probl_i in range(n_probls):
    elpd_tk = np.stack((res_test_A[probl_i], res_test_B[probl_i]), axis=1)
    bma_elpd_s[probl_i] = pseudo_bma(elpd_tk)
print('calc pseudo-bma, done', flush=True)

# ============================================================================
# plot
# ============================================================================



# ============================================================================
# pseudo-bma+s

bin_lims = np.r_[
    0.0, 0.05,
    np.arange(0.10, 0.95, 0.1),
    0.95, 1.0
]
# bin_lims = np.arange(0.0, 1.1, 0.1)


fig, axes = plt.subplots(n_probls, 2, figsize=(8,9), sharex=True, sharey='row')

for probl_i in range(n_probls):

    n_obs, beta_t, prc_out, sigma2_d = run_i_to_params(run_i_s[probl_i])

    loo = loo_s[probl_i]
    loo_u2_idx = np.abs(loo)<=2
    idx_s = [loo_u2_idx, ~loo_u2_idx]

    for col_i, idx_cur in enumerate(idx_s):

        ax = axes[probl_i, col_i]

        # calibration counts
        ax.hist(bma_s[probl_i, idx_cur, 0], bins=bin_lims)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize-4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    probl_name = (
        r'$n=' + str(n_obs) + r'$' + '\n' +
        r'$\beta_t=' + str(beta_t) + r'$' + '\n' +
        r'$\mathrm{out}\%=' + str(int(prc_out*100)) + r'$' + '\n' +
        r'$\sigma^2_*=' + str(sigma2_d) + r'$'
    )
    ax = axes[probl_i, 0]
    ax.set_ylabel(
        probl_name, rotation=0, ha='right', va='center', fontsize=fontsize-2)

for ax in axes[-1,:]:
    ax.set_xlabel('pseudo-bma+', fontsize=fontsize)
axes[0,0].set_title(r'$|\mathrm{\widehat{elpd}_D}| \leq 2$', fontsize=fontsize)
axes[0,1].set_title(r'$|\mathrm{\widehat{elpd}_D}| > 2$', fontsize=fontsize)

fig.tight_layout()




# ============================================================================
# calibrations


cal_limits = np.linspace(0, 1, cal_nbins+1)


fig, axes = plt.subplots(n_probls, 2, figsize=(8,9), sharex=True, sharey='row')

for probl_i in range(n_probls):

    n_obs, beta_t, prc_out, sigma2_d = run_i_to_params(run_i_s[probl_i])

    loo = loo_s[probl_i]
    loo_u2_idx = np.abs(loo)<=2
    idx_s = [loo_u2_idx, ~loo_u2_idx]

    for col_i, idx_cur in enumerate(idx_s):

        ax = axes[probl_i, col_i]

        n_t_cur = np.sum(idx_cur)

        if n_t_cur < 100:
            # ax.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            continue

        # calibration counts
        q005 = stats.binom.ppf(0.005, n_t_cur, 1/cal_nbins)
        q995 = stats.binom.ppf(0.995, n_t_cur, 1/cal_nbins)
        diff_pdf = stats.norm.cdf(
            elpd_s[probl_i][idx_cur],
            loc=loo_s[probl_i][idx_cur],
            scale=np.sqrt(naive_var_s[probl_i][idx_cur])
        )
        cal_counts = np.histogram(diff_pdf, cal_limits)[0]
        ax.bar(
            cal_limits[:-1],
            cal_counts,
            width=1/cal_nbins,
            align='edge'
        )
        ax.axhline(n_t_cur/cal_nbins, color='red', lw=0.5)
        ax.fill_between(
            [0,1], [q995, q995], [q005, q005], color='red', alpha=0.2,
            zorder=2)

        # ax.set_ylim((0, cal_counts.max()))
        ax.set_xlim((0, 1))
        ax.set_yticks([])
        ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize-4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)


    probl_name = (
        r'$n=' + str(n_obs) + r'$' + '\n' +
        r'$\beta_t=' + str(beta_t) + r'$' + '\n' +
        r'$\mathrm{out}\%=' + str(int(prc_out*100)) + r'$' + '\n' +
        r'$\sigma^2_*=' + str(sigma2_d) + r'$'
    )
    ax = axes[probl_i, 0]
    ax.set_ylabel(
        probl_name, rotation=0, ha='right', va='center', fontsize=fontsize-2)

for ax in axes[-1,:]:
    ax.set_xlabel(r'$p(\mathrm{\widehat{elpd}_D}<\mathrm{elpd_D})$', fontsize=fontsize)
axes[0,0].set_title(r'$|\mathrm{\widehat{elpd}_D}| \leq 2$', fontsize=fontsize)
axes[0,1].set_title(r'$|\mathrm{\widehat{elpd}_D}| > 2$', fontsize=fontsize)

fig.tight_layout()
