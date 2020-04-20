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
#   [beta_t, ...],      [0.0, 0.2, 1.0, 4.0]
#   [prc_out, ...]      [0.0, np.nextafter(0,1), 0.01, 0.08]
# ]

idxs = (
    [1, 1, 1],
    [1, 1, 1],
    [0, 1, 2],
    [0, 0, 0],
)

# idxs = (
#     [1, 1, 1, 1],
#     [1, 2, 4, 5],
#     [1, 1, 1, 1],
#     [0, 0, 0, 0],
# )

run_i_s = np.ravel_multi_index(idxs, grid_shape)

# histogram bins
cal_nbins = 7

# mirror looi or mirror resulting distr
mirror_looi = False

# BB sample size
bb_n = 1000
# BB alpha param
bb_a = 1.0

seed_analysis = 123451



# ============================================================================

plt.rcParams['font.family'] = 'serif'
plt.rcParams["mathtext.fontset"] = "dejavuserif"

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

rng = np.random.RandomState(seed=seed_analysis)

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

# calibration counts
cal_limits = np.linspace(0, 1, cal_nbins+1)
# normal
cal_counts_n = np.zeros((n_probls, cal_nbins), dtype=int)
# BB
cal_counts_bb = np.zeros((n_probls, cal_nbins), dtype=int)
# BB mirrored
cal_counts_bbm = np.zeros((n_probls, cal_nbins), dtype=int)
for probl_i in range(n_probls):
    # normal
    err_cdf = stats.norm.cdf(
        elpd_s[probl_i],
        loc=loo_s[probl_i],
        scale=np.sqrt(naive_var_s[probl_i])
    )
    cal_counts_n[probl_i] = np.histogram(err_cdf, cal_limits)[0]
    # BB
    n_cur = res_A[probl_i].shape[-1]
    looi = res_A[probl_i] - res_B[probl_i]
    err_cdf = np.zeros(n_trial)
    for trial_i in range(n_trial):
        weights = rng.dirichlet(
            np.full(n_cur, bb_a),
            size=bb_n
        )
        temp = weights.dot(looi[trial_i])
        temp *= n_cur
        err_cdf[trial_i] = np.mean(temp < elpd_s[probl_i][trial_i])
    cal_counts_bb[probl_i] = np.histogram(err_cdf, cal_limits)[0]
    # BB mirrored
    if mirror_looi:
        n_cur = res_A[probl_i].shape[-1]
        looi = res_A[probl_i] - res_B[probl_i]
        looim = 2*np.mean(looi, axis=-1, keepdims=True) - looi
        err_cdf = np.zeros(n_trial)
        for trial_i in range(n_trial):
            weights = rng.dirichlet(
                np.full(n_cur, bb_a),
                size=bb_n
            )
            temp = weights.dot(looim[trial_i])
            temp *= n_cur
            err_cdf[trial_i] = np.mean(temp < elpd_s[probl_i][trial_i])
        cal_counts_bbm[probl_i] = np.histogram(err_cdf, cal_limits)[0]
    else:
        n_cur = res_A[probl_i].shape[-1]
        looi = res_A[probl_i] - res_B[probl_i]
        err_cdf = np.zeros(n_trial)
        for trial_i in range(n_trial):
            weights = rng.dirichlet(
                np.full(n_cur, bb_a),
                size=bb_n
            )
            temp = weights.dot(looi[trial_i])
            temp *= n_cur
            temp = 2*np.mean(temp) - temp
            err_cdf[trial_i] = np.mean(temp < elpd_s[probl_i][trial_i])
        cal_counts_bbm[probl_i] = np.histogram(err_cdf, cal_limits)[0]


# ============================================================================
# plot sublots
# ============================================================================


q005 = stats.binom.ppf(0.005, n_trial, 1/cal_nbins)
q995 = stats.binom.ppf(0.995, n_trial, 1/cal_nbins)

fontsize = 16

histbins = 14

errorbar_density = 5

cmap = truncate_colormap(cm.get_cmap('Greys'), 0.4)
cmap.set_under('white', 1.0)

rng = np.random.RandomState(seed=12345)

all_cal_counts = [cal_counts_n, cal_counts_bb, cal_counts_bbm]
max_count = max(map(np.max, all_cal_counts))

fig, axes = plt.subplots(
    n_probls, 3,
    figsize=(8,10)
)
for probl_i in range(n_probls):

    # ===============
    # calibrations

    for ax, cal_counts in zip(axes[probl_i, :], all_cal_counts):

        ax.bar(
            cal_limits[:-1],
            cal_counts[probl_i],
            width=1/cal_nbins,
            align='edge'
        )

        ax.axhline(n_trial/cal_nbins, color='C1', lw=0.8)
        # ax.axhline(q005, color='pink')
        # ax.axhline(q995, color='pink')
        ax.fill_between(
            [0,1], [q995, q995], [q005, q005], color='C1', alpha=0.3,
            zorder=2)
        ax.set_ylim((0, max_count))
        ax.set_xlim((0, 1))
        ax.set_yticks([])
        ax.set_xticks([0, 0.5, 1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

for ax in axes.ravel():
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize-2)

for probl_i in range(n_probls):
    axes[probl_i, 0].set_ylabel(
        'beta={}'.format(beta_t_s[idxs[2][probl_i]]), rotation=90)

# for probl_i in range(n_probls):
#     axes[probl_i, 0].set_ylabel(
#         'n={}'.format(n_obs_s[idxs[1][probl_i]]), rotation=90)

fig.tight_layout()
