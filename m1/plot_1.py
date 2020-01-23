"""Script for plot 1."""

import numpy as np
from scipy import linalg, stats

import matplotlib.pyplot as plt
import seaborn as sns


from m1_problem import *



# ============================================================================

fixed_sigma2_m = False

idxs = (
    [1, 1, 1],
    [3, 3, 3],
    [3, 0, 3],
    [0, 0, 1],
)
run_i_s = np.ravel_multi_index(idxs, grid_shape)

# histogram bins
cal_nbins = 7


# ============================================================================

def remove_frame(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

# ============================================================================

print('selected problems:')
print('fixed sigma2_m: {}'.format(fixed_sigma2_m))
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
    res_test_A[probl_i] = res_file['test_ti_A']
    res_test_B[probl_i] = res_file['test_ti_B']
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
    elpd_s[probl_i] = np.sum(
        res_test_A[probl_i] - res_test_B[probl_i], axis=-1)
target_coefvar_s = np.sqrt(target_var_s)/target_mean_s

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

# ============================================================================
# plot
# ============================================================================


for probl_i in range(n_probls):
    x_arr = loo_s[probl_i]
    y_arr = elpd_s[probl_i]
    grid = sns.jointplot(
        x_arr, y_arr,
        joint_kws=dict(s=6),
        height=4,
        color='C0'
    )
    grid.set_axis_labels(
        '$\widehat{\mathrm{elpd}}_\mathrm{D}$',
        '$\mathrm{elpd}_\mathrm{D}$'
    )
    grid.ax_joint.autoscale(enable=False)
    grid.ax_joint.plot(
        [min(x_arr.min(), y_arr.min()),
         max(x_arr.max(), y_arr.max())],
        [min(x_arr.min(), y_arr.min()),
         max(x_arr.max(), y_arr.max())],
        color='C1'
    )
    plt.tight_layout()

size = plt.gcf().get_size_inches()

q005 = stats.binom.ppf(0.005, n_trial, 1/cal_nbins)
q995 = stats.binom.ppf(0.995, n_trial, 1/cal_nbins)

for probl_i in range(n_probls):
    fig = plt.figure(figsize=size)
    plt.bar(
        cal_limits[:-1],
        cal_counts[probl_i],
        width=1/cal_nbins,
        align='edge'
    )

    plt.axhline(n_trial/cal_nbins, color='red', lw=0.5)
    # plt.axhline(q005, color='pink')
    # plt.axhline(q995, color='pink')
    plt.fill_between(
        [0,1], [q995, q995], [q005, q005], color='red', alpha=0.2,
        zorder=2)
    plt.ylim((0, cal_counts.max()))
    plt.xlim((0, 1))
    plt.yticks([])
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xlabel('$p(\widetilde{\mathrm{\widehat{elpd}_{d}}}<\mathrm{elpd}_d)$')
    plt.tight_layout()
