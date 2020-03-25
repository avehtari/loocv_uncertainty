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
    [3, 3, 3],
    [2, 0, 3],
    [0, 0, 2],
)
run_i_s = np.ravel_multi_index(idxs, grid_shape)

# histogram bins
cal_nbins = 7

# errorbar SE multiplier
error_se_multip = 2

# manual zooming
manual_zoom = True


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
# plot separate
# ============================================================================


# cmap = truncate_colormap(cm.get_cmap('Greys'), 0.4)
# cmap.set_under('white', 1.0)
#
# rng = np.random.RandomState(seed=12345)
#
# # marginal hists for the joint
# joint_marginal_hist = True
#
# for probl_i in range(n_probls):
#     y_arr = loo_s[probl_i]
#     y_err = error_se_multip*np.sqrt(naive_var_s[probl_i])
#     x_arr = elpd_s[probl_i]
#
#     if manual_zoom:
#         if probl_i == 0:
#             idxs = (
#                 (-70 < y_arr) & (y_arr < -20) &
#                 (-50 < x_arr) & (x_arr < -37)
#             )
#         if probl_i == 1:
#             idxs = (
#                 (-1.5 < y_arr) & (y_arr < 2.5) &
#                 (-1 < x_arr) & (x_arr < 2.5)
#             )
#         if probl_i == 2:
#             idxs = (
#                 (-18 < y_arr) & (y_arr < 9) &
#                 (-15 < x_arr) & (x_arr < -4)
#             )
#         y_arr = y_arr[idxs]
#         y_err = y_err[idxs]
#         x_arr = x_arr[idxs]
#
#
#     if joint_marginal_hist:
#         grid = sns.jointplot(
#             x_arr, y_arr,
#             kind='hex',
#             height=4,
#             cmap=cmap,
#             # joint_kws=dict(cmin=1),
#         )
#         grid.set_axis_labels(
#             '$\mathrm{elpd}_\mathrm{D}$',
#             '$\widehat{\mathrm{elpd}}_\mathrm{D}$',
#             fontsize=18
#         )
#         fig = grid.fig
#         # custom size
#         fig.set_figwidth(6)
#         fig.set_figheight(4)
#         ax = grid.ax_joint
#         plt.clim(vmin=1)
#     else:
#         fig = plt.figure(figsize=(6,4))
#         ax = plt.gca()
#
#         ax.hexbin(x_arr, y_arr, gridsize=40, cmap=cmap, mincnt=1)
#
#         ax.set_xlabel('$\mathrm{elpd}_\mathrm{D}$', fontsize=18)
#         ax.set_ylabel('$\widehat{\mathrm{elpd}}_\mathrm{D}$', fontsize=18)
#
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#
#     ax.autoscale(enable=False)
#     ax.plot(
#         [min(y_arr.min(), x_arr.min()),
#          max(y_arr.max(), x_arr.max())],
#         [min(y_arr.min(), x_arr.min()),
#          max(y_arr.max(), x_arr.max())],
#         color='C2'
#     )
#
#
#     # error points
#
#     # x_locs = np.linspace(x_arr.min(), x_arr.max(), 10)
#     # idxs = np.unique(np.abs(x_arr - x_locs[:,None]).argmin(axis=-1))
#
#     x_lims = np.linspace(x_arr.min(), x_arr.max()+1e-10, 11)
#     idxs = []
#     for y_l, y_u in zip(x_lims[:-1], x_lims[1:]):
#         cur_idxs = (y_l <= x_arr) & (x_arr < y_u)
#         if cur_idxs.sum() == 0:
#             # no obs in this range
#             continue
#         selected_idx = rng.choice(np.nonzero(cur_idxs)[0])
#         idxs.append(selected_idx)
#
#     ax.errorbar(
#         x_arr[idxs], y_arr[idxs], yerr=y_err[idxs], color='C1', ls='', marker='o')
#
#     if probl_i == 1 and not manual_zoom:
#         ax.set_ylim(top=3.2)
#
#     ax.tick_params(axis='both', which='major', labelsize=16)
#     ax.tick_params(axis='both', which='minor', labelsize=14)
#
#
#
#     fig.tight_layout()





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

fig, axes = plt.subplots(
    4, n_probls,
    gridspec_kw={'height_ratios': [4, 2, 2, 3]},
    figsize=(10,10)
)

for probl_i in range(n_probls):

    # ===============
    # joint

    ax = axes[0, probl_i]

    y_arr = loo_s[probl_i]
    y_err = error_se_multip*np.sqrt(naive_var_s[probl_i])
    x_arr = elpd_s[probl_i]

    if manual_zoom:
        if probl_i == 0:
            idxs = (
                (-70 < y_arr) & (y_arr < -20) &
                (-50 < x_arr) & (x_arr < -37)
            )
        if probl_i == 1:
            idxs = (
                (-1.5 < y_arr) & (y_arr < 2.5) &
                (-1 < x_arr) & (x_arr < 2.5)
            )
        if probl_i == 2:
            idxs = (
                (-18 < y_arr) & (y_arr < 9) &
                (-15 < x_arr) & (x_arr < -4)
            )
        y_arr = y_arr[idxs]
        y_err = y_err[idxs]
        x_arr = x_arr[idxs]

    ax.hexbin(x_arr, y_arr, gridsize=40, cmap=cmap, mincnt=1)

    ax.set_xlabel(r'$\mathrm{elpd}_\mathrm{D}$', fontsize=fontsize)
    ax.set_ylabel(r'$\widehat{\mathrm{elpd}}_\mathrm{D}$', fontsize=fontsize)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.autoscale(enable=False)
    ax.plot(
        [min(y_arr.min(), x_arr.min()),
         max(y_arr.max(), x_arr.max())],
        [min(y_arr.min(), x_arr.min()),
         max(y_arr.max(), x_arr.max())],
        color='C2'
    )


    # error points

    # x_locs = np.linspace(x_arr.min(), x_arr.max(), errorbar_density+1)
    # idxs = np.unique(np.abs(x_arr - x_locs[:,None]).argmin(axis=-1))

    x_lims = np.linspace(x_arr.min(), x_arr.max()+1e-10, errorbar_density+1)
    idxs = []
    x_lim_d = x_lims[1] - x_lims[0]
    for y_l, y_u in zip(x_lims[:-1], x_lims[1:]):
        cur_idxs = (y_l + x_lim_d*0.2 <= x_arr) & (x_arr < y_u - x_lim_d*0.2)
        if cur_idxs.sum() == 0:
            # no obs in this range
            continue
        selected_idx = rng.choice(np.nonzero(cur_idxs)[0])
        idxs.append(selected_idx)

    ax.errorbar(
        x_arr[idxs], y_arr[idxs], yerr=y_err[idxs], color='C1', ls='', marker='o')

    if probl_i == 1 and not manual_zoom:
        ax.set_ylim(top=3.2)

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=14)

    # ===============
    # hist of loo

    ax = axes[1, probl_i]

    x_arr = loo_s[probl_i]

    ax.hist(x_arr, bins=histbins, color='C0')

    # ax.axvline(np.mean(x_arr), color='C1')
    # ax.plot([np.mean(x_arr)]*2, [0, ax.get_ylim()[1]], color='C1')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_yticks([])

    if probl_i == 0:
        ax.set_ylabel(
            r'$\widehat{\mathrm{elpd}}_\mathrm{D}$',
            fontsize=fontsize,
            labelpad=20,
            # rotation=0,
            # ha='right',
        )

    # ===============
    # hist of error

    ax = axes[2, probl_i]
    elpd = elpd_s[probl_i]
    loo = loo_s[probl_i]

    # x_arr = elpd_s[probl_i] - loo_s[probl_i]
    x_arr = (
        (elpd[:n_trial//2] - loo[:n_trial//2])
        + loo[n_trial//2:]
    )

    ax.hist(x_arr, bins=histbins, color='C0')

    ax.axvline(np.mean(x_arr), color='C1')
    # ax.plot([np.mean(x_arr)]*2, [0, ax.get_ylim()[1]], color='C1')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_yticks([])

    if probl_i == 0:
        ax.set_ylabel(
            r'$\mathrm{elpd}_\mathrm{D} - \widehat{\mathrm{elpd}}_\mathrm{D}$',
            fontsize=fontsize,
            labelpad=20,
            # rotation=0,
            # ha='right',
        )

    # ===============
    # calibrations

    ax = axes[3, probl_i]

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
    ax.set_ylim((0, cal_counts.max()))
    ax.set_xlim((0, 1))
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if probl_i == 0:
        ax.set_ylabel(
            r'$p(\mathrm{elpd_D} < \mathrm{elpd_{D \; true}})$',
            fontsize=fontsize,
            labelpad=20,
            # rotation=0,
            # ha='right',
        )

for ax in axes.ravel():
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize-2)


# # share hists ylim
# max_ylim = max(
#     max(map(lambda ax: ax.get_ylim()[1], axes[1,:])),
#     max(map(lambda ax: ax.get_ylim()[1], axes[2,:]))
# )
# for ax_i in [1, 2]:
#     for ax in axes[ax_i,:]:
#         ax.set_ylim(top=max_ylim)

# share hists col ylim
for probl_i in range(n_probls):
    max_ylim = max(map(lambda ax: ax.get_ylim()[1], axes[[1, 2], probl_i]))
    for ax in axes[[1, 2], probl_i]:
        ax.set_ylim(top=max_ylim)


# share hists col x-scale
for probl_i in range(n_probls):
    scale = 0.0
    for ax in axes[[1, 2], probl_i]:
        x_1, x_2 = ax.get_xlim()
        scale = max(scale, x_2-x_1)
    for ax in axes[[1, 2], probl_i]:
        x_1, x_2 = ax.get_xlim()
        if x_2-x_1 < scale:
            d_x = scale - (x_2 - x_1)
            ax.set_xlim([x_1-d_x/2, x_2+d_x/2])


for probl_i, probl_name in enumerate([
    'Clear case',
    'Models similar',
    'Outliers'
]):
    ax = axes[0, probl_i]
    ax.set_title(probl_name, fontsize=fontsize)
    # ax.text(
    #     -0.6,
    #     0.5,
    #     probl_name,
    #     transform=ax.transAxes,
    #     rotation=90,
    #     fontsize=fontsize,
    #     va='center'
    # )

fig.tight_layout()
fig.subplots_adjust(hspace=0.32)
