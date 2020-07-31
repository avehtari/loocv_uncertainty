
import numpy as np
from scipy import linalg, stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import matplotlib.gridspec as gridspec

# import seaborn as sns
# import pandas as pd

from setup_all import *


# ============================================================================
# select problems

# number of obs in one trial
# n_obs_s = [16, 32, 64, 128, 256, 512, 1024]
# n_obs_sel = [16, 64, 256, 1024]
# n_obs_sel = [32, 128, 512]
n_obs_sel = n_obs_s

# last covariate effect not used in model A
# beta_t_s = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
# beta_t_sel = [0.0, 0.1, 0.5, 1.0]
# beta_t_sel = [0.0, 0.2, 1.0]
beta_t_sel = [0.0, 0.1, 0.2, 0.5, 1.0]

# outlier dev
# out_dev_s = [0.0, 20.0, 200.0]
out_dev = 0.0

# tau2
# tau2_s = [None, 1.0]
tau2 = None


# ============================================================================
# other config


# ============================================================================

# set grid
n_obs_sel_grid, beta_t_sel_grid = np.meshgrid(
    n_obs_sel, beta_t_sel, indexing='ij')
sel_grid_shape = n_obs_sel_grid.shape
n_probl = n_obs_sel_grid.size

def probl_i_to_params(probl_i):
    n_obs = n_obs_sel_grid.flat[probl_i]
    beta_t = beta_t_sel_grid.flat[probl_i]
    return n_obs, beta_t

def params_to_probl_i(n_obs, beta_t):
    n_obs_i = n_obs_sel.index(n_obs)
    beta_t_i = beta_t_sel.index(beta_t)
    probl_i = np.ravel_multi_index(
        (n_obs_i, beta_t_i), sel_grid_shape)
    return probl_i


loo_pti_A = np.empty(n_probl, dtype=object)
loo_pti_B = np.empty(n_probl, dtype=object)
elpd_pt_A = np.zeros((n_probl, N_TRIAL))
elpd_pt_B = np.zeros((n_probl, N_TRIAL))

for probl_i in range(n_probl):
    n_obs, beta_t = probl_i_to_params(probl_i)
    run_i = params_to_run_i(n_obs, beta_t, out_dev, tau2)
    # load results
    res_file = np.load(
        '{}/{}.npz'
        .format(res_folder_name, str(run_i).zfill(4))
    )
    # fetch results
    loo_pti_A[probl_i] = res_file['loo_ti_A']
    loo_pti_B[probl_i] = res_file['loo_ti_B']
    elpd_pt_A[probl_i] = res_file['elpd_t_A']
    elpd_pt_B[probl_i] = res_file['elpd_t_B']
    # close file
    res_file.close()

n_trial = loo_pti_A[0].shape[0]

# calc some normally obtainable values
loo_pti = loo_pti_A-loo_pti_B
loo_pt = np.array([np.sum(loo_ti, axis=-1) for loo_ti in loo_pti])
loo_var_hat_pt = np.array(
    [loo_ti.shape[-1]*np.var(loo_ti, ddof=1, axis=-1) for loo_ti in loo_pti])
loo_skew_hat_pt = np.array(
    [stats.skew(loo_ti, axis=-1, bias=False) for loo_ti in loo_pti])
loo_napprox_pneg_pt = stats.norm.cdf(
    0, loc=loo_pt, scale=np.sqrt(loo_var_hat_pt))

# elpd
elpd_pt = elpd_pt_A - elpd_pt_B

# err
err_pt = loo_pt - elpd_pt

# elpd(y)
elpd_y_mean_p = np.mean(elpd_pt, axis=-1)
elpd_y_var_p = np.var(elpd_pt, ddof=1, axis=-1)
elpd_y_skew_p = stats.skew(elpd_pt, bias=False, axis=-1)
elpd_y_pneg_p = np.mean(elpd_pt<0, axis=-1)

# elpdhat(y)
loo_y_mean_p = np.mean(loo_pt, axis=-1)
loo_y_var_p = np.var(loo_pt, ddof=1, axis=-1)
loo_y_skew_p = stats.skew(loo_pt, bias=False, axis=-1)
loo_y_pneg_p = np.mean(loo_pt<0, axis=-1)

# err(y)
err_y_mean_p = np.mean(err_pt, axis=-1)
err_y_var_p = np.var(err_pt, ddof=1, axis=-1)
err_y_skew_p = stats.skew(err_pt, bias=False, axis=-1)
err_y_pneg_p = np.mean((err_pt)<0, axis=-1)

# BB moments
loo_mean_bb_pb, loo_sd_bb_pb, loo_skew_bb_pb = bb_mean_sd_skew(loo_pt)
elpd_mean_bb_pb, elpd_sd_bb_pb, elpd_skew_bb_pb = bb_mean_sd_skew(elpd_pt)
err_mean_bb_pb, err_sd_bb_pb, err_skew_bb_pb = bb_mean_sd_skew(err_pt)


# ============================================================================
# plot stuff
# ============================================================================

plt.rcParams['font.family'] = 'serif'
plt.rcParams["mathtext.fontset"] = "dejavuserif"

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


# # ============================================================================
# # plot moments (no uncertainty)
# # ============================================================================
#
# fontsize = 16
#
#
# datas = [
#     [loo_y_mean_p/np.sqrt(loo_y_var_p), loo_y_skew_p],
#     [elpd_y_mean_p/np.sqrt(elpd_y_var_p), elpd_y_skew_p],
#     [err_y_mean_p/np.sqrt(err_y_var_p), err_y_skew_p],
# ]
# data_names = [
#     r'$\widehat{\mathrm{elpd}}_\mathrm{LOO}$',
#     r'$\mathrm{elpd}$',
#     r'$\mathrm{err}_\mathrm{LOO}$'
# ]
# data_statistic_names = ['mean/SD', 'skewness']
# show_zero_line = [
#     [True, True],
#     [True, True],
#     [True, True],
# ]
#
#
# fig, axes = plt.subplots(
#     len(datas[0]), len(datas), sharex=True, figsize=(10,6))
#
# for d_j, data_i in enumerate(datas):
#
#     for d_i, data_ij in enumerate(data_i):
#
#         ax = axes[d_i, d_j]
#
#         for b_i, beta_t in enumerate(beta_t_sel):
#             color = 'C{}'.format(b_i)
#             label = r'${}$'.format(beta_t)
#
#             data = np.zeros(len(n_obs_sel))
#             for n_i, n_obs in enumerate(n_obs_sel):
#                 probl_i = params_to_probl_i(n_obs, beta_t)
#                 data[n_i] = data_ij[probl_i]
#
#             ax.plot(n_obs_sel, data, color=color, label=label)
#
#         if show_zero_line[d_j][d_i]:
#             ax.axhline(0, color='gray', lw=1.0)#, zorder=0)
#
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
#         ax.tick_params(axis='both', which='minor', labelsize=fontsize-4)
#
# # for ax in axes.ravel():
# #     ax.set_xscale('log')
#
# for ax, name in zip(axes[:, 0], data_statistic_names):
#     ax.set_ylabel(name, fontsize=fontsize)
#
# for ax, name in zip(axes[0, :], data_names):
#     ax.set_title(name, fontsize=fontsize)
#
# for ax in axes[-1, :]:
#     ax.set_xlabel(r'$n$', fontsize=fontsize-2)
#
# fig.tight_layout()
#
# axes[1, -1].legend(
#     loc='center left', bbox_to_anchor=(1, 0.5),
#     fontsize=fontsize-2, fancybox=False,
#     title=r'$\beta_t$',
#     title_fontsize=fontsize-2,
# )
#
# fig.subplots_adjust(right=0.88)



# ============================================================================
# plot moments BB
# ============================================================================

fontsize = 16

datas = [
    [loo_mean_bb_pb/loo_sd_bb_pb, loo_skew_bb_pb],
    [elpd_mean_bb_pb/elpd_sd_bb_pb, elpd_skew_bb_pb],
    [err_mean_bb_pb/err_sd_bb_pb, err_skew_bb_pb],
]
data_names = [
    r'$\widehat{\mathrm{elpd}}_\mathrm{LOO}$',
    r'$\mathrm{elpd}$',
    r'$\mathrm{err}_\mathrm{LOO}$'
]
data_statistic_names = ['mean/SD', 'skewness']
show_zero_line = [
    [True, True],
    [True, True],
    [True, True],
]

n_bb = datas[0][0].shape[-1]


fig, axes = plt.subplots(
    len(datas[0]), len(datas), sharex=True, figsize=(10,6))

for d_j, data_i in enumerate(datas):

    for d_i, data_ij in enumerate(data_i):

        ax = axes[d_i, d_j]

        for b_i, beta_t in enumerate(beta_t_sel):
            color = 'C{}'.format(b_i)
            label = r'${}$'.format(beta_t)

            data = np.zeros((len(n_obs_sel), n_bb))
            for n_i, n_obs in enumerate(n_obs_sel):
                probl_i = params_to_probl_i(n_obs, beta_t)
                data[n_i] = data_ij[probl_i]

            q025 = np.percentile(data, 2.5, axis=-1)
            q975 = np.percentile(data, 97.5, axis=-1)
            ax.fill_between(n_obs_sel, q025, q975, alpha=0.2, color=color)
            median = np.percentile(data, 50, axis=-1)
            ax.plot(n_obs_sel, median, color=color, label=label)

        if show_zero_line[d_j][d_i]:
            ax.axhline(0, color='gray', lw=1.0)#, zorder=0)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize-4)

# for ax in axes.ravel():
#     ax.set_xscale('log')

axes[0,-1].set_ylim([-0.31, 0.31])

for ax, name in zip(axes[:, 0], data_statistic_names):
    ax.set_ylabel(name, fontsize=fontsize)

for ax, name in zip(axes[0, :], data_names):
    ax.set_title(name, fontsize=fontsize)

for ax in axes[-1, :]:
    ax.set_xlabel(r'$n$', fontsize=fontsize-2)

fig.tight_layout()

axes[1, -1].legend(
    loc='center left', bbox_to_anchor=(1, 0.5),
    fontsize=fontsize-2, fancybox=False,
    title=r'$\beta_\Delta$',
    title_fontsize=fontsize-2,
)

fig.subplots_adjust(right=0.88)
