
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
n_obs_sel = [32, 128, 512]

# last covariate effect not used in model A
# beta_t_s = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
beta_t_sel = [0.0, 0.2, 1.0]

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
bb_plooneg = np.zeros((n_probl, N_TRIAL))
loobb_t_A = np.zeros((n_probl, N_TRIAL))

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
    bb_plooneg[probl_i] = res_file['bb_plooneg']
    loobb_t_A[probl_i] = res_file['loobb_t_A']
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


# ============================================================================
# plot
# ============================================================================

fontsize = 16

cmap = truncate_colormap(cm.get_cmap('Greys'), 0.3)
cmap.set_under('white', 1.0)

hex_instead_of_hist2d = True

ylims_always_01 = True

fig, axes = plt.subplots(
    len(beta_t_sel), len(n_obs_sel),
    sharex=True, sharey=True,
    figsize=(8, 8)
)

fig.subplots_adjust(
    top=0.936,
    bottom=0.104,
    left=0.232,
    right=0.968,
    hspace=0.119,
    wspace=0.185
)

empty_plots = [[2, 1], [2, 2]]

for n_obs_i, n_obs in enumerate(n_obs_sel):
    for beta_t_i, beta_t in enumerate(beta_t_sel):
        probl_i = params_to_probl_i(n_obs, beta_t)
        ax = axes[beta_t_i, n_obs_i]

        w_bb = 1-bb_plooneg[probl_i]
        w_loobb = loobb_t_A[probl_i]

        # shared lims
        if ylims_always_01:
            lim_min_w, lim_max_w = 0.0, 1.0
        else:
            lim_min_w = min(w_bb.min(), w_loobb.min())
            lim_max_w = max(w_bb.max(), w_loobb.max())
            lim_min_w, lim_max_w = (
                np.array([lim_min_w, lim_max_w]) +
                np.array([-1, 1])*0.02*(lim_max_w-lim_min_w)
            )
            lim_min_w = max(lim_min_w, 0.0)
            lim_max_w = min(lim_max_w, 1.0)
            if lim_min_w < 0.2:
                lim_min_w = 0.0
            if lim_max_w > 0.8:
                lim_max_w = 1.0

        if not hex_instead_of_hist2d:
            hist2d_lims = [
                np.linspace(lim_min_w, lim_max_w, 21),
                np.linspace(lim_min_w, lim_max_w, 21),
            ]

        if [beta_t_i, n_obs_i] in empty_plots:
            ax.text(
                0.5, 0.5,
                'All weights and\nprobabilities\nare 0.',
                transform=ax.transAxes,
                ha='center',
                va='center',
                fontsize=fontsize-2,
            )
            continue

        if hex_instead_of_hist2d:
            ax.hexbin(
                w_bb, w_loobb,
                gridsize=25,
                extent=(lim_min_w, lim_max_w, lim_min_w, lim_max_w),
                cmap=cmap, mincnt=1
            )
        else:
            ax.hist2d(
                w_bb, w_loobb,
                bins=hist2d_lims,
                cmap=cmap,
                cmin=1
            )

        ax.set_ylim([
            lim_min_w-0.03*(lim_max_w-lim_min_w),
            lim_max_w+0.03*(lim_max_w-lim_min_w)
        ])
        ax.set_xlim([
            lim_min_w-0.03*(lim_max_w-lim_min_w),
            lim_max_w+0.03*(lim_max_w-lim_min_w)
        ])

        ax.autoscale(enable=False)
        ax.plot(
            [max(ax.get_xlim()[0], ax.get_ylim()[0]),
             min(ax.get_xlim()[1], ax.get_ylim()[1])],
            [max(ax.get_xlim()[0], ax.get_ylim()[0]),
             min(ax.get_xlim()[1], ax.get_ylim()[1])],
            # color=adjust_lightness('C3', amount=1.3)
            color='C1'
        )

for ax in axes.ravel():
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize-4)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])

for ax, beta_t in zip(axes[:, 0], beta_t_sel):
    ax.set_ylabel(
        'LOO-BB-weight',
        fontsize=fontsize-2,
    )

for ax, beta_t in zip(axes[-1, :], n_obs_sel):
    ax.set_xlabel(
        r'$p(\widehat{\widetilde{\mathrm{elpd}}} > 0)$',
        rotation=0,
        fontsize=fontsize-2,
    )

for ax, beta_t in zip(axes[:, 0], beta_t_sel):
    ax.text(
        -0.55, 0.5,
        r'$\beta_t={}$'.format(beta_t),
        transform=ax.transAxes,
        ha='right',
        va='center',
        fontsize=fontsize,
    )

# set n labels
for ax, n_obs in zip(axes[0, :], n_obs_sel):
    ax.text(
        0.5, 1.10,
        r'$n={}$'.format(n_obs),
        transform=ax.transAxes,
        ha='center',
        va='bottom',
        fontsize=fontsize,
    )
