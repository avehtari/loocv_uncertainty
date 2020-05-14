
import numpy as np
from scipy import linalg, stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import matplotlib.gridspec as gridspec

import seaborn as sns
import pandas as pd

from setup_all import *


# ============================================================================
# select problems

# number of obs in one trial
# n_obs_s = [16, 32, 64, 128, 256, 512, 1024]
# n_obs_sel = [16, 64, 256, 1024]
n_obs_sel = [32, 128, 512]

# last covariate effect not used in model A
# beta_t_s = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
# beta_t_sel = [0.0, 0.1, 0.5, 1.0]
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
# plot joints
# ============================================================================

fontsize = 16
n_bins = 25

datas = [
    loo_pt,
    np.mean(loo_pt, axis=-1, keepdims=True) - (loo_pt - elpd_pt),
]
names = [
    r'${}^\mathrm{sv}\widehat{\mathrm{elpd}}_{\mathrm{LOO}}(y)$',
    r'${}^\mathrm{sv}\widetilde{\mathrm{elpd}}_{\mathrm{LOO}}(y^\mathrm{obs})$',
]

fig = plt.figure(figsize=(10, 10))
outer = gridspec.GridSpec(
    len(beta_t_sel), len(n_obs_sel),
    hspace=0.38,
    wspace=0.34,
    top=0.94,
    bottom=0.03,
    left=0.2,
    right=0.95
)
gs_s = np.array([
    [gridspec.GridSpecFromSubplotSpec(
        len(datas), 1, subplot_spec=outer[gs_i, gs_j], hspace=0.1)
        for gs_j in range(len(n_obs_sel))]
    for gs_i in range(len(beta_t_sel))
])
axes = np.array([
    [fig.add_subplot(gs_s[gs_i, gs_j][data_i])
        for gs_j in range(len(n_obs_sel))]
    for gs_i in range(len(beta_t_sel))
    for data_i in range(len(datas))
])

for n_obs_i, n_obs in enumerate(n_obs_sel):
    for beta_t_i, beta_t in enumerate(beta_t_sel):
        probl_i = params_to_probl_i(n_obs, beta_t)

        xmin = min(
            np.percentile(datas[data_i][probl_i], 2.5)
            for data_i in range(len(datas))
        )
        xmax = max(
            np.percentile(datas[data_i][probl_i], 97.5)
            for data_i in range(len(datas))
        )
        xmin, xmax = 0.05*(xmax-xmin)*np.array([-1,1])+np.array([xmin,xmax])
        binlims = np.linspace(xmin, xmax, n_bins+1)

        axes_cur = axes[beta_t_i*len(datas):(beta_t_i+1)*len(datas), n_obs_i]
        for data_i, ax in enumerate(axes_cur):
            data = datas[data_i][probl_i]

            ax.hist(
                data,
                bins=binlims,
                color=adjust_lightness('C0', amount=1.6)
            )
            ax.axvline(np.mean(data), color='C1')

            # ax.axvline(np.percentile(data, 50), color='C1')
            # ax.axvline(
            #     np.percentile(data, 2.5),
            #     color=adjust_lightness('C1', amount=1.2)
            # )
            # ax.axvline(
            #     np.percentile(data, 97.5),
            #     color=adjust_lightness('C1', amount=1.2)
            # )

            # set row title
            if n_obs_i == 0:
                ax.set_ylabel(
                    names[data_i],
                    rotation=0,
                    ha='right',
                    va='center',
                    fontsize=fontsize-2,
                )

            if data_i < len(datas)-1:
                ax.set_xticklabels([])

            # set sd text
            sdtext = (
                'SD:\n{:.2}'.format(np.std(data, ddof=1))
                if np.std(data, ddof=1) < 10 else
                'SD:\n{}'.format(int(round(np.std(data, ddof=1))))
            )
            ax.text(
                1.01, 0.1,
                sdtext,
                transform=ax.transAxes,
                ha='left',
                va='bottom',
                fontsize=fontsize-2,
            )

        # share ylim
        maxy = max(ax.get_ylim()[1] for ax in axes_cur)
        for ax in axes_cur:
            ax.set_ylim(top=maxy)

for ax in axes.ravel():
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize-4)

# set n labels
for ax, n_obs in zip(axes[0, :], n_obs_sel):
    # ax.set_title(r'$n={}$'.format(n_obs), fontsize=fontsize)
    ax.text(
        0.5, 1.25,
        r'$n={}$'.format(n_obs),
        transform=ax.transAxes,
        ha='center',
        va='bottom',
        fontsize=fontsize,
    )

# set beta labels
for beta_t_i, beta_t in enumerate(beta_t_sel):
    ax = axes[beta_t_i*len(datas)+1, 0]
    ax.text(
        -0.54, 1.1,
        r'$\beta_t={}$'.format(beta_t),
        transform=ax.transAxes,
        ha='right',
        va='center',
        fontsize=fontsize,
    )
