
import numpy as np
from scipy import linalg, stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import matplotlib.gridspec as gridspec
from matplotlib.collections import PathCollection

import seaborn as sns
import pandas as pd

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
# beta_t_sel = [0.0, 0.1, 0.2, 0.5, 1.0]
beta_t_sel = [0.0, 0.2, 1.0]


# outlier dev
# out_dev_s = [0.0, 20.0, 200.0]
out_dev = 0.0

# tau2
# tau2_s = [None, 1.0]
tau2 = None


# ============================================================================
# other config

bb_seed = 514762934
bb_n = 1000
bb_alpha = 1.0


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

fig, axes = plt.subplots(
    len(beta_t_sel), 1, figsize=(7, 10),
    sharex=True, #sharey=True
)

data_pt = err_pt

for b_i, (ax, beta_t) in enumerate(zip(axes, beta_t_sel)):

    # ax.axhline(1.0, color='red')
    ax.axhline(1.0, color=adjust_lightness('red', amount=1.3))

    data = np.zeros((len(n_obs_sel), data_pt.shape[-1]))
    for n_i, n_obs in enumerate(n_obs_sel):
        probl_i = params_to_probl_i(n_obs, beta_t)
        data[n_i] = data_pt[probl_i]

    # sns.violinplot(
    #     data=pd.DataFrame(data.T),
    #     scale='width',
    #     # cut=0,
    #     ax=ax,
    #     color=adjust_lightness('C0', amount=1.3),
    #     # color='C0',
    #     #saturation=0.6,
    # )

    sns.boxenplot(
        data=pd.DataFrame(data.T),
        ax=ax,
        color=adjust_lightness('C0', amount=1.3),
        # showfliers=False, # requires 0.10.0 and pytohn3.6
        # color='C0',
        #saturation=0.6,
    )
    for l in ax.lines[1:]:
        l.set_linewidth(1.2)
        l.set_color('k')
        l.set_alpha(1)
    # manual remove of outliers
    for child in ax.get_children():
        if not isinstance(child, PathCollection):
            continue
        if child.get_array() is not None:
            continue
        child.set_visible(False)

    ax.set_xticklabels(n_obs_sel)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize-4)

    # ax.set_yticks([0.0, 0.5, 1.0, 1.5])

    ax.set_ylim(top=max(np.percentile(data, 97.5, axis=-1)))
    ax.set_ylim(bottom=min(np.percentile(data, 2.5, axis=-1)))

    ax.set_ylabel(r'$\beta_t={}$'.format(beta_t), fontsize=fontsize)

axes[-1].set_xlabel(r'$n$', fontsize=fontsize)

# set ylable labels
ylabel = (
    r'${}^\mathrm{sv}\mathrm{err}_\mathrm{LOO}(\mathrm{M}_a,\mathrm{M}_b|y)$'
    # r'$\mathrm{err}_\mathrm{LOO}$'
)
ax = axes[len(axes)//2]
ax.text(
    -0.2, 0.5,
    ylabel,
    transform=ax.transAxes,
    ha='right',
    va='center',
    rotation=90,
    fontsize=fontsize,
)

fig.tight_layout()
