
import numpy as np
from scipy import linalg, stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import matplotlib.gridspec as gridspec

# import seaborn as sns
# import pandas as pd

from setup_linreg import *


# ============================================================================
# select problems

# number of obs in one trial
# n_obs_s = [16, 32, 64, 128, 256, 512, 1024]
n_obs_sel = [32, 128, 512]

# last covariate effect not used in model A
# beta_t_s = [0.0, 0.1, 0.2, 0.5, 1.0]
beta_t_sel = [0.0, 0.2, 1.0]

# outlier dev
# out_dev_s = [0.0, 20.0]
out_dev = 0.0

# tau2
# tau2_s = [None]
tau2 = None


# ============================================================================

cal_limits = np.linspace(0, 1, cal_nbins+1)

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

n_trial = N_TRIAL

loo_pt = np.zeros((n_probl, n_trial))
elpd_pt = np.zeros((n_probl, n_trial))
var_hat_n_pt = np.zeros((n_probl, n_trial))
var_hat_bb_pt = np.zeros((n_probl, n_trial))
var_hat_impr_pt = np.zeros((n_probl, n_trial))
var_hat_impr2_pt = np.zeros((n_probl, n_trial))
cal_counts_n = np.zeros((n_probl, cal_nbins))
cal_counts_bb = np.zeros((n_probl, cal_nbins))
cal_counts_impr = np.zeros((n_probl, cal_nbins))
cal_counts_impr2 = np.zeros((n_probl, cal_nbins))

for probl_i in range(n_probl):
    n_obs, beta_t = probl_i_to_params(probl_i)
    run_i = params_to_run_i(n_obs, beta_t, out_dev, tau2)
    # load results
    res_file = np.load(
        '{}/{}.npz'
        .format(res_folder_name, str(run_i).zfill(4))
    )
    # fetch results
    loo_pt[probl_i] = res_file['loo_t']
    elpd_pt[probl_i] = res_file['elpd_t']
    var_hat_n_pt[probl_i] = res_file['var_hat_n_t']
    var_hat_bb_pt[probl_i] = res_file['var_hat_bb_t']
    var_hat_impr_pt[probl_i] = res_file['var_hat_impr_t']
    var_hat_impr2_pt[probl_i] = res_file['var_hat_impr2_t']
    cal_counts_n[probl_i] = res_file['cal_counts_n']
    cal_counts_bb[probl_i] = res_file['cal_counts_bb']
    cal_counts_impr[probl_i] = res_file['cal_counts_impr']
    cal_counts_impr2[probl_i] = res_file['cal_counts_impr2']
    # close file
    res_file.close()

# manually add improved normal calibration
cal_counts_imprn = np.zeros((n_probl, cal_nbins), dtype=int)
for probl_i in range(n_probl):
    cdf_elpd = stats.norm.cdf(
        elpd_pt[probl_i],
        loc=loo_pt[probl_i],
        scale=np.sqrt(var_hat_impr_pt[probl_i])
    )
    cal_counts_imprn[probl_i] = np.histogram(cdf_elpd, cal_limits)[0]
cal_counts_imprn2 = np.zeros((n_probl, cal_nbins), dtype=int)
for probl_i in range(n_probl):
    cdf_elpd = stats.norm.cdf(
        elpd_pt[probl_i],
        loc=loo_pt[probl_i],
        scale=np.sqrt(var_hat_impr2_pt[probl_i])
    )
    cal_counts_imprn2[probl_i] = np.histogram(cdf_elpd, cal_limits)[0]



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
# plot cal hists
# ============================================================================

q005 = stats.binom.ppf(0.005, n_trial, 1/cal_nbins)
q995 = stats.binom.ppf(0.995, n_trial, 1/cal_nbins)

fontsize = 16

datas = [
    cal_counts_n,
    cal_counts_bb,
    cal_counts_impr,
    # cal_counts_impr2,
    # cal_counts_imprn,
    # cal_counts_imprn2,
]
names = [
    'normal',
    'BB',
    'improved',
]


fig = plt.figure(figsize=(11, 14))
outer = gridspec.GridSpec(
    len(beta_t_sel), len(n_obs_sel),
    hspace=0.2,
    wspace=0.34,
    top=0.94,
    bottom=0.06,
    left=0.21,
    right=0.94
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

        for data_i in range(len(datas)):
            data = datas[data_i][probl_i]
            ax = axes[beta_t_i*len(datas)+data_i, n_obs_i]
            ax.bar(
                cal_limits[:-1],
                data,
                width=1/cal_nbins,
                align='edge',
                color=adjust_lightness('C0', amount=1.6),
            )

            # set row title
            if n_obs_i == 0:
                ax.set_ylabel(
                    names[data_i],
                    rotation=0,
                    ha='right',
                    va='center',
                    fontsize=fontsize-2,
                )

        # share ylims among each prolem
        maxy = np.max([
            ax.get_ylim()[1]
            for ax in axes[
                beta_t_i*len(datas):(beta_t_i+1)*len(datas), n_obs_i]
        ])
        for ax in axes[beta_t_i*len(datas):(beta_t_i+1)*len(datas), n_obs_i]:
            ax.set_ylim(0, maxy)

for ax in axes.ravel():
    ax.fill_between(
        [0,1], [q995, q995], [q005, q005], color='C1', alpha=0.3,
        zorder=2)
    ax.set_xlim((0, 1))
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize-4)

for ax in axes[:-1,:].ravel():
    ax.set_xticklabels([])

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

# set approx title
ax = axes[0, 0]
ax.text(
    -0.1, 1.25,
    'approximated\nwith',
    transform=ax.transAxes,
    ha='right',
    va='bottom',
    fontsize=fontsize,

)

for ax in axes[-1,:]:
    ax.set_xlabel(
        r'$p(\widehat{\widetilde{\mathrm{elpd}}} < \mathrm{elpd})$',
        fontsize=fontsize-3
    )

# set beta labels
for beta_t_i, beta_t in enumerate(beta_t_sel):
    ax = axes[beta_t_i*len(datas)+1, 0]
    ax.text(
        -0.65, 0.5,
        r'$\beta_t={}$'.format(beta_t),
        transform=ax.transAxes,
        ha='right',
        va='center',
        fontsize=fontsize,
    )
