
import numpy as np
from scipy import linalg, stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import matplotlib.gridspec as gridspec
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.path as mpath
from matplotlib.transforms import BboxTransformFrom, BboxTransformTo, Bbox

import seaborn as sns
import pandas as pd

from setup_indep import *


# ============================================================================
# select problem

# number of obs in one trial
# n_obs_s = [32, 128, 512]
n_obs = 128

# last covariate effect not used in model A
# beta_t_s = [0.0, 1.0]
beta_t = 0.0

# outlier dev
# out_dev_s = [0.0, 20.0, 200.0]
out_dev = 0.0

# tau2
# tau2_s = [None, 1.0]
tau2 = None


# ============================================================================

run_i = params_to_run_i(n_obs, beta_t, out_dev, tau2)

# load results
res_file = np.load(
    '{}/{}.npz'
    .format(res_folder_name, str(run_i).zfill(4))
)
# fetch results
loo_ti_A = res_file['loo_ti_A']
loo_ti_B = res_file['loo_ti_B']
looindep_ti_A = res_file['looindep_ti_A']
looindep_ti_B = res_file['looindep_ti_B']
elpd_t_A = res_file['elpd_t_A']
elpd_t_B = res_file['elpd_t_B']
# close file
res_file.close()


# calc some normally obtainable values
loo_ti = loo_ti_A-loo_ti_B
loo_t = np.sum(loo_ti, axis=-1)
loo_var_hat_t = n_obs*np.var(loo_ti, ddof=1, axis=-1)
loo_skew_hat_t = stats.skew(loo_ti, axis=-1, bias=False)
loo_napprox_pneg_t = stats.norm.cdf(0, loc=loo_t, scale=np.sqrt(loo_var_hat_t))

# looindep
looindep_ti = looindep_ti_A-looindep_ti_B
looindep_t = np.sum(looindep_ti, axis=-1)
looindep_var_hat_t = n_obs*np.var(looindep_ti, ddof=1, axis=-1)
looindep_skew_hat_t = stats.skew(looindep_ti, axis=-1, bias=False)
looindep_napprox_pneg_t = stats.norm.cdf(
    0, loc=looindep_t, scale=np.sqrt(looindep_var_hat_t))

# elpd
elpd_t = elpd_t_A - elpd_t_B

# elpd(y)
elpd_y_mean = np.mean(elpd_t)
elpd_y_var = np.var(elpd_t, ddof=1)
elpd_y_skew = stats.skew(elpd_t, bias=False)
elpd_y_pneg = np.mean(elpd_t<0)

# elpdhat(y)
loo_y_mean = np.mean(loo_t)
loo_y_var = np.var(loo_t, ddof=1)
loo_y_skew = stats.skew(loo_t, bias=False)
loo_y_pneg = np.mean(loo_t<0)

# elpdhat(y) indep
looindep_y_mean = np.mean(looindep_t)
looindep_y_var = np.var(looindep_t, ddof=1)
looindep_y_skew = stats.skew(looindep_t, bias=False)
looindep_y_pneg = np.mean(looindep_t<0)


# ============================================================================
# plot funcs
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

# for the custom legend marker
class HandlerMiniatureLine(HandlerLine2D):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):

        legline, _ = HandlerLine2D.create_artists(self,legend, orig_handle,
                                xdescent, ydescent, width, height, fontsize, trans)

        legline.set_data(*orig_handle.get_data())

        ext = mpath.get_paths_extents([orig_handle.get_path()])
        if ext.width == 0:
            ext.x0 -= 0.1
            ext.x1 += 0.1
        bbox0 = BboxTransformFrom(ext)
        bbox1 = BboxTransformTo(Bbox.from_bounds(xdescent, ydescent, width, height))

        legline.set_transform(bbox0 + bbox1 + trans)
        return legline,

# ============================================================================
# Plot joint + hist
# ============================================================================

if False:

    figsize = (10, 10)
    fontsize = 16
    hex_gridsize = 50
    n_bins = 20

    cmap = truncate_colormap(cm.get_cmap('Greys'), 0.3)
    cmap.set_under('white', 1.0)

    arr_s = [elpd_t, looindep_t, loo_t]
    name_s = [
        r'$\mathrm{elpd}(y)$',
        r'$\widehat{\mathrm{elpd}}_\mathrm{LOO}(\tilde{y}, y)$',
        r'$\widehat{\mathrm{elpd}}_\mathrm{LOO}(y)$',
    ]

    # ---------------------------------------------------------------------------

    fig, axes = plt.subplots(
        len(arr_s), len(arr_s), figsize=figsize)

    for i, arr_i in enumerate(arr_s):

        ax = axes[i, i]
        ax2 = ax.twinx()
        ax2.hist(arr_i, bins=n_bins)
        ax2.set_yticks([])

        for ax_cur in [ax, ax2]:
            ax_cur.spines['top'].set_visible(False)
            # ax_cur.spines['left'].set_visible(False)
            ax_cur.spines['right'].set_visible(False)

        ax.set_yticks(ax.get_xticks())
        ax.set_ylim(ax.get_xlim())

        for j, arr_j in enumerate(arr_s):
            if j == i:
                continue

            for ax, (arr_x, arr_y) in [
                    [axes[i, j], [arr_j, arr_i]],
                    [axes[j, i], [arr_i, arr_j]]]:

                ax.hexbin(
                    arr_x, arr_y, gridsize=hex_gridsize, cmap=cmap, mincnt=1)

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                ax.autoscale(enable=False)
                ax.plot(
                    [min(arr_y.min(), arr_x.min()),
                     max(arr_y.max(), arr_x.max())],
                    [min(arr_y.min(), arr_x.min()),
                     max(arr_y.max(), arr_x.max())],
                    color=adjust_lightness('C3', amount=1.3)
                )

    for i in range(len(arr_s)):
        for j in range(len(arr_s)):
            if j == i:
                continue
            ax = axes[i, j]
            ax.set_xticks(axes[j, j].get_xticks())
            ax.set_yticks(axes[i, i].get_xticks())
            ax.set_xlim(axes[j, j].get_xlim())
            ax.set_ylim(axes[i, i].get_xlim())

    for i, name in enumerate(name_s):
        axes[-1, i].set_xlabel(name, fontsize=fontsize)
        axes[i, 0].set_ylabel(name, fontsize=fontsize)

    for ax in axes[:-1,:].ravel():
        ax.set_xticklabels([])
    for ax in axes[:,1:].ravel():
        ax.set_yticklabels([])

    fig.tight_layout()

    # ======================================
    # Plot joint + hist seaborn
    # ======================================

    # df = pd.DataFrame(dict(loo=loo_t, looindep=looindep_t, elpd=elpd_t))
    # g = sns.PairGrid(df)
    # g.map_diag(plt.hist, bins=n_bins)
    # g.map_offdiag(plt.hexbin, gridsize=hex_gridsize, cmap=cmap, mincnt=1)


# ============================================================================
# Plot joint + hist
# ============================================================================

if False:
    figsize = (10, 5)
    fontsize = 16
    hex_gridsize = 50
    hist_n_bins = 26
    hist_prctile_min = 1
    hist_prctile_max = 99

    x_arr = loo_t
    x_name = r'$\widehat{\mathrm{elpd}}_\mathrm{LOO}(y)$'
    y_arr = looindep_t
    y_name = r'$\widehat{\mathrm{elpd}}_\mathrm{LOO}(\tilde{y}, y)$'

    cmap = truncate_colormap(cm.get_cmap('Greys'), 0.3)
    cmap.set_under('white', 1.0)



    bin_lims = np.linspace(
        min(np.percentile(x_arr, 1), np.percentile(y_arr, 1)),
        max(np.percentile(x_arr, 99), np.percentile(y_arr, 99)),
        hist_n_bins+1
    )


    fig = plt.figure(figsize=figsize)
    outer = gridspec.GridSpec(
        1, 2, width_ratios = [1, 1],
        wspace=0.10,
        top=0.96,
        bottom=0.15,
        left=0.1,
        right=0.97
    )
    gs1 = gridspec.GridSpecFromSubplotSpec(
        1, 1, subplot_spec=outer[0, 0])
    gs2 = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[0, 1], hspace=0.4)
    ax_joint = fig.add_subplot(gs1[0, 0])
    ax_hists = [fig.add_subplot(gs2[i, 0]) for i in range(2)]


    ax_joint.hexbin(x_arr, y_arr, gridsize=hex_gridsize, cmap=cmap, mincnt=1)
    ax_joint.set_xlabel(x_name, fontsize=fontsize)
    ax_joint.set_ylabel(y_name, fontsize=fontsize)
    ax_joint.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax_joint.spines['top'].set_visible(False)
    ax_joint.spines['right'].set_visible(False)
    ax_joint.autoscale(enable=False)
    ax_joint.plot(
        [min(y_arr.min(), x_arr.min()),
         max(y_arr.max(), x_arr.max())],
        [min(y_arr.min(), x_arr.min()),
         max(y_arr.max(), x_arr.max())],
        color=adjust_lightness('C3', amount=1.3)
    )

    for ax, data, name in zip(ax_hists, [x_arr, y_arr], [x_name, y_name]):
        ax.hist(data, bins=bin_lims)
        mean_line = ax.axvline(
            np.mean(data),
            color=adjust_lightness('C3', amount=1.3),
            # color='C2',
        )
        ax.set_xlabel(name, fontsize=fontsize)
        ax.set_yticks([])
        ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

    mean_line_marker = mpl.lines.Line2D(
        [], [], color=mean_line.get_color(), marker='|', linestyle='None',
        markersize=14, markeredgewidth=1.5
    )
    ax_hists[0].legend(
        [mean_line_marker],
        ['mean'],
        fontsize=fontsize-2,
        fancybox=False,
        shadow=False,
    )



# ============================================================================
# Plot joint + hist also elpd
# ============================================================================

figsize = (10, 6)
fontsize = 16
hex_gridsize = 50
hist_n_bins = 26
hist_prctile_min = 1
hist_prctile_max = 99

fixed_lims = True
lim_min = -2.3
lim_max = 4.3

kde_plot = True

x_arr = loo_t
x_name = r'$\widehat{\mathrm{elpd}}_\mathrm{LOO}(y)$'
y_arr = looindep_t
y_name = r'$\widehat{\mathrm{elpd}}_\mathrm{LOO}(\tilde{y}, y)$'
z_arr = elpd_t
z_name = r'$\mathrm{elpd}(y)$'

cmap = truncate_colormap(cm.get_cmap('Greys'), 0.3)
cmap.set_under('white', 1.0)

# cmap_kde = 'copper'
cmap_kde = truncate_colormap(cm.get_cmap('copper'), 0.3)

if fixed_lims:
    bin_lims = np.linspace(
        min(
            np.percentile(x_arr, 1),
            np.percentile(y_arr, 1),
            np.percentile(z_arr, 1),
        ),
        max(
            np.percentile(x_arr, 99),
            np.percentile(y_arr, 99),
            np.percentile(z_arr, 99),
        ),
        hist_n_bins+1
    )
else:
    bin_lims = np.linspace(lim_min, lim_max, hist_n_bins+1)


fig = plt.figure(figsize=figsize)
outer = gridspec.GridSpec(
    1, 2, width_ratios = [1, 1],
    wspace=0.10,
    top=0.99,
    bottom=0.11,
    left=0.09,
    right=0.97
)
gs1 = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=outer[0, 0], hspace=0.4)
gs2 = gridspec.GridSpecFromSubplotSpec(
    3, 1, subplot_spec=outer[0, 1], hspace=0.4)
ax_joints = [fig.add_subplot(gs1[i, 0]) for i in range(2)]
ax_hists = [fig.add_subplot(gs2[i, 0]) for i in range(3)]

joint_line_diag_s = []
for ax, (data_1, data_2), (name_1, name_2) in zip(
        ax_joints,
        [[y_arr, z_arr], [y_arr, x_arr]],
        [[y_name, z_name], [y_name, x_name]]):
    if fixed_lims:
        idxs = (
            (lim_min<data_1) & (data_1<lim_max) &
            (lim_min<data_2) & (data_2<lim_max)
        )
        data_1 = data_1[idxs]
        data_2 = data_2[idxs]
    ax.hexbin(data_1, data_2, gridsize=hex_gridsize, cmap=cmap, mincnt=1)
    if kde_plot:
        sns.kdeplot(data_1, data_2, n_levels=7, cmap=cmap_kde, ax=ax)
    ax.set_xlabel(name_1, fontsize=fontsize)
    ax.set_ylabel(name_2, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.autoscale(enable=False)
    joint_line_diag, = ax.plot(
        [max(ax.get_xlim()[0], ax.get_ylim()[0]),
         min(ax.get_xlim()[1], ax.get_ylim()[1])],
        [max(ax.get_xlim()[0], ax.get_ylim()[0]),
         min(ax.get_xlim()[1], ax.get_ylim()[1])],
        # color=adjust_lightness('C3', amount=1.3)
        color='C2'
    )
    joint_line_diag_s.append(joint_line_diag)

for ax, data, name in zip(
        ax_hists,
        [z_arr, y_arr, x_arr],
        [z_name, y_name, x_name]):
    ax.hist(data, bins=bin_lims)
    mean_line = ax.axvline(
        np.mean(data),
        # color=adjust_lightness('C3', amount=1.3),
        color='C1',
    )
    ax.set_xlabel(name, fontsize=fontsize)
    ax.set_yticks([])
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ax_joints[0].legend(
#     [joint_line_diag_s[0]],
#     ['x=y'],
#     fontsize=fontsize-2,
#     fancybox=False,
#     shadow=False,
#     handler_map={joint_line_diag_s[0]:HandlerMiniatureLine()}
# )

mean_line_marker = mpl.lines.Line2D(
    [], [], color=mean_line.get_color(), marker='|', linestyle='None',
    markersize=14, markeredgewidth=1.5
)
ax_hists[0].legend(
    [mean_line_marker],
    ['mean'],
    fontsize=fontsize-2,
    fancybox=False,
    shadow=False,
)

for ax in ax_hists[:-1]:
    ax.set_xticklabels([])
