
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
from matplotlib.patches import ConnectionPatch
from matplotlib import patheffects

import seaborn as sns
import pandas as pd

from setup_all import *


# ============================================================================
# select problems

# number of obs in one trial
# n_obs_s = [16, 32, 64, 128, 256, 512, 1024]
probl_n_obs_s = [128, 128, 128]

# last covariate effect not used in model A
# beta_t_s = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
probl_beta_t_s = [1.0, 0.0, 1.0]

# outlier dev
# out_dev_s = [0.0, 20.0, 200.0]
probl_out_dev_s = [0.0, 0.0, 20.0]

# tau2
# tau2_s = [None, 1.0]
tau2 = None


# ============================================================================
# other config



# ============================================================================

n_probl = len(probl_n_obs_s)

loo_pti_A = np.empty(n_probl, dtype=object)
loo_pti_B = np.empty(n_probl, dtype=object)
elpd_pt_A = np.zeros((n_probl, N_TRIAL))
elpd_pt_B = np.zeros((n_probl, N_TRIAL))

for probl_i, (n_obs, beta_t, out_dev) in enumerate(zip(
        probl_n_obs_s, probl_beta_t_s, probl_out_dev_s)):
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
# plot
# ============================================================================


fontsize = 16

cmap = truncate_colormap(cm.get_cmap('Greys'), 0.3)
cmap.set_under('white', 1.0)

# kde_cmap = 'copper'
kde_cmap = truncate_colormap(cm.get_cmap('copper'), 0.3)
kde_n_levels = [6, 4, 5]


# example points (elpdhat, elpd)
custom_points = [
    [[-54.0, -42.0], [-44.0, -44.0], [-47.0, -47.0]],
    [[-0.5, 1.4], [0.6, 0.6], [0.942, 0.102]],
    [[37.0, -4.0], [25.0, -8.0], [6.0, -9.0]],
]
custom_hist_limits = [
    [-75, -25],
    [-3.2, 3.1],
    [-55, 67]
]

kde_plot = True
manual_zoom = True

fig = plt.figure(figsize=(10, 8))
outer = gridspec.GridSpec(
    2, 1, height_ratios = [1, 1],
    hspace=0.28,
    top=0.95,
    bottom=0.15,
    left=0.08,
    right=0.97
)
gs1 = gridspec.GridSpecFromSubplotSpec(
    1, n_probl, subplot_spec=outer[0,:], wspace=0.5)
gs2 = gridspec.GridSpecFromSubplotSpec(
    3, n_probl, subplot_spec=outer[1:,:], wspace=0.5, hspace=0.2)
axes = np.array(
    [[fig.add_subplot(gs1[probl_i]) for probl_i in range(n_probl)]]
    +
    [
        [fig.add_subplot(gs2[probl_i, row_i])for row_i in range(3)]
        for probl_i in range(n_probl)
    ]
)

# fig, axes = plt.subplots(
#     4, n_probl,
#     gridspec_kw={'height_ratios': [3, 1, 1, 1]},
#     figsize=(9, 8)
# )

for probl_i in range(n_probl):

    x_arr = loo_pt[probl_i]
    y_arr = elpd_pt[probl_i]

    # ===============
    # joint

    ax = axes[0, probl_i]

    if manual_zoom:
        if probl_i == 0:
            idxs = (
                (-75 < x_arr) & (x_arr < -25) &
                (-50 < y_arr) & (y_arr < -38)
            )
        if probl_i == 1:
            idxs = (
                (-1.5 < x_arr) & (x_arr < 3.0) &
                (-1 < y_arr) & (y_arr < 1.95)
            )
        if probl_i == 2:
            idxs = (
                (-10 < x_arr) & (x_arr < 60) &
                (-12 < y_arr) & (y_arr < 0)
            )
        x_arr = x_arr[idxs]
        y_arr = y_arr[idxs]

    ax.hexbin(x_arr, y_arr, gridsize=35, cmap=cmap, mincnt=1)
    if kde_plot:
        sns.kdeplot(
            x_arr, y_arr,
            n_levels=kde_n_levels[probl_i],
            cmap=kde_cmap, ax=ax
        )

    if probl_i == 0:
        ax.set_ylabel(r'$\mathrm{elpd}$', fontsize=fontsize-2)
    ax.set_xlabel(r'$\widehat{\mathrm{elpd}}$', fontsize=fontsize-2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if probl_i == 0:
        ax.set_xlim(left=-72)
        ax.set_ylim(bottom=-49.5)
    if probl_i == 1:
        ax.set_ylim(bottom=-0.7, top=2.05)
    if probl_i == 2:
        ax.set_xlim(left=-10)

    ax.autoscale(enable=False)
    ax.plot(
        [max(ax.get_xlim()[0], ax.get_ylim()[0]),
         min(ax.get_xlim()[1], ax.get_ylim()[1])],
        [max(ax.get_xlim()[0], ax.get_ylim()[0]),
         min(ax.get_xlim()[1], ax.get_ylim()[1])],
        # color=adjust_lightness('C3', amount=1.3)
        color='C2'
    )

    # ===============
    # custom points

    # x_min, x_max = axes[0, probl_i].get_xlim()
    x_min, x_max = custom_hist_limits[probl_i]
    x_grid = np.linspace(x_min, x_max, 100)

    for p_i, (ax, (elpdhat_target, elpd_target)) in enumerate(zip(
            axes[1:, probl_i], custom_points[probl_i])):

        # find closest taxi distance
        idx = (
            np.abs(loo_pt[probl_i] - elpdhat_target)
            + np.abs(elpd_pt[probl_i] - elpd_target)
        ).argmin()

        elpd = elpd_pt[probl_i, idx]
        loo = loo_pt[probl_i, idx]
        loo_i = loo_pti[probl_i][idx]
        sehat = np.sqrt(loo_var_hat_pt[probl_i, idx])
        n_obs = len(loo_i)
        elpdtilde = loo - (loo_pt[probl_i] - elpd_pt[probl_i])

        # plot elpdtilde hist
        _, _, hs_elpdtilde = ax.hist(
            elpdtilde,
            density=True,
            color=adjust_lightness('C0', amount=2.0),
            # alpha=0.4,
            bins=int(round((elpdtilde.max()-elpdtilde.min())/(x_max-x_min)*20)),
        )
        # plot normal approx
        dens = stats.norm.pdf(x_grid, loc=loo, scale=sehat)
        #dens *= ax.get_ylim()[1]*0.95/dens.max()
        h_elpdhat_n, = ax.plot(
            x_grid,
            dens,
            color='C1'
        )
        # plot loo
        h_elpdhat = ax.axvline(loo, color='C1')
        # plot elpd
        h_elpd = ax.axvline(elpd, color='C0', linestyle=':', linewidth=2.5)

        ax.set_xlim((x_min, x_max))
        if p_i < len(custom_points[0])-1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r'$\widehat{\mathrm{elpd}}$', fontsize=fontsize-2)

        ax.set_yticks([])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # add annotation
        axes[0, probl_i].plot(
            [loo], [elpd],
            'o', markersize=6,
            # markerfacecolor=adjust_lightness('C1', amount=1.3),
            markerfacecolor='C3',
            markeredgewidth=0.0,# markeredgecolor='k'
        )
        annotation_text = axes[0, probl_i].annotate(
            '{}'.format(p_i+1),
            xy=(loo, elpd),
            xytext=(-5, 0),
            ha='right',
            va='center',
            textcoords='offset points',
            # color=adjust_lightness('C1', amount=1.3),
            color='w',
            fontsize=fontsize,
        )
        annotation_text.set_path_effects([
            mpl.patheffects.Stroke(linewidth=1.8, foreground='k'),
            mpl.patheffects.Normal()
        ])

    # share hists col ylim
    max_ylim = max(map(lambda ax: ax.get_ylim()[1], axes[1:, probl_i]))
    for ax in axes[1:, probl_i]:
        ax.set_ylim(top=max_ylim)

for probl_i, probl_name in enumerate([
    'Clear case',
    'Models similar',
    'Outliers'
]):
    ax = axes[0, probl_i]
    ax.set_title(
        probl_name,
        fontsize=fontsize,
        pad=10,
    )

for row_i, ax in enumerate(axes[1:, 0]):
    label_text = ax.set_ylabel(
        '{}'.format(row_i+1),
        labelpad=25,
        # color=adjust_lightness('C1', amount=1.3),
        color='w',
        ha='right', va='center',
        fontsize=fontsize, rotation=0)
    label_text.set_path_effects([
        mpl.patheffects.Stroke(linewidth=1.8, foreground='k'),
        mpl.patheffects.Normal()
    ])

# legend
# h_elpd_vert = mpl.lines.Line2D(
#     [], [], color=h_elpd.get_color(), marker='|', linestyle='None',
#     markersize=14, markeredgewidth=1.5
# )
h_elpdhat_vert = mpl.lines.Line2D(
    [], [], color=h_elpdhat.get_color(), marker='|', linestyle='None',
    markersize=14, markeredgewidth=1.5
)
axes[-1,1].legend(
    [h_elpdhat_vert, h_elpd, h_elpdhat_n, hs_elpdtilde[0]],
    [r'$\widehat{\mathrm{elpd}}$',
        r'$\mathrm{elpd}$',
        'normal approx.',
        r'$\widetilde{\mathrm{elpd}}$'],
    fontsize=fontsize-2,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.7),
    fancybox=False,
    shadow=False,
    ncol=4,
    handler_map={h_elpd:HandlerMiniatureLine()}
)
