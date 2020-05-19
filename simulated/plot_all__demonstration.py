
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
probl_n_obs_s = [128, 128, 128, 32]

# last covariate effect not used in model A
# beta_t_s = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
probl_beta_t_s = [1.0, 0.0, 1.0, 1.0]

# outlier dev
# out_dev_s = [0.0, 20.0, 200.0]
probl_out_dev_s = [0.0, 0.0, 20.0, 0.0]

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


# example points (elpdhat, elpd)
custom_points = [
    [[-58.0, -41.0], [-44.0, -44.0]],
    [[-0.8, 1.6], [0.95, 0.11]],
    [[25.0, -8.0], [37.0, -4.0]],
    [[-11.0, -11.0], [0.0, -10.0]],
]


kde_plot = True
# kde_cmap = 'copper'
kde_cmap = truncate_colormap(cm.get_cmap('copper'), 0.3)
kde_n_levels = [0.05, 0.3, 0.6, 0.9]


zoom_limits = [
    [(-75, -21), (-53, -31)],
    [(-1.5, 3.0), (-0.8, 1.95)],
    [(-10, 60), (-13, 1.0)],
    [(-30, 5), (-20, 3)],
]

fig, axes = plt.subplots(
    n_probl, max(map(len, custom_points))+1,
    gridspec_kw={'width_ratios': [2, 3, 3]},
    figsize=(9, 8)
)

fig.subplots_adjust(
    top=0.942,
    bottom=0.140,
    left=0.198,
    right=0.979,
    hspace=0.287,
    wspace=0.125
)

for probl_i in range(n_probl):

    x_arr = loo_pt[probl_i]
    y_arr = elpd_pt[probl_i]

    # ===============
    # joint

    ax = axes[probl_i, 0]

    (zoom_xmin, zoom_xmax), (zoom_ymin, zoom_ymax) = zoom_limits[probl_i]
    idxs = (
        (zoom_xmin < x_arr) & (x_arr < zoom_xmax) &
        (zoom_ymin < y_arr) & (y_arr < zoom_ymax)
    )
    x_arr = x_arr[idxs]
    y_arr = y_arr[idxs]

    ax.hexbin(x_arr, y_arr, gridsize=35, cmap=cmap, mincnt=1)
    if kde_plot:
        # sns.kdeplot(
        #     x_arr, y_arr,
        #     levels=kde_n_levels,
        #     cmap=kde_cmap, ax=ax,
        #     linewidths=1.0
        # )
        kde = stats.gaussian_kde(
            np.stack([x_arr, y_arr], axis=0),
            bw_method=0.5
        )
        X, Y = np.mgrid[zoom_xmin:zoom_xmax:100j, zoom_ymin:zoom_ymax:100j]
        kde_positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kde(kde_positions).T, X.shape)
        if not np.isscalar(kde_n_levels):
            levels = (
                Z.min() + (Z.max()-Z.min())*np.asarray(kde_n_levels)
            )
        else:
            levels = kde_n_levels
        ax.contour(X, Y, Z, levels=levels, cmap=kde_cmap)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.autoscale(enable=False)
    ax.plot(
        [max(ax.get_xlim()[0], ax.get_ylim()[0]),
         min(ax.get_xlim()[1], ax.get_ylim()[1])],
        [max(ax.get_xlim()[0], ax.get_ylim()[0]),
         min(ax.get_xlim()[1], ax.get_ylim()[1])],
        # color=adjust_lightness('C3', amount=1.3)
        color='C2'
    )
    if ax.get_xlim()[0] < 0 and ax.get_xlim()[1] > 0:
        ax.axvline(
            0, color=adjust_lightness('gray', amount=1.4),
            zorder=1, lw=0.8)
    if ax.get_ylim()[0] < 0 and ax.get_ylim()[1] > 0:
        ax.axhline(
            0, color=adjust_lightness('gray', amount=1.4),
            zorder=1, lw=0.8)

    # ===============
    # custom points

    for p_i, (ax, (elpdhat_target, elpd_target)) in enumerate(zip(
            axes[probl_i, 1:], custom_points[probl_i])):

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

        elpdtilde_min, elpdtilde_max = np.percentile(elpdtilde, [0.1, 99])
        elpdhat_min, elpdhat_max = loo + 3*sehat*np.array([-1, 1])
        x_min = min(elpdtilde_min, elpdhat_min)
        x_max = max(elpdtilde_max, elpdhat_max)
        x_grid = np.linspace(x_min, x_max, 100)

        # plot elpdtilde hist
        _, _, hs_elpdtilde = ax.hist(
            elpdtilde,
            density=True,
            color=adjust_lightness('C0', amount=2.0),
            # alpha=0.4,
            bins=int(round((elpdtilde.max()-elpdtilde.min())/(x_max-x_min)*32)),
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
        # h_elpdhat = ax.axvline(loo, color='C1')
        # plot elpd
        h_elpd = ax.axvline(
            elpd,
            color='C0',
            # linestyle=':',
            # linewidth=2.5
        )

        ax.set_xlim((x_min, x_max))

        ax.set_yticks([])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # add annotation
        axes[probl_i, 0].plot(
            [loo], [elpd],
            'o', markersize=6,
            # markerfacecolor=adjust_lightness('C1', amount=1.3),
            markerfacecolor='C3',
            markeredgewidth=0.0,# markeredgecolor='k'
        )
        annotation_text = axes[probl_i, 0].annotate(
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

    # share hists row ylim
    max_ylim = max(map(lambda ax: ax.get_ylim()[1], axes[probl_i, 1:]))
    for ax in axes[probl_i, 1:]:
        ax.set_ylim(top=max_ylim)

# for ax in axes[-1, 1:]:
#     ax.set_xlabel(
#         r'$\widehat{\mathrm{elpd}}_\mathrm{LOO}$',
#         fontsize=fontsize-2
#     )

axes[-1, 0].set_xlabel(
    r'$\widehat{\mathrm{elpd}}_\mathrm{LOO}(M_a,M_b|y)$',
    fontsize=fontsize-2
)

for ax in axes[:, 0]:
    ax.text(
        -0.25, 0.5,
        r'$\widehat{\mathrm{elpd}}_\mathrm{LOO}$',
        transform=ax.transAxes,
        ha='right',
        va='center',
        fontsize=fontsize-2,
        rotation=90,
    )

for probl_i, ax in enumerate(axes[:, 0]):
    ax.text(
        -0.52, 0.5,
        (
            r'$\beta_t={}$'.format(probl_beta_t_s[probl_i]) +
            '\n' +
            r'$n={}$'.format(probl_n_obs_s[probl_i]) +
            '\n' +
            r'$\mu_{\star,r}=' + '{}$'.format(probl_out_dev_s[probl_i])
        ),
        transform=ax.transAxes,
        ha='right',
        va='center',
        fontsize=fontsize-2,
    )

for col_i, ax in enumerate(axes[0, 1:]):
    label_text = ax.text(
        0.5, 1.1,
        '{}'.format(col_i+1),
        transform=ax.transAxes,
        ha='center', va='bottom',
        fontsize=fontsize-2,
        color='w',
    )
    label_text.set_path_effects([
        mpl.patheffects.Stroke(linewidth=1.8, foreground='k'),
        mpl.patheffects.Normal()
    ])


# legend
# h_elpd_vert = mpl.lines.Line2D(
#     [], [], color=h_elpd.get_color(), marker='|', linestyle='None',
#     markersize=14, markeredgewidth=1.5
# )
# h_elpdhat_vert = mpl.lines.Line2D(
#     [], [], color=h_elpdhat.get_color(), marker='|', linestyle='None',
#     markersize=14, markeredgewidth=1.5
# )
axes[-1,1].legend(
    [
        # h_elpdhat_vert,
        h_elpd,
        hs_elpdtilde[0],
        h_elpdhat_n,
    ],
    [
        # r'$\widehat{\mathrm{elpd}}_\mathrm{LOO}$',
        r'$\mathrm{elpd}$',
        r'$\widetilde{\mathrm{elpd}}_\mathrm{LOO}$',
        'normal approx.',
    ],
    fontsize=fontsize-2,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.4),
    fancybox=False,
    shadow=False,
    ncol=4,
    handler_map={h_elpd:HandlerMiniatureLine()}
)
