"""Script for plot 1."""

import numpy as np
from scipy import linalg, stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import seaborn as sns

from matplotlib.patches import ConnectionPatch
from matplotlib import patheffects
import matplotlib.gridspec as gridspec

from m1_problem import *





from matplotlib.legend_handler import HandlerLine2D
import matplotlib.path as mpath
from matplotlib.transforms import BboxTransformFrom, BboxTransformTo, Bbox


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

# # calibration counts
# cal_limits = np.linspace(0, 1, cal_nbins+1)
# cal_counts = np.zeros((n_probls, cal_nbins), dtype=int)
# for probl_i in range(n_probls):
#     diff_pdf = stats.norm.cdf(
#         elpd_s[probl_i],
#         loc=loo_s[probl_i],
#         scale=np.sqrt(naive_var_s[probl_i])
#     )
#     cal_counts[probl_i] = np.histogram(diff_pdf, cal_limits)[0]


# ============================================================================
# plot
# ============================================================================

# ============================================================================
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

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

fontsize = 16

cmap = truncate_colormap(cm.get_cmap('Greys'), 0.3)
cmap.set_under('white', 1.0)

# example points (elpdhat, elpd)
custom_points = [
    [[-54.0, -42.0], [-44.0, -44.0], [-47.0, -47.0]],
    [[-0.5, 1.5], [0.6, 0.6], [0.967, 0.073]],
    [[5.7, -6.7], [0.0, -10.0], [-11.0, -11.0]],
]

# custom_limits = [
#     [-75, -25],
#     [-2.5, 3.0],
#     [-30, 20]
# ]

fig = plt.figure(figsize=(10, 8))
outer = gridspec.GridSpec(
    2, 1, height_ratios = [1, 1],
    hspace=0.35,
    top=0.95,
    bottom=0.11,
    left=0.08,
    right=0.97
)
gs1 = gridspec.GridSpecFromSubplotSpec(
    1, n_probls, subplot_spec=outer[0,:], wspace=0.5)
gs2 = gridspec.GridSpecFromSubplotSpec(
    3, n_probls, subplot_spec=outer[1:,:], wspace=0.5, hspace=0.2)
axes = np.array(
    [[fig.add_subplot(gs1[probl_i]) for probl_i in range(n_probls)]]
    +
    [
        [fig.add_subplot(gs2[probl_i, row_i])for row_i in range(3)]
        for probl_i in range(n_probls)
    ]
)

# fig, axes = plt.subplots(
#     4, n_probls,
#     gridspec_kw={'height_ratios': [3, 1, 1, 1]},
#     figsize=(9, 8)
# )

for probl_i in range(n_probls):

    x_arr = loo_s[probl_i]
    y_arr = elpd_s[probl_i]

    # ===============
    # joint

    ax = axes[0, probl_i]

    if manual_zoom:
        if probl_i == 0:
            idxs = (
                (-75 < x_arr) & (x_arr < -25) &
                (-50 < y_arr) & (y_arr < -37)
            )
        if probl_i == 1:
            idxs = (
                (-2.5 < x_arr) & (x_arr < 3.0) &
                (-1 < y_arr) & (y_arr < 2.5)
            )
        if probl_i == 2:
            idxs = (
                (-26 < x_arr) & (x_arr < 15) &
                (-15 < y_arr) & (y_arr < -4)
            )
        x_arr = x_arr[idxs]
        y_arr = y_arr[idxs]

    ax.hexbin(x_arr, y_arr, gridsize=35, cmap=cmap, mincnt=1)

    if probl_i == 0:
        ax.set_ylabel(r'$\mathrm{elpd}$', fontsize=fontsize-2)
    ax.set_xlabel(r'$\widehat{\mathrm{elpd}}$', fontsize=fontsize-2)

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

    if probl_i == 2:
        ax.set_xlim(left=-26)

    # ===============
    # custom points

    x_min, x_max = axes[0, probl_i].get_xlim()
    x_grid = np.linspace(x_min, x_max, 100)

    for p_i, (ax, (elpdhat_target, elpd_target)) in enumerate(zip(
            axes[1:, probl_i], custom_points[probl_i])):

        # find closest taxi distance
        idx = (
            np.abs(loo_s[probl_i] - elpdhat_target)
            + np.abs(elpd_s[probl_i] - elpd_target)
        ).argmin()

        elpd = elpd_s[probl_i, idx]
        elpdhat = loo_s[probl_i, idx]
        elpdhat_i = res_A[probl_i][idx]-res_B[probl_i][idx]
        sehat = np.sqrt(naive_var_s[probl_i, idx])
        n_obs = len(elpdhat_i)
        elpdtilde = elpdhat - (loo_s[probl_i] - elpd_s[probl_i])

        # limits
        # elpdtilde_min, elpdtilde_max = np.percentile(elpdtilde, [2.5, 97.5])
        # x_min = min(elpd, elpdhat-3*sehat, elpdtilde_min)
        # x_max = max(elpd, elpdhat+3*sehat, elpdtilde_max)
        # x_min, x_max = np.array([-1,1])*0.2*(x_max-x_min)+[x_min, x_max]

        # x_min, x_max = custom_limits[probl_i]

        # x_grid = np.linspace(x_min, x_max, 100)

        # elpdtilde sliced
        # elpdtilde = elpdtilde[(x_min<elpdtilde) & (elpdtilde<x_max)]

        # plot elpdtilde hist
        _, _, hs_elpdtilde = ax.hist(
            elpdtilde,
            density=True,
            color=adjust_lightness('C0', amount=2.0),
            # alpha=0.4,
            bins=int(round((elpdtilde.max()-elpdtilde.min())/(x_max-x_min)*20)),
        )
        # plot normal approx
        dens = stats.norm.pdf(x_grid, loc=elpdhat, scale=sehat)
        #dens *= ax.get_ylim()[1]*0.95/dens.max()
        h_elpdhat_n, = ax.plot(
            x_grid,
            dens,
            color='C1'
        )
        # plot elpdhat
        h_elpdhat = ax.axvline(elpdhat, color='C1')
        # plot elpd
        h_elpd = ax.axvline(elpd, color='C0', linestyle=':', linewidth=2.5)

        ax.set_xlim((x_min, x_max))
        ax.set_xticklabels([])

        ax.set_yticks([])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # add annotation
        axes[0, probl_i].plot(
            [elpdhat], [elpd],
            'o', markersize=5,
            markerfacecolor=adjust_lightness('C1', amount=1.3),
            markeredgewidth=1.0, markeredgecolor='k'
        )
        annotation_text = axes[0, probl_i].annotate(
            '{}'.format(p_i),
            xy=(elpdhat, elpd),
            xytext=(-5, 0),
            ha='right',
            va='center',
            textcoords='offset points',
            color=adjust_lightness('C1', amount=1.3),
            fontsize=fontsize,
        )
        annotation_text.set_path_effects([
            mpl.patheffects.Stroke(linewidth=1.5, foreground='k'),
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
        '{}'.format(row_i),
        labelpad=25,
        color=adjust_lightness('C1', amount=1.3),
        ha='right', va='center',
        fontsize=fontsize, rotation=0)
    label_text.set_path_effects([
        mpl.patheffects.Stroke(linewidth=1.0, foreground='k'),
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
    bbox_to_anchor=(0.5, -0.3),
    fancybox=False,
    shadow=False,
    ncol=4,
    handler_map={h_elpd:HandlerMiniatureLine()}
)
