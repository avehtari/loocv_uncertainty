"""Script for plot 1."""

import numpy as np
from scipy import linalg, stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import seaborn as sns

from matplotlib.patches import ConnectionPatch

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

cmap = truncate_colormap(cm.get_cmap('Greys'), 0.4)
cmap.set_under('white', 1.0)

# example points (elpdhat, elpd)
custom_points = [
    [[-54.0, -42.0], [-47.0, -47.0], [-44.0, -44.0]],
    [[-0.5, 1.5], [0.6, 0.6], [0.967, 0.073]],
    [[-11.5, -11.5], [0.0, -10.0], [5.7, -6.7]],
]

custom_limits = [
    [-75, -25],
    [-2.5, 3.0],
    [-30, 20]
]

fig, axes_all = plt.subplots(
    2*n_probls, 3,
    gridspec_kw={'height_ratios': [3, 2]*n_probls},
    figsize=(10, 11)
)

for probl_i in range(n_probls):

    axes = axes_all[probl_i*2:probl_i*2+2,:]

    # blanks
    axes[0, 0].axis('off')
    axes[0, 2].axis('off')

    x_arr = loo_s[probl_i]
    y_arr = elpd_s[probl_i]

    # ===============
    # joint

    ax = axes[0, 1]

    if manual_zoom:
        if probl_i == 0:
            idxs = (
                (-70 < x_arr) & (x_arr < -20) &
                (-50 < y_arr) & (y_arr < -37)
            )
        if probl_i == 1:
            idxs = (
                (-1.5 < x_arr) & (x_arr < 2.5) &
                (-1 < y_arr) & (y_arr < 2.5)
            )
        if probl_i == 2:
            idxs = (
                (-18 < x_arr) & (x_arr < 9) &
                (-15 < y_arr) & (y_arr < -4)
            )
        x_arr = x_arr[idxs]
        y_arr = y_arr[idxs]

    ax.hexbin(x_arr, y_arr, gridsize=40, cmap=cmap, mincnt=1)

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
        color=adjust_lightness('C3', amount=1.3)
    )

    # ===============
    # custom points

    x_y_curs = []

    for ax, (elpdhat_target, elpd_target) in zip(
            axes[1, :], custom_points[probl_i]):

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

        x_y_curs.append((elpdhat, elpd))

        # limits
        # elpdtilde_min, elpdtilde_max = np.percentile(elpdtilde, [2.5, 97.5])
        # x_min = min(elpd, elpdhat-3*sehat, elpdtilde_min)
        # x_max = max(elpd, elpdhat+3*sehat, elpdtilde_max)
        # x_min, x_max = np.array([-1,1])*0.2*(x_max-x_min)+[x_min, x_max]

        x_min, x_max = custom_limits[probl_i]

        x_grid = np.linspace(x_min, x_max, 100)

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
            color='C2'
        )
        # plot elpdhat
        h_elpdhat = ax.axvline(elpdhat, color='C2')
        # plot elpd
        h_elpd = ax.axvline(elpd, color='C0', linestyle=':', linewidth=2.5)

        ax.set_xlim((x_min, x_max))

        ax.set_yticks([])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # share hists row ylim
    max_ylim = max(map(lambda ax: ax.get_ylim()[1], axes_all[probl_i*2+1, :]))
    for ax in axes_all[probl_i*2+1, :]:
        ax.set_ylim(top=max_ylim)

    # connect plots
    for ax, (x_cur, y_cur) in zip(axes[1, :], x_y_curs):

        # con_line = ConnectionPatch(
        #     xyA=(x_cur, ax.get_ylim()[1]), xyB=(x_cur, y_cur),
        #     coordsA="data", coordsB="data",
        #     axesA=ax, axesB=axes[0, 1],
        #     color="C1"
        # )
        # ax.add_artist(con_line)
        # axes[0, 1].plot(x_cur, y_cur, color='C1', marker='o')

        con_line = ConnectionPatch(
            xyB=(0.5, 1.1),
            xyA=(x_cur, y_cur),
            coordsB="axes fraction", coordsA="data",
            axesB=ax, axesA=axes[0, 1],
            color=adjust_lightness('C1', amount=1.3),
            zorder=1,
        )
        axes[0, 1].add_artist(con_line)
        axes[0, 1].plot(
            x_cur, y_cur, color='C1', marker='.', ms=8)

    # legend
    if probl_i == 0:
        # h_elpd_vert = mpl.lines.Line2D(
        #     [], [], color=h_elpd.get_color(), marker='|', linestyle='None',
        #     markersize=14, markeredgewidth=1.5
        # )
        h_elpdhat_vert = mpl.lines.Line2D(
            [], [], color=h_elpdhat.get_color(), marker='|', linestyle='None',
            markersize=14, markeredgewidth=1.5
        )
        axes[0,-1].legend(
            [h_elpdhat_vert, h_elpd, h_elpdhat_n, hs_elpdtilde[0]],
            [r'$\widehat{\mathrm{elpd}}$',
                r'$\mathrm{elpd}$',
                'normal approx.',
                r'$\widetilde{\mathrm{elpd}}$'],
            fontsize=fontsize-2,
            loc='upper right',
            handler_map={h_elpd:HandlerMiniatureLine()}
        )

fig.tight_layout()

for probl_i, probl_name in enumerate([
    'Clear case',
    'Models similar',
    'Outliers'
]):
    ax = axes_all[probl_i*2+1, 0]
    ax.text(
        x=-0.1,
        y=1.5,
        s=probl_name,
        rotation=90,
        fontsize=fontsize,
        va='center',
        ha='right',
        transform=ax.transAxes
    )
