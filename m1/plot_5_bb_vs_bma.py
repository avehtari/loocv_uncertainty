"""Script for plot 1."""

import numpy as np
from scipy import linalg, stats

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap

from m1_problem import *



# ============================================================================

fixed_sigma2_m = False

# [
#   [sigma_d, ...],     [0.01, 1.0, 100.0]
#   [n_obs, ...],       [16, 32, 64, 128, 256, 512, 1024]
#   [beta_t, ...],      [0.0, 0.2, 1.0, 4.0]
#   [prc_out, ...]      [0.0, np.nextafter(0,1), 0.01, 0.08]
# ]

selected_ns = [2, 3, 4]
selected_betas = [0, 1, 2]
selected_prc_outs = [0, 2]
sigma_d_i = 1

# histogram bins
cal_nbins = 7


fontsize = 16


seed_analysis = 123451

# bayes boot samples
bb_n = 1000
bb_a = 1.0


# ============================================================================

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


# custom diverging colormap
cmap_colors = [
    (0.9, 0.7, 0.3),
    (0.85, 0.65, 0.3),
    (0.7, 0.5, 0.3),
    (0.3, 0.3, 0.3),
    (0.8, 0.2, 0.2),
    (0.95, 0.2, 0.2),
    (1, 0.2, 0.2),
]
cmap_div_black = LinearSegmentedColormap.from_list(
        'div_black', cmap_colors, N=256)


# ============================================================================


if fixed_sigma2_m:
    folder_name = 'fixed'
else:
    folder_name = 'unfixed'

rng = np.random.RandomState(seed=seed_analysis)

loo_s = np.zeros((
    len(selected_prc_outs), len(selected_betas), len(selected_ns), n_trial))
naive_var_s = np.zeros((
    len(selected_prc_outs), len(selected_betas), len(selected_ns), n_trial))
cor_loo_i_s = np.zeros((
    len(selected_prc_outs), len(selected_betas), len(selected_ns), n_trial))
skew_loo_i_s = np.zeros((
    len(selected_prc_outs), len(selected_betas), len(selected_ns), n_trial))
bb_plooneg_s = np.zeros((
    len(selected_prc_outs), len(selected_betas), len(selected_ns), n_trial))

target_mean_s = np.zeros((
    len(selected_prc_outs), len(selected_betas), len(selected_ns)))
target_var_s = np.zeros((
    len(selected_prc_outs), len(selected_betas), len(selected_ns)))
# target_Sigma_s = np.zeros((
#     len(selected_prc_outs), len(selected_betas), len(selected_ns), n_obs, n_obs))
target_skew_s = np.zeros((
    len(selected_prc_outs), len(selected_betas), len(selected_ns)))
target_plooneg_s = np.zeros((
    len(selected_prc_outs), len(selected_betas), len(selected_ns)))
elpd_s = np.zeros((
    len(selected_prc_outs), len(selected_betas), len(selected_ns), n_trial))

bma_s = np.zeros((
    len(selected_prc_outs),
    len(selected_betas),
    len(selected_ns),
    n_trial,
    2
))
# bma_pair_s = np.zeros((
#     len(selected_prc_outs),
#     len(selected_betas),
#     len(selected_ns),
#     n_trial,
#     2
# ))
bma_elpd_s = np.zeros((
    len(selected_prc_outs),
    len(selected_betas),
    len(selected_ns),
    n_trial,
    2
))

# load results
for o_i, o_ii in enumerate(selected_prc_outs):
    print('o_i={}/{}'.format(o_i, len(selected_prc_outs)), flush=True)
    prc_out = prc_out_s[o_ii]
    for b_i, b_ii in enumerate(selected_betas):
        print('b_i={}/{}'.format(b_i, len(selected_betas)), flush=True)
        beta_t = beta_t_s[b_ii]
        for n_i, n_ii in enumerate(selected_ns):
            n_obs = n_obs_s[n_ii]
            run_i = np.ravel_multi_index([sigma_d_i, n_ii, b_ii, o_ii], grid_shape)
            res_file = np.load(
                'res_1/{}/{}.npz'
                .format(folder_name, str(run_i).zfill(4))
            )
            # fetch results
            loo_ti_A = res_file['loo_ti_A']
            loo_ti_B = res_file['loo_ti_B']
            test_elpd_t_A = res_file['test_elpd_t_A']
            test_elpd_t_B = res_file['test_elpd_t_B']
            # close file
            res_file.close()

            # calc some normally obtainable values
            n_cur = loo_ti_A.shape[-1]
            loo_i = loo_ti_A-loo_ti_B
            loo_s[o_i, b_i, n_i] = np.sum(loo_i, axis=-1)
            naive_var_s[o_i, b_i, n_i] = n_cur*np.var(
                loo_i, ddof=1, axis=-1)
            for trial_i in range(n_trial):
                cor_loo_i_s[o_i, b_i, n_i, trial_i] = np.corrcoef(
                    loo_ti_A[trial_i], loo_ti_B[trial_i])[0, 1]
            skew_loo_i_s[o_i, b_i, n_i] = stats.skew(
                loo_i, axis=-1, bias=False)
            bb_plooneg_s[o_i, b_i, n_i] = bb_plooneg(loo_i)

            # calc some target values
            target_mean_s[o_i, b_i, n_i] = np.mean(loo_s[o_i, b_i, n_i])
            target_var_s[o_i, b_i, n_i] = np.var(loo_s[o_i, b_i, n_i], ddof=1)
            # target_Sigma_s[o_i, b_i, n_i] = np.cov(loo_i, ddof=1, rowvar=False)
            target_skew_s[o_i, b_i, n_i] = stats.skew(loo_s[o_i, b_i, n_i], bias=False)
            # TODO calc se of this ... formulas online
            target_plooneg_s[o_i, b_i, n_i] = np.mean(loo_s[o_i, b_i, n_i]<0)
            elpd_s[o_i, b_i, n_i] = test_elpd_t_A - test_elpd_t_B

            # bma
            loo_tki = np.stack((loo_ti_A, loo_ti_B), axis=1)
            bma_s[o_i, b_i, n_i] = pseudo_bma_p(loo_tki)
            # bma_pair_s[o_i, b_i, n_i] = pseudo_bma_p_pair(loo_i)
            elpd_tk = np.stack((test_elpd_t_A, test_elpd_t_B), axis=1)
            bma_elpd_s[o_i, b_i, n_i] = pseudo_bma(elpd_tk)

# naive var ratio
naive_se_ratio_s_mean = np.sqrt(
    np.mean(naive_var_s, axis=-1)/target_var_s)
naive_se_ratio_s = np.sqrt(
    np.einsum('bt,ijkt->ijkb',
        rng.dirichlet(
            np.full(n_trial, bb_a),
            size=bb_n
        ),
        naive_var_s
    )/target_var_s[:,:,:,None]
)

# naive var error ratio
error_var_s = np.var(loo_s-elpd_s, ddof=1, axis=-1)
naive_error_ratio_s_mean = np.sqrt(
    np.mean(naive_var_s, axis=-1)/error_var_s)
naive_error_ratio_s = np.sqrt(
    np.einsum('bt,ijkt->ijkb',
        rng.dirichlet(
            np.full(n_trial, bb_a),
            size=bb_n
        ),
        naive_var_s
    )/error_var_s[:,:,:,None]
)


# naive_coef_var_s = np.sqrt(naive_var_s)/loo_s
naive_plooneg_s = stats.norm.cdf(0, loc=loo_s, scale=np.sqrt(naive_var_s))

pelpdneg_s = np.mean(elpd_s<0, axis=-1)
# target_coefvar_s = np.sqrt(target_var_s)/target_mean_s

# misspred: TODO replace with something
naive_misspred_s = np.abs(naive_plooneg_s-pelpdneg_s[:,:,:,None])



# ============================================================================
# plot select grid subset
# ============================================================================

def select_grid_subset(data_x, data_y, nx, ny):
    minx = np.min(data_x)
    miny = np.min(data_y)
    maxx = np.max(data_x)
    maxy = np.max(data_y)
    gridx = np.linspace(minx, maxx, nx)
    gridy = np.linspace(miny, maxy, ny)
    meshx, meshy = np.meshgrid(gridx, gridy, indexing='ij')
    pointx = meshx.ravel()
    pointy = meshy.ravel()
    dist2 = (data_x[:,None]-pointx)**2 + (data_y[:,None]-pointy)**2
    closest = np.argmin(dist2, axis=0)
    idxs = np.unique(closest)
    return idxs



# other
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
# plots
# ============================================================================


cmap = truncate_colormap(cm.get_cmap('Greys'), 0.4)
cmap.set_under('white', 1.0)

cmap_man_fix = truncate_colormap(cm.get_cmap('Greys'), 0.99)
cmap_man_fix.set_under('white', 1.0)

binlims = np.arange(0, 1.01, 0.05)


for o_i, o_ii in enumerate(selected_prc_outs):
    prc_out = prc_out_s[o_ii]

    fig, axes = plt.subplots(
        len(selected_ns), len(selected_betas), figsize=(7,9),
        sharex=True,
        sharey=True
    )

    for n_i, n_ii in enumerate(selected_ns):
        n_obs = n_obs_s[n_ii]

        for b_i, b_ii in enumerate(selected_betas):
            beta_t = beta_t_s[b_ii]

            ax = axes[n_i, b_i]

            ax.plot([0,1], [0,1], color='red', alpha=0.7)

            data_x = 1-bb_plooneg_s[o_i, b_i, n_i]
            data_y = bma_s[o_i, b_i, n_i, :, 0]

            # manual fix for only 1 bin -> too light
            t, _, _ = np.histogram2d(data_x, data_y, bins=[binlims, binlims])
            if np.count_nonzero(t) == 1:
                ax.hist2d(
                    data_x, data_y, bins=binlims, cmap=cmap_man_fix, cmin=1)
            else:
                ax.hist2d(data_x, data_y, bins=binlims, cmap=cmap, cmin=1)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.set_ylim(0, 1)
            ax.set_xlim(0, 1)

            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.tick_params(axis='both', which='minor', labelsize=12)

    for ax in axes[-1,:]:
        ax.set_xlabel(
            r'$\mathrm{Pr}(\mathrm{\widehat{elpd}_d}>0)$', fontsize=16)

    for ax in axes[:,0]:
        ax.set_ylabel('pBMA+',fontsize=16)

    for b_i, b_ii in enumerate(selected_betas):
        beta_t = beta_t_s[b_ii]
        ax = axes[0, b_i]
        ax.set_title(r'$\beta_t={}$'.format(beta_t),fontsize=16)

    fig.tight_layout()
    fig.subplots_adjust(left=0.2)

    for n_i, n_ii in enumerate(selected_ns):
        n_obs = n_obs_s[n_ii]
        ax = axes[n_i, 0]
        ax.text(
            -0.78,
            0.5,
            r'$n={}$'.format(n_obs),
            transform=ax.transAxes,
            rotation=90,
            fontsize=16,
            va='center'
        )
