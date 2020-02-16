"""Analyse LOOCV results

First run results with m1_run.py

"""

import numpy as np
from scipy import linalg, stats

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap


from m3_problem import *


# ============================================================================
# config

# seed for randomisation
# np.random.RandomState().randint(np.iinfo(np.uint32).max)
seed_analysis = 4257151306

# fixed or unfixed sigma
fixed_sigma2_m = False

# bayes boot samples
bb_n = 100
bb_a = 1.0

# ============================================================================

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def remove_frame(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


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

n_obs = 128

loo_s = np.zeros((grid_shape[2], grid_shape[3], n_trial))
naive_var_s = np.zeros((grid_shape[2], grid_shape[3], n_trial))
cor_loo_i_s = np.zeros((grid_shape[2], grid_shape[3], n_trial))
skew_loo_i_s = np.zeros((grid_shape[2], grid_shape[3], n_trial))

target_mean_s = np.zeros((grid_shape[2], grid_shape[3]))
target_var_s = np.zeros((grid_shape[2], grid_shape[3]))
# target_Sigma_s = np.zeros((grid_shape[2], grid_shape[3], n_obs, n_obs))
target_skew_s = np.zeros((grid_shape[2], grid_shape[3]))
target_plooneg_s = np.zeros((grid_shape[2], grid_shape[3]))
elpd_s = np.zeros((grid_shape[2], grid_shape[3], n_trial))

vlpd_t = np.empty((grid_shape[2], grid_shape[3], n_trial))
q025lpd_t = np.empty((grid_shape[2], grid_shape[3], n_trial))
q500lpd_t = np.empty((grid_shape[2], grid_shape[3], n_trial))
q975lpd_t = np.empty((grid_shape[2], grid_shape[3], n_trial))
plpdneg_t = np.empty((grid_shape[2], grid_shape[3], n_trial))

bma_s = np.zeros((grid_shape[2], grid_shape[3], n_trial, 2))
bma_elpd_s = np.zeros((grid_shape[2], grid_shape[3], n_trial, 2))


# load results
for b_i, beta_t in enumerate(beta_t_s):

    print('{}/{}'.format(b_i, len(beta_t_s)), flush=True)

    for o_i, prc_out in enumerate(prc_out_s):

        run_i = np.ravel_multi_index([0, 0, b_i, o_i], grid_shape)
        res_file = np.load(
            'res_3/{}/{}.npz'
            .format(folder_name, str(run_i).zfill(4))
        )
        # fetch results
        loo_ti_A = res_file['loo_ti_A']
        loo_ti_B = res_file['loo_ti_B']
        test_elpd_t_A = res_file['test_elpd_t_A']
        test_elpd_t_B = res_file['test_elpd_t_B']

        vlpd_t[b_i, o_i] = res_file['test_vlpd_t']
        q025lpd_t[b_i, o_i] = res_file['test_q025lpd_t']
        q500lpd_t[b_i, o_i] = res_file['test_q500lpd_t']
        q975lpd_t[b_i, o_i] = res_file['test_q975lpd_t']
        plpdneg_t[b_i, o_i] = res_file['test_plpdneg_t']

        # close file
        res_file.close()

        # calc some normally obtainable values
        n_cur = loo_ti_A.shape[-1]
        loo_i = loo_ti_A-loo_ti_B
        loo_s[b_i, o_i] = np.sum(loo_i, axis=-1)
        naive_var_s[b_i, o_i] = n_cur*np.var(
            loo_i, ddof=1, axis=-1)
        for trial_i in range(n_trial):
            cor_loo_i_s[b_i, o_i, trial_i] = np.corrcoef(
                loo_ti_A[trial_i], loo_ti_B[trial_i])[0, 1]
        skew_loo_i_s[b_i, o_i] = stats.skew(
            loo_i, axis=-1, bias=False)

        # calc some target values
        target_mean_s[b_i, o_i] = np.mean(loo_s[b_i, o_i])
        target_var_s[b_i, o_i] = np.var(loo_s[b_i, o_i], ddof=1)
        # target_Sigma_s[b_i, o_i] = np.cov(loo_i, ddof=1, rowvar=False)
        target_skew_s[b_i, o_i] = stats.skew(loo_s[b_i, o_i], bias=False)
        # TODO calc se of this ... formulas online
        target_plooneg_s[b_i, o_i] = np.mean(loo_s[b_i, o_i]<0)
        elpd_s[b_i, o_i] = test_elpd_t_A - test_elpd_t_B

        # pseudo-bma+
        loo_tki = np.stack((loo_ti_A, loo_ti_B), axis=1)
        bma_s[b_i, o_i] = pseudo_bma_p(loo_tki)

        # pseudo-bma for true elpd
        elpd_tk = np.stack((test_elpd_t_A, test_elpd_t_B), axis=1)
        bma_elpd_s[b_i, o_i] = pseudo_bma(elpd_tk)

# save
if False:
    np.savez_compressed(
        'plot_4_save.npz',
        loo_s=loo_s,
        naive_var_s=naive_var_s,
        cor_loo_i_s=cor_loo_i_s,
        skew_loo_i_s=skew_loo_i_s,
        target_mean_s=target_mean_s,
        target_var_s=target_var_s,
        target_skew_s=target_skew_s,
        target_plooneg_s=target_plooneg_s,
        elpd_s=elpd_s,
        vlpd_t=vlpd_t,
        q025lpd_t=q025lpd_t,
        q500lpd_t=q500lpd_t,
        q975lpd_t=q975lpd_t,
        plpdneg_t=plpdneg_t,
        bma_s=bma_s,
        bma_elpd_s=bma_elpd_s,
    )

# load
if False:
    save_file = np.load('plot_4_save.npz')
    loo_s = save_file['loo_s']
    naive_var_s = save_file['naive_var_s']
    cor_loo_i_s = save_file['cor_loo_i_s']
    skew_loo_i_s = save_file['skew_loo_i_s']
    target_mean_s = save_file['target_mean_s']
    target_var_s = save_file['target_var_s']
    target_skew_s = save_file['target_skew_s']
    target_plooneg_s = save_file['target_plooneg_s']
    elpd_s = save_file['elpd_s']
    vlpd_t = save_file['vlpd_t']
    q025lpd_t = save_file['q025lpd_t']
    q500lpd_t = save_file['q500lpd_t']
    q975lpd_t = save_file['q975lpd_t']
    plpdneg_t = save_file['plpdneg_t']
    bma_s = save_file['bma_s']
    bma_elpd_s = save_file['bma_elpd_s']
    save_file.close()

# naive var ratio
naive_se_ratio_s_mean = np.sqrt(
    np.mean(naive_var_s, axis=-1)/target_var_s)
naive_se_ratio_s = np.sqrt(
    np.einsum('bt,ijt->ijb',
        rng.dirichlet(
            np.full(n_trial, bb_a),
            size=bb_n
        ),
        naive_var_s
    )/target_var_s[:,:,None]
)

# naive var error ratio
error_var_s = np.var(loo_s-elpd_s, ddof=1, axis=-1)
naive_error_ratio_s_mean = np.sqrt(
    np.mean(naive_var_s, axis=-1)/error_var_s)
naive_error_ratio_s = np.sqrt(
    np.einsum('bt,ijt->ijb',
        rng.dirichlet(
            np.full(n_trial, bb_a),
            size=bb_n
        ),
        naive_var_s
    )/error_var_s[:,:,None]
)


# naive_coef_var_s = np.sqrt(naive_var_s)/loo_s
naive_plooneg_s = stats.norm.cdf(0, loc=loo_s, scale=np.sqrt(naive_var_s))

pelpdneg_s = np.mean(elpd_s<0, axis=-1)
# target_coefvar_s = np.sqrt(target_var_s)/target_mean_s

# misspred: TODO replace with something
naive_misspred_s = np.abs(naive_plooneg_s-pelpdneg_s[:,:,None])
naive_misspred2_s = np.abs(naive_plooneg_s-plpdneg_t)


# ===========================================================================
# plot 1

cmap = truncate_colormap(cm.get_cmap('Greys'), 0.4)
cmap.set_under('white', 1.0)

data_x = loo_s[0:5,:].ravel()
# data_y = plpdneg_t[0,0].ravel()
data_y = np.tile(pelpdneg_s[0:5,:], (1,1, n_trial)).ravel()

# idxs = (-3 < data_x) & (data_x < 3)
# data_x = data_x[idxs]
# data_y = data_y[idxs]

# plt.plot(data_x, data_y, '.')
# plt.ylim((0,1))

grid = sns.jointplot(
    data_x, data_y,
    kind='hex',
    height=5,
    cmap=cmap,
    # joint_kws=dict(cmin=1),
)
grid.set_axis_labels(
    '$\widehat{\mathrm{elpd}}_\mathrm{D}$',
    '$\widehat{\mathrm{elpd}}_\mathrm{D}/\widehat{\mathrm{SE}}$'
)
plt.clim(vmin=1)
plt.tight_layout()
plt.xlim((-3, 3))




# ======================================

data_x = beta_t_s[:7]
data_y_s = [
    loo_s[:7],
    loo_s[:7]/np.sqrt(naive_var_s[:7]),
    pelpdneg_s[:7],
    plpdneg_t[:7],
    bma_s[:7,...,0],
    bma_elpd_s[:7,...,0]
]
data_y_name_s = [
    r'$\widehat{\mathrm{elpd}}_\mathrm{D}$',
    r'$\widehat{\mathrm{elpd}}_\mathrm{D}/\widehat{\mathrm{SE}}$',
    r'$p(\mathrm{elpd}<0)$',
    r'$p(\mathrm{lpd}<0)$',
    'pseudo-bma+',
    'pseudo-bma-true',
]

for data_y, data_y_name in zip(data_y_s, data_y_name_s):

    selected_prc_outs = [0, 1, 2]
    fig, axes = plt.subplots(len(selected_prc_outs), 1,
        sharex=False, sharey=False)
    for ax_i, ax in enumerate(axes):
        o_i = selected_prc_outs[ax_i]

        if data_y.ndim == 3:
            median = np.percentile(data_y[:,o_i], 50, axis=-1)
            q025 = np.percentile(data_y[:,o_i], 2.5, axis=-1)
            q975 = np.percentile(data_y[:,o_i], 97.5, axis=-1)
            ax.fill_between(data_x, q025, q975, color='C0', alpha=0.2)
            ax.plot(data_x, median, 'C0')
        else:
            ax.plot(data_x, data_y[:,o_i], 'C0')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel(
            '{} % outliers'.format(int(prc_out_s[o_i]*100))
            + '\n'
            + data_y_name
        )
    axes[-1].set_xlabel(r'$\beta_t$')
    fig.tight_layout()


# ======================================

data_x = loo_s
data_x_name = 'loo_s'

data_y_s = [
    # loo_s/np.sqrt(naive_var_s),
    # np.tile(pelpdneg_s[:,:,None], (1, 1, n_trial)),
    1-plpdneg_t,
    bma_s[...,0],
    bma_elpd_s[...,0],
]
data_y_name_s = [
    # r'$\widehat{\mathrm{elpd}}_\mathrm{D}/\widehat{\mathrm{SE}}$',
    # r'$p(\mathrm{elpd}<0)$',
    r'$p(\mathrm{lpd}>0)$',
    'pseudo-bma+',
    'pseudo-bma-true',
]

selected_prc_outs = [0, 1, 2]
selected_beta_t_s = [0, 1, 2, 5, 10]

for data_y, data_y_name in zip(data_y_s, data_y_name_s):

    fig, axes = plt.subplots(
        len(selected_prc_outs), len(selected_beta_t_s),
        sharex=False, sharey=True)
    for o_ax_i, row in enumerate(axes):
        for b_ax_i, ax in enumerate(row):
            o_i = selected_prc_outs[o_ax_i]
            b_i = selected_beta_t_s[b_ax_i]

            ax.plot(data_x[b_i, o_i], data_y[b_i, o_i], '.', alpha=0.1)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    for ax in axes[-1,:]:
        ax.set_xlabel(data_x_name)
    for ax in axes[:,0]:
        ax.set_ylabel(data_y_name)
    fig.tight_layout()


# ======================================
# corr vs bma-error

cmap = truncate_colormap(cm.get_cmap('Greys'), 0.4)
cmap.set_under('white', 1.0)


data_x = cor_loo_i_s
data_x_name = 'corr'

data_y_s = [
    # loo_s/np.sqrt(naive_var_s),
    # np.tile(pelpdneg_s[:,:,None], (1, 1, n_trial)),
    # 1-plpdneg_t,
    bma_s[...,0],
]
data_y_name_s = [
    # r'$\widehat{\mathrm{elpd}}_\mathrm{D}/\widehat{\mathrm{SE}}$',
    # r'$p(\mathrm{elpd}<0)$',
    # r'$p(\mathrm{lpd}>0)$',
    'pseudo-bma+',
]

data_z = bma_elpd_s[...,0]-bma_s[...,0]
data_z_name = 'pseudo-bma+ error'

selected_prc_outs = [0, 1]
selected_beta_t_s = [0, 4, 6, 10]

for data_y, data_y_name in zip(data_y_s, data_y_name_s):

    fig, axes = plt.subplots(
        len(selected_prc_outs), len(selected_beta_t_s),
        sharex=True, sharey=True,
        figsize=(8,6))
    for o_ax_i, row in enumerate(axes):
        for b_ax_i, ax in enumerate(row):
            o_i = selected_prc_outs[o_ax_i]
            b_i = selected_beta_t_s[b_ax_i]

            data_x_i = data_x[b_i, o_i]
            data_y_i = data_y[b_i, o_i]
            data_z_i = data_z[b_i, o_i]

            # select a sparce representation
            idxs = select_grid_subset(data_x_i, data_y_i, 25, 25)

            scat = ax.scatter(
                data_x_i[idxs], data_y_i[idxs], s=12, c=data_z_i[idxs],
                cmap=cmap_div_black, vmin=-1, vmax=1,
            )

            # ax.plot(data_x[b_i, o_i], data_y[b_i, o_i], '.', alpha=0.1)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.tick_params(axis='both', which='minor', labelsize=14)

    for ax in axes[-1,:]:
        ax.set_xlabel(data_x_name, fontsize=16)
    axes[0,0].set_ylabel('no outliers\n' + data_y_name, fontsize=16)
    axes[1,0].set_ylabel('1 % outliers\n' + data_y_name, fontsize=16)

    for b_ax_i, ax in enumerate(axes[0,:]):
        b_i = selected_beta_t_s[b_ax_i]
        beta_t = beta_t_s[b_i]
        ax.set_title(r'$\beta_t = {:.1f}$'.format(beta_t), fontsize=16)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    cbar_ax = fig.add_axes([0.15, 0.1, 0.8, 0.03])
    cbar = fig.colorbar(scat, orientation="horizontal", cax=cbar_ax)
    cbar.set_label(data_z_name, fontsize=16)
    cbar.ax.tick_params(labelsize=16)
