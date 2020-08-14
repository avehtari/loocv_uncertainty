
import sys, os, time
from functools import partial

import numpy as np
from scipy import linalg, stats

from problem_setting import *



# ============================================================================
# conf

folder_name = 'res_zscore_skew_n_b'
run_moments = True
distributed = False
plot = True

# data seed
# np.random.RandomState().randint(np.iinfo(np.uint32).max)
data_seed = 247169102

# number of trials
n_trial = 2000
# fixed model tau2 value
tau2 = 1.0
# dimensionality of X
n_dim = 3
# first dim intercept
intercept = True
# intercept coef
beta_intercept = 0.0
# other betas
beta_other = 1.0
# model A covariates
idx_a = [0,1]
# model B covariates
idx_b = [0,1,2]

# constant data variance
sigma_d_2 = 1.0

# grid parameters
n_obs_s = np.round(np.linspace(10, 500, 21)).astype(int)
# last beta effect missing in model A
beta_t_s = np.array([0.0, 0.1, 0.2, 0.5, 1.0])

# ============================================================================

if plot:
    import matplotlib.pyplot as plt
    # import seaborn as sns

# ============================================================================
# Funcs

class DataGeneration:

    def __init__(self, n_trial, n_obs_max, data_seed=None):
        self.rng = np.random.RandomState(seed=data_seed)
        if intercept:
            # firs dim (column) ones for intercept
            self.X_mat_all = np.concatenate(
                (
                    np.ones((n_trial, n_obs_max, 1)),
                    self.rng.randn(n_trial, n_obs_max, n_dim-1)
                ),
                axis=-1,
            )
        else:
            self.X_mat_all = self.rng.randn(n_trial, n_obs_max, n_dim)

    def get_x(self, trial_i, n_obs):
        return self.X_mat_all[trial_i, :n_obs, :]

# ============================================================================

# construct beta vectors
beta_s = np.zeros((len(beta_t_s), n_dim))
for b_i, beta_t in enumerate(beta_t_s):
    beta = np.array([beta_other]*(n_dim-1)+[beta_t])
    if intercept:
        beta[0] = beta_intercept
    beta_s[b_i] = beta

data_generation = DataGeneration(n_trial, np.max(n_obs_s), data_seed)


# ============================================================================
# setup the grid

# set grid
n_obs_grid, beta_t_grid = np.meshgrid(n_obs_s, beta_t_s, indexing='ij')
grid_shape = n_obs_grid.shape
n_runs = n_obs_grid.size

def run_i_to_params(run_i):
    n_obs = n_obs_grid.flat[run_i]
    beta_t = beta_t_grid.flat[run_i]
    return n_obs, beta_t

def params_to_run_i(n_obs, beta_t):
    n_obs_i = np.nonzero(n_obs_s == n_obs)[0][0]
    beta_t_i = np.nonzero(beta_t_s == beta_t)[0][0]
    run_i = np.ravel_multi_index((n_obs_i, beta_t_i), grid_shape)
    return run_i


# ============================================================================
# As a function of n

def run_and_save_run_i(run_i):
    run_i_str = str(run_i).zfill(4)

    n_obs, beta_t = run_i_to_params(run_i)
    beta = np.array([beta_other]*(n_dim-1)+[beta_t])
    if intercept:
        beta[0] = beta_intercept

    mean_loo_t = np.full((n_trial,), np.nan)
    var_loo_t = np.full((n_trial,), np.nan)
    moment3_loo_t = np.full((n_trial,), np.nan)
    mean_elpd_t = np.full((n_trial,), np.nan)
    var_elpd_t = np.full((n_trial,), np.nan)
    moment3_elpd_t = np.full((n_trial,), np.nan)
    mean_err_t = np.full((n_trial,), np.nan)
    var_err_t = np.full((n_trial,), np.nan)
    moment3_err_t = np.full((n_trial,), np.nan)

    for t_i in range(n_trial):
        X_mat = data_generation.get_x(t_i, n_obs)
        (
            mean_loo, var_loo, moment3_loo,
            mean_elpd, var_elpd, moment3_elpd,
            mean_err, var_err, moment3_err
        ) = (
            get_analytic_res(
                X_mat, beta, tau2, idx_a, idx_b,
                Sigma_d=sigma_d_2, mu_d=None
            )
        )
        mean_loo_t[t_i] = mean_loo
        var_loo_t[t_i] = var_loo
        moment3_loo_t[t_i] = moment3_loo
        mean_elpd_t[t_i] = mean_elpd
        var_elpd_t[t_i] = var_elpd
        moment3_elpd_t[t_i] = moment3_elpd
        mean_err_t[t_i] = mean_err
        var_err_t[t_i] = var_err
        moment3_err_t[t_i] = moment3_err

    np.savez_compressed(
        folder_name+'/'+'res_'+run_i_str+'.npz',
        mean_loo_t=mean_loo_t,
        var_loo_t=var_loo_t,
        moment3_loo_t=moment3_loo_t,
        mean_elpd_t=mean_elpd_t,
        var_elpd_t=var_elpd_t,
        moment3_elpd_t=moment3_elpd_t,
        mean_err_t=mean_err_t,
        var_err_t=var_err_t,
        moment3_err_t=moment3_err_t,
    )

if run_moments:
    # os.makedirs(folder_name, exist_ok=True)  # make beforehand
    if distributed:
        # parse cmd input for run id
        if len(sys.argv) > 1:
            # get run_i
            run_i = int(sys.argv[1])
        else:
            raise ValueError('Provide run_i as cmd line arg.')
        if run_i < 0 or run_i >= n_runs:
            raise ValueError('invalid run_i, max is {}'.format(n_runs-1))
        n_obs, beta_t = run_i_to_params(run_i)
        print('n_obs:{}, beta_t:{}'.format(n_obs, beta_t))
        start_time = time.time()
        run_and_save_run_i(run_i)
        cur_time_min = (time.time() - start_time)/60
        print('elapsed time: {:.2} min'.format(cur_time_min), flush=True)
    else:
        start_time = time.time()
        for run_i in range(n_runs):
            # progress
            cur_time_min = (time.time() - start_time)/60
            print('{}/{}, elapsed time: {:.2} min'.format(
                run_i+1, n_runs, cur_time_min), flush=True)
            run_and_save_run_i(run_i)
    print('done', flush=True)

# ============================================================================
if not plot:
    # all done if not plotting anything
    raise SystemExit

# ============================================================================
# Load results

mean_loo_s = np.full((len(n_obs_s), len(beta_t_s), n_trial,), np.nan)
var_loo_s = np.full((len(n_obs_s), len(beta_t_s), n_trial,), np.nan)
moment3_loo_s = np.full((len(n_obs_s), len(beta_t_s), n_trial,), np.nan)
mean_elpd_s = np.full((len(n_obs_s), len(beta_t_s), n_trial,), np.nan)
var_elpd_s = np.full((len(n_obs_s), len(beta_t_s), n_trial,), np.nan)
moment3_elpd_s = np.full((len(n_obs_s), len(beta_t_s), n_trial,), np.nan)
mean_err_s = np.full((len(n_obs_s), len(beta_t_s), n_trial,), np.nan)
var_err_s = np.full((len(n_obs_s), len(beta_t_s), n_trial,), np.nan)
moment3_err_s = np.full((len(n_obs_s), len(beta_t_s), n_trial,), np.nan)

for run_i in range(n_runs):
    run_i_str = str(run_i).zfill(4)
    n_obs, beta_t = run_i_to_params(run_i)
    n_obs_i = np.nonzero(n_obs_s == n_obs)[0][0]
    beta_t_i = np.nonzero(beta_t_s == beta_t)[0][0]
    res_file = np.load(folder_name+'/'+'res_'+run_i_str+'.npz')
    mean_loo_s[n_obs_i, beta_t_i, :] = res_file['mean_loo_t']
    var_loo_s[n_obs_i, beta_t_i, :] = res_file['var_loo_t']
    moment3_loo_s[n_obs_i, beta_t_i, :] = res_file['moment3_loo_t']
    mean_elpd_s[n_obs_i, beta_t_i, :] = res_file['mean_elpd_t']
    var_elpd_s[n_obs_i, beta_t_i, :] = res_file['var_elpd_t']
    moment3_elpd_s[n_obs_i, beta_t_i, :] = res_file['moment3_elpd_t']
    mean_err_s[n_obs_i, beta_t_i, :] = res_file['mean_err_t']
    var_err_s[n_obs_i, beta_t_i, :] = res_file['var_err_t']
    moment3_err_s[n_obs_i, beta_t_i, :] = res_file['moment3_err_t']
    res_file.close()

# ============================================================================

# cal skews
skew_loo_s = moment3_loo_s/np.sqrt(var_loo_s)**3
skew_elpd_s = moment3_elpd_s/np.sqrt(var_elpd_s)**3
skew_err_s = moment3_err_s/np.sqrt(var_err_s)**3


# calc total mean, variance and skew
# loo
mean_tot_loo_s, var_tot_loo_s, moment3_tot_loo_s = (
    calc_tot_mean_var_moment3_form_given_x(
        mean_loo_s, var_loo_s, moment3_loo_s)
)
skew_tot_loo_s = moment3_tot_loo_s/np.sqrt(var_tot_loo_s)**3
# elpd
mean_tot_elpd_s, var_tot_elpd_s, moment3_tot_elpd_s = (
    calc_tot_mean_var_moment3_form_given_x(
        mean_elpd_s, var_elpd_s, moment3_elpd_s)
)
skew_tot_elpd_s = moment3_tot_elpd_s/np.sqrt(var_tot_elpd_s)**3
# err
mean_tot_err_s, var_tot_err_s, moment3_tot_err_s = (
    calc_tot_mean_var_moment3_form_given_x(
        mean_err_s, var_err_s, moment3_err_s)
)
skew_tot_err_s = moment3_tot_err_s/np.sqrt(var_tot_err_s)**3


# calc total mean, variance and skew with bb
# loo
mean_tot_loo_sb, var_tot_loo_sb, moment3_tot_loo_sb = (
    calc_tot_mean_var_moment3_form_given_x_bb(
        mean_loo_s, var_loo_s, moment3_loo_s)
)
skew_tot_loo_sb = moment3_tot_loo_sb/np.sqrt(var_tot_loo_sb)**3
# elpd
mean_tot_elpd_sb, var_tot_elpd_sb, moment3_tot_elpd_sb = (
    calc_tot_mean_var_moment3_form_given_x_bb(
        mean_elpd_s, var_elpd_s, moment3_elpd_s)
)
skew_tot_elpd_sb = moment3_tot_elpd_sb/np.sqrt(var_tot_elpd_sb)**3
# err
mean_tot_err_sb, var_tot_err_sb, moment3_tot_err_sb = (
    calc_tot_mean_var_moment3_form_given_x_bb(
        mean_err_s, var_err_s, moment3_err_s)
)
skew_tot_err_sb = moment3_tot_err_sb/np.sqrt(var_tot_err_sb)**3


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


# ===========================================================================
# plots
# ===========================================================================

# configs

fontsize = 16

plot_multilines = False
multilines_max = 50
multilines_alpha = 0.05

plot_only_median = False

datas = [
    [mean_loo_s/np.sqrt(var_loo_s), skew_loo_s],
    [mean_elpd_s/np.sqrt(var_elpd_s), skew_elpd_s],
    [mean_err_s/np.sqrt(var_err_s), skew_err_s],
]
data_names = [
    r'$\widehat{\mathrm{elpd}}_\mathrm{LOO}$',
    r'$\mathrm{elpd}$',
    r'$\mathrm{err}_\mathrm{LOO}$'
]
data_statistic_names = ['mean/SD', 'skewness']
show_zero_line = [
    [True, True],
    [True, True],
    [True, True],
]


fig, axes = plt.subplots(
    len(datas[0]), len(datas), sharex=True, figsize=(10,6))

for d_j, data_i in enumerate(datas):

    for d_i, data_ij in enumerate(data_i):

        ax = axes[d_i, d_j]

        for b_i, beta_t in enumerate(beta_t_s):
            color = 'C{}'.format(b_i)
            label = r'${}$'.format(beta_t)
            data = data_ij[:, b_i]
            if plot_multilines:
                median = np.percentile(data, 50, axis=-1)
                ax.plot(n_obs_s, median, color=color, label=label)
                ax.plot(
                    n_obs_s,
                    data[:,:multilines_max],
                    color=color,
                    alpha=multilines_alpha
                )
            else:
                if not plot_only_median:
                    q025 = np.percentile(data, 2.5, axis=-1)
                    q975 = np.percentile(data, 97.5, axis=-1)
                    ax.fill_between(n_obs_s, q025, q975, alpha=0.2, color=color)
                median = np.percentile(data, 50, axis=-1)
                ax.plot(n_obs_s, median, color=color, label=label)

        if show_zero_line[d_j][d_i]:
            ax.axhline(0, color='gray', lw=1.0)#, zorder=0)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize-4)

for ax, name in zip(axes[:, 0], data_statistic_names):
    ax.set_ylabel(name, fontsize=fontsize)

for ax, name in zip(axes[0, :], data_names):
    ax.set_title(name, fontsize=fontsize)

for ax in axes[-1, :]:
    ax.set_xlabel(r'$n$', fontsize=fontsize-2)

fig.tight_layout()

axes[1, -1].legend(
    loc='center left', bbox_to_anchor=(1, 0.5),
    fontsize=fontsize-2, fancybox=False,
    title=r'$\beta_t$',
    title_fontsize=fontsize-2,
)

fig.subplots_adjust(right=0.88)


# ============================================================================
# plot tot moments
# ============================================================================

# fontsize = 16
#
# datas = [
#     [mean_tot_loo_s/np.sqrt(var_tot_loo_s), skew_tot_loo_s],
#     [mean_tot_elpd_s/np.sqrt(var_tot_elpd_s), skew_tot_elpd_s],
#     [mean_tot_err_s/np.sqrt(var_tot_err_s), skew_tot_err_s],
# ]
# data_names = [
#     r'$\widehat{\mathrm{elpd}}_\mathrm{LOO}$',
#     r'$\mathrm{elpd}$',
#     r'$\mathrm{err}_\mathrm{LOO}$'
# ]
# data_statistic_names = ['mean/SD', 'skewness']
# show_zero_line = [
#     [True, True],
#     [True, True],
#     [True, True],
# ]
#
# fig, axes = plt.subplots(
#     len(datas[0]), len(datas), sharex=True, figsize=(10,6))
#
# for d_j, data_i in enumerate(datas):
#
#     for d_i, data_ij in enumerate(data_i):
#
#         ax = axes[d_i, d_j]
#
#         for b_i, beta_t in enumerate(beta_t_s):
#             color = 'C{}'.format(b_i)
#             label = r'${}$'.format(beta_t)
#             data = data_ij[:, b_i]
#             ax.plot(n_obs_s, data, color=color, label=label)
#
#         if show_zero_line[d_j][d_i]:
#             ax.axhline(0, color='gray', lw=1.0)#, zorder=0)
#
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
#         ax.tick_params(axis='both', which='minor', labelsize=fontsize-4)
#
# for ax, name in zip(axes[:, 0], data_statistic_names):
#     ax.set_ylabel(name, fontsize=fontsize)
#
# for ax, name in zip(axes[0, :], data_names):
#     ax.set_title(name, fontsize=fontsize)
#
# for ax in axes[-1, :]:
#     ax.set_xlabel(r'$n$', fontsize=fontsize-2)
#
# fig.tight_layout()
#
# axes[1, -1].legend(
#     loc='center left', bbox_to_anchor=(1, 0.5),
#     fontsize=fontsize-2, fancybox=False,
#     title=r'$\beta_t$',
#     title_fontsize=fontsize-2,
# )
#
# fig.subplots_adjust(right=0.88)


# ============================================================================
# plot tot moments bb
# ============================================================================

fontsize = 16

datas = [
    [mean_tot_loo_sb/np.sqrt(var_tot_loo_sb), skew_tot_loo_sb],
    [mean_tot_elpd_sb/np.sqrt(var_tot_elpd_sb), skew_tot_elpd_sb],
    [mean_tot_err_sb/np.sqrt(var_tot_err_sb), skew_tot_err_sb],
]
data_names = [
    r'$\widehat{\mathrm{elpd}}_\mathrm{LOO}$',
    r'$\mathrm{elpd}$',
    r'$\mathrm{err}_\mathrm{LOO}$'
]
data_statistic_names = ['mean/SD', 'skewness']
show_zero_line = [
    [True, True],
    [True, True],
    [True, True],
]

fig, axes = plt.subplots(
    len(datas[0]), len(datas), sharex=True, figsize=(10,6))

for d_j, data_i in enumerate(datas):

    for d_i, data_ij in enumerate(data_i):

        ax = axes[d_i, d_j]

        for b_i, beta_t in enumerate(beta_t_s):
            color = 'C{}'.format(b_i)
            label = r'${}$'.format(beta_t)
            data = data_ij[:, b_i]
            q025 = np.percentile(data, 2.5, axis=-1)
            q975 = np.percentile(data, 97.5, axis=-1)
            ax.fill_between(n_obs_s, q025, q975, alpha=0.2, color=color)
            median = np.percentile(data, 50, axis=-1)
            ax.plot(n_obs_s, median, color=color, label=label)

        if show_zero_line[d_j][d_i]:
            ax.axhline(0, color='gray', lw=1.0)#, zorder=0)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize-4)

for ax, name in zip(axes[:, 0], data_statistic_names):
    ax.set_ylabel(name, fontsize=fontsize)

for ax, name in zip(axes[0, :], data_names):
    ax.set_title(name, fontsize=fontsize)

for ax in axes[-1, :]:
    ax.set_xlabel(r'$n$', fontsize=fontsize-2)

fig.tight_layout()

axes[1, -1].legend(
    loc='center left', bbox_to_anchor=(1, 0.5),
    fontsize=fontsize-2, fancybox=False,
    title=r'$\beta_\Delta$',
    title_fontsize=fontsize-2,
)

fig.subplots_adjust(right=0.88)
