
import sys, os, time
from functools import partial

import numpy as np
from scipy import linalg, stats

from problem_setting import *

import seaborn as sns
import pandas as pd


# ============================================================================
# conf

load_res = False
filename = 'res_nonnested_n_xcorr.npz'

plot = True

# data seed
# np.random.RandomState().randint(np.iinfo(np.uint32).max)
data_seed = 247169102

# number of trials
n_trial = 100
# fixed model tau2 value
tau2 = 1.0
# dimensionality of X
n_dim = 3
# correlated dimension pairs in X
X_corr_dims = [[1, 2]]
# first dim intercept
intercept = True
# intercept coef
beta_intercept = 0.0
# other betas
beta_other = 1.0
# beta for model A
beta_A = 1.0
# beta for model B
beta_B = 1.0
# model A covariates
idx_a = [0,1]
# model B covariates
idx_b = [0,2]

# constant data variance
sigma_d_2 = 1.0

# tot mean knwon to be zero
known_zero_mean_tot = True

# grid parameters
# n_obs_s = np.round(np.linspace(10, 1000, 2)).astype(int)
n_obs_s = np.array([32, 128, 512])
# correlation coefs for correlated dims in X
# xcorrs_s = np.array([-0.9, -0.5, 0.0, 0.5, 0.9])
xcorrs_s = np.array([-0.8, 0.0, 0.8])

# ============================================================================

if plot:
    import matplotlib.pyplot as plt
    # import seaborn as sns

# ============================================================================
# Funcs

class DataGeneration:

    def __init__(self, n_trial, n_obs_max, xcorrs, data_seed=None):
        self.rng = np.random.RandomState(seed=data_seed)
        self.X_mat_all = self._make_x(n_trial, n_obs_max, xcorrs)

    def _make_x(self, n_trial, n_obs, xcorrs):
        # X
        if intercept:
            # firs dim (column) ones for intercept
            if np.abs(xcorrs) < 1.0:
                cov = np.eye(n_dim-1)
                for d_i, d_j in X_corr_dims:
                    cov[d_i-1, d_j-1] = xcorrs
                    cov[d_j-1, d_i-1] = xcorrs
                X_mat = np.dstack((
                    np.ones((n_trial, n_obs, 1)),
                    self.rng.multivariate_normal(
                        mean=np.zeros(n_dim-1),
                        cov=cov,
                        size=(n_trial, n_obs),
                        check_valid='raise',
                    )
                ))
            else:
                cov = np.eye(n_dim-1)
                X_mat = np.dstack((
                    np.ones((n_trial, n_obs, 1)),
                    self.rng.multivariate_normal(
                        mean=np.zeros(n_dim-1),
                        cov=cov,
                        size=(n_trial, n_obs),
                        check_valid='raise',
                    )
                ))
                for d_i, d_j in X_corr_dims:
                    if xcorrs < 0:
                        X_mat[:,:,d_j] = -X_mat[:,:,d_i]
                    else:
                        X_mat[:,:,d_j] = X_mat[:,:,d_i]
        else:
            if np.abs(xcorrs) < 1.0:
                cov = np.eye(n_dim)
                for d_i, d_j in X_corr_dims:
                    cov[d_i, d_j] = xcorrs
                    cov[d_j, d_i] = xcorrs
                X_mat = self.rng.multivariate_normal(
                    mean=np.zeros(n_dim),
                    cov=cov,
                    size=(n_trial, n_obs),
                    check_valid='raise',
                )
            else:
                cov = np.eye(n_dim)
                X_mat = self.rng.multivariate_normal(
                    mean=np.zeros(n_dim),
                    cov=cov,
                    size=(n_trial, n_obs),
                    check_valid='raise',
                )
                for d_i, d_j in X_corr_dims:
                    if xcorrs < 0:
                        X_mat[:,:,d_j] = -X_mat[:,:,d_i]
                    else:
                        X_mat[:,:,d_j] = X_mat[:,:,d_i]
        return X_mat

    def get_x(self, trial_i, n_obs):
        return self.X_mat_all[trial_i, :n_obs, :]

# ============================================================================

# construct beta vector
beta = np.array([beta_other]*(n_dim-2)+[beta_A]+[beta_B])
if intercept:
    beta[0] = beta_intercept


# ============================================================================
# As a function of n

if load_res:
    res_file = np.load(filename)
    mean_loo_s = res_file['mean_loo_s']
    var_loo_s = res_file['var_loo_s']
    moment3_loo_s = res_file['moment3_loo_s']
    mean_elpd_s = res_file['mean_elpd_s']
    var_elpd_s = res_file['var_elpd_s']
    moment3_elpd_s = res_file['moment3_elpd_s']
    mean_err_s = res_file['mean_err_s']
    var_err_s = res_file['var_err_s']
    moment3_err_s = res_file['moment3_err_s']
    res_file.close()

else:

    start_time = time.time()
    mean_loo_s = np.full((len(xcorrs_s), len(n_obs_s), n_trial), np.nan)
    var_loo_s = np.full((len(xcorrs_s), len(n_obs_s), n_trial), np.nan)
    moment3_loo_s = np.full((len(xcorrs_s), len(n_obs_s), n_trial), np.nan)
    mean_elpd_s = np.full((len(xcorrs_s), len(n_obs_s), n_trial), np.nan)
    var_elpd_s = np.full((len(xcorrs_s), len(n_obs_s), n_trial), np.nan)
    moment3_elpd_s = np.full((len(xcorrs_s), len(n_obs_s), n_trial), np.nan)
    mean_err_s = np.full((len(xcorrs_s), len(n_obs_s), n_trial), np.nan)
    var_err_s = np.full((len(xcorrs_s), len(n_obs_s), n_trial), np.nan)
    moment3_err_s = np.full((len(xcorrs_s), len(n_obs_s), n_trial), np.nan)

    for i0, xcorrs in enumerate(xcorrs_s):

        # progress
        cur_time_min = (time.time() - start_time)/60
        print('{}/{}, elapsed time: {:.2} min'.format(
            i0+1, len(xcorrs_s), cur_time_min), flush=True)

        data_generation = DataGeneration(
            n_trial, np.max(n_obs_s), xcorrs, data_seed)

        for i1, n_obs in enumerate(n_obs_s):

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
                mean_loo_s[i0, i1, t_i] = mean_loo
                var_loo_s[i0, i1, t_i] = var_loo
                moment3_loo_s[i0, i1, t_i] = moment3_loo
                mean_elpd_s[i0, i1, t_i] = mean_elpd
                var_elpd_s[i0, i1, t_i] = var_elpd
                moment3_elpd_s[i0, i1, t_i] = moment3_elpd
                mean_err_s[i0, i1, t_i] = mean_err
                var_err_s[i0, i1, t_i] = var_err
                moment3_err_s[i0, i1, t_i] = moment3_err
    print('done', flush=True)

    np.savez_compressed(
        filename,
        mean_loo_s=mean_loo_s,
        var_loo_s=var_loo_s,
        moment3_loo_s=moment3_loo_s,
        mean_elpd_s=mean_elpd_s,
        var_elpd_s=var_elpd_s,
        moment3_elpd_s=moment3_elpd_s,
        mean_err_s=mean_err_s,
        var_err_s=var_err_s,
        moment3_err_s=moment3_err_s,
    )

# ============================================================================
if not plot:
    # all done if not plotting anything
    raise SystemExit
# ============================================================================

# cal skews
skew_loo_s = moment3_loo_s/np.sqrt(var_loo_s)**3
skew_elpd_s = moment3_elpd_s/np.sqrt(var_elpd_s)**3
skew_err_s = moment3_err_s/np.sqrt(var_err_s)**3


# calc total mean, variance and skew
# loo
mean_tot_loo_s, var_tot_loo_s, moment3_tot_loo_s = (
    calc_tot_mean_var_moment3_form_given_x(
        mean_loo_s, var_loo_s, moment3_loo_s, known_zero_mean_tot)
)
skew_tot_loo_s = moment3_tot_loo_s/np.sqrt(var_tot_loo_s)**3
# elpd
mean_tot_elpd_s, var_tot_elpd_s, moment3_tot_elpd_s = (
    calc_tot_mean_var_moment3_form_given_x(
        mean_elpd_s, var_elpd_s, moment3_elpd_s, known_zero_mean_tot)
)
skew_tot_elpd_s = moment3_tot_elpd_s/np.sqrt(var_tot_elpd_s)**3
# err
mean_tot_err_s, var_tot_err_s, moment3_tot_err_s = (
    calc_tot_mean_var_moment3_form_given_x(
        mean_err_s, var_err_s, moment3_err_s, known_zero_mean_tot)
)
skew_tot_err_s = moment3_tot_err_s/np.sqrt(var_tot_err_s)**3



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

datas = [
    [np.sqrt(var_tot_loo_s)/n_obs_s[None, :], skew_tot_loo_s],
    [np.sqrt(var_tot_elpd_s)/n_obs_s[None, :], skew_tot_elpd_s],
    [np.sqrt(var_tot_err_s)/n_obs_s[None, :], skew_tot_err_s],
]
data_names = [
    r'$\widehat{\mathrm{elpd}}_\mathrm{LOO}$',
    r'$\mathrm{elpd}$',
    r'$\mathrm{err}_\mathrm{LOO}$'
]
data_statistic_names = [r'$\mathrm{SD}/n$', 'skewness']
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

        for xc_i, xcorrs in enumerate(xcorrs_s):
            color = 'C{}'.format(xc_i)
            label = '{}'.format(xcorrs)
            data = data_ij[xc_i]
            ax.plot(n_obs_s, data, color=color, label=label)

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
    title=r'$\mathrm{corr}(X_{[\cdot, 1]},X_{[\cdot, 2]})$',
    title_fontsize=fontsize-2,
)

fig.subplots_adjust(right=0.80)


# ===========================================================================
# violins
# ===========================================================================


fontsize = 16

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

        # construct DF (so inconvenient and wasteful :( )
        dfarr_xcorrs = np.zeros(np.prod(data_ij.shape))
        dfarr_ns = np.zeros(np.prod(data_ij.shape))
        dfarr_data = np.zeros(np.prod(data_ij.shape))
        tot_i = 0
        for xc_i, xcorrs in enumerate(xcorrs_s):
            for n_obs_i, n_obs in enumerate(n_obs_s):
                dfarr_xcorrs[tot_i:tot_i+n_trial] = xcorrs
                dfarr_ns[tot_i:tot_i+n_trial] = n_obs
                dfarr_data[tot_i:tot_i+n_trial] = data_ij[xc_i, n_obs_i, :]
                tot_i += n_trial
        df = pd.DataFrame(dict(
            xcorr=dfarr_xcorrs,
            n_obs=dfarr_ns,
            data=dfarr_data
        ))

        sns.violinplot(
            x='n_obs', y='data', hue='xcorr', data=df,
            scale='width',
            palette='Set2',
            linewidth=0.5,
            ax=ax,
        )
        ax.get_legend().set_visible(False)
        ax.set_xlabel(None)
        ax.set_ylabel(None)

        if show_zero_line[d_j][d_i]:
            ax.axhline(0, color='red', lw=1.0)#, zorder=0)

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
    ax.set_xticklabels(n_obs_s)

fig.tight_layout()

axes[-1, 1].legend(
    loc='upper center', bbox_to_anchor=(0.5, -0.3),
    fontsize=fontsize-2, fancybox=False,
    title=r'$\mathrm{corr}(X_{[\cdot, 1]},X_{[\cdot, 2]})$',
    title_fontsize=fontsize-2,
    ncol=len(xcorrs_s)
)
fig.subplots_adjust(bottom=0.25)
