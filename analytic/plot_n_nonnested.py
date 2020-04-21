
import sys, os, time
from functools import partial

import numpy as np
from scipy import linalg, stats

from setup import *



# ============================================================================
# conf

load_res = True
plot = True

# data seed
# np.random.RandomState().randint(np.iinfo(np.uint32).max)
data_seed = 247169102

# number of trials
n_trial = 8
# fixed model tau2 value
tau2 = 1.0
# dimensionality of X
n_dim = 3
# first dim intercept
intercept = True
# correlated dimension pairs in X
X_corr_dims = [[1, 2]]
# intercept coef
beta_intercept = 0.0
# other betas
beta_other = 1.0
# beta for model A
beta_A = 1.0
# beta for model B
beta_B = 0.2
# model A covariates
idx_a = [0,1]
# model B covariates
idx_b = [0,2]

# constant data variance
sigma_d_2 = 1.0

# plot configs
plot_multilines = False
multilines_alpha = 0.05

# grid parameters
n_obs_s = np.round(np.linspace(10, 300, 10)).astype(int)
# correlation coefs for correlated dims in X
xcorrs_s = np.array([-1.0, -0.9, -0.5, 0.0, 0.5, 0.9, 1.0])

# ============================================================================

if plot:
    import matplotlib.pyplot as plt
    # import seaborn as sns

# ============================================================================
# Funcs

class DataGeneration:

    def __init__(self, data_seed=None):
        self.rng = np.random.RandomState(seed=data_seed)

    def make_x(self, n_obs, n_dim, xcorrs):
        # X
        if intercept:
            # firs dim (column) ones for intercept
            if np.abs(xcorrs) < 1.0:
                cov = np.eye(n_dim-1)
                for d_i, d_j in X_corr_dims:
                    cov[d_i-1, d_j-1] = xcorrs
                    cov[d_j-1, d_i-1] = xcorrs
                X_mat = np.hstack((
                    np.ones((n_obs, 1)),
                    self.rng.multivariate_normal(
                        mean=np.zeros(n_dim-1),
                        cov=cov,
                        size=n_obs,
                        check_valid='raise',
                    )
                ))
            else:
                cov = np.eye(n_dim-1)
                X_mat = np.hstack((
                    np.ones((n_obs, 1)),
                    self.rng.multivariate_normal(
                        mean=np.zeros(n_dim-1),
                        cov=cov,
                        size=n_obs,
                        check_valid='raise',
                    )
                ))
                for d_i, d_j in X_corr_dims:
                    if xcorrs < 0:
                        X_mat[:,d_j] = -X_mat[:,d_i]
                    else:
                        X_mat[:,d_j] = X_mat[:,d_i]
        else:
            if np.abs(xcorrs) < 1.0:
                cov = np.eye(n_dim)
                for d_i, d_j in X_corr_dims:
                    cov[d_i, d_j] = xcorrs
                    cov[d_j, d_i] = xcorrs
                X_mat = self.rng.multivariate_normal(
                    mean=np.zeros(n_dim),
                    cov=cov,
                    size=n_obs,
                    check_valid='raise',
                )
            else:
                cov = np.eye(n_dim)
                X_mat = self.rng.multivariate_normal(
                    mean=np.zeros(n_dim),
                    cov=cov,
                    size=n_obs,
                    check_valid='raise',
                )
                for d_i, d_j in X_corr_dims:
                    if xcorrs < 0:
                        X_mat[:,d_j] = -X_mat[:,d_i]
                    else:
                        X_mat[:,d_j] = X_mat[:,d_i]
        return X_mat

# ============================================================================

# construct beta vector
beta = np.array([beta_other]*(n_dim-2)+[beta_A]+[beta_B])
if intercept:
    beta[0] = beta_intercept


# ============================================================================
# As a function of n

if load_res:
    res_file = np.load('res_n_nonnested.npz')
    mean_loo_s = res_file['mean_loo_s']
    var_loo_s = res_file['var_loo_s']
    skew_loo_s = res_file['skew_loo_s']
    mean_err_s = res_file['mean_err_s']
    var_err_s = res_file['var_err_s']
    skew_err_s = res_file['skew_err_s']
    res_file.close()

else:

    start_time = time.time()
    mean_loo_s = np.full((len(xcorrs_s), len(n_obs_s), n_trial), np.nan)
    var_loo_s = np.full((len(xcorrs_s), len(n_obs_s), n_trial), np.nan)
    skew_loo_s = np.full((len(xcorrs_s), len(n_obs_s), n_trial), np.nan)
    mean_err_s = np.full((len(xcorrs_s), len(n_obs_s), n_trial), np.nan)
    var_err_s = np.full((len(xcorrs_s), len(n_obs_s), n_trial), np.nan)
    skew_err_s = np.full((len(xcorrs_s), len(n_obs_s), n_trial), np.nan)

    for i0, xcorrs in enumerate(xcorrs_s):

        # progress
        cur_time_min = (time.time() - start_time)/60
        print('{}/{}, elapsed time: {:.2} min'.format(
            i0+1, len(xcorrs_s), cur_time_min), flush=True)

        # reset data seed for each xcorrs
        data_generation = DataGeneration(data_seed)
        make_x = data_generation.make_x

        for i1, n_obs in enumerate(n_obs_s):

            for t_i in range(n_trial):

                X_mat = make_x(n_obs, n_dim, xcorrs)
                mean_loo, var_loo, skew_loo, mean_err, var_err, skew_err = (
                    get_analytic_res(
                        X_mat, beta, tau2, idx_a, idx_b,
                        Sigma_d=sigma_d_2, mu_d=None
                    )
                )
                mean_loo_s[i0, i1, t_i] = mean_loo
                var_loo_s[i0, i1, t_i] = var_loo
                skew_loo_s[i0, i1, t_i] = skew_loo
                mean_err_s[i0, i1, t_i] = mean_err
                var_err_s[i0, i1, t_i] = var_err
                skew_err_s[i0, i1, t_i] = skew_err
    print('done', flush=True)

    np.savez_compressed(
        'res_n_nonnested.npz',
        mean_loo_s=mean_loo_s,
        var_loo_s=var_loo_s,
        skew_loo_s=skew_loo_s,
        mean_err_s=mean_err_s,
        var_err_s=var_err_s,
        skew_err_s=skew_err_s,
    )


# plots
if plot:

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8,8))

    # skew
    ax = axes[0]
    ax.axhline(0, color='red', lw=1.0)#, zorder=0)
    for i0, xcorrs in enumerate(xcorrs_s):
        color = 'C{}'.format(i0)
        label = 'xcorrs={}'.format(xcorrs)
        data = skew_err_s[i0]
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
            median = np.percentile(data, 50, axis=-1)
            q025 = np.percentile(data, 2.5, axis=-1)
            q975 = np.percentile(data, 97.5, axis=-1)
            ax.fill_between(n_obs_s, q025, q975, alpha=0.2, color=color)
            # ax.plot(n_obs_s, q025, alpha=0.2, color=color)
            # ax.plot(n_obs_s, q975, alpha=0.2, color=color)
            ax.plot(n_obs_s, median, color=color, label=label)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    ax.set_ylabel('skewness', fontsize=18)

    # z-score
    ax = axes[1]
    ax.axhline(0, color='red', lw=1.0)#, zorder=0)
    for i0, xcorrs in enumerate(xcorrs_s):
        color = 'C{}'.format(i0)
        label = 'xcorrs={}'.format(xcorrs)

        # data = 1-stats.norm.cdf(
        #     0, loc=mean_err_s[i0], scale=np.sqrt(var_err_s[i0]))
        data = mean_err_s[i0] / np.sqrt(var_err_s[i0])

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
            median = np.percentile(data, 50, axis=-1)
            q025 = np.percentile(data, 2.5, axis=-1)
            q975 = np.percentile(data, 97.5, axis=-1)
            ax.fill_between(n_obs_s, q025, q975, alpha=0.2, color=color)
            # ax.plot(n_obs_s, q025, alpha=0.2, color=color)
            # ax.plot(n_obs_s, q975, alpha=0.2, color=color)
            ax.plot(n_obs_s, median, color=color, label=label)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    # ax.set_ylabel(r'$p(\mathrm{\widehat{elpd}_D}>0)$', fontsize=18)
    ax.set_ylabel('mean/sd', fontsize=18)
    ax.set_xlabel(r'$n$', fontsize=18)

    fig.tight_layout()
    for ax in axes:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.74, box.height])
    axes[0].legend(loc='center left', bbox_to_anchor=(1, -0.1), fontsize=16, fancybox=False)
