
import sys, os, time
from functools import partial

import numpy as np
from scipy import linalg, stats

from problem_setting import *



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

# plot configs
plot_multilines = False
multilines_alpha = 0.05

# grid parameters
n_obs_s = np.round(np.linspace(10, 300, 10)).astype(int)
# last beta effect missing in model A
beta_t_s = np.array([0.0, 0.05, 0.1, 0.2, 0.5, 1.0])

# ============================================================================

if plot:
    import matplotlib.pyplot as plt
    # import seaborn as sns

# ============================================================================
# Funcs

class DataGeneration:

    def __init__(self, data_seed=None):
        self.rng = np.random.RandomState(seed=data_seed)

    def make_x(self, n_obs, n_dim):
        # X
        if intercept:
            # firs dim (column) ones for intercept
            # X_mat = np.hstack((
            #     np.ones((n_obs, 1)),
            #     self.rng.uniform(low=-1.0, high=1.0, size=(n_obs, n_dim-1))
            # ))
            X_mat = np.hstack((
                np.ones((n_obs, 1)),
                self.rng.randn(n_obs, n_dim-1)
            ))
        else:
            # X_mat = self.rng.uniform(low=-1.0, high=1.0, size=(n_obs, n_dim))
            X_mat = self.rng.randn(n_obs, n_dim)
        return X_mat

# ============================================================================

# construct beta vectors
beta_s = np.zeros((len(beta_t_s), n_dim))
for b_i, beta_t in enumerate(beta_t_s):
    beta = np.array([beta_other]*(n_dim-1)+[beta_t])
    if intercept:
        beta[0] = beta_intercept
    beta_s[b_i] = beta

data_generation = DataGeneration(data_seed)
make_x = data_generation.make_x


# ============================================================================
# As a function of n

if load_res:
    res_file = np.load('res_n_nested.npz')
    mean_loo_s = res_file['mean_loo_s']
    var_loo_s = res_file['var_loo_s']
    skew_loo_s = res_file['skew_loo_s']
    mean_err_s = res_file['mean_err_s']
    var_err_s = res_file['var_err_s']
    skew_err_s = res_file['skew_err_s']
    res_file.close()

else:

    start_time = time.time()
    mean_loo_s = np.full((len(beta_t_s), len(n_obs_s), n_trial), np.nan)
    var_loo_s = np.full((len(beta_t_s), len(n_obs_s), n_trial), np.nan)
    skew_loo_s = np.full((len(beta_t_s), len(n_obs_s), n_trial), np.nan)
    mean_err_s = np.full((len(beta_t_s), len(n_obs_s), n_trial), np.nan)
    var_err_s = np.full((len(beta_t_s), len(n_obs_s), n_trial), np.nan)
    skew_err_s = np.full((len(beta_t_s), len(n_obs_s), n_trial), np.nan)

    for i0, beta in enumerate(beta_s):

        # progress
        cur_time_min = (time.time() - start_time)/60
        print('{}/{}, elapsed time: {:.2} min'.format(
            i0+1, len(beta_t_s), cur_time_min), flush=True)

        for i1, n_obs in enumerate(n_obs_s):

            for t_i in range(n_trial):

                X_mat = make_x(n_obs, n_dim)
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
        'res_n_nested.npz',
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
    for b_i, beta_t in enumerate(beta_t_s):
        color = 'C{}'.format(b_i)
        label = r'$\beta_t={}$'.format(beta_t)
        data = skew_err_s[b_i]
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
            ax.plot(n_obs_s, median, color=color, label=label)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    ax.set_ylabel('skewness', fontsize=18)

    # z-score
    ax = axes[1]
    ax.axhline(0, color='red', lw=1.0)#, zorder=0)
    for b_i, beta_t in enumerate(beta_t_s):
        color = 'C{}'.format(b_i)
        label = r'$\beta_t={}$'.format(beta_t)

        # data = 1-stats.norm.cdf(
        #     0, loc=mean_err_s[b_i], scale=np.sqrt(var_err_s[b_i]))
        data = mean_err_s[b_i] / np.sqrt(var_err_s[b_i])

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
