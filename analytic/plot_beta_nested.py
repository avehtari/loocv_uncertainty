
import sys, os, time
from functools import partial

import numpy as np
from scipy import linalg, stats

from problem_setting import *



# ============================================================================
# conf

load_res = False
plot = True
filename = 'res_n_nested.npz'


# data seed
# np.random.RandomState().randint(np.iinfo(np.uint32).max)
data_seed = 247169102

# number of trials
n_trial = 6
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
multilines_max = 50
multilines_alpha = 0.05

# grid parameters
n_obs_s = np.array([64, 128])
# last beta effect missing in model A
beta_t_s = np.linspace(0, 100, 10)

# ============================================================================

if plot:
    import matplotlib.pyplot as plt
    # import seaborn as sns

# ============================================================================
# Funcs

class DataGeneration:

    def __init__(self, n_obs_max, data_seed=None):
        self.rng = np.random.RandomState(seed=data_seed)
        if intercept:
            # firs dim (column) ones for intercept
            self.X_mat_all = np.hstack((
                np.ones((n_obs_max, 1)),
                self.rng.randn(n_obs_max, n_dim-1)
            ))
        else:
            self.X_mat_all = self.rng.randn(n_obs_max, n_dim)

    def make_x(self, n_obs):
        return self.X_mat_all[:n_obs,:]

# ============================================================================

# construct beta vectors
beta_s = np.zeros((len(beta_t_s), n_dim))
for b_i, beta_t in enumerate(beta_t_s):
    beta = np.array([beta_other]*(n_dim-1)+[beta_t])
    if intercept:
        beta[0] = beta_intercept
    beta_s[b_i] = beta

data_generation = DataGeneration(np.max(n_obs_s), data_seed)
make_x = data_generation.make_x


# ============================================================================
# As a function of n

if load_res:
    res_file = np.load(filename)
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

                X_mat = make_x(n_obs)
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
        filename,
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
    for n_i, n_obs in enumerate(n_obs_s):
        color = 'C{}'.format(n_i)
        label = r'$n={}$'.format(n_obs)
        data = skew_err_s[:,n_i]
        if plot_multilines:
            median = np.percentile(data, 50, axis=-1)
            ax.plot(beta_t_s, median, color=color, label=label)
            ax.plot(
                beta_t_s,
                data[:,:multilines_max],
                color=color,
                alpha=multilines_alpha
            )
        else:
            median = np.percentile(data, 50, axis=-1)
            q025 = np.percentile(data, 2.5, axis=-1)
            q975 = np.percentile(data, 97.5, axis=-1)
            ax.fill_between(beta_t_s, q025, q975, alpha=0.2, color=color)
            ax.plot(beta_t_s, median, color=color, label=label)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    ax.set_ylabel('skewness', fontsize=18)

    # mean
    ax = axes[1]
    ax.axhline(0, color='red', lw=1.0)#, zorder=0)
    for n_i, n_obs in enumerate(n_obs_s):
        color = 'C{}'.format(n_i)
        label = r'$n={}$'.format(n_obs)

        # data = 1-stats.norm.cdf(
        #     0, loc=mean_err_s[:,n_i], scale=np.sqrt(var_err_s[:,n_i]))
        data = mean_err_s[:,n_i] / np.sqrt(var_err_s[:,n_i])
        # data = mean_err_s[:,n_i]

        if plot_multilines:
            median = np.percentile(data, 50, axis=-1)
            ax.plot(beta_t_s, median, color=color, label=label)
            ax.plot(
                beta_t_s,
                data[:,:multilines_max],
                color=color,
                alpha=multilines_alpha
            )
        else:
            median = np.percentile(data, 50, axis=-1)
            q025 = np.percentile(data, 2.5, axis=-1)
            q975 = np.percentile(data, 97.5, axis=-1)
            ax.fill_between(beta_t_s, q025, q975, alpha=0.2, color=color)
            ax.plot(beta_t_s, median, color=color, label=label)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    # ax.set_ylabel(r'$p(\mathrm{\widehat{elpd}_D}>0)$', fontsize=18)
    ax.set_ylabel('mean/sd', fontsize=18)
    # ax.set_ylabel('mean', fontsize=18)
    ax.set_xlabel(r'$n$', fontsize=18)

    fig.tight_layout()
    for ax in axes:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.74, box.height])
    axes[0].legend(loc='center left', bbox_to_anchor=(1, -0.1), fontsize=16, fancybox=False)
