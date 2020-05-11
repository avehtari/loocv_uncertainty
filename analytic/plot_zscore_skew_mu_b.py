
import sys, os, time
from functools import partial

import numpy as np
from scipy import linalg, stats

from problem_setting import *



# ============================================================================
# conf

load_res = False
filename = 'res_zscore_skew_mu_b.npz'

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

# ------------
# fixed params
# ------------

# number of observations
n_obs = 32

# base outlier mu vector

# # 1
mu_base = np.zeros(n_obs)
mu_base[0] = 1

# # 2
# mu_base = np.zeros(n_obs)
# mu_base[0] = 1
# mu_base[1] = -1

# # 3
# mu_base = np.zeros(n_obs)
# mu_base[:n_obs//2] = 1

# 4
# mu_base = np.ones(n_obs)

# ---------------
# grid parameters
# ---------------

# outliers
mu_r_s = np.linspace(-200, 200, 21)

# last beta effect missing in model A
beta_t_s = np.array([0.0, 0.1, 0.2, 0.5, 1.0])
# beta_t_s = np.array([1.0])

# ============================================================================

if plot:
    import matplotlib.pyplot as plt
    # import seaborn as sns

# ============================================================================
# Funcs

class DataGeneration:

    def __init__(self, n_trial, data_seed=None):
        self.rng = np.random.RandomState(seed=data_seed)
        if intercept:
            # firs dim (column) ones for intercept
            self.X_mat_all = np.concatenate(
                (
                    np.ones((n_trial, n_obs, 1)),
                    self.rng.randn(n_trial, n_obs, n_dim-1)
                ),
                axis=-1,
            )
        else:
            self.X_mat_all = self.rng.randn(n_trial, n_obs, n_dim)

    def get_x(self, trial_i):
        return self.X_mat_all[trial_i, :, :]

# ============================================================================

# construct beta vectors
beta_s = np.zeros((len(beta_t_s), n_dim))
for b_i, beta_t in enumerate(beta_t_s):
    beta = np.array([beta_other]*(n_dim-1)+[beta_t])
    if intercept:
        beta[0] = beta_intercept
    beta_s[b_i] = beta

# construct mu vectors
mu_s =  mu_r_s[:,None]*mu_base

data_generation = DataGeneration(n_trial, data_seed)


# ============================================================================
# As a function of n

if load_res:
    res_file = np.load(filename)
    mean_loo_s = res_file['mean_loo_s']
    var_loo_s = res_file['var_loo_s']
    skew_loo_s = res_file['skew_loo_s']
    mean_elpd_s = res_file['mean_elpd_s']
    var_elpd_s = res_file['var_elpd_s']
    skew_elpd_s = res_file['skew_elpd_s']
    mean_err_s = res_file['mean_err_s']
    var_err_s = res_file['var_err_s']
    skew_err_s = res_file['skew_err_s']
    res_file.close()

else:

    start_time = time.time()
    mean_loo_s = np.full((len(beta_t_s), len(mu_r_s), n_trial), np.nan)
    var_loo_s = np.full((len(beta_t_s), len(mu_r_s), n_trial), np.nan)
    skew_loo_s = np.full((len(beta_t_s), len(mu_r_s), n_trial), np.nan)
    mean_elpd_s = np.full((len(beta_t_s), len(mu_r_s), n_trial), np.nan)
    var_elpd_s = np.full((len(beta_t_s), len(mu_r_s), n_trial), np.nan)
    skew_elpd_s = np.full((len(beta_t_s), len(mu_r_s), n_trial), np.nan)
    mean_err_s = np.full((len(beta_t_s), len(mu_r_s), n_trial), np.nan)
    var_err_s = np.full((len(beta_t_s), len(mu_r_s), n_trial), np.nan)
    skew_err_s = np.full((len(beta_t_s), len(mu_r_s), n_trial), np.nan)

    for i0, beta in enumerate(beta_s):

        # progress
        cur_time_min = (time.time() - start_time)/60
        print('{}/{}, elapsed time: {:.2} min'.format(
            i0+1, len(beta_t_s), cur_time_min), flush=True)

        for i1, mu in enumerate(mu_s):

            for t_i in range(n_trial):

                X_mat = data_generation.get_x(t_i)
                (
                    mean_loo, var_loo, moment3_loo,
                    mean_elpd, var_elpd, moment3_elpd,
                    mean_err, var_err, moment3_err
                ) = (
                    get_analytic_res(
                        X_mat, beta, tau2, idx_a, idx_b,
                        Sigma_d=sigma_d_2, mu_d=mu
                    )
                )
                mean_loo_s[i0, i1, t_i] = mean_loo
                var_loo_s[i0, i1, t_i] = var_loo
                skew_loo_s[i0, i1, t_i] = moment3_loo/np.sqrt(var_loo)**3
                mean_elpd_s[i0, i1, t_i] = mean_elpd
                var_elpd_s[i0, i1, t_i] = var_elpd
                skew_elpd_s[i0, i1, t_i] = moment3_elpd/np.sqrt(var_elpd)**3
                mean_err_s[i0, i1, t_i] = mean_err
                var_err_s[i0, i1, t_i] = var_err
                skew_err_s[i0, i1, t_i] = moment3_err/np.sqrt(var_err)**3
    print('done', flush=True)

    np.savez_compressed(
        filename,
        mean_loo_s=mean_loo_s,
        var_loo_s=var_loo_s,
        skew_loo_s=skew_loo_s,
        mean_elpd_s=mean_elpd_s,
        var_elpd_s=var_elpd_s,
        skew_elpd_s=skew_elpd_s,
        mean_err_s=mean_err_s,
        var_err_s=var_err_s,
        skew_err_s=skew_err_s,
    )

if not plot:
    # all done if not plotting anything
    raise SystemExit


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

plot_only_median = True

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
    [True, False],
    [True, True],
]


fig, axes = plt.subplots(
    len(datas[0]), len(datas), sharex=True, figsize=(10,6))

for d_j, data_i in enumerate(datas):

    for d_i, data_ij in enumerate(data_i):

        ax = axes[d_i, d_j]

        for b_i, beta_t in enumerate(beta_t_s):
            color = 'C{}'.format(b_i)
            label = r'$\beta_t={}$'.format(beta_t)
            data = data_ij[b_i]
            if plot_multilines:
                median = np.percentile(data, 50, axis=-1)
                ax.plot(mu_r_s, median, color=color, label=label)
                ax.plot(
                    mu_r_s,
                    data[:,:multilines_max],
                    color=color,
                    alpha=multilines_alpha
                )
            else:
                if not plot_only_median:
                    q025 = np.percentile(data, 2.5, axis=-1)
                    q975 = np.percentile(data, 97.5, axis=-1)
                    ax.fill_between(mu_r_s, q025, q975, alpha=0.2, color=color)
                median = np.percentile(data, 50, axis=-1)
                ax.plot(mu_r_s, median, color=color, label=label)

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
    ax.set_xlabel(r'$\mu_{\star,\mathrm{r}}$', fontsize=fontsize-2)

fig.tight_layout()

axes[1, -1].legend(
    loc='center left', bbox_to_anchor=(1, 0.5),
    fontsize=fontsize-2, fancybox=False,
)

fig.subplots_adjust(right=0.80)
