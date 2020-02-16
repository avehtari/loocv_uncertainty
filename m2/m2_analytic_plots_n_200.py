
import sys, os, time

import numpy as np
from scipy import linalg, stats

from m2_setup import *



# conf
load_res = False
plot = True

plot_multilines = True
multilines_max = 100
multilines_alpha = 0.05


# ============================================================================

if plot:
    import matplotlib.pyplot as plt
    import seaborn as sns


# ============================================================================
# As a function of n

if load_res:
    res_file = np.load('m2_res_n_200.npz')
    analytic_mean_s = res_file['analytic_mean_s']
    analytic_var_s = res_file['analytic_var_s']
    analytic_skew_s = res_file['analytic_skew_s']
    analytic_coefvar_s = res_file['analytic_coefvar_s']
    n_obs_s = res_file['n_obs_s']
    prc_out_s = res_file['prc_out_s']
    beta_t_s = res_file['beta_t_s']
    sigma2_d = res_file['sigma2_d']
    res_file.close()

else:

    # variables

    # n_obs_s
    # n_obs_s = [16, 32, 64, 128, 256, 512, 1024]
    # n_obs_s = [16, 32, 64, 128, 256]
    n_obs_s = np.round(np.linspace(20, 300, 10)).astype(int)

    # beta_t
    beta_t_s = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]

    # constants
    prc_out = 0.0
    sigma2_d = 1.0


    start_time = time.time()
    analytic_mean_s = np.full(
        (len(beta_t_s), len(n_obs_s), n_trial), np.nan)
    analytic_var_s = np.full(
        (len(beta_t_s), len(n_obs_s), n_trial), np.nan)
    analytic_skew_s = np.full(
        (len(beta_t_s), len(n_obs_s), n_trial), np.nan)
    analytic_coefvar_s = np.full(
        (len(beta_t_s), len(n_obs_s), n_trial), np.nan)
    for i3, beta_t in enumerate(beta_t_s):
        cur_time_min = (time.time() - start_time)/60
        print('{}/{}, elapsed time: {:.2} min'.format(
            i3+1, len(beta_t_s), cur_time_min), flush=True)
        for i1, n_obs in enumerate(n_obs_s):
            n_obs_out = determine_n_obs_out(n_obs, prc_out)
            for t_i in range(n_trial):
                X_mat, mu_d = make_x_mu(
                    n_obs, n_obs_out, sigma2_d, beta_t)
                if n_obs_out == 0:
                    mu_d = None
                mean, var, _, coefvar, skew = get_analytic_res(
                    X_mat, beta_t, sigma2_d, mu_d)
                analytic_mean_s[i3, i1, t_i] = mean
                analytic_var_s[i3, i1, t_i] = var
                analytic_coefvar_s[i3, i1, t_i] = coefvar
                analytic_skew_s[i3, i1, t_i] = skew
    print('done', flush=True)

    np.savez_compressed(
        'm2_res_n_200.npz',
        analytic_mean_s=analytic_mean_s,
        analytic_var_s=analytic_var_s,
        analytic_skew_s=analytic_skew_s,
        analytic_coefvar_s=analytic_coefvar_s,
        n_obs_s=n_obs_s,
        prc_out=prc_out,
        beta_t_s=beta_t_s,
        sigma2_d=sigma2_d,
    )


# plots
if plot:

n_obs_s = np.asarray(n_obs_s)

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8,8))

# skew
ax = axes[0]
for b_i, beta_t in enumerate(beta_t_s):
    ax.axhline(0, color='red', lw=1.0)#, zorder=0)
    color = 'C{}'.format(b_i)
    label = r'$\beta_t={}$'.format(beta_t)
    data = analytic_skew_s[b_i]
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

# p_looneg_naive
ax = axes[1]
for b_i, beta_t in enumerate(beta_t_s):
    color = 'C{}'.format(b_i)
    label = r'$\beta_t={}$'.format(beta_t)

    data = 1-stats.norm.cdf(
        0, loc=analytic_mean_s[b_i], scale=np.sqrt(analytic_var_s[b_i]))

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

ax.set_ylim(-0.1, 1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.tick_params(axis='both', which='minor', labelsize=14)
ax.set_ylabel(r'$p(\mathrm{\widehat{elpd}_D}>0)$', fontsize=18)
ax.set_xlabel(r'$n$', fontsize=18)

fig.tight_layout()
for ax in axes:
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.74, box.height])
axes[0].legend(loc='center left', bbox_to_anchor=(1, -0.1), fontsize=16, fancybox=False)
