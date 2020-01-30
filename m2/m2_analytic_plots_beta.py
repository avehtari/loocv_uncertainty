
import sys, os, time

import numpy as np
from scipy import linalg, stats

from m2_setup import *



# conf
load_res = True
plot = True

plot_multilines = True
multilines_max = 100
multilines_alpha = 0.05


# ============================================================================

if plot:
    import matplotlib.pyplot as plt
    import seaborn as sns


# ============================================================================
# As a function of beta_t


if load_res:
    res_file = np.load('m2_res_beta.npz')
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
    beta_t_s = np.linspace(0.0, 2.0, 20)
    # beta_t_s = np.linspace(0.0, 2.0, 5)

    n_obs_s = [64, 512]
    # n_obs_s = [64, 128]

    # prc_out
    prc_out_s = [0.0, 0.01]
    # prc_out_s = [0.0, np.nextafter(0,1), 0.01]

    # constants
    sigma2_d = 1.0

    start_time = time.time()
    analytic_mean_s = np.full(
        (len(prc_out_s), len(beta_t_s), len(n_obs_s), n_trial), np.nan)
    analytic_var_s = np.full(
        (len(prc_out_s), len(beta_t_s), len(n_obs_s), n_trial), np.nan)
    analytic_skew_s = np.full(
        (len(prc_out_s), len(beta_t_s), len(n_obs_s), n_trial), np.nan)
    analytic_coefvar_s = np.full(
        (len(prc_out_s), len(beta_t_s), len(n_obs_s), n_trial), np.nan)
    for i3, beta_t in enumerate(beta_t_s):
        cur_time_min = (time.time() - start_time)/60
        print('{}/{}, elapsed time: {:.2} min'.format(
            i3+1, len(beta_t_s), cur_time_min), flush=True)
        for i1, n_obs in enumerate(n_obs_s):
            for i2, prc_out in enumerate(prc_out_s):
                n_obs_out = determine_n_obs_out(n_obs, prc_out)
                for t_i in range(n_trial):
                    X_mat, mu_d = make_x_mu(
                        n_obs, n_obs_out, sigma2_d, beta_t)
                    if n_obs_out == 0:
                        mu_d = None
                    mean, var, _, coefvar, skew = get_analytic_res(
                        X_mat, beta_t, sigma2_d, mu_d)
                    analytic_mean_s[i2, i3, i1, t_i] = mean
                    analytic_var_s[i2, i3, i1, t_i] = var
                    analytic_coefvar_s[i2, i3, i1, t_i] = coefvar
                    analytic_skew_s[i2, i3, i1, t_i] = skew
    print('done', flush=True)

    np.savez_compressed(
        'm2_res_beta.npz',
        analytic_mean_s=analytic_mean_s,
        analytic_var_s=analytic_var_s,
        analytic_skew_s=analytic_skew_s,
        analytic_coefvar_s=analytic_coefvar_s,
        n_obs_s=n_obs_s,
        prc_out_s=prc_out_s,
        beta_t_s=beta_t_s,
        sigma2_d=sigma2_d,
    )


# plots
if plot:

    # mean
    fig, axes = plt.subplots(
        len(n_obs_s), len(prc_out_s), sharex=True, sharey=False, figsize=(9,6))
    for n_i, ax in enumerate(axes):
        for o_i, prc_out in enumerate(prc_out_s):
            ax = axes[n_i, o_i]
            ax.axhline(0, color='gray', lw=0.8, zorder=0)
            data = analytic_mean_s[o_i, :, n_i]/n_obs_s[n_i]
            if plot_multilines:
                ax.plot(
                    beta_t_s,
                    data[:,:multilines_max],
                    color='C0',
                    alpha=multilines_alpha
                )
                # limit y for outliers cases
                ax.set_ylim((
                    min(np.percentile(data, 2.5, axis=1).min(), 0),
                    max(np.percentile(data, 97.5, axis=1).max(), 0),
                ))
            else:
                median = np.percentile(data, 50, axis=-1)
                q025 = np.percentile(data, 2.5, axis=-1)
                q975 = np.percentile(data, 97.5, axis=-1)
                ax.fill_between(beta_t_s, q025, q975, alpha=0.2)
                ax.plot(beta_t_s, median)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    for ax, prc_out in zip(axes[0,:], prc_out_s):
        ax.set_title('prc_out={}'.format(prc_out))
    for ax in axes[-1,:]:
        ax.set_xlabel(r'$\beta_t$')
    for ax, n_obs in zip(axes[:,0], n_obs_s):
        ax.set_ylabel(r'$n={}$'.format(n_obs))
    # fig.suptitle('mean')
    fig.tight_layout()

    # coefvar
    fig, axes = plt.subplots(
        len(n_obs_s), len(prc_out_s), sharex=True, sharey=False, figsize=(9,6))
    for n_i, ax in enumerate(axes):
        for o_i, prc_out in enumerate(prc_out_s):
            ax = axes[n_i, o_i]
            ax.axhline(0, color='gray', lw=0.8, zorder=0)
            data = analytic_coefvar_s[o_i, :, n_i]
            if plot_multilines:
                ax.plot(
                    beta_t_s,
                    data[:,:multilines_max],
                    color='C0',
                    alpha=multilines_alpha
                )
                # limit y for outliers cases
                ax.set_ylim((
                    min(np.percentile(data, 2.5, axis=1).min(), 0),
                    max(np.percentile(data, 97.5, axis=1).max(), 0),
                ))
            else:
                median = np.percentile(data, 50, axis=-1)
                q025 = np.percentile(data, 2.5, axis=-1)
                q975 = np.percentile(data, 97.5, axis=-1)
                ax.fill_between(beta_t_s, q025, q975, alpha=0.2)
                ax.plot(beta_t_s, median)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    for ax, prc_out in zip(axes[0,:], prc_out_s):
        ax.set_title('prc_out={}'.format(prc_out))
    for ax in axes[-1,:]:
        ax.set_xlabel(r'$\beta_t$')
    for ax, n_obs in zip(axes[:,0], n_obs_s):
        ax.set_ylabel(r'$n={}$'.format(n_obs))
    # fig.suptitle('coefvar')
    fig.tight_layout()


    # skew
    fig, axes = plt.subplots(
        len(n_obs_s), len(prc_out_s), sharex=True, sharey='row', figsize=(9,6))
    for n_i, ax in enumerate(axes):
        for o_i, prc_out in enumerate(prc_out_s):
            ax = axes[n_i, o_i]
            ax.axhline(0, color='gray', lw=0.8, zorder=0)
            data = analytic_skew_s[o_i, :, n_i]
            if plot_multilines:
                ax.plot(
                    beta_t_s,
                    data[:,:multilines_max],
                    color='C0',
                    alpha=multilines_alpha
                )
            else:
                median = np.percentile(data, 50, axis=-1)
                q025 = np.percentile(data, 2.5, axis=-1)
                q975 = np.percentile(data, 97.5, axis=-1)
                ax.fill_between(beta_t_s, q025, q975, alpha=0.2)
                ax.plot(beta_t_s, median)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    for ax, prc_out in zip(axes[0,:], prc_out_s):
        ax.set_title('prc_out={}'.format(prc_out))
    for ax in axes[-1,:]:
        ax.set_xlabel(r'$\beta_t$')
    for ax, n_obs in zip(axes[:,0], n_obs_s):
        ax.set_ylabel(r'$n={}$'.format(n_obs))
    # fig.suptitle('skew')
    fig.tight_layout()
