"""Analyse LOOCV results

First run results with m1_run.py

"""

import numpy as np
from scipy import linalg, stats

import matplotlib.pyplot as plt
import seaborn as sns

import sobol_seq

from m1_problem import (
    get_analytic_params,
    calc_analytic_mean,
    calc_analytic_var,
    calc_analytic_skew,
    calc_analytic_coefvar,
)


# ============================================================================
# Config

# fixed model sigma2_m value
sigma2_m = 1.0
# outlier loc deviation
outlier_dev = 20.0
# dimensionality of beta
n_dim = 3
# first covariate as intercept
intercept = True
# other covariates' effects
beta_other = 1.0
# intercept coef (if applied)
beta_intercept = 0.0

# shuffle outlier observations
shuffle_mu_d = True
# shuffle seed
# np.random.RandomState().randint(np.iinfo(np.uint32).max)
shuffle_seed = 3911746089


# ============================================================================


def determine_n_obs_out_p_m(n_obs, prc_out, outliers_style):
    """Determine outliers numbers.

    Parameters
    ----------
    outliers_style : str ['even', 'pos', 'neg']
        style of outliers (if `prc_out > 0`)

    """
    if prc_out > 0.0:
        if outliers_style == 'even':
            n_obs_out = max(int(np.round(prc_out*n_obs/2)), 1)
            n_obs_out_p = n_obs_out
            n_obs_out_m = n_obs_out
        elif outliers_style == 'pos':
            n_obs_out = max(int(np.round(prc_out*n_obs)), 1)
            n_obs_out_p = n_obs_out
            n_obs_out_m = 0
        elif outliers_style == 'neg':
            n_obs_out = max(int(np.round(prc_out*n_obs)), 1)
            n_obs_out_p = 0
            n_obs_out_m = n_obs_out
        else:
            raise ValueError('invalid arg `outliers_style`')
    else:
        n_obs_out = 0
        n_obs_out_p = 0
        n_obs_out_m = 0
    return n_obs_out_p, n_obs_out_m


def make_x(n_obs):
    if intercept:
        # firs dim (column) ones for intercept
        X_mat = np.hstack((
            np.ones((n_obs, 1)),
            sobol_seq.i4_sobol_generate_std_normal(n_dim-1, n_obs)
        ))
    else:
        X_mat = sobol_seq.i4_sobol_generate_std_normal(n_dim, n_obs)
    return X_mat


def make_mu_d(n_obs, n_obs_out_p, n_obs_out_m, sigma2_d, beta_t):
    beta = np.array([beta_other]*(n_dim-1)+[beta_t])
    if intercept:
        beta[0] = beta_intercept
    outlier_dev_eps = outlier_dev*np.sqrt(sigma2_d + np.sum(beta**2))
    n_obs_out = n_obs_out_p + n_obs_out_m
    mu_d = np.zeros(n_obs)
    mu_d[n_obs-n_obs_out:n_obs-n_obs_out+n_obs_out_p] = outlier_dev_eps
    mu_d[n_obs-n_obs_out+n_obs_out_p:] = -outlier_dev_eps
    if shuffle_mu_d:
        rng = np.random.RandomState(seed=shuffle_seed)
        rng.shuffle(mu_d)
    return mu_d


# ============================================================================
# As a function of n

# variables
n_obs_s = np.concatenate((
    np.arange(8,64),
    2**np.arange(7,10)
))
# (prc_out, outliers_style)
out_confs = [(0.0, 'none'), (0.04, 'even'), (0.04, 'pos'), (0.04, 'neg')]

# constants
beta_t = 1.0
sigma2_d = 1.0


analytic_mean_s = np.full((len(out_confs), len(n_obs_s)), np.nan)
analytic_var_s = np.full((len(out_confs), len(n_obs_s)), np.nan)
analytic_skew_s = np.full((len(out_confs), len(n_obs_s)), np.nan)
analytic_coefvar_s = np.full((len(out_confs), len(n_obs_s)), np.nan)
for i1, n_obs in enumerate(n_obs_s):
    for i, (prc_out, outliers_style) in enumerate(out_confs):
        n_obs_out_p, n_obs_out_m = determine_n_obs_out_p_m(
            n_obs, prc_out, outliers_style)
        n_obs_out = n_obs_out_p + n_obs_out_m
        prc_out_fixed = n_obs_out/n_obs
        X_mat = make_x(n_obs)
        mu_d = make_mu_d(n_obs, n_obs_out_p, n_obs_out_m, sigma2_d, beta_t)
        if prc_out_fixed == 0.0:
            mu_d = None
        A_mat, b_vec, c_sca = get_analytic_params(X_mat, beta_t)
        analytic_mean_s[i2, i1] = calc_analytic_mean(
            A_mat, b_vec, c_sca, sigma2_d, mu_d)
        analytic_var_s[i2, i1] = calc_analytic_var(
            A_mat, b_vec, c_sca, sigma2_d, mu_d)
        analytic_skew_s[i2, i1] = calc_analytic_skew(
            A_mat, b_vec, c_sca, sigma2_d, mu_d)
        analytic_coefvar_s[i2, i1] = calc_analytic_coefvar(
            A_mat, b_vec, c_sca, sigma2_d, mu_d)

# plots
do_subplots = False

# mean
if do_subplots:
    fig, axes = plt.subplots(len(out_confs), 1, sharex=True, sharey=True, figsize=(16,8))
    for i, ax in enumerate(axes):
        ax.plot(n_obs_s, analytic_mean_s[i])
        ax.set_ylabel('mean')
        ax.set_xlabel('n_obs')
        ax.set_title(out_confs[i][1])
    fig.tight_layout()
else:
    fig = plt.figure()
    for i in range(len(out_confs)):
        plt.plot(n_obs_s, analytic_mean_s[i])
    plt.ylabel('mean')
    plt.xlabel('n_obs')
    plt.legend([out_conf[1] for out_conf in out_confs])
    fig.tight_layout()

# coefvar
if do_subplots:
    fig, axes = plt.subplots(len(out_confs), 1, sharex=True, sharey=True, figsize=(16,8))
    for i, ax in enumerate(axes):
        ax.plot(n_obs_s, analytic_coefvar_s[i])
        ax.set_ylabel('coef var')
        ax.set_xlabel('n_obs')
        ax.set_title(out_confs[i][1])
    fig.tight_layout()
else:
    fig = plt.figure()
    for i in range(len(out_confs)):
        plt.plot(n_obs_s, analytic_coefvar_s[i])
    plt.ylabel('coef var')
    plt.xlabel('n_obs')
    plt.legend([out_conf[1] for out_conf in out_confs])
    fig.tight_layout()

# skew
if do_subplots:
    fig, axes = plt.subplots(len(out_confs), 1, sharex=True, sharey=True, figsize=(16,8))
    for i, ax in enumerate(axes):
        ax.plot(n_obs_s, analytic_skew_s[i])
        ax.set_ylabel('skew')
        ax.set_xlabel('n_obs')
        ax.set_title(out_confs[i][1])
    fig.tight_layout()
else:
    fig = plt.figure()
    for i in range(len(out_confs)):
        plt.plot(n_obs_s, analytic_skew_s[i])
    plt.ylabel('skew')
    plt.xlabel('n_obs')
    plt.legend([out_conf[1] for out_conf in out_confs])
    fig.tight_layout()



# ============================================================================
# As a function of beta_t

# variables
# beta_t_s = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 200.0]
beta_t_s = np.linspace(0.0, 20.0, 60)
# beta_t_s = np.linspace(1.0, 1.5, 20)
# (prc_out, outliers_style)
out_confs = [(0.0, 'none'), (2/32, 'even'), (2/32, 'pos'), (2/32, 'neg')]

# constants
n_obs = 32
sigma2_d = 1.0


analytic_mean_s = np.full((len(out_confs), len(beta_t_s)), np.nan)
analytic_var_s = np.full((len(out_confs), len(beta_t_s)), np.nan)
analytic_skew_s = np.full((len(out_confs), len(beta_t_s)), np.nan)
analytic_coefvar_s = np.full((len(out_confs), len(beta_t_s)), np.nan)
for i1, beta_t in enumerate(beta_t_s):
    for i2, (prc_out, outliers_style) in enumerate(out_confs):
        n_obs_out_p, n_obs_out_m = determine_n_obs_out_p_m(
            n_obs, prc_out, outliers_style)
        n_obs_out = n_obs_out_p + n_obs_out_m
        prc_out_fixed = n_obs_out/n_obs
        X_mat = make_x(n_obs)
        mu_d = make_mu_d(n_obs, n_obs_out_p, n_obs_out_m, sigma2_d, beta_t)
        if prc_out_fixed == 0.0:
            mu_d = None
        A_mat, b_vec, c_sca = get_analytic_params(X_mat, beta_t)
        analytic_mean_s[i2, i1] = calc_analytic_mean(
            A_mat, b_vec, c_sca, sigma2_d, mu_d)
        analytic_var_s[i2, i1] = calc_analytic_var(
            A_mat, b_vec, c_sca, sigma2_d, mu_d)
        analytic_skew_s[i2, i1] = calc_analytic_skew(
            A_mat, b_vec, c_sca, sigma2_d, mu_d)
        analytic_coefvar_s[i2, i1] = calc_analytic_coefvar(
            A_mat, b_vec, c_sca, sigma2_d, mu_d)

# plots
do_subplots = False

# mean
if do_subplots:
    fig, axes = plt.subplots(len(out_confs), 1, sharex=True, sharey=True, figsize=(16,8))
    for i, ax in enumerate(axes):
        ax.plot(beta_t_s, analytic_mean_s[i])
        ax.set_ylabel('mean')
        ax.set_xlabel('beta_t')
        ax.set_title(out_confs[i][1])
    fig.tight_layout()
else:
    fig = plt.figure()
    for i in range(len(out_confs)):
        plt.plot(beta_t_s, analytic_mean_s[i])
    plt.ylabel('mean')
    plt.xlabel('beta_t')
    plt.legend([out_conf[1] for out_conf in out_confs])
    fig.tight_layout()

# coefvar
if do_subplots:
    fig, axes = plt.subplots(len(out_confs), 1, sharex=True, sharey=True, figsize=(16,8))
    for i, ax in enumerate(axes):
        ax.plot(beta_t_s, analytic_coefvar_s[i])
        ax.set_ylabel('coef var')
        ax.set_xlabel('beta_t')
        ax.set_title(out_confs[i][1])
    fig.tight_layout()
else:
    fig = plt.figure()
    for i in range(len(out_confs)):
        plt.plot(beta_t_s, analytic_coefvar_s[i])
    plt.ylabel('coef var')
    plt.xlabel('beta_t')
    plt.legend([out_conf[1] for out_conf in out_confs])
    fig.tight_layout()

# skew
if do_subplots:
    fig, axes = plt.subplots(len(out_confs), 1, sharex=True, sharey=True, figsize=(16,8))
    for i, ax in enumerate(axes):
        ax.plot(beta_t_s, analytic_skew_s[i])
        ax.set_ylabel('skew')
        ax.set_xlabel('beta_t')
        ax.set_title(out_confs[i][1])
    fig.tight_layout()
else:
    fig = plt.figure()
    for i in range(len(out_confs)):
        plt.plot(beta_t_s, analytic_skew_s[i])
    plt.ylabel('skew')
    plt.xlabel('beta_t')
    plt.legend([out_conf[1] for out_conf in out_confs])
    fig.tight_layout()



# ============================================================================
# As a function of sigma2_d

# variables

sigma2_d_s = [0.01, 0.1, 1.0, 10.0, 100.0]

# sigma2_d_s = np.concatenate((
#     np.linspace(0.01, 8.0, 40),
#     np.linspace(3.1, 3.8, 20),
# ))
# sigma2_d_s.sort()

# (prc_out, outliers_style)
out_confs = [(0.0, 'none'), (2/32, 'even'), (2/32, 'pos'), (2/32, 'neg')]

# constants
n_obs = 32
beta_t = 1.0


analytic_mean_s = np.full((len(out_confs), len(sigma2_d_s)), np.nan)
analytic_var_s = np.full((len(out_confs), len(sigma2_d_s)), np.nan)
analytic_skew_s = np.full((len(out_confs), len(sigma2_d_s)), np.nan)
analytic_coefvar_s = np.full((len(out_confs), len(sigma2_d_s)), np.nan)
for i1, sigma2_d in enumerate(sigma2_d_s):
    for i2, (prc_out, outliers_style) in enumerate(out_confs):
        n_obs_out_p, n_obs_out_m = determine_n_obs_out_p_m(
            n_obs, prc_out, outliers_style)
        n_obs_out = n_obs_out_p + n_obs_out_m
        prc_out_fixed = n_obs_out/n_obs
        X_mat = make_x(n_obs)
        mu_d = make_mu_d(n_obs, n_obs_out_p, n_obs_out_m, sigma2_d, beta_t)
        if prc_out_fixed == 0.0:
            mu_d = None
        A_mat, b_vec, c_sca = get_analytic_params(X_mat, beta_t)
        analytic_mean_s[i2, i1] = calc_analytic_mean(
            A_mat, b_vec, c_sca, sigma2_d, mu_d)
        analytic_var_s[i2, i1] = calc_analytic_var(
            A_mat, b_vec, c_sca, sigma2_d, mu_d)
        analytic_skew_s[i2, i1] = calc_analytic_skew(
            A_mat, b_vec, c_sca, sigma2_d, mu_d)
        analytic_coefvar_s[i2, i1] = calc_analytic_coefvar(
            A_mat, b_vec, c_sca, sigma2_d, mu_d)

# plots
do_subplots = False

# mean
if do_subplots:
    fig, axes = plt.subplots(len(out_confs), 1, sharex=True, sharey=True, figsize=(16,8))
    for i, ax in enumerate(axes):
        ax.plot(sigma2_d_s, analytic_mean_s[i])
        ax.set_ylabel('mean')
        ax.set_xlabel('sigma2_d')
        ax.set_title(out_confs[i][1])
    fig.tight_layout()
else:
    fig = plt.figure()
    for i in range(len(out_confs)):
        plt.plot(sigma2_d_s, analytic_mean_s[i])
    plt.ylabel('mean')
    plt.xlabel('sigma2_d')
    plt.legend([out_conf[1] for out_conf in out_confs])
    fig.tight_layout()

# coefvar
if do_subplots:
    fig, axes = plt.subplots(len(out_confs), 1, sharex=True, sharey=True, figsize=(16,8))
    for i, ax in enumerate(axes):
        ax.plot(sigma2_d_s, analytic_coefvar_s[i])
        ax.set_ylabel('coef var')
        ax.set_xlabel('sigma2_d')
        ax.set_title(out_confs[i][1])
    fig.tight_layout()
else:
    fig = plt.figure()
    for i in range(len(out_confs)):
        plt.plot(sigma2_d_s, analytic_coefvar_s[i])
    plt.ylabel('coef var')
    plt.xlabel('sigma2_d')
    plt.legend([out_conf[1] for out_conf in out_confs])
    fig.tight_layout()

# skew
if do_subplots:
    fig, axes = plt.subplots(len(out_confs), 1, sharex=True, sharey=True, figsize=(16,8))
    for i, ax in enumerate(axes):
        ax.plot(sigma2_d_s, analytic_skew_s[i])
        ax.set_ylabel('skew')
        ax.set_xlabel('sigma2_d')
        ax.set_title(out_confs[i][1])
    fig.tight_layout()
else:
    fig = plt.figure()
    for i in range(len(out_confs)):
        plt.plot(sigma2_d_s, analytic_skew_s[i])
    plt.ylabel('skew')
    plt.xlabel('sigma2_d')
    plt.legend([out_conf[1] for out_conf in out_confs])
    fig.tight_layout()
