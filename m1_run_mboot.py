"""Run LOOCV linear regression."""

import sys, os, time

import numpy as np
from scipy import linalg
from scipy import stats

from m1_problem import *


# ============================================================================


def sample_scaled_inv_chi2(df, scale, n_samp, rng):
    samp = rng.chisquare(df, size=n_samp)
    np.divide(scale*df, samp, out=samp)
    return samp


def sample_n_invcho(mu, cho_inv_Sigma, n_samp, rng):
    dim = mu.shape[0]
    z = rng.randn(dim, n_samp)
    samp = linalg.solve_triangular(cho_inv_Sigma, z).T
    samp += mu
    return samp


def calc_loo_ti(ys, X_mat):
    n_trial_cur = ys.shape[0]
    n_obs_cur, n_dim_cur = X_mat.shape
    # working arrays
    x_tilde = np.empty((n_obs_cur-1, n_dim_cur))
    y_tilde = np.empty((n_obs_cur-1,))
    # pred distr params
    mu_preds = np.empty((n_trial_cur, n_obs_cur))
    sigma2_preds = np.empty((n_trial_cur, n_obs_cur))
    # As we have fixed X for each trial,
    # reuse some calcs for the trial iterations.
    cho_s = []
    xSx_p1_s = []
    for i in range(n_obs_cur):
        x_i = X_mat[i]
        # x_tilde = np.delete(X_mat, i, axis=0)
        x_tilde[:i,:] = X_mat[:i,:]
        x_tilde[i:,:] = X_mat[i+1:,:]
        cho = linalg.cho_factor(x_tilde.T.dot(x_tilde).T, overwrite_a=True)
        xSx_p1 = x_i.dot(linalg.cho_solve(cho, x_i)) + 1.0
        cho_s.append(cho)
        xSx_p1_s.append(xSx_p1)
    # loop for each trial
    for t in range(n_trial_cur):
        # LOO params for each data point
        for i in range(n_obs_cur):
            x_i = X_mat[i]
            # x_tilde = np.delete(X_mat, i, axis=0)
            # y_tilde = np.delete(ys[t], i, axis=0)
            x_tilde[:i,:] = X_mat[:i,:]
            x_tilde[i:,:] = X_mat[i+1:,:]
            y_tilde[:i] = ys[t,:i]
            y_tilde[i:] = ys[t,i+1:]
            beta_hat = linalg.cho_solve(cho_s[i], x_tilde.T.dot(y_tilde))
            mu_preds[t, i] = x_i.dot(beta_hat)
            if fixed_sigma2_m:
                sigma2_preds[t, i] = xSx_p1_s[i]*sigma2_m
            else:
                y_xm = x_tilde.dot(beta_hat)
                y_xm -= y_tilde
                s2 = y_xm.dot(y_xm)
                s2 /= n_obs_cur - 1 - n_dim_cur
                sigma2_preds[t, i] = xSx_p1_s[i]*s2
    # calc logpdf for loos
    if fixed_sigma2_m:
        loo_ti = stats.norm.logpdf(
            ys, loc=mu_preds, scale=np.sqrt(sigma2_preds))
    else:
        loo_ti = stats.t.logpdf(
            ys,
            n_obs_cur - 1 - n_dim_cur,
            loc=mu_preds,
            scale=np.sqrt(sigma2_preds)
        )
    return loo_ti


def calc_mbootloo_tb_AB(ys, X_mat):
    rng = np.random.RandomState(seed=seed_mboot)
    out = np.zeros((n_trial, n_mboot_trial))
    X_a = X_mat[:,:-1]
    n_dim_a = n_dim - 1
    X_b = X_mat
    n_dim_b = n_dim
    # posterior stuff
    # A
    A_cho_v = linalg.cho_factor(X_a.T.dot(X_a).T, overwrite_a=True)
    A_vx = linalg.cho_solve(A_cho_v, X_a.T)
    A_xvx = X_a.dot(A_vx)
    # B
    B_cho_v = linalg.cho_factor(X_b.T.dot(X_b).T, overwrite_a=True)
    B_vx = linalg.cho_solve(B_cho_v, X_b.T)
    B_xvx = X_b.dot(B_vx)
    for t in range(n_trial):
        y_t = ys[t]
        # A
        A_beta_hat = A_vx.dot(y_t)
        if not fixed_sigma2_m:
            y_xm = y_t - A_xvx.dot(y_t)
            A_s2 = (y_xm.dot(y_xm))/(n_obs-n_dim_a)
        # B
        B_beta_hat = B_vx.dot(y_t)
        if not fixed_sigma2_m:
            y_xm = y_t - B_xvx.dot(y_t)
            B_s2 = (y_xm.dot(y_xm))/(n_obs-n_dim_b)
        # do boot
        for boot_i in range(n_mboot_trial):
            # A
            if not fixed_sigma2_m:
                A_s2_sampled = sample_scaled_inv_chi2(
                    n_obs-n_dim_a, A_s2, 1, rng)[0]
            else:
                A_s2_sampled = sigma2_m
            beta_sampled = sample_n_invcho(
                A_beta_hat,
                A_cho_v[0]/np.sqrt(A_s2_sampled),
                1,
                rng
            )[0]
            A_samp_y = X_a.dot(beta_sampled)
            A_samp_y += rng.randn(n_obs)*np.sqrt(A_s2_sampled)
            # B
            if not fixed_sigma2_m:
                B_s2_sampled = sample_scaled_inv_chi2(
                    n_obs-n_dim_b, B_s2, 1, rng)[0]
            else:
                B_s2_sampled = sigma2_m
            beta_sampled = sample_n_invcho(
                B_beta_hat,
                B_cho_v[0]/np.sqrt(B_s2_sampled),
                1,
                rng
            )[0]
            B_samp_y = X_b.dot(beta_sampled)
            B_samp_y += rng.randn(n_obs)*np.sqrt(B_s2_sampled)
            # calc loos
            bootloo_A = calc_loo_ti(A_samp_y[None,:], X_a)
            bootloo_B = calc_loo_ti(B_samp_y[None,:], X_b)
            # sum dif
            out[t, boot_i] = np.sum(bootloo_A) - np.sum(bootloo_B)
    return out


# ============================================================================

# parse cmd input for run id
if len(sys.argv) > 1:
    # get run_i
    run_i = int(sys.argv[1])
else:
    run_i = 0
if run_i < 0 or run_i >= n_runs:
    raise ValueError('invalid run_i, max is {}'.format(n_runs-1))
run_i_str = str(run_i).zfill(4)
# results save folder name in `res_1`
if len(sys.argv) > 2:
    # results save folder name in `res_1`
    folder_name = sys.argv[2]
else:
    folder_name = None
if folder_name == 'unfixed':
    fixed_sigma2_m = False
elif folder_name == 'fixed':
    fixed_sigma2_m = True
else:
    folder_name = 'unfixed'
    fixed_sigma2_m = False

# get params and data
n_obs, beta_t, prc_out, sigma2_d = run_i_to_params(run_i)
X_mat, ys, X_test, ys_test = make_data(run_i)

print('Run {}/{}'.format(run_i, n_runs-1))
print(
    'model sigma fixed: {}'.format(fixed_sigma2_m),
    'n_obs: {} / {}'.format(n_obs, n_obs_s),
    'beta_t: {} / {}'.format(beta_t, beta_t_s),
    'prc_out: {} / {}'.format(prc_out, prc_out_s),
    'sigma2_d: {} / {}'.format(sigma2_d, sigma2_d_s),
    sep='\n'
)

outer_start_time = time.time()

# mbootloo
print('mbootloo')
mbootloo_tb = calc_mbootloo_tb_AB(ys, X_mat)

# progress print
time_per = (
    (time.time() - outer_start_time)*100/n_mboot_trial*1000/n_trial)
time_per_unit = 's'
if time_per >= 60:
    time_per /= 60
    time_per_unit = 'min'
    if time_per >= 60:
        time_per /= 60
        time_per_unit = 'h'
time_per = int(np.ceil(time_per))
print(
    'time per 100 n_boot per 1000 trials: {} {}'
    .format(time_per, time_per_unit)
)

# save
os.makedirs('res_1/{}/mboot'.format(folder_name), exist_ok=True)
np.savez_compressed(
    'res_1/{}/mboot/{}.npz'.format(folder_name, run_i_str),
    mbootloo_tb=mbootloo_tb,
    run_i=run_i,
    fixed_sigma2_m=fixed_sigma2_m,
)
