"""Run LOOCV linear regression."""

import sys, os, time

import numpy as np
from scipy import linalg
from scipy import stats

from m1_problem import *


# ============================================================================


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


def calc_bootloo_tb_AB(ys, X_mat):
    rng = np.random.RandomState(seed=seed_boot)
    out = np.zeros((n_trial, n_boot_trial))
    inner_start_time = time.time()
    for boot_i in range(n_boot_trial):
        # progress print
        if boot_i % (n_boot_trial//10) == 0 and boot_i != 0:
            elapsed_time = time.time() - inner_start_time
            etr = (n_boot_trial - boot_i)*(elapsed_time/boot_i)
            etr_unit = 's'
            if etr >= 60:
                etr /= 60
                etr_unit = 'min'
                if etr >= 60:
                    etr /= 60
                    etr_unit = 'h'
            etr = int(np.ceil(etr))
            print(
                '{}/{}, etr: {} {}'
                .format(boot_i, n_boot_trial, etr, etr_unit),
                flush=True
            )
        # take subsample
        idxs = rng.choice(n_obs, size=n_obs, replace=True)
        idxs.sort()
        X_mat_i = X_mat[idxs]
        ys_i = ys[:,idxs]
        # model A
        bootloo_A = calc_loo_ti(ys_i, X_mat_i[:,:-1])
        # model B
        bootloo_B = calc_loo_ti(ys_i, X_mat_i)
        # sum dif
        out[:, boot_i] = (
            np.sum(bootloo_A, axis=-1) - np.sum(bootloo_B, axis=-1))
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
X_mat, ys, X_test, ys_test, _ = make_data(n_obs, beta_t, prc_out, sigma2_d)

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

# bootloo
print('bootloo', flush=True)
bootloo_tb = calc_bootloo_tb_AB(ys, X_mat)
print('done')

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
os.makedirs('res_1/{}/boot'.format(folder_name), exist_ok=True)
np.savez_compressed(
    'res_1/{}/boot/{}.npz'.format(folder_name, run_i_str),
    bootloo_tb=bootloo_tb,
    run_i=run_i,
    fixed_sigma2_m=fixed_sigma2_m,
)
