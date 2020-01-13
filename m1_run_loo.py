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


def calc_test_t(ys, X_mat, ys_test, X_test):
    _, n_dim_cur = X_mat.shape
    elpd_test_set_size, _ = X_test.shape
    # test set pred distr params working array
    mu_pred_test = np.empty((elpd_test_set_size,))
    sigma2_pred_test = np.empty((elpd_test_set_size,))
    # test set logpdf
    test_t = np.empty((n_trial,))
    # As we have fixed X for each trial,
    # reuse some calcs for the trial iterations.
    cho_test = linalg.cho_factor(X_mat.T.dot(X_mat).T, overwrite_a=True)
    xSx_p1_test = np.einsum(
        'td,dt->t',
        X_test,
        linalg.cho_solve(cho_test, X_test.T)
    )
    xSx_p1_test += 1.0
    # loop for each trial
    for t in range(n_trial):
        # test set pred params
        beta_hat = linalg.cho_solve(cho_test, X_mat.T.dot(ys[t]))
        X_test.dot(beta_hat, out=mu_pred_test)
        if fixed_sigma2_m:
            np.multiply(xSx_p1_test, sigma2_m, out=sigma2_pred_test)
        else:
            y_xm = X_mat.dot(beta_hat)
            y_xm -= ys[t]
            s2 = y_xm.dot(y_xm)
            s2 /= n_obs - n_dim_cur
            np.multiply(xSx_p1_test, s2, out=sigma2_pred_test)
        # calc logpdf for test
        if fixed_sigma2_m:
            test_logpdf = stats.norm.logpdf(
                ys_test,
                loc=mu_pred_test,
                scale=np.sqrt(sigma2_pred_test)
            )
            test_t[t] = n_obs*np.mean(test_logpdf)
        else:
            test_logpdf = stats.t.logpdf(
                ys_test,
                n_obs - n_dim_cur,
                loc=mu_pred_test,
                scale=np.sqrt(sigma2_pred_test)
            )
            test_t[t] = n_obs*np.mean(test_logpdf)
    return test_t


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

elpd_test_set_size, _ = X_test.shape


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

# model A LOO
print('model A LOO', flush=True)
loo_ti_A = calc_loo_ti(ys, X_mat[:,:-1])
# model B LOO
print('model B LOO', flush=True)
loo_ti_B = calc_loo_ti(ys, X_mat)

# model A test
print('model A test', flush=True)
test_t_A = calc_test_t(ys, X_mat[:,:-1], ys_test, X_test[:,:-1])
# model B test
print('model B test', flush=True)
test_t_B = calc_test_t(ys, X_mat, ys_test, X_test)

# progress print
time_per_1000 = (time.time() - outer_start_time) * 1000 / n_trial
time_per_1000_unit = 's'
if time_per_1000 >= 60:
    time_per_1000 /= 60
    time_per_1000_unit = 'min'
    if time_per_1000 >= 60:
        time_per_1000 /= 60
        time_per_1000_unit = 'h'
time_per_1000 = int(np.ceil(time_per_1000))
print(
    'time per 1000 trials: {} {}'
    .format(time_per_1000, time_per_1000_unit)
)

# save
os.makedirs('res_1/{}'.format(folder_name), exist_ok=True)
np.savez_compressed(
    'res_1/{}/{}.npz'.format(folder_name, run_i_str),
    loo_ti_A=loo_ti_A,
    loo_ti_B=loo_ti_B,
    test_t_A=test_t_A,
    test_t_B=test_t_B,
    run_i=run_i,
    fixed_sigma2_m=fixed_sigma2_m,
)


# double check calculations
if False:
    t = 3
    i = 7

    def calc_logpdf_for_one_pred(x_cur, y_cur, x_pred, y_pred):
        n_obs_cur, n_dim_cur = x_cur.shape
        v_cho = linalg.cho_factor(x_cur.T.dot(x_cur))
        beta_hat = linalg.cho_solve(v_cho, x_cur.T.dot(y_cur))
        if fixed_sigma2_m:
            s2 = sigma2_m
        else:
            y_xm = y_cur - x_cur.dot(beta_hat)
            s2 = y_xm.dot(y_xm)/(n_obs_cur - n_dim_cur)
        pred_mu = x_pred.dot(beta_hat)
        pred_sigma2 = (x_pred.dot(linalg.cho_solve(v_cho, x_pred)) + 1.0)*s2
        if fixed_sigma2_m:
            out = stats.norm.logpdf(
                y_pred, loc=pred_mu, scale=np.sqrt(pred_sigma2))
        else:
            out = stats.t.logpdf(
                y_pred,
                n_obs_cur - n_dim_cur,
                loc=pred_mu,
                scale=np.sqrt(pred_sigma2)
            )
        return out

    print('Double checking results (values should match)...')

    # LOO A
    test_loo_a = calc_logpdf_for_one_pred(
        np.delete(X_mat[:,:-1], i, axis=0),
        np.delete(ys[t], i, axis=0),
        X_mat[i,:-1],
        ys[t,i]
    )
    print('loo_a:\n{}\n{}'.format(test_loo_a, loo_ti_A[t,i]))
    # ok

    # LOO B
    test_loo_b = calc_logpdf_for_one_pred(
        np.delete(X_mat, i, axis=0),
        np.delete(ys[t], i, axis=0),
        X_mat[i],
        ys[t,i]
    )
    print('loo_b:\n{}\n{}'.format(test_loo_b, loo_ti_B[t,i]))
    # ok

    # test A
    test_test_a = np.zeros(elpd_test_set_size)
    for t_i in range(elpd_test_set_size):
        test_test_a[t_i] = calc_logpdf_for_one_pred(
            X_mat[:,:-1],
            ys[t],
            X_test[t_i,:-1],
            ys_test[t_i]
        )
    test_test_a = n_obs*np.mean(test_test_a)
    print('test_a:\n{}\n{}'.format(test_test_a, test_t_A[t]))
    # not ok

    # test B
    test_test_b = np.zeros(elpd_test_set_size)
    for t_i in range(elpd_test_set_size):
        test_test_b[t_i] = calc_logpdf_for_one_pred(
            X_mat,
            ys[t],
            X_test[t_i],
            ys_test[t_i]
        )
    test_test_b = n_obs*np.mean(test_test_b)
    print('test_b:\n{}\n{}'.format(test_test_b, test_t_B[t]))
    # not ok