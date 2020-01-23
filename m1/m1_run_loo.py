"""Run LOOCV linear regression."""

import sys, os, time

import numpy as np
from scipy import linalg
from scipy import stats

from m1_problem import *


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
X_mat, _, ys, ys_test = make_data(n_obs, beta_t, prc_out, sigma2_d)


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
loo_ti_A = calc_loo_ti(ys, X_mat[:,:-1], fixed_sigma2_m)
# model B LOO
print('model B LOO', flush=True)
loo_ti_B = calc_loo_ti(ys, X_mat, fixed_sigma2_m)

# model A test
print('model A test', flush=True)
test_ti_A = calc_test_ti(
    ys, X_mat[:,:-1], ys_test, X_mat[:,:-1], fixed_sigma2_m)
# model B test
print('model B test', flush=True)
test_ti_B = calc_test_ti(ys, X_mat, ys_test, X_mat, fixed_sigma2_m)

print('done', flush=True)

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
    test_ti_A=test_ti_A,
    test_ti_B=test_ti_B,
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
    X_test = X_mat
    test_test_a = np.zeros(elpd_test_n)
    for t_i in range(elpd_test_n):
        test_test_a[t_i] = calc_logpdf_for_one_pred(
            X_mat[:,:-1],
            ys[t],
            X_test[i,:-1],
            ys_test[t_i,i]
        )
    test_test_a = np.mean(test_test_a)
    print('test_a:\n{}\n{}'.format(test_test_a, test_ti_A[t,i]))
    # not ok

    # test B
    X_test = X_mat
    test_test_b = np.zeros(elpd_test_n)
    for t_i in range(elpd_test_n):
        test_test_b[t_i] = calc_logpdf_for_one_pred(
            X_mat,
            ys[t],
            X_test[i],
            ys_test[t_i,i]
        )
    test_test_b = np.mean(test_test_b)
    print('test_b:\n{}\n{}'.format(test_test_b, test_ti_B[t,i]))
    # not ok
