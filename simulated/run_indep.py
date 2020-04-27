
import sys, os, time

import numpy as np
from scipy import linalg, stats

from setup_indep import *


# parse cmd input for run id
if len(sys.argv) > 1:
    # get run_i
    run_i = int(sys.argv[1])
else:
    run_i = 0
if run_i < 0 or run_i >= n_runs:
    raise ValueError('invalid run_i, max is {}'.format(n_runs-1))
run_i_str = str(run_i).zfill(4)
# run additional results into provided sub-folder
if len(sys.argv) > 2:
    # results save sub-folder name
    subfolder_name = sys.argv[2]
else:
    subfolder_name = ''
out_folder_name = res_folder_name + subfolder_name

# get params and data
n_obs, beta_t, out_dev, tau2 = run_i_to_params(run_i)

# make problem run instance
probl_run = ProblemRun(
    n_obs=n_obs,
    n_obs_max=n_obs_max,
    beta_t=beta_t,
    out_dev=out_dev,
    tau2=tau2
)
probl_args = probl_run.get_args()

# basic yobs
X_tid, y_ti = probl_run.make_data()
# yobs2 for independent predics
X2_tid, y2_ti = probl_run.make_data()
# yobs_test for target calculations
X_test_tid, y_test_ti = probl_run.make_data(probl_run.elpd_test_n)

print('Run {}/{}'.format(run_i, n_runs-1))
print('Results into folder {}'.format(out_folder_name))
print(
    'n_obs: {} / {}'.format(n_obs, n_obs_s),
    'beta_t: {} / {}'.format(beta_t, beta_t_s),
    'out_dev: {} / {}'.format(out_dev, out_dev_s),
    'tau2: {} / {}'.format(tau2, tau2_s),
    sep='\n'
)

# ============================================================================
# Run
outer_start_time = time.time()

# model A LOO
print('model A LOO', flush=True)
loo_ti_A = probl_run.calc_loo_ti(y_ti, X_tid[:,:,:-1])
# model B LOO
print('model B LOO', flush=True)
loo_ti_B = probl_run.calc_loo_ti(y_ti, X_tid)

# model A LOO indep
print('model A LOO indep', flush=True)
looindep_ti_A = probl_run.calc_loo_ti(
    y_ti, X_tid[:,:,:-1], y2_ti=y2_ti, X2_tid=X2_tid[:,:,:-1])
# model B LOO indep
print('model B LOO indep', flush=True)
looindep_ti_B = probl_run.calc_loo_ti(
    y_ti, X_tid, y2_ti=y2_ti, X2_tid=X2_tid)

# model A test
print('model A test', flush=True)
elpd_tl_A = probl_run.calc_elpd_tl(
    y_ti, X_tid[:,:,:-1], y_test_ti, X_test_tid[:,:,:-1])
# model B test
print('model B test', flush=True)
elpd_tl_B = probl_run.calc_elpd_tl(
    y_ti, X_tid, y_test_ti, X_test_tid)

# calc elpd_t
elpd_t_A = np.mean(elpd_tl_A, axis=1)
elpd_t_B = np.mean(elpd_tl_B, axis=1)

print('done', flush=True)

# precalculate some quantites

# calc BB approx var and plooneg
loo_ti = loo_ti_A - loo_ti_B
bb_mean, bb_var, bb_025, bb_500, bb_975, bb_plooneg = (
    calc_bb_mean_var_prctiles_plooneg(loo_ti))

# calc LOO-BB
# loo_tki = np.stack((loo_ti_A, loo_ti_B), axis=1)
# loobb_t_A = calc_loo_bb(loo_tki)[:,0]
loobb_t_A = calc_loo_bb_pair(loo_ti)

# calc P-BMA
# loo_tk = loo_tki.sum(axis=-1)
# pbma_t_A = calc_p_bma(loo_tk)[:,0]
pbma_t_A = calc_p_bma_pair(loo_ti.sum(axis=-1))

# calc target P-BMA with true elpd
# elpd_tk = np.stack((elpd_t_A, elpd_t_B), axis=1)
# pbma_target_t_A = calc_p_bma(elpd_tk)[:,0]
pbma_target_t_A = calc_p_bma_pair(elpd_t_A - elpd_t_B)

# progress print
time_per_1000 = (time.time() - outer_start_time) * 1000 / probl_run.n_trial
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

# ============================================================================
# save
os.makedirs(out_folder_name, exist_ok=True)
np.savez_compressed(
    '{}/{}.npz'.format(out_folder_name, run_i_str),
    run_i=run_i,
    probl_args=probl_args,
    loo_ti_A=loo_ti_A,
    loo_ti_B=loo_ti_B,
    looindep_ti_A=looindep_ti_A,
    looindep_ti_B=looindep_ti_B,
    elpd_t_A=elpd_t_A,
    elpd_t_B=elpd_t_B,
    bb_mean=bb_mean,
    bb_var=bb_var,
    bb_025=bb_025,
    bb_500=bb_500,
    bb_975=bb_975,
    bb_plooneg=bb_plooneg,
    loobb_t_A=loobb_t_A,
    pbma_t_A=pbma_t_A,
    pbma_target_t_A=pbma_target_t_A,
)

# ============================================================================
# double check calculations
if False:
    t = 3
    i = 7

    def calc_logpdf_for_one_pred(x_cur, y_cur, x_pred, y_pred):
        n_obs_cur, n_dim_cur = x_cur.shape
        v_cho = linalg.cho_factor(x_cur.T.dot(x_cur))
        beta_hat = linalg.cho_solve(v_cho, x_cur.T.dot(y_cur))
        if probl_run.tau2 is not None:
            s2 = tau2
        else:
            y_xm = y_cur - x_cur.dot(beta_hat)
            s2 = y_xm.dot(y_xm)/(n_obs_cur - n_dim_cur)
        pred_mu = x_pred.dot(beta_hat)
        pred_sigma2 = (x_pred.dot(linalg.cho_solve(v_cho, x_pred)) + 1.0)*s2
        if probl_run.tau2 is not None:
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
        np.delete(X_tid[t,:,:-1], i, axis=0),
        np.delete(y_ti[t], i, axis=0),
        X_tid[t,i,:-1],
        y_ti[t,i]
    )
    print('loo_a:\n{}\n{}'.format(test_loo_a, loo_ti_A[t,i]))
    # ok

    # LOO B
    test_loo_b = calc_logpdf_for_one_pred(
        np.delete(X_tid[t], i, axis=0),
        np.delete(y_ti[t], i, axis=0),
        X_tid[t,i],
        y_ti[t,i]
    )
    print('loo_b:\n{}\n{}'.format(test_loo_b, loo_ti_B[t,i]))
    # ok

    # LOO A
    test_looindep_a = calc_logpdf_for_one_pred(
        np.delete(X_tid[t,:,:-1], i, axis=0),
        np.delete(y_ti[t], i, axis=0),
        X2_tid[t,i,:-1],
        y2_ti[t,i]
    )
    print('looindep_a:\n{}\n{}'.format(test_looindep_a, looindep_ti_A[t,i]))
    # ok

    # LOO B
    test_looindep_b = calc_logpdf_for_one_pred(
        np.delete(X_tid[t], i, axis=0),
        np.delete(y_ti[t], i, axis=0),
        X2_tid[t,i],
        y2_ti[t,i]
    )
    print('looindep_b:\n{}\n{}'.format(test_looindep_b, looindep_ti_B[t,i]))
    # ok

    # test A
    test_test_a = np.zeros((elpd_test_n, n_obs))
    for t_i in range(elpd_test_n):
        for ii in range(n_obs):
            test_test_a[t_i,ii] = calc_logpdf_for_one_pred(
                X_tid[t,:,:-1],
                y_ti[t],
                X_test_tid[t_i,ii,:-1],
                y_test_ti[t_i,ii]
            )
    test_test_a = np.sum(np.mean(test_test_a, axis=0))
    print('test_a:\n{}\n{}'.format(test_test_a, test_elpd_t_A[t]))
    # ok

    # test B
    test_test_b = np.zeros((elpd_test_n, n_obs))
    for t_i in range(elpd_test_n):
        for ii in range(n_obs):
            test_test_b[t_i,ii] = calc_logpdf_for_one_pred(
                X_tid[t,:,:],
                y_ti[t],
                X_test_tid[t_i,ii,:],
                y_test_ti[t_i,ii]
            )
    test_test_b = np.sum(np.mean(test_test_b, axis=0))
    print('test_a:\n{}\n{}'.format(test_test_b, test_elpd_t_B[t]))
    # ok
