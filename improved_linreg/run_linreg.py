
import sys, os, time

import numpy as np
from scipy import linalg, stats

from setup_linreg import *


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

n_trial = probl_run.n_trial
n_dim = probl_run.n_dim

# basic yobs
X_tid, y_ti = probl_run.make_data()
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
# diff
loo_ti = loo_ti_A - loo_ti_B
loo_t = np.sum(loo_ti, axis=-1)

# model A test
print('model A test', flush=True)
elpd_tl_A = probl_run.calc_elpd_tl(
    y_ti, X_tid[:,:,:-1], y_test_ti, X_test_tid[:,:,:-1])
elpd_t_A = np.mean(elpd_tl_A, axis=1)
# model B test
print('model B test', flush=True)
elpd_tl_B = probl_run.calc_elpd_tl(
    y_ti, X_tid, y_test_ti, X_test_tid)
elpd_t_B = np.mean(elpd_tl_B, axis=1)
# diff
elpd_t = elpd_t_A - elpd_t_B


# ============================================================================
# calibration + var estims
cal_limits = np.linspace(0, 1, cal_nbins+1)

# normal
print('norm approx', flush=True)
# variance
var_hat_n_t = loo_ti.shape[-1]*np.var(loo_ti, ddof=1, axis=-1)
# calibration
cdf_elpd = stats.norm.cdf(
    elpd_t,
    loc=loo_t,
    scale=np.sqrt(var_hat_n_t)
)
cal_counts_n = np.histogram(cdf_elpd, cal_limits)[0]

# bb
print('BB approx', flush=True)
bb_rng = np.random.RandomState(seed=bb_seed)
alpha = bb_rng.dirichlet(np.ones(n_obs), size=bb_n)
# bb_approx_tb = np.sum(alpha.T*loo_ti[..., None], axis=-2)
bb_approx_tb = np.einsum('bi,...ib->...b', alpha, loo_ti[..., None])
bb_approx_tb *= n_obs
# variance
var_hat_bb_t = np.var(bb_approx_tb, ddof=1, axis=-1)
# calibration
cdf_elpd = np.mean(bb_approx_tb < elpd_t[:,None], axis=-1)
cal_counts_bb = np.histogram(cdf_elpd, cal_limits)[0]

# improved approx
print('improved approx', flush=True)
impr_linreg_rng = np.random.RandomState(seed=impr_linreg_seed)
impr_approx_ts = np.tile(loo_t[:,None], (1, impr_linreg_n))
var_hat_impr_t = np.empty((n_trial,))
impr2_approx_ts = np.tile(loo_t[:,None], (1, impr_linreg_n))
for t_i in range(n_trial):
    beta_hat, s2_hat, R_inv = lin_reg_fit(y_ti[t_i], X_tid[t_i])
    analytic_err = AnalyticErr(
        X_tid[t_i],
        idx_a=np.arange(X_tid.shape[-1]-1),
        idx_b=np.arange(X_tid.shape[-1])
    )
    A_err, b_err, c_err = analytic_err.calc_params(
        beta_ma=beta_hat[[-1]],
        beta_mb=None,
        tau2=s2_hat,
        sigma_star=np.sqrt(s2_hat)
    )
    var_hat_impr_t[t_i] = analytic_err.calc_var(
        A_err, b_err, c_err, Sigma_d=s2_hat)
    for s_i in range(impr_linreg_n):
        s2_hat_2 = inv_chi2_rng(n_obs - n_dim, s2_hat, rng=impr_linreg_rng)
        beta_hat_2 = beta_hat + np.sqrt(s2_hat_2)*R_inv.dot(
            impr_linreg_rng.randn(n_dim))
        A_err_2, b_err_2, c_err_2 = analytic_err.calc_params(
            beta_ma=beta_hat_2[[-1]],
            beta_mb=None,
            tau2=s2_hat_2,
            sigma_star=np.sqrt(s2_hat_2)
        )
        eps_is = impr_linreg_rng.randn(n_obs)

        eps_is_1 = eps_is*np.sqrt(s2_hat)
        # err_hat_1 = np.einsum('ib,ij,jb->b', eps_is_1, A_err, eps_is_1)
        # err_hat_1 += eps_is_1.T.dot(b_err)
        # err_hat_1 += c_err
        err_hat_1 = (
            eps_is_1.T.dot(A_err).dot(eps_is_1)
            + eps_is_1.T.dot(b_err)
            + c_err
        )
        impr_approx_ts[t_i, s_i] -= err_hat_1

        eps_is_2 = eps_is*np.sqrt(s2_hat_2)
        err_hat_2 = (
            eps_is_2.T.dot(A_err_2).dot(eps_is_2)
            + eps_is_2.T.dot(b_err_2)
            + c_err_2
        )
        impr2_approx_ts[t_i, s_i] -= err_hat_2

var_hat_impr2_t = np.var(impr2_approx_ts, ddof=1, axis=-1)
# calibration
cdf_elpd = np.mean(impr_approx_ts < elpd_t[:,None], axis=-1)
cal_counts_impr = np.histogram(cdf_elpd, cal_limits)[0]
cdf_elpd = np.mean(impr2_approx_ts < elpd_t[:,None], axis=-1)
cal_counts_impr2 = np.histogram(cdf_elpd, cal_limits)[0]


# ============================================================================
# progress print
print('done')
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
    # loo_ti_A=loo_ti_A,
    # loo_ti_B=loo_ti_B,
    # elpd_t_A=elpd_t_A,
    # elpd_t_B=elpd_t_B,
    loo_t=loo_t,
    elpd_t=elpd_t,
    var_hat_n_t=var_hat_n_t,
    var_hat_bb_t=var_hat_bb_t,
    var_hat_impr_t=var_hat_impr_t,
    var_hat_impr2_t=var_hat_impr2_t,
    cal_limits=cal_limits,
    cal_counts_n=cal_counts_n,
    cal_counts_bb=cal_counts_bb,
    cal_counts_impr=cal_counts_impr,
    cal_counts_impr2=cal_counts_impr2,
)
