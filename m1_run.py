"""Run LOOCV linear regression.

data:
y = beta'*x + eps
x = norm_inv_cdf(unif_rng(0, 1))
eps = sqrt(sigma2_d) * (
    norm_rng(0,1) if nonoutlier else norm_rng(+-outlier_dev,1))

run:
python m1_run.py [<run_i> [fixed/unfixed]]

"""

import sys, os, time

import numpy as np
from scipy import linalg
from scipy import stats


# ===========================================================================
# conf (remember to save all confs in the result file)



# random seed for data
seed = 11

# fixed model sigma2_m value
sigma2_m = 1.0
# epsilon sigma2_d
sigma2_d = 10.0**(2*np.arange(-1, 3))
# number of obs in one trial
n_obs = [16, 32, 64, 128, 256, 512]
# percentage of outliers
prc_out = [0.0, 0.01, 0.05, 0.1]
# outlier loc deviation
outlier_dev = 10.0
# suffle observations
suffle_obs = False
# number of trials
n_trial = 200
# dimensionality of true beta
n_dim = 3
# first covariate as intercept
intercept = True
# last covariate effect not used in model A
beta_t = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
# other covariates' effects
beta_other = 1.0
# intercept coef (if applied)
beta_intercept = 0.0
# independent test set for true elpd
elpd_test_set_size = 10000
# outliers in the independent test set for true elpd
elpd_test_outliers = True


# ===========================================================================

# set grid
n_obs_s, beta_t_s, prc_out_s, sigma2_d_s = np.meshgrid(
    n_obs,
    beta_t,
    prc_out,
    sigma2_d
)
n_obs_s = n_obs_s.ravel()
beta_t_s = beta_t_s.ravel()
prc_out_s = prc_out_s.ravel()
sigma2_d_s = sigma2_d_s.ravel()
n_runs = len(n_obs_s)

# parse cmd input for run id
if len(sys.argv) > 1:
    # get run_i
    run_i = int(sys.argv[1])
else:
    run_i = 0
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

# set params for this run
n_obs_i = n_obs_s[run_i]
beta_t_i = beta_t_s[run_i]
prc_out_i = prc_out_s[run_i]
sigma2_d_i = sigma2_d_s[run_i]

print('Run {}/{}'.format(run_i, n_runs-1))
print(
    'model sigma fixed: {}'.format(fixed_sigma2_m),
    'n_obs_i: {} / {}'.format(n_obs_i, n_obs),
    'beta_t_i: {} / {}'.format(beta_t_i, beta_t),
    'prc_out_i: {} / {}'.format(prc_out_i, prc_out),
    'sigma2_d_i: {} / {}'.format(sigma2_d_i, sigma2_d),
    sep='\n'
)

# beta vector
beta = np.array([beta_other]*(n_dim-1)+[beta_t_i])
if intercept:
    beta[0] = beta_intercept

# random number generator
rng = np.random.RandomState(seed=seed)

# data for all cases
X_last = stats.norm.ppf(
    rng.uniform(size=(n_obs[-1], n_dim)))
if intercept:
    # firs dim ones for intercept
    X_last[:,0] = 1.0
# data for the biggest sample size
n_obs_out_last = max(int(np.ceil(prc_out[-1]*n_obs[-1])), 2)
n_obs_out_m_last = n_obs_out_last // 2
n_obs_out_p_last = n_obs_out_last - n_obs_out_m_last
eps_in_last = rng.normal(size=(n_trial, n_obs[-1]))
eps_out_p_last = rng.normal(
    loc=outlier_dev, size=(n_trial, n_obs_out_p_last))
eps_out_m_last = rng.normal(
    loc=-outlier_dev, size=(n_trial, n_obs_out_m_last))
# take current size observations (copy in order to get contiguous)
X_mat = X_last[:n_obs_i,:].copy()
if prc_out_i > 0.0:
    n_obs_out = max(int(np.ceil(prc_out_i*n_obs_i)), 2)
    n_obs_out_m = n_obs_out // 2
    n_obs_out_p = n_obs_out - n_obs_out_m
    eps_in = eps_in_last[:,:n_obs_i-n_obs_out]
    eps_out_p = eps_out_p_last[:,:n_obs_out_p]
    eps_out_m = eps_out_m_last[:,:n_obs_out_m]
    eps = np.concatenate((eps_in, eps_out_p, eps_out_m), axis=1)
else:
    n_obs_out = 0
    n_obs_out_m = 0
    n_obs_out_p = 0
    eps = eps_in_last[:,:n_obs_i].copy()
eps *= np.sqrt(sigma2_d_i)
if suffle_obs:
    rng.shuffle(eps.T)
# drop unnecessary data
del(X_last, eps_in_last, eps_out_p_last, eps_out_m_last)
# calc y
ys = X_mat.dot(beta) + eps

# elpd test set
X_test = stats.norm.ppf(
    rng.uniform(size=(elpd_test_set_size, n_dim)))
if intercept:
    # firs dim ones for intercept
    X_test[:,0] = 1.0
if elpd_test_outliers and prc_out_i > 0.0:
    n_obs_out_test = max(int(np.ceil(prc_out_i*elpd_test_set_size)), 2)
    n_obs_out_test_m = n_obs_out_test // 2
    n_obs_out_test_p = n_obs_out_test - n_obs_out_test_m
    eps_test_in = rng.normal(
        size=(elpd_test_set_size-n_obs_out_test,))
    eps_test_out_p = rng.normal(
        loc=outlier_dev, size=(n_obs_out_test_p,))
    eps_test_out_m = rng.normal(
        loc=outlier_dev, size=(n_obs_out_test_m,))
    eps_test = np.concatenate(
        (eps_test_in, eps_test_out_p, eps_test_out_m), axis=0)
    del(eps_test_in, eps_test_out_p, eps_test_out_m)
else:
    n_obs_out_test = 0
    n_obs_out_test_m = 0
    n_obs_out_test_p = 0
    eps_test = rng.normal(size=(elpd_test_set_size,))
eps_test *= np.sqrt(sigma2_d_i)
if suffle_obs:
    rng.shuffle(eps_test.T)
# calc y
ys_test = X_test.dot(beta) + eps_test


def calc_loo_ti(ys, X_mat):
    n_dim_cur = X_mat.shape[-1]
    if not fixed_sigma2_m:
        pred_tdf = n_obs_i - 1 - n_dim_cur
    # working arrays
    temp_mat = np.empty((n_dim_cur, n_dim_cur))
    temp_vec = np.empty((n_dim_cur,))
    if not fixed_sigma2_m:
        temp_loo_vec = np.empty((n_obs_i-1,))
    x_tilde = np.empty((n_obs_i-1, n_dim_cur))
    y_tilde = np.empty((n_obs_i-1,))
    # calc pred distr params
    mu_preds = np.empty((n_trial, n_obs_i))
    sigma2_preds = np.empty((n_trial, n_obs_i))
    # for each trial
    inner_start_time = time.time()
    for t in range(n_trial):
        if t % (n_trial//10) == 0 and t != 0:
            elapsed_time = time.time() - inner_start_time
            etr = (n_trial - t)*(elapsed_time/t)
            etr_unit = 's'
            if etr >= 60:
                etr /= 60
                etr_unit = 'min'
                if etr >= 60:
                    etr /= 60
                    etr_unit = 'h'
            etr = int(np.ceil(etr))
            print(
                '{}/{}, etr: {} {}'.format(t, n_trial, etr, etr_unit),
                flush=True
            )
        # for each data point
        for i in range(n_obs_i):
            x_i = X_mat[i]
            # x_tilde = np.delete(X_mat, i, axis=0)
            # y_tilde = np.delete(ys[t], i, axis=0)
            x_tilde[:i,:] = X_mat[:i,:]
            x_tilde[i:,:] = X_mat[i+1:,:]
            y_tilde[:i] = ys[t,:i]
            y_tilde[i:] = ys[t,i+1:]
            Q_i = x_tilde.T.dot(x_tilde, out=temp_mat)
            r_i = x_tilde.T.dot(y_tilde, out=temp_vec)
            cho = linalg.cho_factor(Q_i, overwrite_a=True)
            xSx = linalg.cho_solve(cho, x_i).dot(x_i)
            rS = linalg.cho_solve(cho, r_i)
            mu_preds[t, i] = rS.dot(x_i)
            if fixed_sigma2_m:
                sigma2_preds[t, i] = (xSx + 1)*sigma2_m
            else:
                y_xm = x_tilde.dot(rS, out=temp_loo_vec)
                y_xm -= y_tilde
                s2 = y_xm.dot(y_xm)
                s2 /= pred_tdf
                sigma2_preds[t, i] = (xSx + 1)*s2
    print('done', flush=True)
    # calc logpdf for pred distributions
    if fixed_sigma2_m:
        loo_ti = stats.norm.logpdf(
            ys, loc=mu_preds, scale=np.sqrt(sigma2_preds))
    else:
        loo_ti = stats.t.logpdf(
            ys, pred_tdf, loc=mu_preds, scale=np.sqrt(sigma2_preds))
    # calc test elpd
    TODO
    if fixed_sigma2_m:
        ...
        elpd_test =
    else:
        ...
        elpd_test =
    return loo_ti, elpd_test

outer_start_time = time.time()

# model A
print('model A')
loo_ti_A, elpd_test_A = calc_loo_ti(ys, X_mat[:,:-1])
# model B
print('model B')
loo_ti_B, elpd_test_B = calc_loo_ti(ys, X_mat)

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
    elpd_test_A=elpd_test_A,
    elpd_test_B=elpd_test_B,
    run_i=run_i,
    seed=seed,
    fixed_sigma2_m=fixed_sigma2_m,
    sigma2_m=sigma2_m,
    sigma2_d=sigma2_d,
    n_obs=n_obs,
    prc_out=prc_out,
    outlier_dev=outlier_dev,
    suffle_obs=suffle_obs,
    n_trial=n_trial,
    n_dim=n_dim,
    intercept=intercept,
    beta_t=beta_t,
    beta_other=beta_other,
    beta_intercept=beta_intercept,
    elpd_test_set_size=elpd_test_set_size,
    elpd_test_outliers=elpd_test_outliers,
)
