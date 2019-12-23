"""Run LOOCV linear regression.

data:
y = beta'*x + eps
x = norm_inv_cdf(unif_rng(0, 1))
eps = sqrt(sigma2_d) * (
    norm_rng(0,1) if nonoutlier else norm_rng(+-outlier_dev,1))

run:
python m1_run.py [run_i]

"""

import sys, os

import numpy as np
from scipy import linalg
from scipy import stats


# ===========================================================================
# conf (remember to save all confs in the result file)

# random seed for data
seed = 11

# fixed model sigma
fixed_sigma2_m = True
# fixed model sigma2_m value
sigma2_m = 1.0
# epsilon sigma2_d
sigma2_d = 10.0**(2*np.arange(-1, 2))
# number of obs in one trial
n_obs = [16, 32, 64, 128]
# percentage of outliers
prc_out = [0.01, 0.05, 0.1]
# outlier loc deviation
outlier_dev = 10.0
# suffle observations
suffle_obs = False
# number of trials
n_trial = 13
# dimensionality of true beta
n_dim = 3
# first covariate as intercept
intercept = True
# last covariate effect not used in model A
beta_t = [0.1, 1.0, 10.0]
# other covariates' effects
beta_other = 1.0
# intercept coef (if applied)
beta_intercept = 0.0


# ===========================================================================
# parse cmd input for run id
if len(sys.argv) > 1:
    # get run_i
    run_i = int(sys.argv[1])
else:
    run_i = 0
run_i_str = str(run_i).zfill(4)

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

# set params for this run
n_obs_i = n_obs_s[run_i]
beta_t_i = beta_t_s[run_i]
prc_out_i = prc_out_s[run_i]
sigma2_d_i = sigma2_d_s[run_i]


# ===========================================================================
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
n_obs_out = max(int(np.ceil(prc_out_i*n_obs_i)), 2)
n_obs_out_m = n_obs_out // 2
n_obs_out_p = n_obs_out - n_obs_out_m
eps_in = eps_in_last[:,:n_obs_i-n_obs_out].copy()
eps_out_p = eps_out_p_last[:,:n_obs_out_p].copy()
eps_out_m = eps_out_m_last[:,:n_obs_out_m].copy()
eps = np.concatenate((eps_in, eps_out_p, eps_out_m), axis=1)
eps *= np.sqrt(sigma2_d_i)
if suffle_obs:
    rng.shuffle(eps.T)

# drop unnecessary data
del(X_last, eps_in_last, eps_out_p_last, eps_out_m_last)

# calc y
ys = X_mat.dot(beta) + eps


def calc_loo_ti(ys, X_mat):

    n_dim_cur = X_mat.shape[-1]

    if not fixed_sigma2_m:
        pred_tdf = n_obs_i - 1 - n_dim_cur

    # natural parameters
    # pointwise products for each trial and data point
    Q_i = X_mat[:,:,None]*X_mat[:,None,:]
    r_ti = X_mat[None,:,:]*ys[:,:,None]
    # full posterior for each trial
    Q_full = Q_i.sum(axis=0)
    r_t_full = r_ti.sum(axis=1)

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
    for t in range(n_trial):
        if t % (n_trial//10) == 0:
            print('{}/{}'.format(t, n_trial), flush=True)
        # for each data point
        for i in range(n_obs_i):
            Q_i = np.subtract(Q_full, Q_i[i], out=temp_mat)
            r_i = np.subtract(r_t_full[t], r_ti[t, i], out=temp_vec)
            x_i = X_mat[i]
            # x_tilde = np.delete(X_mat, i, axis=0)
            # y_tilde = np.delete(ys[t], i, axis=0)
            x_tilde[:i,:] = X_mat[:i,:]
            x_tilde[i:,:] = X_mat[i+1:,:]
            y_tilde[:i] = ys[t,:i]
            y_tilde[i:] = ys[t,i+1:]
            cho = linalg.cho_factor(Q_i, overwrite_a=True)
            xS = linalg.cho_solve(cho, x_i)
            rS = linalg.cho_solve(cho, r_i)
            mu_preds[t, i] = rS.dot(x_i)
            if fixed_sigma2_m:
                sigma2_preds[t, i] = (xS.dot(x_i) + 1)*sigma2_m
            else:
                y_xm = x_tilde.dot(rS, out=temp_loo_vec)
                y_xm -= y_tilde
                s2 = y_xm.dot(y_xm)
                s2 /= pred_tdf
                sigma2_preds[t, i] = (xS.dot(x_i) + 1)*s2
    print('done', flush=True)

    # calc logpdf for pred distributions
    if fixed_sigma2_m:
        loo_ti = stats.norm.logpdf(
            ys, loc=mu_preds, scale=np.sqrt(sigma2_preds))
    else:
        loo_ti = stats.t.logpdf(
            ys, pred_tdf, loc=mu_preds, scale=np.sqrt(sigma2_preds))

    return loo_ti

# model A
print('model A')
loo_ti_A = calc_loo_ti(ys, X_mat[:,:-1])
# model B
print('model B')
loo_ti_B = calc_loo_ti(ys, X_mat)

# save
os.makedirs('res_1', exist_ok=True)
np.savez_compressed(
    'res_1/{}.npz'.format(run_i_str),
    loo_ti_A=loo_ti_A,
    loo_ti_B=loo_ti_B,
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
)
