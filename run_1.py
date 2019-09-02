"""Run LOOCV.

run:
python run_1.py [run_id]


"""

import sys, os

import numpy as np
from scipy import linalg
from scipy import stats


# ===========================================================================
# conf (remember to save all confs in the result file)

# random seed
seed = 11

# number of obs in one trial
n_obs = 10
# number of trials
n_trial = 50

# number of true coefficients, >= 1 (inc. possible intercept)
n_coef = 3
# first covariate as intercept
intercept = True
# last covariate effect not used in model A
beta_t = 0.1
# other covariate effects
beta_other = 1.0
# intercept coef (if applied)
beta_intercept = 0.0

# data epsilon degres of freedom
t_df_data = 4


# ===========================================================================
# parse cmd input for run id
if len(sys.argv) > 1:
    run_id = sys.argv[1].zfill(4)
else:
    run_id = 'runi'

# ===========================================================================
beta = np.array([beta_other]*(n_coef-1)+[beta_t])
if intercept:
    beta[0] = beta_intercept

# random number generator
rng = np.random.RandomState(seed=seed)

# data
xs = rng.uniform(low=-1.0, high=1.0, size=(n_trial, n_obs, n_coef))
if intercept:
    # firs dim ones for intercept
    xs[:,:,0] = 1.0
eps = rng.standard_t(t_df_data, size=(n_trial, n_obs))
ys = xs.dot(beta) + eps

def calc_loo_ti(ys, xs):
    n_coef_cur = xs.shape[2]
    pred_t_df = n_obs - 1 - n_coef_cur

    # natural parameters
    # pointwise products for each trial and data point
    Q_ti = xs[:,:,:,None]*xs[:,:,None,:]
    r_ti = xs[:,:,:]*ys[:,:,None]
    # full posterior for each trial
    Q_t_full = Q_ti.sum(axis=1)
    r_t_full = r_ti.sum(axis=1)

    # working arrays
    temp_mat = np.empty((n_coef_cur, n_coef_cur))
    temp_vec = np.empty((n_coef_cur,))
    temp_loo_vec = np.empty((n_obs-1,))
    x_tilde = np.empty((n_obs-1, n_coef_cur))
    y_tilde = np.empty((n_obs-1,))

    # calc pred distr params
    mu_preds = np.empty((n_trial, n_obs))
    sigma2_preds = np.empty((n_trial, n_obs))
    # for each trial
    for t in range(n_trial):
        # for each data point
        for i in range(n_obs):
            Q_i = np.subtract(Q_t_full[t], Q_ti[t, i], out=temp_mat)
            r_i = np.subtract(r_t_full[t], r_ti[t, i], out=temp_vec)
            x_i = xs[t, i]
            # x_tilde = np.delete(xs[t], i, axis=0)
            # y_tilde = np.delete(ys[t], i, axis=0)
            x_tilde[:i,:] = xs[t,:i,:]
            x_tilde[i:,:] = xs[t,i+1:,:]
            y_tilde[:i] = ys[t,:i]
            y_tilde[i:] = ys[t,i+1:]
            cho = linalg.cho_factor(Q_i, overwrite_a=True)
            xS = linalg.cho_solve(cho, x_i)
            rS = linalg.cho_solve(cho, r_i)
            y_xm = x_tilde.dot(rS, out=temp_loo_vec)
            y_xm -= y_tilde
            s2 = y_xm.dot(y_xm)
            s2 /= pred_t_df
            mu_preds[t, i] = rS.dot(x_i)
            sigma2_preds[t, i] = (xS.dot(x_i) + 1)*s2

    # calc logpdf for pred distributions
    loo_ti = stats.t.logpdf(
        ys, pred_t_df, loc=mu_preds, scale=np.sqrt(sigma2_preds))

    return loo_ti

# model A
loo_ti_A = calc_loo_ti(ys, xs[:,:,:-1])
# model B
loo_ti_B = calc_loo_ti(ys, xs)

# save
os.makedirs('res_1', exist_ok=True)
np.savez_compressed(
    'res_1/{}.npz'.format(run_id),
    loo_ti_A=loo_ti_A,
    loo_ti_B=loo_ti_B,
    seed=seed,
    n_obs=n_obs,
    n_trial=n_trial,
    n_coef=n_coef,
    intercept=intercept,
    beta_t=beta_t,
    beta_other=beta_other,
    beta_intercept=beta_intercept,
    t_df_data=t_df_data,
)
