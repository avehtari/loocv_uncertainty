"""LOOCV linear regression problem setup module.

data:
y = beta'*x + eps
x = standard normal Quasi MC with Sobol sequence
eps = sqrt(sigma2_d_s) * (
    norm_rng(0,1) if nonoutlier else norm_rng(+-outlier_dev,1))

"""

import numpy as np
import sobol_seq


# ===========================================================================
# confs

# Random seed for data. Generate one with:
# np.random.RandomState().randint(np.iinfo(np.uint32).max)
seed = 2958319585

# grid params and default values
# number of obs in one trial
n_obs_s = [16, 32, 64, 128, 256, 512, 1024]
n_obs_def = 256
# epsilon sigma2_d_s
sigma2_d_s = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
sigma2_d_def = 1.0
# last covariate effect not used in model A
beta_t_s = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
beta_t_def = 1.0
# percentage of outliers
prc_out_s = [0.0, 0.01, 0.02, 0.05, 0.1]
prc_out_def = 0.02

# fixed model sigma2_m value
sigma2_m = 1.0
# outlier loc deviation
outlier_dev = 20.0
# number of trials
n_trial = 2000
# dimensionality of beta
n_dim = 3
# first covariate as intercept
intercept = True
# other covariates' effects
beta_other = 1.0
# intercept coef (if applied)
beta_intercept = 0.0
# suffle observations
suffle_obs = True

# independent test set for true elpd
elpd_test_set_size = 20000
# outliers in the independent test set for true elpd
elpd_test_outliers = True

# bootstrap loo repetitions
n_boot_trial = 100
# bootrap sampling random seed
# np.random.RandomState().randint(np.iinfo(np.uint32).max)
seed_boot = 1584981854

# model based bootstrap loo repetitions
n_mboot_trial = 100
# bootrap sampling random seed
# np.random.RandomState().randint(np.iinfo(np.uint32).max)
seed_mboot = 1118669156


# ===========================================================================

# set grid
n_runs = sum(map(len, (n_obs_s, sigma2_d_s, beta_t_s, prc_out_s)))
n_obs_grid = np.array(
    n_obs_s +
    sum(map(len, (sigma2_d_s, beta_t_s, prc_out_s)))*[n_obs_def]
)
sigma2_d_grid = np.array(
    len(n_obs_s)*[sigma2_d_def] +
    sigma2_d_s +
    sum(map(len, (beta_t_s, prc_out_s)))*[sigma2_d_def]
)
beta_t_grid = np.array(
    sum(map(len, (n_obs_s, sigma2_d_s)))*[beta_t_def] +
    beta_t_s +
    len(prc_out_s)*[beta_t_def]
)
prc_out_grid = np.array(
    sum(map(len, (n_obs_s, sigma2_d_s, beta_t_s)))*[prc_out_def] +
    prc_out_s
)


def run_i_to_params(run_i):
    n_obs = n_obs_grid[run_i]
    beta_t = beta_t_grid[run_i]
    prc_out = prc_out_grid[run_i]
    sigma2_d = sigma2_d_grid[run_i]
    return n_obs, beta_t, prc_out, sigma2_d


def make_data(run_i):

    # set params for this run
    n_obs, beta_t, prc_out, sigma2_d = run_i_to_params(run_i)

    # random number generator
    rng = np.random.RandomState(seed=seed)

    # beta vector
    beta = np.array([beta_other]*(n_dim-1)+[beta_t])
    if intercept:
        beta[0] = beta_intercept

    # data for all cases
    if intercept:
        # firs dim (column) ones for intercept
        X_last = np.hstack((
            np.ones((n_obs_s[-1], 1)),
            sobol_seq.i4_sobol_generate_std_normal(n_dim-1, n_obs_s[-1])
        ))
    else:
        X_last = sobol_seq.i4_sobol_generate_std_normal(n_dim, n_obs_s[-1])
    # data for the biggest sample size
    n_obs_out_last = max(int(np.ceil(prc_out_s[-1]*n_obs_s[-1])), 2)
    n_obs_out_m_last = n_obs_out_last // 2
    n_obs_out_p_last = n_obs_out_last - n_obs_out_m_last
    eps_in_last = rng.normal(size=(n_trial, n_obs_s[-1]))
    eps_out_p_last = rng.normal(
        loc=outlier_dev, size=(n_trial, n_obs_out_p_last))
    eps_out_m_last = rng.normal(
        loc=-outlier_dev, size=(n_trial, n_obs_out_m_last))
    # take current size observations (copy in order to get contiguous)
    X_mat = X_last[:n_obs,:].copy()
    if prc_out > 0.0:
        n_obs_out = max(int(np.ceil(prc_out*n_obs)), 2)
        n_obs_out_m = n_obs_out // 2
        n_obs_out_p = n_obs_out - n_obs_out_m
        eps_in = eps_in_last[:,:n_obs-n_obs_out]
        eps_out_p = eps_out_p_last[:,:n_obs_out_p]
        eps_out_m = eps_out_m_last[:,:n_obs_out_m]
        eps = np.concatenate((eps_in, eps_out_p, eps_out_m), axis=1)
    else:
        n_obs_out = 0
        n_obs_out_m = 0
        n_obs_out_p = 0
        eps = eps_in_last[:,:n_obs].copy()
    eps *= np.sqrt(sigma2_d)
    if suffle_obs:
        rng.shuffle(eps.T)
    # drop unnecessary data
    del(X_last, eps_in_last, eps_out_p_last, eps_out_m_last)
    # calc y
    ys = X_mat.dot(beta) + eps

    # elpd test set
    if intercept:
        # firs dim (column) ones for intercept
        X_test = np.hstack((
            np.ones((elpd_test_set_size, 1)),
            sobol_seq.i4_sobol_generate_std_normal(
                n_dim-1, elpd_test_set_size, skip=n_obs_s[-1]+1)
        ))
    else:
        X_test = sobol_seq.i4_sobol_generate_std_normal(
            n_dim, elpd_test_set_size, skip=n_obs_s[-1]+1)
    if elpd_test_outliers and prc_out > 0.0:
        n_obs_out_test = max(int(np.ceil(prc_out*elpd_test_set_size)), 2)
        n_obs_out_test_m = n_obs_out_test // 2
        n_obs_out_test_p = n_obs_out_test - n_obs_out_test_m
        eps_test_in = rng.normal(
            size=(elpd_test_set_size-n_obs_out_test,))
        eps_test_out_p = rng.normal(
            loc=outlier_dev, size=(n_obs_out_test_p,))
        eps_test_out_m = rng.normal(
            loc=-outlier_dev, size=(n_obs_out_test_m,))
        eps_test = np.concatenate(
            (eps_test_in, eps_test_out_p, eps_test_out_m), axis=0)
        del(eps_test_in, eps_test_out_p, eps_test_out_m)
    else:
        n_obs_out_test = 0
        n_obs_out_test_m = 0
        n_obs_out_test_p = 0
        eps_test = rng.normal(size=(elpd_test_set_size,))
    eps_test *= np.sqrt(sigma2_d)
    if suffle_obs:
        rng.shuffle(eps_test.T)
    # calc y
    ys_test = X_test.dot(beta) + eps_test

    return X_mat, ys, X_test, ys_test
