"""LOOCV linear regression problem setup module.

data:
y = beta'*x + eps
x = standard normal Quasi MC with Sobol sequence
eps = sqrt(sigma2_d_s) * (
    norm_rng(0,1) if nonoutlier else norm_rng(+-outlier_dev,1))

"""

import numpy as np
from scipy import linalg
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
# outliers style (if ``prc_out>0.0``): ['even', 'pos', 'neg']
outliers_style = 'even'
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
# shuffle observations
shuffle_obs = True

# independent test set for true elpd (should be divisible by all n_ob_s)
elpd_test_set_size_target = 2**15  # 32768
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

n_obs_idxs = np.arange(len(n_obs_s))
n_obs_grid = np.full(n_runs, n_obs_def)
n_obs_grid[n_obs_idxs] = n_obs_s

sigma2_d_idxs = np.arange(len(sigma2_d_s)) + n_obs_idxs[-1] + 1
sigma2_d_grid = np.full(n_runs, sigma2_d_def)
sigma2_d_grid[sigma2_d_idxs] = sigma2_d_s

beta_t_idxs = np.arange(len(beta_t_s)) + sigma2_d_idxs[-1] + 1
beta_t_grid = np.full(n_runs, beta_t_def)
beta_t_grid[beta_t_idxs] = beta_t_s

prc_out_idxs = np.arange(len(prc_out_s)) + beta_t_idxs[-1] + 1
prc_out_grid = np.full(n_runs, prc_out_def)
prc_out_grid[prc_out_idxs] = prc_out_s


def determine_n_obs_out_p_m(n_obs, prc_out, outliers_style):
    if prc_out > 0.0:
        if outliers_style == 'even':
            n_obs_out = max(int(np.round(prc_out*n_obs/2)), 1)
            n_obs_out_p = n_obs_out
            n_obs_out_m = n_obs_out
        elif outliers_style == 'pos':
            n_obs_out = max(int(np.round(prc_out*n_obs)), 1)
            n_obs_out_p = n_obs_out
            n_obs_out_m = 0
        elif outliers_style == 'neg':
            n_obs_out = max(int(np.round(prc_out*n_obs)), 1)
            n_obs_out_p = 0
            n_obs_out_m = n_obs_out
        else:
            raise ValueError('invalid arg `outliers_style`')
    else:
        n_obs_out = 0
        n_obs_out_p = 0
        n_obs_out_m = 0
    return n_obs_out_p, n_obs_out_m


def run_i_to_params(run_i):
    n_obs = n_obs_grid[run_i]
    beta_t = beta_t_grid[run_i]
    prc_out = prc_out_grid[run_i]
    sigma2_d = sigma2_d_grid[run_i]
    # fix prc_out
    n_obs_out_p, n_obs_out_m = determine_n_obs_out_p_m(
        n_obs, prc_out, outliers_style)
    n_obs_out = n_obs_out_p + n_obs_out_m
    prc_out = n_obs_out/n_obs
    return n_obs, beta_t, prc_out, sigma2_d


def make_data(n_obs, beta_t, prc_out, sigma2_d):

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
    n_obs_out_p_last, n_obs_out_m_last = determine_n_obs_out_p_m(
        n_obs_s[-1], prc_out_s[-1], outliers_style)
    eps_in_last = rng.normal(size=(n_trial, n_obs_s[-1]))
    eps_out_p_last = rng.normal(
        loc=outlier_dev, size=(n_trial, n_obs_out_p_last))
    eps_out_m_last = rng.normal(
        loc=-outlier_dev, size=(n_trial, n_obs_out_m_last))
    # take current size observations (copy in order to get contiguous)
    X_mat = X_last[:n_obs,:].copy()
    if prc_out > 0.0:
        n_obs_out_p, n_obs_out_m = determine_n_obs_out_p_m(
            n_obs, prc_out, outliers_style)
        n_obs_out = n_obs_out_p + n_obs_out_m
        eps_in = eps_in_last[:,:n_obs-n_obs_out]
        eps_out_p = eps_out_p_last[:,:n_obs_out_p]
        eps_out_m = eps_out_m_last[:,:n_obs_out_m]
        eps = np.concatenate((eps_in, eps_out_p, eps_out_m), axis=1)
        mu_d = np.zeros(n_obs)
        mu_d[n_obs-n_obs_out:n_obs-n_obs_out+n_obs_out_p] = outlier_dev
        mu_d[n_obs-n_obs_out+n_obs_out_p:] = -outlier_dev
    else:
        n_obs_out = 0
        n_obs_out_m = 0
        n_obs_out_p = 0
        eps = eps_in_last[:,:n_obs].copy()
        mu_d = np.zeros(n_obs)
    eps *= np.sqrt(sigma2_d)
    mu_d *= np.sqrt(sigma2_d)
    if shuffle_obs:
        idx = np.arange(n_obs)
        rng.shuffle(idx)
        eps = eps[:,idx]
        mu_d = mu_d[idx]
    # drop unnecessary data
    del(X_last, eps_in_last, eps_out_p_last, eps_out_m_last)
    # calc y
    ys = X_mat.dot(beta) + eps

    # elpd test set
    elpd_size_multip = elpd_test_set_size_target // n_obs
    elpd_test_set_size = elpd_test_set_size_target*elpd_size_multip
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
        n_obs_out_test = elpd_size_multip*n_obs_out
        n_obs_out_test_p = elpd_size_multip*n_obs_out_p
        n_obs_out_test_m = n_obs_out_test - n_obs_out_test_p
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
    if shuffle_obs:
        rng.shuffle(eps_test.T)
    # calc y
    ys_test = X_test.dot(beta) + eps_test

    return X_mat, ys, X_test, ys_test, mu_d


def get_analytic_params(X_mat, beta_t):
    """Analytic result for fixed sigma2 parameters."""
    n_obs, _ = X_mat.shape
    # calc Ps
    Pa = np.zeros((n_obs, n_obs))
    Pb = np.zeros((n_obs, n_obs))
    for i in range(n_obs):
        X_mi = np.delete(X_mat, i, axis=0)
        # a
        XXinvX_a = linalg.solve(
            X_mi[:,:-1].T.dot(X_mi[:,:-1]),
            X_mat[i,:-1],
            assume_a='sym'
        )
        sXX_a = np.sqrt(X_mat[i,:-1].dot(XXinvX_a) + 1)
        # b
        XXinvX_b = linalg.solve(
            X_mi.T.dot(X_mi),
            X_mat[i,:],
            assume_a='sym'
        )
        sXX_b = np.sqrt(X_mat[i,:].dot(XXinvX_b) + 1)
        for j in range(n_obs):
            if i == j:
                # diag
                Pa[i,i] = -1.0/sXX_a
                Pb[i,i] = -1.0/sXX_b
            else:
                # off-diag
                Pa[i,j] = X_mat[j,:-1].dot(XXinvX_a)/sXX_a
                Pb[i,j] = X_mat[j,:].dot(XXinvX_b)/sXX_b
    #
    PaX = Pa.dot(X_mat[:,-1])
    # calc A
    A_mat = Pa.T.dot(Pa)
    A_mat -= Pb.T.dot(Pb)
    A_mat /= -2*sigma2_m
    # calc b
    b_vec = Pa.T.dot(PaX)
    b_vec *= -beta_t/sigma2_m
    # calc c
    c_sca = PaX.T.dot(PaX)
    c_sca *= -beta_t**2/(2*sigma2_m)
    c_sca += np.sum(np.log(-np.diag(Pa))) - np.sum(np.log(-np.diag(Pb)))
    return A_mat, b_vec, c_sca


def calc_analytic_mean(A_mat, b_vec, c_sca, sigma2_d, mu_d=None):
    """Calc analytic loo mean for fixed sigma2."""
    out = c_sca
    if mu_d is not None:
        out += np.sqrt(sigma2_d)*(b_vec.T.dot(mu_d))
        out += sigma2_d*(mu_d.T.dot(A_mat).dot(mu_d))
    return out

def calc_analytic_var(A_mat, b_vec, c_sca, sigma2_d, mu_d=None):
    """Calc analytic loo var for fixed sigma2."""
    A2 = A_mat.dot(A_mat)
    out = 2*sigma2_d**2*np.trace(A2)
    out += sigma2_d*(b_vec.T.dot(b_vec))
    if mu_d is not None:
        out += 4*np.sqrt(sigma2_d)**3*(b_vec.T.dot(A_mat).dot(mu_d))
        out += 4*sigma2_d**2*(mu_d.T.dot(A2).dot(mu_d))
    return out

def calc_analytic_moment3(A_mat, b_vec, c_sca, sigma2_d, mu_d=None):
    """Calc analytic loo 3rd central moment for fixed sigma2."""
    A2 = A_mat.dot(A_mat)
    A3 = A2.dot(A_mat)
    out = 8*sigma2_d**3*np.trace(A3)
    out += 6*sigma2_d**2*(b_vec.T.dot(A_mat).dot(b_vec))
    if mu_d is not None:
        out += 24*np.sqrt(sigma2_d)**5*(b_vec.T.dot(A2).dot(mu_d))
        out += 24*sigma2_d**3*(mu_d.T.dot(A3).dot(mu_d))
    return out

def calc_analytic_coefvar(A_mat, b_vec, c_sca, sigma2_d, mu_d=None):
    """Calc analytic loo coefficient of variation for fixed sigma2."""
    mean = calc_analytic_mean(A_mat, b_vec, c_sca, sigma2_d, mu_d)
    var = calc_analytic_var(A_mat, b_vec, c_sca, sigma2_d, mu_d)
    return np.sqrt(var)/mean

def calc_analytic_skew(A_mat, b_vec, c_sca, sigma2_d, mu_d=None):
    """Calc analytic loo skewness for fixed sigma2."""
    var = calc_analytic_var(A_mat, b_vec, c_sca, sigma2_d, mu_d)
    moment3 = calc_analytic_moment3(A_mat, b_vec, c_sca, sigma2_d, mu_d)
    return moment3 / np.sqrt(var)**3
