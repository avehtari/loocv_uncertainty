"""LOOCV linear regression problem setup module.

data:
y = X*beta + eps
x = standard normal
eps = (
    norm_rng(0, sigma2_d)
    if nonoutlier else
    norm_rng(+-outlier_dev*sqrt(sigma2_d + sum_i beta_i^2), sigma2_d)
)

"""


import numpy as np
from scipy import linalg, stats


# ===========================================================================
# confs

# Random seed for data. Generate one with:
# np.random.RandomState().randint(np.iinfo(np.uint32).max)
seed = 2958319585


# grid params and default values
# number of obs in one trial
n_obs_s = [16, 32, 64, 128, 256, 512, 1024]
# epsilon sigma2_d_s
sigma2_d_s = [0.01, 1.0, 100.0]
# last covariate effect not used in model A
beta_t_s = [0.0, 0.5, 1.0, 4.0]
# percentage of outliers (np.nextafter(0,1) corresponds to always 1 out)
prc_out_s = [0.0, np.nextafter(0,1), 0.01, 0.08]

# fixed model tau2 value
tau2 = 1.0
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

# number of independent test data sets of size n_obs for true elpd
elpd_test_n = 4000

# pseudo_bma_p seed
seed_bma = 1022464040
# pseudo_bma_p bb sample size
n_bb_bma = 500



# ===========================================================================

# set grid
n_obs_grid, sigma2_d_grid, beta_t_grid, prc_out_grid = np.meshgrid(
    n_obs_s, sigma2_d_s, beta_t_s, prc_out_s)
grid_shape = n_obs_grid.shape
n_runs = n_obs_grid.size


def determine_n_obs_out(n_obs, prc_out):
    """Determine outliers numbers."""
    return int(np.ceil(prc_out*n_obs))


def run_i_to_params(run_i):
    n_obs = n_obs_grid.flat[run_i]
    beta_t = beta_t_grid.flat[run_i]
    prc_out = prc_out_grid.flat[run_i]
    sigma2_d = sigma2_d_grid.flat[run_i]
    return n_obs, beta_t, prc_out, sigma2_d


def make_data(n_obs, beta_t, prc_out, sigma2_d):

    # random number generator
    rng = np.random.RandomState(seed=seed)

    # beta vector
    beta = np.array([beta_other]*(n_dim-1)+[beta_t])
    if intercept:
        beta[0] = beta_intercept

    # calc outlier deviation for eps
    outlier_dev_eps = outlier_dev*np.sqrt(sigma2_d + np.sum(beta**2))
    sigma_d = np.sqrt(sigma2_d)

    # data
    n_obs_out = determine_n_obs_out(n_obs, prc_out)
    # X
    if intercept:
        # firs dim (column) ones for intercept
        # X_mat = np.dstack((
        #     np.ones((elpd_test_n, n_obs, 1)),
        #     rng.uniform(low=-1.0, high=1.0, size=(elpd_test_n, n_obs, n_dim-1))
        # ))
        X_mat = np.dstack((
            np.ones((n_trial, n_obs, 1)),
            rng.randn(n_trial, n_obs, n_dim-1)
        ))
    else:
        # X_mat = rng.uniform(low=-1.0, high=1.0, size=(elpd_test_n, n_obs, n_dim))
        X_mat = rng.randn(n_trial, n_obs, n_dim)
    # mu
    mu_d = np.zeros((n_trial, n_obs))
    for trial_i in range(n_trial):
        out_idx = rng.choice(n_obs, size=n_obs_out, replace=False)
        mu_d[trial_i, out_idx] = rng.choice(
            [outlier_dev_eps, -outlier_dev_eps],
            size=n_obs_out,
            replace=True
        )
    # eps
    eps = rng.normal(loc=mu_d, scale=sigma_d)
    ys = X_mat.dot(beta) + eps

    # elpd test set
    # X
    if intercept:
        # firs dim (column) ones for intercept
        # X_test = np.dstack((
        #     np.ones((elpd_test_n, n_obs, 1)),
        #     rng.uniform(low=-1.0, high=1.0, size=(elpd_test_n, n_obs, n_dim-1))
        # ))
        X_test = np.dstack((
            np.ones((elpd_test_n, n_obs, 1)),
            rng.randn(elpd_test_n, n_obs, n_dim-1)
        ))
    else:
        # X_test = rng.uniform(low=-1.0, high=1.0, size=(elpd_test_n, n_obs, n_dim))
        X_test = rng.randn(elpd_test_n, n_obs, n_dim)
    # mu
    mu_d_test = np.zeros((elpd_test_n, n_obs))
    for trial_i in range(elpd_test_n):
        out_idx = rng.choice(n_obs, size=n_obs_out, replace=False)
        mu_d_test[trial_i, out_idx] = rng.choice(
            [outlier_dev_eps, -outlier_dev_eps],
            size=n_obs_out,
            replace=True
        )
    # eps
    eps_test = rng.normal(loc=mu_d_test, scale=sigma_d)
    ys_test = X_test.dot(beta) + eps_test

    return X_mat, mu_d, ys, X_test, ys_test


def calc_loo_ti(ys, X_mat, fixed_sigma2_m):
    n_trial_cur = ys.shape[0]
    _, n_obs_cur, n_dim_cur = X_mat.shape
    # working arrays
    x_tilde = np.empty((n_obs_cur-1, n_dim_cur))
    y_tilde = np.empty((n_obs_cur-1,))
    # pred distr params
    mu_preds = np.empty((n_trial_cur, n_obs_cur))
    sigma2_preds = np.empty((n_trial_cur, n_obs_cur))
    # loop for each trial
    for t in range(n_trial_cur):
        # LOO params for each data point
        for i in range(n_obs_cur):
            x_i = X_mat[t,i]
            # x_tilde = np.delete(X_mat[t], i, axis=0)
            # y_tilde = np.delete(ys[t], i, axis=0)
            x_tilde[:i,:] = X_mat[t,:i,:]
            x_tilde[i:,:] = X_mat[t,i+1:,:]
            y_tilde[:i] = ys[t,:i]
            y_tilde[i:] = ys[t,i+1:]

            cho = linalg.cho_factor(x_tilde.T.dot(x_tilde).T, overwrite_a=True)
            xSx_p1 = x_i.dot(linalg.cho_solve(cho, x_i)) + 1.0

            beta_hat = linalg.cho_solve(cho, x_tilde.T.dot(y_tilde))
            mu_preds[t, i] = x_i.dot(beta_hat)
            if fixed_sigma2_m:
                sigma2_preds[t, i] = xSx_p1*tau2
            else:
                y_xm = x_tilde.dot(beta_hat)
                y_xm -= y_tilde
                s2 = y_xm.dot(y_xm)
                s2 /= n_obs_cur - 1 - n_dim_cur
                sigma2_preds[t, i] = xSx_p1*s2
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


def calc_test_t(ys, X_mat, ys_test, X_test, fixed_sigma2_m):
    n_trial, n_obs, n_dim_cur = X_mat.shape
    elpd_test_n, _ = ys_test.shape
    # test set pred distr params working array
    mu_pred_test = np.empty((elpd_test_n, n_obs,))
    sigma2_pred_test = np.empty((elpd_test_n, n_obs,))
    # test set logpdf
    test_ti = np.empty((n_trial, n_obs))
    # loop for each trial
    for t in range(n_trial):
        # test set pred params
        cho_test = linalg.cho_factor(X_mat[t].T.dot(X_mat[t]).T, overwrite_a=True)
        beta_hat = linalg.cho_solve(cho_test, X_mat[t].T.dot(ys[t]))
        if fixed_sigma2_m:
            s2 = tau2
        else:
            y_xm = X_mat[t].dot(beta_hat)
            y_xm -= ys[t]
            s2 = y_xm.dot(y_xm)
            s2 /= n_obs - n_dim_cur
        for tt in range(elpd_test_n):
            xSx_p1_test = np.einsum(
                'td,dt->t',
                X_test[tt],
                linalg.cho_solve(cho_test, X_test[tt].T)
            )
            xSx_p1_test += 1.0
            X_test[tt].dot(beta_hat, out=mu_pred_test[tt])
            np.multiply(xSx_p1_test, s2, out=sigma2_pred_test[tt])
        # calc logpdf for test
        if fixed_sigma2_m:
            test_logpdf = stats.norm.logpdf(
                ys_test,
                loc=mu_pred_test,
                scale=np.sqrt(sigma2_pred_test)
            )
            test_ti[t] = np.mean(test_logpdf, axis=0)
        else:
            test_logpdf = stats.t.logpdf(
                ys_test,
                n_obs - n_dim_cur,
                loc=mu_pred_test,
                scale=np.sqrt(sigma2_pred_test)
            )
            test_ti[t] = np.mean(test_logpdf, axis=0)
    test_t = np.sum(test_ti, axis=1)
    return test_t


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
    A_mat /= -2*tau2
    # calc b
    b_vec = Pa.T.dot(PaX)
    b_vec *= -beta_t/tau2
    # calc c
    c_sca = PaX.T.dot(PaX)
    c_sca *= -beta_t**2/(2*tau2)
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


def pseudo_bma_p(loo_tki):
    rng = np.random.RandomState(seed=seed_bma)
    n_obs = loo_tki.shape[-1]
    alpha = rng.dirichlet(np.ones(n_obs), size=n_bb_bma)
    z_tkb = np.sum(alpha.T*loo_tki[...,None], axis=-2)
    n_z_tkb = np.multiply(z_tkb, n_obs, out=z_tkb)
    exp_n_z_tkb = np.exp(n_z_tkb, out=n_z_tkb)
    sum_exp_n_z_t1b = np.sum(exp_n_z_tkb, axis=-2, keepdims=True)
    # might contain zeroes
    w_tkb = np.divide(exp_n_z_tkb, sum_exp_n_z_t1b, out=exp_n_z_tkb)
    # nanmean because of zeroes
    w_tk = np.nanmean(w_tkb, axis=-1)
    return w_tk
