"""General setup module.

data:
y = X*beta + eps
x = standard normal
eps = (
    norm_rng(0, sigma2_d)
    if nonoutlier else
    norm_rng(outlier_dev*sqrt(sigma2_d + sum_i beta_i^2), sigma2_d)
)

"""


import sys

import numpy as np
from scipy import linalg, stats

from scipy.special import expit as logistic_fun


# ============================================================================
# config

# default values for ProblemRun
BETA_T = 1.0
OUT_DEV = 0.0
N_OBS_OUT = 1
SIGMA2_D = 1.0
TAU2 = None
N_DIM = 3
INTERCEPT = True
BETA_OTHER = 1.0
BETA_INTERCEPT = 0.0
N_TRIAL = 2000
ELPD_TEST_N = 4000
SEED = 2958319585

# config for BB uncertainty approximation
# BB seed
BB_SEED = 555830760
# BB sample size
BB_N = 2000

# config for BMA weight calculation
# BB seed
BMA_BB_SEED = 1022464040
# BB sample size
BMA_BB_N = 2000

# config for BB moment estimation
# BB seed
BB_MOMENT_SEED = 243682451
# BB sample size
BB_MOMENT_N = 2000


# ============================================================================


class ProblemRun:

    def __init__(
            self, n_obs, n_obs_max, beta_t=BETA_T, out_dev=OUT_DEV,
            n_obs_out=N_OBS_OUT, sigma2_d=SIGMA2_D, tau2=TAU2, n_dim=N_DIM,
            intercept=INTERCEPT, beta_other=BETA_OTHER,
            beta_intercept=BETA_INTERCEPT, n_trial=N_TRIAL,
            elpd_test_n=ELPD_TEST_N, seed=SEED):
        """Problem setup.

        Parameters
        ----------
        n_obs : int
            number of obs
        n_obs_max : int
            maximum number of observations in a set of runs
        beta_t : float, optional
            effect of the covariate missing in model A
        out_dev : float or ndarray, optional
            mean of the outlier observations
        n_obs_out : int, optional
            number of outlier observations
        sigma2_d : float, optional
            data residual variance: sigma2_d * eye(n_obs)
        tau2 : float, optional
            Fixed model tau2 value. None corresponds to non-fixed tau2 model.
        n_dim : int, optional
            dimensionality of beta
        intercept : bool, optional
            first covariate as intercept
        beta_other : float, optional
            other covariates' effects
        beta_intercept : float, optional
            intercept coef (if applied)
        n_trial : int, optional
            number of trials
        elpd_test_n : int, optional
            number of independent test data sets of size n_obs for true elpd
        seed : int, optional
            Random seed for data. Generate one with:
            ``np.random.RandomState().randint(np.iinfo(np.uint32).max)``
        """
        self.n_obs = n_obs
        self.n_obs_max = n_obs_max
        self.beta_t = beta_t
        self.out_dev = out_dev
        self.n_obs_out = n_obs_out
        self.sigma2_d = sigma2_d
        self.tau2 = tau2
        self.n_dim = n_dim
        self.intercept = intercept
        self.beta_other = beta_other
        self.beta_intercept = beta_intercept
        self.n_trial = n_trial
        self.elpd_test_n = elpd_test_n
        self.seed = seed

        self.args = (
            self.n_obs, self.n_obs_max, self.beta_t, self.out_dev,
            self.n_obs_out, self.sigma2_d, self.tau2, self.n_dim,
            self.intercept, self.beta_other, self.beta_intercept, self.n_trial,
            self.elpd_test_n, self.seed,
        )

        self.sigma_d = np.sqrt(self.sigma2_d)

        # random number generator
        self.rng = np.random.RandomState(seed=self.seed)

        # beta vector
        self.beta = np.array([self.beta_other]*(self.n_dim-1)+[self.beta_t])
        if self.intercept:
            self.beta[0] = self.beta_intercept


    def get_args(self):
        """Return the args this problem was initialised with."""
        return self.args


    def make_data(self, n_sets=None):
        """Make data sets.

        n_sets : int, optional
            Number of data sets to make. ``self.n_trial`` by default

        """
        if n_sets is None:
            n_sets = self.n_trial
        # calc outlier deviation for eps
        outlier_dev_eps = self.out_dev*np.sqrt(
            self.sigma2_d + np.sum(self.beta**2))
        mu_d = np.zeros(self.n_obs_max)
        mu_d[:self.n_obs_out] = outlier_dev_eps
        # X
        if self.intercept:
            # firs dim (column) ones for intercept
            # X_tid = np.dstack((
            #     np.ones((n_sets, self.n_obs_max, 1)),
            #     self.rng.uniform(
            #         low=-1.0, high=1.0,
            #         size=(n_sets, self.n_obs_max, self.n_dim-1))
            # ))
            X_tid = np.dstack((
                np.ones((n_sets, self.n_obs_max, 1)),
                self.rng.randn(n_sets, self.n_obs_max, self.n_dim-1)
            ))
        else:
            # X_tid = self.rng.uniform(
            #     low=-1.0, high=1.0,
            #     size=(n_sets, self.n_obs_max, self.n_dim)
            # )
            X_tid = self.rng.randn(n_sets, self.n_obs_max, self.n_dim)
        # eps
        eps = self.rng.normal(
            loc=mu_d, scale=self.sigma_d, size=(n_sets, self.n_obs_max))
        # slice n_obs
        X_tid = X_tid[:, :self.n_obs, :]
        eps = eps[:, :self.n_obs]
        # calc y_vec
        y_ti = X_tid.dot(self.beta) + eps
        return X_tid, y_ti


    def calc_loo_ti(self, y_ti, X_tid, y2_ti=None, X2_tid=None):
        """Calc pointwise LOO terms.

        Providing parameters `y2_ti`, `X2_tid` use these observations for each
        predicted observation instead of the ones from `y_ti`, `X_tid`.

        """
        n_trial, n_obs, n_dim = X_tid.shape
        # working arrays
        x_tilde = np.empty((n_obs-1, n_dim))
        y_tilde = np.empty((n_obs-1,))
        # pred distr params
        mu_preds = np.empty((n_trial, n_obs))
        sigma2_preds = np.empty((n_trial, n_obs))
        # loop for each trial
        for t in range(n_trial):
            # LOO params for each data point
            for i in range(n_obs):
                x_i = X_tid[t,i] if X2_tid is None else X2_tid[t,i]
                # x_tilde = np.delete(X_tid[t], i, axis=0)
                # y_tilde = np.delete(y_ti[t], i, axis=0)
                x_tilde[:i,:] = X_tid[t,:i,:]
                x_tilde[i:,:] = X_tid[t,i+1:,:]
                y_tilde[:i] = y_ti[t,:i]
                y_tilde[i:] = y_ti[t,i+1:]

                cho = linalg.cho_factor(
                    x_tilde.T.dot(x_tilde).T, overwrite_a=True)
                xSx_p1 = x_i.dot(linalg.cho_solve(cho, x_i)) + 1.0

                beta_hat = linalg.cho_solve(cho, x_tilde.T.dot(y_tilde))
                mu_preds[t, i] = x_i.dot(beta_hat)
                if self.tau2 is not None:
                    sigma2_preds[t, i] = xSx_p1*self.tau2
                else:
                    y_xm = x_tilde.dot(beta_hat)
                    y_xm -= y_tilde
                    s2 = y_xm.dot(y_xm)
                    s2 /= n_obs - 1 - n_dim
                    sigma2_preds[t, i] = xSx_p1*s2
        # calc logpdf for loos
        if self.tau2 is not None:
            loo_ti = stats.norm.logpdf(
                y_ti if y2_ti is None else y2_ti,
                loc=mu_preds,
                scale=np.sqrt(sigma2_preds)
            )
        else:
            loo_ti = stats.t.logpdf(
                y_ti if y2_ti is None else y2_ti,
                n_obs - 1 - n_dim,
                loc=mu_preds,
                scale=np.sqrt(sigma2_preds)
            )
        return loo_ti


    def calc_elpd_tl(self, y_ti, X_tid, y_test_ti, X_test_tid):
        n_trial, n_obs, n_dim = X_tid.shape
        elpd_test_n, _ = y_test_ti.shape
        # test set pred distr params working array
        mu_pred_test = np.empty((elpd_test_n, n_obs,))
        sigma2_pred_test = np.empty((elpd_test_n, n_obs,))
        # test set logpdf
        test_lpd_tl = np.empty((n_trial, elpd_test_n))
        # loop for each trial
        for t in range(n_trial):
            # test set pred params
            cho_test = linalg.cho_factor(
                X_tid[t].T.dot(X_tid[t]).T, overwrite_a=True)
            beta_hat = linalg.cho_solve(cho_test, X_tid[t].T.dot(y_ti[t]))
            if self.tau2 is not None:
                s2 = self.tau2
            else:
                y_xm = X_tid[t].dot(beta_hat)
                y_xm -= y_ti[t]
                s2 = y_xm.dot(y_xm)
                s2 /= n_obs - n_dim
            for tt in range(elpd_test_n):
                xSx_p1_test = np.einsum(
                    'td,dt->t',
                    X_test_tid[tt],
                    linalg.cho_solve(cho_test, X_test_tid[tt].T)
                )
                xSx_p1_test += 1.0
                X_test_tid[tt].dot(beta_hat, out=mu_pred_test[tt])
                np.multiply(xSx_p1_test, s2, out=sigma2_pred_test[tt])
            # calc logpdf for test
            if self.tau2 is not None:
                test_logpdf = stats.norm.logpdf(
                    y_test_ti,
                    loc=mu_pred_test,
                    scale=np.sqrt(sigma2_pred_test)
                )
            else:
                test_logpdf = stats.t.logpdf(
                    y_test_ti,
                    n_obs - n_dim,
                    loc=mu_pred_test,
                    scale=np.sqrt(sigma2_pred_test)
                )
            test_lpd_tl[t] = np.sum(test_logpdf, axis=1)
        return test_lpd_tl


def calc_ls_estim(y_ti, X_tid):
    """Calculate classical least-squares linear regression point estimates."""
    n_trial, n_obs, n_dim = X_tid
    # loop for each trial
    beta_hat = np.zeros((n_trial, n_dim))
    s2_hat = np.zeros((n_trial,))
    for t in range(n_trial):
        beta_hat[t] = linalg.lstsq(X_tid[t], y_ti[t])[0]
        temp = X_tid[t].dot(beta_hat[t])
        temp -= y_ti[t]
        s2_hat[t] = temp.T.dot(temp) / (n_obs - n_dim)
    return beta_hat, s2_hat


def calc_bb_mean_var_prctiles_plooneg(loo_ti, seed=BB_SEED):
    """Calc BB mean, var, 2.5-50-97.5 prc, and p(elpdhat<0) estimates."""
    rng = np.random.RandomState(seed=seed)
    n_obs = loo_ti.shape[-1]
    alpha = rng.dirichlet(np.ones(n_obs), size=BB_N)
    # z_tb = np.sum(alpha.T*loo_ti[..., None], axis=-2)
    z_tb = np.einsum('bi,...ib->...b', alpha, loo_ti[..., None])
    n_z_tb = np.multiply(z_tb, n_obs, out=z_tb)
    bb_mean = np.mean(n_z_tb, axis=-1)
    bb_var = np.var(n_z_tb, axis=-1, ddof=1)
    bb_025, bb_500, bb_975 = np.percentile(n_z_tb, [2.5, 50.0, 97.5], axis=-1)
    bb_plooneg = np.mean(n_z_tb<0, axis=-1)
    return bb_mean, bb_var, bb_025, bb_500, bb_975, bb_plooneg


def calc_loo_bb(loo_tki, seed=BMA_BB_SEED):
    """Calc LOO-BB-weight for multiple trials t, and models k, given loo_i."""
    rng = np.random.RandomState(seed=seed)
    n_obs = loo_tki.shape[-1]
    alpha = rng.dirichlet(np.ones(n_obs), size=BMA_BB_N)
    # z_tkb = np.sum(alpha.T*loo_tki[..., None], axis=-2)
    z_tkb = np.einsum('bi,...ib->...b', alpha, loo_tki[..., None])
    n_z_tkb = np.multiply(z_tkb, n_obs, out=z_tkb)
    # try to avoid precision problems
    n_z_tkb -= np.max(n_z_tkb, axis=-2, keepdims=True)
    sum_exp_n_z_t1b_shift = np.sum(np.exp(n_z_tkb), axis=-2, keepdims=True)
    log_sum = np.log(sum_exp_n_z_t1b_shift, out=sum_exp_n_z_t1b_shift)
    n_z_tkb -= log_sum
    w_tkb = np.exp(n_z_tkb, out=n_z_tkb)
    # nanmean because of possible precision problems
    w_tk = np.nanmean(w_tkb, axis=-1)
    return w_tk


def calc_loo_bb_pair(loo_ti, seed=BMA_BB_SEED):
    """Calc LOO-BB-weight for diff, for multiple trials t, given loo_i.

    The resulting weight corresponds to the weights of the model M_a in
    comparison M_a - M_b.

    """
    rng = np.random.RandomState(seed=seed)
    n_obs = loo_ti.shape[-1]
    alpha = rng.dirichlet(np.ones(n_obs), size=BMA_BB_N)
    # z_tb = np.sum(alpha.T*loo_ti[..., None], axis=-2)
    z_tb = np.einsum('bi,...ib->...b', alpha, loo_ti[..., None])
    n_z_tb = np.multiply(z_tb, n_obs, out=z_tb)
    w_tb = logistic_fun(n_z_tb, out=n_z_tb)
    w_t = np.mean(w_tb, axis=-1)
    return w_t


def calc_p_bma(loo_tk):
    """Calc P-BMA-weight for multiple trials t, and models k, given elpdhat."""
    shift = np.max(loo_tk, axis=-1, keepdims=True)
    loo_tk_shift = loo_tk - shift
    w_tk = np.exp(
        loo_tk_shift
        - np.log(
            np.sum(np.exp(loo_tk_shift), axis=-1, keepdims=True)
        )
    )
    return w_tk


def calc_p_bma_pair(loo_t):
    """Calc P-BMA-weight for diff, for multiple trials t, given elpdhat.

    The resulting weight corresponds to the weights of the model M_a in
    comparison M_a - M_b.

    """
    w_t = logistic_fun(loo_t)
    return w_t


def bb_mean_sd_skew(data_pi, seed=BB_MOMENT_SEED):
    # BB moments
    n_probl, n_data = data_pi.shape
    rng = np.random.RandomState(seed=seed)
    w_bi = rng.dirichlet(np.ones(n_data), size=BB_MOMENT_N)

    # sum_w_b = np.sum(w_bi, axis=-1)  # = 1
    # sum_w_2_b = sum_w_b**2  # = 1
    # sum_w_3_b = sum_w_2_b*sum_w_b  # = 1
    sum_w2_b = np.sum(w_bi**2, axis=-1)
    sum_w3_b = np.sum(w_bi**3, axis=-1)
    norm_var = 1 - sum_w2_b
    norm_skew = 1 - 3*sum_w2_b + 2*sum_w3_b

    mean_pb = data_pi.dot(w_bi.T)
    # looping though p because of memory reasons (should be cythonised)
    # data_centered_pib = data_pi[:, :, None] - mean_pb[:, None, :]
    sd_pb = np.zeros((n_probl, BB_MOMENT_N))
    skew_pb = np.zeros((n_probl, BB_MOMENT_N))
    for p in range(n_probl):
        work_data_bi = data_pi[p, None, :] - mean_pb[p, :, None]
        # sd
        sd_pb[p] = np.einsum(
            'bi,bi,bi->b', w_bi, work_data_bi, work_data_bi)
        sd_pb[p] /= norm_var
        np.sqrt(sd_pb[p], out=sd_pb[p])
        # skew
        work_data_bi /= sd_pb[p][:, None]
        skew_pb[p] = np.einsum(
            'bi,bi,bi,bi->b',
            w_bi, work_data_bi, work_data_bi, work_data_bi
        )
        skew_pb[p] /= norm_skew
    return mean_pb, sd_pb, skew_pb
