
import numpy as np
from scipy import linalg, stats

# ============================================================================
# Config

# data seed
# np.random.RandomState().randint(np.iinfo(np.uint32).max)
data_seed = 247169102

n_trial = 10

# fixed model tau2 value
tau2 = 1.0
# outlier loc deviation
outlier_dev = 20.0
# dimensionality of beta
n_dim = 3
# first covariate as intercept
intercept = True
# other covariates' effects
beta_other = 1.0
# intercept coef (if applied)
beta_intercept = 0.0


# ============================================================================
# Funcs

def determine_n_obs_out(n_obs, prc_out):
    """Determine outliers numbers."""
    return int(np.ceil(prc_out*n_obs))


class _DataGeneration:

    def __init__(self):
        self.rng = np.random.RandomState(seed=data_seed)

    def make_x_mu(self, n_obs, n_obs_out, sigma2_d, beta_t):
        beta = np.array([beta_other]*(n_dim-1)+[beta_t])
        if intercept:
            beta[0] = beta_intercept
        # X
        if intercept:
            # firs dim (column) ones for intercept
            # X_mat = np.hstack((
            #     np.ones((n_obs, 1)),
            #     self.rng.uniform(low=-1.0, high=1.0, size=(n_obs, n_dim-1))
            # ))
            X_mat = np.hstack((
                np.ones((n_obs, 1)),
                self.rng.randn(n_obs, n_dim-1)
            ))
        else:
            # X_mat = self.rng.uniform(low=-1.0, high=1.0, size=(n_obs, n_dim))
            X_mat = self.rng.randn(n_obs, n_dim)
        # mu
        outlier_dev_eps = outlier_dev*np.sqrt(sigma2_d + np.sum(beta**2))
        mu_d = np.zeros(n_obs)
        out_idx = self.rng.choice(n_obs, size=n_obs_out, replace=False)
        mu_d[out_idx] = self.rng.choice(
            [outlier_dev_eps, -outlier_dev_eps],
            size=n_obs_out,
            replace=True
        )
        return X_mat, mu_d

# make this function to share seed when ever called
make_x_mu = _DataGeneration().make_x_mu



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
        out += b_vec.T.dot(mu_d)
        out += mu_d.T.dot(A_mat).dot(mu_d)
    return out

def calc_analytic_var(A_mat, b_vec, c_sca, sigma2_d, mu_d=None):
    """Calc analytic loo var for fixed sigma2."""
    A2 = A_mat.dot(A_mat)
    out = 2*sigma2_d**2*np.trace(A2)
    out += sigma2_d*(b_vec.T.dot(b_vec))
    if mu_d is not None:
        out += 4*sigma2_d*(b_vec.T.dot(A_mat).dot(mu_d))
        out += 4*sigma2_d*(mu_d.T.dot(A2).dot(mu_d))
    return out

def calc_analytic_moment3(A_mat, b_vec, c_sca, sigma2_d, mu_d=None):
    """Calc analytic loo 3rd central moment for fixed sigma2."""
    A2 = A_mat.dot(A_mat)
    A3 = A2.dot(A_mat)
    out = 8*sigma2_d**3*np.trace(A3)
    out += 6*sigma2_d**2*(b_vec.T.dot(A_mat).dot(b_vec))
    if mu_d is not None:
        out += 24*sigma2_d**2*(b_vec.T.dot(A2).dot(mu_d))
        out += 24*sigma2_d**2*(mu_d.T.dot(A3).dot(mu_d))
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



def get_analytic_res(X_mat, beta_t, sigma2_d, mu_d=None):
    """Analytic result for fixed sigma2 measure."""
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

    #
    A2 = A_mat.dot(A_mat)
    A3 = A2.dot(A_mat)
    b_vec_A = b_vec.T.dot(A_mat)
    if mu_d is not None:
        mu_d_A = mu_d.T.dot(A_mat)

    # mean
    mean = c_sca
    if mu_d is not None:
        mean += b_vec.T.dot(mu_d)
        mean += mu_d_A.dot(mu_d)

    # var
    var = 2*sigma2_d**2*np.trace(A2)
    var += sigma2_d*(b_vec.T.dot(b_vec))
    if mu_d is not None:
        var += 4*sigma2_d*(mu_d_A.dot(b_vec))
        var += 4*sigma2_d*(mu_d_A.dot(mu_d_A.T))

    # moment3
    moment3 = 8*sigma2_d**3*np.trace(A3)
    moment3 += 6*sigma2_d**2*(b_vec_A.dot(b_vec))
    if mu_d is not None:
        moment3 += 24*sigma2_d**2*(b_vec_A.dot(mu_d_A.T))
        moment3 += 24*sigma2_d**2*(mu_d_A.dot(A_mat.dot(mu_d_A.T)))

    # coefvar
    coefvar = np.sqrt(var)/mean

    # skew
    skew = moment3 / np.sqrt(var)**3


    return mean, var, moment3, coefvar, skew
