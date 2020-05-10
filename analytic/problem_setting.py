
import numpy as np
from scipy import linalg, stats



def get_analytic_res(X_mat, beta, tau2, idx_a, idx_b, Sigma_d, mu_d=None):
    """Analytic results.

    Parameters
    ----------
    Sigma_d : ndarray, scalar
        scalar -> Sigma_d is the shared data variance
        vector -> Sigma_d is the data variance for each independent obs
        matrix -> Sigma_d is the matrix sqrt of the data covariance matrix

    """

    n_obs, n_dim = X_mat.shape

    if np.isscalar(Sigma_d):
        sigma_d = np.full(n_obs, np.sqrt(Sigma_d))
    else:
        if Sigma_d.ndim == 1:
            sigma_d = np.sqrt(Sigma_d)
        else:
            sigma_d = np.sqrt(np.diag(Sigma_d.dot(Sigma_d)))

    # model covariate missing indices
    idx_ma = np.array(sorted(set(range(n_dim)) - set(idx_a)))
    idx_mb = np.array(sorted(set(range(n_dim)) - set(idx_b)))

    if idx_ma.size:
        yhat_ma = X_mat[:,idx_ma].dot(beta[idx_ma])
    else:
        yhat_ma = np.zeros(n_obs)
    if idx_mb.size:
        yhat_mb = X_mat[:,idx_mb].dot(beta[idx_mb])
    else:
        yhat_mb = np.zeros(n_obs)

    # ----------------------------------------------------
    # loo
    # ----------------------------------------------------

    Pt_a = np.zeros((n_obs, n_obs))
    Pt_b = np.zeros((n_obs, n_obs))
    Dt_a = np.zeros((n_obs, n_obs))
    Dt_b = np.zeros((n_obs, n_obs))
    for i in range(n_obs):
        X_mi = np.delete(X_mat, i, axis=0)
        XXinvX_a = linalg.solve(
            X_mi[:,idx_a].T.dot(X_mi[:,idx_a]),
            X_mat[i,idx_a],
            assume_a='sym'
        )
        Dt_a[i,i] = 1/(X_mat[i,idx_a].dot(XXinvX_a) + 1)
        XXinvX_b = linalg.solve(
            X_mi[:,idx_b].T.dot(X_mi[:,idx_b]),
            X_mat[i,idx_b],
            assume_a='sym'
        )
        Dt_b[i,i] = 1/(X_mat[i,idx_b].dot(XXinvX_b) + 1)
        for j in range(n_obs):
            if i == j:
                # diag
                Pt_a[i,i] = -1.0
                Pt_b[i,i] = -1.0
            else:
                # off-diag
                Pt_a[i,j] = X_mat[j,idx_a].dot(XXinvX_a)
                Pt_b[i,j] = X_mat[j,idx_b].dot(XXinvX_b)

    Pt_Dt_Pt_a = Pt_a.T.dot(Dt_a).dot(Pt_a)
    Pt_Dt_Pt_b = Pt_b.T.dot(Dt_b).dot(Pt_b)

    # A_loo_a_1 = -0.5*Pt_Dt_Pt_a
    # A_loo_b_1 = -0.5*Pt_Dt_Pt_b
    B_loo_a_1 = -Pt_Dt_Pt_a
    B_loo_b_1 = -Pt_Dt_Pt_b
    C_loo_a_1 = -0.5*Pt_Dt_Pt_a
    C_loo_b_1 = -0.5*Pt_Dt_Pt_b
    # c_loo_a_4 = 0.5*np.sum(np.log(np.diag(Dt_a))) - n_obs/2*np.log(2*np.pi*tau2)
    # c_loo_b_4 = 0.5*np.sum(np.log(np.diag(Dt_b))) - n_obs/2*np.log(2*np.pi*tau2)

    A_loo_1 = -0.5*(Pt_Dt_Pt_a - Pt_Dt_Pt_b)
    C_loo_4 = 0.5*np.sum(np.log(np.diag(Dt_a))) - 0.5*np.sum(np.log(np.diag(Dt_b)))

    A_loo = 1/tau2 * A_loo_1
    b_loo = 1/tau2 * (B_loo_a_1.dot(yhat_ma) - B_loo_b_1.dot(yhat_mb))
    c_loo =(
        1/tau2 * (
            yhat_ma.dot(C_loo_a_1).dot(yhat_ma)
            - yhat_mb.dot(C_loo_b_1).dot(yhat_mb)
        )
        + C_loo_4
    )

    # ----------------------------------------------------
    # elpd
    # ----------------------------------------------------

    P_a = X_mat[:,idx_a].dot(
        linalg.solve(X_mat[:,idx_a].T.dot(X_mat[:,idx_a]), X_mat[:,idx_a].T,
        assume_a='sym'))
    P_b = X_mat[:,idx_b].dot(
        linalg.solve(X_mat[:,idx_b].T.dot(X_mat[:,idx_b]), X_mat[:,idx_b].T,
        assume_a='sym'))

    D_a = np.diag(1.0/(np.diag(P_a) + 1.0))
    D_b = np.diag(1.0/(np.diag(P_b) + 1.0))

    # A_elpd_a_1 = -0.5*P_a.dot(D_a).dot(P_a)
    # A_elpd_b_1 = -0.5*P_b.dot(D_b).dot(P_b)
    B_elpd_a_1 = -P_a.dot(D_a).dot(P_a - np.eye(n_obs))
    B_elpd_b_1 = -P_b.dot(D_b).dot(P_b - np.eye(n_obs))
    # B_elpd_a_2 = P_a.dot(D_a)
    # B_elpd_b_2 = P_b.dot(D_b)
    C_elpd_a_1 = -0.5*(P_a - np.eye(n_obs)).dot(D_a).dot(P_a - np.eye(n_obs))
    C_elpd_b_1 = -0.5*(P_b - np.eye(n_obs)).dot(D_b).dot(P_b - np.eye(n_obs))
    C_elpd_a_2 = (P_a - np.eye(n_obs)).dot(D_a)
    C_elpd_b_2 = (P_b - np.eye(n_obs)).dot(D_b)
    # C_elpd_a_3 = -0.5*D_a
    # C_elpd_b_3 = -0.5*D_b
    # c_elpd_a_4 = 0.5*np.sum(np.log(np.diag(D_a))) - n_obs/2*np.log(2*np.pi*tau2)
    # c_elpd_b_4 = 0.5*np.sum(np.log(np.diag(D_b))) - n_obs/2*np.log(2*np.pi*tau2)

    A_elpd_1 = -0.5*(P_a.dot(D_a).dot(P_a) - P_b.dot(D_b).dot(P_b))
    B_elpd_2 = P_a.dot(D_a) - P_b.dot(D_b)
    C_elpd_3 = -0.5*(D_a - D_b)
    c_elpd_4 = 0.5*(np.sum(np.log(np.diag(D_a))) - np.sum(np.log(np.diag(D_b))))

    A_elpd = 1/tau2 * A_elpd_1
    b_elpd = 1/tau2 * (
        B_elpd_a_1.dot(yhat_ma)
        - B_elpd_b_1.dot(yhat_mb)
        + (B_elpd_2.dot(mu_d) if mu_d is not None else 0.0)
    )
    c_elpd =(
        1/tau2 * (
            yhat_ma.dot(C_elpd_a_1).dot(yhat_ma)
            - yhat_mb.dot(C_elpd_b_1).dot(yhat_mb)
            + (yhat_ma.dot(C_elpd_a_2).dot(mu_d) if mu_d is not None else 0.0)
            - (yhat_mb.dot(C_elpd_b_2).dot(mu_d) if mu_d is not None else 0.0)
            + (mu_d.dot(C_elpd_3).dot(mu_d) if mu_d is not None else 0.0)
            + sigma_d.dot(C_elpd_3).dot(sigma_d)
        )
        + c_elpd_4
    )

    # ----------------------------------------------------
    # error
    # ----------------------------------------------------

    A_err_1 = 0.5*(
        - Pt_Dt_Pt_a
        + P_a.dot(D_a).dot(P_a)
        + Pt_Dt_Pt_b
        - P_b.dot(D_b).dot(P_b)
    )
    B_err_a_1 = -Pt_Dt_Pt_a + P_a.dot(D_a).dot(P_a - np.eye(n_obs))
    B_err_b_1 = -Pt_Dt_Pt_b + P_b.dot(D_b).dot(P_b - np.eye(n_obs))
    C_err_a_1 = 0.5*(
        -Pt_Dt_Pt_a + (P_a-np.eye(n_obs)).dot(D_a).dot(P_a-np.eye(n_obs)))
    C_err_b_1 = 0.5*(
        -Pt_Dt_Pt_b + (P_b-np.eye(n_obs)).dot(D_b).dot(P_b-np.eye(n_obs)))
    c_err_4 = -0.5*(
        np.sum(np.log(np.diag(D_a))) + np.sum(np.log(np.diag(Dt_b)))
        - np.sum(np.log(np.diag(D_b))) - np.sum(np.log(np.diag(Dt_a)))
    )

    A_err = 1/tau2 * A_err_1
    b_err = 1/tau2 * (
        B_err_a_1.dot(yhat_ma)
        - B_err_b_1.dot(yhat_mb)
        - (B_elpd_2.dot(mu_d) if mu_d is not None else 0.0)
    )
    c_err =(
        1/tau2 * (
            yhat_ma.dot(C_err_a_1).dot(yhat_ma)
            - yhat_mb.dot(C_err_b_1).dot(yhat_mb)
            - (yhat_ma.dot(C_elpd_a_2).dot(mu_d) if mu_d is not None else 0.0)
            + (yhat_mb.dot(C_elpd_b_2).dot(mu_d) if mu_d is not None else 0.0)
            - (mu_d.dot(C_elpd_3).dot(mu_d) if mu_d is not None else 0.0)
            - sigma_d.dot(C_elpd_3).dot(sigma_d)
        )
        + c_err_4
    )

    # ----------------------------------------------------
    # moments
    # ----------------------------------------------------

    # loo
    mean_loo, var_loo, skew_loo = moments_from_a_b_c(
        A_loo, b_loo, c_loo, Sigma_d, mu_d)
    # elpd
    mean_elpd, var_elpd, skew_elpd = moments_from_a_b_c(
        A_elpd, b_elpd, c_elpd, Sigma_d, mu_d)
    # error
    mean_err, var_err, skew_err = moments_from_a_b_c(
        A_err, b_err, c_err, Sigma_d, mu_d)

    return (
        mean_loo, var_loo, skew_loo,
        mean_elpd, var_elpd, skew_elpd,
        mean_err, var_err, skew_err
    )


def moments_from_a_b_c(A_mat, b_vec, c_sca, Sigma_d, mu_d=None):

    if np.isscalar(Sigma_d):

        # shared data variance
        sigma2_d = Sigma_d

        A2 = A_mat.dot(A_mat)
        A3 = A2.dot(A_mat)
        b_vec_A = b_vec.T.dot(A_mat)
        if mu_d is not None:
            mu_d_A = mu_d.T.dot(A_mat)

        # mean
        mean = sigma2_d*np.trace(A_mat)
        mean += c_sca
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

        # skew
        skew = moment3 / np.sqrt(var)**3

    else:
        if Sigma_d.ndim == 1:
            # indep
            Sigma_d_sqrt = np.diag(np.sqrt(Sigma_d))
            Sigma_d = np.diag(Sigma_d)
        else:
            # matrix square root provided
            Sigma_d_sqrt = Sigma_d
            Sigma_d = Sigma_d_sqrt.dot(Sigma_d_sqrt)

        if mu_d is not None:
            mu_d_A = mu_d.T.dot(A_mat)
            mu_d_A_Sigma = mu_d_A.dot(Sigma_d)
            mu_d_A_Sigma_A = mu_d_A_Sigma.dot(A_mat)


        Sigma_d_sqrt_A_Sigma_d_sqrt = Sigma_d_sqrt.dot(A_mat).dot(Sigma_d_sqrt)
        Sigma_d_sqrt_A_Sigma_d_sqrt_2 = (
            Sigma_d_sqrt_A_Sigma_d_sqrt.dot(Sigma_d_sqrt_A_Sigma_d_sqrt))
        Sigma_d_sqrt_A_Sigma_d_sqrt_3 = (
            Sigma_d_sqrt_A_Sigma_d_sqrt_2.dot(Sigma_d_sqrt_A_Sigma_d_sqrt))

        b_Sigma = b_vec.T.dot(Sigma_d)

        # mean
        mean = np.trace(Sigma_d_sqrt_A_Sigma_d_sqrt)
        mean += c_sca
        if mu_d is not None:
            mean += b_vec.T.dot(mu_d)
            mean += mu_d_A.dot(mu_d)

        # var
        var = 2*np.trace(Sigma_d_sqrt_A_Sigma_d_sqrt_2)
        var += b_Sigma.dot(b_vec)
        if mu_d is not None:
            var += 4*(b_vec.dot(mu_d_A_Sigma.T))
            var += 4*(mu_d_A.dot(mu_d_A_Sigma.T))

        # moment3
        moment3 = 8*np.trace(Sigma_d_sqrt_A_Sigma_d_sqrt_3)
        moment3 += 6*(b_Sigma.dot(A_mat).dot(b_Sigma.T))
        if mu_d is not None:
            moment3 += 24*(b_Sigma.dot(mu_d_A_Sigma_A.T))
            moment3 += 24*(mu_d_A_Sigma.dot(mu_d_A_Sigma_A.T))

        # skew
        skew = moment3 / np.sqrt(var)**3

    return mean, var, skew
