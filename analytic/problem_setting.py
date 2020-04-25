
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

    Pt_a = np.zeros((n_obs, n_obs))
    Pt_b = np.zeros((n_obs, n_obs))
    for i in range(n_obs):
        X_mi = np.delete(X_mat, i, axis=0)
        XXinvX_a = linalg.solve(
            X_mi[:,idx_a].T.dot(X_mi[:,idx_a]),
            X_mat[i,idx_a],
            assume_a='sym'
        )
        sXX_a = np.sqrt(X_mat[i,idx_a].dot(XXinvX_a) + 1)
        XXinvX_b = linalg.solve(
            X_mi[:,idx_b].T.dot(X_mi[:,idx_b]),
            X_mat[i,idx_b],
            assume_a='sym'
        )
        sXX_b = np.sqrt(X_mat[i,idx_b].dot(XXinvX_b) + 1)
        for j in range(n_obs):
            if i == j:
                # diag
                Pt_a[i,i] = -1.0/sXX_a
                Pt_b[i,i] = -1.0/sXX_b
            else:
                # off-diag
                Pt_a[i,j] = X_mat[j,idx_a].dot(XXinvX_a)/sXX_a
                Pt_b[i,j] = X_mat[j,idx_b].dot(XXinvX_b)/sXX_b

    Pt_a_T_Pt_a = Pt_a.T.dot(Pt_a)
    Pt_b_T_Pt_b = Pt_b.T.dot(Pt_b)

    # At_a_1 = -0.5*Pt_a_T_Pt_a
    # At_b_1 = -0.5*Pt_b_T_Pt_b
    Bt_a_1 = -Pt_a_T_Pt_a
    Bt_b_1 = -Pt_b_T_Pt_b
    Ct_a_1 = -0.5*Pt_a_T_Pt_a
    Ct_b_1 = -0.5*Pt_b_T_Pt_b
    # ct_a_4 = np.sum(np.log(-np.diag(Pt_a))) - n_obs/2*np.log(2*np.pi*tau2)
    # ct_b_4 = np.sum(np.log(-np.diag(Pt_b))) - n_obs/2*np.log(2*np.pi*tau2)

    At_1 = -0.5*(Pt_a_T_Pt_a - Pt_b_T_Pt_b)
    Ct_4 = np.sum(np.log(-np.diag(Pt_a))) - np.sum(np.log(-np.diag(Pt_b)))

    At = 1/tau2 * At_1
    bt = 1/tau2 * (Bt_a_1.dot(yhat_ma) - Bt_b_1.dot(yhat_mb))
    ct =(
        1/tau2 * (
            yhat_ma.dot(Ct_a_1).dot(yhat_ma)
            - yhat_mb.dot(Ct_b_1).dot(yhat_mb)
        )
        + Ct_4
    )

    P_a = X_mat[:,idx_a].dot(
        linalg.solve(X_mat[:,idx_a].T.dot(X_mat[:,idx_a]), X_mat[:,idx_a].T,
        assume_a='sym'))
    P_b = X_mat[:,idx_b].dot(
        linalg.solve(X_mat[:,idx_b].T.dot(X_mat[:,idx_b]), X_mat[:,idx_b].T,
        assume_a='sym'))

    D_a = np.diag(1.0/(np.diag(P_a) + 1.0))
    D_b = np.diag(1.0/(np.diag(P_b) + 1.0))

    # Ae_a_1 = -0.5*P_a.dot(D_a).dot(P_a)
    # Ae_b_1 = -0.5*P_b.dot(D_b).dot(P_b)
    # Be_a_1 = -P_a.dot(D_a).dot(P_a - np.eye(n_obs))
    # Be_b_1 = -P_b.dot(D_b).dot(P_b - np.eye(n_obs))
    # Be_a_2 = P_a.dot(D_a)
    # Be_b_2 = P_b.dot(D_b)
    # Ce_a_1 = -0.5*(P_a - np.eye(n_obs)).dot(D_a).dot(P_a - np.eye(n_obs))
    # Ce_b_1 = -0.5*(P_b - np.eye(n_obs)).dot(D_b).dot(P_b - np.eye(n_obs))
    Ce_a_2 = (P_a - np.eye(n_obs)).dot(D_a)
    Ce_b_2 = (P_b - np.eye(n_obs)).dot(D_b)
    # Ce_a_3 = -0.5*D_a
    # Ce_b_3 = -0.5*D_b
    # ce_a_4 = 0.5*np.sum(np.log(np.diag(D_a))) - n_obs/2*np.log(2*np.pi*tau2)
    # ce_b_4 = 0.5*np.sum(np.log(np.diag(D_b))) - n_obs/2*np.log(2*np.pi*tau2)

    # Ae_a = 1/tau2 * Ae_a_1
    # be_a = 1/tau2 * (
    #     Be_a_1.dot(yhat_ma) + (Be_a_2.dot(mu_d) if mu_d is not None else 0.0))
    # ce_a =(
    #     1/tau2 * (
    #         yhat_ma.dot(Ce_a_1).dot(yhat_ma)
    #         + (yhat_ma.dot(Ce_a_2).dot(mu_d) if mu_d is not None else 0.0)
    #         + (mu_d.dot(Ce_a_3).dot(mu_d) if mu_d is not None else 0.0)
    #         + sigma_d.dot(Ce_a_3).dot(sigma_d)
    #     )
    #     + ce_a_4
    # )

    # Ae_b = 1/tau2 * Ae_b_1
    # be_b = 1/tau2 * (
    #     Be_b_1.dot(yhat_mb) + (Be_b_2.dot(mu_d) if mu_d is not None else 0.0))
    # ce_b =(
    #     1/tau2 * (
    #         yhat_mb.dot(Ce_b_1).dot(yhat_mb)
    #         + (yhat_mb.dot(Ce_b_2).dot(mu_d) if mu_d is not None else 0.0)
    #         + (mu_d.dot(Ce_b_3).dot(mu_d) if mu_d is not None else 0.0)
    #         + sigma_d.dot(Ce_b_3).dot(sigma_d)
    #     )
    #     + ce_b_4
    # )

    # Ae_1 = -0.5*(P_a.dot(D_a).dot(P_a) - P_b.dot(D_b).dot(P_b))
    Be_2 = P_a.dot(D_a) - P_b.dot(D_b)
    Ce_3 = -0.5*(D_a - D_b)
    # ce_4 = 0.5*(np.sum(np.log(np.diag(D_a))) - np.sum(np.log(np.diag(D_b))))

    # Ae = 1/tau2 * Ae_1
    # be = 1/tau2 * (
    #     Be_a_1.dot(yhat_ma)
    #     - Be_b_1.dot(yhat_mb)
    #     + (Be_2.dot(mu_d) if mu_d is not None else 0.0)
    # )
    # ce =(
    #     1/tau2 * (
    #         yhat_ma.dot(Ce_a_1).dot(yhat_ma)
    #         - yhat_mb.dot(Ce_b_1).dot(yhat_mb)
    #         + (yhat_ma.dot(Ce_a_2).dot(mu_d) if mu_d is not None else 0.0)
    #         - (yhat_mb.dot(Ce_b_2).dot(mu_d) if mu_d is not None else 0.0)
    #         + (mu_d.dot(Ce_3).dot(mu_d) if mu_d is not None else 0.0)
    #         + sigma_d.dot(Ce_3).dot(sigma_d)
    #     )
    #     + ce_4
    # )

    A_err_1 = 0.5*(
        Pt_a_T_Pt_a
        - P_a.dot(D_a).dot(P_a)
        - Pt_b_T_Pt_b
        + P_b.dot(D_b).dot(P_b)
    )
    B_err_a_1 = Pt_a_T_Pt_a - P_a.dot(D_a).dot(P_a - np.eye(n_obs))
    B_err_b_1 = Pt_b_T_Pt_b - P_b.dot(D_b).dot(P_b - np.eye(n_obs))
    C_err_a_1 = 0.5*(
        Pt_a_T_Pt_a - (P_a-np.eye(n_obs)).dot(D_a).dot(P_a-np.eye(n_obs)))
    C_err_b_1 = 0.5*(
        Pt_b_T_Pt_b - (P_b-np.eye(n_obs)).dot(D_b).dot(P_b-np.eye(n_obs)))
    c_err_4 = 0.5*(
        np.sum(np.log(np.diag(D_a))) + np.sum(np.log(np.diag(Pt_b)**2))
        - np.sum(np.log(np.diag(D_b))) - np.sum(np.log(np.diag(Pt_a)**2))
    )

    A_err = 1/tau2 * A_err_1
    b_err = 1/tau2 * (
        B_err_a_1.dot(yhat_ma)
        - B_err_b_1.dot(yhat_mb)
        + (Be_2.dot(mu_d) if mu_d is not None else 0.0)
    )
    c_err =(
        1/tau2 * (
            yhat_ma.dot(C_err_a_1).dot(yhat_ma)
            - yhat_mb.dot(C_err_b_1).dot(yhat_mb)
            + (yhat_ma.dot(Ce_a_2).dot(mu_d) if mu_d is not None else 0.0)
            - (yhat_mb.dot(Ce_b_2).dot(mu_d) if mu_d is not None else 0.0)
            + (mu_d.dot(Ce_3).dot(mu_d) if mu_d is not None else 0.0)
            + sigma_d.dot(Ce_3).dot(sigma_d)
        )
        + c_err_4
    )

    # loo
    mean_loo, var_loo, skew_loo = moments_from_a_b_c(
        At, bt, ct, Sigma_d, mu_d)
    # error
    mean_err, var_err, skew_err = moments_from_a_b_c(
        A_err, b_err, c_err, Sigma_d, mu_d)

    return mean_loo, var_loo, skew_loo, mean_err, var_err, skew_err




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
            # shared data variance
            Sigma_d = np.diag(Sigma_d)
            Sigma_d_sqrt = np.diag(np.sqrt(Sigma_d))
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
