import numpy as np
from scipy import linalg, stats

n = 6
tau2 = 1.7

n_trial = 4000

seed = 11
rng = np.random.RandomState(seed)

# ===========================================================================

s_star = 1.37
m_star = 0.0
# m_star = 12.3
mu_star = np.zeros(n)
mu_star[0] = m_star
Sigma_star = s_star**2*np.eye(n)
Sigma_star_12 = s_star*np.eye(n)


# mu_star = np.zeros(n)
# mu_star[1] = 45.9
# mu_star[3] = -21.7
#
# Sigma_star = np.eye(n)
# Sigma_star[0,0] = 2.1
# Sigma_star[2,2] = 0.7
# Sigma_star[0,1] = 0.3
# Sigma_star[1,0] = Sigma_star[0,1]
# Sigma_star[-1,2] = -0.25
# Sigma_star[2,-1] = Sigma_star[-1,2]
# Sigma_star_12 = linalg.sqrtm(Sigma_star)

sigma_star = np.sqrt(np.diag(Sigma_star))
linalg.cholesky(Sigma_star)  # ensure pos-def


# ===========================================================================
# X and model setting

# rand
# -----------------
# k = 3
# k_a = np.array([0, 1])
# k_b = np.array([0, 1, 2])
# X = rng.rand(n, k)*2-1


# randn noncenter
# -----------------
k = 4
k_a = np.array([0, 1])
k_b = np.array([0, 2, 3])
X = rng.randn(n, k) - 123.4


# first dim as intercept, linspace
# -----------------
# k = 2
# k_a = np.array([0])
# k_b = np.array([0, 1])
# X = np.vstack((np.ones(n), np.linspace(-1,1,n))).T


# first dim as intercept, half -1 1
# -----------------
# k = 2
# k_a = np.array([0])
# k_b = np.array([0, 1])
# X = np.ones(n)
# X[rng.choice(n, n//2, replace=False)] = -1.0
# X = np.vstack((np.ones(n), X)).T
# x = X[:,1]
# xx = (x[:,None] * x[None,:])


# first dim as intercept, rand unif -1 1
# -----------------
# k = 2
# k_a = np.array([0])
# k_b = np.array([0, 1])
# X = np.vstack((np.ones(n), rng.rand(n)*2-1)).T


# ===========================================================================

k_ma = np.array(sorted(set(range(k)) - set(k_a)))
k_mb = np.array(sorted(set(range(k)) - set(k_b)))

beta = rng.randn(k)


# ===========================================================================
# trial

eps_t = stats.multivariate_normal.rvs(
    mean=mu_star, cov=Sigma_star, size=n_trial)
y_t = X.dot(beta) + eps_t

yhat_ma = X[:,k_ma].dot(beta[k_ma]) if k_ma.size else np.zeros(n)
yhat_mb = X[:,k_mb].dot(beta[k_mb]) if k_mb.size else np.zeros(n)

# ===========================================================================
# elpdhat
# =========================================================================

# ==========================================
# calc P and D
Pt_a = np.zeros((n, n))
Pt_b = np.zeros((n, n))
Dt_a = np.zeros((n, n))
Dt_b = np.zeros((n, n))
for i in range(n):
    X_mi = np.delete(X, i, axis=0)
    XXinvX_a = linalg.solve(
        X_mi[:,k_a].T.dot(X_mi[:,k_a]),
        X[i,k_a],
        assume_a='sym'
    )
    ixx_a = 1/(X[i,k_a].dot(XXinvX_a) + 1)
    XXinvX_b = linalg.solve(
        X_mi[:,k_b].T.dot(X_mi[:,k_b]),
        X[i,k_b],
        assume_a='sym'
    )
    ixx_b = 1/(X[i,k_b].dot(XXinvX_b) + 1)
    for j in range(n):
        if i == j:
            # diag
            Pt_a[i,i] = -1.0
            Pt_b[i,i] = -1.0
            Dt_a[i,i] = ixx_a
            Dt_b[i,i] = ixx_b
        else:
            # off-diag
            Pt_a[i,j] = X[j,k_a].dot(XXinvX_a)
            Pt_b[i,j] = X[j,k_b].dot(XXinvX_b)

Pt_D_Pt_a = Pt_a.T.dot(Dt_a).dot(Pt_a)
Pt_D_Pt_b = Pt_b.T.dot(Dt_b).dot(Pt_b)

A_loo_a_1 = -0.5*Pt_D_Pt_a
A_loo_b_1 = -0.5*Pt_D_Pt_b
B_loo_a_1 = -Pt_D_Pt_a
B_loo_b_1 = -Pt_D_Pt_b
C_loo_a_1 = -0.5*Pt_D_Pt_a
C_loo_b_1 = -0.5*Pt_D_Pt_b
c_loo_a_4 = 0.5*np.sum(np.log(np.diag(Dt_a))) - n/2*np.log(2*np.pi*tau2)
c_loo_b_4 = 0.5*np.sum(np.log(np.diag(Dt_b))) - n/2*np.log(2*np.pi*tau2)

A_loo_1 = -0.5*(Pt_D_Pt_a - Pt_D_Pt_b)
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


# =========================================================================
# elpdhat for trials
loos = np.einsum('ti,ij,jt->t', eps_t, A_loo, eps_t.T)
loos += eps_t.dot(b_loo)
loos += c_loo

# =========================================================================
# calc reference loos by hand

loos_i_a_target = np.zeros((n_trial, n))
loos_i_b_target = np.zeros((n_trial, n))
for i in range(n):
    X_mi = np.delete(X, i, axis=0)
    y_mi = np.delete(y_t, i, axis=1)
    # A
    betahat = (
        y_mi.dot(linalg.solve(
            X_mi[:,k_a].T.dot(X_mi[:,k_a]),
            X_mi[:,k_a].T,
            assume_a='sym'
        ).T)
    )
    mu_pred = betahat.dot(X[i,k_a].T)
    sigma2_pred = tau2*(
        1 + X[i,k_a].dot(
            linalg.solve(
                X_mi[:,k_a].T.dot(X_mi[:,k_a]),
                X[i,k_a],
                assume_a='sym'
            )
        )
    )
    loos_i_a_target[:, i] = stats.norm.logpdf(
        y_t[:,i], loc=mu_pred, scale=np.sqrt(sigma2_pred))
    # B
    betahat = (
        y_mi.dot(linalg.solve(
            X_mi[:,k_b].T.dot(X_mi[:,k_b]),
            X_mi[:,k_b].T,
            assume_a='sym'
        ).T)
    )
    mu_pred = betahat.dot(X[i,k_b].T)
    sigma2_pred = tau2*(
        1 + X[i,k_b].dot(
            linalg.solve(
                X_mi[:,k_b].T.dot(X_mi[:,k_b]),
                X[i,k_b],
                assume_a='sym'
            )
        )
    )
    loos_i_b_target[:, i] = stats.norm.logpdf(
        y_t[:,i], loc=mu_pred, scale=np.sqrt(sigma2_pred))
loos_target = loos_i_a_target.sum(axis=1) - loos_i_b_target.sum(axis=1)
# ok, match with loos


# ============================================================================
# elpds
# ============================================================================

P_a = X[:,k_a].dot(
    linalg.solve(X[:,k_a].T.dot(X[:,k_a]), X[:,k_a].T, assume_a='sym'))
P_b = X[:,k_b].dot(
    linalg.solve(X[:,k_b].T.dot(X[:,k_b]), X[:,k_b].T, assume_a='sym'))

D_a = np.diag(1.0/(np.diag(P_a) + 1.0))
D_b = np.diag(1.0/(np.diag(P_b) + 1.0))

A_elpd_a_1 = -0.5*P_a.dot(D_a).dot(P_a)
A_elpd_b_1 = -0.5*P_b.dot(D_b).dot(P_b)
B_elpd_a_1 = -P_a.dot(D_a).dot(P_a - np.eye(n))
B_elpd_b_1 = -P_b.dot(D_b).dot(P_b - np.eye(n))
B_elpd_a_2 = P_a.dot(D_a)
B_elpd_b_2 = P_b.dot(D_b)
C_elpd_a_1 = -0.5*(P_a - np.eye(n)).dot(D_a).dot(P_a - np.eye(n))
C_elpd_b_1 = -0.5*(P_b - np.eye(n)).dot(D_b).dot(P_b - np.eye(n))
C_elpd_a_2 = (P_a - np.eye(n)).dot(D_a)
C_elpd_b_2 = (P_b - np.eye(n)).dot(D_b)
C_elpd_a_3 = -0.5*D_a
C_elpd_b_3 = -0.5*D_b
c_elpd_a_4 = 0.5*np.sum(np.log(np.diag(D_a))) - n/2*np.log(2*np.pi*tau2)
c_elpd_b_4 = 0.5*np.sum(np.log(np.diag(D_b))) - n/2*np.log(2*np.pi*tau2)

A_elpd_a = 1/tau2 * A_elpd_a_1
b_elpd_a = 1/tau2 * (B_elpd_a_1.dot(yhat_ma) + B_elpd_a_2.dot(mu_star))
c_elpd_a =(
    1/tau2 * (
        yhat_ma.dot(C_elpd_a_1).dot(yhat_ma)
        + yhat_ma.dot(C_elpd_a_2).dot(mu_star)
        + mu_star.dot(C_elpd_a_3).dot(mu_star)
        + sigma_star.dot(C_elpd_a_3).dot(sigma_star)
    )
    + c_elpd_a_4
)

A_elpd_b = 1/tau2 * A_elpd_b_1
b_elpd_b = 1/tau2 * (B_elpd_b_1.dot(yhat_mb) + B_elpd_b_2.dot(mu_star))
c_elpd_b =(
    1/tau2 * (
        yhat_mb.dot(C_elpd_b_1).dot(yhat_mb)
        + yhat_mb.dot(C_elpd_b_2).dot(mu_star)
        + mu_star.dot(C_elpd_b_3).dot(mu_star)
        + sigma_star.dot(C_elpd_b_3).dot(sigma_star)
    )
    + c_elpd_b_4
)

A_elpd_1 = -0.5*(P_a.dot(D_a).dot(P_a) - P_b.dot(D_b).dot(P_b))
B_elpd_2 = P_a.dot(D_a) - P_b.dot(D_b)
C_elpd_3 = -0.5*(D_a - D_b)
c_elpd_4 = 0.5*(np.sum(np.log(np.diag(D_a))) - np.sum(np.log(np.diag(D_b))))

A_elpd = 1/tau2 * A_elpd_1
b_elpd = 1/tau2 * (
    B_elpd_a_1.dot(yhat_ma) - B_elpd_b_1.dot(yhat_mb) + B_elpd_2.dot(mu_star))
c_elpd =(
    1/tau2 * (
        yhat_ma.dot(C_elpd_a_1).dot(yhat_ma)
        - yhat_mb.dot(C_elpd_b_1).dot(yhat_mb)
        + yhat_ma.dot(C_elpd_a_2).dot(mu_star)
        - yhat_mb.dot(C_elpd_b_2).dot(mu_star)
        + mu_star.dot(C_elpd_3).dot(mu_star)
        + sigma_star.dot(C_elpd_3).dot(sigma_star)
    )
    + c_elpd_4
)

elpds_a = np.einsum('ti,ij,jt->t', eps_t, A_elpd_a, eps_t.T)
elpds_a += eps_t.dot(b_elpd_a)
elpds_a += c_elpd_a

elpds_b = np.einsum('ti,ij,jt->t', eps_t, A_elpd_b, eps_t.T)
elpds_b += eps_t.dot(b_elpd_b)
elpds_b += c_elpd_b

elpds = np.einsum('ti,ij,jt->t', eps_t, A_elpd, eps_t.T)
elpds += eps_t.dot(b_elpd)
elpds += c_elpd


# =========================================================================
# calc reference elpd s by hand

# analytic
elpds_i_a_target = np.zeros((n_trial, n))
elpds_i_b_target = np.zeros((n_trial, n))

# sampled
samp_n = 10000
elpds_i_a_samp_target = np.zeros((n_trial, n))
elpds_i_b_samp_target = np.zeros((n_trial, n))


betahat_A = (
    y_t.dot(linalg.solve(
        X[:,k_a].T.dot(X[:,k_a]),
        X[:,k_a].T,
        assume_a='sym'
    ).T)
)
betahat_B = (
    y_t.dot(linalg.solve(
        X[:,k_b].T.dot(X[:,k_b]),
        X[:,k_b].T,
        assume_a='sym'
    ).T)
)

for i in range(n):

    mu_true = mu_star[i] + X[i,:].dot(beta)
    sigma_true = sigma_star[i]
    samp_true = rng.normal(loc=mu_true, scale=sigma_true, size=samp_n)

    # A
    mu_pred = betahat_A.dot(X[i,k_a].T)
    sigma2_pred = tau2*(
        1 + X[i,k_a].dot(
            linalg.solve(
                X[:,k_a].T.dot(X[:,k_a]),
                X[i,k_a],
                assume_a='sym'
            )
        )
    )
    elpds_i_a_target[:, i] = (
        -0.5*((mu_pred-mu_true)**2 + sigma_true**2)/sigma2_pred
        -0.5*np.log(2*np.pi*sigma2_pred)
    )
    elpds_i_a_samp_target[:, i] = stats.norm.logpdf(
        samp_true[None,:],
        loc=mu_pred[:,None],
        scale=np.sqrt(sigma2_pred)
    ).mean(axis=-1)

    # B
    mu_pred = betahat_B.dot(X[i,k_b].T)
    sigma2_pred = tau2*(
        1 + X[i,k_b].dot(
            linalg.solve(
                X[:,k_b].T.dot(X[:,k_b]),
                X[i,k_b],
                assume_a='sym'
            )
        )
    )
    elpds_i_b_target[:, i] = (
        -0.5*((mu_pred-mu_true)**2 + sigma_true**2)/sigma2_pred
        -0.5*np.log(2*np.pi*sigma2_pred)
    )
    elpds_i_b_samp_target[:, i] = stats.norm.logpdf(
        samp_true[None,:],
        loc=mu_pred[:,None],
        scale=np.sqrt(sigma2_pred)
    ).mean(axis=-1)


elpds_a_target = elpds_i_a_target.sum(axis=1)
elpds_b_target = elpds_i_b_target.sum(axis=1)
elpds_target = elpds_a_target - elpds_b_target

elpds_a_samp_target = elpds_i_a_samp_target.sum(axis=1)
elpds_b_samp_target = elpds_i_b_samp_target.sum(axis=1)
elpds_samp_target = elpds_a_samp_target - elpds_b_samp_target


# ===========================================================================
# error
# ===========================================================================

# Pt_a, Pt_b, P_a, P_b, D_a, D_b

A_err_1 = 0.5*(
    - Pt_D_Pt_a
    + P_a.dot(D_a).dot(P_a)
    + Pt_D_Pt_b
    - P_b.dot(D_b).dot(P_b)
)
B_err_a_1 = -Pt_D_Pt_a + P_a.dot(D_a).dot(P_a - np.eye(n))
B_err_b_1 = -Pt_D_Pt_b + P_b.dot(D_b).dot(P_b - np.eye(n))
C_err_a_1 = 0.5*(
    -Pt_D_Pt_a + (P_a-np.eye(n)).dot(D_a).dot(P_a-np.eye(n)))
C_err_b_1 = 0.5*(
    -Pt_D_Pt_b + (P_b-np.eye(n)).dot(D_b).dot(P_b-np.eye(n)))
c_err_4 = -0.5*(
    np.sum(np.log(np.diag(D_a))) + np.sum(np.log(np.diag(Dt_b)))
    - np.sum(np.log(np.diag(D_b))) - np.sum(np.log(np.diag(Dt_a)))
)

A_err = 1/tau2 * A_err_1
b_err = 1/tau2 * (
    B_err_a_1.dot(yhat_ma) - B_err_b_1.dot(yhat_mb) - B_elpd_2.dot(mu_star))
c_err =(
    1/tau2 * (
        yhat_ma.dot(C_err_a_1).dot(yhat_ma)
        - yhat_mb.dot(C_err_b_1).dot(yhat_mb)
        - yhat_ma.dot(C_elpd_a_2).dot(mu_star)
        + yhat_mb.dot(C_elpd_b_2).dot(mu_star)
        - mu_star.dot(C_elpd_3).dot(mu_star)
        - sigma_star.dot(C_elpd_3).dot(sigma_star)
    )
    + c_err_4
)

errs = np.einsum('ti,ij,jt->t', eps_t, A_err, eps_t.T)
errs += eps_t.dot(b_err)
errs += c_err

# ===========================================================================
# error moments
# ===========================================================================

Ss_A_Ss = Sigma_star_12.dot(A_err).dot(Sigma_star_12)

m1_err = (
    np.trace(Ss_A_Ss)
    + c_err
    + b_err.dot(mu_star)
    + mu_star.dot(A_err).dot(mu_star)
)

m2_err = (
    2*np.trace(Ss_A_Ss.dot(Ss_A_Ss))
    + b_err.dot(Sigma_star).dot(b_err)
    + 4*b_err.dot(Sigma_star).dot(A_err).dot(mu_star)
    + 4*mu_star.dot(A_err).dot(Sigma_star).dot(A_err).dot(mu_star)
)

m3_err = (
    8*np.trace(Ss_A_Ss.dot(Ss_A_Ss).dot(Ss_A_Ss))
    + 6*b_err.dot(Sigma_star).dot(A_err).dot(Sigma_star).dot(b_err)
    + 24*b_err.dot(Sigma_star).dot(A_err).dot(Sigma_star).dot(A_err).dot(mu_star)
    + 24*mu_star.dot(A_err).dot(Sigma_star).dot(A_err).dot(Sigma_star).dot(A_err).dot(mu_star)
)

# from problem_setting import moments_from_a_b_c
# print(moments_from_a_b_c(A_err, b_err, c_err, np.diag(Sigma_star), mu_d=None))

print(np.mean(errs))
print(m1_err)

print(np.std(errs, ddof=1))
print(np.sqrt(m2_err))

print(stats.skew(errs, bias=False))
print(m3_err/(m2_err)**(3/2))

# ===========================================================================
# outlier effect orders
# ===========================================================================
if False:
    Q_m1 = 1/tau2*(
        A_err_1
        -B_elpd_2
        -C_elpd_3
    )
    q_m1 = 1/tau2*(
        (B_err_a_1 - C_elpd_a_2).dot(yhat_ma)
        -(B_err_b_1 - C_elpd_b_2).dot(yhat_mb)
    )

    Q_m2 = 1/tau2**2*(
        4*A_err_1.dot(Sigma_star).dot(A_err_1)
        -4*A_err_1.dot(Sigma_star).dot(B_elpd_2)
        + B_elpd_2.T.dot(Sigma_star).dot(B_elpd_2)
    )

    Q_m3 = 1/tau2**3*(
        24*A_err_1.dot(Sigma_star).dot(A_err_1).dot(Sigma_star).dot(A_err_1)
        -24*A_err_1.dot(Sigma_star).dot(A_err_1).dot(Sigma_star).dot(B_elpd_2)
        + 6*B_elpd_2.T.dot(Sigma_star).dot(A_err_1).dot(Sigma_star).dot(B_elpd_2)
    )

    # zero for Q_m1
    ia = 0
    ib = 2
    mu_a = 1
    mu_b = (-(Q_m1[ib,ia]+Q_m1[ia,ib]) + np.sqrt(
            (Q_m1[ib,ia]+Q_m1[ia,ib])**2 - 4*Q_m1[ia,ia]*Q_m1[ib,ib]
        ))/(2*Q_m1[ib,ib])
    mu_cur = np.zeros(n)
    mu_cur[ia] = mu_a
    mu_cur[ib] = mu_b

    # zero for Q_m2
    Q_m2_s = (Q_m2 + Q_m2.T)/2
    l, S = linalg.eigh(Q_m2_s)
    print(l)
    mu_cur2 = np.zeros(n)
    mu_cur2[0] = 1.0
    mu_cur = linalg.solve(S.T, mu_cur2)
    print(mu_cur.dot(Q_m2).dot(mu_cur))
    print(mu_cur.dot(Q_m3).dot(mu_cur))


# # temp
# m1_test = np.zeros(17)
# for i, mu_r in enumerate(np.linspace(-100, 100, 17)):
#     mu_cur = mu_r*np.ones(n)
#     A_err_cur = 1/tau2 * A_err_1
#     b_err_cur = 1/tau2 * (
#         B_err_a_1.dot(yhat_ma) - B_err_b_1.dot(yhat_mb) - B_elpd_2.dot(mu_cur))
#     c_err_cur =(
#         1/tau2 * (
#             yhat_ma.dot(C_err_a_1).dot(yhat_ma)
#             - yhat_mb.dot(C_err_b_1).dot(yhat_mb)
#             - yhat_ma.dot(C_elpd_a_2).dot(mu_cur)
#             + yhat_mb.dot(C_elpd_b_2).dot(mu_cur)
#             - mu_cur.dot(C_elpd_3).dot(mu_cur)
#             - sigma_star.dot(C_elpd_3).dot(sigma_star)
#         )
#         + c_err_4
#     )
#
#     m1_test[i] = (
#         np.trace(Sigma_star_12.dot(A_err_cur).dot(Sigma_star_12))
#         + c_err_cur
#         + b_err_cur.dot(mu_cur)
#         + mu_cur.dot(A_err_cur).dot(mu_cur)
#     )
