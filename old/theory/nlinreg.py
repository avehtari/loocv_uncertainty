import numpy as np
from scipy import linalg, stats

n = 5
tau2 = 1.7
beta_t = 12.5

n_trial = 3

seed = 11

mu_star = np.zeros(n)
mu_star[1] = 45.9
mu_star[3] = -21.7

Sigma_star = np.eye(n)
Sigma_star[0,0] = 2.1
Sigma_star[2,2] = 0.7
Sigma_star[0,1] = 0.3
Sigma_star[1,0] = Sigma_star[0,1]
Sigma_star[-1,2] = -0.25
Sigma_star[2,-1] = Sigma_star[-1,2]
sigma_star = np.sqrt(np.diag(Sigma_star))
linalg.cholesky(Sigma_star)  # ensure pos-def

rng = np.random.RandomState(seed)


# ===========================================================================
# X

# rand
k = 4
X = rng.rand(n, k)*2-1

# # randn noncenter
# k = 4
# X = rng.randn(n, k) - 123.4

# # first dim as intercept, linspace
# k = 2
# X = np.vstack((np.ones(n), np.linspace(-1,1,n))).T

# # first dim as intercept, half -1 1
# k = 2
# X = np.ones(n)
# X[rng.choice(n, n//2, replace=False)] = -1.0
# X = np.vstack((np.ones(n), X)).T

# # first dim as intercept, rand unif -1 1
# k = 2
# X = np.vstack((np.ones(n), rng.rand(n)*2-1)).T


# ===========================================================================
# trial

# take random betas[0:-1] which should not affect outcome
betas = np.hstack((rng.randn(k-1)*78.9, beta_t))

eps_t = stats.multivariate_normal.rvs(
    mean=mu_star, cov=Sigma_star, size=n_trial)
y_t = X.dot(betas) + eps_t


# ===========================================================================
# elpdhat
# =========================================================================

A_mat_l_a = np.zeros((n,n))
b_vec_l_a = np.zeros(n)
c_sca_l_a = 0.0

A_mat_l_b = np.zeros((n,n))
c_sca_l_b = 0.0

temp_mat_nn = np.zeros((n,n))
temp_vec_n = np.zeros(n)

log2pi05 = 0.5*np.log(2*np.pi)

v_A_s = np.zeros((n,n))
v_B_s = np.zeros((n,n))

for i in range(n):
    X_mi = np.delete(X, i, axis=0)

    # A
    inv_x_i = linalg.solve(
        X_mi[:,:-1].T.dot(X_mi[:,:-1]), X[i,:-1], assume_a='sym')
    v_A = X[:,:-1].dot(inv_x_i)
    v_A_s[i] = v_A
    v_A_i = v_A[i]
    v_A[i] = -1

    betavxt = beta_t*v_A.dot(X[:,-1])
    v1s = (v_A_i+1)*tau2
    v1s2 = 2*v1s

    np.divide(v_A, v1s2, out=temp_vec_n)
    np.multiply(temp_vec_n, v_A[:,None], out=temp_mat_nn)
    A_mat_l_a -= temp_mat_nn

    np.multiply(v_A, betavxt/v1s, out=temp_vec_n)
    b_vec_l_a -= temp_vec_n

    c_sca_l_a -= (betavxt)**2/v1s2 + log2pi05 + 0.5*np.log(v1s)

    # B
    inv_x_i = linalg.solve(
        X_mi.T.dot(X_mi), X[i,:], assume_a='sym')
    v_B = X.dot(inv_x_i)
    v_B_s[i] = v_B
    v_B_i = v_B[i]
    v_B[i] = -1

    v1s = (v_B_i+1)*tau2
    v1s2 = 2*v1s

    np.divide(v_B, v1s2, out=temp_vec_n)
    np.multiply(temp_vec_n, v_B[:,None], out=temp_mat_nn)
    A_mat_l_b -= temp_mat_nn

    c_sca_l_b -= log2pi05 + 0.5*np.log(v1s)


A_mat_l = A_mat_l_a - A_mat_l_b
b_vec_l = b_vec_l_a
c_sca_l = c_sca_l_a - c_sca_l_b

# ==========================================
# calc P
Pt_a = np.zeros((n, n))
Pt_b = np.zeros((n, n))
for a in range(n):
    X_ma = np.delete(X, a, axis=0)
    XXinvX_a = linalg.solve(
        X_ma[:,:-1].T.dot(X_ma[:,:-1]),
        X[a,:-1],
        assume_a='sym'
    )
    sXX_a = np.sqrt(X[a,:-1].dot(XXinvX_a) + 1)
    XXinvX_b = linalg.solve(
        X_ma[:,:].T.dot(X_ma[:,:]),
        X[a,:],
        assume_a='sym'
    )
    sXX_b = np.sqrt(X[a,:].dot(XXinvX_b) + 1)
    for b in range(n):
        if a == b:
            # diag
            Pt_a[a,a] = -1.0/sXX_a
            Pt_b[a,a] = -1.0/sXX_b
        else:
            # off-diag
            Pt_a[a,b] = X[b,:-1].dot(XXinvX_a)/sXX_a
            Pt_b[a,b] = X[b,:].dot(XXinvX_b)/sXX_b

# ==========================================
if False:
    # check P

    # A
    -1/(2*tau2)*(Pt_a.T.dot(Pt_a)-Pt_b.T.dot(Pt_b))
    # ok

    # b
    -beta_t/tau2 * Pt_a.T.dot(Pt_a).dot(X[:,-1])
    # ok

    # c
    (
        -(beta_t**2)/(2*tau2)*X[:,-1].dot(Pt_a.T.dot(Pt_a).dot(X[:,-1]))
        +np.log(np.prod(np.diag(Pt_a)/np.diag(Pt_b)))
    )
    # ok

# ==========================================
if False:
    # check P rows square = 1

    (Pt_a**2).sum(axis=1)
    (Pt_b**2).sum(axis=1)
    # ok

# ==========================================
if False:
    # check P.T.P
    a = 1
    b = 2
    i_noab = list(range(n))
    i_noab.remove(a)
    i_noab.remove(b)

    summand = 0
    summand -= v_A_s[a,b] / (v_A_s[a,a]+1)
    summand -= v_A_s[b,a] / (v_A_s[b,b]+1)
    for i in i_noab:
        summand += v_A_s[i,a]*v_A_s[i,b] / (v_A_s[i,i]+1)
    # ok

    a = 1
    i_noa = list(range(n))
    i_noa.remove(a)

    summand = 0
    summand += 1 / (v_A_s[a,a]+1)
    for i in i_noa:
        summand += v_A_s[i,a]**2 / (v_A_s[i,i]+1)
    # ok

# =========================================
if False:
    # test formula for A_aa
    a = 1
    i_noa = list(range(n))
    i_noa.remove(a)
    A_aa = -1/(2*tau2)*(
        np.sum(
              v_A_s[i_noa,a]**2/(np.diag(v_A_s)[i_noa]+1)
            - v_B_s[i_noa,a]**2/(np.diag(v_B_s)[i_noa]+1)
        )
        + 1/(v_A_s[a,a]+1)
        - 1/(v_B_s[a,a]+1)
    )
    # ok

# =========================================
if False:
    # test formula for A_ab
    a = 1
    b = 2
    i_noab = list(range(n))
    i_noab.remove(a)
    i_noab.remove(b)
    A_ab = -1/(2*tau2)*(
        np.sum(
              v_A_s[i_noab,a]*v_A_s[i_noab,b]/(np.diag(v_A_s)[i_noab]+1)
            - v_B_s[i_noab,a]*v_B_s[i_noab,b]/(np.diag(v_B_s)[i_noab]+1)
        )
        - v_A_s[a,b]/(v_A_s[a,a]+1)
        - v_A_s[b,a]/(v_A_s[b,b]+1)
        + v_B_s[a,b]/(v_B_s[a,a]+1)
        + v_B_s[b,a]/(v_B_s[b,b]+1)
    )
    # ok

# =========================================
if False:
    # test formula v_B_ab for one covariate case
    x = X[:,1]
    a = 1
    b = 2
    i_noa = list(range(n))
    i_noa.remove(a)
    v_B_ab = (
        ((x[i_noa]-x[a]-x[b]).dot(x[i_noa]) + (n-1)*x[a]*x[b])
        / (x[i_noa].dot((n-1)*np.eye(n-1)-1).dot(x[i_noa]))
    )
    # ok

# =========================================
if False:
    # test one cov formulas

    # assume Sigma_star = sigma_star * eye
    sigma_star_sca = sigma_star[0]

    # trace(A)
    np.trace(A_mat_l) # == 0
    # ok

    # trace(A^2)
    np.trace(A_mat_l.dot(A_mat_l))
    # ==
    sigma_star_sca**4/(4*tau2**2)*n**2/((n-1)*(n-2))
    # ok

    # trace(A^3)
    np.trace(A_mat_l.dot(A_mat_l).dot(A_mat_l))
    # ==
    -sigma_star_sca**6/(8*tau2**3)*n**3*(n-3)/((n-1)**2*(n-2)**2)
    # ok

    # b^T b
    b_vec_l.dot(b_vec_l)
    # ==
    beta_t**2*sigma_star_sca**2/tau2**2*n**3/(n-1)**2
    # ok

    # b^T A b
    b_vec_l.dot(A_mat_l).dot(b_vec_l)
    # ==
    -beta_t**2*sigma_star_sca**4/(2*tau2**3)*n**4/(n-1)**3
    # ok


# =========================================================================
# elpdhat for trials
loos = np.einsum('ti,ij,jt->t', eps_t, A_mat_l, eps_t.T)
loos += eps_t.dot(b_vec_l)
loos += c_sca_l

# =========================================================================
# calc reference elpdhat by hand

loos_i_a_target = np.zeros((n_trial, n))
loos_i_b_target = np.zeros((n_trial, n))
for i in range(n):
    X_mi = np.delete(X, i, axis=0)
    y_mi = np.delete(y_t, i, axis=1)
    # A
    betahat = (
        y_mi.dot(linalg.solve(
            X_mi[:,:-1].T.dot(X_mi[:,:-1]),
            X_mi[:,:-1].T,
            assume_a='sym'
        ).T)
    )
    mu_pred = betahat.dot(X[i,:-1].T)
    sigma2_pred = tau2*(
        1 + X[i,:-1].dot(
            linalg.solve(
                X_mi[:,:-1].T.dot(X_mi[:,:-1]),
                X[i,:-1],
                assume_a='sym'
            )
        )
    )
    loos_i_a_target[:, i] = stats.norm.logpdf(
        y_t[:,i], loc=mu_pred, scale=np.sqrt(sigma2_pred))
    # B
    betahat = (
        y_mi.dot(linalg.solve(
            X_mi[:,:].T.dot(X_mi[:,:]),
            X_mi.T,
            assume_a='sym'
        ).T)
    )
    mu_pred = betahat.dot(X[i,:].T)
    sigma2_pred = tau2*(
        1 + X[i,:].dot(
            linalg.solve(
                X_mi[:,:].T.dot(X_mi[:,:]),
                X[i,:],
                assume_a='sym'
            )
        )
    )
    loos_i_b_target[:, i] = stats.norm.logpdf(
        y_t[:,i], loc=mu_pred, scale=np.sqrt(sigma2_pred))
loos_target = loos_i_a_target.sum(axis=1) - loos_i_b_target.sum(axis=1)
# ok, match with loos, regardles of betas




# ============================================================================
# elpds
# ============================================================================

P_a = X[:,:-1].dot(
    linalg.solve(X[:,:-1].T.dot(X[:,:-1]), X[:,:-1].T, assume_a='sym'))
P_b = X.dot(
    linalg.solve(X.T.dot(X), X.T, assume_a='sym'))

D_a = np.diag(1.0/(np.diag(P_a) + 1.0))
D_b = np.diag(1.0/(np.diag(P_b) + 1.0))

temp_vec = beta_t*(P_a-np.eye(n)).dot(X[:,-1]) - mu_star
A_e_mat_A = -1/(2*tau2) * P_a.dot(D_a).dot(P_a)
b_e_vec_A = -1/(tau2) * P_a.dot(D_a).dot(temp_vec)
c_e_sca_A = (
    -1/(2*tau2) * (
        temp_vec.dot(D_a).dot(temp_vec)
        +sigma_star.dot(D_a).dot(sigma_star)
    )
    - n/2*np.log(2*np.pi*tau2)
    +0.5*(np.sum(np.log(np.diag(D_a))))
    # + 0.5*np.log(np.prod(np.diag(D_a)))
)

A_e_mat_B = -1/(2*tau2) * P_b.dot(D_b).dot(P_b)
b_e_vec_B = 1/(tau2) * P_b.dot(D_b).dot(mu_star)
c_e_sca_B = (
    -1/(2*tau2) * (
        mu_star.dot(D_b).dot(mu_star)
        +sigma_star.dot(D_b).dot(sigma_star)
    )
    - n/2*np.log(2*np.pi*tau2)
    + 0.5*(np.sum(np.log(np.diag(D_b))))
    # + 0.5*np.log(np.prod(np.diag(D_b)))
)

temp_vec = beta_t*(P_a-np.eye(n)).dot(X[:,-1]) - mu_star
A_e_mat_D = -1/(2*tau2) * (
    P_a.dot(D_a).dot(P_a) - P_b.dot(D_b).dot(P_b))
b_e_vec_D = -1/(tau2) * (
    P_a.dot(D_a).dot(temp_vec)
    + P_b.dot(D_b).dot(mu_star)
)
c_e_sca_D = (
    -1/(2*tau2) * (
        temp_vec.dot(D_a).dot(temp_vec)
        - mu_star.dot(D_b).dot(mu_star)
        +sigma_star.dot(D_a - D_b).dot(sigma_star)
    )
    +0.5*(np.sum(np.log(np.diag(D_a))) - np.sum(np.log(np.diag(D_b))))
    #+0.5*np.log(np.prod(np.diag(D_a)/np.diag(D_b)))
)

elpds_A = np.einsum('ti,ij,jt->t', eps_t, A_e_mat_A, eps_t.T)
elpds_A += eps_t.dot(b_e_vec_A)
elpds_A += c_e_sca_A

elpds_B = np.einsum('ti,ij,jt->t', eps_t, A_e_mat_B, eps_t.T)
elpds_B += eps_t.dot(b_e_vec_B)
elpds_B += c_e_sca_B

elpds_D = np.einsum('ti,ij,jt->t', eps_t, A_e_mat_D, eps_t.T)
elpds_D += eps_t.dot(b_e_vec_D)
elpds_D += c_e_sca_D


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
        X[:,:-1].T.dot(X[:,:-1]),
        X[:,:-1].T,
        assume_a='sym'
    ).T)
)
betahat_B = (
    y_t.dot(linalg.solve(
        X.T.dot(X),
        X.T,
        assume_a='sym'
    ).T)
)

for i in range(n):

    mu_true = mu_star[i] + X[i,:].dot(betas)
    sigma_true = sigma_star[i]
    samp_true = rng.normal(loc=mu_true, scale=sigma_true, size=samp_n)

    # A
    mu_pred = betahat_A.dot(X[i,:-1].T)
    sigma2_pred = tau2*(
        1 + X[i,:-1].dot(
            linalg.solve(
                X[:,:-1].T.dot(X[:,:-1]),
                X[i,:-1],
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
    mu_pred = betahat_B.dot(X[i,:].T)
    sigma2_pred = tau2*(
        1 + X[i,:].dot(
            linalg.solve(
                X.T.dot(X),
                X[i,:],
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

A_mat_err = 1/(2*tau2) * (
    Pt_a.T.dot(Pt_a) - P_a.dot(D_a).dot(P_a)
    - Pt_b.T.dot(Pt_b) + P_b.dot(D_b).dot(P_b)
)
b_vec_err = 1/(tau2) * (
    beta_t*(
        Pt_a.T.dot(Pt_a)
        - P_a.dot(D_a).dot(P_a-np.eye(n))
    ).dot(X[:,-1])
    + (P_a.dot(D_a) - P_b.dot(D_b)).dot(mu_star)
)
c_sca_err = (
    1/(2*tau2) * (
        beta_t**2*X[:,-1].T.dot(
            Pt_a.T.dot(Pt_a) - (P_a-np.eye(n)).dot(D_a).dot(P_a-np.eye(n))
        ).dot(X[:,-1])
        +2*beta_t*X[:,-1].T.dot(P_a-np.eye(n)).dot(D_a).dot(mu_star)
        - mu_star.dot(D_a - D_b).dot(mu_star)
        - sigma_star.dot(D_a - D_b).dot(sigma_star)
    ) +
    (
        + 0.5*(np.sum(np.log(np.diag(D_a))) - np.sum(np.log(np.diag(D_b))))
        - (np.sum(np.log(-np.diag(Pt_a))) - np.sum(np.log(-np.diag(Pt_b))))
    )
    # +0.5*np.log(np.prod(np.diag(D_a)*np.diag(Pt_b**2)/(np.diag(D_b)*np.diag(Pt_a**2))))
)
