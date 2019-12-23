import numpy as np
from scipy import linalg, stats

n = 34
sigma2_m = 1.7
sigma2_d = 0.9
beta_t = 12.5

n_trial = 4000
n_bb = 600

seed = 11

rng = np.random.RandomState(seed)

# rand
# k = 4
# X = rng.rand(n, k)*2-1

# randn noncenter
k = 4
X = rng.randn(n, k) - 123.4

# # first dim as intercept, linspace
# k = 2
# X = np.vstack((np.ones(n), np.linspace(-1,1,n))).T

# # first dim as intercept, half -1 1
# k = 2
# X = np.ones(n)
# X[rng.choice(n, n//2)] = -1.0
# X = np.vstack((np.ones(n), X)).T

# # first dim as intercept, rand unif -1 1
# k = 2
# X = np.vstack((np.ones(n), rng.rand(n)*2-1)).T

A_mat_A = np.zeros((n,n))
b_vec_A = np.zeros(n)
c_sca_A = 0.0

A_mat_B = np.zeros((n,n))
c_sca_B = 0.0

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
    v1s = (v_A_i+1)*sigma2_m
    v1s2 = 2*v1s

    np.divide(v_A, v1s2, out=temp_vec_n)
    np.multiply(temp_vec_n, v_A[:,None], out=temp_mat_nn)
    A_mat_A -= temp_mat_nn

    np.multiply(v_A, betavxt/v1s, out=temp_vec_n)
    b_vec_A -= temp_vec_n

    c_sca_A -= (betavxt)**2/v1s2 + log2pi05 + 0.5*np.log(v1s)

    # B
    inv_x_i = linalg.solve(
        X_mi.T.dot(X_mi), X[i,:], assume_a='sym')
    v_B = X.dot(inv_x_i)
    v_B_s[i] = v_B
    v_B_i = v_B[i]
    v_B[i] = -1

    v1s = (v_B_i+1)*sigma2_m
    v1s2 = 2*v1s

    np.divide(v_B, v1s2, out=temp_vec_n)
    np.multiply(temp_vec_n, v_B[:,None], out=temp_mat_nn)
    A_mat_B -= temp_mat_nn

    c_sca_B -= log2pi05 + 0.5*np.log(v1s)


A_mat = sigma2_d*(A_mat_A - A_mat_B)
b_vec = np.sqrt(sigma2_d)*b_vec_A
c_sca = c_sca_A - c_sca_B

# =========================================
if False:
    Pa = np.zeros((n,n))
    Pb = np.zeros((n,n))
    for a in range(n):
        i_noa = list(range(n))
        i_noa.remove(a)
        part_a = linalg.solve(
            X[i_noa,:-1].T.dot(X[i_noa,:-1]),
            X[a,:-1].T,
            assume_a='sym'
        )
        part_b = linalg.solve(
            X[i_noa,:].T.dot(X[i_noa,:]),
            X[a,:].T,
            assume_a='sym'
        )
        dA = np.sqrt(X[a,:-1].dot(part_a) + 1)
        dB = np.sqrt(X[a,:].dot(part_b) + 1)
        for b in range(n):
            if a == b:
                Pa[a,b] = -1/dA
                Pb[a,b] = -1/dB
            else:
                Pa[a,b] = X[b,:-1].dot(part_a)/dA
                Pb[a,b] = X[b,:].dot(part_b)/dB

    # A
    -sigma2_d/(2*sigma2_m)*(Pa.T.dot(Pa) - Pb.T.dot(Pb))
    # ok

    # b
    -beta_t*np.sqrt(sigma2_d)/sigma2_m*(Pa.T.dot(Pa).dot(X[:,-1]))
    # ok

    # c
    (
        -beta_t**2/(2*sigma2_m)*(X[:,-1].T.dot(Pa.T.dot(Pa).dot(X[:,-1])))
        + np.log(np.prod(np.diag(Pa)/np.diag(Pb)))
    )
    # ok


# =========================================
if False:
    # test formula for A_aa
    a = 1
    i_noa = list(range(n))
    i_noa.remove(a)
    A_aa = -sigma2_d/(2*sigma2_m)*(
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
    A_ab = -sigma2_d/(2*sigma2_m)*(
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

    # trace(A)
    np.trace(A_mat) # == 0
    # ok

    # trace(A^2)
    np.trace(A_mat.dot(A_mat))
    # ==
    sigma2_d**2/(4*sigma2_m**2)*n**2/((n-1)*(n-2))
    # ok

    # trace(A^3)
    np.trace(A_mat.dot(A_mat).dot(A_mat))
    # ==
    -sigma2_d**3/(8*sigma2_m**3)*n**3*(n-3)/((n-1)**2*(n-2)**2)
    # ok

    # b^T b
    b_vec.dot(b_vec)
    # ==
    beta_t**2*sigma2_d/sigma2_m**2*n**3/(n-1)**2
    # ok

    # b^T A b
    b_vec.dot(A_mat).dot(b_vec)
    # ==
    -beta_t**2*sigma2_d**2/(2*sigma2_m**3)*n**4/(n-1)**3
    # ok


# trial
eps_t = rng.randn(n_trial, n)
loos = np.einsum('ti,ij,jt->t', eps_t, A_mat, eps_t.T)
loos += eps_t.dot(b_vec)
loos += c_sca


# calc reference loos by hand
# take random betas[0:-1] which should not affect outcome
betas = np.hstack((rng.randn(k-1)*78.9, beta_t))
y_t = X.dot(betas) + eps_t*np.sqrt(sigma2_d)
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
    sigma2_pred = sigma2_m*(
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
    sigma2_pred = sigma2_m*(
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



# calc P
Pa = np.zeros((n, n))
Pb = np.zeros((n, n))
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
            Pa[a,a] = -1.0/sXX_a
            Pb[a,a] = -1.0/sXX_b
        else:
            # off-diag
            Pa[a,b] = X[b,:-1].dot(XXinvX_a)/sXX_a
            Pb[a,b] = X[b,:].dot(XXinvX_b)/sXX_b

if False:
    # check P

    # A
    -sigma2_d/(2*sigma2_m)*(Pa.T.dot(Pa)-Pb.T.dot(Pb))
    # ok

    # b
    -beta_t*np.sqrt(sigma2_d)/sigma2_m * Pa.T.dot(Pa).dot(X[:,-1])
    # ok

    # c
    (
        -(beta_t**2)/(2*sigma2_m)*X[:,-1].dot(Pa.T.dot(Pa).dot(X[:,-1]))
        +np.log(np.prod(np.diag(Pa)/np.diag(Pb)))
    )
    # ok

if False:
    # check P rows square = 1

    (Pa**2).sum(axis=1)
    (Pb**2).sum(axis=1)
    # ok

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


def calc_bb_var(X, n_replications):
    """Bayesian bootstrap variance.

    By: https://github.com/lmc2179/bayesian_bootstrap

    """
    samples = []
    weights = np.random.dirichlet([1]*len(X), n_replications)
    for w in weights:
        samples.append(np.dot([x ** 2 for x in X], w) - np.dot(X, w) ** 2)
    return samples


if False:
    # test var
    bb_samp = calc_bb_var(loos, n_bb)
    analytic_val = 2*np.trace(A_mat.dot(A_mat)) + b_vec.dot(b_vec)
    plt.hist(bb_samp)
    plt.axvline(analytic_val, color='red')

A = A_mat
b = b_vec
c = c_sca
