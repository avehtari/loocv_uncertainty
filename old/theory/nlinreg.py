import numpy as np
from scipy import linalg

n = 8
sigma2_m = 1.7
sigma2_d = 0.9
beta_t = 0.5

seed = 11

rng = np.random.RandomState(seed)

# # rand
# k = 3
# X = rng.rand(n, k)*2-1

# # first dim as intercept, linspace
# k = 2
# X = np.vstack((np.ones(n), np.linspace(-1,1,n))).T

# first dim as intercept, half -1 1
k = 2
X = np.ones(n)
X[rng.choice(n, n//2)] = -1.0
X = np.vstack((np.ones(n), X)).T

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
    -sigma2_d**3/(8*sigma2_m**3)*n**3*(n-3)/((n-1)**2*(n-2)**2
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
