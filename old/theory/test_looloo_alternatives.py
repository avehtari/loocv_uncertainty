"""Test LOO-LOO approximation."""

import itertools
import numpy as np
from scipy import linalg

import matplotlib.pyplot as plt


# conf
seed = 11
n = 16
p = 2
n_trial = 1000


num_of_pairs = (n**2-n) // 2

LOG2PI = np.log(2*np.pi)
def norm_logpdf(x, mu, sigma2):
    """Univariate normal logpdf."""
    return -0.5*(LOG2PI + np.log(sigma2) + (x-mu)**2/sigma2)


rng = np.random.RandomState(seed=seed)
xs = rng.randn(n_trial, n, p)
ys = rng.randn(n_trial, n)

# pointwise products for each test and data point
Q_ti = xs[:,:,:,None]*xs[:,:,None,:]
r_ti = xs[:,:,:]*ys[:,:,None]
# full posterior for each test
Q_t_full = Q_ti.sum(axis=1)
r_t_full = r_ti.sum(axis=1)

# loos
loos = np.empty((n_trial, n))
# for each test
for t in range(n_trial):
    # for each data point
    for i in range(n):
        Q = Q_t_full[t] - Q_ti[t, i]
        r = r_t_full[t] - r_ti[t, i]
        x = xs[t, i]
        y = ys[t, i]
        xS = linalg.solve(Q, x, assume_a='pos')
        mu_pred = xS.dot(r)
        sigma2_pred = xS.dot(x) + 1
        loos[t, i] = norm_logpdf(y, mu_pred, sigma2_pred)

# loo-loos
looloos_full = np.full((n_trial, n, n), np.nan)
# for each test
for t in range(n_trial):
    # for each data point
    for k in range(n):
        Q = Q_t_full[t] - Q_ti[t, k]
        r = r_t_full[t] - r_ti[t, k]
        # add loo_i to the diagonal
        # looloos_full[t, k, k] = loos[t, k]
        # for each data point i != k
        for i in itertools.chain(range(k), range(k+1, n)):
            x = xs[t, i]
            y = ys[t, i]
            Q2 = Q - Q_ti[t, i]
            r2 = r - r_ti[t, i]
            xS = linalg.solve(Q2, x, assume_a='pos')
            mu_pred = xS.dot(r2)
            sigma2_pred = xS.dot(x) + 1
            looloos_full[t, k, i] = norm_logpdf(y, mu_pred, sigma2_pred)

# make view in which k=i elements are skipped
# strides could be used here also
idx = (
    n*n*np.arange(n_trial)[:,None,None] +
    (n+1)*np.arange(n-1)[:,None] +
    np.arange(1, n+1)
).reshape((n_trial, n, n-1))
looloos = looloos_full.ravel()[idx]

# form sample cov matrix for loo
cov_loo = np.cov(loos, rowvar=False, ddof=1)
sigma2s_loo = np.diag(cov_loo)
gammas_loo = cov_loo[np.triu_indices_from(cov_loo, 1)]

# form sample cov matrices for looloos
looloos_c = looloos - np.mean(looloos, axis=0)
cov_looloo = np.einsum('tka,tkb->kab', looloos_c, looloos_c)
cov_looloo /= n_trial - 1
inds = np.diag_indices(n-1)
sigma2s_looloo = cov_looloo[:,inds[0],inds[1]]
inds = np.triu_indices(n-1, 1)
gammas_looloo = cov_looloo[:,inds[0],inds[1]]

# form estims for Cov(Y_ij, Y_ji)
looloos_full_c = looloos_full - np.mean(looloos_full, axis=0)
cov_looloo2 = np.einsum('tab,tba->ab', looloos_full_c, looloos_full_c)
cov_looloo2 /= n_trial - 1
inds = np.triu_indices(n, 1)
gammas_looloo2 = cov_looloo2[np.triu_indices_from(cov_looloo2, 1)]

print('E[sigma2s_loo]: {}'.format(np.mean(sigma2s_loo)))
print('E[sigma2s_looloo]: {}'.format(np.mean(sigma2s_looloo)))
print()
print('E[gammas_loo]: {}'.format(np.mean(gammas_loo)))
print('E[gammas_looloo]: {}'.format(np.mean(gammas_looloo)))
print('E[gammas_looloo2]: {}'.format(np.mean(gammas_looloo2)))


# targets

# loo target
sumvar_target = np.var(np.sum(loos, axis=1), ddof=1)
# this should be approx same as
# np.sum(cov_loo)

# sigma2
sigma2_target = np.mean(sigma2s_loo)
# gamma
gamma_target = np.mean(gammas_loo)


naive_estims = n*np.var(loos, axis=1, ddof=1)
# expectation of this
# np.mean(naive_estims)
# should be approx same as
# np.sum(sigma2s_loo)

# plot
plt.hist(naive_estims, 20)
plt.axvline(sumvar_target, color='red')
plt.axvline(np.mean(naive_estims), color='green')


# looloo gamma estims
gamma_estims = np.empty((n_trial, num_of_pairs))
# for each test
for t in range(n_trial):
    looloo_t = looloos_full[t]
    # for each pair or points i, j
    for pair_i, (i, j) in enumerate(itertools.combinations(range(n), 2)):
        # slice into new arrays dropping k=i,j
        looloo_ti = np.delete(looloo_t[:, i], [i, j])
        looloo_tj = np.delete(looloo_t[:, j], [i, j])
        # center
        looloo_ti -= looloo_ti.mean()
        looloo_tj -= looloo_tj.mean()
        # multiply
        prod = np.multiply(looloo_ti, looloo_tj, out=looloo_ti)
        # sum
        gamma_estims[t, pair_i] = prod.sum() / (n-3)
print("gamma_estims")
print(np.percentile(gamma_estims, [10, 50, 90]))
print(np.mean(gamma_estims))
print()


# looloo2 gamma estims
gamma_estim2 = np.empty(n_trial)
idx_1 = np.arange(0, n-1 if n%2 else n, 2)
idx_2 = np.arange(1, n, 2)
# for each test
for t in range(n_trial):
    looloo_t = looloos_full[t]
    # exclusive pairs of points
    t1 = looloo_t[idx_1, idx_2]
    t2 = looloo_t[idx_2, idx_1]
    gamma_estim2[t] = np.sum((t1-t1.mean())*(t2-t2.mean())) / (len(t1) - 1)
# limited version
gamma_estim2_limited = np.where(gamma_estim2<0, 0, gamma_estim2)
print("gamma_estim2")
print(np.percentile(gamma_estim2, [10, 50, 90]))
print(np.mean(gamma_estim2))
print()

# looloo3 gamma estims
gamma_estim3 = np.empty(n_trial)
triu_inds = np.triu_indices(n, 1)
# for each test
for t in range(n_trial):
    # for each pair or points i, j
    looloo_t = looloos_full[t]
    centered = looloo_t - np.nanmean(looloo_t, axis=0)
    pairprods = centered*centered.T
    gamma_estim3[t] = np.sum(pairprods[triu_inds]) / (num_of_pairs-1)
# limited version
gamma_estim3_limited = np.where(gamma_estim3<0, 0, gamma_estim3)
print("gamma_estim3")
print(np.percentile(gamma_estim3, [10, 50, 90]))
print(np.mean(gamma_estim3))
print()

# old var estim
old_var_estim = np.mean(np.var(looloos, ddof=1, axis=1), axis=1)
print("old_var_estim")
print(np.percentile(old_var_estim, [10, 50, 90]))
print(np.mean(old_var_estim))
print()
