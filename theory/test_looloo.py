"""Test LOO-LOO approximation."""

import itertools
import numpy as np
from scipy import linalg

import matplotlib.pyplot as plt


# conf
seed = 11
n = 16
p = 2
n_trial = 4000


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

# run loos or load precalculated
if False:
    # run new loos

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

    # save
    np.savez_compressed('test_looloo.npz', loos=loos, looloos_full=looloos_full)

else:
    # load precalculated loos
    saved_file = np.load('test_looloo.npz')
    loos = saved_file['loos']
    looloos_full = saved_file['looloos_full']
    saved_file.close()

# make view in which k=i elements are skipped
# strides could be used here also
idx = (
    n*n*np.arange(n_trial)[:,None,None] +
    (n+1)*np.arange(n-1)[:,None] +
    np.arange(1, n+1)
).reshape((n_trial, n, n-1))
looloos = looloos_full.ravel()[idx]


# ====== estimate covariances

# form sample cov matrix for loo
cov_loo = np.cov(loos, rowvar=False, ddof=1)
sigma2s_loo = np.diag(cov_loo)
gammas_loo = cov_loo[np.triu_indices_from(cov_loo, 1)]

# form estims for Cov(Y_ij, Y_ji)
looloos_full_c = looloos_full - np.mean(looloos_full, axis=0)
prodsums = np.einsum('tab,tba->ab', looloos_full_c, looloos_full_c)
prodsums /= n_trial - 1
gammas_looloo = prodsums[np.triu_indices_from(prodsums, 1)]

print('E[sigma2s_loo]: {}'.format(np.mean(sigma2s_loo)))
print('E[gammas_loo]: {}'.format(np.mean(gammas_loo)))
print('E[gammas_looloo]: {}'.format(np.mean(gammas_looloo)))


# create looloo_full with loo in diag
looloos_inc_diag = looloos_full.copy()
idx = np.diag_indices(n)
looloos_inc_diag[:, idx[0], idx[1]] = loos
looloos_inc_diag_flat = looloos_inc_diag.reshape((n_trial, n*n))
cov_looloo_inc_diag = np.cov(
    looloos_inc_diag.reshape((n_trial, n*n)),
    rowvar=False,
    ddof=1
)



# ====== targets

# sigma2
sigma2_target = np.mean(sigma2s_loo)
# gamma
gamma_target = np.mean(gammas_loo)

# loo target
sumvar_target = np.var(np.sum(loos, axis=1), ddof=1)
# this should be approx same as
# np.sum(cov_loo)
# i.e. approx
# n*sigma2_target + (n**2-n)*gamma_target


# ====== estimates for gamma

# looloo gamma estims
looloos_full_c = (looloos_full - np.nanmean(looloos_full, axis=1)[:, None, :])
looloos_full_c_T = np.lib.stride_tricks.as_strided(
    looloos_full_c,
    strides=(
        looloos_full_c.strides[0],
        looloos_full_c.strides[2],
        looloos_full_c.strides[1]
    ),
    writeable=False
)
pairprods = looloos_full_c*looloos_full_c_T
triu_inds = np.triu_indices(n, 1)
gamma_estims = np.sum(pairprods[:, triu_inds[0], triu_inds[1]], axis=1)
gamma_estims /= num_of_pairs - 1
# limited version
# gamma_estims_limited = np.where(gamma_estims<0, 0, gamma_estims)
print("gamma_estims")
print(np.percentile(gamma_estims, [10, 50, 90]))
print(np.mean(gamma_estims))
print()

# old looloo var estim
old_looloo_var_estim = np.sum(
    np.var(looloos, ddof=1, axis=1), axis=1
) / (n**2 - n)
print("old_looloo_var_estim")
print(np.percentile(old_looloo_var_estim, [10, 50, 90]))
print(np.mean(old_looloo_var_estim))
print()

# ====== estimates for loo

naive_estims = np.var(loos, axis=1, ddof=1)
naive_estims *= n
# expectation of this
# np.mean(naive_estims)
# should be approx same as
# np.sum(sigma2s_loo)

# new looloo estims
looloo_estims = naive_estims + (n**2 - n)*gamma_estims

# old looloo estims
looloo_old_estims = naive_estims + (n**2 - n)*old_looloo_var_estim

# plot
def plot_one_hist(ax, estims, bins=50):
    ax.hist(estims, bins=bins)
    ax.axvline(sumvar_target, color='red')
    ax.axvline(np.mean(estims), color='green')
fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
plot_one_hist(axes[0], naive_estims)
plot_one_hist(axes[1], looloo_estims)
plot_one_hist(axes[2], looloo_old_estims)
axes[0].set_xlim([0, 40])
