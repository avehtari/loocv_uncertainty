"""Test LOO-LOO approximation."""

import itertools
import numpy as np
from scipy import linalg
from scipy import stats

import matplotlib.pyplot as plt


# conf
seed = 11
n = 10
p = 2
n_trial = 4000
PLOT = False


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
    loo_ti = np.empty((n_trial, n))
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
            loo_ti[t, i] = norm_logpdf(y, mu_pred, sigma2_pred)

    # loo-loos
    loo_tik = np.full((n_trial, n, n), np.nan)
    # for each test
    for t in range(n_trial):
        # for each datapoint i
        for i in range(n):
            Q = Q_t_full[t] - Q_ti[t, i]
            r = r_t_full[t] - r_ti[t, i]
            x = xs[t, i]
            y = ys[t, i]
            # for each other data point k != i
            for k in itertools.chain(range(i), range(i+1, n)):
                Q2 = Q - Q_ti[t, k]
                r2 = r - r_ti[t, k]
                xS = linalg.solve(Q2, x, assume_a='pos')
                mu_pred = xS.dot(r2)
                sigma2_pred = xS.dot(x) + 1
                loo_tik[t, i, k] = norm_logpdf(y, mu_pred, sigma2_pred)
            # optionally add loo_i to the diagonal
            # loo_tik[t, i, i] = loo_ti[t, i]

    # save
    np.savez_compressed(
        'test_looloo.npz', loo_ti=loo_ti, loo_tik=loo_tik)

else:
    # load precalculated loos
    saved_file = np.load('test_looloo.npz')
    loo_ti = saved_file['loo_ti']
    loo_tik = saved_file['loo_tik']
    saved_file.close()


# ====== moments

# for A
mean_i = np.mean(loo_ti, axis=0)
var_i = np.var(loo_ti, axis=0, ddof=1)
cov_ij = np.cov(loo_ti, rowvar=False, ddof=1)

# for B
mean_ik = np.mean(loo_tik, axis=0)
var_ik =  np.var(loo_tik, axis=0, ddof=1)
t_cent = loo_tik - mean_ik
cov_ikjl = np.einsum('tik,tjl->ikjl', t_cent, t_cent)
cov_ikjl /= n_trial - 1

# select different parts of cov_ikjl
i_i, i_k, i_j, i_l = np.unravel_index(np.arange(n**4), cov_ikjl.shape)
i_valids = (i_i != i_k) & (i_j != i_l)
# i != j, k == l
cov_ineqj_keql = cov_ikjl.ravel()[i_valids & (i_i != i_j) & (i_k == i_l)]
# i == j, k != l
cov_ieqj_kneql = cov_ikjl.ravel()[i_valids & (i_i == i_j) & (i_k != i_l)]
# i == l, j == k (also i != j, k != l)
cov_ieql_jeqk = cov_ikjl.ravel()[i_valids & (i_i == i_l) & (i_j == i_k)]
# i == l, j != k
cov_ieql_jneqk = cov_ikjl.ravel()[i_valids & (i_i == i_l) & (i_j != i_k)]
# i != l, j == k
cov_ineql_jeqk = cov_ikjl.ravel()[i_valids & (i_i != i_l) & (i_j == i_k)]
# i, k, j, l all not equal
cov_all_neq = cov_ikjl.ravel()[
    i_valids & (i_i != i_j) & (i_k != i_l) & (i_i != i_l) & (i_j != i_k)]

# moment estims
mu_A_hat = np.mean(mean_i)
sigma2_A_hat = np.mean(var_i)
gamma_A_hat = np.mean(cov_ij[np.triu_indices_from(cov_ij, 1)])
mu_B_hat = np.mean(mean_ik[np.arange(n) != np.arange(n)[:,None]])
sigma2_B_hat = np.mean(var_ik[np.arange(n) != np.arange(n)[:,None]])
gamma_1_B_hat = np.mean(cov_ineqj_keql)
gamma_2_B_hat = np.mean(cov_ieqj_kneql)
gamma_3_B_hat = np.mean(cov_ieql_jeqk)
gamma_4_B_hat = np.mean(cov_ieql_jneqk)
gamma_5_B_hat = np.mean(cov_ineql_jeqk)
gamma_6_B_hat = np.mean(cov_all_neq)


# ==== check moments correspondence

if PLOT:

    fig, axes = plt.subplots(3, 1, sharex=True)

    # var_i
    plot_arr = var_i
    axes[0].hist(plot_arr)
    axes[0].axvline(np.mean(plot_arr), color='C1')
    axes[0].set_title('Var(A_i)')
    # var_ik
    plot_arr = var_ik[np.arange(n) != np.arange(n)[:,None]]  # skip diagonal
    axes[1].hist(plot_arr)
    axes[1].axvline(np.mean(plot_arr), color='C1')
    axes[1].set_title('Var(B_ik)')
    # cov_ieqj_kneql
    plot_arr = cov_ieqj_kneql
    axes[2].hist(plot_arr)
    axes[2].axvline(np.mean(plot_arr), color='C1')
    axes[2].set_title('Cov(B_ik,B_il)')


    fig, axes = plt.subplots(6, 1, sharex=True)

    # cov_ij
    plot_arr = cov_ij[np.triu_indices_from(cov_ij, 1)]  # take upper diagonal
    axes[0].hist(plot_arr)
    axes[0].axvline(np.mean(plot_arr), color='C1')
    axes[0].set_title('Cov(A_i,A_j)')
    # cov_ineqj_keql
    plot_arr = cov_ineqj_keql
    axes[1].hist(plot_arr)
    axes[1].axvline(np.mean(plot_arr), color='C1')
    axes[1].set_title('Cov(B_ik,B_jk)')
    # cov_ieql_jeqk
    plot_arr = cov_ieql_jeqk
    axes[2].hist(plot_arr)
    axes[2].axvline(np.mean(plot_arr), color='C1')
    axes[2].set_title('Cov(B_ij,B_ji)')
    # cov_ieql_jneqk
    plot_arr = cov_ieql_jneqk
    axes[3].hist(plot_arr)
    axes[3].axvline(np.mean(plot_arr), color='C1')
    axes[3].set_title('Cov(B_ij,B_jk)')
    # cov_ineql_jeqk
    plot_arr = cov_ineql_jeqk
    axes[4].hist(plot_arr)
    axes[4].axvline(np.mean(plot_arr), color='C1')
    axes[4].set_title('Cov(B_ij,B_ki')
    # cov_all_neq
    plot_arr = cov_all_neq
    axes[5].hist(plot_arr)
    axes[5].axvline(np.mean(plot_arr), color='C1')
    axes[5].set_title('Cov(B_ik,B_jh')


# ==== check true variance
print(
    'target loo sum: Var[Sum A_i]={:.5} approx. {:.5}'
    .format(
        n*sigma2_A_hat + (n**2-n)*gamma_A_hat,
        np.sum(loo_ti, axis=1).var(ddof=1)
    )
)


# ==== estimates

# naive
estims_naive = n*loo_ti.var(ddof=1, axis=1)
# check if theory correct
print(
    'naive: E[V]={:.5} approx. {:.5}'
    .format(n*(sigma2_A_hat - gamma_A_hat), estims_naive.mean())
)

# W estim
estims_w = np.sum(np.nanvar(loo_tik, ddof=1, axis=2), axis=1)
# check if theory correct
print(
    'g2: E[W]={:.5} approx. {:.5}'
    .format(n*(sigma2_B_hat - gamma_2_B_hat), estims_w.mean())
)

# G estim
loo_tik_c = loo_tik - np.nanmean(loo_tik, axis=-1)[:,:,None]
# it would be nice if there was naneinsum
t_arr = np.nansum(np.einsum('tik,tjk->tijk', loo_tik_c, loo_tik_c), axis=-1)
t_arr /= n-3
# select last 2 axis upper triangular
i_i, i_j = np.triu_indices(n, 1)
estim_g = t_arr[:, i_i, i_j].mean(axis=-1)

# G2 estim with loops (mean only considers k neq i,j)
# res = np.zeros((n_trial, num_of_pairs))
# for t in range(n_trial):
#     pair_i = 0
#     for i in range(n-1):
#         for j in range(i+1, n):
#             k = tuple(
#                 itertools.chain(range(i), range(i+1, j), range(j+1, n)))
#             res[t, pair_i] = np.sum(
#                 (loo_tik[t,i,k] - np.mean(loo_tik[t,i,k])) *
#                 (loo_tik[t,j,k] - np.mean(loo_tik[t,j,k]))
#             ) / (n-3)
#             pair_i += 1
# np.mean(res)







t = loo_tik[0]
tc = tc = t - np.nanmean(t, axis=-1)[:,None]
np.sum(tc[0,2:]*tc[1,2:])/(n-3)  # 0,1
ts = np.nansum(np.einsum('ik,jk->ijk', tc, tc), axis=-1)/(n-3)  # i,j







# ==== old


# calc trial loo sums
loo_sums = np.sum(loo_ti, axis=1)


# form estims for Cov(Y_ij, Y_ji)
loo_tik_c = loo_tik - np.mean(loo_tik, axis=0)
prodsums = np.einsum('tab,tba->ab', loo_tik_c, loo_tik_c)
prodsums /= n_trial - 1
gammas_looloo = prodsums[np.triu_indices_from(prodsums, 1)]

print('E[sigma2s_loo]: {}'.format(np.mean(sigma2s_loo)))
print('E[gammas_loo]: {}'.format(np.mean(gammas_loo)))
print('E[gammas_looloo]: {}'.format(np.mean(gammas_looloo)))


# create looloo_full with loo in diag
looloos_inc_diag = loo_tik.copy()
idx = np.diag_indices(n)
looloos_inc_diag[:, idx[0], idx[1]] = loo_ti
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
sumvar_target = np.var(loo_sums, ddof=1)
# this should be approx same as
# np.sum(cov_loo)
# i.e. approx
# n*sigma2_target + (n**2-n)*gamma_target

# mad target
mad_k = 1 / stats.norm.ppf(3/4)
mad_target = (mad_k * np.median(np.abs(loo_sums - np.median(loo_sums))))**2


# ====== estimates for gamma

# looloo gamma estims
loo_tik_c = (loo_tik - np.nanmean(loo_tik, axis=1)[:, None, :])
loo_tik_c_T = np.lib.stride_tricks.as_strided(
    loo_tik_c,
    strides=(
        loo_tik_c.strides[0],
        loo_tik_c.strides[2],
        loo_tik_c.strides[1]
    ),
    writeable=False
)
pairprods = loo_tik_c*loo_tik_c_T
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

# ====== estimates for var loo

naive_estims = np.var(loo_ti, axis=1, ddof=1)
naive_estims *= n
# expectation of this
# np.mean(naive_estims)
# should be approx same as
# np.sum(sigma2s_loo)

# new looloo estims
looloo_estims = naive_estims + (n**2 - n)*gamma_estims

# old looloo estims
looloo_old_estims = naive_estims + (n**2 - n)*old_looloo_var_estim

# MAD
mad_estims = n * np.square(
    mad_k * np.median(
        np.abs(loo_ti - np.median(loo_ti, axis=1)[:, None]),
        axis=1
    )
)

# plot
def plot_one_hist(ax, estims, bins=50):
    ax.hist(estims, bins=bins)
    ax.axvline(sumvar_target, color='red')
    ax.axvline(np.mean(estims), color='green')
fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
plot_one_hist(axes[0], naive_estims)
plot_one_hist(axes[1], looloo_estims)
plot_one_hist(axes[2], looloo_old_estims)
# plot_one_hist(axes[3], mad_estims)
axes[0].set_xlim([0, 40])
