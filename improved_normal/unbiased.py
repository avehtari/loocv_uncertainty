"""Simulated experiment for the unbiased LOO-CV variance estimator.

Numerical results
-----------------

1. well-fitting data
    mu = 0.0,
    sigma2 = 1.4,
    skew = 0.0,
    exkurt = 0.0
    naive analytic: 0.966125189
    naive BB mean: 0.9664053747
    naive BB median: 0.9664015508
    naive BB 95 % CI: [0.95996186 0.97273841]
    naive BB 99 % CI: [0.95756668 0.97492683]
    unbiased analytic: 1.0
    unbiased BB mean: 1.00030043
    unbiased BB median: 1.000301568
    unbiased BB 95 % CI: [0.9934566  1.00704534]
    unbiased BB 99 % CI: [0.9908942  1.00942793]

2. under-dispersed data
    mu = 2.0,
    sigma2 = 0.01,
    skew = 0.0,
    exkurt = 0.0
    naive analytic: 1.160312657
    naive BB mean: 1.15877344
    naive BB median: 1.158743842
    naive BB 95 % CI: [1.15169156 1.16562419]
    naive BB 99 % CI: [1.14959731 1.16764143]
    unbiased analytic: 1.0
    unbiased BB mean: 1.000248493
    unbiased BB median: 1.00025908
    unbiased BB 95 % CI: [0.99339451 1.00696045]
    unbiased BB 99 % CI: [0.99082462 1.00938296]

3. under-dispersed, skewed, heavy-tailed data
    mu = -1.9,
    sigma2 = 0.0095,
    skew = 0.96,
    exkurt = 0.82
    naive analytic: 0.8094516078
    naive BB mean: 0.8098450276
    naive BB median: 0.8098479447
    naive BB 95 % CI: [0.80231211 0.81750817]
    naive BB 99 % CI: [0.80000032 0.82010894]
    unbiased analytic: 1.0
    unbiased BB mean: 0.9994987331
    unbiased BB median: 0.999523508
    unbiased BB 95 % CI: [0.98842999 1.01048438]
    unbiased BB 99 % CI: [0.98563382 1.01432708]

"""


import itertools

import numpy as np
from scipy import linalg, stats
from scipy.special import comb

import matplotlib.pyplot as plt


# ==============================================================================
# config

seed = 57346234
n_trial = 20000

# data params
n_obs = 16

# data 1
datadist = stats.norm(loc=0, scale=1.2)

# data 2
# datadist = stats.norm(loc=2, scale=0.1)

# data 3
# datadist = stats.skewnorm(10, loc=-2, scale=0.16)


# model params
sigma2_m = 1.2**2
sigma2_p = 2.0**2


# ==============================================================================

rng = np.random.RandomState(seed)

# data moments
d_stats = datadist.stats(moments='mvsk')
mu = d_stats[0][()]
sigma2 = d_stats[1][()]
skew = d_stats[2][()]
exkurt = d_stats[3][()]
mu_3 = skew*np.sqrt(sigma2)**3
mu_4 = (exkurt+3)*sigma2**2

print(f'mu = {mu:.2},\nsigma2 = {sigma2:.2},\nskew = {skew:.2},\nexkurt = {exkurt:.2}')

const_a = (
    -0.5*(sigma2_m + (n_obs-1)*sigma2_p)
    / (sigma2_m*(sigma2_m+n_obs*sigma2_p))
)
const_b = (n_obs-1)*sigma2_p/(sigma2_m*(sigma2_m+n_obs*sigma2_p))
const_c = (
    -0.5*(n_obs-1)**2*sigma2_p**2
    / (sigma2_m*(sigma2_m + (n_obs-1)*sigma2_p)*(sigma2_m+n_obs*sigma2_p))
)
const_d = -0.5*np.log(
    2*np.pi*(sigma2_m*(sigma2_m+n_obs*sigma2_p))/(sigma2_m+(n_obs-1)*sigma2_p))

# # test consts
# ntest = n_obs-1
# ytilde = np.random.randn(5)
# ybar = 3.46
# ppred_tau = 1/(1/sigma2_p + ntest/sigma2_m)
# ppred_mu = ppred_tau*ntest/sigma2_m * ybar
# ppred_sigma2 = sigma2_m + ppred_tau
# stats.norm.logpdf(ytilde, ppred_mu, np.sqrt(ppred_sigma2))
# const_a*ytilde**2 + const_b*ybar*ytilde + const_c*ybar**2 + const_d
# # match ok

# def get_loo_i(data_i):
#     n_obs = len(data_i)
#     # sum_all = np.sum(data_i)
#     # out = np.zeros(n_obs)
#     # for i in range(n_obs):
#     #     ytilde = data_i[i]
#     #     ybar_i = sum_all - ytilde
#     #     ybar_i /= n_obs-1
#     #     out[i] = (
#     #         const_a*ytilde**2
#     #         + const_b*ytilde*ybar_i
#     #         + const_c*ybar_i**2
#     #         + const_d
#     #     )
#     ybar_i = (np.sum(data_i)-data_i)/(n_obs-1)
#     out = (
#         const_a*data_i**2
#         + const_b*data_i*ybar_i
#         + const_c*ybar_i**2
#         + const_d
#     )
#     return out


def get_loo_ti(data_ti):
    n_trial, n_obs = data_ti.shape
    ybar_ti = np.sum(data_ti, axis=-1, keepdims=True) - data_ti
    ybar_ti /= n_obs-1
    # a
    out = np.square(data_ti)
    out *= const_a
    # + b
    temp = data_ti*ybar_ti
    temp *= const_b
    out += temp
    # + c
    temp = np.square(ybar_ti, out=temp)
    temp *= const_c
    out += temp
    # + d
    out += const_d
    return out


def get_true_var():
    n, m, s2, m3, m4, a, b, c = (
        n_obs, mu, sigma2, mu_3, mu_4, const_a, const_b, const_c)
    true_var = (
        m**2*s2 * 4*n*(a+b+c)**2
        + s2**2 * (
                -n*a**2
                + 2*n/(n-1)*b**2
                + n*(2*n-3)*(n-3)/(n-1)**3*c**2
                - 2*n/(n-1)*a*c
                + 4*n*(n-2)/(n-1)**2*b*c
            )
        + m*m3 * 4*n*(a+b+c)*(a*(n-1)+c)/(n-1)
        + m4 * (n*a**2 + n/(n-1)**2*c**2 + 2*n/(n-1)*a*c)
    )
    return true_var


def get_true_var_i():
    n, m, s2, m3, m4, a, b, c = (
        n_obs, mu, sigma2, mu_3, mu_4, const_a, const_b, const_c)
    true_var = (
        m**2*s2 * (4*a**2 + n/(n-1)*b**2 + 4/(n-1)*c**2 + 4*a*b + 4/(n-1)*b*c)
        + s2**2 * (-a**2 +1/(n-1)*b**2 + (2*n-5)/(n-1)**3*c**2)
        + m*m3 * (4*a**2 + 4/(n-1)**2*c**2 + 2*a*b + 2/(n-1)**2*b*c)
        + m4 * (a**2 + 1/(n-1)**3*c**2)
    )
    return true_var


def get_true_cov_ij():
    n, m, s2, m3, m4, a, b, c = (
        n_obs, mu, sigma2, mu_3, mu_4, const_a, const_b, const_c)
    true_cov = (
        m**2*s2 * (
            (3*n-4)/(n-1)**2*b**2
            + 4*(n-2)/(n-1)**2*c**2
            + 4/(n-1)*a*b
            + 8/(n-1)*a*c
            + 4*(2*n-3)/(n-1)**2*b*c
        )
        + s2**2 * (
            1/(n-1)**2*b**2
            + (n-2)*(2*n-7)/(n-1)**4*c**2
            - 2/(n-1)**2*a*c
            + 4*(n-2)/(n-1)**3*b*c
        )
        + m*m3 * (
            4*(n-2)/(n-1)**3*c**2
            + 2/(n-1)*a*b
            +4*n/(n-1)**2*a*c
            + (4*n-6)/(n-1)**3*b*c
        )
        + m4 * ((n-2)/(n-1)**4*c**2 + 2/(n-1)**2*a*c)
    )
    return true_cov


def get_moment_estims(data_ti):
    n = n_obs
    a1 = np.mean(data_ti, axis=-1)
    a2 = np.mean(data_ti**2, axis=-1)
    a3 = np.mean(data_ti**3, axis=-1)
    a4 = np.mean(data_ti**4, axis=-1)
    a1p2 = a1**2
    a1p4 = a1p2**2
    a2p2 = a2**2
    a2_a1p2 = a2*a1p2
    a3_a1 = a3*a1
    # mup4_hat
    # mup4_hat = np.zeros(n_trial)
    # for i1, i2, i3, i4 in itertools.permutations(range(n), 4):
    #     mup4_hat += data_ti[:, i1]*data_ti[:, i2]*data_ti[:, i3]*data_ti[:, i4]
    # mup4_hat /= n*(n-1)*(n-2)*(n-3)
    mup4_hat = np.zeros(n_trial)
    for i1, i2, i3, i4 in itertools.combinations(range(n), 4):
        mup4_hat += data_ti[:, i1]*data_ti[:, i2]*data_ti[:, i3]*data_ti[:, i4]
    mup4_hat /= comb(n, 4, exact=True)
    # mup2_s2_hat
    mup2_s2_hat = (
        - n**3*a1p4
        + 2*n**3*a2_a1p2
        - 4*(n-1)*n*a3_a1
        - (2*n**2 - 3*n)*a2p2
        +2*(2*n-3)*a4
    ) / (2*(n-1)*(n-2)*(n-3)) - 0.5*mup4_hat
    # s4_hat
    s4_hat = (
        n**3*a1p4
        - 2*n**3*a2_a1p2
        + (n**3-3*n**2+3*n)*a2p2
        + 4*n*(n-1)*a3_a1
        + n*(1-n)*a4
    ) / ((n-1)*(n-2)*(n-3))
    # mu_mu3_hat
    mu_mu3_hat = (
        - 2*(n**2 + n-3)*a4
        - 6*n**3*a2_a1p2
        + n*(6*n-9)*a2p2
        + 3*n**3*a1p4
        + 2*n**2*(n+1)*a3_a1
    ) / (2*(n-1)*(n-2)*(n-3)) + 0.5*mup4_hat
    # mu4_hat
    mu4_hat = (
        - 3*n**3*a1p4
        + 6*n**3*a2_a1p2
        + (9-6*n)*n*a2p2
        + (-12 + 8*n-4*n**2)*n*a3_a1
        + (3*n-2*n**2+n**3)*a4
    ) / ((n-3)*(n-2)*(n-1))
    return mup2_s2_hat, s4_hat, mu_mu3_hat, mu4_hat


def improved_estim(mup2_s2_hat, s4_hat, mu_mu3_hat, mu4_hat):
    n, m, s2, m3, m4, a, b, c = (
        n_obs, mu, sigma2, mu_3, mu_4, const_a, const_b, const_c)
    var_hat = (
        mup2_s2_hat * 4*n*(a+b+c)**2
        + s4_hat * (
                -n*a**2
                + 2*n/(n-1)*b**2
                + n*(2*n-3)*(n-3)/(n-1)**3*c**2
                - 2*n/(n-1)*a*c
                + 4*n*(n-2)/(n-1)**2*b*c
            )
        + mu_mu3_hat * 4*n*(a+b+c)*(a*(n-1)+c)/(n-1)
        + mu4_hat * (n*a**2 + n/(n-1)**2*c**2 + 2*n/(n-1)*a*c)
    )
    return var_hat


true_var = get_true_var()
true_var_i = get_true_var_i()
true_cov_ij = get_true_cov_ij()


# ==============================================================================
# simulate

data_ti = datadist.rvs(size=(n_trial, n_obs), random_state=rng)
loo_ti = get_loo_ti(data_ti)


# --------------------------------------------------
# compare true var
# np.var(loo_ti.sum(axis=-1), ddof=1)
# true_var
# ok


# --------------------------------------------------
# compare moment estims
mup2_s2_hat, s4_hat, mu_mu3_hat, mu4_hat = get_moment_estims(data_ti)

# np.mean(mup2_s2_hat)
# mu**2*sigma2
# ok

# np.mean(s4_hat)
# sigma2**2
# ok

# np.mean(mu_mu3_hat)
# mu*mu_3
# ok

# np.mean(mu4_hat)
# mu_4
# ok


# --------------------------------------------------
# get improved estims

var_hat_impr_t = improved_estim(mup2_s2_hat, s4_hat, mu_mu3_hat, mu4_hat)

# np.mean(var_hat_impr_t)
# true_var
# ok

# compare to the naive
var_hat_naiv_t = n_obs*np.var(loo_ti, ddof=1, axis=-1)
# np.mean(var_hat_naiv_t)



# ============================================================================
# plot stuff
# ============================================================================

plt.rcParams['font.family'] = 'serif'
plt.rcParams["mathtext.fontset"] = "dejavuserif"

bb_n = 4000
alpha_bt = rng.dirichlet(np.ones(n_trial), size=bb_n)


# ==============================================================================
# plot datadistr

# priorps = stats.norm.rvs(
#     loc=stats.norm.rvs(loc=0, scale=np.sqrt(sigma2_p), size=4000),
#     scale=np.sqrt(sigma2_m)
# )


# ppred_tau = 1/(1/sigma2_p + (n_obs-1)/sigma2_m)
# ppred_mu = ppred_tau*(n_obs-1)/sigma2_m * data_ti[:, :-1].mean(axis=-1)
# ppred_sigma2 = sigma2_m + ppred_tau
# loopps = stats.norm.rvs(
#     loc=ppred_mu,
#     scale=np.sqrt(ppred_sigma2)
# )
#
# r_0, r_1 = datadist.interval(0.98)
# # p_0, p_1 = np.percentile(priorps, [1, 99])
# pp_0, pp_1 = np.percentile(loopps, [1, 99])
# lim_0 = min(r_0, pp_0)
# lim_1 = max(r_1, pp_1)
#
# fig, axes = plt.subplots(2, 1, sharex=True, figsize=(4, 3))
# x = np.linspace(lim_0, lim_1, 100)
#
# ax = axes[0]
# ax.plot(x, datadist.pdf(x), label='data')
#
# # ax = axes[1]
# # ax.hist(
# #     priorps[(priorps > lim_0) & (priorps < lim_1)],
# #     label='prior pred',
# #     bins=25
# # )
#
# ax = axes[1]
# ax.hist(
#     loopps[(loopps > lim_0) & (loopps < lim_1)],
#     label='loo post pred',
#     bins=25
# )


# # ==============================================================================
# # bb var
# loo_t = loo_ti.sum(axis=-1)
# temp_bt = loo_t - alpha_bt.dot(loo_t)[:,None]
# temp_bt = np.square(temp_bt, out=temp_bt)
# true_var_b = np.einsum('bt,bt->b', alpha_bt, temp_bt)
# true_var_b /= 1 - np.sum(alpha_bt**2, axis=-1)
#
# fig = plt.figure()
# ax = fig.gca()
# ax.hist(true_var_b, bins=30, color='C0', label='BB')
# ax.axvline(true_var, color='C1', label='analytic')
# ax.axvline(np.percentile(true_var_b, 25), color='C2', label='IQR')
# ax.axvline(np.percentile(true_var_b, 75), color='C2')
# ax.legend()


# ==============================================================================
# plot BB ratio

datas = [
    np.sqrt(alpha_bt.dot(var_hat_naiv_t)/true_var),
    np.sqrt(alpha_bt.dot(var_hat_impr_t)/true_var)
]
# datas_point = [
#     np.sqrt(np.mean(var_hat_naiv_t)/true_var),
#     np.sqrt(np.mean(var_hat_impr_t)/true_var)
# ]
datas_point = [
    np.sqrt(n_obs*(true_var_i-true_cov_ij)/true_var),
    1.0
]
names = ['naive', 'unbiased']

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(4, 3))
for ax, data, data_point, name in zip(axes, datas, datas_point, names):
    # data_filtered = data[data<uplim]
    ax.hist(data, bins=20, label='BB')
    ax.axvline(1.0, color='C2', lw=2.0, label='target')
    ax.axvline(data_point, color='C1', ls='--', label='analytic')

    # ax.set_xticks([0, 1, 2, 3])

    # ax.set_xlim(left=0)

    ax.set_yticks([])
    ax.spines['left'].set_visible(False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel(name)

# axes[1].legend(
#     loc='lower left',
#     fancybox=False, shadow=False, framealpha=1.0)

axes[-1].set_xlabel(
    r'$\sqrt{\left.\mathrm{E}\left[\widehat{\sigma^2_\mathrm{LOO}}\right]'
    r'\;\right/\;'
    r'\sigma^2_\mathrm{LOO}}$',
)

fig.tight_layout()


# print
for data, data_point, name in zip(
        [alpha_bt.dot(var_hat_naiv_t), alpha_bt.dot(var_hat_impr_t)],
        datas_point, names):
    print(f'{name} analytic: {data_point:.10}')
    print(f'{name} BB mean: {np.sqrt(np.mean(data)/true_var):.10}')
    print(f'{name} BB median: {np.sqrt(np.median(data)/true_var):.10}')
    print(f'{name} BB 95 % CI: '
        f'{np.sqrt(np.percentile(data, [2.5, 97.5])/true_var)}')
    print(f'{name} BB 99 % CI: '
        f'{np.sqrt(np.percentile(data, [0.5, 99.5])/true_var)}')
