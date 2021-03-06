
import itertools

from tqdm import tqdm

import numpy as np
from scipy import linalg, stats
from scipy.special import comb

import matplotlib.pyplot as plt


# ==============================================================================
# config

seed = 57346234
n_trial = 2000

n_loo_mc = 1000
n_impr_mc = 1000

# data params
n_obs = 16

datadist = stats.norm(loc=0, scale=1.2)
# mu = 0.0,
# sigma2 = 1.4,
# skew = 0.0,
# exkurt = 0.0

# datadist = stats.norm(loc=50, scale=1.2)
# tai

# datadist = stats.norm(loc=2, scale=0.1)
# mu = 2.0,
# sigma2 = 0.01,
# skew = 0.0,
# exkurt = 0.0


# datadist = stats.skewnorm(10, loc=-2, scale=0.16)
# mu = -1.9,
# sigma2 = 0.0095,
# skew = 0.96,
# exkurt = 0.82


# priors
# conjugate prior BDA 3rd ed pp. 67-68
sigma2_0 = 2.0**2
k_0 = 0.1
v_0 = 10


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


def get_loo_ti(data_ti):

    # noninformative prior
    # out = np.zeros((n_trial, n_obs))
    # for i in range(n_obs):
    #     ytilde_t = data_ti[:, i]
    #     y_t = np.delete(data_ti, i, axis=-1)
    #     n_obs_cur = y_t.shape[-1]
    #     ybar_t = np.mean(y_t, axis=-1)
    #     s2_t = np.var(y_t, ddof=1, axis=-1)
    #     scale_t = s2_t*np.sqrt(1+1/n_obs_cur)
    #     df = n_obs_cur - 1
    #     out[:, i] = stats.t.logpdf(ytilde_t, df, loc=ybar_t, scale=scale_t)

    # conjugate prior BDA 3rd ed pp. 67-68
    out = np.zeros((n_trial, n_obs))
    for i in tqdm(range(n_obs)):
        ytilde_t = data_ti[:, i]
        y_t = np.delete(data_ti, i, axis=-1)
        n_obs_cur = y_t.shape[-1]
        ybar_t = np.mean(y_t, axis=-1)
        s2_t = np.var(y_t, ddof=1, axis=-1)
        mu_n = (n_obs_cur/(k_0 + n_obs_cur))*ybar_t
        k_n = k_0 + n_obs_cur
        v_n = v_0 + n_obs_cur
        v_n_s2_n = (
            v_0*sigma2_0
            + (n_obs_cur-1)*s2_t
            + n_obs_cur*k_0/(n_obs_cur+k_0)*ybar_t**2
        )
        alpha = 0.5*v_n
        # beta = 0.5*s2_n*v_n
        beta = 0.5*v_n_s2_n
        s2_samp = stats.invgamma.rvs(
            alpha, scale=beta, size=(n_loo_mc, n_trial), random_state=rng)
        temp = s2_samp/k_n
        np.sqrt(temp, out=temp)
        mu_samp = stats.norm.rvs(
            loc=mu_n, scale=temp, random_state=rng)
        np.sqrt(s2_samp, out=temp)
        out[:, i] = np.log(np.mean(
            stats.norm.pdf(ytilde_t, loc=mu_samp, scale=temp), axis=0))
    return out


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


def improved_estim(data_ti):
    n = n_obs

    # full data marginal posterior for sigma_m
    ybar_t = np.mean(data_ti, axis=-1)
    s2_t = np.var(data_ti, ddof=1, axis=-1)
    # mu_n = (n/(k_0 + n))*ybar_t
    # k_n = k_0 + n
    v_n = v_0 + n
    v_n_s2_n = (
        v_0*sigma2_0
        + (n-1)*s2_t
        + n*k_0/(n+k_0)*ybar_t**2
    )
    alpha = 0.5*v_n
    # beta = 0.5*s2_n*v_n
    beta = 0.5*v_n_s2_n
    s2_m_samp = stats.invgamma.rvs(
        alpha, scale=beta, size=(n_impr_mc, n_trial), random_state=rng)

    sigma2_p_samp = s2_m_samp / k_0

    # continue as with fixed var
    mup2_s2_hat, s4_hat, mu_mu3_hat, mu4_hat = get_moment_estims(data_ti)

    a = (
        -0.5*(s2_m_samp + (n-1)*sigma2_p_samp)
        / (s2_m_samp*(s2_m_samp+n*sigma2_p_samp))
    )
    b = (n-1)*sigma2_p_samp/(s2_m_samp*(s2_m_samp+n*sigma2_p_samp))
    c = (
        -0.5*(n-1)**2*sigma2_p_samp**2
        / (s2_m_samp*(s2_m_samp + (n-1)*sigma2_p_samp)
            *(s2_m_samp+n*sigma2_p_samp))
    )

    var_hat_samp = (
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
    return np.mean(var_hat_samp, axis=0)


# ==============================================================================
# simulate

data_ti = datadist.rvs(size=(n_trial, n_obs), random_state=rng)
print('getting loos ...')
loo_ti = get_loo_ti(data_ti)
loo_t = np.sum(loo_ti, axis=-1)

# target
target_var = np.var(loo_t, ddof=1)

# naive
var_hat_naiv_t = n_obs*np.var(loo_ti, ddof=1, axis=-1)

# get improved estims
var_hat_impr_t = improved_estim(data_ti)


# ============================================================================
# plot stuff
# ============================================================================

plt.rcParams['font.family'] = 'serif'
plt.rcParams["mathtext.fontset"] = "dejavuserif"

bb_n = 2000
alpha_bt = rng.dirichlet(np.ones(n_trial), size=bb_n)


# ==============================================================================
# plot BB ratio

datas = [
    np.sqrt(alpha_bt.dot(var_hat_naiv_t)/target_var),
    np.sqrt(alpha_bt.dot(var_hat_impr_t)/target_var)
]
datas_point = [
    np.sqrt(np.mean(var_hat_naiv_t)/target_var),
    np.sqrt(np.mean(var_hat_impr_t)/target_var)
]
names = ['naive', 'improved']

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(4, 3))
for ax, data, name in zip(axes, datas, names):
    # data_filtered = data[data<uplim]
    ax.hist(data, bins=20, label='BB')
    ax.axvline(1.0, color='k', lw=0.8)

    # ax.set_xticks([0, 1, 2, 3])

    # ax.set_xlim(left=0)

    ax.set_yticks([])
    ax.spines['left'].set_visible(False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel(name)



fig.suptitle(f'n={n_obs}')

axes[-1].set_xlabel(
    r'$\sqrt{\left.\mathrm{E}\left[\widehat{\sigma^2_\mathrm{LOO}}\right]'
    r'\;\right/\;'
    r'\sigma^2_\mathrm{LOO}}$',
)

fig.tight_layout()
