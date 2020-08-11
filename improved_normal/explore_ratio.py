
import itertools

import numpy as np
from scipy import linalg, stats
from scipy.special import comb

import matplotlib.pyplot as plt


# ==============================================================================
# config

# varying params

# params = np.linspace(-10, 10, 21)
# params_name = 'mu'

params = 2.0**np.linspace(-6, 0, 25)
params_name = 'sigma_d'

# params = 2.0**np.linspace(-6, 3, 25)
# params_name = 'sigma_m'

# params = 2.0**np.linspace(-6, 3, 25)
# params_name = 'sigma_p'



def get_all_params(param):

    n_obs = 16

    # datadist = stats.norm(loc=param, scale=1.0)
    # datadist = stats.norm(loc=2, scale=param)

    datadist = stats.skewnorm(10, loc=-2, scale=param)
    # datadist = stats.skewnorm(3, loc=-1.7, scale=1.2)
    # datadist = stats.skewnorm(-3, loc=-1.7, scale=1.2)

    # datadist = stats.nct(df=5, nc=0, loc=-0.3, scale=1.1)
    # datadist = stats.chi2(df=8, loc=param, scale=1.0)

    # model params
    sigma2_m = 1.0
    # sigma2_m = param


    sigma2_p = 2.0**2
    # sigma2_p = param

    return n_obs, datadist, sigma2_m, sigma2_p



# ==============================================================================



def get_true_var(n, m, s2, m3, m4, a, b, c):
    # n, m, s2, m3, m4, a, b, c = (
    #     n_obs, mu, sigma2, mu_3, mu_4, const_a, const_b, const_c)
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


def get_true_var_i(n, m, s2, m3, m4, a, b, c):
    # n, m, s2, m3, m4, a, b, c = (
    #     n_obs, mu, sigma2, mu_3, mu_4, const_a, const_b, const_c)
    true_var = (
        m**2*s2 * (4*a**2 + n/(n-1)*b**2 + 4/(n-1)*c**2 + 4*a*b + 4/(n-1)*b*c)
        + s2**2 * (-a**2 +1/(n-1)*b**2 + (2*n-5)/(n-1)**3*c**2)
        + m*m3 * (4*a**2 + 4/(n-1)**2*c**2 + 2*a*b + 2/(n-1)**2*b*c)
        + m4 * (a**2 + 1/(n-1)**3*c**2)
    )
    return true_var


def get_true_cov_ij(n, m, s2, m3, m4, a, b, c):
    # n, m, s2, m3, m4, a, b, c = (
    #     n_obs, mu, sigma2, mu_3, mu_4, const_a, const_b, const_c)
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


ratios = np.zeros(len(params))

for i, param in enumerate(params):

    n_obs, datadist, sigma2_m, sigma2_p = get_all_params(param)

    # data moments
    d_stats = datadist.stats(moments='mvsk')
    mu = d_stats[0][()]
    sigma2 = d_stats[1][()]
    skew = d_stats[2][()]
    exkurt = d_stats[3][()]
    mu_3 = skew*np.sqrt(sigma2)**3
    mu_4 = (exkurt+3)*sigma2**2


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

    true_var = get_true_var(
        n_obs, mu, sigma2, mu_3, mu_4, const_a, const_b, const_c)
    true_var_i = get_true_var_i(
        n_obs, mu, sigma2, mu_3, mu_4, const_a, const_b, const_c)
    true_cov_ij = get_true_cov_ij(
        n_obs, mu, sigma2, mu_3, mu_4, const_a, const_b, const_c)

    naive_var_mean = n_obs*(true_var_i - true_cov_ij)

    ratios[i] = np.sqrt(naive_var_mean/true_var)

# ============================================================================
# plot stuff
# ============================================================================

plt.rcParams['font.family'] = 'serif'
plt.rcParams["mathtext.fontset"] = "dejavuserif"

plt.axhline(1, color='red', lw=1)
plt.plot(params, ratios)
plt.ylabel('ratio')
plt.xlabel(params_name)


print(min(ratios), '\n', max(ratios))
