import numpy as np
from scipy import linalg
from scipy import stats
from scipy.special import gamma, beta


# conf
seed = 11
# n = 10
# n = 100
n = 1000
# n = 10000
n_trial = 1000
PLOT = False


num_of_pairs = (n**2-n) // 2

LOG2PI = np.log(2*np.pi)
def norm_logpdf(x, mu, sigma2):
    """Univariate normal logpdf."""
    return -0.5*(LOG2PI + np.log(sigma2) + (x-mu)**2/sigma2)

def calc_lpred(Q, r, x, y):
    x_S = linalg.solve(Q, x, assume_a='pos')
    mu_pred = x_S.dot(r)
    sigma2_pred = x_S.dot(x) + 1
    return norm_logpdf(y, mu_pred, sigma2_pred)

rng = np.random.RandomState(seed=seed)

# data
n = np.round(np.linspace(10, 1000, 50)).astype(int)
mu = 20.0
s2 = 1.0
skew = -4
ex_kurt = 15
# model
s2_m = 20.0
# prior
s2_0 = 1.0

assert ex_kurt+3 >= skew**2 + 1

m_3 = skew*np.sqrt(s2)**3
m_4 = (ex_kurt+3)*s2**2

tau = 1/(1/s2_0+n/s2_m)
s2_pp = s2_m + tau

a = -1/(2*s2_pp)
b = tau/(s2_m*s2_pp)*n
c = -tau**2/(2*s2_m**2*s2_pp)*n**2
d = -0.5*np.log(2*np.pi*s2_pp)

a2 = a**2
b2 = b**2
c2 = c**2

ab = a*b
ac = a*c
bc = b*c

var = (
    (4*a2 + n/(n-1)*b2 + 4/(n-1)*c2 + 4*ab + 4/(n-1)*bc)*mu**2*s2
    + (-a2 + 1/(n-1)*b2 + (2*n-5)/((n-1)**3)*c2)*s2**2
    + (4*a2 + 4/((n-1)**2)*c2 + 2*ab + 2/((n-1)**2)*bc)*mu*m_3
    + (a2 + 1/((n-1)**3)*c2)*m_4
)

cov = (
    (
        (3*n-4)/((n-1)**2)*b2
        +(4*(n-2))/((n-1)**2)*c2
        +4/(n-1)*ab
        +8/(n-1)*ac
        +(4*(2*n-3))/((n-1)**2)*bc
    )*mu**2*s2
    + (
        1/((n-1)**2)*b2
        +((n-2)*(2*n-7))/((n-1)**4)*c2
        -2/((n-1)**2)*ac
        +(4*(n-2))/((n-1)**3)*bc
    )*s2**2
    + (
        (4*(n-2))/((n-1)**3)*c2
        +2/(n-1)*ab
        +(4*n)/((n-1)**2)*ac
        +(4*n-6)/((n-1)**3)*bc
    )*mu*m_3
    + (
        (n-2)/((n-1)**4)*c2
        +2/((n-1)**2)*ac
    )*m_4
)



# plot
fig, axes = plt.subplots(2, 1, sharex=True)

ax = axes[0]
ax.plot(n, np.array([var, cov]).T)
# ax.set_xlabel('n')
ax.legend(['Var(loo_i)', 'Cov(loo_i, loo_j)'])

ax = axes[1]
ax.plot(n, (n-1)*cov/var)
ax.set_xlabel('n')
ax.set_title('ratio of n(n-1) cov : n var')

fig.tight_layout()



# sample random

from skewstudent import SkewStudent

mu = -2
s2 = 1.6
lam = -0.7
q = 2.4  # must be q > 2



s = np.sqrt(s2)
v = 1/np.sqrt(
    q*((3*lam**2+1)/(2*q-2) - 4*lam**2/np.pi*(gamma(q-0.5)/gamma(q))**2))

beta1p = beta(0.5,q)
beta2p = gamma(q-0.5)/gamma(q+0.5)
beta3p = beta(1.5, q-1)
beta4p = gamma(q-1.5)/gamma(q+0.5)
beta5p = beta(2.5, q-2)

mu_3 = (2*q**1.5*lam*(v*s)**3)/beta1p**3*(
    8*lam**2*beta2p**3
    -3*(1+3*lam**2)*beta1p*beta2p*beta3p
    +2*(1+lam**2)*beta1p**2*beta4p
)
mu_4 = (q**2*(v*s)**4)/beta1p**4*(
    -48*lam**4*beta2p**4
    +24*lam**2*(1+3*lam**2)*beta1p*beta2p**2*beta3p
    -32*lam**2*(1+lam**2)*beta1p**2*beta2p*beta4p
    +(1+10*lam**2+5*lam**4)*beta1p**3*beta5p
)
