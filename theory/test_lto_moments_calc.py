"""Test LTO omega and gamma."""

import numpy as np
from scipy import linalg
from scipy import stats


# conf
seed = 11
# n = 10
# n = 100
n = 1000
# n = 10000
d0 = 1
n_trial = 10000
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

beta = np.array([3.0]*d0 + [0.0, 3.0])

xs = rng.rand(n_trial, n, d0+2)
xs *= 2
xs -= 1

xs0 = xs[:,:,:d0]
xs1 = xs[:,:,:d0+1]

ys = stats.t.rvs(df=4, loc=xs.dot(beta), scale=1.0, random_state=rng)


# points of interest
ps = [0, 1, 2, 3]



# calc loos
loo_p1_s = np.empty((2, n_trial))
loo_p2_s = np.empty((2, n_trial))
lto_p12_s = np.empty((2, n_trial))
lto_p21_s = np.empty((2, n_trial))
lto_p13_s = np.empty((2, n_trial))
lto_p43_s = np.empty((2, n_trial))
for model_i, xs_cur in enumerate((xs0, xs1)):
    print('model:{}'.format(model_i))

    # full posterior for each trial
    Q_full_t = np.einsum('tia,tib->tab', xs_cur, xs_cur)
    r_full_t = np.einsum('tia,ti->ta', xs_cur, ys)
    # pointwise products for each trial and tested observation
    Q_t_ps = xs_cur[:,ps,:,None]*xs_cur[:,ps,None,:]
    r_t_ps = xs_cur[:,ps,:]*ys[:,ps,None]

    # leave-one-out loo
    print('loo')
    # for each trial
    for t in range(n_trial):
        if t%(n_trial//10) == 0:
            print('trial:{}'.format(t))
        for pi, loo_pi_s in zip((0, 1), (loo_p1_s, loo_p2_s)):
            idx = ps[pi]
            loo_pi_s[model_i, t] = calc_lpred(
                Q=Q_full_t[t]-Q_t_ps[t, pi],
                r=r_full_t[t]-r_t_ps[t, pi],
                x=xs_cur[t, idx],
                y=ys[t, idx]
            )

    # leave-two-out lto
    print('lto')
    # for each trial
    for t in range(n_trial):
        if t%(n_trial//10) == 0:
            print('trial:{}'.format(t))
        for (pi, pj), lto_pij_s in zip(
                ((0, 1), (1, 0), (0, 2), (3, 2)),
                (lto_p12_s, lto_p21_s, lto_p13_s, lto_p43_s)):
            idx_i = ps[pi]
            lto_pij_s[model_i, t] = calc_lpred(
                Q=Q_full_t[t]-Q_t_ps[t, pi]-Q_t_ps[t, pj],
                r=r_full_t[t]-r_t_ps[t, pi]-r_t_ps[t, pj],
                x=xs_cur[t, idx_i],
                y=ys[t, idx_i]
            )

# save
np.savez_compressed(
    'test_lto_moments_{}.npz'.format(n),
    loo_p1_s=loo_p1_s,
    loo_p2_s=loo_p2_s,
    lto_p12_s=lto_p12_s,
    lto_p21_s=lto_p21_s,
    lto_p13_s=lto_p13_s,
    lto_p43_s=lto_p43_s,
)
