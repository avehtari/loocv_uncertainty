"""Test LTO omega and gamma."""

import numpy as np
from scipy import linalg
from scipy import stats

import matplotlib.pyplot as plt

from bayesian_bootstrap.bootstrap import (
    mean as bb_mean,
    var as bb_var,
    covar as bb_covar
)

# conf
seed = 11
# n = 10
# n = 100
# n = 1000
n = 10000
d0 = 1
n_trial = 10000
PLOT = False

bb_n = n_trial


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

# load precalculated loos
saved_file = np.load('test_lto_moments_{}.npz'.format(n))
loo_p1_s = saved_file['loo_p1_s']
loo_p2_s = saved_file['loo_p2_s']
lto_p12_s = saved_file['lto_p12_s']
lto_p21_s = saved_file['lto_p21_s']
lto_p13_s = saved_file['lto_p13_s']
lto_p43_s = saved_file['lto_p43_s']
saved_file.close()

# calc the diff into the third dim
loo_p1_s = np.concatenate(
    (loo_p1_s, loo_p1_s[None, 0]-loo_p1_s[None, 1]), axis=0)
loo_p2_s = np.concatenate(
    (loo_p2_s, loo_p2_s[None, 0]-loo_p2_s[None, 1]), axis=0)
lto_p12_s = np.concatenate(
    (lto_p12_s, lto_p12_s[None, 0]-lto_p12_s[None, 1]), axis=0)
lto_p21_s = np.concatenate(
    (lto_p21_s, lto_p21_s[None, 0]-lto_p21_s[None, 1]), axis=0)
lto_p13_s = np.concatenate(
    (lto_p13_s, lto_p13_s[None, 0]-lto_p13_s[None, 1]), axis=0)
lto_p43_s = np.concatenate(
    (lto_p43_s, lto_p43_s[None, 0]-lto_p43_s[None, 1]), axis=0)

# pack
arrs = (loo_p1_s, loo_p2_s, lto_p12_s, lto_p21_s, lto_p13_s, lto_p43_s)


# ====== moments

def sample_cov(ar1, ar2):
    out = np.einsum(
        'ma,ma->m',
        (ar1 - np.mean(ar1, axis=1)[:,None]),
        (ar2 - np.mean(ar2, axis=1)[:,None])
    )
    out /= (n_trial - 1)
    return out

loo_mean_i = np.mean(loo_p1_s, axis=1)
loo_var_i = np.var(loo_p1_s, ddof=1, axis=1)
loo_cov_ij = sample_cov(loo_p1_s, loo_p2_s)

lto_mean_ij = np.mean(lto_p12_s, axis=1)
lto_var_ij = np.var(lto_p12_s, ddof=1, axis=1)
lto_cov_ij_ji = sample_cov(lto_p12_s, lto_p21_s)
lto_cov_ij_ik = sample_cov(lto_p12_s, lto_p13_s)
lto_cov_ij_kj = sample_cov(lto_p13_s, lto_p43_s)
lto_cov_ij_jk = sample_cov(lto_p21_s, lto_p13_s)
lto_cov_ij_kl = sample_cov(lto_p12_s, lto_p43_s)


# bayesian bootstrap
if True:
    # load precalculated
    saved_file = np.load('test_lto_moments_{}_bb.npz'.format(n))
    loo_mean_i_bb = saved_file['loo_mean_i_bb']
    loo_var_i_bb = saved_file['loo_var_i_bb']
    loo_cov_ij_bb = saved_file['loo_cov_ij_bb']
    lto_mean_ij_bb = saved_file['lto_mean_ij_bb']
    lto_var_ij_bb = saved_file['lto_var_ij_bb']
    lto_cov_ij_ji_bb = saved_file['lto_cov_ij_ji_bb']
    lto_cov_ij_ik_bb = saved_file['lto_cov_ij_ik_bb']
    lto_cov_ij_kj_bb = saved_file['lto_cov_ij_kj_bb']
    lto_cov_ij_jk_bb = saved_file['lto_cov_ij_jk_bb']
    lto_cov_ij_kl_bb = saved_file['lto_cov_ij_kl_bb']
    saved_file.close()

else:
    # calculate and save
    loo_mean_i_bb = np.array(tuple(
        bb_mean(loo_p1_s[model_i], bb_n)
        for model_i in range(3)
    ))
    loo_var_i_bb = np.array(tuple(
        bb_var(loo_p1_s[model_i], bb_n)
        for model_i in range(3)
    ))
    loo_cov_ij_bb = np.array(tuple(
        bb_covar(loo_p1_s[model_i], loo_p2_s[model_i], bb_n)
        for model_i in range(3)
    ))

    lto_mean_ij_bb = np.array(tuple(
        bb_mean(lto_p12_s[model_i], bb_n)
        for model_i in range(3)
    ))
    lto_var_ij_bb = np.array(tuple(
        bb_var(lto_p12_s[model_i], bb_n)
        for model_i in range(3)
    ))
    lto_cov_ij_ji_bb = np.array(tuple(
        bb_covar(lto_p12_s[model_i], lto_p21_s[model_i], bb_n)
        for model_i in range(3)
    ))
    lto_cov_ij_ik_bb = np.array(tuple(
        bb_covar(lto_p12_s[model_i], lto_p13_s[model_i], bb_n)
        for model_i in range(3)
    ))
    lto_cov_ij_kj_bb = np.array(tuple(
        bb_covar(lto_p13_s[model_i], lto_p43_s[model_i], bb_n)
        for model_i in range(3)
    ))
    lto_cov_ij_jk_bb = np.array(tuple(
        bb_covar(lto_p21_s[model_i], lto_p13_s[model_i], bb_n)
        for model_i in range(3)
    ))
    lto_cov_ij_kl_bb = np.array(tuple(
        bb_covar(lto_p12_s[model_i], lto_p43_s[model_i], bb_n)
        for model_i in range(3)
    ))
    # save
    np.savez_compressed(
        'test_lto_moments_{}_bb.npz'.format(n),
        loo_mean_i_bb=loo_mean_i_bb,
        loo_var_i_bb=loo_var_i_bb,
        loo_cov_ij_bb=loo_cov_ij_bb,
        lto_mean_ij_bb=lto_mean_ij_bb,
        lto_var_ij_bb=lto_var_ij_bb,
        lto_cov_ij_ji_bb=lto_cov_ij_ji_bb,
        lto_cov_ij_ik_bb=lto_cov_ij_ik_bb,
        lto_cov_ij_kj_bb=lto_cov_ij_kj_bb,
        lto_cov_ij_jk_bb=lto_cov_ij_jk_bb,
        lto_cov_ij_kl_bb=lto_cov_ij_kl_bb,
    )


# ============================================================================
# plot covs

fig, axes = plt.subplots(5, 2, sharex='col')
# fig.suptitle('bb for covs')
for col_i, model_i in enumerate((0, 2)):

    ax = axes[0, col_i]
    ax.hist(loo_cov_ij_bb[model_i], bins=30)
    ax.axvline(loo_cov_ij[model_i], color='orange')
    ax.set_title('Model: {}'.format(model_i))
    if col_i == 0:
        ax.set_ylabel('i j', rotation=0, size='large', ha='right')

    ax = axes[1, col_i]
    ax.hist(lto_cov_ij_ji_bb[model_i], bins=30)
    ax.axvline(lto_cov_ij_ji[model_i], color='orange')
    if col_i == 0:
        ax.set_ylabel('ij ji', rotation=0, size='large', ha='right')

    ax = axes[2, col_i]
    ax.hist(lto_cov_ij_kj_bb[model_i], bins=30)
    ax.axvline(lto_cov_ij_kj[model_i], color='orange')
    if col_i == 0:
        ax.set_ylabel('ij kj', rotation=0, size='large', ha='right')

    ax = axes[3, col_i]
    ax.hist(lto_cov_ij_jk_bb[model_i], bins=30)
    ax.axvline(lto_cov_ij_jk[model_i], color='orange')
    if col_i == 0:
        ax.set_ylabel('ij jk', rotation=0, size='large', ha='right')

    ax = axes[4, col_i]
    ax.hist(lto_cov_ij_kl_bb[model_i], bins=30)
    ax.axvline(lto_cov_ij_kl[model_i], color='orange')
    if col_i == 0:
        ax.set_ylabel('ij kl', rotation=0, size='large', ha='right')

    # ax = axes[5, col_i]
    # ax.hist(lto_cov_ij_ik_bb[model_i], bins=30)
    # ax.axvline(lto_cov_ij_ik[model_i], color='orange')
    # if col_i == 0:
    #     ax.set_ylabel('lto_ij_ik', rotation=0, size='large', ha='right')

for ax in axes.ravel():
    ax.set_yticks([], [])

lims = np.array((
    np.min(np.percentile(
        (loo_cov_ij_bb,
        lto_cov_ij_ji_bb,
        lto_cov_ij_kj_bb,
        lto_cov_ij_jk_bb,
        lto_cov_ij_kl_bb),
        2.5, axis=-1
    ), axis=0),
    np.max(np.percentile(
        (loo_cov_ij_bb,
        lto_cov_ij_ji_bb,
        lto_cov_ij_kj_bb,
        lto_cov_ij_jk_bb,
        lto_cov_ij_kl_bb),
        97.5, axis=-1
    ), axis=0)
)).T
for col_i, model_i in enumerate((0, 2)):
    axes[0,col_i].set_xlim(lims[model_i])

fig.tight_layout()
plt.show()



# ============================================================================
# plot vars

fig, axes = plt.subplots(3, 2, sharex='col')
# fig.suptitle('bb for vars')
for col_i, model_i in enumerate((0, 2)):

    ax = axes[0, col_i]
    ax.hist(loo_var_i_bb[model_i], bins=30)
    ax.axvline(loo_var_i[model_i], color='orange')
    ax.set_title('Model: {}'.format(model_i))
    if col_i == 0:
        ax.set_ylabel('i', rotation=0, size='large', ha='right')

    ax = axes[1, col_i]
    ax.hist(lto_var_ij_bb[model_i], bins=30)
    ax.axvline(lto_var_ij[model_i], color='orange')
    if col_i == 0:
        ax.set_ylabel('ij', rotation=0, size='large', ha='right')

    ax = axes[2, col_i]
    ax.hist(lto_cov_ij_ik_bb[model_i], bins=30)
    ax.axvline(lto_cov_ij_ik[model_i], color='orange')
    if col_i == 0:
        ax.set_ylabel('ij ik', rotation=0, size='large', ha='right')

for ax in axes.ravel():
    ax.set_yticks([], [])

if False:
    lims = np.array((
        np.min(np.percentile(
            (loo_var_i_bb,
            lto_var_ij_bb,
            lto_cov_ij_ik_bb),
            2.5, axis=-1
        ), axis=0),
        np.max(np.percentile(
            (loo_var_i_bb,
            lto_var_ij_bb,
            lto_cov_ij_ik_bb),
            97.5, axis=-1
        ), axis=0)
    )).T
    for col_i, model_i in enumerate((0, 2)):
        axes[0,col_i].set_xlim(lims[model_i])

fig.tight_layout()
plt.show()
