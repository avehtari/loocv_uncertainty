
import numpy as np
from scipy import linalg, stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import matplotlib.gridspec as gridspec

# import seaborn as sns
# import pandas as pd

from setup_all import *


# ============================================================================
# select problems

# number of obs in one trial
# n_obs_s = [16, 32, 64, 128, 256, 512, 1024]
probl_n_obs_s = [32, 32, 32, 128, 128, 128, 512, 512, 512]

# last covariate effect not used in model A
# beta_t_s = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
probl_beta_t_s = [0.0, 0.1, 1.0, 0.0, 0.1, 1.0, 0.0, 0.1, 1.0]

# outlier dev
# out_dev_s = [0.0, 20.0, 200.0]
probl_out_dev_s = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# tau2
# tau2_s = [None, 1.0]
tau2 = None


# ============================================================================
# other config

cal_nbins = 7


# ============================================================================

n_probl = len(probl_n_obs_s)

loo_pti_A = np.empty(n_probl, dtype=object)
loo_pti_B = np.empty(n_probl, dtype=object)
elpd_pt_A = np.zeros((n_probl, N_TRIAL))
elpd_pt_B = np.zeros((n_probl, N_TRIAL))

for probl_i, (n_obs, beta_t, out_dev) in enumerate(zip(
        probl_n_obs_s, probl_beta_t_s, probl_out_dev_s)):
    run_i = params_to_run_i(n_obs, beta_t, out_dev, tau2)
    # load results
    res_file = np.load(
        '{}/{}.npz'
        .format(res_folder_name, str(run_i).zfill(4))
    )
    # fetch results
    loo_pti_A[probl_i] = res_file['loo_ti_A']
    loo_pti_B[probl_i] = res_file['loo_ti_B']
    elpd_pt_A[probl_i] = res_file['elpd_t_A']
    elpd_pt_B[probl_i] = res_file['elpd_t_B']
    # close file
    res_file.close()

# calc some normally obtainable values
loo_pti = loo_pti_A-loo_pti_B
loo_pt = np.array([np.sum(loo_ti, axis=-1) for loo_ti in loo_pti])
loo_var_hat_pt = np.array(
    [loo_ti.shape[-1]*np.var(loo_ti, ddof=1, axis=-1) for loo_ti in loo_pti])
loo_skew_hat_pt = np.array(
    [stats.skew(loo_ti, axis=-1, bias=False) for loo_ti in loo_pti])
loo_napprox_pneg_pt = stats.norm.cdf(
    0, loc=loo_pt, scale=np.sqrt(loo_var_hat_pt))

# elpd
elpd_pt = elpd_pt_A - elpd_pt_B

# elpd(y)
elpd_y_mean_p = np.mean(elpd_pt, axis=-1)
elpd_y_var_p = np.var(elpd_pt, ddof=1, axis=-1)
elpd_y_skew_p = stats.skew(elpd_pt, bias=False, axis=-1)
elpd_y_pneg_p = np.mean(elpd_pt<0, axis=-1)

# elpdhat(y)
loo_y_mean_p = np.mean(loo_pt, axis=-1)
loo_y_var_p = np.var(loo_pt, ddof=1, axis=-1)
loo_y_skew_p = stats.skew(loo_pt, bias=False, axis=-1)
loo_y_pneg_p = np.mean(loo_pt<0, axis=-1)

# calibration
cal_limits = np.linspace(0, 1, cal_nbins+1)
# normal approx
loo_n_cal_counts = np.zeros((n_probl, cal_nbins), dtype=int)
for probl_i in range(n_probl):
    cdf_elpd = stats.norm.cdf(
        elpd_pt[probl_i],
        loc=loo_pt[probl_i],
        scale=np.sqrt(loo_var_hat_pt[probl_i])
    )
    loo_n_cal_counts[probl_i] = np.histogram(cdf_elpd, cal_limits)[0]
# BB
bb_n = 1000
bb_seed = 1234
loo_bb_cal_counts = np.zeros((n_probl, cal_nbins), dtype=int)
for probl_i in range(n_probl):
    loo_ti = loo_pti[probl_i]
    n_trial, n_obs = loo_ti.shape
    cdf_elpd = np.zeros(n_trial)
    rng = np.random.RandomState(seed=bb_seed)
    weights = rng.dirichlet(np.ones(n_obs), size=bb_n)
    for trial_i in range(n_trial):
        temp = weights.dot(loo_ti[trial_i])
        temp *= n_obs
        cdf_elpd[trial_i] = np.mean(temp < elpd_pt[probl_i][trial_i])
    loo_bb_cal_counts[probl_i] = np.histogram(cdf_elpd, cal_limits)[0]
# BB mirror
bb_n = 1000
bb_seed = 1234
loo_bbmir_cal_counts = np.zeros((n_probl, cal_nbins), dtype=int)
for probl_i in range(n_probl):
    loo_ti = loo_pti[probl_i]
    n_trial, n_obs = loo_ti.shape
    cdf_elpd = np.zeros(n_trial)
    rng = np.random.RandomState(seed=bb_seed)
    weights = rng.dirichlet(np.ones(n_obs), size=bb_n)
    for trial_i in range(n_trial):
        temp = weights.dot(loo_ti[trial_i])
        temp *= n_obs
        temp = 2*np.mean(temp) - temp
        cdf_elpd[trial_i] = np.mean(temp < elpd_pt[probl_i][trial_i])
    loo_bbmir_cal_counts[probl_i] = np.histogram(cdf_elpd, cal_limits)[0]
# elpdhat(y)
loo_elpdhaty_cal_counts = np.zeros((n_probl, cal_nbins), dtype=int)
for probl_i in range(n_probl):
    loo_t = loo_pt[probl_i]
    loo_t_center = loo_t - np.mean(loo_t)
    elpd_hat_unk_hat_t = loo_t[:,None] + loo_t_center
    cdf_elpd = np.mean(elpd_hat_unk_hat_t<elpd_pt[probl_i][:,None], axis=-1)
    loo_elpdhaty_cal_counts[probl_i] = np.histogram(cdf_elpd, cal_limits)[0]
# mirror elpdhat(y)
loo_mirror_cal_counts = np.zeros((n_probl, cal_nbins), dtype=int)
for probl_i in range(n_probl):
    loo_t = loo_pt[probl_i]
    loo_t_mirror_center = np.mean(loo_t) - loo_t
    elpd_hat_unk_hat_t = loo_t[:,None] + loo_t_mirror_center
    cdf_elpd = np.mean(elpd_hat_unk_hat_t<elpd_pt[probl_i][:,None], axis=-1)
    loo_mirror_cal_counts[probl_i] = np.histogram(cdf_elpd, cal_limits)[0]


# ============================================================================
# plot stuff
# ============================================================================

plt.rcParams['font.family'] = 'serif'
plt.rcParams["mathtext.fontset"] = "dejavuserif"

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

q005 = stats.binom.ppf(0.005, n_trial, 1/cal_nbins)
q995 = stats.binom.ppf(0.995, n_trial, 1/cal_nbins)

# ============================================================================
# plot cal hists
# ============================================================================

fig, axes = plt.subplots(
    n_probl, 4, sharex=True, sharey='row', figsize=(8, 15))

col_data = np.array([
    loo_n_cal_counts,
    loo_bb_cal_counts,
    loo_bbmir_cal_counts,
    loo_mirror_cal_counts,
])
col_names = [
    'normal approx.',
    'bb',
    'bb mirror',
    'elpd_hat(y) mirror',
]

for probl_i in range(n_probl):

    for col_i, ax in enumerate(axes[probl_i]):
        data = col_data[col_i][probl_i]
        col_name = col_names[col_i]

        ax.bar(
            cal_limits[:-1],
            data,
            width=1/cal_nbins,
            align='edge'
        )

for ax in axes.ravel():
    ax.fill_between(
        [0,1], [q995, q995], [q005, q005], color='C1', alpha=0.3,
        zorder=2)
    ax.set_xlim((0, 1))
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

for ax, name in zip(axes[0, :], col_names):
    ax.set_title(name)

for probl_i, (n_obs, beta_t, out_dev) in enumerate(zip(
        probl_n_obs_s, probl_beta_t_s, probl_out_dev_s)):
    axes[probl_i,0].set_ylabel(
        r'$n={}$'.format(n_obs) + '\n' +
        r'$\beta_t={}$'.format(beta_t) + '\n' +
        'out_dev={}'.format(out_dev),
        rotation=0,
        ha='right',
        va='center',
    )

fig.tight_layout()
