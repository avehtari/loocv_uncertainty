
import numpy as np
from scipy import linalg, stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import matplotlib.gridspec as gridspec

# import seaborn as sns
# import pandas as pd

from setup_out import *


# ============================================================================
# select problems

# number of obs in one trial
# n_obs_s = [32, 128]
n_obs = 128

# last covariate effect not used in model A
# beta_t_s = [0.0, 1.0]
beta_t = 1.0

# outlier dev
out_dev_s = [0.0, 1.0, 2.0, 4.0, 10.0, 20.0, 40.0, 100.0, 1000.0]

# tau2
# tau2_s = [None, 1.0]
tau2 = None


# ============================================================================

n_probl = len(out_dev_s)
n_trial = None
for probl_i, out_dev in enumerate(out_dev_s):

    run_i = params_to_run_i(n_obs, beta_t, out_dev, tau2)

    # load results
    res_file = np.load(
        '{}/{}.npz'
        .format(res_folder_name, str(run_i).zfill(4))
    )

    # first file > check n_trial
    if n_trial is None:
        n_trial = int(res_file['probl_args'][11])
        loo_pti_A = np.zeros((n_probl, n_trial, n_obs))
        loo_pti_B = np.zeros((n_probl, n_trial, n_obs))
        elpd_pt_A = np.zeros((n_probl, n_trial))
        elpd_pt_B = np.zeros((n_probl, n_trial))

    # fetch results
    loo_pti_A[probl_i] = res_file['loo_ti_A']
    loo_pti_B[probl_i] = res_file['loo_ti_B']
    elpd_pt_A[probl_i] = res_file['elpd_t_A']
    elpd_pt_B[probl_i] = res_file['elpd_t_B']

    # close file
    res_file.close()

# calc some normally obtainable values
loo_pti = loo_pti_A-loo_pti_B
loo_pt = np.sum(loo_pti, axis=-1)
loo_var_hat_pt = n_obs*np.var(loo_pti, ddof=1, axis=-1)
loo_skew_hat_pt = stats.skew(loo_pti, axis=-1, bias=False)
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

title_str = 'n_obs={}, beta_t={}, tau2={}'.format(n_obs, beta_t, tau2)


# ============================================================================
# plot means
# ============================================================================

plt.figure()
plt.plot(out_dev_s, np.zeros(n_probl), label=None, color='r')
plt.plot(out_dev_s, loo_y_mean_p, label='loo_y_mean')
plt.plot(out_dev_s, elpd_y_mean_p, label='elpd_y_mean')
plt.xlabel('out_dev')
plt.title(title_str, fontsize=10)
plt.legend()
