"""Analyse LOOCV results

First run results with m1_run.py

"""

import numpy as np
from scipy import linalg, stats

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import seaborn as sns


from m1_problem import *


# ============================================================================
# config

# seed for randomisation
# np.random.RandomState().randint(np.iinfo(np.uint32).max)
seed_analysis = 4257151306

# fixed or unfixed sigma
fixed_sigma2_m = False

# plot using seaborn
use_sea = True

# bootstrap on LOO_i
n_booti_trial = 1000

# calibration nbins
cal_nbins = 5



# ============================================================================
# Select a lot of problems

# grid dims:
#   sigma2_d  [0.01, 1.0, 100.0]
#   n_obs     [16, 32, 64, 128, 256, 512, 1024]
#   beta_t    [0.0, 0.01, 0.1, 1.0, 10.0]
#   prc_out   [0/128, eps, 2/128, 12/128]

# as a function of beta_t with no out and out
idxs = list(map(
    np.ravel,
    np.meshgrid(
        [1],
        [1, 2, 3, 4, 5],
        np.arange(grid_shape[2]),
        [0],
    )
))
run_i_s = np.ravel_multi_index(idxs, grid_shape)

# ============================================================================

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def remove_frame(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

# ============================================================================

if fixed_sigma2_m:
    folder_name = 'fixed'
else:
    folder_name = 'unfixed'

rng = np.random.RandomState(seed=seed_analysis)

n_probls = len(run_i_s)


loo_s = np.zeros((n_probls, n_trial))
naive_var_s = np.zeros((n_probls, n_trial))
cor_loo_i_s = np.zeros((n_probls, n_trial))
skew_loo_i_s = np.zeros((n_probls, n_trial))

# calc some target values
# target_mean_s = np.zeros((n_probls))
# target_var_s = np.zeros((n_probls))
target_skew_s = np.zeros((n_probls))
target_plooneg_s = np.zeros((n_probls))
elpd_s = np.zeros((n_probls, n_trial))

bma_s = np.zeros((n_probls, n_trial, 2))
bma_elpd_s = np.zeros((n_probls, n_trial, 2))


# load results
res_A = np.empty(n_probls, dtype=object)
res_B = np.empty(n_probls, dtype=object)
res_test_A = np.empty(n_probls, dtype=object)
res_test_B = np.empty(n_probls, dtype=object)
for probl_i in range(n_probls):
    run_i = run_i_s[probl_i]
    res_file = np.load(
        'res_1/{}/{}.npz'
        .format(folder_name, str(run_i).zfill(4))
    )
    # fetch results
    res_A[probl_i] = res_file['loo_ti_A']
    res_B[probl_i] = res_file['loo_ti_B']
    res_test_A[probl_i] = res_file['test_elpd_t_A']
    res_test_B[probl_i] = res_file['test_elpd_t_B']
    # close file
    res_file.close()



    # calc some normally obtainable values

for probl_i in range(n_probls):
    n_cur = res_A[probl_i].shape[-1]
    loo_s[probl_i] = np.sum(res_A[probl_i]-res_B[probl_i], axis=-1)
    naive_var_s[probl_i] = n_cur*np.var(
        res_A[probl_i]-res_B[probl_i], ddof=1, axis=-1)
    for trial_i in range(n_trial):
        cor_loo_i_s[probl_i, trial_i] = np.corrcoef(
            res_A[probl_i][trial_i], res_B[probl_i][trial_i])[0, 1]
    skew_loo_i_s[probl_i] = stats.skew(
        res_A[probl_i]-res_B[probl_i], axis=-1, bias=False)
# naive_coef_var_s = np.sqrt(naive_var_s)/loo_s
naive_plooneg_s = stats.norm.cdf(0, loc=loo_s, scale=np.sqrt(naive_var_s))


for probl_i in range(n_probls):
    # target_mean_s[probl_i] = np.mean(loo_s[probl_i])
    # target_var_s[probl_i] = np.var(loo_s[probl_i], ddof=1)
    target_skew_s[probl_i] = stats.skew(loo_s[probl_i], bias=False)
    # TODO calc se of this ... formulas online
    target_plooneg_s[probl_i] = np.mean(loo_s[probl_i]<0)
    elpd_s[probl_i] = res_test_A[probl_i] - res_test_B[probl_i]
    # bma
    loo_tki = np.stack((res_A[probl_i], res_B[probl_i]), axis=1)
    bma_s[probl_i] = pseudo_bma_p(loo_tki)
    # bma_pair_s[o_i, b_i, n_i] = pseudo_bma_p_pair(loo_i)
    elpd_tk = np.stack((res_test_A[probl_i], res_test_B[probl_i]), axis=1)
    bma_elpd_s[probl_i] = pseudo_bma(elpd_tk)
pelpdneg_s = np.mean(elpd_s<0, axis=-1)
# target_coefvar_s = np.sqrt(target_var_s)/target_mean_s

# misspred: TODO replace with something
naive_misspred_s = np.abs(naive_plooneg_s-pelpdneg_s[:,None])


del(res_A, res_B, res_test_A, res_test_B)




# ===========================================================================
# plot


# loo_i_cors vs |naive_plooneg-plooneg|
plt.plot(cor_loo_i_s.flat, naive_misspred_s.flat, '.')


idxs = (np.abs(loo_s.ravel())>4.0)
plt.plot(bma_s[:,:,0].ravel()[idxs], 1-naive_plooneg_s.ravel()[idxs], '.')
plt.plot([0,1], [0,1], 'red')


idxs = (np.abs(loo_s.ravel())>4.0)
data_x = np.abs(bma_s[:,:,0].ravel()[idxs] - (1-naive_plooneg_s.ravel()[idxs]))
data_y = np.abs(naive_plooneg_s-pelpdneg_s[:,None]).ravel()[idxs]
plt.plot(data_x, data_y, '.')

sns.jointplot(data_x[data_x>0.02], data_y[data_x>0.02],  kind="hex", color="k")


###
# find idx for big diff in bme vs p SE
(bma_s[:,:,0].ravel() - (1-naive_plooneg_s).ravel()).argmax()
idx = (bma_s[:,:,0].ravel() - (1-naive_plooneg_s).ravel()).argmax()
np.unravel_index(idx, grid_shape+(n_trial,))


###

idxs = (np.abs(loo_s.ravel())>4.0)
plt.plot(
    cor_loo_i_s.ravel()[idxs],
    bma_elpd_s[:,:,0].ravel()[idxs] - bma_s[:,:,0].ravel()[idxs],
    '.'
)

data_x = cor_loo_i_s.ravel()[idxs]
data_y = bma_elpd_s[:,:,0].ravel()[idxs] - bma_s[:,:,0].ravel()[idxs]
sns.jointplot(
    data_x[np.abs(data_y)>0.08],
    data_y[np.abs(data_y)>0.08],
    kind='hex',
    color='k'
)








sns.jointplot(cor_loo_i_s.flat, naive_misspred_s.flat, kind="hex", color="k")

# 2d histogram one or nothing
counts, x_edges, y_edges = np.histogram2d(
    cor_loo_i_s.ravel(), naive_misspred_s.ravel(), 50)
found = counts > 0
plt.imshow(found.T, origin='lower',
    extent=(x_edges[0], x_edges[-1], y_edges[0], x_edges[-1]),
    cmap='Greys'
)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.hist2d(cor_loo_i_s.ravel(), naive_misspred_s.ravel(), 50, cmin=1,
    cmap=truncate_colormap(cm.get_cmap('Greens'), 0.4))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlim((-0.5, 1.1))
plt.ylim((-0.1, 1.1))
plt.xlabel('$\mathrm{Corr}(\pi_{a,\,i}, \pi_{b,\,i})$')
plt.ylabel('sss')
