
import sys, os, time
from functools import partial

import numpy as np
from scipy import linalg, stats

from problem_setting import *



# ============================================================================
# conf

folder_name = 'res_skew_b_n'
run_moments = False
distributed = True
plot = True

# data seed
# np.random.RandomState().randint(np.iinfo(np.uint32).max)
data_seed = 247169102

# number of trials
n_trial = 2000
# fixed model tau2 value
tau2 = 1.0
# first dim intercept
intercept = True


# dimensionality of X
n_dim_s = [3, 4, 4]
# model A covariates
idx_a_s = [
    [0,1],
    [0,1,2],
    [0,1,2],
]
# model B covariates
idx_b_s = [
    [0,1,2],
    [0,1,3],
    [0,1,3],
]
beta_rate_s = [
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 0.5, 1.0],
]

# constant data variance
sigma_d_2 = 1.0

# grid parameters
n_obs_s = np.array([16, 64, 256])
# n_obs_s = np.array([16, 64])
# last beta effect missing in model A
beta_r_s = np.linspace(0, 8, 51)



# ============================================================================

if plot:
    import matplotlib.pyplot as plt
    # import seaborn as sns

# ============================================================================
# Funcs

class DataGeneration:

    def __init__(self, n_trial, data_seed=None):
        self.rng = np.random.RandomState(seed=data_seed)
        if intercept:
            # firs dim (column) ones for intercept
            self.X_mat_all = np.concatenate(
                (
                    np.ones((n_trial, np.max(n_obs_s), 1)),
                    self.rng.randn(n_trial, np.max(n_obs_s), np.max(n_dim_s)-1)
                ),
                axis=-1,
            )
        else:
            self.X_mat_all = self.rng.randn(
                n_trial, np.max(n_obs_s), np.max(n_dim_s))

    def get_x(self, probl_i, trial_i, n_obs):
        return self.X_mat_all[trial_i, :n_obs, :n_dim_s[probl_i]]

# ============================================================================

data_generation = DataGeneration(n_trial, data_seed)

n_probl = len(n_dim_s)

beta_rate_s = list(map(np.array, beta_rate_s))

# ============================================================================
# setup the grid

# set grid
n_obs_grid, beta_r_grid = np.meshgrid(n_obs_s, beta_r_s, indexing='ij')
grid_shape = n_obs_grid.shape
n_runs = n_obs_grid.size

def run_i_to_params(run_i):
    n_obs = n_obs_grid.flat[run_i]
    beta_r = beta_r_grid.flat[run_i]
    return n_obs, beta_r

def params_to_run_i(n_obs, beta_r):
    n_obs_i = np.nonzero(n_obs_s == n_obs)[0][0]
    beta_r_i = np.nonzero(beta_r_s == beta_r)[0][0]
    run_i = np.ravel_multi_index((n_obs_i, beta_r_i), grid_shape)
    return run_i


# ============================================================================
# As a function of n

def run_and_save_run_i(run_i):
    run_i_str = str(run_i).zfill(4)

    n_obs, beta_r = run_i_to_params(run_i)

    # mean_loo_t = np.full((n_probl, n_trial,), np.nan)
    # var_loo_t = np.full((n_probl, n_trial,), np.nan)
    # moment3_loo_t = np.full((n_probl, n_trial,), np.nan)
    # mean_elpd_t = np.full((n_probl, n_trial,), np.nan)
    # var_elpd_t = np.full((n_probl, n_trial,), np.nan)
    # moment3_elpd_t = np.full((n_probl, n_trial,), np.nan)
    # mean_err_t = np.full((n_probl, n_trial,), np.nan)
    # var_err_t = np.full((n_probl, n_trial,), np.nan)
    # moment3_err_t = np.full((n_probl, n_trial,), np.nan)

    skew_loo_t = np.full((n_probl, n_trial,), np.nan)
    skew_elpd_t = np.full((n_probl, n_trial,), np.nan)
    skew_err_t = np.full((n_probl, n_trial,), np.nan)

    for t_i in range(n_trial):
        for probl_i in range(n_probl):
            X_mat = data_generation.get_x(probl_i, t_i, n_obs)
            beta = beta_r*beta_rate_s[probl_i]
            idx_a = idx_a_s[probl_i]
            idx_b = idx_b_s[probl_i]
            (
                mean_loo, var_loo, moment3_loo,
                mean_elpd, var_elpd, moment3_elpd,
                mean_err, var_err, moment3_err
            ) = (
                get_analytic_res(
                    X_mat, beta, tau2, idx_a, idx_b,
                    Sigma_d=sigma_d_2, mu_d=None
                )
            )
            # mean_loo_t[probl_i, t_i] = mean_loo
            # var_loo_t[probl_i, t_i] = var_loo
            # moment3_loo_t[probl_i, t_i] = moment3_loo
            # mean_elpd_t[probl_i, t_i] = mean_elpd
            # var_elpd_t[probl_i, t_i] = var_elpd
            # moment3_elpd_t[probl_i, t_i] = moment3_elpd
            # mean_err_t[probl_i, t_i] = mean_err
            # var_err_t[probl_i, t_i] = var_err
            # moment3_err_t[probl_i, t_i] = moment3_err

            skew_loo_t[probl_i, t_i] = moment3_loo/np.sqrt(var_loo)**3
            skew_elpd_t[probl_i, t_i] = moment3_elpd/np.sqrt(var_elpd)**3
            skew_err_t[probl_i, t_i] = moment3_err/np.sqrt(var_err)**3

    np.savez_compressed(
        folder_name+'/'+'res_'+run_i_str+'.npz',
        # mean_loo_t=mean_loo_t,
        # var_loo_t=var_loo_t,
        # moment3_loo_t=moment3_loo_t,
        # mean_elpd_t=mean_elpd_t,
        # var_elpd_t=var_elpd_t,
        # moment3_elpd_t=moment3_elpd_t,
        # mean_err_t=mean_err_t,
        # var_err_t=var_err_t,
        # moment3_err_t=moment3_err_t,
        skew_loo_t=skew_loo_t,
        skew_elpd_t=skew_elpd_t,
        skew_err_t=skew_err_t,
    )

if run_moments:
    # os.makedirs(folder_name, exist_ok=True)  # make beforehand
    if distributed:
        # parse cmd input for run id
        if len(sys.argv) > 1:
            # get run_i
            run_i = int(sys.argv[1])
        else:
            raise ValueError('Provide run_i as cmd line arg.')
        if run_i < 0 or run_i >= n_runs:
            raise ValueError('invalid run_i, max is {}'.format(n_runs-1))
        n_obs, beta_r = run_i_to_params(run_i)
        print('n_obs:{}, beta_r:{}'.format(n_obs, beta_r))
        start_time = time.time()
        run_and_save_run_i(run_i)
        cur_time_min = (time.time() - start_time)/60
        print('elapsed time: {:.2} min'.format(cur_time_min), flush=True)
    else:
        start_time = time.time()
        for run_i in range(n_runs):
            # progress
            cur_time_min = (time.time() - start_time)/60
            print('{}/{}, elapsed time: {:.2} min'.format(
                run_i+1, n_runs, cur_time_min), flush=True)
            run_and_save_run_i(run_i)
    print('done', flush=True)

# ============================================================================
if not plot:
    # all done if not plotting anything
    raise SystemExit

# ============================================================================
# Load results

# mean_loo_s = np.full((n_probl, len(n_obs_s), len(beta_r_s), n_trial,), np.nan)
# var_loo_s = np.full((n_probl, len(n_obs_s), len(beta_r_s), n_trial,), np.nan)
# moment3_loo_s = np.full((n_probl, len(n_obs_s), len(beta_r_s), n_trial,), np.nan)
# mean_elpd_s = np.full((n_probl, len(n_obs_s), len(beta_r_s), n_trial,), np.nan)
# var_elpd_s = np.full((n_probl, len(n_obs_s), len(beta_r_s), n_trial,), np.nan)
# moment3_elpd_s = np.full((n_probl, len(n_obs_s), len(beta_r_s), n_trial,), np.nan)
# mean_err_s = np.full((n_probl, len(n_obs_s), len(beta_r_s), n_trial,), np.nan)
# var_err_s = np.full((n_probl, len(n_obs_s), len(beta_r_s), n_trial,), np.nan)
# moment3_err_s = np.full((n_probl, len(n_obs_s), len(beta_r_s), n_trial,), np.nan)

skew_loo_s = np.full((n_probl, len(n_obs_s), len(beta_r_s), n_trial,), np.nan)
skew_elpd_s = np.full((n_probl, len(n_obs_s), len(beta_r_s), n_trial,), np.nan)
skew_err_s = np.full((n_probl, len(n_obs_s), len(beta_r_s), n_trial,), np.nan)

for run_i in range(n_runs):
    run_i_str = str(run_i).zfill(4)
    n_obs, beta_r = run_i_to_params(run_i)
    n_obs_i = np.nonzero(n_obs_s == n_obs)[0][0]
    beta_r_i = np.nonzero(beta_r_s == beta_r)[0][0]
    res_file = np.load(folder_name+'/'+'res_'+run_i_str+'.npz')
    # mean_loo_s[:, n_obs_i, beta_r_i, :] = res_file['mean_loo_t']
    # var_loo_s[:, n_obs_i, beta_r_i, :] = res_file['var_loo_t']
    # moment3_loo_s[:, n_obs_i, beta_r_i, :] = res_file['moment3_loo_t']
    # mean_elpd_s[:, n_obs_i, beta_r_i, :] = res_file['mean_elpd_t']
    # var_elpd_s[:, n_obs_i, beta_r_i, :] = res_file['var_elpd_t']
    # moment3_elpd_s[:, n_obs_i, beta_r_i, :] = res_file['moment3_elpd_t']
    # mean_err_s[:, n_obs_i, beta_r_i, :] = res_file['mean_err_t']
    # var_err_s[:, n_obs_i, beta_r_i, :] = res_file['var_err_t']
    # moment3_err_s[:, n_obs_i, beta_r_i, :] = res_file['moment3_err_t']
    skew_loo_s[:, n_obs_i, beta_r_i, :] = res_file['skew_loo_t']
    skew_elpd_s[:, n_obs_i, beta_r_i, :] = res_file['skew_elpd_t']
    skew_err_s[:, n_obs_i, beta_r_i, :] = res_file['skew_err_t']
    res_file.close()


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


# ===========================================================================
# plots
# ===========================================================================

# configs

probl_names = [
    r'unique cov. in $\mathrm{M_B}$',
    r'unique cov. in $\mathrm{M_A}$ and $\mathrm{M_B}$'+'\n'
        +r'with effect ratio 1:1',
    r'unique cov. in $\mathrm{M_A}$ and $\mathrm{M_B}$'+'\n'
        +r'with effect ratio 1:2',
]

fontsize = 16

plot_sample_instead_of_median = False

fig, axes = plt.subplots(
    n_probl, len(n_obs_s), sharex=True, sharey='row', figsize=(11,6))

for probl_i in range(n_probl):
    for n_i, n_obs in enumerate(n_obs_s):

        ax = axes[probl_i, n_i]

        data = skew_err_s[probl_i, n_i]
        q025 = np.percentile(data, 2.5, axis=-1)
        q975 = np.percentile(data, 97.5, axis=-1)
        ax.fill_between(beta_r_s, q025, q975, alpha=0.2, color='C0')
        if plot_sample_instead_of_median:
            ax.plot(beta_r_s, data[:, 0], color='C0')
        else:
            median = np.percentile(data, 50, axis=-1)
            ax.plot(beta_r_s, median, color='C0')

        ax.axhline(0, color='gray', lw=1.0)#, zorder=0)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize-4)

for ax, name in zip(axes[:, 0], probl_names):
    ax.set_ylabel(
        name,
        fontsize=fontsize,
        rotation=0,
        ha='right',
        va='center',
    )

for ax, n_obs in zip(axes[0, :], n_obs_s):
    ax.set_title(r'$n={}$'.format(n_obs), fontsize=fontsize)

for ax in axes[-1, :]:
    ax.set_xlabel(r'$\beta_\mathrm{r}$', fontsize=fontsize-2)

fig.tight_layout()
