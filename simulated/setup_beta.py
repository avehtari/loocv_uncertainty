
import numpy as np

from general_setup import *


# folder name
res_folder_name = 'res_beta'


# ============================================================================
# grid params

# number of obs in one trial
n_obs_s = [32, 128]

# last covariate effect not used in model A
beta_t_s = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 4.0]

# outlier dev
out_dev_s = [0.0, 20.0]

# tau2
tau2_s = [None, 1.0]


# ============================================================================
# setup the grid

# set grid
n_obs_grid, beta_t_grid, out_dev_grid, tau2_grid = np.meshgrid(
    n_obs_s, beta_t_s, out_dev_s, tau2_s, indexing='ij')
grid_shape = n_obs_grid.shape
n_runs = n_obs_grid.size
n_obs_max = n_obs_grid.max()

def run_i_to_params(run_i):
    n_obs = n_obs_grid.flat[run_i]
    beta_t = beta_t_grid.flat[run_i]
    out_dev = out_dev_grid.flat[run_i]
    tau2 = tau2_grid.flat[run_i]
    return n_obs, beta_t, out_dev, tau2

def params_to_run_i(n_obs, beta_t, out_dev, tau2):
    n_obs_i = n_obs_s.index(n_obs)
    beta_t_i = beta_t_s.index(beta_t)
    out_dev_i = out_dev_s.index(out_dev)
    tau2_i = tau2_s.index(tau2)
    run_i = np.ravel_multi_index(
        (n_obs_i, beta_t_i, out_dev_i, tau2_i), grid_shape)
    return run_i
