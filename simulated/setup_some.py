"""Setup for test 'some'."""

import numpy as np

from general_setup import *


# folder name
res_folder_name = 'res_some'


# ============================================================================
# grid params

# number of obs in one trial
n_obs_s = [16, 32, 64, 128, 256, 512, 1024]
# n_obs_s = [16, 22, 32, 45, 64, 90, 128, 181, 256, 362, 512, 724, 1024]

# last covariate effect not used in model A
beta_t_s = [0.0, 0.05, 0.1, 1.0]
# beta_t_s = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 2.0]

# outlier dev
out_dev_s = [0.0, 100.0]


# ============================================================================
# setup the grid

# set grid
n_obs_grid, beta_t_grid, out_dev_grid = np.meshgrid(
    n_obs_s, beta_t_s, out_dev_s, indexing='ij')
grid_shape = n_obs_grid.shape
n_runs = n_obs_grid.size
n_obs_max = n_obs_grid.max()

def run_i_to_params(run_i):
    n_obs = n_obs_grid.flat[run_i]
    beta_t = beta_t_grid.flat[run_i]
    out_dev = out_dev_grid.flat[run_i]
    return n_obs, beta_t, out_dev

def params_to_run_i(n_obs, beta_t, out_dev):
    n_obs_i = n_obs_s.index(n_obs)
    beta_t_i = beta_t_s.index(beta_t)
    out_dev_i = out_dev_s.index(out_dev)
    run_i = np.ravel_multi_index((n_obs_i, beta_t_i, out_dev_i), grid_shape)
    return run_i
