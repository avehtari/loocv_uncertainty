"""Analyse LOOCV results

First run results with m1_run.py

"""

import sys, os

import numpy as np
from scipy import linalg
from scipy import stats

import matplotlib.pyplot as plt

from m1_problem import *


# ============================================================================
# config

# fixed or unfixed sigma
fixed_sigma2_m = False

# ============================================================================

if fixed_sigma2_m:
    folder_name = 'fixed'
else:
    folder_name = 'unfixed'


# As a function of n



# load results
res_A = np.empty(n_runs, dtype=object)
res_B = np.empty(n_runs, dtype=object)
res_test_A = np.empty(n_runs, dtype=object)
res_test_B = np.empty(n_runs, dtype=object)
for run_i in range(n_runs):
    res_file = np.load(
        'res_1/{}/{}.npz'
        .format(folder_name, str(run_i).zfill(4))
    )
    # fetch results
    res_A[run_i_idx] = res_file['loo_ti_A']
    res_B[run_i_idx] = res_file['loo_ti_B']
    res_test_A[run_i_idx] = res_file['test_t_A']
    res_test_B[run_i_idx] = res_file['test_t_B']
    # close file
    res_file.close()

# ============================================================================
# test some plots

# pick one
idx = (2, 3, 1, 1)
res_A_i = res_A[idx]
res_B_i = res_B[idx]

plt.hist(res_A_i.sum(axis=1)-res_B_i.sum(axis=1))


# vary beta_t
idx = (2, None, 1, 1)
# ...
