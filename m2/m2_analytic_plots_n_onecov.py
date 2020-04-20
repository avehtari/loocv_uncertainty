
import sys, os, time

import numpy as np
from scipy import linalg, stats

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# As a function of n


# variables

# n_obs_s
n_obs_s = np.round(np.linspace(100, 1e6, 20)).astype(int)

n_obs_s = np.asarray(n_obs_s)

# beta_t
beta_t_s = [0.0, 1.0]

# constants
sigma2_d = 1.0
tau2 = 1.0



analytic_skew_s = np.full((len(beta_t_s), len(n_obs_s)), np.nan)
for i1, beta_t in enumerate(beta_t_s):
    for i2, n_obs in enumerate(n_obs_s):

        k1 = sigma2_d**2/(2*tau2**2)
        k2 = beta_t**2*sigma2_d/tau2**2
        k3 = (sigma2_d/tau2)**3
        k4 = 3*beta_t**2*sigma2_d**2/tau2**3

        skew = (
            -( k3*(n_obs-3)*(n_obs-1) )
            /
            ( (k1*(n_obs-1)+k2*n_obs*(n_obs-2))**(3/2) * np.sqrt(n_obs-2) )
            -( k4**n_obs*(n_obs-2)**(3/2) )
            /
            ( (k1*(n_obs-1)+k2*n_obs*(n_obs-2))**(3/2) )
        )

        analytic_skew_s[i1, i2] = skew
print('done', flush=True)



# plots

# skew
fig, axes = plt.subplots(
    len(beta_t_s), 1, sharex=True, figsize=(7,6))
for b_i, beta_t in enumerate(beta_t_s):
    ax = axes[b_i]
    ax.axhline(0, color='red', lw=1.0)#, zorder=0)
    data = analytic_skew_s[b_i]
    ax.plot(n_obs_s, data)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=14)

axes[-1].set_xlabel(r'$n$', fontsize=18)
for ax, beta_t in zip(axes, beta_t_s):
    ax.set_ylabel(r'$\beta_t={}$'.format(beta_t), fontsize=18)
# fig.suptitle('skew')
fig.tight_layout()
