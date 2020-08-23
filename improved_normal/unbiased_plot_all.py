"""Plot simulated experiment results for the unbiased LOO-CV variance estimator.

Uses the hard-coded results from `unbiased.py`

"""

import numpy as np

import matplotlib.pyplot as plt


# ==============================================================================
# config

seed = 57346234
n_trial = 20000

# data params
n_obs = 16


# model params
sigma2_m = 1.2**2
sigma2_p = 2.0**2


# ==============================================================================
# hard-coded results from `unbiased.py`
bb_mean_pe = [
    [0.9663997725, 1.000294244],
    [1.15876797, 1.000242305],
    [0.8098354968, 0.999482842],
]
bb_median_pe = [
    [0.9664015508, 1.000301568],
    [1.158743842, 1.00025908],
    [0.8098479447, 0.999523508],
]
analytic_pe = [
    [0.966125189, 1.0],
    [1.160312657, 1.0],
    [0.8094516078, 1.0],
]
bb_ci_95_pe = [
    [[0.95996186, 0.97273841], [0.9934566,  1.00704534]],
    [[1.15169156, 1.16562419], [0.99339451, 1.00696045]],
    [[0.80231211, 0.81750817], [0.98842999, 1.01048438]],
]
bb_ci_99_pe = [
    [[0.95756668, 0.97492683], [0.9908942,  1.00942793]],
    [[1.14959731, 1.16764143], [0.99082462, 1.00938296]],
    [[0.80000032, 0.82010894], [0.98563381, 1.01432708]],
]




# ==============================================================================

rng = np.random.RandomState(seed)


# ============================================================================
# plot stuff
# ============================================================================

plt.rcParams['font.family'] = 'serif'
plt.rcParams["mathtext.fontset"] = "dejavuserif"


# ============================================================================
# plot
# ============================================================================

# # -------------------------------------
# # y-axis
#
# problem_names = [
#     'Well-matching data',
#     'Under-dispersed data',
#     'Under-dispersed, skewed,\nand heavy-tailed data'
# ]
# estim_names = ['naive', 'unbiased']
#
# x_sect_lims = np.linspace(0, 1, len(estim_names)+1)
# x_cents = np.linspace(0, 1, 2*len(estim_names)+1)[1::2]
# x_space = 0.8  # between [0, 1]
# x_delta = (x_sect_lims[1]-x_sect_lims[0])*x_space/3
# x_grid = (
#     x_sect_lims[:-1] + (x_sect_lims[1]-x_sect_lims[0])*(1-x_space)
#     + x_delta*np.arange(1, 3)[:, None]
# )
# x_analytic = x_grid[0]
# x_bb = x_grid[1]
#
# err_pe = (
#     (np.array(bb_mean_pe)[: ,:, None] - np.array(bb_ci_99_pe)) *
#     np.array([[[1, -1]]])
# )
#
#
# fig, axes = plt.subplots(1, len(problem_names), figsize=(6, 3))
# # fig, axes = plt.subplots(1, len(problem_names), sharey=True, figsize=(6, 5))
# for ax, probl_name, bb_median_p, err_p, analytic_p in zip(
#         axes, problem_names, bb_median_pe, err_pe, analytic_pe):
#
#     ax.axhline(1.0, lw=1.0, color='C2', label='target')
#
#     # analytic
#     ax.plot(
#         x_analytic,
#         analytic_p,
#         '.',
#         markersize=11,
#         label='analytic',
#         color='C0',
#     )
#
#     # bb
#     ax.errorbar(
#         x_bb,
#         bb_median_p,
#         yerr=err_p,
#         fmt='.',
#         markersize=11,
#         label='BB',
#         color='C1',
#     )
#
#     ax.set_xlim([0, 1])
#     ax.set_xticks(x_cents)
#     ax.set_xticklabels(estim_names)
#
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#
#     ax.set_title(probl_name, fontsize=11)
#
# axes[-1].legend(fancybox=False, shadow=False, framealpha=1.0)
#
# fig.tight_layout()

# -------------------------------------
# x-axis

problem_names = [
    'Well-matching data',
    'Under-dispersed data',
    'Under-dispersed, skewed, and heavy-tailed data'
]
estim_names = ['naive', 'unbiased']

x_sect_lims = np.linspace(0, 1, len(estim_names)+1)
x_cents = np.linspace(0, 1, 2*len(estim_names)+1)[1::2]
x_space = 0.8  # between [0, 1]
x_delta = (x_sect_lims[1]-x_sect_lims[0])*x_space/3
x_grid = (
    x_sect_lims[:-1] + (x_sect_lims[1]-x_sect_lims[0])*(1-x_space)
    + x_delta*np.arange(1, 3)[:, None]
)
x_grid = 1 - x_grid
x_analytic = x_grid[0]
x_bb = x_grid[1]

err_pe = (
    (np.array(bb_mean_pe)[: ,:, None] - np.array(bb_ci_99_pe)) *
    np.array([[[1, -1]]])
)

# fig, axes = plt.subplots(len(problem_names), 1, figsize=(6, 5))
fig, axes = plt.subplots(len(problem_names), 1, sharex=True, figsize=(7, 4.5))
for ax, probl_name, bb_median_p, err_p, analytic_p in zip(
        axes, problem_names, bb_median_pe, err_pe, analytic_pe):

    l1 = ax.axvline(1.0, lw=1.0, color='C2', label='target')

    # analytic
    l2, = ax.plot(
        analytic_p,
        x_analytic,
        '.',
        markersize=11,
        label='analytic',
        color='C0',
    )

    # bb
    l3 = ax.errorbar(
        bb_median_p,
        x_bb,
        xerr=err_p,
        fmt='.',
        markersize=11,
        label='simulated',
        color='C1',
    )

    ax.set_ylim([0, 1])
    ax.set_yticks(x_cents)
    ax.set_yticklabels(np.array(estim_names)[::-1])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ax.set_ylabel(probl_name, fontsize=11, rotation=0, ha='right', va='center')
    ax.set_title(probl_name, fontsize=11)

axes[-1].set_xlabel(
    r'$\sqrt{\left.\mathrm{E}\left[\widehat{\sigma^2_\mathrm{LOO}}\right]'
    r'\;\right/\;'
    r'\sigma^2_\mathrm{LOO}}$',
)

axes[-1].legend(
    # [l1, l3, l2],
    # ['target', 'BB', 'analytic'],
    ncol=3,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.9),
    fancybox=False,
    shadow=False,
    framealpha=1.0,
)

fig.tight_layout()
fig.subplots_adjust(bottom=0.24)
