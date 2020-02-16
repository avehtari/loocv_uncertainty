"""Script for plot 1."""

import numpy as np
from scipy import linalg, stats

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import seaborn as sns


from m1_problem import *



# ============================================================================

fixed_sigma2_m = False

# [
#   [sigma_d, ...],     [0.01, 1.0, 100.0]
#   [n_obs, ...],       [16, 32, 64, 128, 256, 512, 1024]
#   [beta_t, ...],      [0.0, 0.2, 1.0, 4.0]
#   [prc_out, ...]      [0.0, np.nextafter(0,1), 0.01, 0.08]
# ]

selected_betas = [0, 1, 2]
selected_prc_outs = [0, 2]
sigma_d_i = 1

# histogram bins
cal_nbins = 7


fontsize = 16


seed_analysis = 123451

# bayes boot samples
bb_n = 1000
bb_a = 1.0


# ============================================================================

def remove_frame(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

# ============================================================================


if fixed_sigma2_m:
    folder_name = 'fixed'
else:
    folder_name = 'unfixed'

rng = np.random.RandomState(seed=seed_analysis)

loo_s = np.zeros((
    len(selected_prc_outs), len(selected_betas), len(n_obs_s), n_trial))
naive_var_s = np.zeros((
    len(selected_prc_outs), len(selected_betas), len(n_obs_s), n_trial))
cor_loo_i_s = np.zeros((
    len(selected_prc_outs), len(selected_betas), len(n_obs_s), n_trial))
skew_loo_i_s = np.zeros((
    len(selected_prc_outs), len(selected_betas), len(n_obs_s), n_trial))

target_mean_s = np.zeros((
    len(selected_prc_outs), len(selected_betas), len(n_obs_s)))
target_var_s = np.zeros((
    len(selected_prc_outs), len(selected_betas), len(n_obs_s)))
# target_Sigma_s = np.zeros((
#     len(selected_prc_outs), len(selected_betas), len(n_obs_s), n_obs, n_obs))
target_skew_s = np.zeros((
    len(selected_prc_outs), len(selected_betas), len(n_obs_s)))
target_plooneg_s = np.zeros((
    len(selected_prc_outs), len(selected_betas), len(n_obs_s)))
elpd_s = np.zeros((
    len(selected_prc_outs), len(selected_betas), len(n_obs_s), n_trial))


# load results
for o_i, o_ii in enumerate(selected_prc_outs):
    prc_out = prc_out_s[o_ii]
    for b_i, b_ii in enumerate(selected_betas):
        beta_t = beta_t_s[b_ii]
        for n_i, n_obs in enumerate(n_obs_s):
            run_i = np.ravel_multi_index([sigma_d_i, n_i, b_ii, o_ii], grid_shape)
            res_file = np.load(
                'res_1/{}/{}.npz'
                .format(folder_name, str(run_i).zfill(4))
            )
            # fetch results
            loo_ti_A = res_file['loo_ti_A']
            loo_ti_B = res_file['loo_ti_B']
            test_elpd_t_A = res_file['test_elpd_t_A']
            test_elpd_t_B = res_file['test_elpd_t_B']
            # close file
            res_file.close()

            # calc some normally obtainable values
            n_cur = loo_ti_A.shape[-1]
            loo_i = loo_ti_A-loo_ti_B
            loo_s[o_i, b_i, n_i] = np.sum(loo_i, axis=-1)
            naive_var_s[o_i, b_i, n_i] = n_cur*np.var(
                loo_i, ddof=1, axis=-1)
            for trial_i in range(n_trial):
                cor_loo_i_s[o_i, b_i, n_i, trial_i] = np.corrcoef(
                    loo_ti_A[trial_i], loo_ti_B[trial_i])[0, 1]
            skew_loo_i_s[o_i, b_i, n_i] = stats.skew(
                loo_i, axis=-1, bias=False)

            # calc some target values
            target_mean_s[o_i, b_i, n_i] = np.mean(loo_s[o_i, b_i, n_i])
            target_var_s[o_i, b_i, n_i] = np.var(loo_s[o_i, b_i, n_i], ddof=1)
            # target_Sigma_s[o_i, b_i, n_i] = np.cov(loo_i, ddof=1, rowvar=False)
            target_skew_s[o_i, b_i, n_i] = stats.skew(loo_s[o_i, b_i, n_i], bias=False)
            # TODO calc se of this ... formulas online
            target_plooneg_s[o_i, b_i, n_i] = np.mean(loo_s[o_i, b_i, n_i]<0)
            elpd_s[o_i, b_i, n_i] = test_elpd_t_A - test_elpd_t_B

# naive var ratio
naive_se_ratio_s_mean = np.sqrt(
    np.mean(naive_var_s, axis=-1)/target_var_s)
naive_se_ratio_s = np.sqrt(
    np.einsum('bt,ijkt->ijkb',
        rng.dirichlet(
            np.full(n_trial, bb_a),
            size=bb_n
        ),
        naive_var_s
    )/target_var_s[:,:,:,None]
)

# naive var error ratio
error_var_s = np.var(loo_s-elpd_s, ddof=1, axis=-1)
naive_error_ratio_s_mean = np.sqrt(
    np.mean(naive_var_s, axis=-1)/error_var_s)
naive_error_ratio_s = np.sqrt(
    np.einsum('bt,ijkt->ijkb',
        rng.dirichlet(
            np.full(n_trial, bb_a),
            size=bb_n
        ),
        naive_var_s
    )/error_var_s[:,:,:,None]
)


# naive_coef_var_s = np.sqrt(naive_var_s)/loo_s
naive_plooneg_s = stats.norm.cdf(0, loc=loo_s, scale=np.sqrt(naive_var_s))

pelpdneg_s = np.mean(elpd_s<0, axis=-1)
# target_coefvar_s = np.sqrt(target_var_s)/target_mean_s

# misspred: TODO replace with something
naive_misspred_s = np.abs(naive_plooneg_s-pelpdneg_s[:,:,:,None])



# ============================================================================
# plots option 2 with lines
# ============================================================================

fig, axes = plt.subplots(len(selected_prc_outs), 1, figsize=(6,6), sharex=True)
for o_i, ax in enumerate(axes):
    o_ii = selected_prc_outs[o_i]
    prc_out = prc_out_s[o_ii]

    ax.axhline(1.0, color='red')

    for b_i, b_ii in enumerate(selected_betas):
        beta_t = beta_t_s[b_ii]
        colorstr = 'C{}'.format(b_i)
        beta_t_name = r'$\beta_t=' + str(beta_t) + r'$'

        data = naive_error_ratio_s[o_i, b_i]

        median = np.percentile(data, 50, axis=-1)
        q025 = np.percentile(data, 2.5, axis=-1)
        q975 = np.percentile(data, 97.5, axis=-1)
        ax.fill_between(n_obs_s, q025, q975, color=colorstr.format(b_i), alpha=0.2)
        ax.plot(
            n_obs_s, median,
            color=colorstr,
            label=beta_t_name
        )
        ax.set_xscale('log')

    ax.set_xticks(n_obs_s)
    ax.set_xticklabels(n_obs_s)
    ax.minorticks_off()
    ax.set_ylim(bottom=0)

    out_name = r'$\mathrm{out}\%=' + str(int(prc_out*100)) + r'$'
    y_axis_name = r'$\widehat{\mathrm{SE}}\,/\,\mathrm{SE}(\mathrm{\widehat{elpd}_D} - \mathrm{elpd_D})$'
    ax.set_ylabel(out_name + '\n' + y_axis_name, fontsize=fontsize)

    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize-4)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[-1].set_xlabel(r'$n$', fontsize=fontsize)

axes[-1].legend(fontsize=fontsize-2)

fig.tight_layout()
