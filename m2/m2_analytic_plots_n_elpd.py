
import sys, os, time

import numpy as np
from scipy import linalg, stats

from m2_setup import *



# conf
load_res = False
plot = True

plot_multilines = True
multilines_max = 100
multilines_alpha = 0.05


# ============================================================================

if plot:
    import matplotlib.pyplot as plt
    import seaborn as sns



def get_A_Pa_Pb(X_mat):
    n_obs, _ = X_mat.shape
    # calc Ps
    Pa = np.zeros((n_obs, n_obs))
    Pb = np.zeros((n_obs, n_obs))
    for i in range(n_obs):
        X_mi = np.delete(X_mat, i, axis=0)
        # a
        XXinvX_a = linalg.solve(
            X_mi[:,:-1].T.dot(X_mi[:,:-1]),
            X_mat[i,:-1],
            assume_a='sym'
        )
        sXX_a = np.sqrt(X_mat[i,:-1].dot(XXinvX_a) + 1)
        # b
        XXinvX_b = linalg.solve(
            X_mi.T.dot(X_mi),
            X_mat[i,:],
            assume_a='sym'
        )
        sXX_b = np.sqrt(X_mat[i,:].dot(XXinvX_b) + 1)
        for j in range(n_obs):
            if i == j:
                # diag
                Pa[i,i] = -1.0/sXX_a
                Pb[i,i] = -1.0/sXX_b
            else:
                # off-diag
                Pa[i,j] = X_mat[j,:-1].dot(XXinvX_a)/sXX_a
                Pb[i,j] = X_mat[j,:].dot(XXinvX_b)/sXX_b
    # calc A
    A_mat = Pa.T.dot(Pa)
    A_mat -= Pb.T.dot(Pb)
    A_mat /= -2*tau2
    return A_mat, Pa, Pb


def determine_beta_t_s(X_mat, A_mat, Pa, Pb):
    PaX = Pa.dot(X_mat[:,-1])

    b2 = -PaX.T.dot(PaX)/(2*tau2)

    b0 = np.sum(np.log(-np.diag(Pa))) - np.sum(np.log(-np.diag(Pb)))

    # = 4
    t = (4 - b0)/(b2)
    if t < 0:
        t = (4 + b0)/(-b2)
    if t < 0:
        beta_t_4 = 0.0
    else:
        beta_t_4 = np.sqrt(t)

    # # = 10 se
    # A2 = A_mat.dot(A_mat)
    # b02 = 2*sigma2_d**2*np.trace(A2)
    # t = Pa.T.dot(PaX)
    # b22 = t.T.dot(t)/tau2**2
    #
    # b02 *= 10
    # b22 *= 10
    #
    # t = (b02 - b0)/(b2-b22)
    # if t < 0:
    #     t = (b02 + b0)/(-b2-b22)
    # if t < 0:
    #     beta_t_10se = 100.0
    # else:
    #     beta_t_10se = np.sqrt(t)

    return beta_t_4  #, beta_t_10se


def get_analytic_res(X_mat, beta_t, sigma2_d, A_mat, Pa, Pb):
    """Analytic result for fixed sigma2 measure."""
    n_obs, _ = X_mat.shape
    #
    PaX = Pa.dot(X_mat[:,-1])
    # calc b
    b_vec = Pa.T.dot(PaX)
    b_vec *= -beta_t/tau2
    # calc c
    c_sca = PaX.T.dot(PaX)
    c_sca *= -beta_t**2/(2*tau2)
    c_sca += np.sum(np.log(-np.diag(Pa))) - np.sum(np.log(-np.diag(Pb)))

    #
    A2 = A_mat.dot(A_mat)
    A3 = A2.dot(A_mat)
    b_vec_A = b_vec.T.dot(A_mat)
    # if mu_d is not None:
    #     mu_d_A = mu_d.T.dot(A_mat)

    # mean
    mean = c_sca
    # if mu_d is not None:
    #     mean += b_vec.T.dot(mu_d)
    #     mean += mu_d_A.dot(mu_d)

    # var
    var = 2*sigma2_d**2*np.trace(A2)
    var += sigma2_d*(b_vec.T.dot(b_vec))
    # if mu_d is not None:
    #     var += 4*sigma2_d*(mu_d_A.dot(b_vec))
    #     var += 4*sigma2_d*(mu_d_A.dot(mu_d_A.T))

    # moment3
    moment3 = 8*sigma2_d**3*np.trace(A3)
    moment3 += 6*sigma2_d**2*(b_vec_A.dot(b_vec))
    # if mu_d is not None:
    #     moment3 += 24*sigma2_d**2*(b_vec_A.dot(mu_d_A.T))
    #     moment3 += 24*sigma2_d**2*(mu_d_A.dot(A_mat.dot(mu_d_A.T)))

    # coefvar
    coefvar = np.sqrt(var)/mean

    # skew
    skew = moment3 / np.sqrt(var)**3


    return mean, var, moment3, coefvar, skew




# ============================================================================
# As a function of n

if load_res:
    res_file = np.load('m2_res_n_elpd.npz')
    beta_t_s_4 = res_file['beta_t_s_4']
    beta_t_multip = res_file['beta_t_multip']
    analytic_mean_s = res_file['analytic_mean_s']
    analytic_var_s = res_file['analytic_var_s']
    analytic_skew_s = res_file['analytic_skew_s']
    analytic_coefvar_s = res_file['analytic_coefvar_s']
    n_obs_s = res_file['n_obs_s']
    sigma2_d = res_file['sigma2_d']
    res_file.close()

else:

    # variables

    # n_obs_s
    # n_obs_s = [16, 32, 64, 128, 256, 512, 1024]
    n_obs_s = [16, 32, 64, 128]
    # n_obs_s = np.round(np.linspace(20, 1000, 20)).astype(int)

    # constants
    prc_out = 0.0
    n_obs_out = 0
    sigma2_d = 1.0

    # determine these from the equations
    beta_t_s_4 = np.full((len(n_obs_s), n_trial), np.nan)
    beta_t_multip = [1, 2, 5, 10]


    start_time = time.time()
    analytic_mean_s = np.full(
        (len(n_obs_s), n_trial, len(beta_t_multip)), np.nan)
    analytic_var_s = np.full(
        (len(n_obs_s), n_trial, len(beta_t_multip)), np.nan)
    analytic_skew_s = np.full(
        (len(n_obs_s), n_trial, len(beta_t_multip)), np.nan)
    analytic_coefvar_s = np.full(
        (len(n_obs_s), n_trial, len(beta_t_multip)), np.nan)
    for i1, n_obs in enumerate(n_obs_s):
        cur_time_min = (time.time() - start_time)/60
        print('{}/{}, elapsed time: {:.2} min'.format(
            i1+1, len(n_obs_s), cur_time_min), flush=True)
        for t_i in range(n_trial):
            X_mat, mu_d = make_x_mu(
                n_obs, n_obs_out, sigma2_d)

            A_mat, Pa, Pb = get_A_Pa_Pb(X_mat)
            beta_t_s_4[i1, t_i] = determine_beta_t_s(
                X_mat, A_mat, Pa, Pb)

            for b_i, b_multip in enumerate(beta_t_multip):

                # 10se
                beta_t = b_multip*beta_t_s_4[i1, t_i]
                mean, var, _, coefvar, skew = get_analytic_res(
                    X_mat, beta_t, sigma2_d, A_mat, Pa, Pb)
                analytic_mean_s[i1, t_i, b_i] = mean
                analytic_var_s[i1, t_i, b_i] = var
                analytic_coefvar_s[i1, t_i, b_i] = coefvar
                analytic_skew_s[i1, t_i, b_i] = skew


    print('done', flush=True)

    np.savez_compressed(
        'm2_res_n_elpd.npz',
        beta_t_s_4=beta_t_s_4,
        beta_t_multip=beta_t_multip,
        analytic_mean_s=analytic_mean_s,
        analytic_var_s=analytic_var_s,
        analytic_skew_s=analytic_skew_s,
        analytic_coefvar_s=analytic_coefvar_s,
        n_obs_s=n_obs_s,
        sigma2_d=sigma2_d,
    )


# plots
if plot:

    n_obs_s = np.asarray(n_obs_s)

    # ============== mean
    fig = plt.figure()
    ax = plt.gca()
    ax.axhline(0, color='red', lw=1.0)#, zorder=0)

    for b_i, b_multip in enumerate(beta_t_multip):

        data = analytic_mean_s[:,:,b_i]
        ax.plot(
            n_obs_s,
            data[:,:multilines_max],
            color='C{}'.format(b_i),
            alpha=multilines_alpha,
        )
        median = np.percentile(data, 50, axis=-1)
        ax.plot(n_obs_s, median,
            color='C{}'.format(b_i),
            label=(r'$\beta_t = ' + str(b_multip) +
                r'\,\beta_{t\,4}$')
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=14)

    ax.set_xlabel(r'$n$', fontsize=18)
    ax.set_ylabel(r'$m_1$', fontsize=18)
    ax.legend()
    fig.tight_layout()


    # ============== coefvar
    fig = plt.figure()
    ax = plt.gca()
    ax.axhline(0, color='red', lw=1.0)#, zorder=0)

    for b_i, b_multip in enumerate(beta_t_multip):

        data = analytic_coefvar_s[:,:,b_i]
        ax.plot(
            n_obs_s,
            data[:,:multilines_max],
            color='C{}'.format(b_i),
            alpha=multilines_alpha,
        )
        median = np.percentile(data, 50, axis=-1)
        ax.plot(n_obs_s, median,
            color='C{}'.format(b_i),
            label=(r'$\beta_t = ' + str(b_multip) +
                r'\,\beta_{t\,4}$')
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=14)

    ax.set_xlabel(r'$n$', fontsize=18)
    ax.set_ylabel(r'$\sqrt{\overline{m}_2}/m_1$', fontsize=18)
    ax.legend()
    fig.tight_layout()


    # ============== skew
    fig = plt.figure()
    ax = plt.gca()
    ax.axhline(0, color='red', lw=1.0)#, zorder=0)

    for b_i, b_multip in enumerate(beta_t_multip):

        data = analytic_skew_s[:,:,b_i]
        ax.plot(
            n_obs_s,
            data[:,:multilines_max],
            color='C{}'.format(b_i),
            alpha=multilines_alpha,
        )
        median = np.percentile(data, 50, axis=-1)
        ax.plot(n_obs_s, median,
            color='C{}'.format(b_i),
            label=(r'$\beta_t = ' + str(b_multip) +
                r'\,\beta_{t\,4}$')
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=14)

    ax.set_xlabel(r'$n$', fontsize=18)
    ax.set_ylabel(r'$\widetilde{m}_3}$', fontsize=18)
    ax.legend()
    fig.tight_layout()
