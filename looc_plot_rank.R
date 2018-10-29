
library(matrixStats)
library(extraDistr)
library(ggplot2)
library(ggExtra)


Ns = c(10, 20, 40, 60, 100, 140, 200, 260)
p0 = 1

# truedist = 'n'; modeldist = 'n'; priordist = 'n'
truedist = 't4'; modeldist = 'tnu'; priordist = 'n'
# truedist = 'b'; modeldist = 'b'; priordist = 'n'
# truedist = 'n'; modeldist = 'tnu'; priordist = 'n'
# truedist = 't4'; modeldist = 'n'; priordist = 'n'

ni = 1
n = Ns[ni]

# load data in variable out
load(sprintf('res_looc/%s_%s_%s_%g_%g.RData',
    truedist, modeldist, priordist, p0, n))
# remove some last trials (to make binning easier)
# also drop singleton dimensions
Niter_full = dim(out$loos)[2]
Niter = Niter_full - 1  # drop one sample to get 1999
for (name in names(out)) {
    dims = dim(out[[name]])
    if (length(dims) == 2) {
        out[[name]] = out[[name]][,1:Niter]
    } else if (length(dims) == 3) {
        out[[name]] = out[[name]][,1:Niter,]
    } else if (length(dims) == 4) {
        out[[name]] = out[[name]][,,1:Niter,]
    } else {
        stop('Unrecognised name in out part')
    }
}



# ==============================================================================
# select measure

# Mi
m_i = 1
loo_name = sprintf('M%d', m_i)
loos = out$loos[,,m_i]
tls = out$tls[,m_i]
g2s = out$g2s[,m_i]
g3s = out$g3s[,m_i]
# g2s = out$g2s_nod[,m_i]
# g3s = out$g3s_nod[,m_i]

# M1-M2
# loo_name = 'M1-M2'
# loos = out$loos[,,1] - out$loos[,,2]
# tls = out$tls[,1] - out$tls[,2]
# g2s = out$g2s_d
# g3s = out$g3s_d
# # g2s = out$g2s_nod_d
# # g3s = out$g3s_nod_d


# ==============================================================================
# calc

loosums = colSums(loos)

# var estimates
loovars_naive = n*colVars(loos)
loovars_2 = 2*loovars_naive
loovars_3 = loovars_naive + n*g2s
loovars_4 = loovars_naive + (n^2)*g3s
# pack them
var_estim_names = list('naive', 'x2', 'g2', 'g3')
var_estims = list(loovars_naive, loovars_2, loovars_3, loovars_4)

# calc p(elpd<elpd_i)
ref_p = sapply(tls, function(x) mean(tls < x))
# calc p(loo<elpd_i), normal approx. using all var estimates
test_p_s = vector("list", length(var_estims))
for (var_e_i in 1:length(var_estims)) {
    test_p_s[[var_e_i]] = pnorm(
        tls, mean=loosums, sd=sqrt(var_estims[[var_e_i]]))
}


# ==============================================================================
# plot pairwise plots
if (T) {

    dev.new()
    g = ggplot()
    g = g + geom_abline(intercept=0, slope=1, colour='red')
    g = g + geom_point(aes(x=tls, y=loosums))
    g = g +
        xlab("elpd_i") +
        ylab("loo_i") +
        ggtitle(sprintf(
            "%s, %s_%s_%s, p0=%g n=%g",
            loo_name, truedist, modeldist, priordist, p0, n
        ))
    g = ggMarginal(g, type='histogram')
    print(g)

    for (var_e_i in 1:length(var_estims)) {
        test_p = test_p_s[[var_e_i]]
        test_p_name = var_estim_names[[var_e_i]]

        dev.new()
        g = ggplot()
        g = g + geom_abline(intercept=0, slope=1, colour='red')
        g = g + geom_point(aes(x=ref_p, y=test_p))
        g = g +
            xlab("p(elpd<elpd_i)") +
            ylab(sprintf(
                "p(loo<elpd_i) normal approx., %s var", test_p_name)) +
            ggtitle(sprintf(
                "%s, %s_%s_%s, p0=%g n=%g",
                loo_name, truedist, modeldist, priordist, p0, n
            ))
        g = ggMarginal(g, type='histogram')
        print(g)
    }
}

# ==============================================================================
# plot ranks

for (var_e_i in 1:length(var_estims)) {

    test_p = test_p_s[[var_e_i]]
    test_p_name = var_estim_names[[var_e_i]]

    # calc
    Ntg = Niter
    ranks = sapply(test_p, function(x) sum(x > ref_p))

    # combine bins
    comb = 200
    B = (Ntg+1)/comb
    if (comb > 1) {
        ranks_comb = array(NA, Niter)
        for (bi in 0:(B-1)) {
            selected = array(FALSE, Niter)
            for (ci in 0:(comb-1)) {
                selected = selected | (ranks == (bi*comb+ci))
            }
            ranks_comb[selected] = bi
        }
    } else {
        ranks_comb = ranks
    }

    # plot
    dev.new()
    g = ggplot()
    g = g + geom_bar(aes(x=ranks_comb))
    g = g +
        geom_hline(yintercept=qbinom(0.005, Niter, 1/B)) +
        geom_hline(yintercept=Niter/B) +
        geom_hline(yintercept=qbinom(0.995, Niter, 1/B)) +
        xlab(sprintf("rank with %s", test_p_name)) +
        ylab("") +
        ggtitle(sprintf(
            "%s, %s_%s_%s, p0=%g n=%g",
            loo_name, truedist, modeldist, priordist, p0, n
        ))
    print(g)
}
