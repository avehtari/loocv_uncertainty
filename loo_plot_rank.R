
library(ggplot2)
library(matrixStats)
library(extraDistr)


Ns = c(10, 20, 40, 60, 100, 140, 200, 260)
Ps = c(1, 2, 5, 10)

truedist = 'n'; modeldist = 'n'; priordist = 'n'
# truedist = 't4'; modeldist = 'tnu'; priordist = 'n'
# truedist = 'b'; modeldist = 'b'; priordist = 'n'
# truedist = 'n'; modeldist = 'tnu'; priordist = 'n'
# truedist = 't4'; modeldist = 'n'; priordist = 'n'

p_i = 1
ni = 1
n = Ns[ni]

# load data in variable out
load(sprintf('res_loo/%s_%s_%s_%g_%g.RData',
    truedist, modeldist, priordist, Ps[p_i], n))
# modify 1d matrices into vectors in out
out$peff = out$peff[1,]
out$tls = out$tls[1,]
out$ets = out$ets[1,]
out$es = out$es[1,]
out$tes = out$tes[1,]
out$bs = out$bs[1,]
out$gs = out$gs[1,]
out$gms = out$gms[1,]
out$g2s = out$g2s[1,]
out$gm2s = out$gm2s[1,]
out$g2s_new = out$g2s_new[1,]
out$g2s_new2 = out$g2s_new2[1,]
out$loovar1 = out$loovar1[1,]
out$loovar2 = out$loovar2[1,]
out$loovar3 = out$loovar3[1,]
out$loovar4 = out$loovar4[1,]
out$loovar5 = out$loovar5[1,]
out$loovar1_rank = out$loovar1_rank[1,]
out$loovar2_rank = out$loovar2_rank[1,]
out$loovar3_rank = out$loovar3_rank[1,]
out$loovar4_rank = out$loovar4_rank[1,]
out$loovar5_rank = out$loovar5_rank[1,]
out$test_target = out$test_target[1,]
# populate local environment with the named stored variables in selected out
list2env(out, envir=environment())
rm(out)

Niter = dim(loos)[2]
Ntg = Niter-1
comb = 200
B = (Ntg+1)/comb


test_arrays = list(loovar1, loovar2, loovar3, loovar4)
names = list('naive', 'x2', 'g2s', 'g2s_new')

for (plot_i in 1:length(test_arrays)) {
    name = names[[plot_i]]
    test_array = test_arrays[[plot_i]]

    # calc ranks
    ranks = array(NaN, Niter)
    for (i1 in 1:Niter) {
        ranks[i1] = sum(test_target[1:Niter-1] < test_array[i1])
    }

    # combine bins
    if (comb > 1) {
        ranks_combined = array(NaN, Niter)
        for (bi in 0:(B-1)) {
            selected = array(FALSE, Niter)
            for (ci in 0:(comb-1)) {
                selected = selected | (ranks == (bi*comb+ci))
            }
            ranks_combined[selected] = bi
        }
    } else {
        ranks_combined = ranks
    }

    # plot
    dev.new()
    g = ggplot()
    g = g + geom_bar(
        data = data.frame(
            x = ranks_combined
        ),
        aes(x=x),
    )
    g = g +
        geom_hline(yintercept=qbinom(0.005, Niter, 1/B)) +
        geom_hline(yintercept=Niter/B) +
        geom_hline(yintercept=qbinom(0.995, Niter, 1/B)) +
        xlab("rank") +
        ylab("") +
        ggtitle(sprintf(
            "rankplot, %s, model: %s_%s_%s_%g_%g",
            name, truedist, modeldist, priordist, Ps[p_i], n
        ))
    print(g)
    # ggsave(
    #     plot=g, width=8, height=5,
    #     filename = sprintf(
    #         "figs/p2_%s_%s_%s_%i_%i.pdf",
    #         truedist, modeldist, priordist, Ps[p_i], n
    #     )
    # )

}
