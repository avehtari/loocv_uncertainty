
library(ggplot2)
library(matrixStats)
library(extraDistr)


Ns = c(10, 20, 40, 60, 100, 140, 200, 260)
Ps = c(1, 2, 5, 10)

# truedist = 'n'; modeldist = 'n'; priordist = 'n'
truedist = 't4'; modeldist = 'tnu'; priordist = 'n'
# truedist = 'b'; modeldist = 'b'; priordist = 'n'
# truedist = 'n'; modeldist = 'tnu'; priordist = 'n'
# #truedist = 't4'; modeldist = 'n'; priordist = 'n'

p_i = 1

# ==============================================================================
# peformance of variance terms

# settings for bayesian bootstrap
bbsamples_perf = 2000
bbalpha_perf = 1

# output arrays
pvs = array(0, c(4, length(Ns)))
bbs = array(0, c(4, length(Ns), bbsamples_perf))

cat('processing n=')
for (ni in 1:length(Ns)) {
    n = Ns[ni]
    cat(sprintf('%g,', n))

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
    # populate local environment with the named stored variables in selected out
    list2env(out, envir=environment())
    rm(out)
    Niter = dim(loos)[2]

    pvz = sd(tls-colSums(loos))  # sd of loo error
    # pvz = sd(colSums(loos))    # sd of loo
    colvars_loos_n = colVars(loos)*n

    # basic
    t = colvars_loos_n
    # t = loovar1
    pvs[1,ni] = sqrt(mean(t)) / pvz
    bbs[1,ni,] = sqrt(
        rdirichlet(bbsamples_perf, rep(bbalpha_perf, length(t))) %*% t) / pvz

    # g2s
    t = colvars_loos_n + g2s*n
    # t = loovar3
    pvs[2,ni] = sqrt(mean(t)) / pvz
    bbs[2,ni,] = sqrt(
        rdirichlet(bbsamples_perf, rep(bbalpha_perf, length(t))) %*% t) / pvz

    # g2s_new
    t = colvars_loos_n + (n^2)*g2s_new
    # t = loovar4
    pvs[3,ni] = sqrt(mean(t)) / pvz
    bbs[3,ni,] = sqrt(
        rdirichlet(bbsamples_perf, rep(bbalpha_perf, length(t))) %*% t) / pvz

    # # g2s_new2
    # t = colvars_loos_n + (n^2)*g2s_new2
    # # t = loovar5
    # pvs[3,ni] = sqrt(mean(t)) / pvz
    # bbs[3,ni,] = sqrt(
    #     rdirichlet(bbsamples_perf, rep(bbalpha_perf, length(t))) %*% t) / pvz

    # x2
    # t = 2*colvars_loos_n
    t = loovar2
    pvs[4,ni] = sqrt(mean(t)) / pvz
    bbs[4,ni,] = sqrt(
        rdirichlet(bbsamples_perf, rep(bbalpha_perf, length(t))) %*% t) / pvz
}
cat('\ndone processing\n')


## plot ------------------------------------------------------
fillcats = c("basic", "g2s", "g2s_new", "x2")

## Plot all Ns
g = ggplot()
for (fi in 1:length(fillcats)) {
    for (ni in 1:length(Ns)) {
        g = g + geom_violin(
            data = data.frame(
                y = bbs[fi,ni,],
                x = Ns[ni],
                fill = fillcats[fi]
            ),
            aes(x=x, y=y, fill=fill),
            width = 8,
            alpha = 0.5
        )
    }
}
g = g +
    geom_hline(yintercept=1) +
    scale_fill_manual(values=c("green", "red", "blue", "gray")) +
    xlab("number of samples") +
    ylab("") +
    ggtitle(sprintf(
        "peformance of variance terms (model: %s_%s_%s_%g)\nbbsamples=%g",
        truedist, modeldist, priordist, Ps[p_i],
        bbsamples_perf
    ))
g
# ggsave(
#     plot=g, width=8, height=5,
#     filename = sprintf(
#         "figs/p1_%s_%s_%s_%i.pdf",
#         truedist, modeldist, priordist, Ps[p_i]
#     )
# )


## plot ------------------------------------------------------
# plot one ni
ni = 7
g = ggplot()
for (fi in 1:length(fillcats)) {
    g = g + geom_violin(
        data = data.frame(
            y = bbs[fi,ni,],
            x = fillcats[fi]
        ),
        aes(x=x, y=y)
    )
}
g = g +
    geom_hline(yintercept=1) +
    ylab("") +
    ggtitle(sprintf(
        "peformance of variance terms (model: %s_%s_%s_%g_%g)\nbbsamples=%g",
        truedist, modeldist, priordist, Ps[p_i], Ns[ni],
        bbsamples_perf
    ))
g
# ggsave(
#     plot=g, width=8, height=5,
#     filename = sprintf(
#         "figs/p1_%s_%s_%s_%i_%i.pdf",
#         truedist, modeldist, priordist, Ps[p_i], Ns[ni]
#     )
# )
