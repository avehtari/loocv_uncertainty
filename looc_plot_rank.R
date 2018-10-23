
library(ggplot2)
library(matrixStats)
library(extraDistr)


Ns = c(10, 20, 40, 60, 100, 140, 200, 260)
p0 = 1

truedist = 'n'; modeldist = 'n'; priordist = 'n'
# truedist = 't4'; modeldist = 'tnu'; priordist = 'n'
# truedist = 'b'; modeldist = 'b'; priordist = 'n'
# truedist = 'n'; modeldist = 'tnu'; priordist = 'n'
# truedist = 't4'; modeldist = 'n'; priordist = 'n'

ni = 7
n = Ns[ni]

# load data in variable out
load(sprintf('res_looc/%s_%s_%s_%g_%g.RData',
    truedist, modeldist, priordist, p0, n))
# remove some last trials (to make binning easier)
# also drop singleton first dimensions
Niter_full = dim(out$loos)[2]
for (name in names(out)) {
    out[[name]] = out[[name]][,1:(Niter_full-1),]
}
# populate local environment with the named stored variables in selected out
list2env(out, envir=environment())
rm(out)


Niter = dim(loos)[2]

loos_d = loos[,,1] - loos[,,2]
tls_d = tls[,1] - tls[,2]

loosums = colSums(loos_d)

# var estims
loovars_1 = n*colVars(loos_d)

# true var
target_var = var(loosums)

# calc true p(loo<elpd)
ref_p = sapply(tls_d, function(x) mean(loosums < x))
# calc p(loo<elpd), normal approx., naive var estim
test_p = pnorm(tls_d, mean=loosums, sd=sqrt(loovars_1))


# ==============================================================================
# plot histograms of p(loo<elpd)
if (F) {
    dev.new()
    g = ggplot()
    g = g + geom_histogram(data = data.frame(x=ref_p), aes(x=x))
    g = g +
        xlab("p(loo<elpd)") +
        ylab("") +
        ggtitle(sprintf(
            "p(loo<elpd) using normal approx. with true var, %s, model: %s_%s_%s_%g_%g",
            name, truedist, modeldist, priordist, p0, n
        ))
    print(g)
}
if (F) {
    dev.new()
    g = ggplot()
    g = g + geom_histogram(data = data.frame(x=test_p), aes(x=x))
    g = g +
        xlab("p(loo<elpd)") +
        ylab("") +
        ggtitle(sprintf(
            "p(loo<elpd) using normal approx. with naive var estimate, %s, model: %s_%s_%s_%g_%g",
            name, truedist, modeldist, priordist, p0, n
        ))
    print(g)
}
# plot pairwise p(loo<elpd)
if (T) {
    dev.new()
    g = ggplot()
    g = g + geom_point(data = data.frame(x=ref_p, y=test_p), aes(x=x, y=y))
    g = g +
        xlab("with true var") +
        ylab("with naive var") +
        ggtitle(sprintf(
            "p(loo<elpd) using normal approx., %s, model: %s_%s_%s_%g_%g",
            name, truedist, modeldist, priordist, p0, n
        ))
    print(g)
}
# plot pairwise p(loo<elpd) error
if (F) {
    dev.new()
    g = ggplot()
    g = g + geom_point(data = data.frame(x=ref_p, y=test_p-ref_p), aes(x=x, y=y))
    g = g +
        xlab("with true var") +
        ylab("with naive var") +
        ggtitle(sprintf(
            "p(loo<elpd) using normal approx., %s, model: %s_%s_%s_%g_%g",
            name, truedist, modeldist, priordist, p0, n
        ))
    print(g)
}


# ==============================================================================
# calc ranks
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
g = g + geom_bar(
    data = data.frame(x=ranks_comb),
    aes(x=x),
)
g = g +
    geom_hline(yintercept=qbinom(0.005, Niter, 1/B)) +
    geom_hline(yintercept=Niter/B) +
    geom_hline(yintercept=qbinom(0.995, Niter, 1/B)) +
    xlab("rank") +
    ylab("") +
    ggtitle(sprintf(
        "rankplot, model: %s_%s_%s_%g_%g",
        truedist, modeldist, priordist, p0, n
    ))
print(g)
