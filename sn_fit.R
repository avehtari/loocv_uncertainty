
skewness_weighted = function(x, w){
    sum_w = sum(w)
    sum_w_2 = sum_w^2
    sum_w_3 = sum_w^3
    sum_w2 = sum(w^2)
    sum_w3 = sum(w^3)
    resid_x_w = x - sum(w*x)/sum_w
    sd_x_w = sqrt(sum(w*resid_x_w^2) * sum_w_2 / (sum_w * (sum_w_2 - sum_w2)))
    skew_x_w = sum(w*resid_x_w^3) * sum_w_3 /
        (sum_w * sd_x_w^3 * (sum_w_3 - 3*sum_w*sum_w2 + 2*sum_w3))
    skew_x_w
}


sn_from_moments = function(mu, sigma, skew){
    # fit skew normal from sample moments
    # returns xi, omega, alpha

    # limit skew
    skew[skew>0.995] = 0.99
    skew[skew<(-0.995)] = -0.99

    delta2 = pi/(2*(1+((4-pi)/(2*abs(skew)))**(2/3)))
    delta = sqrt(delta2) * sign(skew)

    alpha = delta/sqrt(1-delta2)
    omega = sigma/sqrt(1-2*delta2/pi)
    xi = mu - omega*delta*sqrt(2/pi)
    out = list(xi=xi, omega=omega, alpha=alpha)
    return(out)
}
