sn_loo_fit = function(mu, sigma, skew){
    # fit skew normal from loo sample moments
    # returns xi, omega, alpha

    # limit skew
    skew[skew>0.99] = 0.99
    skew[skew<(-0.99)] = -0.99

    delta2 = pi/(2*(1+((4-pi)/(2*abs(skew)))**(2/3)))
    delta = sqrt(delta2) * sign(skew)

    alpha = delta/sqrt(1-delta2)
    omega = sigma/sqrt(1-2*delta2/pi)
    xi = mu - omega*delta*sqrt(2/pi)
    out = list(xi=xi, omega=omega, alpha=alpha)
    return(out)
}
