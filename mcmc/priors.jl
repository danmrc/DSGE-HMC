include("../src/likelihood.jl")

using Distributions

gamma_mv(mu,std) = Gamma(mu^2/std^2,std^2/mu)
gammainv_mv(mu,std) = Gamma(mu^2/std^2,(std^2/mu)^(-1))

prior_bet = Beta(100,1.01010101)
prior_epsilon = Normal(6,1.5)
prior_theta = Beta(3,1.5)
prior_sig = gamma_mv(1,0.5)
prior_phi = gamma_mv(1,0.5)
prior_phi_pi = gamma_mv(1.5,0.4)
prior_phi_y = gamma_mv(0.5/4,0.2)
prior_rho_v = Beta(0.88,0.88)
prior_s2 = InverseGamma(3.76,2.76)

gammainv(alpha,mu) = InverseGamma(alpha,(alpha-1)*mu)

Plots.plot(3.75:0.05:3.8, std.(gammainv.(3.75:0.05:3.8,1)))

function posterior(par,data)
    aa,bb,llh = log_like_dsge(par,data)
    return llh+logpdf(prior_bet,par[2])+logpdf(prior_epsilon,par[3])+logpdf(prior_theta,par[4])+logpdf(prior_sig,par[5])+logpdf(prior_s2,par[6])+logpdf(prior_phi,par[7])+logpdf(prior_phi_pi,par[8])+logpdf(prior_phi_y,par[9])+logpdf(prior_rho_v,par[10])
end
