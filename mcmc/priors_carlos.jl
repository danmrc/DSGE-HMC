include("../src/likelihood.jl")

using Distributions, TransformVariables

gamma_mv(mu,std) = Gamma(mu^2/std^2, std^2/mu)
gammainv_mv(mu,std) = Gamma(mu^2/std^2, (std^2/mu)^(-1))
beta_mv(mu,std) = Beta(((1-mu)/std^2-1/mu)*mu^2, ((1-mu)/std^2-1/mu)*mu^2*(1/mu-1))

prior_h           = beta_mv(0.85, 0.05)
prior_sig         = gamma_mv(1.3, 0.05)
prior_eta         = gamma_mv(1, 0.5)
prior_delta_D     = beta_mv(0.65, 0.2)
prior_theta_D     = beta_mv(0.65, 0.1)
prior_delta_I     = beta_mv(0.65, 0.2)
prior_theta_I     = beta_mv(0.65, 0.1)
prior_chi         = gamma_mv(0.02, 0.01)
prior_rho_a       = beta_mv(0.5, 0.25)
prior_rho_gamma   = beta_mv(0.5, 0.25)
prior_rho_cp      = beta_mv(0.5, 0.25)
prior_rho_phi     = beta_mv(0.5, 0.25)
prior_sigma_a     = gammainv_mv(1, 0.75)
prior_sigma_gamma = gammainv_mv(1, 0.75)
prior_sigma_cp    = gammainv_mv(1, 0.75)
prior_sigma_phi   = gammainv_mv(1, 0.75)
prior_rho_1       = beta_mv(0.6, 0.15)
prior_rho_2       = beta_mv(0.6, 0.15)
prior_lambda_pi   = gamma_mv(2, 0.5)
prior_lambda_y    = gamma_mv(0.25, 0.1)
prior_lambda_s    = gamma_mv(1.5, 0.5)
prior_sigma_i     = gammainv_mv(1, 0.75)



#gammainv(alpha,mu) = InverseGamma(alpha,(alpha-1)*mu)

#Plots.plot(3.75:0.05:3.8, std.(gammainv.(3.75:0.05:3.8,1)))

# function posterior(par,data)
#     aa,bb,llh = log_like_dsge(par,data)
#     return llh+logpdf(prior_bet,par[2])+logpdf(prior_epsilon,par[3])+logpdf(prior_theta,par[4])+logpdf(prior_sig,par[5])+logpdf(prior_s2,par[6])+logpdf(prior_phi,par[7])+logpdf(prior_phi_pi,par[8])+logpdf(prior_phi_y,par[9])+logpdf(prior_rho_v,par[10])
# end
