using Parameters

struct DSGE_Model{data <: AbstractArray, alfa <: Float64, bet <: Float64, phi <:Float64, var_pars <: Array{Float64}}
    "Data"
    data::data
    "Alpha"
    alfa::alfa
    "Beta"
    bet::bet
    "Phi"
    phi::phi
    "VAR EUA"
    var_pars::var_pars
end

function (problem::DSGE_Model)(pars)
    @unpack alfa,bet,phi,var_pars, data = problem
    @unpack h,sig,eta,delta_D,theta_D,delta_I,theta_I, chi, rho_a, rho_gamma, rho_cp, rho_phi, sigma_a,sigma_gamma, sigma_cp,sigma_phi, rho_1, rho_2, lambda_pi, lambda_y, lambda_s, sigma_i = pars
    llh = log_like_dsge([alfa;bet;phi;var_pars;h;sig;eta;delta_D;theta_D;delta_I;theta_I; chi; rho_a; rho_gamma; rho_cp; rho_phi; sigma_a;sigma_gamma; sigma_cp;sigma_phi; rho_1; rho_2; lambda_pi; lambda_y; lambda_s; sigma_i],data)
        return llh+logpdf(prior_h,h)+logpdf(prior_sig,sig)+logpdf(prior_eta,eta)+logpdf(prior_delta_D,delta_D)+logpdf(prior_theta_D,theta_D)+logpdf(prior_delta_I,delta_I)+logpdf(prior_theta_I,theta_I)+logpdf(prior_chi,chi)+logpdf(prior_rho_a,rho_a)+logpdf(prior_rho_gamma,rho_gamma)+logpdf(prior_rho_cp,rho_cp)+logpdf(prior_rho_phi,rho_phi)+logpdf(prior_sigma_a,sigma_a)+logpdf(prior_sigma_gamma,sigma_gamma)+logpdf(prior_sigma_cp,sigma_cp)+logpdf(prior_sigma_phi,sigma_phi)+logpdf(prior_rho_1,rho_1)+logpdf(prior_rho_2,rho_2)+logpdf(prior_lambda_pi,lambda_pi)+logpdf(prior_lambda_y,lambda_y)+logpdf(prior_lambda_s,lambda_s)+logpdf(prior_sigma_i,sigma_i)
end

function problem_transform(p::DSGE_Model)
    as((h = as_unit_interval, sig = as_positive_real, eta = as_positive_real, delta_D = as_unit_interval, theta_D = as_unit_interval, delta_I = as_unit_interval, theta_I = as_unit_interval, chi = as_positive_real, rho_a = as_unit_interval, rho_gamma = as_unit_interval, rho_cp = as_unit_interval, rho_phi = as_unit_interval, sigma_a = as_positive_real, sigma_gamma = as_positive_real, sigma_cp = as_positive_real, sigma_phi = as_positive_real, rho_1 = as_unit_interval,rho_2 = as_unit_interval,lambda_pi = as_positive_real,lambda_y = as_positive_real,lambda_s = as_positive_real,sigma_i = as_positive_real,))
end
