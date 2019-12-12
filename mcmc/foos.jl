using Parameters

struct DSGE_Model{data <: AbstractArray, alfa <: Float64}
    "Data"
    data::data
    "Alpha"
    alfa::alfa
end

function (problem::DSGE_Model)(pars)
    @unpack alfa, data = problem
    @unpack bet, epsilon, theta, sig, s2, phi, phi_pi, phi_y, rho_v = pars
    llh = log_like_dsge([alfa,bet,epsilon,theta,sig,s2,phi,phi_pi,phi_y,rho_v],data)
        return llh+logpdf(prior_bet,bet)+logpdf(prior_epsilon,epsilon)+logpdf(prior_theta,theta)+logpdf(prior_sig,sig)+logpdf(prior_s2,s2)+logpdf(prior_phi,phi)+logpdf(prior_phi_pi,phi_pi)+logpdf(prior_phi_y,phi_y)+logpdf(prior_rho_v,rho_v)
end

function problem_transform(p::DSGE_Model)
    as((bet = as_unit_interval, epsilon=as_positive_real,theta = as_unit_interval, sig = as_positive_real, s2 = as_positive_real, phi = as_positive_real, phi_pi = as_positive_real, phi_y = as_positive_real, rho_v = as_unit_interval))
end
