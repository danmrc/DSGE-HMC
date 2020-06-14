using Parameters, ForwardDiff

struct DSGE_Model{data <: AbstractArray, alfa <: Float64}
    "Data"
    data::data
    "Alpha"
    alfa::alfa
end

function (problem::DSGE_Model)(pars)
    @unpack alfa, data = problem
    @unpack bet, epsilon, theta, sig, s2, phi, phi_pi, phi_y, rho_v = pars
    log_p_bet(bet) = logpdf(prior_bet,bet)
    log_p_epsilon(epsilon) = logpdf(prior_epsilon,epsilon)
    log_p_theta(theta) = logpdf(prior_theta,theta)
    log_p_sig(sig) = logpdf(prior_sig,sig)
    log_p_s2(s2) = logpdf(prior_s2,s2)
    log_p_phi(phi) = logpdf(prior_phi,phi)
    log_p_phi_pi(phi_pi) = logpdf(prior_phi_pi,phi_pi)
    log_p_phi_y(phi_y) = logpdf(prior_phi_y,phi_y)
    log_p_rho_v(rho_v) = logpdf(prior_rho_v,rho_v)
    llh,dll = log_like_dsge([alfa,bet,epsilon,theta,sig,s2,phi,phi_pi,phi_y,rho_v],data)
    llh = llh + log_p_bet(bet) + log_p_epsilon(epsilon) + log_p_theta(theta) + log_p_sig(sig) + log_p_s2(s2) + log_p_phi(phi) + log_p_phi_pi(phi_pi) + log_p_phi_y(phi_y) + log_p_rho_v(rho_v)
    dll = dll + [0 ForwardDiff.derivative(log_p_bet,bet) ForwardDiff.derivative(log_p_epsilon,epsilon) ForwardDiff.derivative(log_p_theta,theta) ForwardDiff.derivative(log_p_sig,sig) ForwardDiff.derivative(log_p_s2,s2) ForwardDiff.derivative(log_p_phi,phi) ForwardDiff.derivative(log_p_phi_pi,phi_pi) ForwardDiff.derivative(log_p_phi_y,phi_y) ForwardDiff.derivative(log_p_rho_v,rho_v)]

    return llh,dll
end

function problem_transform(p::DSGE_Model)
    as((bet = as_unit_interval, epsilon=as_positive_real,theta = as_unit_interval, sig = as_positive_real, s2 = as_positive_real, phi = as_positive_real, phi_pi = as_positive_real, phi_y = as_positive_real, rho_v = as_unit_interval))
end
