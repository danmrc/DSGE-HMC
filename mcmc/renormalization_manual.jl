using ForwardDiff

include(string(pwd(),"/src/likelihood_and_diff.jl"))

to_unit(x) = 1/(1+exp(-x))
to_positive(x) = exp(x)
to_one_inf(x) = exp(x) + 1

d_to_unit(x) = exp(x)/(exp(-x)+1)^2
d_to_positive(x) = exp(x)
d_to_one_inf(x) = exp(x)

function renormalization(par)

    J_1 = d_to_unit(par[2])
    J_2 = d_to_positive(par[3])
    J_3 = d_to_unit(par[4])
    J_4 = d_to_positive(par[5])
    J_5 = d_to_positive(par[6])
    J_6 = d_to_positive(par[7])
    J_7 = d_to_one_inf(par[8])
    J_8 = d_to_positive(par[9])
    J_9 = d_to_unit(par[10])

    return([J_1;J_2;J_3;J_4;J_5;J_6;J_7;J_8;J_9])
end

function dens(par,data)
    alfa = par[1]
    bet = to_unit(par[2])
    epsilon = to_positive(par[3])
    theta = to_unit(par[4])
    sig = to_positive(par[5])
    s2 = to_positive(par[6])
    phi = to_positive(par[7])
    phi_pi = to_one_inf(par[8])
    phi_y = to_positive(par[9])
    rho_v = to_unit(par[10])

    renorm_par = [alfa;bet;epsilon;theta;sig;s2;phi;phi_pi;phi_y;rho_v]

    renorm = renormalization(renorm_par)
    jacob = abs(prod(renorm))

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
    llh = llh + size(data,1)*log(jacob) + log_p_bet(bet) + log_p_epsilon(epsilon) + log_p_theta(theta) + log_p_sig(sig) + log_p_s2(s2) + log_p_phi(phi) + log_p_phi_pi(phi_pi) + log_p_phi_y(phi_y) + log_p_rho_v(rho_v) + sum(log.(renorm))
    return llh
end

function dens_and_grad(par,data)
    alfa = par[1]
    bet = to_unit(par[2])
    epsilon = to_positive(par[3])
    theta = to_unit(par[4])
    sig = to_positive(par[5])
    s2 = to_positive(par[6])
    phi = to_positive(par[7])
    phi_pi = to_one_inf(par[8])
    phi_y = to_positive(par[9])
    rho_v = to_unit(par[10])

    renorm_par = [alfa;bet;epsilon;theta;sig;s2;phi;phi_pi;phi_y;rho_v]

    renorm = renormalization(renorm_par)
    jacob = abs(prod(renorm))

    renorm_diff = ForwardDiff.gradient(x->abs(prod(renormalization(x))),par) #derivate of abs det of Jacobian matrix

    diff_correction = 1/jacob*renorm_diff
    diff_correction = diff_correction[2:10]

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
    llh = llh + size(data,1)*log(jacob) + log_p_bet(bet) + log_p_epsilon(epsilon) + log_p_theta(theta) + log_p_sig(sig) + log_p_s2(s2) + log_p_phi(phi) + log_p_phi_pi(phi_pi) + log_p_phi_y(phi_y) + log_p_rho_v(rho_v) + sum(log.(abs.(renorm)))
    dll = dll[2:10]
    dll = dll .*renorm + size(data,1)*diff_correction + [ForwardDiff.derivative(log_p_bet,bet); ForwardDiff.derivative(log_p_epsilon,epsilon); ForwardDiff.derivative(log_p_theta,theta); ForwardDiff.derivative(log_p_sig,sig); ForwardDiff.derivative(log_p_s2,s2); ForwardDiff.derivative(log_p_phi,phi); ForwardDiff.derivative(log_p_phi_pi,phi_pi); ForwardDiff.derivative(log_p_phi_y,phi_y); ForwardDiff.derivative(log_p_rho_v,rho_v);]

    return llh,dll
end
