include("llh_diff.jl")
include("gali_bayesian.jl")
include("simulation.jl")

yy,shocks = simulate_dsge(GAMMA_0,GAMMA_1,PSI,PI,500)

bet = 0.99
sig = 1
phi = 1
alfa = 1/3
epsilon = 6
theta = 2/3
phi_pi = 1.5
phi_y = 0.5/4
rho_v = 0.5
s2 = 1

true_pars = [alfa,bet,epsilon,theta, sig, s2, phi,phi_pi, phi_y,rho_v]

ll,dif = loglike_dsge(true_pars,yy[:,2])
