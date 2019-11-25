using Plots

include(string(pwd(),"/src/simulation.jl"))
include(string(pwd(),"/gali_bayesian.jl"))
include(string(pwd(),"/src/likelihood.jl"))

dados = simulate_dsge(GAMMA_0,GAMMA_1,PSI,PI,100)

plot(dados[:,1], label = "Quartely CPI")
plot!(dados[:,2], label = "GDP")
plot!(dados[:,3], label = "i")
plot(dados[:,4])

param = [alfa,bet,epsilon,theta,sig,1,rho_v,phi_pi,phi_y]

log_like_dsge(param,dados[:,1])
