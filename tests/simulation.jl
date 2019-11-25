using Plots

include(string(pwd(),"/src/simulation.jl"))
include(string(pwd(),"/gali_bayesian.jl"))
include(string(pwd(),"/src/likelihood.jl"))

dados,choques = simulate_dsge(GAMMA_0,GAMMA_1,PSI,PI,5000)

plot(dados[:,1], label = "Quartely CPI")
plot!(dados[:,2], label = "GDP")
plot!(dados[:,3], label = "i")

plot(dados[:,4], label = "v")
plot!(dados[:,3], label = "i")

param = [alfa,bet,epsilon,theta,sig,1,phi,phi_pi,phi_y,rho_v]

ss, filtered,loglike = log_like_dsge(param,dados[:,2])

plot(dados[1:100,1])
plot!(filtered[1:100,1])

param1 = [alfa*2,bet,epsilon,theta,sig,1,phi,phi_pi,phi_y,rho_v]

a2,ll = log_like_dsge(param1,dados[:,2])

bett = 0.70:0.005:1.15

vals = collect(bett)

for i in 1:length(bett)
    param1 = [alfa,bett[i],epsilon,theta,sig,1,phi,phi_pi,phi_y,rho_v]
    ss,filtered,vals[i] = log_like_dsge(param1,dados[:,2])
    println(i)
end

plot(bett,vals)
