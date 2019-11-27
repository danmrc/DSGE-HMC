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

vals_bet = collect(bett)

for i in 1:length(bett)
    param1 = [alfa,bett[i],epsilon,theta,sig,1,phi,phi_pi,phi_y,rho_v]
    ss,filtered,vals_bet[i] = log_like_dsge(param1,dados[:,2])
    println(i)
end

plot(bett,vals_bet)

alff = 0.10:0.01:0.9

vals_alfa = collect(alff)

for i in 1:length(alff)
    param1 = [alff[i],bet,epsilon,theta,sig,1,phi,phi_pi,phi_y,rho_v]
    ss,filtered,vals_alfa[i] = log_like_dsge(param1,dados[:,2])
end

plot(alff,vals)


bett = 0.70:0.005:1.15
alff = 0.10:0.01:0.9

vals = zeros(length(alff),length(bett))

for i in 1:length(alff), j in 1:length(bett)
    param1 = [alff[i],bett[j],epsilon,theta,sig,1,phi,phi_pi,phi_y,rho_v]
    ss,filtered,vals[i,j] = log_like_dsge(param1,dados[:,2])
end

plotly()

plot(alff,bett,vals, st=:surface, camera = (-45,45))
xaxis!("alfa")
yaxis!("beta")

using JLD2

@save "surface.jld2" alff bett vals
