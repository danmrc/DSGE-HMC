pars_convertidos = zeros(100000,9)

for i in 50001:100000
    @unpack bet,epsilon,theta,sig,s2,phi,phi_pi,phi_y,rho_v = TransformVariables.transform(t,pars_aceitos[i,2:10])
    pars_convertidos[i,:] = [bet,epsilon,theta,sig,s2,phi,phi_pi,phi_y,rho_v]#(bet = pars_aceitos[i,2], epsilon = pars_aceitos[i,3], theta = pars_aceitos[i,4],sig = pars_aceitos[i,5],s2 = pars_aceitos[i,6],phi = pars_aceitos[i,7],phi_pi = pars_aceitos[i,8],phi_y = pars_aceitos[i,9],rho_v = pars_aceitos[i,10]))
end

using StatsPlots, Latexify

StatsPlots.density(pars_convertidos[50001:100000,1], label= latexify("beta()")) #)"β")
StatsPlots.density(pars_convertidos[50001:100000,2], label="epsilon") #)"ε")
StatsPlots.density(pars_convertidos[50001:100000,3], label="theta") #)"θ")
StatsPlots.density(pars_convertidos[50001:100000,4], label="sigma") #)"σ")
StatsPlots.density(pars_convertidos[50001:100000,5], label="sigma_v") #)"σ_v")
StatsPlots.density(pars_convertidos[50001:100000,6], label="phi") #)"φ")
StatsPlots.density(pars_convertidos[50001:100000,7], label="phi_pi") #)"φ_π")
StatsPlots.density(pars_convertidos[50001:100000,8], label="phi_y") #)"φ_y")
StatsPlots.density(pars_convertidos[50001:100000,9], label="rho_v") #)"ρ_ν")
