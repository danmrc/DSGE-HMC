include("priors.jl")

unif = Uniform(0,1)

include("../src/simulation.jl")
include("../gali_bayesian.jl")

yy,shocks = simulate_dsge(GAMMA_0,GAMMA_1,PSI,PI,500)

pars_aceitos = zeros(5000,10)
pars_aceitos[:,1] .= 2/3

pars_aceitos[1,2:10] = [rand(prior_bet),rand(prior_epsilon),rand(prior_theta),rand(prior_sig),rand(prior_s2),rand(prior_phi),rand(prior_phi_pi),rand(prior_phi_y),rand(prior_rho_v)]

matriz_escala = 1

for i in 2:5000
    kernel_velho = MvNormal(pars_aceitos[i-1,2:10],matriz_escala)
    novo_par = rand(kernel_velho)
    kernel_novo = MvNormal(novo_par,matriz_escala)
    num = posterior([2/3;novo_par],yy[:,2]) + logpdf(kernel_novo,pars_aceitos[i-1,2:10])
    dem = posterior(par[i-1,:],yy[:,2]) + logpdf(kernel_velho,novo_par)
    alpha = exp(min(0,num - dem))
    p = rand(unif)
    if alpha < p
        pars_aceitos[i,2:10] = pars_aceitos[i-1,2:10]
    else
        pars_aceitos[i,2:10] = novo_par
    end
    println("Iteração", i)
end
