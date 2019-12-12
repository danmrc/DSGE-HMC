include("priors.jl")

unif = Uniform(0,1)

num_iter = 50000

include("../src/simulation.jl")
include("../gali_bayesian.jl")

pars_aceitos = zeros(num_iter,10)
pars_aceitos[:,1] .= 2/3

pars_aceitos[1,2:10] = TransformVariables.inverse(t,(bet = 0.99,epsilon = 6,theta = 2/3,sig = 1,s2 = 1, phi = 1,phi_pi = 1.5,phi_y = 0.5/4, rho_v = 0.5))# [rand(prior_bet),rand(prior_epsilon),rand(prior_theta),rand(prior_sig),rand(prior_s2),rand(prior_phi),rand(prior_phi_pi),rand(prior_phi_y),rand(prior_rho_v)]

matriz_escala = 0.01

j = 2

while j <= num_iter
    kernel_velho = MvNormal(pars_aceitos[j-1,2:10],matriz_escala)
    novo_par = rand(kernel_velho)
    pars_aceitos[j,2:10] = novo_par
    kernel_novo = MvNormal(novo_par,matriz_escala)

    teste = LogDensityProblems.logdensity(P,novo_par)
    if isnan(teste)
        println("Rejeitado, repetindo")
        continue
    else
        num = teste + logpdf(kernel_novo,pars_aceitos[j-1,2:10])
        dem = LogDensityProblems.logdensity(P,pars_aceitos[j,2:10]) + logpdf(kernel_velho,novo_par)
        alpha = exp(min(0,num - dem))
        p = rand(unif)
        if alpha < p
            pars_aceitos[j,2:10] = pars_aceitos[j-1,2:10]
        else
            pars_aceitos[j,2:10] = novo_par
        end
        println("Iteração ", j)
        global j += 1
    end
end
