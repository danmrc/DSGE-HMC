using LogDensityProblems
using Distributions, Parameters
using Calculus, StatsPlots, PositiveFactorizations

include(string(pwd(),"/src/simulation.jl"))
include(string(pwd(),"/gali_bayesian.jl"))
include(string(pwd(),"/mcmc/priors.jl"))
include(string(pwd(),"/mcmc/foos.jl"))

yy,shocks = simulate_dsge(GAMMA_0,GAMMA_1,PSI,PI,500)

prob = DSGE_Model(yy[:,2],1/3)

prob((bet = 0.99,epsilon = 6,theta=2/3,sig=1,s2=1,phi=1,phi_pi=1.5,phi_y=0.5/4,rho_v=0.5))

t = problem_transform(prob)

P = TransformedLogDensity(t,prob)

unif = Uniform(0,1)

num_iter = 100000

pars_aceitos = zeros(num_iter,10)
pars_aceitos[:,1] .= 1/3

true_vals = TransformVariables.inverse(t,(bet = 0.99,epsilon = 6,theta = 2/3,sig = 1,s2 = 1, phi = 1,phi_pi = 1.5,phi_y = 0.5/4, rho_v = 0.5))#

#[rand(prior_bet),rand(prior_epsilon),rand(prior_theta),rand(prior_sig),rand(prior_s2),rand(prior_phi),rand(prior_phi_pi),rand(prior_phi_y),rand(prior_rho_v)]

npar = 9

coef_escala = 0.0075 #2.4/sqrt(npar)

hes = -Calculus.hessian(x->LogDensityProblems.logdensity(P,x),true_vals)

hes_inv = inv(hes)

hes_inv = (hes_inv + hes_inv')/2

hes_inv = cholesky(Positive,hes_inv)

hes_inv = hes_inv.L*hes_inv.L'

#hes = round.(hes;digits=3)

isposdef(hes_inv)

pars_aceitos[1,2:10] = rand(MvNormal(hes_inv),1)

j = 2
rejec = 0

while j <= num_iter
    kernel_velho = MvNormal(pars_aceitos[j-1,2:10],coef_escala*hes_inv)
    novo_par = rand(kernel_velho)
    #pars_aceitos[j,2:10] = novo_par
    kernel_novo = MvNormal(novo_par,coef_escala*hes_inv)

    teste = LogDensityProblems.logdensity(P,novo_par)
    if isnan(teste)
        #pars_aceitos[j,2:10] = pars_aceitos[j-1,2:10]
        #global rejec += 1
        #global j += 1
        continue
    else
        num = teste + logpdf(kernel_novo,pars_aceitos[j-1,2:10])
        #println(string(teste," ",logpdf(kernel_novo, pars_aceitos[j-1,2:10])))
        dem = LogDensityProblems.logdensity(P,pars_aceitos[j-1,2:10]) + logpdf(kernel_velho,novo_par)
        alpha = exp(min(0,num - dem))
        p = rand(unif)
        if alpha < p
            pars_aceitos[j,2:10] = pars_aceitos[j-1,2:10]
            global rejec += 1
        else
            pars_aceitos[j,2:10] = novo_par
        end
        acc = 1 - rejec/j
        if j % 50 == 0
            println("Iteração ", j, " taxa de aceitação ", acc)
            sleep(0.4)
        end
        global j += 1
    end
end
