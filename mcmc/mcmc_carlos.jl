using LogDensityProblems, CSV, Optim, DataFrames
using Distributions, Parameters
using Calculus, StatsPlots, PositiveFactorizations

include(string(pwd(),"/src/simulation.jl"))
include(string(pwd(),"/mcmc/priors_carlos.jl"))
include(string(pwd(),"/mcmc/foos_carlos.jl"))

dados = DataFrame(CSV.file("data/canada_data.csv"))

dados = Array(dados)[:,2:9]

par_cal = [0.67; 0.989; 1.66; # alfa, beta, phi
      #-0.34; -7.12; -8.98; # A0
       -40.66; -0.4467; -84.6557;
       0.872; 1.34e-3; 4.911e-05; -29.87; -0.212738; 0.002511; -84.1538; 0.31; 0.9410; # A1
       0.005; 0.798; 1.36; # shocks_star
       0.68; 1.27; 0.58; 0.07; 0.82; 0.13; 0.92; 0.02; 0.79; 0.60; 0.12; 0.88; 5.57; 10.34; 2.08; 1.10; 0.54; 0.30; 1.99; 0.70; 0.31; 0.79]

prob = DSGE_Model(dados,par_cal[1],par_cal[2],par_cal[3],par_cal[4:18])

t = problem_transform(prob)

P = TransformedLogDensity(t,prob)

unif = Uniform(0,1)

num_iter = 1_000_000

pars_aceitos = zeros(num_iter,22)

par_chute = randn(22)

nll(x) = -1*LogDensityProblems.logdensity(P,x)

mode = optimize(nll,par_chute, NelderMead(),Optim.Options(iterations =50_000))

start_par = mode.minimizer

#[rand(prior_bet),rand(prior_epsilon),rand(prior_theta),rand(prior_sig),rand(prior_s2),rand(prior_phi),rand(prior_phi_pi),rand(prior_phi_y),rand(prior_rho_v)]

npar = 22

coef_escala = 0.0075

hes = -Calculus.hessian(x->LogDensityProblems.logdensity(P,x),start_par)

hes_inv = pinv(hes)

hes_inv = (hes_inv + hes_inv')/2

hes_inv = cholesky(Positive,hes_inv)

hes_inv = hes_inv.L*hes_inv.L'

#hes = round.(hes;digits=3)

isposdef(hes_inv)

pars_aceitos[1,:] = pars_inicio

scale_matrix = cov(pars_aceitos[1:4000,:])

j = 2
rejec = 0

while j <= num_iter
    kernel_velho = MvNormal(pars_aceitos[j-1,:],coef_escala*scale_matrix)
    novo_par = rand(kernel_velho)
    #pars_aceitos[j,2:10] = novo_par
    kernel_novo = MvNormal(novo_par,coef_escala*scale_matrix)

    teste = LogDensityProblems.logdensity(P,novo_par)
    if isnan(teste)
        #pars_aceitos[j,2:10] = pars_aceitos[j-1,2:10]
        #global rejec += 1
        #global j += 1
        continue
    else
        num = teste + logpdf(kernel_novo,pars_aceitos[j-1,:])
        #println(string(teste," ",logpdf(kernel_novo, pars_aceitos[j-1,2:10])))
        dem = LogDensityProblems.logdensity(P,pars_aceitos[j-1,:]) + logpdf(kernel_velho,novo_par)
        alpha = exp(min(0,num - dem))
        p = rand(unif)
        if alpha < p
            pars_aceitos[j,:] = pars_aceitos[j-1,:]
            global rejec += 1
        else
            pars_aceitos[j,:] = novo_par
        end
        acc = 1 - rejec/j
        if j % 50 == 0
            println("Iteração ", j, " taxa de aceitação ", acc)
            sleep(0.4)
        end
        global j += 1
    end
end
