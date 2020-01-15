using Distributions, ForwardDiff, Plots, LinearAlgebra
using LogDensityProblems, Parameters, PositiveFactorizations

include(string(pwd(),"/hmc/rkhs.jl"))

include(string(pwd(),"/src/simulation.jl"))
include(string(pwd(),"/gali_bayesian.jl"))
include(string(pwd(),"/mcmc/priors_v2.jl"))
include(string(pwd(),"/mcmc/foos.jl"))

yy,shocks = simulate_dsge(GAMMA_0,GAMMA_1,PSI,PI,500)

prob = DSGE_Model(yy[:,2],1/3)

prob((bet = 0.99,epsilon = 6,theta=2/3,sig=1,s2=1,phi=1,phi_pi=1.5,phi_y=0.5/4,rho_v=0.5))

t = problem_transform(prob)

P = TransformedLogDensity(t,prob)

true_vals = TransformVariables.inverse(t,(bet = 0.99,epsilon = 6,theta=2/3,sig=1,s2=1,phi=1,phi_pi=1.5,phi_y=0.5/4,rho_v=0.5))

LogDensityProblems.logdensity(P,true_vals)

unif = Uniform(0,1)

M = I(9)
M_inv = inv(M)

replics = 3000
warm_up_size = 2000

moment_dist = MvNormal(M)

par = zeros(replics,9)

par[1,:] = [rand(prior_bet);rand(prior_epsilon);rand(prior_theta);rand(prior_sig);rand(prior_phi);rand(prior_phi_pi);rand(prior_phi_y);rand(prior_rho_v);rand(prior_s2)]

N = 300

# dados = zeros(N,10)
#
# for i in 1:N
#     xx = [rand(prior_bet);rand(prior_epsilon);rand(prior_theta);rand(prior_sig);rand(prior_phi);rand(prior_phi_pi);rand(prior_phi_y);rand(prior_rho_v);rand(prior_s2)]
#     dados[i,:] = [xx;LogDensityProblems.logdensity(P,xx)]
# end

par_warmup = zeros(warm_up_size,9)

par_warmup[1,:] = par[1,:]

coef_escala = 0.2#2.4/sqrt(9)

j = 2
rejec = 0

while j <= warm_up_size
    kernel_velho = MvNormal(par_warmup[j-1,:],coef_escala*M)
    novo_par = rand(kernel_velho)
    #pars_aceitos[j,2:10] = novo_par
    kernel_novo = MvNormal(novo_par,coef_escala*M)

    teste = LogDensityProblems.logdensity(P,novo_par)
    if isnan(teste)
        #pars_aceitos[j,2:10] = pars_aceitos[j-1,2:10]
        #global rejec += 1
        #global j += 1
        continue
    else
        num = teste + logpdf(kernel_novo,par_warmup[j-1,:])
        #println(string(teste," ",logpdf(kernel_novo, pars_aceitos[j-1,2:10])))
        dem = LogDensityProblems.logdensity(P,par_warmup[j-1,:]) + logpdf(kernel_velho,novo_par)
        alpha = exp(min(0,num - dem))
        p = rand(unif)
        if alpha > p
            par_warmup[j,:] = novo_par
        else
            par_warmup[j,:] = par_warmup[j-1,:]
            global rejec += 1
        end
        acc = 1 - rejec/j
        if j % 50 == 0
            println("Iteração ", j, " taxa de aceitação ", acc)
            sleep(0.4)
            cov_aux = cov(par_warmup[1:j,:])
            cov_aux = cholesky(Positive,cov_aux)
            global M = cov_aux.L*cov_aux.L'
            par_warmup[j,:] = rand(MvNormal(M))
        end
        global j += 1
    end
end

M_aux = cov(par_warmup[1000:2000,:])
M_aux = inv(M_aux)

M = diagm(diag(M_aux))*1e-9
M_inv = inv(M)

sigg = I(9)

# wp_idx = sample(1:size(dados,1),N)
#
# ll_val = [LogDensityProblems.logdensity(P,dados[wp_idx[j],1:9]) for j in 1:N]
#
# rkhs_approx = rkhs_gaussian(dados[wp_idx,1:9],ll_val,sigg,0)

wp_idx = sample(1000:size(par_warmup,1),N)

ll_val = [LogDensityProblems.logdensity(P,par_warmup[wp_idx[j],1:9]) for j in 1:N]

rkhs_approx = rkhs_gaussian(par_warmup[wp_idx,1:9],ll_val,sigg,0)

f_hat(x) = rkhs_approx(x)

dif_aprox(x) = ForwardDiff.gradient(f_hat,x)

moment_dist = MvNormal(M)

#moments = zeros(replics,5000,9)

i = 2
accep = 0

while i <= replics
    moment = rand(moment_dist)
    old_moment = moment
    par_aux = par[i-1,:]

    L = 1e4#ceil(rand(Uniform(1e4,6e4)))
    leapfrog_step = 1/L#rand(Uniform(1e-9,6e-9))

    for j in 1:L
        new_moment = moment + 1/2*leapfrog_step*ForwardDiff.gradient(pp->f_hat(pp),par_aux)
         par_aux = par_aux .+ leapfrog_step*M_inv*new_moment
         moment = new_moment + 1/2*leapfrog_step*ForwardDiff.gradient(pp->f_hat(pp),par_aux)
         #moments[i,Int64(j),:] = moment
    end

    #println(par_aux)
    num = LogDensityProblems.logdensity(P, par_aux) + logpdf(moment_dist,moment)#+ logpdf(kernel_novo,par[i-1])

    dem = LogDensityProblems.logdensity(P, par[i-1,:]) + logpdf(moment_dist,old_moment)#+ logpdf(kernel,novo_par)

    alpha = exp(min(0,num - dem))

    p = rand(unif)
    if alpha < p
        par[i,:] = par[i-1,:]
    else
        par[i,:] = par_aux
        global accep += 1
    end
    println("Step ",i, " Acceptance ", accep/i)
    # if i % 300 == 0
    #     idxs = sample(1:i,N)
    #     novos_dados = par[idxs,:]
    #     novas_ll = [LogDensityProblems.logdensity(P,novos_dados[j,:]) for j in 1:N]
    #     rkhs_approx = rkhs_gaussian(novos_dados,novas_ll,sig,0)
    #     global f_hat(x) = rkhs_approx(x)
    # end
    global i += 1
end

pars_convertidos = zeros(replics,9)

for i in 1:replics
    @unpack bet,epsilon,theta,sig,s2,phi,phi_pi,phi_y,rho_v = TransformVariables.transform(t,par[i,:])
    pars_convertidos[i,:] = [bet,epsilon,theta,sig,s2,phi,phi_pi,phi_y,rho_v]
end

using StatsPlots

StatsPlots.density(pars_convertidos[1000:3000,1])
StatsPlots.density(pars_convertidos[1000:5000,2])
StatsPlots.density(pars_convertidos[1000:5000,3])
StatsPlots.density(pars_convertidos[1000:5000,4])
StatsPlots.density(pars_convertidos[1000:5000,5])
