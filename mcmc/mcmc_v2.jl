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

adapt = 50000
adapt_warm_up = 5000

num_iter = 100000

npar = 9

include("../src/simulation.jl")
include("../gali_bayesian.jl")

pars_aceitos = zeros(num_iter,npar+1)
pars_aceitos[:,1] .= 1/3

true_vals = TransformVariables.inverse(t,(bet = 0.99,epsilon = 6,theta = 2/3,sig = 1,s2 = 1, phi = 1,phi_pi = 1.5,phi_y = 0.5/4, rho_v = 0.5))#

#[rand(prior_bet),rand(prior_epsilon),rand(prior_theta),rand(prior_sig),rand(prior_s2),rand(prior_phi),rand(prior_phi_pi),rand(prior_phi_y),rand(prior_rho_v)]

hes = -Calculus.hessian(x->LogDensityProblems.logdensity(P,x),true_vals)

hes_inv = inv(hes)

########################################################
### Adaptative phase: finding optimal c and hes_inv ###
########################################################

warm_up_pars = zeros(adapt_warm_up,npar+1)
warm_up_pars[:,1] .= 1/3

hes_inv = inv(hes)

hes_inv = (hes_inv + hes_inv')/2

hes_inv = cholesky(Positive,hes_inv)

hes_inv = hes_inv.L*hes_inv.L'

isposdef(hes_inv)

warm_up_pars[1,2:10] = rand(MvNormal(hes_inv),1)

coef_escala = 2.4/sqrt(npar)

rejec = 0

k = 2

while k <= adapt_warm_up
    kernel_velho = MvNormal(warm_up_pars[k-1,2:10],coef_escala*hes_inv)
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
        num = teste + logpdf(kernel_novo,warm_up_pars[k-1,2:10])
        #println(string(teste," ",logpdf(kernel_novo, pars_aceitos[j-1,2:10])))
        dem = LogDensityProblems.logdensity(P,warm_up_pars[k-1,2:10]) + logpdf(kernel_velho,novo_par)
        alpha = exp(min(0,num - dem))
        p = rand(unif)
        if alpha < p
            warm_up_pars[k,2:10] = warm_up_pars[k-1,2:10]
            global rejec += 1
        else
            warm_up_pars[k,2:10] = novo_par
        end
    end

    if k % 500 == 0
        acc = 1 - rejec/k
        println("Warm up phase, iteration " ,k, " acceptance ", acc*100,"%")
    end

    if k % 1000 == 0
        pars_space = warm_up_pars[(k-499):k,2:10]
        global hes_inv = cov(pars_space)
        global hes_inv = (hes_inv + hes_inv' )/2
        global hes_inv = cholesky(Positive,hes_inv)
        global hes_inv = hes_inv.L*hes_inv.L'
        global rejec = 0
    end
    global k += 1
end

hes_inv_backup = hes_inv

j = 2
rejec = 0
test = true

coef_escala = 2.4/sqrt(npar)

final_countdown = 0

pars_aceitos[1,2:10] = warm_up_pars[adapt_warm_up,2:10]

while j <= adapt && test
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
            if j % 1200 > 900
                global rejec += 1
            end
        else
            pars_aceitos[j,2:10] = novo_par
        end
    end
    if j % 1200 == 0
        acc = 1 - rejec/300
        #pars_space = pars_aceitos[(j-299):j,2:10]
        # global hes_inv = cov(pars_space)
        # global hes_inv = (hes_inv + hes_inv' )/2
        # global hes_inv = cholesky(Positive,hes_inv)
        # global hes_inv = hes_inv.L*hes_inv.L'
        if round(acc;digits=4) > 0.3
            rescale = 1.1#acc/0.23
            global coef_escala = coef_escala*rescale
            println("Adaptação: iteração ", j, " taxa de aceitação ", acc*100, "% new c = ", coef_escala)
            sleep(0.4)
            #pars_aceitos[j,2:10] = pars_aceitos[1,2:10]
            global j += 1
            global rejec = 0
        elseif round(acc;digits=4) < 0.18
            rescale = 0.8#acc/0.23
            global coef_escala = coef_escala*rescale
            println("Adaptação: iteração ", j, " taxa de aceitação ", acc*100, "% new c = ", coef_escala)
            sleep(0.4)
            #pars_aceitos[j,2:10] = pars_aceitos[1,2:10]
            global j += 1
            global rejec = 0
        else
            acc = 1 - rejec/300
            println("Adaptação: iteração ", j, " taxa de aceitação ", acc*100, "% c = ", coef_escala)
            global rejec = 0
            while final_countdown <= 500
                kernel_velho = MvNormal(pars_aceitos[j+final_countdown-1,2:10],coef_escala*hes_inv)
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
                    num = teste + logpdf(kernel_novo,pars_aceitos[j+final_countdown-1,2:10])
                    #println(string(teste," ",logpdf(kernel_novo, pars_aceitos[j-1,2:10])))
                    dem = LogDensityProblems.logdensity(P,pars_aceitos[j-1,2:10]) + logpdf(kernel_velho,novo_par)
                    alpha = exp(min(0,num - dem))
                    p = rand(unif)
                    if alpha < p
                        pars_aceitos[j+final_countdown,2:10] = pars_aceitos[j+final_countdown-1,2:10]
                        global rejec += 1
                    else
                        pars_aceitos[j+final_countdown,2:10] = novo_par
                    end
                    acc = 1 - rejec/final_countdown
                end
                println("Adaptação: iteração ", final_countdown, " taxa de aceitação ", acc*100, "% c = ", coef_escala)
                global final_countdown += 1
            end
            global test = false
        end
    else
        global j += 1
    end
end

####################################################
#### RWMC for sampling the true distribution ######
###################################################


j = 2
rejec = 0

pars_aceitos[1,2:10] = rand(MvNormal(hes_inv),1)

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
