using Distributions, ForwardDiff, Plots, LinearAlgebra

include(string(pwd(),"/hmc/rkhs.jl"))

replics = 3000

x = randn(500)

y = 1.5*x + randn(500)

prior = Normal(0.5,1)

likelihood(y,x,par) = sum(logpdf.(Normal(0,1),y - x .*par))

loglike(y,x,par) = likelihood(y,x,par)+logpdf.(prior,par)

unif = Uniform(0,1)

moment_dist = Normal(0,2)

N = 200

dados = zeros(N,2)

for i in 1:N
    xx = rand(prior)
    dados[i,:] = [xx;loglike(y,x,xx)]
end

sig = 1

rkhs_approx = rkhs_gaussian(dados[:,1],dados[:,2],sig,0)

f_hat(x) = rkhs_approx(x)

dif_aprox(x) = ForwardDiff.derivative(f_hat,x)
dif_exact(xx) = ForwardDiff.derivative(z->loglike(y,x,z),xx)

gridd = -2:0.05:2

diff_exact_vals = dif_exact.(gridd)
diff_approx_vals = dif_aprox.(gridd)

plot(gridd,diff_approx_vals)
plot!(gridd,diff_exact_vals)

par = zeros(replics)
par[1] = rand(prior)

moments = zeros(replics,500)

for i in 2:replics
    moment = rand(moment_dist)
    old_moment = moment
    par_aux = par[i-1]

    L = ceil(rand(Uniform(100,500)))
    leapfrog_step = 1/L

    for j in 1:L
        new_moment = moment + 1/2*leapfrog_step*ForwardDiff.derivative(pp->f_hat(pp),par_aux)
         par_aux = par_aux .+ leapfrog_step*1*new_moment
         moment = new_moment + 1/2*leapfrog_step*ForwardDiff.derivative(pp->f_hat(pp),par_aux)
         moments[i,Int64(j)] = new_moment
    end

    num = loglike(y,x,par_aux) + logpdf(moment_dist,moment)#+ logpdf(kernel_novo,par[i-1])

    dem = loglike(y,x,par[i-1]) + logpdf(moment_dist,old_moment)#+ logpdf(kernel,novo_par)
    alpha = exp(min(0,num - dem))
    p = rand(unif)
    if alpha < p
        par[i] = par[i-1]
    else
        par[i] = par_aux
    end
    println(i)
    if i % 400 == 0
        novos_dados = sample(par[1:i],N)
        novas_ll = [loglike(y,x,novos_dados[j]) for j in 1:N]
        rkhs_approx = rkhs_gaussian(novos_dados,novas_ll,sig,0)
        global f_hat(x) = rkhs_approx(x)
        dif_aprox(x) = ForwardDiff.derivative(f_hat,x)
        dif_exact(xx) = ForwardDiff.derivative(z->loglike(y,x,z),xx)

        gridd = -2:0.05:2

        diff_exact_vals = dif_exact.(gridd)
        diff_approx_vals = dif_aprox.(gridd)
        #println("Dif Error", sqrt(sum((diff_exact_vals - diff_approx_vals).^2)))
    end
end

histogram(par[100:3000])

par_rwmh = zeros(replics)
par_rwmh[1] = par[1]

for i in 2:replics
    kernel = Normal(par_rwmh[i-1],0.5)
    novo_par = rand(kernel)
    kernel_novo = Normal(novo_par,0.5)

    num = likelihood(y,x,novo_par) + logpdf(prior,novo_par) + logpdf(kernel_novo,par_rwmh[i-1])

    dem = likelihood(y,x,par_rwmh[i-1]) + logpdf(prior,par_rwmh[i-1]) + logpdf(kernel,novo_par)
    alpha = exp(min(0,num - dem))
    p = rand(unif)
    if alpha < p
        par_rwmh[i] = par_rwmh[i-1]
    else
        par_rwmh[i] = novo_par
    end
    println(i)
end

using StatsPlots

StatsPlots.density(par[100:1000])
StatsPlots.density!(par_rwmh[100:1000])
vline!([1.5])
