using Distributions, ForwardDiff, Plots, LinearAlgebra

replics = 1000

x = randn(100)

y = 1.5*x + randn(100)

prior = Normal(0,1)

par = zeros(replics)

likelihood(y,x,par) = sum(logpdf.(Normal(0,1),y - x .*par))

loglike(y,x,par) = likelihood(y,x,par)+logpdf.(prior,par)

unif = Uniform(0,1)

M = I(9)

M_inv = inv(M)

moment_dist = Normal(0,1.5)

par[1] = rand(prior)

for i in 2:replics
    moment = rand(moment_dist)
    old_moment = moment
    par_aux = par[i-1]

    L = ceil(rand(Uniform(0,500)))
    leapfrog_step = 1/L

    for j in 1:L
        new_moment = moment + 1/2*leapfrog_step*ForwardDiff.derivative(pp->loglike(y,x,pp),par_aux)
         par_aux = par_aux .+ leapfrog_step*1*new_moment
         moment = new_moment + 1/2*leapfrog_step*ForwardDiff.derivative(pp->loglike(y,x,pp),par_aux)
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
end

histogram(par[100:1000])
