using Distributions, Tracker, Plots

x = randn(100)

y = 1.5*x + randn(100)

prior = Normal(0,1)

par = zeros(10_000)

likelihood(y,x,par) = sum(logpdf.(Normal(0,1),y - x .*par))

loglike(y,x,par) = likelihood(y,x,par)+logpdf.(prior,par)[1]

unif = Uniform(0,1)

M = I(9)
leapfrog_step = 1e-8
L = 50

M_inv = inv(M)

moment_dist = Normal(0,1)

moment = randn(1)#rand(moment_dist)

par_aux = randn(1)

for i in 2:10_000
    for j in 1:L
        new_moment = moment[1] + 1/2*leapfrog_step*Tracker.gradient(pp->loglike(y,x,pp),par_aux)[1][1]
        global par_aux = par_aux .+ leapfrog_step*1*new_moment
        global moment = new_moment[1] + 1/2*leapfrog_step*Tracker.gradient(pp->loglike(y,x,pp),par_aux)[1][1]
    end
    kernel = Normal(par[i-1],0.5)
    novo_par = rand(kernel)
    kernel_novo = Normal(novo_par,0.5)

    num = likelihood(y,x,novo_par) + logpdf(prior,novo_par) #+ logpdf(kernel_novo,par[i-1])

    dem = likelihood(y,x,par[i-1]) + logpdf(prior,par[i-1]) #+ logpdf(kernel,novo_par)
    alpha = exp(min(0,num - dem))
    p = rand(unif)
    if alpha < p
        par[i] = par[i-1]
    else
        par[i] = novo_par
    end
    println(i)
end

histogram(par)
