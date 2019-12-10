include("C:\\Users\\daniel.coutinho\\github\\DSGE-HMC\\src\\priors.jl")

using Distributions
using Plots

x = randn(100)

y = 0.5*x + randn(100)

prior = Normal(0,1)

par = zeros(10000)

likelihood(y,x,par) = sum(logpdf.(Normal(0,1),y - x*par))

unif = Uniform(0,1)

for i in 2:10000
    kernel = Normal(par[i-1],0.5)
    novo_par = rand(kernel)
    kernel_novo = Normal(novo_par,0.5)
    
    num = likelihood(y,x,novo_par) + logpdf(prior,novo_par) + pdf(kernel_novo,par[i-1])

    dem = likelihood(y,x,par[i-1]) + logpdf(prior,par[i-1]) + pdf(kernel,novo_par)
    alpha = exp(min(0,num - dem))
    p = rand(unif)
    if alpha < p
        par[i] = par[i-1]
    else
        par[i] = novo_par
    end
end

histogram(par)
