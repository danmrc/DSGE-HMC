using Distributions,StatsPlots, ForwardDiff, LinearAlgebra, PositiveFactorizations, Optim

n_samples = 2000

n = 100

x = randn(n,50)
theta = [repeat([1],10); repeat([0],40)]
y = x*theta + randn(n)

prior = Laplace(0,2)

ll(theta) = logpdf(MvNormal(I(n)),y-x*theta)

post(theta) = ll(theta) + sum(logpdf.(prior,theta))

init = rand(prior,50)

grad(theta) = ForwardDiff.gradient(post,theta)

ot = optimize(x->-1*ll(x),zeros(50),BFGS(), autodiff=:forward)

M = grad(ot.minimizer)*grad(ot.minimizer)'#mass matrix
M = inv(diagm(diag(M)))

mom_dist = MvNormal(M)

tol_psi = 1e15

#######
### start here
#

pars = zeros(n_samples,50)

uni = Uniform(0,1)

leap_step = 0.1

L = max(Int(ceil(1/leap_step)),20)

accept = 0

old_par = init#repeat([0],50)

leap_dist = Normal(leap_step,0.05)

for j = 1:500
    psi = rand(mom_dist)
    old_psi = psi
    new_par = old_par
    leap_step = rand(leap_dist)
    for i = 1:L
        psi = psi + 1/2*leap_step*grad(new_par)
        new_par = new_par + inv(M)*leap_step*psi
        psi = psi + 1/2*leap_step*grad(new_par)
    end
    #println(psi)
    if any(psi .> tol_psi)
        @info "Energy diverging. Quitting"
        break
    end
    r = post(new_par) + logpdf(mom_dist,psi) - (post(old_par)+ logpdf(mom_dist,old_psi))
    a = rand(uni)
    r = min(1,exp(r))
    if r > a
        pars[j,:] = new_par
        global old_par = new_par
        global accept = accept + 1
    else
        pars[j,:] = old_par
    end
    println("Iteration ", j, " Accept ratio ", accept/j*100, "%")
end

pars_tuning = pars[1:500,:]

leap_step = 0.145

L = max(Int(ceil(1/leap_step)),20)

accept = 0

old_par = reshape(mapslices(mode,pars_tuning,dims = 1),50)

leap_dist = Normal(leap_step,0.05)

for j = 1:n_samples
    psi = rand(mom_dist)
    old_psi = psi
    new_par = old_par
    for i = 1:L
        psi = psi + 1/2*leap_step*grad(new_par)
        new_par = new_par + inv(M)*leap_step*psi
        psi = psi + 1/2*leap_step*grad(new_par)
    end
    #println(psi)
    r = post(new_par) + logpdf(mom_dist,psi) - (post(old_par)+ logpdf(mom_dist,old_psi))
    a = rand(uni)
    r = min(1,exp(r))
    if r > a
        pars[j,:] = new_par
        global old_par = new_par
        global accept = accept + 1
    else
        pars[j,:] = old_par
    end
    println("Iteration ", j, " Accept ratio ", accept/j*100, "%")
end

StatsPlots.density(pars[500:2000,4])

mm = reshape(mapslices(mode,pars[500:2000,:],dims = 1),50)

scatter(mm)

for i = 1:50
    p = StatsPlots.density(pars[500:2000,i], title = string("Par ",i), legend = :none)
    title!
    png(p,string("/home/danielc/Documentos/GitHub/azul/content/post/mcmc/par",i,".png"))
end
