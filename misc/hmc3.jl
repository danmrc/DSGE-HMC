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

tol_psi = 1e18

#######
### start here
#

pars = zeros(n_samples,50)

uni = Uniform(0,1)

leap_step = 0.09

L = max(Int(ceil(1/leap_step)),20)

window_size = 10

window_dist = DiscreteUniform(1,window_size)

accept = 0

old_par = init#repeat([0],50)

leap_dist = Normal(leap_step,0.05)

for j = 1:500
    psi = rand(mom_dist)
    old_psi = psi
    new_par = old_par
    leap_step = rand(leap_dist)
    s = rand(window_dist)
    forward_leap = zeros(50,2)
    sum_p = 0
    for i = 1:(s*L) #forward leapfrog
        psi = psi + 1/2*leap_step*grad(new_par)
        new_par = new_par + inv(M)*leap_step*psi
        psi = psi + 1/2*leap_step*grad(new_par)
        pp = exp(logpdf(mom_dist,psi)+post(new_par))
        sum_p = sum_p + pp
        if pp/sum_p > rand(uni)
            forward_leap = [psi new_par]
        end
    end
    backward_leap = zeros(50,2)
    sum_p = 0
    psi = old_psi
    new_par = old_par
    for i = 1:((window_size-s)*L) #backward leapfrog
        psi = psi - 1/2*leap_step*grad(new_par)
        new_par = new_par - inv(M)*leap_step*psi
        psi = psi - 1/2*leap_step*grad(new_par)
        pp = exp(logpdf(mom_dist,psi)+post(new_par))
        sum_p = sum_p + pp
        if pp/sum_p > rand(uni)
            backward_leap = [psi new_par]
        end
    end
    #println(psi)
    if any(psi .> tol_psi)
        @info "Energy diverging. Quitting"
        break
    end
    r = post(forward_leap[:,2]) + logpdf(mom_dist,forward_leap[:,1]) - (post(backward_leap[:,2])+ logpdf(mom_dist,backward_leap[:,1]))
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

leap_step = 0.15

L = max(Int(ceil(1/leap_step)),20)

accept = 0

old_par = reshape(mapslices(mode,pars[1:500,:],dims = 1),50)

for j = 1:2000
    psi = rand(mom_dist)
    old_psi = psi
    new_par = old_par
    leap_step = rand(leap_dist)
    s = rand(window_dist)
    forward_leap = zeros(50,2)
    sum_p = 0
    for i = 1:(s*L) #forward leapfrog
        psi = psi + 1/2*leap_step*grad(new_par)
        new_par = new_par + inv(M)*leap_step*psi
        psi = psi + 1/2*leap_step*grad(new_par)
        pp = exp(logpdf(mom_dist,psi)+post(new_par))
        sum_p = sum_p + pp
        if pp/sum_p > rand(uni)
            forward_leap = [psi new_par]
        end
    end
    backward_leap = zeros(50,2)
    sum_p = 0
    psi = old_psi
    new_par = old_par
    for i = 1:((window_size-s)*L) #backward leapfrog
        psi = psi - 1/2*leap_step*grad(new_par)
        new_par = new_par - inv(M)*leap_step*psi
        psi = psi - 1/2*leap_step*grad(new_par)
        pp = exp(logpdf(mom_dist,psi)+post(new_par))
        sum_p = sum_p + pp
        if pp/sum_p > rand(uni)
            backward_leap = [psi new_par]
        end
    end
    #println(psi)
    if any(psi .> tol_psi)
        @info "Energy diverging. Quitting"
        break
    end
    r = post(forward_leap[:,2]) + logpdf(mom_dist,forward_leap[:,1]) - (post(backward_leap[:,2])+ logpdf(mom_dist,backward_leap[:,1]))
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

StatsPlots.density(pars[200:1700,1])

mm = mapslices(mode,pars[500:1126,:],dims = 1)

scatter(mm')


StatsPlots.density(pars[200:1700,1])
StatsPlots.density(pars[200:1700,2])
StatsPlots.density(pars[200:1700,3])
StatsPlots.density(pars[200:1700,4])
StatsPlots.density(pars[200:1700,5])
StatsPlots.density(pars[200:1700,6])
StatsPlots.density(pars[200:1700,7])
StatsPlots.density(pars[200:1700,8])
StatsPlots.density(pars[200:1700,9])
StatsPlots.density(pars[200:1700,10])
StatsPlots.density(pars[200:1700,11])
StatsPlots.density(pars[200:1700,12])
StatsPlots.density(pars[200:1700,13])
StatsPlots.density(pars[200:1700,14])
StatsPlots.density(pars[200:1700,15])
StatsPlots.density(pars[200:1700,16])
