using Distributions,StatsPlots, ForwardDiff, LinearAlgebra

n_samples = 1000

n = 10_000

x = randn(n,50)
theta = [repeat([1],10); repeat([0],40)]
y = x*theta + randn(n)

prior = Laplace(0,2)

ll(theta) = logpdf(MvNormal(I(n)),y-x*theta)

post(theta) = ll(theta) + sum(logpdf.(prior,theta))

init = rand(prior,50)

grad(theta) = ForwardDiff.gradient(post,theta)

M = 1.5*I(50)#mass matrix

mom_dist = MvNormal(M)


#######
### start here
#


pars = zeros(n_samples,50)

uni = Uniform(0,1)

leap_step = 0.023

L = Int(ceil(1/leap_step))

accept = 0

old_par = init#repeat([0],50)

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

StatsPlots.density(pars[100:1000])

mm = mapslices(mode,pars,dims = 1)

scatter(mm')
