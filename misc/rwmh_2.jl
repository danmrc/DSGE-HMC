using Distributions,StatsPlots, ForwardDiff

n_samples = 1000
n = 10_000

x = randn(n,50)
theta = [repeat([1],10); repeat([0],40)]
y = x*theta + randn(n)

prior = Laplace(0,2)

ll(theta) = sum(logpdf.(Normal.(0,1),y-x*theta))

post(theta) = ll(theta) + sum(logpdf.(prior,theta))

init = rand(prior,50)

grad(theta) = ForwardDiff.grad(post,theta)

### start here
#

old_par = init

pars = zeros(n_samples,50)

uni = Uniform(0,1)

accept = 0

var_trans = 0.20

for i = 1:n_samples
    old_transition = MvNormal(old_par,var_trans)
    new_par = rand(old_transition)
    new_transition = MvNormal(new_par,var_trans)
    r = post(new_par) + logpdf(new_transition,old_par) - (post(old_par)+ logpdf(old_transition,new_par))
    a = rand(uni)
    r = min(1,exp(r))
    if r > a
        pars[i,:] = new_par
        global old_par = new_par
        global accept = accept + 1
    else
        pars[i,:] = old_par
    end
    println("Accept ratio ", accept/i*100, "%")
end

StatsPlots.density(pars[:,21])

mm = mapslices(mode,pars,dims = 1)

scatter(mm')
