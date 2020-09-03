using LinearAlgebra, Distributions

# stepsize: the stepsize
# n_steps: number of steps for the leapfrog
# mm: metric matrix
# post: the posterior distribution. Should be a function!
# grad: the gradient. Should be a function!
# mom: the moment for the beggining of the leapfrog
# par: the paramater for the beggining of the leapfrog

function sympl_integrator(stepsize,n_steps,mm,post,grad,mom,par)
    new_mom = mom
    new_par = par
    direction = 1
    for l in 1:n_steps
        new_mom = new_mom + 1/2*stepsize*grad(new_par)
        new_par = new_par + stepsize*mm*new_mom
        if isnan(post(new_par)) || isinf(post(new_par))
            @warn "Invalid value of the distribution. Going back"
            direction = -1
        end
        new_mom = new_mom + 1/2*stepsize*grad(new_par)
    end
    return new_par,new_mom
end

function sympl_integrator_window(stepsize,n_steps,mm,mom_dist,post,grad,mom,par)
    unif = Uniform(0,1)
    new_mom = mom
    new_par = par
    prob = 0
    prob_cur = 0
    cur_par, cur_mom = new_par,new_mom
    direction = 1
    for l in 1:n_steps
        new_mom = new_mom + 1/2*stepsize*grad(new_par)
        new_par = new_par + stepsize*mm*new_mom
        if isnan(loglike(new_par)) || isinf(loglike(new_par))
            direction = -1
        end
        new_mom = new_mom + 1/2*stepsize*grad(new_par)
        prob = exp(log_pdf(mom_dist,new_mom) + post(new_par))
        prob_cur += prob
        if prob/prob_cur > rand(unif)
            cur_par,cur_mom = new_par,new_mom
        end
    end
    return new_par,new_mom, cur_par,cur_mom
end

function metropolis_step(par_new,mom_new,par_old,mom_old,post,dist_mom)
    metropolis = post(par_new) + logpdf(dist_mom,mom_new) - post(par_old) -logpdf(dist_mom,mom_old)
    metropolis = min(exp(metropolis),1)
    unif = Uniform(0,1)
    if metropolis > rand(unif)
        return par_new, mom_new, 1
    else
        return par_old, mom_old, 0
    end
end
