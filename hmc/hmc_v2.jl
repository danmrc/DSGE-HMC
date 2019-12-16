using DynamicHMC, LogDensityProblems
using Distributions, Parameters, Random, Flux, Tracker
using Calculus, StatsPlots

include(string(pwd(),"/src/simulation.jl"))
include(string(pwd(),"/gali_bayesian.jl"))
include(string(pwd(),"/mcmc/priors.jl"))
include(string(pwd(),"/mcmc/foos.jl"))

yy,shocks = simulate_dsge(GAMMA_0,GAMMA_1,PSI,PI,500)

prob = DSGE_Model(yy[:,2],2/3)

prob((bet = 0.99,epsilon = 6,theta=2/3,sig=1,s2=1,phi=1,phi_pi=1.5,phi_y=0.5/4,rho_v=0.5))

t = problem_transform(prob)

P = TransformedLogDensity(t,prob)

aa = [0.99,6,2/3,1,1,1,1.5,0.5/4,0.5]

bb = TransformVariables.inverse(t,(bet = 0.99,epsilon = 6,theta=2/3,sig=1,s2=1,phi=1,phi_pi=1.5,phi_y=0.5/4,rho_v=0.5))

LogDensityProblems.logdensity(P,bb)

function LogDensityProblems.capabilities(::Type{TransformedLogDensity{TransformVariables.TransformTuple{NamedTuple{(:bet, :epsilon, :theta, :sig, :s2, :phi, :phi_pi, :phi_y, :rho_v),Tuple{TransformVariables.ScaledShiftedLogistic{Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ScaledShiftedLogistic{Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ScaledShiftedLogistic{Float64}}}},DSGE_Model{Array{Float64,1},Float64}}})
    LogDensityProblems.LogDensityOrder{1}()
end

function LogDensityProblems.logdensity_and_gradient(P::TransformedLogDensity{TransformVariables.TransformTuple{NamedTuple{(:bet, :epsilon, :theta, :sig, :s2, :phi, :phi_pi, :phi_y, :rho_v),Tuple{TransformVariables.ScaledShiftedLogistic{Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ScaledShiftedLogistic{Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ScaledShiftedLogistic{Float64}}}},DSGE_Model{Array{Float64,1},Float64}},x)
    logdens = LogDensityProblems.logdensity(P,x)
    grad = Calculus.gradient(y->LogDensityProblems.logdensity(P,y),x)
    return logdens,grad
end

grad_p(x) = LogDensityProblems.logdensity_and_gradient(P,x)

grads = Calculus.gradient(P)

res = mcmc_with_warmup(Random.GLOBAL_RNG, P,5000)

posterior = transform.(t, res.chain)

DynamicHMC.Diagnostics.summarize_tree_statistics(res.tree_statistics)

post_array = zeros(5000,9)

for i in 1:5000
    @unpack bet, epsilon, theta, sig, s2,phi,phi_pi,phi_y,rho_v = posterior[i,:][1]
    post_array[i,:] = [bet epsilon theta sig s2 phi phi_pi phi_y rho_v]
end

StatsPlots.density(post_array[:,1], legend = :none)
title!(latexify("beta()"))
vline!([true_pars[:bet]])

StatsPlots.density(post_array[:,2], legend = :none)
title!(latexify("epsilon()"))
vline!([true_pars[:epsilon]])

StatsPlots.density(post_array[:,3], legend = :none)
title!(latexify("theta()"))
vline!([true_pars[:theta]])

StatsPlots.density(post_array[:,4], legend = :none)
title!(latexify("sigma()"))
vline!([true_pars[:sig]])

StatsPlots.density(post_array[:,5], legend = :none)
title!(latexify("sigma()^2"))
vline!([true_pars[:s2]])

StatsPlots.density(post_array[:,6], legend = :none)
title!(latexify("phi()"))
vline!([true_pars[:phi]])

StatsPlots.density(post_array[:,7], legend = :none)
title!(latexify("phi()_pi",cdot = false))
vline!([true_pars[:phi_pi]])

StatsPlots.density(post_array[:,8], legend = :none)
title!(latexify("phi()_y",cdot = false))
vline!([true_pars[:phi_y]])

StatsPlots.density(post_array[:,9], legend = :none)
title!(latexify("rho()_v",cdot = false))
vline!([true_pars[:rho_v]])
