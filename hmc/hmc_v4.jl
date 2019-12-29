using DynamicHMC, LogDensityProblems
using Distributions, Parameters, Random
using StatsPlots, PositiveFactorizations, Calculus
using Flux

include(string(pwd(),"/src/simulation.jl"))
include(string(pwd(),"/gali_bayesian.jl"))
include(string(pwd(),"/mcmc/priors_v2.jl"))
include(string(pwd(),"/mcmc/foos.jl"))

yy,shocks = simulate_dsge(GAMMA_0,GAMMA_1,PSI,PI,500)

prob = DSGE_Model(yy[:,2],1/3)

prob((bet = 0.99,epsilon = 6,theta=2/3,sig=1,s2=1,phi=1,phi_pi=1.5,phi_y=0.5/4,rho_v=0.5))

t = problem_transform(prob)

P = TransformedLogDensity(t,prob)

aa = [0.99,6,2/3,1,1,1,1.5,0.5/4,0.5]

true_vals = TransformVariables.inverse(t,(bet = 0.99,epsilon = 6,theta=2/3,sig=1,s2=1,phi=1,phi_pi=1.5,phi_y=0.5/4,rho_v=0.5))

LogDensityProblems.logdensity(P,true_vals)

#grads = Calculus.gradient(P)

function LogDensityProblems.capabilities(::TransformedLogDensity{TransformVariables.TransformTuple{NamedTuple{(:bet, :epsilon, :theta, :sig, :s2, :phi, :phi_pi, :phi_y, :rho_v),Tuple{TransformVariables.ScaledShiftedLogistic{Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ScaledShiftedLogistic{Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ScaledShiftedLogistic{Float64}}}},DSGE_Model{Array{Float64,1},Float64}})
    LogDensityProblems.LogDensityOrder{1}() # can do gradient
end

#LogDensityProblems.dimension(::NormalPosterior) = 2 # for this problem

function LogDensityProblems.logdensity_and_gradient(problem::TransformedLogDensity{TransformVariables.TransformTuple{NamedTuple{(:bet, :epsilon, :theta, :sig, :s2, :phi, :phi_pi, :phi_y, :rho_v),Tuple{TransformVariables.ScaledShiftedLogistic{Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ScaledShiftedLogistic{Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ScaledShiftedLogistic{Float64}}}},DSGE_Model{Array{Float64,1},Float64}}, x)
    logdens = LogDensityProblems.logdensity(P,x)
    grad = -1*Flux.gradient(y->model_flux(y)[1],x)[1] .*exp.(model_flux(x)[1])
    logdens, grad
end

res = @time mcmc_with_warmup(Random.GLOBAL_RNG, P,1000)

posterior = transform.(t, res.chain)

post_array = zeros(1000,9)

for i in 1:1000
    @unpack bet, epsilon, theta, sig, s2,phi,phi_pi,phi_y,rho_v = posterior[i,:][1]
    post_array[i,:] = [bet epsilon theta sig s2 phi phi_pi phi_y rho_v]
end

StatsPlots.density(post_array[:,1], legend = :none)
title!(latexify("beta()"))
vline!([true_pars[:bet]])
vline!([mean(post_array[:,1])])

StatsPlots.density(post_array[:,2], legend = :none)
title!(latexify("epsilon()"))
vline!([true_pars[:epsilon]])
vline!([mean(post_array[:,2])])

StatsPlots.density(post_array[:,3], legend = :none)
title!(latexify("theta()"))
vline!([true_pars[:theta]])
vline!([mean(post_array[:,3])])

StatsPlots.density(post_array[:,4], legend = :none)
title!(latexify("sigma()"))
vline!([true_pars[:sig]])
vline!([mean(post_array[:,4])])

StatsPlots.density(post_array[:,5], legend = :none)
title!(latexify("sigma()^2"))
vline!([true_pars[:s2]])
vline!([mean(post_array[:,5])])

StatsPlots.density(post_array[:,6], legend = :none)
title!(latexify("phi()"))
vline!([true_pars[:phi]])
vline!([mean(post_array[:,6])])

StatsPlots.density(post_array[:,7], legend = :none)
title!(latexify("phi()_pi",cdot = false))
vline!([true_pars[:phi_pi]])
vline!([mean(post_array[:,7])])

StatsPlots.density(post_array[:,8], legend = :none)
title!(latexify("phi()_y",cdot = false))
vline!([true_pars[:phi_y]])
vline!([mean(post_array[:,8])])

StatsPlots.density(post_array[:,9], legend = :none)
title!(latexify("rho()_v",cdot = false))
vline!([true_pars[:rho_v]])
vline!([mean(post_array[:,9])])
