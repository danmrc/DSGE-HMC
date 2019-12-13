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

posterior = transform.(t, res.chains)

StatsPlots.density(posterior_bet)
StatsPlots.density(posterior_theta)
