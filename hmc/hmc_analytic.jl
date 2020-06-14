using DynamicHMC, LogDensityProblems
using Distributions, Parameters, Random
using ForwardDiff, StatsPlots

include(string(pwd(),"/src/simulation.jl"))
include(string(pwd(),"/gali_bayesian.jl"))
include(string(pwd(),"/mcmc/priors_v2.jl"))
include(string(pwd(),"/mcmc/foos_v2.jl"))

yy,shocks = simulate_dsge(GAMMA_0,GAMMA_1,PSI,PI,500)

prob = DSGE_Model(yy[:,2],1/3)

prob((bet = 0.99,epsilon = 6,theta=2/3,sig=1,s2=1,phi=1,phi_pi=1.5,phi_y=0.5/4,rho_v=0.5))

t = problem_transform(prob)

P = TransformedLogDensity(t,prob)

aa = [0.99,6,2/3,1,1,1,1.5,0.5/4,0.5]

true_vals = TransformVariables.inverse(t,(bet = 0.99,epsilon = 6,theta=2/3,sig=1,s2=1,phi=1,phi_pi=1.5,phi_y=0.5/4,rho_v=0.5))

LogDensityProblems.logdensity(P,true_vals)

function LogDensityProblems.capabilities(::TransformedLogDensity{TransformVariables.TransformTuple{NamedTuple{(:bet, :epsilon, :theta, :sig, :s2, :phi, :phi_pi, :phi_y, :rho_v),Tuple{TransformVariables.ScaledShiftedLogistic{Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ScaledShiftedLogistic{Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ScaledShiftedLogistic{Float64}}}},DSGE_Model{Array{Float64,1},Float64}})
    LogDensityProblems.LogDensityOrder{1}() # can do gradient
end

#LogDensityProblems.dimension(::NormalPosterior) = 2 # for this problem

function LogDensityProblems.logdensity_and_gradient(problem::TransformedLogDensity{TransformVariables.TransformTuple{NamedTuple{(:bet, :epsilon, :theta, :sig, :s2, :phi, :phi_pi, :phi_y, :rho_v),Tuple{TransformVariables.ScaledShiftedLogistic{Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ScaledShiftedLogistic{Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ShiftedExp{true,Float64},TransformVariables.ScaledShiftedLogistic{Float64}}}},DSGE_Model{Array{Float64,1},Float64}}, x)
    logdens, grad = LogDensityProblems.logdensity(P,x)
    logdens, grad
end

res = @time mcmc_with_warmup(Random.GLOBAL_RNG, P,1000)
