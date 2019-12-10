using Mamba
using Distributions

include("src/simulation.jl")
include("gali_bayesian.jl")
include("hmc/hmc_aux.jl")

yy,shocks = simulate_dsge(GAMMA_0,GAMMA_1,PSI,PI,500)

mm = Model(
    y = Stochastic(1,
    (mu)-> MvNormal(mu,1),
    false
    ),

    mu = Logical(1,
    (par,x) -> dsge_fit(par,x),
    false
    ),

    alfa = Stochastic(
    ()-> Beta(3,3),
    false
    ),

    bet = Stochastic(
    ()-> Beta(4,2),
    false
    ),

    epsilon = Stochastic(
    () -> Gamma(7,1),
    false
    ),

    sig = Stochastic(
    () -> Gamma(2,1),
    false
    ),

    phi = Stochastic(
    () -> Gamma(2,1),
    false
    ),

    phi_pi = Stochastic(
    () -> Normal(1,2),
    false
    ),

    phi_y = Stochastic(
    () -> Normal(1,2),
    false
    ),

    rho_v = Stochastic(
    () -> Normal(0,sqrt(0.1)),
    false
    ),

    theta = Stochastic(
    () -> Beta(3,3),
    false
    ),

    s2 = Stochastic(
    () -> InverseGamma(0.001,0.001),
    false
    ),

    par = Logical(1,
    (alfa,bet,epsilon,theta,sig,s2,phi,phi_pi,phi_y,rho_v) -> [alfa,bet,epsilon,theta,sig,s2,phi,phi_pi,phi_y,rho_v]
    )#() -> [0.3,0.99,6,2/3,2,1,2,1.5,1.5,0.4]#
)

sampling1 = [NUTS([:par])]

sampling2 = [RWM([:par])]

dados = Dict{Symbol, Any}(:y => yy[:,2], :x => yy[:,2])

inits = [
    Dict(:y => dados[:y],
    :alfa => rand(Beta(3,3),1),
    :bet => rand(Beta(4,2),1),
    :epsilon => rand(Gamma(7,1),1),
    :sig => rand(Gamma(2,1),1),
    :phi => rand(Gamma(2,1),1),
    :phi_pi => rand(Normal(1,2),1),
    :phi_y => rand(Normal(1,2),1),
    :rho_v => rand(Normal(0,sqrt(0.1)),1),
    :theta => rand(Beta(3,3),1),
    :s2 => rand(Gamma(1,1),1))
    for i in 1:3
]

setsamplers!(mm,sampling1)

sim1 = mcmc(mm,dados,inits,10000, burnin=250, thin=2, chains=3)

describe(sim1)

Gadfly.draw(pp, filename = "plots.svg")
