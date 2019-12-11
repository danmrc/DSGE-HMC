using Mamba
using Distributions

include("../src/simulation.jl")
include("../gali_bayesian.jl")
include("../hmc/hmc_aux.jl")

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

    # alfa = Stochastic(
    # ()-> Beta(3,3),
    # false
    # ),

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
    () -> Gamma(2,1),
    false
    ),

    phi_y = Stochastic(
    () -> Gamma(2,1),
    false
    ),

    rho_v = Stochastic(
    () -> Beta(1,1),
    false
    ),

    theta = Stochastic(
    () -> Beta(3,3),
    false
    ),

    # s2 = Stochastic(
    # () -> Gamma(2,2),
    # false
    # ),

    par = Logical(1,
    (bet,epsilon,theta,sig,phi,phi_pi,phi_y,rho_v) -> [2/3,bet,epsilon,theta,sig,1,phi,phi_pi,phi_y,rho_v]#() -> [0.3,0.99,6,2/3,2,1,2,1.5,1.5,0.4]#
    )
)

sampling1 = [NUTS([:par])]

sampling2 = [RWM([:par],10)]

dados = Dict{Symbol, Any}(:y => yy[1:499,2], :x => yy[:,2])

inits = [
    Dict(:y => dados[:y],
    #:alfa => rand(Beta(3,3)),
    :bet => rand(Beta(4,2)),
    :epsilon => rand(Gamma(7,1)),
    :sig => rand(Gamma(2,1)),
    :phi => rand(Gamma(2,1)),
    :phi_pi => rand(Gamma(2,1)),
    :phi_y => rand(Gamma(2,1)),
    :rho_v => rand(Beta(1,1)),
    :theta => rand(Beta(3,3)),
    #:s2 => rand(Gamma(2,2))
    )
    for i in 1:2
]

setsamplers!(mm,sampling1)

sim1 = mcmc(mm,dados,inits,5000, burnin=1000, thin=2, chains=2)

describe(sim1)

p = Gadfly.plot(sim1)

Gadfly.draw(p, filename = "plots.svg")

pp = [2/3,inits[1][:bet],inits[1][:epsilon],inits[1][:theta],inits[1][:sig],inits[1][:s2],inits[1][:phi],inits[1][:phi_pi],inits[1][:phi_y],inits[1][:rho_v]]

pp[6] = 0.05

dsge_fit(pp,yy[:,2])
