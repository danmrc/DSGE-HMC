using Mamba
using Distributions

xx = randn(100)
yy = 0.5 .+ 1*xx + randn(100)

model = Model(

  y = Stochastic(1,
    (mu, s2) ->  MvNormal(mu, sqrt(s2)),
    false
  ),

  mu = Logical(1,
    (xmat, beta) -> xmat * beta,
    false
  ),

  beta1 = Stochastic(1,
    () -> Normal(0, sqrt(0.1)),false
  ),

  #beta2 = Stochastic(1,
  #  () -> Normal(0, sqrt(1000)),false
  #),

  beta = Logical(1,
    (beta1,beta2) -> [beta1; beta2]
  ),

  s2 = Stochastic(
    () -> InverseGamma(0.001, 0.001)
  )

)

scheme = [NUTS([:beta, :s2])]

setsamplers!(model, scheme)

line = Dict{Symbol, Any}(
  :x => xx,
  :y => yy
)

line[:xmat] = [ones(100) line[:x]]

inits = [
  Dict{Symbol, Any}(
    :y => line[:y],
    :beta1 => rand(Normal(0, 1),1),
    :beta2 => rand(Normal(0, 1),1),
    :s2 => rand(Gamma(1, 1))
  ) for i in 1:3]

sim1 = mcmc(model, line, inits, 10000, burnin=250, thin=2, chains=3)

describe(sim1)
