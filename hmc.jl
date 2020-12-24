using AdvancedHMC
using StatsPlots

include("renormalization_manual.jl")
include("gali_bayesian.jl")
include("simulation.jl")

yy,shocks = simulate_dsge(GAMMA_0,GAMMA_1,PSI,PI,500)

bet = 0.99
sig = 1
phi = 1
alfa = 1/3
epsilon = 6
theta = 2/3
phi_pi = 1.5
phi_y = 0.5/4
rho_v = 0.5
s2 = 1

true_pars = [alfa,bet,epsilon,theta, sig, s2, phi,phi_pi, phi_y,rho_v]

ll,dif = log_like_dsge(true_pars,yy[:,2])


initial_par = true_pars
ell_and_grad(par) = dens_and_grad([1/3;par],yy[:,2])
ell(par) = dens([1/3;par],yy[:,2])

from_unit(x) = -log(1/x-1)
from_pos(x) = log(x)
from_one_to_inf(x) = log(x-1)

trans_true_par = [from_unit(0.99),from_pos(6),from_unit(2/3),from_pos(1),from_pos(1),from_pos(1),from_one_to_inf(1.5),from_pos(0.5/4),from_unit(0.5)]

ell_and_grad(trans_true_par)

n_samples, n_adapts = 2_000, 1_000

# Define a Hamiltonian system
metric = DiagEuclideanMetric(9)
hamiltonian = Hamiltonian(metric, ell, ell_and_grad)

# Define a leapfrog solver, with initial step size chosen heuristically
initial_ϵ = find_good_stepsize(hamiltonian, trans_true_par)
integrator = Leapfrog(initial_ϵ)

# Define an HMC sampler, with the following components
#   - multinomial sampling scheme,
#   - generalised No-U-Turn criteria, and
#   - windowed adaption for step-size and diagonal mass matrix
proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator,9)
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.65, integrator))

# Run the sampler to draw samples from the specified Gaussian, where
#   - `samples` will store the samples
#   - `stats` will store diagnostic statistics for each sample
samples, stats = sample(hamiltonian, proposal, trans_true_par, n_samples, adaptor, n_adapts; progress=true)

tab = zeros(2000,9)

for j = 1:size(samples,1)
    tab[j,:] = samples[j]
end

StatsPlots.density(to_unit.(tab[1:2000,1]), legend = :topleft, label = "Distribution", title = "β")
vline!([true_pars[2]],label = "True Value")
png("imgs/beta.png")

StatsPlots.density(to_positive.(tab[1:2000,2]), label = "Distribution",title = "ϵ")
vline!([true_pars[3]],label = "True Value")
png("imgs/epsilon.png")

StatsPlots.density(to_unit.(tab[1:2000,3]), label = "Distribution", title = "θ")
vline!([true_pars[4]],label = "True Value")
png("imgs/theta.png")

StatsPlots.density(to_positive.(tab[1:2000,4]), label = "Distribution",title = "σ")
vline!([true_pars[5]],label = "True Value")
png("imgs/sigma.png")

StatsPlots.density(to_positive.(tab[1:2000,5]), label = "Distribution", title = "σ_v")
vline!([true_pars[6]],label = "True Value")
png("imgs/sigma_v.png")

StatsPlots.density(to_positive.(tab[1:2000,6]), label = "Distribution",title = "ϕ")
vline!([true_pars[7]],label = "True Value")
png("imgs/phi.png")

StatsPlots.density(to_one_inf.(tab[1:2000,7]), label = "Distribution", title = "ϕ_π")
vline!([true_pars[8]],label = "True Value")
png("imgs/phi_pi.png")

StatsPlots.density(to_positive.(tab[1:2000,8]), label = "Distribution", title = "ϕ_y")
vline!([true_pars[9]],label = "True Value")
png("imgs/phi_y.png")

StatsPlots.density(to_unit.(tab[1:2000,9]), label = "Distribution", title = "ρ")
vline!([true_pars[10]],label = "True Value")
png("imgs/rho.png")

using JLD

save("values.jld","samples",tab,"stats",stats)

load("values.jld")
